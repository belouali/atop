"""Saliency processing: IG computation per split, salient token selection."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from atop.config import PAD_IDX, CLS_IDX, VISIT_IDX, SPECIAL_TOKENS
from atop.utils import format_token_readable, token_stream

_token_stream = token_stream


def process_split_for_sequences(
    model: SingleStreamTransformer,
    ig: IntegratedGradientsCustom,
    loader: DataLoader,
    device: torch.device,
    vocab_inv: Dict[int, str],
    icd_titles: Dict,
    ig_mass: float,
    ig_max_tokens: int,
    store_tensors: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run IG over all batches, return:
      - df_seq: per-sample records with token sequences and saliency
      - df_ig: long-form IG rows for plotting
      - tensors_by_key: (pid, hadm_id) -> input_ids tensor on CPU
    """
    records = []
    ig_rows = []
    tensors_by_key = {}

    processed = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        B = input_ids.shape[0]

        # Compute IG (processes one sample at a time internally for memory safety)
        attr = ig.attribute(input_ids, target_class=1)  # (B, L)
        processed += B
        print(f"\r  IG progress: {processed} samples", end="", flush=True)

        for b in range(B):
            pid = batch["patient_id"][b]
            hadm_id = batch["index_hadm_id"][b]
            label = float(batch["label"][b].item())
            flat_tokens = batch["flat_tokens"][b]
            token_visit_idx = batch["token_visit_idx"][b]

            ids_b = input_ids[b].detach().cpu()
            attr_b = attr[b].detach().cpu().numpy()

            if store_tensors:
                tensors_by_key[(pid, hadm_id)] = ids_b.unsqueeze(0).clone()

            # Precompute truncation mapping for visit_idx
            L_encoded = len(ids_b)
            L_original = len(flat_tokens)
            max_seq_len = input_ids.shape[1]
            was_truncated = (L_original > max_seq_len)

            def _enc_pos_to_visit(pos):
                """Map encoded position → visit_idx via original flat_tokens."""
                if not was_truncated:
                    orig_idx = pos
                elif pos == 0:
                    orig_idx = 0
                else:
                    tail_offset = L_original - (L_encoded - 1)
                    orig_idx = tail_offset + (pos - 1)
                if 0 <= orig_idx < len(token_visit_idx):
                    v = token_visit_idx[orig_idx]
                    return v if v is not None else -1
                return -1

            # Decode tokens and gather IG
            combined_full = []
            combined_map = []  # (position_in_encoded, token_str)
            all_items = []

            for pos in range(len(ids_b)):
                tid = ids_b[pos].item()
                if tid == PAD_IDX:
                    continue
                tok_str = vocab_inv.get(tid, f"UNK_{tid}")
                ig_signed = float(attr_b[pos])
                igv = abs(ig_signed)
                stream = _token_stream(tok_str)

                combined_full.append(tok_str)
                combined_map.append((pos, tok_str))

                if tok_str not in SPECIAL_TOKENS:
                    tok_readable = format_token_readable(tok_str, icd_titles)
                    vidx = _enc_pos_to_visit(pos)
                    all_items.append({
                        "stream": stream, "pos": pos, "ig_abs": igv,
                        "ig_signed": ig_signed,
                        "token_str": tok_str,
                    })
                    ig_rows.append({
                        "patient_id": pid, "index_hadm_id": hadm_id, "readmission": label,
                        "stream": stream, "pos": pos,
                        "visit_idx": vidx,
                        "token_str": tok_str,
                        "token_str_readable": tok_readable,
                        "ig_abs": igv,
                        "ig_signed": ig_signed,
                    })

            # Select salient tokens (mass-based)
            combined_salient, salient_positions = _select_salient(
                all_items, mass=ig_mass, max_tokens=ig_max_tokens)

            # Tag ig_rows with is_salient by POSITION (not token string).
            # This prevents repeated tokens (e.g. HTN in multiple visits) from
            # being marked salient everywhere just because HTN is salient once.
            n_items = len(all_items)
            for j in range(n_items):
                ig_rows[-(n_items - j)]["is_salient"] = all_items[j]["pos"] in salient_positions

            # Build salient visit blocks for episode mining
            salient_visit_blocks, visit_to_block = _build_salient_visit_blocks(
                all_items, combined_salient, flat_tokens, token_visit_idx, ids_b, vocab_inv,
                max_seq_len=max_seq_len)

            records.append({
                "patient_id": pid, "index_hadm_id": hadm_id, "readmission": label,
                "n_visits": batch["n_visits"][b],
                "combined_full": combined_full,
                "combined_map": combined_map,
                "combined_salient": combined_salient,
                "salient_visit_blocks": salient_visit_blocks,
                "visit_to_block": visit_to_block,
                "len_full": len(combined_full),
                "len_salient": len(combined_salient),
            })

    print()  # newline after progress
    return pd.DataFrame(records), pd.DataFrame(ig_rows), tensors_by_key



def _select_salient(items: List[Dict], mass: float, max_tokens: int):
    """Select top tokens by IG mass, preserving sequential order.

    Returns:
        salient_tokens: List[str] — token strings in sequence order
        salient_positions: Set[int] — encoded positions of salient tokens
    """
    if not items:
        return [], set()
    igs = np.array([it["ig_abs"] for it in items])
    tot = igs.sum() + 1e-12
    order = np.argsort(-igs)
    keep_idx = []
    cum = 0.0
    for j in order:
        keep_idx.append(j)
        cum += items[j]["ig_abs"]
        if cum / tot >= mass:
            break
        if max_tokens and len(keep_idx) >= max_tokens:
            break
    # Re-sort by position to preserve sequence order
    keep_idx.sort(key=lambda j: items[j]["pos"])
    salient_tokens = [items[j]["token_str"] for j in keep_idx]
    salient_positions = {items[j]["pos"] for j in keep_idx}
    return salient_tokens, salient_positions



def _build_salient_visit_blocks(all_items, combined_salient, flat_tokens,
                                 token_visit_idx, ids_b, vocab_inv, max_seq_len):
    """
    Build visit-block representation of salient tokens for episode mining.
    Returns: List[frozenset] — one frozenset per visit containing salient token strings.
    Only visits that contribute at least one salient token are included.

    Handles CLS-preserving truncation:
      encoded[0] = original[0] (CLS)
      encoded[1:] = original[-(max_seq_len-1):]  (tail of original)
    """
    salient_set = set(combined_salient)
    if not salient_set:
        return [], {}

    L_encoded = len(ids_b)
    L_original = len(flat_tokens)
    was_truncated = (L_original > max_seq_len)

    def _encoded_pos_to_orig(pos):
        """Map encoded position back to original flat_tokens index."""
        if not was_truncated:
            return pos
        if pos == 0:
            return 0  # CLS is always original position 0
        # positions 1.. map to the tail of the original
        tail_offset = L_original - (L_encoded - 1)
        return tail_offset + (pos - 1)

    visit_to_tokens = {}
    for pos in range(L_encoded):
        tid = ids_b[pos].item()
        if tid == PAD_IDX or tid in (CLS_IDX, VISIT_IDX):
            continue
        tok_str = vocab_inv.get(tid, f"UNK_{tid}")
        if tok_str not in salient_set:
            continue
        orig_idx = _encoded_pos_to_orig(pos)
        if 0 <= orig_idx < len(token_visit_idx):
            vidx = token_visit_idx[orig_idx]
        else:
            vidx = -1
        if vidx is not None and vidx >= 0:
            if vidx not in visit_to_tokens:
                visit_to_tokens[vidx] = set()
            visit_to_tokens[vidx].add(tok_str)

    # Build ordered list of frozensets and visit_idx→block_ordinal mapping
    blocks = []
    visit_to_block = {}
    for bi, vidx in enumerate(sorted(visit_to_tokens.keys())):
        blocks.append(frozenset(visit_to_tokens[vidx]))
        visit_to_block[vidx] = bi
    return blocks, visit_to_block


# ============================================================================
# PATTERN MINING — Episode (itemset-sequence), n-gram, and PrefixSpan
# ============================================================================
