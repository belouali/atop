"""Pattern validation: masking, shuffling, reliance measurement."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from atop.config import PAD_IDX, CLS_IDX, VISIT_IDX, SPECIAL_TOKENS
from atop.utils import predict_one, format_token_readable
from atop.mining.patterns import (
    _is_subsequence, _is_subpattern, _parse_episode_pattern,
)


def find_subsequence_occurrence(combined_full, combined_map, pattern, contiguous):
    """Find first occurrence; return list of (encoded_pos, tok_str) for matched tokens.
    Works for flat patterns (ngram/prefixspan). For episode patterns, use find_episode_occurrence."""
    tokens_only = [t for t in combined_full if t not in SPECIAL_TOKENS]
    map_only = [(pos, tok) for pos, tok in combined_map if tok not in SPECIAL_TOKENS]
    n = len(pattern)
    if len(tokens_only) < n:
        return []
    if contiguous:
        for i in range(len(tokens_only) - n + 1):
            if tokens_only[i:i + n] == pattern:
                return [map_only[i + j] for j in range(n)]
        return []
    else:
        matched = []
        start = 0
        for tok in pattern:
            found = False
            for i in range(start, len(tokens_only)):
                if tokens_only[i] == tok:
                    matched.append(map_only[i])
                    start = i + 1
                    found = True
                    break
            if not found:
                return []
        return matched



def find_episode_occurrence(combined_full, combined_map, episode_pattern):
    """
    Find first occurrence of an episode pattern (list of frozensets) in the token sequence.
    Uses [VISIT] tokens as visit boundaries to identify visit blocks, then matches
    each itemset in the pattern against visit blocks in order (greedy earliest-match).

    Returns the encoded positions of matched tokens within matched blocks only,
    ensuring masking targets the specific instantiation, not all occurrences.

    Returns list of (encoded_pos, tok_str) for all matched tokens, or [] if not found.
    """
    # Split into visit blocks based on [VISIT] tokens
    visit_blocks = []  # list of (list of (pos, tok_str))
    current_block = []
    for pos, tok in combined_map:
        if tok == "[VISIT]":
            if current_block:
                visit_blocks.append(current_block)
            current_block = []
        elif tok not in SPECIAL_TOKENS:
            current_block.append((pos, tok))
    if current_block:
        visit_blocks.append(current_block)

    # Match episode pattern against visit blocks
    matched_positions = []
    block_pos = 0
    for itemset in episode_pattern:
        found = False
        while block_pos < len(visit_blocks):
            block_tokens = {tok for _, tok in visit_blocks[block_pos]}
            if itemset.issubset(block_tokens):
                # Found: collect positions for matched items
                for pos, tok in visit_blocks[block_pos]:
                    if tok in itemset:
                        matched_positions.append((pos, tok))
                block_pos += 1
                found = True
                break
            block_pos += 1
        if not found:
            return []
    return matched_positions


# ============================================================================
# PERTURBATION HELPERS
# ============================================================================


def mask_positions(input_ids: torch.Tensor, positions: List[int]) -> torch.Tensor:
    """Zero out specific positions in input_ids (B=1, L)."""
    masked = input_ids.clone()
    for pos in positions:
        masked[0, pos] = PAD_IDX
    return masked



def shuffle_within_visits(input_ids: torch.Tensor, vocab: Dict[str, int],
                          vocab_inv: Dict[int, str]) -> torch.Tensor:
    """Shuffle non-special tokens within each visit block."""
    out = input_ids.clone()
    ids = out[0].numpy().tolist()

    # Find visit boundaries
    visit_starts = [i for i, tid in enumerate(ids) if tid == VISIT_IDX]

    for vi, vstart in enumerate(visit_starts):
        vend = visit_starts[vi + 1] if vi + 1 < len(visit_starts) else len(ids)
        # Collect non-special, non-pad token positions in this visit
        positions = []
        for j in range(vstart + 1, vend):
            if ids[j] != PAD_IDX and ids[j] not in (CLS_IDX, VISIT_IDX):
                positions.append(j)
        if len(positions) > 1:
            vals = [ids[p] for p in positions]
            np.random.shuffle(vals)
            for p, v in zip(positions, vals):
                out[0, p] = v

    return out


def shuffle_visit_blocks(input_ids: torch.Tensor) -> torch.Tensor:
    """Permute visit blocks while keeping tokens within each block intact.
    
    This tests whether the model learned cross-visit temporal structure.
    If the model only cares about which codes are present (bag-of-codes),
    shuffling visits won't change the prediction. If it learned temporal
    patterns (e.g. HTN in visit 1 → CKD in visit 3), shuffling will
    degrade predictions for patients with those patterns.
    
    Preserves [CLS] at position 0. Each visit block = [VISIT] + tokens.
    """
    out = input_ids.clone()
    ids = out[0].numpy().tolist()
    
    # Find [CLS] and visit boundaries
    cls_end = 1  # [CLS] is at position 0
    visit_starts = [i for i, tid in enumerate(ids) if tid == VISIT_IDX]
    
    if len(visit_starts) < 2:
        return out  # nothing to shuffle with 0 or 1 visits
    
    # Extract each visit block as a contiguous slice
    blocks = []
    for vi, vstart in enumerate(visit_starts):
        vend = visit_starts[vi + 1] if vi + 1 < len(visit_starts) else len(ids)
        # Don't include trailing PAD in the last block
        block = ids[vstart:vend]
        # Strip trailing PADs from block
        while block and block[-1] == PAD_IDX:
            block = block[:-1]
        if block:
            blocks.append(block)
    
    # Permute block order
    perm = np.random.permutation(len(blocks)).tolist()
    shuffled_blocks = [blocks[p] for p in perm]
    
    # Reconstruct: [CLS] + shuffled blocks + PAD
    new_ids = [ids[0]]  # [CLS]
    for block in shuffled_blocks:
        new_ids.extend(block)
    
    # Pad to original length
    seq_len = len(ids)
    while len(new_ids) < seq_len:
        new_ids.append(PAD_IDX)
    new_ids = new_ids[:seq_len]  # truncate if longer
    
    out[0] = torch.tensor(new_ids, dtype=out.dtype)
    return out


# ============================================================================
# BATCHED PREDICTION
# ============================================================================


def predict_batch(model, device: torch.device, batch_tensors: List[torch.Tensor],
                  batch_size: int = 64) -> np.ndarray:
    """Run batched forward passes. Each tensor is (1, L). Returns array of probabilities."""
    if not batch_tensors:
        return np.array([])

    model.eval()
    all_probs = []

    for start in range(0, len(batch_tensors), batch_size):
        chunk = batch_tensors[start:start + batch_size]
        max_len = max(t.shape[1] for t in chunk)

        # Pad each to max_len
        padded = []
        for t in chunk:
            if t.shape[1] < max_len:
                pad = torch.full((1, max_len - t.shape[1]), PAD_IDX, dtype=t.dtype)
                t = torch.cat([t, pad], dim=1)
            padded.append(t)

        batch = torch.cat(padded, dim=0).to(device)  # (B, L)
        with torch.no_grad():
            out = model(batch)
            probs = out["y_prob"].cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)


# ============================================================================
# MAIN VALIDATION
# ============================================================================


def run_validation(model, device, df_seq_test, df_patterns_train, tensors_by_key_test,
                   icd_titles, vocab, vocab_inv, top_k, max_adm_per_pattern,
                   n_shuffle_draws, mining_method, out_dir,
                   df_ig=None, pattern_whitelist=None) -> pd.DataFrame:
    """Validate top patterns via masking, within-visit shuffle, and visit-block shuffle.
    
    Pattern selection: If pattern_whitelist is provided, only those patterns are
    validated (ensuring alignment with fig3 Panel C). Otherwise, patterns are
    ranked by |mean Σ IG_i| (consistent with fig3/fig6), or falls back to OR.
    """
    from atop.explain.matching import PatternIndex
    
    is_episode = (mining_method == "episode")
    contiguous = (mining_method == "ngram")
    res = []

    # ── Pattern selection ────────────────────────────────────────────────
    if pattern_whitelist is not None:
        # Use fig3's pattern list — but limit to top_k by |IG| for validation
        top = df_patterns_train[df_patterns_train["pattern"].isin(pattern_whitelist)].copy()
        print(f"  [validate] Using fig3 whitelist: {len(top)}/{len(pattern_whitelist)} patterns found in mined set")
        
        # Rank whitelist patterns by |IG| and take top_k risk + top_k protective
        if df_ig is not None and not df_ig.empty and len(top) > top_k * 2:
            print(f"  [validate] Ranking {len(top)} whitelist patterns by |IG| to select top {top_k}...")
            ig_col = "ig_signed" if "ig_signed" in df_ig.columns else "ig_abs"
            from atop.explain.figures import _build_ig_cache_by_block, _score_pattern_instance
            
            block_cache = _build_ig_cache_by_block(df_ig, df_seq_test)
            pat_index = PatternIndex.from_df(df_seq_test)
            seq_rows = df_seq_test.reset_index(drop=True)
            
            ig_scores = []
            for i, (_, pat_row) in enumerate(top.iterrows()):
                pat_str = pat_row["pattern"]
                if is_episode:
                    ep = _parse_episode_pattern(pat_str)
                    pat_fsets = [frozenset(s) for s in ep]
                    pat_tokens = set()
                    for s in ep:
                        pat_tokens.update(s)
                else:
                    flat = [p.strip() for p in pat_str.split(" -> ")]
                    pat_tokens = set(flat)
                    pat_fsets = [frozenset([t]) for t in flat]

                matched_with_blocks = pat_index.patients_matching_with_blocks(
                    pat_fsets, pat_tokens, is_episode)
                if not matched_with_blocks:
                    ig_scores.append(0.0)
                else:
                    sums = []
                    for ri, matched_blocks in matched_with_blocks:
                        row = seq_rows.iloc[ri]
                        key = (str(row["patient_id"]), str(row["index_hadm_id"]))
                        bl = block_cache.get(key, [])
                        if bl:
                            s = _score_pattern_instance(pat_fsets, matched_blocks, bl)
                            sums.append(s)
                    ig_scores.append(float(np.mean(sums)) if sums else 0.0)

                if (i + 1) % 500 == 0:
                    print(f"\r    IG ranking: {i+1}/{len(top)}", end="", flush=True)
            if len(top) > 500:
                print()
            
            top["_ig_score"] = ig_scores
            top["_ig_abs"] = top["_ig_score"].abs()
            top_risk = top[top["_ig_score"] > 0].nlargest(top_k, "_ig_abs")
            top_prot = top[top["_ig_score"] < 0].nlargest(top_k, "_ig_abs")
            top = pd.concat([top_risk, top_prot], ignore_index=True)
            print(f"  [validate] Selected {len(top_risk)} risk + {len(top_prot)} protective from whitelist")
    elif df_ig is not None and not df_ig.empty and not df_seq_test.empty:
        # Rank by |mean signed IG sum| — same metric as fig3/fig6
        print("  [validate] Ranking patterns by |mean Σ IG_i| for selection...")
        ig_col = "ig_signed" if "ig_signed" in df_ig.columns else "ig_abs"
        from atop.explain.figures import _build_ig_cache_by_block, _score_pattern_instance
        from atop.mining.patterns import _match_subpattern

        block_cache = _build_ig_cache_by_block(df_ig, df_seq_test)
        pat_index = PatternIndex.from_df(df_seq_test)
        seq_rows = df_seq_test.reset_index(drop=True)

        n_pats = len(df_patterns_train)
        ig_scores = []
        for i, (pat_idx, pat_row) in enumerate(df_patterns_train.iterrows()):
            pat_str = pat_row["pattern"]
            if is_episode:
                ep = _parse_episode_pattern(pat_str)
                pat_fsets = [frozenset(s) for s in ep]
                pat_tokens = set()
                for s in ep:
                    pat_tokens.update(s)
            else:
                flat = [p.strip() for p in pat_str.split(" -> ")]
                pat_tokens = set(flat)
                pat_fsets = [frozenset([t]) for t in flat]

            matched_with_blocks = pat_index.patients_matching_with_blocks(
                pat_fsets, pat_tokens, is_episode)
            if not matched_with_blocks:
                ig_scores.append(0.0)
            else:
                sums = []
                for ri, matched_blocks in matched_with_blocks:
                    row = seq_rows.iloc[ri]
                    key = (str(row["patient_id"]), str(row["index_hadm_id"]))
                    bl = block_cache.get(key, [])
                    if bl:
                        s = _score_pattern_instance(pat_fsets, matched_blocks, bl)
                        sums.append(s)
                ig_scores.append(float(np.mean(sums)) if sums else 0.0)

            if (i + 1) % 100 == 0 or (i + 1) == n_pats:
                print(f"\r    IG ranking: {i+1}/{n_pats}", end="", flush=True)
        print()

        df_patterns_train = df_patterns_train.copy()
        df_patterns_train["_ig_score"] = ig_scores
        df_patterns_train["_ig_abs"] = df_patterns_train["_ig_score"].abs()

        # Top risk (positive IG = pushes toward readmission) and protective (negative IG)
        top_risk = df_patterns_train[df_patterns_train["_ig_score"] > 0].nlargest(top_k, "_ig_abs")
        top_prot = df_patterns_train[df_patterns_train["_ig_score"] < 0].nlargest(top_k, "_ig_abs")
        top = pd.concat([top_risk, top_prot], ignore_index=True)
        print(f"  [validate] Selected {len(top_risk)} risk + {len(top_prot)} protective patterns by IG")
    else:
        # Fallback: OR-based ranking
        top_risk = df_patterns_train[df_patterns_train["odds_ratio"] > 1.0].head(top_k)
        top_prot = df_patterns_train[df_patterns_train["odds_ratio"] < 1.0].sort_values("odds_ratio").head(top_k)
        top = pd.concat([top_risk, top_prot], ignore_index=True)

    n_patterns = len(top)
    for pi, (_, row) in enumerate(top.iterrows()):
        pat_str = row["pattern"]

        if is_episode:
            episode_pat = _parse_episode_pattern(pat_str)
            all_items = []
            for itemset in episode_pat:
                all_items.extend(sorted(itemset))
            pat_readable = " → ".join([format_token_readable(t, icd_titles) for t in all_items])
        else:
            flat_pat = [p.strip() for p in pat_str.split(" -> ")]
            pat_readable = " -> ".join([format_token_readable(t, icd_titles) for t in flat_pat])

        # Find matches
        matches = []
        for _, r in df_seq_test.iterrows():
            key = (r["patient_id"], r["index_hadm_id"])
            if key not in tensors_by_key_test:
                continue
            if is_episode:
                occ = find_episode_occurrence(r["combined_full"], r["combined_map"], episode_pat)
            else:
                occ = find_subsequence_occurrence(r["combined_full"], r["combined_map"],
                                                  flat_pat, contiguous=contiguous)
            if occ:
                matches.append((key, occ))
            if len(matches) >= max_adm_per_pattern:
                break

        if not matches:
            res.append({
                "pattern": pat_str, "pattern_readable": pat_readable,
                "n_matched_test": 0,
                "mean_delta_mask": np.nan,
                "mean_delta_shuffle_within": np.nan,
                "mean_delta_shuffle_visits": np.nan,
                "pct_positive_delta_mask": np.nan,
                "pct_positive_delta_shuffle_within": np.nan,
                "pct_positive_delta_shuffle_visits": np.nan,
            })
            print(f"\r  [validate] Pattern {pi+1}/{n_patterns}: 0 matches", end="", flush=True)
            continue

        # ── Batched base predictions ────────────────────────────────────
        base_tensors = [tensors_by_key_test[key] for key, _ in matches]
        base_probs = predict_batch(model, device, base_tensors)

        # ── Batched masked predictions ──────────────────────────────────
        masked_tensors = []
        for (key, occ), base_t in zip(matches, base_tensors):
            positions = [pos for pos, _ in occ]
            masked_tensors.append(mask_positions(base_t, positions))
        masked_probs = predict_batch(model, device, masked_tensors)

        dm = base_probs - masked_probs

        # ── Batched within-visit shuffle predictions ─────────────────────
        all_shuffled_within = []
        n_matches = len(matches)
        for (key, occ), base_t in zip(matches, base_tensors):
            for _ in range(n_shuffle_draws):
                all_shuffled_within.append(shuffle_within_visits(base_t, vocab, vocab_inv))

        shuffled_within_probs = predict_batch(model, device, all_shuffled_within)
        shuffled_within_probs = shuffled_within_probs.reshape(n_matches, n_shuffle_draws)
        ds_within = base_probs[:, None] - shuffled_within_probs
        ds_within_mean = ds_within.mean(axis=1)

        # ── Batched visit-block shuffle predictions ──────────────────────
        # This is the key temporal test: does cross-visit order matter?
        all_shuffled_visits = []
        for (key, occ), base_t in zip(matches, base_tensors):
            for _ in range(n_shuffle_draws):
                all_shuffled_visits.append(shuffle_visit_blocks(base_t))

        shuffled_visit_probs = predict_batch(model, device, all_shuffled_visits)
        shuffled_visit_probs = shuffled_visit_probs.reshape(n_matches, n_shuffle_draws)
        ds_visit = base_probs[:, None] - shuffled_visit_probs
        ds_visit_mean = ds_visit.mean(axis=1)

        res.append({
            "pattern": pat_str, "pattern_readable": pat_readable,
            "odds_ratio": float(row.get("odds_ratio", np.nan)),
            "ig_direction": float(row.get("_ig_score", 0.0)),
            "n_matched_test": len(matches),
            "mean_delta_mask": float(dm.mean()),
            "mean_delta_shuffle_within": float(ds_within_mean.mean()),
            "mean_delta_shuffle_visits": float(ds_visit_mean.mean()),
            "pct_positive_delta_mask": float((dm > 0).mean() * 100),
            "pct_positive_delta_shuffle_within": float((ds_within_mean > 0).mean() * 100),
            "pct_positive_delta_shuffle_visits": float((ds_visit_mean > 0).mean() * 100),
            # Raw predictions for OOD diagnostic
            "_base_probs": base_probs.tolist(),
            "_masked_probs": masked_probs.tolist(),
            "_shuffled_within_probs_flat": shuffled_within_probs.flatten().tolist(),
            "_shuffled_visit_probs_flat": shuffled_visit_probs.flatten().tolist(),
        })
        print(f"\r  [validate] Pattern {pi+1}/{n_patterns}: {len(matches)} matches", end="", flush=True)

    print()
    dfv = pd.DataFrame(res)
    if not dfv.empty:
        dfv = dfv.sort_values("mean_delta_mask", ascending=False)
    return dfv


# ============================================================================
# REVERSED-ORDER ANALYSIS
# ============================================================================


def run_reversed_order_analysis(
    df_patterns, df_seq, df_ig, mining_method,
    icd_titles=None, top_k=15, pattern_whitelist=None,
) -> pd.DataFrame:
    """Compare forward vs reversed pattern matching to test temporal specificity.

    For each multi-token pattern {A} → {B}, finds:
      - Patients matching forward order (A before B)
      - Patients matching reversed order (B before A)
      - Mean Σ IG_i for each group

    If the model is sensitive to temporal order, forward-matched patients
    should have different (typically higher) Σ IG_i than reverse-matched.

    If forward ≈ reversed, the model treats the codes as bag-of-tokens
    regardless of ordering.
    """
    from atop.explain.matching import PatternIndex
    from atop.explain.figures import (
        _build_ig_cache_by_block, _score_pattern_instance, _count_pattern_tokens,
    )

    is_episode = (mining_method == "episode")
    if not is_episode:
        print("  [reversed] Skipping — only implemented for episode mining")
        return pd.DataFrame()

    # Filter to multi-token patterns only (reversal is meaningless for single tokens)
    multi_mask = df_patterns["pattern"].apply(_count_pattern_tokens) >= 2
    df_multi = df_patterns[multi_mask].copy()
    if df_multi.empty:
        return pd.DataFrame()

    # If whitelist provided (from fig3), use only those patterns
    if pattern_whitelist is not None:
        df_multi = df_multi[df_multi["pattern"].isin(pattern_whitelist)].copy()
        print(f"  [reversed] Using fig3 whitelist: {len(df_multi)} multi-token patterns")
        if df_multi.empty:
            return pd.DataFrame()
        top = df_multi  # No further selection needed
    else:
        # Build instance-based cache for IG scoring
        block_cache = _build_ig_cache_by_block(df_ig, df_seq)
        pat_index_sel = PatternIndex.from_df(df_seq)
        seq_rows_sel = df_seq.reset_index(drop=True)

        # Quick IG scoring for selection (instance-based)
        ig_scores = []
        for _, pat_row in df_multi.iterrows():
            ep = _parse_episode_pattern(pat_row["pattern"])
            pat_fsets = [frozenset(s) for s in ep]
            pat_tokens = set()
            for s in ep:
                pat_tokens.update(s)
            matches_wb = pat_index_sel.patients_matching_with_blocks(pat_fsets, pat_tokens, True)
            if not matches_wb:
                ig_scores.append(0.0)
                continue
            sums = []
            for ri, mb in matches_wb:
                row = seq_rows_sel.iloc[ri]
                key = (str(row["patient_id"]), str(row["index_hadm_id"]))
                bl = block_cache.get(key, [])
                if bl:
                    sums.append(_score_pattern_instance(pat_fsets, mb, bl))
            ig_scores.append(float(np.mean(sums)) if sums else 0.0)

        df_multi = df_multi.copy()
        df_multi["_ig_score"] = ig_scores
        df_multi["_ig_abs"] = df_multi["_ig_score"].abs()
        top_risk = df_multi[df_multi["_ig_score"] > 0].nlargest(top_k, "_ig_abs")
        top_prot = df_multi[df_multi["_ig_score"] < 0].nlargest(top_k, "_ig_abs")
        top = pd.concat([top_risk, top_prot], ignore_index=True)

    print(f"  [reversed] Analyzing {len(top)} patterns (forward vs reversed)...")

    # Build caches needed for forward/reverse matching and IG scoring
    block_cache = _build_ig_cache_by_block(df_ig, df_seq)
    pat_index = PatternIndex.from_df(df_seq)
    seq_rows = df_seq.reset_index(drop=True)

    results = []
    for pi, (_, pat_row) in enumerate(top.iterrows()):
        pat_str = pat_row["pattern"]
        ep = _parse_episode_pattern(pat_str)
        pat_fsets = [frozenset(s) for s in ep]
        pat_tokens = set()
        for s in ep:
            pat_tokens.update(s)

        # Reversed pattern: reverse the itemset order
        rev_fsets = list(reversed(pat_fsets))

        # Find patients matching forward (with block indices)
        fwd_matches = pat_index.patients_matching_with_blocks(pat_fsets, pat_tokens, True)
        # Find patients matching reversed
        rev_matches = pat_index.patients_matching_with_blocks(rev_fsets, pat_tokens, True)

        fwd_set = set(ri for ri, _ in fwd_matches)
        rev_set = set(ri for ri, _ in rev_matches)
        both_set = fwd_set & rev_set
        fwd_only = fwd_set - rev_set
        rev_only = rev_set - fwd_set

        def _mean_ig_instance(matches_wb, pat_fs):
            sums = []
            for ri, mb in matches_wb:
                row = seq_rows.iloc[ri]
                key = (str(row["patient_id"]), str(row["index_hadm_id"]))
                bl = block_cache.get(key, [])
                if bl:
                    sums.append(_score_pattern_instance(pat_fs, mb, bl))
            return float(np.mean(sums)) if sums else np.nan

        def _mean_ig_subset(matches_wb, pat_fs, subset):
            filtered = [(ri, mb) for ri, mb in matches_wb if ri in subset]
            return _mean_ig_instance(filtered, pat_fs)

        ig_fwd = _mean_ig_instance(fwd_matches, pat_fsets)
        ig_rev = _mean_ig_instance(rev_matches, rev_fsets)
        ig_fwd_only = _mean_ig_subset(fwd_matches, pat_fsets, fwd_only)
        ig_rev_only = _mean_ig_subset(rev_matches, rev_fsets, rev_only)

        _titles = icd_titles or {}
        from atop.explain.figures import _fmt_pat
        pat_readable = _fmt_pat(pat_str, _titles, max_tok_len=80)
        rev_str = " -> ".join(
            "{" + ", ".join(sorted(fs)) + "}" if len(fs) > 1 else sorted(fs)[0]
            for fs in rev_fsets
        )

        results.append({
            "pattern": pat_str,
            "pattern_readable": pat_readable,
            "reversed_pattern": rev_str,
            "n_forward": len(fwd_matches),
            "n_reversed": len(rev_matches),
            "n_both": len(both_set),
            "n_forward_only": len(fwd_only),
            "n_reversed_only": len(rev_only),
            "ig_mean_forward": ig_fwd,
            "ig_mean_reversed": ig_rev,
            "ig_mean_forward_only": ig_fwd_only,
            "ig_mean_reversed_only": ig_rev_only,
            "ig_diff": ig_fwd - ig_rev if not (np.isnan(ig_fwd) or np.isnan(ig_rev)) else np.nan,
            "odds_ratio": float(pat_row.get("odds_ratio", np.nan)),
        })

        if (pi + 1) % 5 == 0 or (pi + 1) == len(top):
            print(f"\r    Progress: {pi+1}/{len(top)}", end="", flush=True)
    print()

    df_rev = pd.DataFrame(results)
    return df_rev


# ============================================================================
# FIGURES
# ============================================================================
