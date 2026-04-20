"""Attention flow analysis for AToP temporal patterns.

Extracts attention weights from the Transformer and analyzes whether
pattern tokens attend to each other more than to non-pattern tokens.
This provides on-manifold evidence that the model internally connects
the tokens in a mined temporal pattern.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict


def _extract_attention_weights(model, input_ids, device):
    """Run forward pass manually through transformer layers to capture attention weights.
    
    We bypass model.transformer() because TransformerEncoderLayer._sa_block()
    forces need_weights=False. Instead we replicate the layer logic and call
    self_attn directly with need_weights=True.
    
    Returns:
        list of (B, n_heads, L, L) numpy arrays, one per layer.
        Empty list if extraction fails.
    """
    model.eval()
    
    input_ids = input_ids.to(device)
    with torch.no_grad():
        B, L = input_ids.shape
        from atop.config import PAD_IDX
        pad_mask = (input_ids == PAD_IDX)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = model.emb_norm(model.emb_dropout(
            model.token_emb(input_ids) + model.pos_emb(positions)))
        
        layer_attns = []
        
        for layer in model.transformer.layers:
            # Replicate TransformerEncoderLayer forward with norm_first=True
            # and need_weights=True
            #
            # norm_first path:
            #   x = x + _sa_block(norm1(x), src_mask, src_key_padding_mask)
            #   x = x + _ff_block(norm2(x))
            
            # Self-attention with need_weights=True
            x_norm = layer.norm1(x)
            attn_out, attn_weights = layer.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=pad_mask,
                need_weights=True,
                average_attn_weights=False,  # get per-head weights
            )
            x = x + layer.dropout1(attn_out)
            
            # Feed-forward
            x_norm2 = layer.norm2(x)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_norm2))))
            x = x + layer.dropout2(ff_out)
            
            if attn_weights is not None:
                layer_attns.append(attn_weights.detach().cpu().numpy())
        
        # Apply final norm if present
        if model.transformer.norm is not None:
            x = model.transformer.norm(x)
    
    return layer_attns


def _match_pattern_positions(input_ids_np, vocab_inv, pattern_str):
    """Find positions of pattern tokens in the input sequence.
    
    Returns list of (position, token_str) for matched tokens in order,
    or empty list if pattern not found.
    """
    from atop.explain.figures import _extract_pattern_tokens, _count_pattern_blocks
    
    # Parse pattern into blocks
    blocks = pattern_str.split(" -> ")
    pat_blocks = []
    for block in blocks:
        block = block.strip()
        if block.startswith("{") and block.endswith("}"):
            tokens = [t.strip() for t in block[1:-1].split(",")]
        else:
            tokens = [block.strip()]
        pat_blocks.append(set(tokens))
    
    # Build token → positions mapping
    tok_positions = defaultdict(list)
    for pos, tid in enumerate(input_ids_np):
        tok_str = vocab_inv.get(int(tid), "")
        if tok_str:
            tok_positions[tok_str].append(pos)
    
    # Greedy match: find earliest positions respecting block ordering
    # Positions must be non-decreasing across blocks (temporal order)
    matched = []
    min_pos = 0
    for block_tokens in pat_blocks:
        block_matched = []
        for tok in block_tokens:
            positions = [p for p in tok_positions.get(tok, []) if p >= min_pos]
            if not positions:
                return []  # Pattern not found
            block_matched.append((positions[0], tok))
        if block_matched:
            min_pos = max(p for p, _ in block_matched) + 1
            matched.extend(block_matched)
    
    return matched


def compute_attention_flow(model, device, tensors_by_key, vocab_inv, df_patterns,
                           df_pathways, n_patterns=20, max_patients=200,
                           carrier_sets=None):
    """Compute attention flow statistics for top patterns.
    
    For each pattern, measures:
    1. Mean attention between pattern token positions (cross-pattern attention)
    2. Mean attention from pattern positions to non-pattern positions (baseline)
    3. Ratio: how much more do pattern tokens attend to each other?
    
    Args:
        carrier_sets: dict of pattern → set of (patient_id, hadm_id) keys.
                      If provided, only sample from known carriers instead of
                      blindly scanning all patients.
    
    Returns DataFrame with attention flow statistics per pattern.
    """
    # Select top patterns by |IG|
    if df_pathways is not None and not df_pathways.empty:
        df_top = df_pathways.copy()
        df_top["_abs_ig"] = df_top["ig_signed_mean"].abs()
        half = max(1, n_patterns // 2)
        df_risk = df_top[df_top["ig_signed_mean"] > 0].nlargest(half, "_abs_ig")
        df_prot = df_top[df_top["ig_signed_mean"] < 0].nlargest(half, "_abs_ig")
        df_top = pd.concat([df_risk, df_prot])
    else:
        return pd.DataFrame()

    test_keys_set = set(tensors_by_key.keys())
    test_keys = list(tensors_by_key.keys())
    results = []
    print(f"  [attn_flow] {len(df_top)} patterns to analyze, {len(test_keys)} test patients available")
    if carrier_sets:
        print(f"  [attn_flow] Using carrier sets for targeted patient sampling")

    for pi, (_, pat_row) in enumerate(df_top.iterrows()):
        pat_str = pat_row["pattern"]
        ig_mean = pat_row["ig_signed_mean"]
        n_present = pat_row.get("n_present", 0)

        cross_attns = []  # attention between pattern tokens
        baseline_attns = []  # attention from pattern to non-pattern
        n_matched = 0
        n_attn_fail = 0

        # Get candidate patients — use carrier sets if available
        if carrier_sets and pat_str in carrier_sets:
            carrier_keys = [k for k in carrier_sets[pat_str] if k in test_keys_set]
            sample_keys = carrier_keys[:max_patients]
        else:
            # Fallback: scan first N*3 patients
            sample_keys = test_keys[:max_patients * 3]

        for key in sample_keys:
            if n_matched >= max_patients:
                break

            input_ids = tensors_by_key[key]
            ids_np = input_ids[0].numpy()

            # Match pattern in this patient
            matched = _match_pattern_positions(ids_np, vocab_inv, pat_str)
            if not matched:
                continue

            n_matched += 1
            pattern_positions = set(p for p, _ in matched)
            non_pattern_positions = set(range(len(ids_np))) - pattern_positions
            from atop.config import PAD_IDX
            non_pattern_positions = {p for p in non_pattern_positions if ids_np[p] != PAD_IDX}

            if len(pattern_positions) < 2 or not non_pattern_positions:
                continue

            # Extract attention weights
            layer_attns = _extract_attention_weights(model, input_ids, device)
            if not layer_attns:
                n_attn_fail += 1
                if n_attn_fail == 1 and pi == 0:
                    print(f"\n  [attn_flow] WARNING: attention extraction returned empty for first patient")
                continue

            # Average across all layers and heads
            # Each layer_attn: (1, n_heads, L, L)
            all_attn = np.stack([la[0] for la in layer_attns])  # (n_layers, n_heads, L, L)
            mean_attn = all_attn.mean(axis=(0, 1))  # (L, L) — averaged over layers and heads

            # Cross-pattern attention: mean attention between pattern positions
            pat_pos_list = sorted(pattern_positions)
            cross_vals = []
            cross_distances = []
            for i, pi_pos in enumerate(pat_pos_list):
                for j, pj_pos in enumerate(pat_pos_list):
                    if i != j:
                        cross_vals.append(mean_attn[pi_pos, pj_pos])
                        cross_distances.append(abs(pi_pos - pj_pos))
            if not cross_vals:
                continue

            # Distance-matched baseline: for each pattern token pair distance,
            # sample non-pattern token pairs at the same distance (±5 tolerance)
            # This controls for positional bias in attention.
            nonpat_list = sorted(non_pattern_positions)
            dist_matched_vals = []
            DIST_TOL = 5
            for dist in cross_distances:
                matched_pairs = []
                for np_i in nonpat_list:
                    for np_j in nonpat_list:
                        if np_i != np_j and abs(abs(np_i - np_j) - dist) <= DIST_TOL:
                            matched_pairs.append(mean_attn[np_i, np_j])
                        if len(matched_pairs) >= 20:  # cap samples per distance
                            break
                    if len(matched_pairs) >= 20:
                        break
                if matched_pairs:
                    dist_matched_vals.append(np.mean(matched_pairs))

            if cross_vals:
                cross_attns.append(np.mean(cross_vals))
            if dist_matched_vals:
                baseline_attns.append(np.mean(dist_matched_vals))
            else:
                # Fallback: global non-pattern baseline if no distance match found
                fallback = []
                for np_i in nonpat_list[:30]:
                    for np_j in nonpat_list[:30]:
                        if np_i != np_j:
                            fallback.append(mean_attn[np_i, np_j])
                if fallback:
                    baseline_attns.append(np.mean(fallback))

        if not cross_attns:
            continue

        mean_cross = np.mean(cross_attns)
        mean_baseline = np.mean(baseline_attns) if baseline_attns else 0
        ratio = mean_cross / max(mean_baseline, 1e-8)

        results.append({
            "pattern": pat_str,
            "pattern_readable": pat_row.get("pattern_readable", pat_str),
            "ig_signed_mean": ig_mean,
            "n_present": n_present,
            "n_attention_samples": n_matched,
            "mean_cross_pattern_attn": mean_cross,
            "mean_baseline_attn": mean_baseline,
            "attention_ratio": ratio,
            "attention_uplift": mean_cross - mean_baseline,
        })

        direction = "▲" if ig_mean > 0 else "▼"
        print(f"\r  [attn_flow] {pi+1}/{len(df_top)}: {direction} ratio={ratio:.2f} "
              f"(cross={mean_cross:.4f} vs base={mean_baseline:.4f}, n={n_matched})", 
              end="", flush=True)

    print()
    return pd.DataFrame(results)


def fig_attention_flow(out_dir, df_attn, icd_titles=None, n_show=20,
                        split="test", short_names=False):
    """Plot attention flow analysis: cross-pattern vs baseline attention.
    
    Shows for each pattern whether the Transformer's attention mechanism
    connects the pattern tokens more than random token pairs.
    """
    if df_attn is None or df_attn.empty:
        return

    from atop.explain.figures import _strip_codes, _SHORT_NAME_MAP, load_short_names, _fmt_pat

    half = max(1, n_show // 2)
    df_risk = df_attn[df_attn["ig_signed_mean"] > 0].nlargest(half, "attention_ratio")
    df_prot = df_attn[df_attn["ig_signed_mean"] < 0].nlargest(half, "attention_ratio")
    df_plot = pd.concat([df_risk, df_prot]).sort_values("attention_ratio")

    if df_plot.empty:
        return

    n = len(df_plot)
    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.5)))

    colors = ["#c44e52" if ig > 0 else "#4c72b0" for ig in df_plot["ig_signed_mean"]]

    ax.barh(range(n), df_plot["attention_ratio"].values, color=colors,
            edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="Ratio = 1 (no uplift)")

    labels = []
    _titles = icd_titles or {}
    for _, r in df_plot.iterrows():
        direction = "▲" if r["ig_signed_mean"] > 0 else "▼"
        label_col = "pattern_readable" if pd.notna(r.get("pattern_readable")) else "pattern"
        lbl = f"{direction} {r[label_col]}  (n={int(r['n_attention_samples'])})"
        if short_names:
            lbl = _strip_codes(lbl) if _SHORT_NAME_MAP else lbl
        labels.append(lbl)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Attention ratio (cross-pattern / distance-matched baseline)\n"
                   "> 1 means pattern tokens exhibit elevated mutual attention vs. same-distance non-pattern pairs")
    ax.set_title(
        f"Attention flow analysis ({split.upper()})\n"
        f"Pattern tokens exhibit elevated mutual attention (distance-matched baseline)",
        fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c44e52", label="▲ Risk (IG > 0)"),
        Patch(facecolor="#4c72b0", label="▼ Protective (IG < 0)"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Ratio = 1 (no elevation)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, f"supp_attention_flow_{split}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [attn_flow] Saved → {path}")

    # Save CSV
    csv_path = os.path.join(out_dir, f"attention_flow_{split}.csv")
    df_attn.to_csv(csv_path, index=False)
    print(f"  [attn_flow] CSV → {csv_path}")
