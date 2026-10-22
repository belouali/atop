"""All figure/table generation for AToP reports."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
# torch is imported lazily inside the few functions that actually use it
# (SHAP panel in fig6_global_importance_comparison). Keeps this module
# importable without torch for the torch-free paths (reversed-order,
# label/helper utilities).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from atop.config import PAD_IDX, SPECIAL_TOKENS
from atop.utils import (
    format_token_readable, truncate_label, stream_color,
    token_stream, predict_one, auroc_np, pr_auc_np,
    roc_curve_np, pr_curve_np,
)
from atop.mining.patterns import _parse_episode_pattern, _is_subpattern, _is_subsequence
from atop.explain.scoring import (
    build_patient_token_occurrences, build_presence_index, candidate_rows,
    episode_to_itemsets, match_and_score_episode,
)
from atop.explain.matching import (
    _match_all_patterns, _compute_pattern_mass, _extract_pattern_tokens,
    PatternIndex,
)
from atop.explain.label_utils import (
    format_token_short, format_pattern_short, shorten_title,
    active_stream_legend,
)


# Aliases
_truncate_label = truncate_label
_stream_color = stream_color
_token_stream = token_stream
_fmt_tok = format_token_short  # shorter alias
_fmt_pat = format_pattern_short


_SHORT_NAME_MAP = {}  # populated by load_short_names()

def load_short_names(csv_path=None):
    """Load short name mapping from CSV. Call once at startup."""
    global _SHORT_NAME_MAP
    if csv_path is None:
        # Default: look next to the package
        import importlib.resources
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "paper", "jamia2026", "figures", "short_names.csv")
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _SHORT_NAME_MAP[row["token_readable"].strip()] = row["short_name"].strip()
    return _SHORT_NAME_MAP


def _strip_codes(label):
    """Strip clinical codes from formatted labels, using short name mapping if available.
    
    With mapping: 'C:F329 (MDD, single ep.)' → 'MDD'
    Without mapping: 'C:F329 (MDD, single ep.)' → 'MDD, single ep.'
    """
    import re

    def _shorten_one(tok):
        tok = tok.strip()
        # Try exact match in short name map first
        if tok in _SHORT_NAME_MAP:
            return _SHORT_NAME_MAP[tok]
        # Try code-based match: extract "C:F329" from "C:F329 (MDD...)" and match
        m_code = re.match(r'^([CPD]:[A-Za-z0-9_.]+)', tok)
        if m_code:
            code_prefix = m_code.group(1)
            for long_name, short in _SHORT_NAME_MAP.items():
                if long_name.startswith(code_prefix + " ") or long_name == code_prefix:
                    return short
        # Try matching the description part (after stripping C:CODE prefix)
        desc = tok
        m_desc = re.match(r'^[CPD]:[A-Za-z0-9_.]+ \((.+?)[\)]*$', tok)
        if m_desc:
            desc = m_desc.group(1).rstrip(')')
        # Try exact match on description
        if desc in _SHORT_NAME_MAP:
            return _SHORT_NAME_MAP[desc]
        # Try prefix match on description (handles truncation)
        if len(desc) > 15:
            for long_name, short in _SHORT_NAME_MAP.items():
                if long_name.startswith(desc[:15]) and abs(len(long_name) - len(desc)) < 10:
                    return short
        # Fallback: strip prefix and code, return raw description
        if tok.startswith("D:"):
            return tok[2:]
        if m_desc:
            return desc
        m = re.match(r'^[CPD]:[A-Za-z0-9_.]+ \((.+)$', tok)
        if m:
            return m.group(1).rstrip(')')
        if tok.startswith(("C:", "P:", "D:")):
            return tok[2:]
        return tok

    parts = label.split("\u2192")
    short_parts = []
    for part in parts:
        part = part.strip()
        if part.startswith("{") and part.endswith("}"):
            inner = part[1:-1]
            # Smart split: don't split on commas inside parentheses
            tokens = re.split(r',\s*(?=[CPD]:)', inner)
            tokens = [t.strip() for t in tokens]
            short_parts.append("{" + ", ".join(_shorten_one(t) for t in tokens) + "}")
        else:
            short_parts.append(_shorten_one(part))
    return " \u2192 ".join(short_parts)


def _count_pattern_tokens(pat_str: str) -> int:
    """Count unique tokens in a pattern string (episode or flat)."""
    tokens = set(_extract_pattern_tokens(pat_str))
    return len(tokens)


def _count_pattern_blocks(pat_str: str) -> int:
    """Count the number of visit blocks (itemsets) in a pattern string.

    '{A, B} -> {C}' has 2 blocks; 'A -> B -> C' has 3 blocks.
    A single-block pattern like '{A, B}' or 'A' has 1 block.
    """
    parts = pat_str.split(" -> ")
    return len(parts)


def _filter_multi_token(df_patterns: pd.DataFrame, min_tokens: int = 2) -> pd.DataFrame:
    """Filter patterns to those with at least min_tokens unique tokens."""
    if df_patterns.empty:
        return df_patterns
    mask = df_patterns["pattern"].apply(lambda s: _count_pattern_tokens(s) >= min_tokens)
    return df_patterns[mask].copy()


def _filter_cross_visit(df_patterns: pd.DataFrame, min_blocks: int = 2) -> pd.DataFrame:
    """Filter patterns to those spanning at least min_blocks visit blocks.

    This eliminates same-visit co-occurrence patterns (e.g. drug pairs
    within a single admission) and retains only cross-visit temporal
    progressions that demonstrate genuine sequential structure.
    """
    if df_patterns.empty:
        return df_patterns
    mask = df_patterns["pattern"].apply(lambda s: _count_pattern_blocks(s) >= min_blocks)
    return df_patterns[mask].copy()


# ── IG cache helpers ─────────────────────────────────────────────────────

def _build_ig_cache(df_ig: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Build per-patient IG cache with raw signed IG values.
    
    Primary metric: raw signed IG preserves completeness property.
    IG(P) = Σ IG_i for i in P gives additive pattern contribution
    (positive = pushes toward readmission, negative = protective).
    
    If a token appears at multiple positions (e.g. HTN in visit 1 and visit 3),
    we sum the IG values across all occurrences. This is correct because
    the completeness property operates over positions, not unique tokens:
      f(x) - f(baseline) = Σ_{pos} IG_{pos}
    
    Returns:
        {(str_pid, str_hadm): {token_str: sum_of_signed_ig_across_positions}}
    """
    ig_col = "ig_signed" if "ig_signed" in df_ig.columns else "ig_abs"
    cache = {}
    for (pid, hid), grp in df_ig.groupby(["patient_id", "index_hadm_id"]):
        # Sum IG across all positions for each token_str
        tok_ig = grp.groupby("token_str")[ig_col].sum().to_dict()
        cache[(str(pid), str(hid))] = tok_ig
    return cache


def _build_ig_cache_by_block(df_ig: pd.DataFrame, df_seq: pd.DataFrame
                             ) -> Dict[Tuple[str, str], List[Dict[str, float]]]:
    """Build per-patient, per-salient-visit-block IG cache for instance-based scoring.

    Uses visit_idx from df_ig (position-based, computed during IG) and
    visit_to_block from df_seq (maps raw visit_idx → salient block ordinal)
    to assign each IG value to the correct salient visit block unambiguously.

    Only tokens that land in a salient block are included, ensuring scoring
    consistency with the same token universe used for mining.

    Returns:
        {(str_pid, str_hadm): [
            {token_str: signed_ig, ...},   # salient block 0
            {token_str: signed_ig, ...},   # salient block 1
            ...
        ]}
    """
    ig_col = "ig_signed" if "ig_signed" in df_ig.columns else "ig_abs"
    has_visit_idx = "visit_idx" in df_ig.columns
    has_salient = "is_salient" in df_ig.columns
    if not has_visit_idx:
        print("  [WARN] df_ig missing visit_idx, instance-based scoring unavailable. "
              "Re-run IG computation to enable position-aware scoring.")
    if not has_salient:
        print("  [WARN] df_ig missing is_salient, block cache includes all tokens in salient visits. "
              "Re-run IG computation to restrict to salient tokens only.")

    # Build visit_to_block lookup per patient
    v2b_lookup = {}
    for _, row in df_seq.iterrows():
        key = (str(row["patient_id"]), str(row["index_hadm_id"]))
        v2b = row.get("visit_to_block")
        n_blocks = len(row.get("salient_visit_blocks", []))
        v2b_lookup[key] = (v2b if v2b else {}, n_blocks)

    cache = {}
    for (pid, hid), grp in df_ig.groupby(["patient_id", "index_hadm_id"]):
        key = (str(pid), str(hid))
        v2b, n_blocks = v2b_lookup.get(key, ({}, 0))
        if n_blocks == 0:
            cache[key] = []
            continue

        block_ig: List[Dict[str, float]] = [{} for _ in range(n_blocks)]

        for _, row in grp.iterrows():
            tok = row["token_str"]
            ig_val = float(row[ig_col])
            vidx = int(row["visit_idx"]) if has_visit_idx else -1

            if vidx < 0:
                continue
            # Skip non-salient tokens, ensures scoring uses the same
            # token universe as mining/matching
            if has_salient and not row["is_salient"]:
                continue
            bi = v2b.get(vidx)
            if bi is None:
                # Token's visit is not a salient block, skip
                continue

            if tok in block_ig[bi]:
                block_ig[bi][tok] += ig_val
            else:
                block_ig[bi][tok] = ig_val

        cache[key] = block_ig
    return cache


def _score_pattern_instance(pattern_fsets, matched_blocks, block_ig_list):
    """Compute instance-based IG for a pattern match.

    Args:
        pattern_fsets: List[frozenset], the pattern's itemsets
        matched_blocks: List[(pattern_idx, block_idx)] from _match_subpattern
        block_ig_list: List[Dict[str, float]], per-block IG for this patient

    Returns:
        float: sum of signed IG for the tokens in the matched blocks only
    """
    total = 0.0
    for pi, bi in matched_blocks:
        if bi >= len(block_ig_list):
            continue
        block_ig = block_ig_list[bi]
        pset = pattern_fsets[pi]
        for tok in pset:
            total += block_ig.get(tok, 0.0)
    return total


def _compute_pattern_ig_stats(ig_cache, pat_tokens, matched_rows, seq_rows,
                              odds_ratio=None):
    """Compute pattern IG statistics across matched patients.
    
    For each matched patient, computes:
      - Signed sum: Σ IG_i for i in P (additive, primary metric)
      - Abs share: Σ|IG_i| for i in P / Σ|IG_j| for all j (descriptive)
      - Conditional impact: E[ΣIG | present, IG concordant with OR direction]
        (surfaces rare-but-powerful patterns)
    
    Returns dict with ig_signed_mean, ig_signed_median, ig_abs_share_mean,
    ig_cond_impact, n_concordant, n_matched, or None if no matches.
    """
    signed_sums = []
    abs_shares = []
    
    for row_i in matched_rows:
        row = seq_rows.iloc[row_i]
        key = (str(row["patient_id"]), str(row["index_hadm_id"]))
        tok_ig = ig_cache.get(key, {})
        if not tok_ig:
            continue
        
        # Signed sum: additive pattern contribution
        pat_vals = [tok_ig.get(t, 0.0) for t in pat_tokens if t in tok_ig]
        if not pat_vals:
            continue
        signed_sum = sum(pat_vals)
        signed_sums.append(signed_sum)
        
        # Abs share: fraction of total attribution magnitude (descriptive)
        pat_abs = sum(abs(v) for v in pat_vals)
        total_abs = sum(abs(v) for v in tok_ig.values()) + 1e-12
        abs_shares.append(pat_abs / total_abs)
    
    if not signed_sums:
        return None
    
    # Conditional impact: only patients where IG aligns with OR direction
    is_risk = (odds_ratio is not None and odds_ratio > 1)
    if odds_ratio is not None:
        if is_risk:
            concordant = [s for s in signed_sums if s > 0]
        else:
            concordant = [s for s in signed_sums if s < 0]
        n_concordant = len(concordant)
        ig_cond = float(np.mean(concordant)) if concordant else 0.0
    else:
        n_concordant = len(signed_sums)
        ig_cond = float(np.mean(signed_sums))

    return {
        "ig_signed_mean": float(np.mean(signed_sums)),
        "ig_signed_median": float(np.median(signed_sums)),
        "ig_abs_share_mean": float(np.mean(abs_shares)),
        "ig_cond_impact": ig_cond,
        "n_concordant": n_concordant,
        "n_matched": len(signed_sums),
    }


def _make_pattern_readable(pat_str: str, icd_titles: Dict, max_tok_len: int = 30) -> str:
    """Make a pattern string readable, handling both flat and episode formats."""
    parts = pat_str.split(" -> ")
    readable_parts = []
    for p in parts:
        p = p.strip()
        if p.startswith("{") and p.endswith("}"):
            # Episode itemset: {item1, item2, ...}
            items = [x.strip() for x in p[1:-1].split(",")]
            readable_items = [_truncate_label(format_token_readable(it, icd_titles), max_tok_len) for it in items]
            readable_parts.append("{" + ", ".join(readable_items) + "}")
        else:
            readable_parts.append(_truncate_label(format_token_readable(p, icd_titles), max_tok_len))
    return " → ".join(readable_parts)



def fig1_dataset_performance(out_dir, summary, test_metrics, y_true, y_prob):
    fpr, tpr = roc_curve_np(y_true, y_prob)
    recall, precision = pr_curve_np(y_true, y_prob)
    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.bar(["No readmit", "Readmit"], [summary["n_no_readmit"], summary["n_readmit"]])
    ax1.set_title("Readmission label counts")
    ax1.set_ylabel("Samples")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(fpr, tpr)
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_title("ROC curve")
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(recall, precision)
    ax3.set_title("Precision-Recall curve")
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")

    fig.suptitle(f"Test AUROC={test_metrics['auroc']:.3f} | PR-AUC={test_metrics['pr_auc']:.3f} | "
                 f"Readmission rate={summary['readmission_rate']*100:.1f}% | "
                 f"Multi-visit: {summary['pct_multi_visit']:.1f}%")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(os.path.join(out_dir, "fig1_dataset_performance.png"), dpi=300)
    plt.close(fig)



# ── Removed (Apr 2026 cleanup) ──────────────────────────────────────────────
# fig2_patient_ig (~55 LOC) and fig3_top_patterns (~130 LOC) used to live
# here. Both were AMIA-era single-panel figures replaced by make_fig2.py
# (3-panel patient exemplar) and make_fig3.py (3-panel global comparison)
# respectively. Neither had any live caller in atop_package/.


# ============================================================================
# SHAP COMPARISON FIGURE (Fig 4)
# ============================================================================


def build_shap_comparison_figure(out_dir, model, device, icd_titles, pick, input_ids_tensor,
                                 vocab_inv, ig, df_patterns_train, df_seq_test,
                                 shap_max_features, shap_nsamples, mining_method,
                                 split="test", csv_dir=None, filename_suffix=""):
    """Figure 2: Single-patient explanation (SHAP vs IG vs visit timeline vs patterns).
    
    Uses KernelSHAP for local per-patient explanation.
    filename_suffix: appended to output filenames (e.g., "_ex2" for 2nd exemplar)
    """
    if csv_dir is None:
        csv_dir = out_dir
    import shap

    is_episode = (mining_method == "episode")
    method_labels = {"prefixspan": "PrefixSpan", "ngram": "n-gram", "episode": "Episode"}
    method_label = method_labels.get(mining_method, mining_method)

    # ── IG for this patient ──────────────────────────────────────────────
    attr = ig.attribute(input_ids_tensor.to(device), target_class=1)
    attr_np = attr[0].detach().cpu().numpy()
    ids_np = input_ids_tensor[0].numpy()

    items = []
    for pos in range(len(ids_np)):
        tid = ids_np[pos]
        if tid == PAD_IDX:
            continue
        tok_str = vocab_inv.get(tid, f"UNK_{tid}")
        if tok_str in SPECIAL_TOKENS:
            continue
        ig_signed = float(attr_np[pos])
        igv = abs(ig_signed)
        items.append({
            "pos": pos, "token_str": tok_str,
            "token_str_readable": format_token_readable(tok_str, icd_titles),
            "ig_abs": igv, "ig_signed": ig_signed,
        })
    items.sort(key=lambda d: d["ig_abs"], reverse=True)
    K = min(shap_max_features, len(items))
    items_sel = items[:K]
    if K == 0:
        return

    # ── SHAP computation ─────────────────────────────────────────────────
    base_ids = input_ids_tensor.clone()

    # Clear GPU cache before SHAP
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def predict_masked(z):
        batch = []
        for row in z:
            inp = base_ids.clone()
            for j, keep in enumerate(row.tolist()):
                if keep < 0.5:
                    inp[0, items_sel[j]["pos"]] = PAD_IDX
            batch.append(inp)
        stacked = torch.cat(batch, dim=0)
        # Chunked forward to avoid OOM
        CHUNK = 32
        all_probs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(stacked), CHUNK):
                chunk = stacked[i:i+CHUNK].to(device)
                out = model(chunk)
                all_probs.append(out["y_prob"].cpu().numpy())
                del chunk
        return np.concatenate(all_probs, axis=0)

    rng_bg = np.random.RandomState(42)
    background = rng_bg.randint(0, 2, size=(10, K)).astype(float)
    x_instance = np.ones((1, K))
    explainer = shap.KernelExplainer(predict_masked, background)
    shap_vals = explainer.shap_values(x_instance, nsamples=shap_nsamples)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.array(shap_vals).reshape(-1)

    N_SHOW = min(10, K)

    # Token → signed IG lookup
    tok_ig_signed = {it["token_str"]: it["ig_signed"] for it in items}

    # ── Patient visit blocks ─────────────────────────────────────────────
    pid = pick["patient_id"]
    hadm_id = pick["index_hadm_id"]
    patient_row = df_seq_test[
        (df_seq_test["patient_id"].astype(str) == str(pid)) &
        (df_seq_test["index_hadm_id"].astype(str) == str(hadm_id))
    ]
    salient_visit_blocks = []
    if len(patient_row) > 0:
        salient_visit_blocks = patient_row.iloc[0].get("salient_visit_blocks", [])

    # ── Match ALL patterns ───────────────────────────────────────────────
    all_matched = _match_all_patterns(
        df_patterns_train, salient_visit_blocks, is_episode)

    # Compute signed sum for each matched pattern (not mass_pct)
    matched_scored = []
    for pat_str, orr, sup, pat_tokens in all_matched:
        ig_sum = sum(tok_ig_signed.get(t, 0.0) for t in pat_tokens)
        matched_scored.append({
            "pattern": pat_str, "odds_ratio": orr,
            "support": int(sup), "tokens": pat_tokens,
            "ig_signed_sum": ig_sum,
        })

    # ── SHAP panel data ──────────────────────────────────────────────────
    shap_order = np.argsort(-np.abs(shap_vals))[:N_SHOW]
    shap_tokens = [_fmt_tok(items_sel[i]["token_str"], icd_titles) for i in shap_order][::-1]
    shap_plot_vals = shap_vals[shap_order][::-1]
    shap_colors = ["#c44e52" if v > 0 else "#4c72b0" for v in shap_plot_vals]

    # ── IG panel data (signed) ───────────────────────────────────────────
    ig_top = sorted(items[:N_SHOW], key=lambda d: d["ig_signed"])  # bottom-to-top by signed
    ig_tokens = [_fmt_tok(it["token_str"], icd_titles) for it in ig_top]
    ig_vals = [it["ig_signed"] for it in ig_top]
    ig_colors = ["#c44e52" if v > 0 else "#4c72b0" for v in ig_vals]

    # ── Build figure ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"Representative {split} patient, {pick.get('n_visits', '?')} visits, "
        f"readmitted, {len(all_matched)} matched patterns",
        fontsize=11, fontweight="bold", y=0.98)

    # ── Panel A: SHAP (signed) ───────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.barh(range(len(shap_tokens)), shap_plot_vals, color=shap_colors)
    ax1.set_yticks(range(len(shap_tokens)))
    ax1.set_yticklabels(shap_tokens, fontsize=8)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_title("A: SHAP (masking-based, top tokens by |SHAP|)", fontsize=10)
    ax1.set_xlabel("SHAP value\n← protective    risk →")

    # ── Panel B: IG (signed) ─────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.barh(range(len(ig_tokens)), ig_vals, color=ig_colors)
    ax2.set_yticks(range(len(ig_tokens)))
    ax2.set_yticklabels(ig_tokens, fontsize=8)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title("B: Integrated Gradients (top tokens by |IG|, signed)", fontsize=10)
    ax2.set_xlabel("Signed IG\n← protective    risk →")

    # ── Panel C: AToP visit-ordered salient tokens (signed) ──────────────
    ax3 = fig.add_subplot(2, 2, 3)
    if salient_visit_blocks:
        visit_entries = []
        for vi, block in enumerate(salient_visit_blocks):
            for tok in sorted(block):
                ig_s = tok_ig_signed.get(tok, 0.0)
                visit_entries.append({
                    "visit": vi, "token": tok,
                    "ig": ig_s,
                    "label": f"V{vi+1}: {_fmt_tok(tok, icd_titles)}",
                    "color": "#c44e52" if ig_s > 0 else "#4c72b0",
                })

        n_show_c = min(25, len(visit_entries))
        if n_show_c > 0:
            display = visit_entries[:n_show_c]
            y_pos = list(range(len(display)))
            ax3.barh(y_pos[::-1],
                     [e["ig"] for e in display],
                     color=[e["color"] for e in display],
                     edgecolor="white", linewidth=0.5)
            ax3.set_yticks(y_pos[::-1])
            ax3.set_yticklabels([e["label"] for e in display], fontsize=8.5)
            ax3.axvline(0, color="black", linewidth=0.8)
            ax3.set_xlabel("Signed IG\n← protective    risk →")

            # Annotate visit boundaries
            prev_v = -1
            for i, e in enumerate(display):
                if e["visit"] != prev_v and prev_v >= 0:
                    ax3.axhline(y_pos[::-1][i] + 0.5, color="#cccccc",
                                linestyle="--", linewidth=0.5)
                prev_v = e["visit"]
    else:
        ax3.text(0.5, 0.5, "No salient tokens found", ha="center", va="center", fontsize=10)

    ax3.set_title(
        f"C: AToP salient tokens, chronological visit order\n"
        f"({len(salient_visit_blocks)} visits, "
        f"{sum(len(b) for b in salient_visit_blocks)} tokens; "
        f"visit structure preserved for temporal context)",
        fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c44e52", label="Risk (IG > 0)"),
        Patch(facecolor="#4c72b0", label="Protective (IG < 0)"),
    ]
    ax3.legend(handles=legend_elements, loc="lower right", fontsize=8.5, framealpha=0.9)

    # ── Panel D: Matched patterns ranked by signed IG sum ────────────────
    ax4 = fig.add_subplot(2, 2, 4)

    n_matched = len(all_matched)
    N_SHOW_PAT = 10

    if n_matched > 0:
        # Rank by |signed IG sum|, take top N
        sorted_matched = sorted(matched_scored, key=lambda m: abs(m["ig_signed_sum"]), reverse=True)
        top_matched = sorted_matched[:N_SHOW_PAT]
        top_matched = sorted(top_matched, key=lambda m: abs(m["ig_signed_sum"]))

        labels_d = []
        colors_d = []
        vals_d = []
        for m in top_matched:
            pat_label = _fmt_pat(m["pattern"], icd_titles, max_tok_len=45)
            direction = "▲" if m["ig_signed_sum"] > 0 else "▼"
            labels_d.append(f"{direction} {pat_label}\n    OR={m['odds_ratio']:.2f} n={m['support']}")
            colors_d.append("#c44e52" if m["ig_signed_sum"] > 0 else "#4c72b0")
            vals_d.append(m["ig_signed_sum"])

        n_d = len(labels_d)
        ax4.barh(range(n_d), vals_d, color=colors_d, edgecolor="white", linewidth=0.5)
        ax4.set_yticks(range(n_d))
        ax4.set_yticklabels(labels_d, fontsize=8)
        ax4.axvline(0, color="black", linewidth=0.8)
        ax4.set_xlabel("Signed IG(P) = Σ IG_i\n← protective    risk →", fontsize=9)
        ax4.set_title(f"D: Matched temporal patterns ({method_label})\n"
                       f"{n_matched} matched of {len(df_patterns_train)} total",
                       fontsize=10)
    else:
        ax4.text(0.5, 0.5,
                 f"No patterns matched this patient's\n"
                 f"salient visit blocks ({len(salient_visit_blocks)} blocks,\n"
                 f"{sum(len(b) for b in salient_visit_blocks)} salient tokens).\n"
                 f"Searched all {len(df_patterns_train)} patterns.",
                 ha="center", va="center", fontsize=10)
        ax4.set_title(f"D: Matched temporal patterns ({method_label})", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, f"fig2_patient_explanation_{split}{filename_suffix}.png"), dpi=300)
    plt.close(fig)

    # Save backing CSVs for fig4
    fig4_rows = []
    for it in items_sel:
        fig4_rows.append({
            "token_str": it["token_str"],
            "token_readable": _fmt_tok(it["token_str"], icd_titles),
            "ig_abs": it["ig_abs"],
            "ig_signed": it["ig_signed"],
        })
    pd.DataFrame(fig4_rows).to_csv(
        os.path.join(csv_dir, f"fig2_patient_ig_tokens_{split}{filename_suffix}.csv"), index=False)

    if matched_scored:
        pd.DataFrame(matched_scored).to_csv(
            os.path.join(csv_dir, f"fig2_matched_patterns_{split}.csv"), index=False)



def _render_fig6(out_dir, icd_titles, ig_global, shap_global, df_pathways,
                 n_show, n_shap_patients, split, panel_c_label=""):
    """Render the 3-panel fig6 figure. Returns fig object (caller saves)."""
    N = n_show
    fig, axes = plt.subplots(1, 3, figsize=(32, max(10, N * 0.6)))

    suffix = f", {panel_c_label}" if panel_c_label else ""
    fig.suptitle(
        f"Global importance comparison: IG vs GradientSHAP vs AToP ({split.upper()}){suffix}\n"
        "All panels computed on the same test cohort, "
        "single-token (A, B) vs occurrence-aggregated pattern attribution (C)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # --- Panel A: Global IG ---
    ax = axes[0]
    if not ig_global.empty:
        ig_top = ig_global.reindex(ig_global.abs().sort_values(ascending=False).head(N).index)
        ig_top = ig_top.reindex(ig_top.abs().sort_values(ascending=True).index)
        labels_a = [_fmt_tok(t, icd_titles) for t in ig_top.index]
        vals_a = ig_top.values.tolist()
        colors_a = ["#c44e52" if v > 0 else "#4c72b0" for v in vals_a]
        ax.barh(range(len(labels_a)), vals_a, color=colors_a, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(labels_a)))
        ax.set_yticklabels(labels_a, fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean IG\n\u2190 protective    risk \u2192", fontsize=9)
    ax.set_title(f"A: Integrated Gradients\n(single-token, all {split} patients)", fontsize=10)

    # --- Panel B: Global SHAP ---
    ax = axes[1]
    if not shap_global.empty:
        shap_top = shap_global.reindex(shap_global.abs().sort_values(ascending=False).head(N).index)
        shap_top = shap_top.reindex(shap_top.abs().sort_values(ascending=True).index)
        labels_b = [_fmt_tok(t, icd_titles) for t in shap_top.index]
        vals_b = shap_top.values.tolist()
        colors_b = ["#c44e52" if v > 0 else "#4c72b0" for v in vals_b]
        ax.barh(range(len(labels_b)), vals_b, color=colors_b, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(labels_b)))
        ax.set_yticklabels(labels_b, fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Mean GradientSHAP ({n_shap_patients} patients)\n\u2190 protective    risk \u2192", fontsize=9)
    ax.set_title(f"B: GradientSHAP\n(single-token, all {split} patients)", fontsize=10)

    # --- Panel C: AToP pathway importance (occurrence-aggregated) ---
    ax = axes[2]
    if not df_pathways.empty:
        # Select top N by |mean signed IG|, breaking ties in favor of
        # longer patterns (more visit blocks = richer clinical narrative).
        df_pw = df_pathways.copy()
        df_pw["_abs_ig"] = df_pw["ig_signed_mean"].abs()
        df_pw["_abs_ig_bin"] = df_pw["_abs_ig"].round(2)
        df_pw["_n_blocks"] = df_pw["pattern"].apply(_count_pattern_blocks)
        top_paths = (df_pw
                     .sort_values(["_abs_ig_bin", "_n_blocks", "_abs_ig"],
                                  ascending=[False, False, False])
                     .head(N)
                     .sort_values("ig_signed_mean", key=abs, ascending=True))
        labels_c = []
        colors_c = []
        for _, r in top_paths.iterrows():
            pat_label = _fmt_pat(r["pattern"], icd_titles, max_tok_len=55)
            direction = "\u25B2" if r["ig_signed_mean"] > 0 else "\u25BC"
            labels_c.append(f"{direction} {pat_label}  (n={r['n_present']})")
            colors_c.append("#c44e52" if r["ig_signed_mean"] > 0 else "#4c72b0")
        ax.barh(range(len(labels_c)), top_paths["ig_signed_mean"].values,
                color=colors_c, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(labels_c)))
        ax.set_yticklabels(labels_c, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No patterns with presence in this split",
                ha="center", va="center")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean \u03A3IG over pattern carriers (n shown per bar)\n\u2190 protective    risk \u2192", fontsize=9)
    c_title = "C: AToP temporal patterns"
    if panel_c_label:
        c_title += f"\n({panel_c_label})"
    else:
        c_title += f"\n(occurrence-aggregated, all {split} patients)"
    ax.set_title(c_title, fontsize=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c44e52", label="\u25B2 Risk (positive IG)"),
        Patch(facecolor="#4c72b0", label="\u25BC Protective (negative IG)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


# ── Removed (Apr 2026 cleanup) ──────────────────────────────────────────────
# fig_supp_conditional_impact (~125 LOC) and fig_supp_jaccard_clusters
# (~200 LOC) used to live here. Both had zero live callers in atop_package
# and produced supplementary figures explicitly marked "skip" in
# TASKS/_shared_context.md (overlapped Fig S5 / methodology detail).
# fig_supp_jaccard_clusters in particular hosted a slow O(N\u00B2) Jaccard sweep
# that was the 3-hour bottleneck on 2026-04-23. Inline jaccard clustering
# for the fig3 \u2192 fig4 whitelist now lives in atop/explainer.py.


def _compute_panel_c(df_patterns, df_seq, df_ig, mining_method, label=""):
    """Score every pattern in ``df_patterns`` over the split with the single
    per-patient rule in :mod:`atop.explain.scoring`.

    Rule (shared with ``cooccurrence.score_all_raw_patterns``): each patient's
    IG rows are indexed as ``token -> [(visit_idx, ig_signed), ...]``; a
    pattern's visit steps are matched greedily left to right, each step to the
    earliest visit strictly later than the previous step's visit in which all
    of the step's tokens occur; the patient's score is the sum of the signed
    IG of each step token at its matched visit.  ``ig_signed_mean`` is the
    mean over carriers and ``n_present`` the number of carriers.  Presence is
    taken from ``df_ig`` (every token position the model saw), so carriers
    are full-presence carriers; ``df_seq`` is accepted for API compatibility
    only.

    Vectorized: a sparse ``patients x tokens`` presence matrix is built once
    and each pattern is reduced to its candidate carriers before the (still
    Python) temporal match.
    """
    is_episode = (mining_method == "episode")
    pattern_results = []
    if df_patterns.empty or df_ig.empty:
        return pd.DataFrame(pattern_results)

    if "visit_idx" not in df_ig.columns:
        print(f"  [{label}] WARNING: df_ig missing visit_idx, "
              "cannot verify temporal order, falling back to token co-occurrence only")

    # ── Per-patient token -> [(visit_idx, ig_value)] index (rule step 1) ──
    print(f"  [{label}] Building per-patient token occurrence index...")
    patient_tokens = build_patient_token_occurrences(df_ig)

    n_patients_total = len(patient_tokens)
    prefix = f"    [{label}] " if label else "    "
    print(f"{prefix}{n_patients_total} test patients indexed", flush=True)

    # ── Sparse patient x token presence matrix for the candidate pre-filter ─
    print(f"  [{label}] Building sparse presence matrix for vectorized carrier lookup...", flush=True)
    index = build_presence_index(patient_tokens)
    presence = index.presence
    print(f"{prefix}presence matrix: {presence.shape[0]} patients × "
          f"{presence.shape[1]} tokens, {presence.nnz:,} non-zeros "
          f"({100 * presence.nnz / (presence.shape[0] * presence.shape[1]):.2f}% density)",
          flush=True)

    # ── Score each pattern ───────────────────────────────────────────────
    n_no_match = 0
    n_total_patterns = len(df_patterns)
    progress_step = max(1000, n_total_patterns // 20)  # ~20 progress lines
    for pat_idx, (_, pat_row) in enumerate(df_patterns.iterrows()):
        pat_str = pat_row["pattern"]
        orr = pat_row["odds_ratio"]
        sup = pat_row["n_admissions_present"]

        if is_episode:
            # Episode: one itemset per visit step.
            pat_itemsets = episode_to_itemsets(_parse_episode_pattern(pat_str))
        else:
            # Flat pattern: each token is its own singleton step.
            flat = [p.strip() for p in pat_str.split(" -> ")]
            pat_itemsets = [[t] for t in flat]

        all_tokens_needed = set()
        for itemset in pat_itemsets:
            all_tokens_needed.update(itemset)

        # Vectorized carrier filter: patients with ALL required tokens.
        # If any required token isn't in the cohort vocab, no patient can carry → skip.
        candidate_rows_ = candidate_rows(index, all_tokens_needed)
        if candidate_rows_ is None:
            n_no_match += 1
            if (pat_idx + 1) % progress_step == 0:
                print(f"{prefix}{pat_idx + 1}/{n_total_patterns} patterns scored "
                      f"({len(pattern_results)} with carriers, {n_no_match} no match)",
                      flush=True)
            continue

        # Periodic progress print (every ~5%)
        if (pat_idx + 1) % progress_step == 0:
            print(f"{prefix}{pat_idx + 1}/{n_total_patterns} patterns scored "
                  f"({len(pattern_results)} with carriers, {n_no_match} no match)",
                  flush=True)

        patient_scores = []
        n_concordant = 0
        matched_patient_keys = set()
        for r in candidate_rows_:
            key = index.patient_keys[r]
            # Rule steps 2-3: greedy earliest-visit match + signed-IG sum.
            is_carrier, patient_ig_sum = match_and_score_episode(pat_itemsets, patient_tokens[key])
            if not is_carrier:
                continue
            patient_scores.append(patient_ig_sum)
            matched_patient_keys.add(key)

            # Track concordance for conditional score
            if (orr > 1 and patient_ig_sum > 0) or (orr <= 1 and patient_ig_sum < 0):
                n_concordant += 1

        if not patient_scores:
            n_no_match += 1
            continue

        # Conditional impact: E[IG | present, concordant with OR direction]
        if orr > 1:
            concordant_vals = [s for s in patient_scores if s > 0]
        else:
            concordant_vals = [s for s in patient_scores if s < 0]
        cond_impact = float(np.mean(concordant_vals)) if concordant_vals else 0.0

        pattern_results.append({
            "pattern": pat_str, "odds_ratio": orr, "support": int(sup),
            "n_present": len(patient_scores),
            "ig_signed_mean": float(np.mean(patient_scores)),
            "ig_signed_median": float(np.median(patient_scores)),
            "ig_cond_impact": cond_impact,
            "n_concordant": n_concordant,
            "pat_tokens": all_tokens_needed,
            "_carrier_keys": matched_patient_keys,
        })

    print(f"{prefix}{len(pattern_results)} patterns with occurrences, "
          f"{n_no_match} with 0 matches across {n_patients_total} patients")
    return pd.DataFrame(pattern_results)


def fig6_global_importance_comparison(
    out_dir: str, model, device: torch.device, icd_titles: Dict,
    vocab_inv: Dict[int, str], ig: "IntegratedGradientsCustom",
    df_ig: pd.DataFrame, df_patterns: pd.DataFrame, df_seq: pd.DataFrame,
    tensors_by_key: Dict, mining_method: str,
    shap_nsamples: int = 200, shap_n_patients: int = 50,
    n_show: int = 15, split: str = "test",
    supp_dir: str = None, csv_dir: str = None,
):
    """Figure 3: IG vs GradientSHAP vs AToP.
    
    Generates two PNGs:
      - Multi-token (≥2) in out_dir (main paper figure)
      - All patterns in supp_dir (supplementary, for completeness)
    
    Panel A: Integrated Gradients (all patients, gradient-based)
    Panel B: GradientSHAP (all patients, gradient-based SHAP)
    Panel C: AToP temporal patterns (all matched patients, signed sum of IG)
    """
    if supp_dir is None:
        supp_dir = out_dir
    if csv_dir is None:
        csv_dir = out_dir
    import shap

    print(f"[mine_global] Building global 3-panel importance comparison ({split})...")
    if df_ig.empty:
        print(f"  [mine_global] Skipped: no IG data available.")
        return

    ig_col = "ig_signed" if "ig_signed" in df_ig.columns else "ig_abs"

    # ── Panel A: IG on all test patients ─────────────────────────────────
    # Consistent with Panel C: for each token, sum IG per patient (across
    # all salient positions of that token), then average across patients.
    # This makes Panel A and Panel C directly comparable, both report
    # mean patient-level IG contribution.
    print(f"  [mine_global] Panel A: aggregating IG (signed, all {split} patients)...")
    ig_filtered = df_ig[~df_ig["token_str"].isin(SPECIAL_TOKENS)].copy()
    tok_patient_counts = ig_filtered.groupby("token_str")["patient_id"].nunique()
    common_tokens = set(tok_patient_counts[tok_patient_counts >= 10].index)
    ig_common = ig_filtered[ig_filtered["token_str"].isin(common_tokens)]
    # Step 1: sum IG per (patient, token), a patient with SI at 3 positions gets one sum
    ig_per_patient = ig_common.groupby(["token_str", "patient_id"])[ig_col].sum().reset_index()
    # Step 2: mean across patients, each patient contributes equally
    ig_global = ig_per_patient.groupby("token_str")[ig_col].mean()

    # ── Panel B: GradientSHAP ─────────────────────────────────────────────────
    shap_global = pd.Series(dtype=float)
    n_shap_patients = 0
    test_keys = list(tensors_by_key.keys()) if tensors_by_key else []

    if not test_keys:
        print("  [mine_global] Panel B: skipping GradientSHAP (no tensors available)")
    else:
        print(f"  [mine_global] Panel B: computing GradientSHAP (all {len(test_keys)} patients)...")
        rng = np.random.RandomState(42)

        # Background: small random subset for reference distribution
        n_bg = min(20, len(test_keys))
        bg_keys = [test_keys[i] for i in rng.choice(len(test_keys), n_bg, replace=False)]

    n_shap_patients = len(test_keys)

    _skip_shap = (not test_keys)
    if _skip_shap:
        pass  # shap_global already set to empty Series above
    else:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Wrapper model: embeddings (B, L, D) → probability (B,)
    # GradientSHAP needs a module that takes continuous input
    class EmbeddingToProb(torch.nn.Module):
        """Wrapper that takes pre-computed embeddings and runs transformer + classifier.
        
        Note: We do NOT use pad masking here because GradientExplainer passes
        interpolated single samples with unpredictable batch sizes. PAD token
        embeddings are near-zero so their contribution is negligible for a
        global importance comparison. The slight approximation is acceptable
        for Panel B (which shows relative token rankings, not exact values).
        """
        def __init__(self, transformer_model):
            super().__init__()
            self.transformer = transformer_model.transformer
            self.classifier = transformer_model.classifier

        def forward(self, x):
            # x: (B, L, D) embeddings
            out = self.transformer(x)
            cls_out = out[:, 0, :]
            logits = self.classifier(cls_out)
            # Return (B, 1), GradientExplainer indexes output with 2D indexing
            probs = torch.sigmoid(logits.view(-1, 1))
            return probs

    if not _skip_shap:
        model.eval()
        shap_accum: Counter = Counter()
        shap_count: Counter = Counter()

        # Check for cached SHAP results
        shap_cache_path = os.path.join(csv_dir, f"shap_global_{split}.csv") if csv_dir else None
        if shap_cache_path and os.path.exists(shap_cache_path):
            print(f"  [mine_global] Panel B: loading cached GradientSHAP from {shap_cache_path}")
            _shap_df = pd.read_csv(shap_cache_path)
            shap_global = pd.Series(_shap_df["shap_mean"].values, index=_shap_df["token"].values)
            _skip_shap = True  # skip computation, use cached
        elif shap_cache_path and os.path.exists(shap_cache_path + ".partial"):
            print(f"  [mine_global] Panel B: found partial SHAP cache, loading as fallback")
            _shap_df = pd.read_csv(shap_cache_path + ".partial")
            shap_global = pd.Series(_shap_df["shap_mean"].values, index=_shap_df["token"].values)
            # Save as full cache, partial is good enough for global ranking
            _shap_df.to_csv(shap_cache_path, index=False)
            print(f"  [mine_global] Panel B: promoted partial ({len(_shap_df)} tokens) to full cache")
            _skip_shap = True

        if not _skip_shap:
            # Create wrapper and background embeddings ONCE
            SHAP_BATCH = 50  # larger batch on GPU
            n_processed = 0

            # Compute background embeddings once
            bg_ids_all = torch.cat([tensors_by_key[k] for k in bg_keys], dim=0).to(device)
            with torch.no_grad():
                B_bg, L_bg = bg_ids_all.shape
                pos_bg = torch.arange(L_bg, device=device).unsqueeze(0)
                bg_emb = model.emb_norm(model.emb_dropout(
                    model.token_emb(bg_ids_all) + model.pos_emb(pos_bg.expand(B_bg, L_bg))))

            wrapper = EmbeddingToProb(model).to(device)
            wrapper.eval()
            explainer_grad = shap.GradientExplainer(wrapper, bg_emb)

            for batch_start in range(0, len(test_keys), SHAP_BATCH):
                batch_keys = test_keys[batch_start:batch_start + SHAP_BATCH]

                # Get input_ids for this batch
                batch_ids = torch.cat([tensors_by_key[k] for k in batch_keys], dim=0).to(device)

                # Compute embeddings
                with torch.no_grad():
                    B_batch, L = batch_ids.shape
                    pos = torch.arange(L, device=device).unsqueeze(0)
                    batch_emb = model.emb_norm(model.emb_dropout(
                        model.token_emb(batch_ids) + model.pos_emb(pos.expand(B_batch, L))))

                try:
                    sv = explainer_grad.shap_values(batch_emb, nsamples=shap_nsamples)

                    if isinstance(sv, list):
                        sv = sv[0]
                    sv_np = np.array(sv)

                    n_batch = len(batch_keys)
                    if sv_np.ndim == 4:
                        sv_np = sv_np.squeeze(-1)
                        sv_per_token = sv_np.sum(axis=-1)
                    elif sv_np.ndim == 3:
                        sv_per_token = sv_np.sum(axis=-1)
                    elif sv_np.ndim == 2:
                        if sv_np.shape[0] == n_batch:
                            sv_per_token = sv_np
                        else:
                            sv_per_token = sv_np.sum(axis=-1).reshape(1, -1)
                    elif sv_np.ndim == 1:
                        sv_per_token = sv_np.reshape(1, -1)
                    else:
                        raise ValueError(f"Unexpected SHAP output ndim={sv_np.ndim}, shape={sv_np.shape}")

                    if sv_per_token.ndim == 1:
                        sv_per_token = sv_per_token.reshape(1, -1)

                    for bi, key in enumerate(batch_keys):
                        if bi >= sv_per_token.shape[0]:
                            break
                        ids_np = tensors_by_key[key][0].numpy()
                        for pos in range(min(len(ids_np), sv_per_token.shape[1])):
                            tid = ids_np[pos]
                            if tid == PAD_IDX:
                                continue
                            tok_str = vocab_inv.get(tid, f"UNK_{tid}")
                            if tok_str in SPECIAL_TOKENS:
                                continue
                            shap_accum[tok_str] += float(sv_per_token[bi, pos])
                            shap_count[tok_str] += 1

                except Exception as e:
                    if batch_start == 0:
                        import traceback; traceback.print_exc()
                    if batch_start % 500 == 0:
                        print(f"\n    [GradientSHAP] Batch {batch_start} failed: {e}")
                    continue
                finally:
                    del batch_emb, batch_ids
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                n_processed += len(batch_keys)
                print(f"\r    GradientSHAP progress: {n_processed}/{len(test_keys)}", end="", flush=True)

                # Intermediate save every 5000 patients to survive disconnects
                if shap_cache_path and n_processed % 5000 < SHAP_BATCH:
                    _shap_partial = {tok: shap_accum[tok] / shap_count[tok] for tok in shap_accum}
                    _shap_partial_s = pd.Series(_shap_partial)
                    if not _shap_partial_s.empty:
                        _tmp = pd.DataFrame({"token": _shap_partial_s.index, "shap_mean": _shap_partial_s.values})
                        _tmp.to_csv(shap_cache_path + ".partial", index=False)

            print()

            # Clean up explainer
            del wrapper, explainer_grad, bg_emb, bg_ids_all
            if device.type == "cuda":
                torch.cuda.empty_cache()

            shap_mean = {tok: shap_accum[tok] / shap_count[tok] for tok in shap_accum}
            shap_global = pd.Series(shap_mean)

            # Cache SHAP results
            if shap_cache_path and not shap_global.empty:
                _shap_out = pd.DataFrame({"token": shap_global.index, "shap_mean": shap_global.values})
                _shap_out.to_csv(shap_cache_path, index=False)
                print(f"  [mine_global] Panel B: cached GradientSHAP → {shap_cache_path}")

    # ── Panel C: occurrence-aggregated pattern attribution ──────────────
    # Check for cached carrier sets, skip expensive recomputation if available
    print("  [mine_global] Panel C: computing occurrence-aggregated pattern IG...")
    
    _panel_c_cached = False
    if csv_dir:
        import pickle as _pkl
        _all_pkl = os.path.join(csv_dir, f"carrier_sets_{split}_all.pkl")
        # Cache file: prefer migrated/canonical name; fall back to legacy fig3_ prefix.
        _all_csv_v2 = os.path.join(csv_dir, f"pathway_importance_{split}_all.csv")
        _all_csv_legacy = os.path.join(csv_dir, f"fig3_pathway_importance_{split}_all.csv")
        _all_csv = _all_csv_v2 if os.path.exists(_all_csv_v2) else _all_csv_legacy
        if os.path.exists(_all_pkl) and os.path.exists(_all_csv):
            try:
                with open(_all_pkl, "rb") as _f:
                    _carriers_all = _pkl.load(_f)
                _df_cached = pd.read_csv(_all_csv)
                # Verify: same pattern set?
                cached_pats = set(_df_cached["pattern"].tolist())
                current_pats = set(df_patterns["pattern"].tolist())
                if cached_pats == current_pats:
                    # Restore carrier keys
                    _df_cached["_carrier_keys"] = _df_cached["pattern"].map(_carriers_all)
                    df_pathways_all = _df_cached
                    print(f"  [mine_global] Panel C: loaded {len(df_pathways_all)} cached patterns "
                          f"from {_all_csv}")
                    _panel_c_cached = True
                else:
                    print(f"  [mine_global] Panel C: cache mismatch ({len(cached_pats)} cached vs "
                          f"{len(current_pats)} current), recomputing")
            except Exception as e:
                print(f"  [mine_global] Panel C: cache load failed ({e}), recomputing")
    
    if not _panel_c_cached:
        df_pathways_all = _compute_panel_c(df_patterns, df_seq, df_ig, mining_method, label="all")

    # Main figure: multi-token patterns (≥2 tokens, incl. same-visit co-occurrences)
    df_patterns_multi_set = set(_filter_multi_token(df_patterns, min_tokens=2)["pattern"].tolist())
    n_multi = len(df_patterns_multi_set)
    print(f"  [mine_global] Panel C multi-token: {n_multi}/{len(df_patterns)} patterns with \u22652 tokens")
    df_pathways_multi = df_pathways_all[df_pathways_all["pattern"].isin(df_patterns_multi_set)].copy()

    # Supplement: cross-visit only (≥2 visit blocks, ≥2 tokens)
    df_patterns_xvisit_set = set(_filter_cross_visit(
        _filter_multi_token(df_patterns, min_tokens=2), min_blocks=2)["pattern"].tolist())
    n_xvisit = len(df_patterns_xvisit_set)
    print(f"  [mine_global] Panel C cross-visit: {n_xvisit}/{len(df_patterns)} patterns "
          f"with \u22652 tokens AND \u22652 visit blocks")
    df_pathways_xvisit = df_pathways_all[df_pathways_all["pattern"].isin(df_patterns_xvisit_set)].copy()

    # ── CSVs (written FIRST so an interrupt during render doesn't lose work) ─
    # Test cohort size for prevalence denominator. ``test_keys`` is the set
    # of stored input-id tensors and is empty under the ``mine_global``
    # step (we skip GradientSHAP to save 8h). Fall back to ``len(df_seq)``
    # so prevalence stays a fraction in either path.
    n_test_total = len(test_keys) or len(df_seq)
    for lbl, df_pw, sfx in [("all", df_pathways_all, "_all"),
                             ("multi", df_pathways_multi, ""),
                             ("xvisit", df_pathways_xvisit, "_xvisit")]:
        if not df_pw.empty:
            df_out = df_pw.drop(columns=["pat_tokens", "_carrier_keys"], errors="ignore").copy()
            df_out["prevalence"] = df_out["n_present"] / max(n_test_total, 1)
            df_out["pattern_readable"] = df_out["pattern"].apply(
                lambda p: _fmt_pat(p, icd_titles, max_tok_len=80))
            df_out.to_csv(
                os.path.join(csv_dir, f"pathway_importance_{split}{sfx}.csv"), index=False)

            # Save carrier sets for offline Jaccard analysis / replot
            if "_carrier_keys" in df_pw.columns:
                import pickle
                carriers = {row["pattern"]: row["_carrier_keys"]
                            for _, row in df_pw.iterrows()
                            if isinstance(row.get("_carrier_keys"), set)}
                if carriers:
                    pkl_path = os.path.join(csv_dir, f"carrier_sets_{split}{sfx}.pkl")
                    with open(pkl_path, "wb") as f:
                        pickle.dump(carriers, f)
                    print(f"  [carriers] Saved {len(carriers)} carrier sets \u2192 {pkl_path}")

    all_toks = sorted(set(ig_global.index) | set(shap_global.index))
    tok_rows = []
    # Same fallback as above so token_importance.csv prevalence is a fraction
    # under both ``mine_global`` (no tensors) and the SHAP path (tensors).
    n_test_total = len(test_keys) or len(df_seq)
    for tok in all_toks:
        n_pr = int(tok_patient_counts.get(tok, 0))
        tok_rows.append({
            "token": tok, "token_readable": _fmt_tok(tok, icd_titles),
            "ig_mean": float(ig_global.get(tok, np.nan)),
            "shap_mean": float(shap_global.get(tok, np.nan)),
            "n_present": n_pr,
            "prevalence": n_pr / max(n_test_total, 1),
        })
    pd.DataFrame(tok_rows).to_csv(
        os.path.join(csv_dir, f"token_importance_{split}.csv"), index=False)

    # ── Render: multi-token sanity-check PNG (NOT the manuscript figure) ──
    # All CSVs above are already on disk by now \u2014 if this render is interrupted
    # or fails, Panel C work is preserved. The submission's Figure 3 is
    # rendered by paper/jamia2026/figures/make_fig3.py from those CSVs; this PNG exists only
    # as a quick eyeball-check after a fresh mining run.
    #
    # Removed in Apr 2026 cleanup: 3 supplementary PNG variants
    # (_split / _xvisit / _all) and three supplementary functions
    # (supp_conditional_impact, supp_prevalence_vs_impact, supp_jaccard_clusters)
    # \u2014 none shipped, all replaced by make_*.py scripts. supp_jaccard in
    # particular hosted an O(N\u00B2) Jaccard sweep that was the 3-hour bottleneck
    # on 2026-04-23.
    fig = _render_fig6(out_dir, icd_titles, ig_global, shap_global,
                       df_pathways_multi, n_show, len(test_keys), split,
                       panel_c_label="\u22652-token patterns (incl. same-visit)")
    path_main = os.path.join(out_dir, f"fig3_global_comparison_{split}.png")
    fig.savefig(path_main, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [mine_global] Saved sanity-check PNG \u2192 {path_main}")

    return df_pathways_multi


# ============================================================================
# REVERSED-ORDER ANALYSIS FIGURE
# ============================================================================


def fig5_reversed_order(out_dir, df_rev, icd_titles=None, split="test",
                        csv_dir=None):
    """Figure 5: Forward vs reversed pattern order comparison.

    Paired bar chart showing mean Σ IG_i for patients matching the pattern
    in forward order vs reversed order. If the model is sensitive to temporal
    ordering, forward-matched patients should show different attribution.
    """
    if df_rev is None or df_rev.empty:
        print("  [reversed_order] No reversed-order data, skipping")
        return
    if csv_dir is None:
        csv_dir = out_dir

    _titles = icd_titles or {}

    # Filter to patterns with both forward and reversed matches
    df_plot = df_rev[(df_rev["n_forward"] > 0) & (df_rev["n_reversed"] > 0)].copy()
    if df_plot.empty:
        print("  [reversed_order] No patterns with both forward and reversed matches, skipping")
        return

    # Sort by |ig_diff| descending
    df_plot["_abs_diff"] = df_plot["ig_diff"].abs()
    df_plot = df_plot.sort_values("_abs_diff", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(14, max(7, len(df_plot) * 0.65)))

    y = np.arange(len(df_plot))
    bar_h = 0.35

    # Forward bars
    fwd_colors = ["#c44e52" if v > 0 else "#4c72b0" for v in df_plot["ig_mean_forward"]]
    ax.barh(y + bar_h/2, df_plot["ig_mean_forward"], height=bar_h,
            color=fwd_colors, edgecolor="white", alpha=0.9)

    # Reversed bars (lighter)
    rev_colors = ["#e8a0a2" if v > 0 else "#a0b8d0" for v in df_plot["ig_mean_reversed"]]
    ax.barh(y - bar_h/2, df_plot["ig_mean_reversed"], height=bar_h,
            color=rev_colors, edgecolor="white", alpha=0.9)

    # Labels
    labels = []
    for _, r in df_plot.iterrows():
        n_fwd = r["n_forward"]
        n_rev = r["n_reversed"]
        pat_label = r["pattern_readable"] if "pattern_readable" in r.index else r["pattern"]
        if len(pat_label) > 70:
            pat_label = pat_label[:67] + "..."
        labels.append(f"{pat_label}\n(fwd={n_fwd}, rev={n_rev})")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Mean signed \u03A3 IG\u1d62\n\u2190 protective    risk \u2192", fontsize=10)
    ax.set_title(
        f"Temporal order specificity ({split.upper()}): forward vs reversed pattern matching\n"
        "If the model is sensitive to temporal order, forward and reversed bars should differ",
        fontsize=10)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#c44e52", label="Forward order (A \u2192 B)"),
        Patch(facecolor="#e8a0a2", label="Reversed order (B \u2192 A)"),
        Patch(facecolor="#4c72b0", label="Forward order (protective)"),
        Patch(facecolor="#a0b8d0", label="Reversed order (protective)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    path = os.path.join(out_dir, f"fig5_reversed_order_{split}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [reversed_order] Saved \u2192 {path}")

    # CSV
    df_rev.to_csv(os.path.join(csv_dir, f"fig5_reversed_order_{split}.csv"), index=False)


# ============================================================================
# SUPPLEMENTAL FIGURES
# ============================================================================


# ── Removed (Apr 2026 cleanup) ──────────────────────────────────────────────
# fig_supp_ig_heatmap_multi_patient (~50 LOC) and fig_supp_ig_stream_heatmap
# (~30 LOC) used to live here. Both had `def` + an unused import in
# explainer.py and zero call sites. Not part of the JAMIA submission.


# ── Removed (Apr 2026 cleanup) ──────────────────────────────────────────────
# fig_supp_pattern_admission_heatmap (~85 LOC) and fig_supp_pattern_decomposition
# (~125 LOC) used to live here. Both were gated behind the `supp_heatmap` and
# `supp_decomposition` step names \u2014 locked decision: skip ("methodology
# detail / not in submission"). Their step branches in atop/explainer.py
# report() were removed in the same pass.


# ============================================================================
# TABLE 1
# ============================================================================


def save_table1(df_seq_all, summary, out_path):
    tab = [
        ("Unique patients", summary["n_unique_patients"]),
        ("Samples (admissions)", summary["n_samples"]),
        ("Multi-visit samples, n (%)", f'{summary["n_multi_visit"]} ({summary["pct_multi_visit"]:.1f}%)'),
        ("Readmission, n (%)", f'{summary["n_readmit"]} ({summary["readmission_rate"]*100:.1f}%)'),
        ("Sequence length, mean (SD)",
         f'{df_seq_all["len_full"].mean():.1f} ({df_seq_all["len_full"].std(ddof=1):.1f})'),
        ("IG-salient seq length, mean (SD)",
         f'{df_seq_all["len_salient"].mean():.1f} ({df_seq_all["len_salient"].std(ddof=1):.1f})'),
        ("Visits per sample, mean (SD)",
         f'{df_seq_all["n_visits"].mean():.1f} ({df_seq_all["n_visits"].std(ddof=1):.1f})'),
    ]
    pd.DataFrame(tab, columns=["Characteristic", "Value"]).to_csv(out_path, index=False)


# ── Removed (Apr 2026 cleanup) ──────────────────────────────────────────────
# fig_supp_ood_diagnostic (~75 LOC) used to live here. Gated behind the
# `supp_ood` step name \u2014 not in submission. The OOD validity story is now
# narrated in the manuscript text via the validation-distribution stats
# from validation_test.csv (no separate figure shipped). Its step branch in
# atop/explainer.py report() was removed in the same pass.


def fig_train_curves(out_dir: str, train_log: list) -> None:
    """Plot training loss and validation AUROC per epoch."""
    if not train_log:
        return
    import pandas as pd
    df = pd.DataFrame(train_log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(df["epoch"], df["train_loss"], "o-", color="#4C72B0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["epoch"], df["val_auroc"], "o-", color="#DD8452", label="AUROC")
    if "val_pr_auc" in df.columns:
        ax2.plot(df["epoch"], df["val_pr_auc"], "s--", color="#55A868", label="PR-AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Validation Performance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_train_curves.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_train_curves.png")




# ============================================================================
# MAIN
