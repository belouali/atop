"""All figure/table generation for AToP reports."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
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
                                "scripts", "short_names.csv")
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
        print("  [WARN] df_ig missing visit_idx — instance-based scoring unavailable. "
              "Re-run IG computation to enable position-aware scoring.")
    if not has_salient:
        print("  [WARN] df_ig missing is_salient — block cache includes all tokens in salient visits. "
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
            # Skip non-salient tokens — ensures scoring uses the same
            # token universe as mining/matching
            if has_salient and not row["is_salient"]:
                continue
            bi = v2b.get(vidx)
            if bi is None:
                # Token's visit is not a salient block — skip
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
        pattern_fsets: List[frozenset] — the pattern's itemsets
        matched_blocks: List[(pattern_idx, block_idx)] from _match_subpattern
        block_ig_list: List[Dict[str, float]] — per-block IG for this patient

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



def fig2_patient_ig(out_dir, df_ig_long, pick, icd_titles, split="test"):
    pid = pick["patient_id"]
    hadm_id = pick["index_hadm_id"]

    # Coerce types for matching (int vs str mismatch)
    df = df_ig_long.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["index_hadm_id"] = df["index_hadm_id"].astype(str)
    pid_s = str(pid)
    hadm_s = str(hadm_id)

    sub = df[(df.patient_id == pid_s) & (df.index_hadm_id == hadm_s)].copy()
    if sub.empty:
        print(f"  [fig2_{split}] EMPTY — no IG rows for pid={pid_s}, hadm={hadm_s}")
        print(f"    df_ig has {len(df_ig_long)} rows, {df_ig_long['patient_id'].nunique()} patients")
        print(f"    sample pids: {df_ig_long['patient_id'].head(3).tolist()}, types: {df_ig_long['patient_id'].dtype}")
        return False

    ig_col = "ig_signed" if "ig_signed" in sub.columns else "ig_abs"

    # Top tokens by |IG| but display signed values
    sub["_abs"] = sub[ig_col].abs()
    sub2 = sub.sort_values("_abs", ascending=False).head(20)
    sub2 = sub2.sort_values(ig_col, ascending=True)  # bottom-to-top by signed

    labels = [_fmt_tok(t, icd_titles) for t in sub2["token_str"]]
    vals = sub2[ig_col].values
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in vals]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top IG tokens ({split.upper()})\npatient={pid} | index_hadm={hadm_id} | "
                 f"readmission={pick['readmission']} | n_visits={pick.get('n_visits', '?')}")
    ax.set_xlabel("Signed IG\n← protective          risk →")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c44e52", label="Risk (IG > 0)"),
        Patch(facecolor="#4c72b0", label="Protective (IG < 0)"),
    ]
    legend_elements.extend(active_stream_legend(sub2["token_str"].tolist(), "line"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig2_patient_ig_{split}.png"), dpi=300)
    plt.close(fig)
    print(f"  [fig2_{split}] Saved fig2_patient_ig_{split}.png")

    # Save backing CSV
    sub2.to_csv(os.path.join(out_dir, f"fig2_patient_ig_{split}.csv"), index=False)
    return True



def fig3_top_patterns(out_dir, df_patterns, icd_titles, mining_method,
                      df_ig=None, df_seq=None, split="train", n_show=15):
    """Top mined patterns ranked by mean signed IG mass.
    
    If df_ig and df_seq are provided, computes mean IG mass per pattern
    and ranks by that. Otherwise falls back to OR ranking.
    """
    if df_patterns.empty:
        return
    method_labels = {"prefixspan": "PrefixSpan", "ngram": "n-gram", "episode": "Episode"}
    method_label = method_labels.get(mining_method, mining_method)
    is_episode = (mining_method == "episode")

    # Compute mean signed IG per pattern if IG data available
    df_p = df_patterns.copy()
    if df_ig is not None and df_seq is not None and not df_seq.empty:
        ig_cache = _build_ig_cache(df_ig)

        pat_index = PatternIndex.from_df(df_seq)
        seq_rows = df_seq.reset_index(drop=True)

        ig_means = []
        ig_medians = []
        abs_shares = []
        cond_impacts = []
        n_concordants = []
        n_presents = []
        carrier_sets = {}  # pattern_str → set of matched row indices
        for _, pat_row in df_p.iterrows():
            pat_str = pat_row["pattern"]
            if is_episode:
                ep = _parse_episode_pattern(pat_str)
                pat_fsets = [frozenset(s) for s in ep]
                pat_tokens = set()
                for s in ep:
                    pat_tokens.update(s)
            else:
                flat = [p.strip() for p in pat_str.split(" -> ")]
                pat_fsets = [frozenset([t]) for t in flat]
                pat_tokens = set(flat)

            matched = pat_index.patients_matching_pattern(pat_fsets, pat_tokens, is_episode)
            carrier_sets[pat_str] = set(matched)
            or_val = pat_row.get("odds_ratio", None)
            stats = _compute_pattern_ig_stats(ig_cache, pat_tokens, matched, seq_rows,
                                              odds_ratio=or_val)
            if stats:
                ig_means.append(stats["ig_signed_mean"])
                ig_medians.append(stats["ig_signed_median"])
                abs_shares.append(stats["ig_abs_share_mean"])
                cond_impacts.append(stats["ig_cond_impact"])
                n_concordants.append(stats["n_concordant"])
                n_presents.append(stats["n_matched"])
            else:
                ig_means.append(0.0)
                ig_medians.append(0.0)
                abs_shares.append(0.0)
                cond_impacts.append(0.0)
                n_concordants.append(0)
                n_presents.append(0)

        df_p["ig_signed_mean"] = ig_means
        df_p["ig_signed_median"] = ig_medians
        df_p["ig_abs_share_mean"] = abs_shares
        df_p["ig_cond_impact"] = cond_impacts
        df_p["n_concordant"] = n_concordants
        df_p["n_present"] = n_presents
        rank_col = "ig_signed_mean"
    else:
        # Fallback: use OR-derived pseudo-score
        df_p["ig_signed_mean"] = np.where(df_p["odds_ratio"] > 1, df_p["odds_ratio"], -1/df_p["odds_ratio"])
        rank_col = "ig_signed_mean"

    # Split into risk (positive IG mass) and protective (negative)
    half = max(1, n_show // 2)
    df_risk = df_p[df_p[rank_col] > 0].nlargest(half, rank_col).copy()
    df_prot = df_p[df_p[rank_col] < 0].nsmallest(half, rank_col).copy()

    n_risk = len(df_risk)
    n_prot = len(df_prot)
    if n_risk + n_prot == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(24, max(6, max(n_risk, n_prot) * 0.45)),
                              gridspec_kw={"width_ratios": [1, 1]})

    title_note = ("Within-visit items are unordered sets; cross-visit ordering is chronological"
                  if mining_method == "episode" else
                  "Patterns reflect genuine cross-visit temporal ordering")
    fig.suptitle(f"Top mined patterns ({split.upper()}, {method_label})\n"
                 f"Ranked by mean signed IG contribution: IG(P) = Σ IG_i for tokens in pattern\n{title_note}",
                 fontsize=11, fontweight="bold", y=1.02)

    # Left: Risk patterns
    ax_risk = axes[0]
    if n_risk > 0:
        df_risk = df_risk.sort_values(rank_col, ascending=True).reset_index(drop=True)
        labels_risk = []
        for _, r in df_risk.iterrows():
            pat_label = _fmt_pat(r["pattern"], icd_titles, max_tok_len=55)
            n = int(r.get("n_present", r.get("n_admissions_present", 0)))
            labels_risk.append(f"{pat_label}  (n={n})")
        ax_risk.barh(range(n_risk), df_risk[rank_col].values, color="#c44e52")
        ax_risk.set_yticks(range(n_risk))
        ax_risk.set_yticklabels(labels_risk, fontsize=8)
    ax_risk.set_xlabel("Mean signed IG(P) over pattern carriers\n→ pushes logit toward readmission")
    ax_risk.set_title("Risk patterns (IG > 0)", fontsize=10)

    # Right: Protective patterns
    ax_prot = axes[1]
    if n_prot > 0:
        df_prot = df_prot.sort_values(rank_col, ascending=False).reset_index(drop=True)
        labels_prot = []
        for _, r in df_prot.iterrows():
            pat_label = _fmt_pat(r["pattern"], icd_titles, max_tok_len=55)
            n = int(r.get("n_present", r.get("n_admissions_present", 0)))
            labels_prot.append(f"{pat_label}  (n={n})")
        ax_prot.barh(range(n_prot), df_prot[rank_col].abs().values, color="#4c72b0")
        ax_prot.set_yticks(range(n_prot))
        ax_prot.set_yticklabels(labels_prot, fontsize=8)
    else:
        ax_prot.text(0.5, 0.5, "No protective patterns found",
                     ha="center", va="center", fontsize=10)
    ax_prot.set_xlabel("|Mean signed IG(P)| over pattern carriers\n← pushes logit away from readmission")
    ax_prot.set_title("Protective patterns (IG < 0)", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig3_top_patterns_{split}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    df_p.to_csv(os.path.join(out_dir, f"fig3_top_patterns_{split}.csv"), index=False)



def fig5_validation(out_dir, dfv, n_shuffle_draws, icd_titles=None, split="test",
                    n_show=15):
    dfv_clean = dfv.dropna(subset=["mean_delta_mask"]).copy()
    if dfv_clean.empty:
        return

    # Show top patterns — split by IG direction (risk vs protective)
    # If ig_direction available, use it; otherwise fall back to mean_delta_mask
    if "ig_direction" in dfv_clean.columns:
        df_risk = dfv_clean[dfv_clean["ig_direction"] > 0].head(n_show)
        df_prot = dfv_clean[dfv_clean["ig_direction"] <= 0].head(n_show)
    else:
        df_risk = dfv_clean[dfv_clean["mean_delta_mask"] > 0].head(n_show)
        df_prot = dfv_clean[dfv_clean["mean_delta_mask"] <= 0].head(n_show)
    # Take balanced set, up to n_show total
    n_each = n_show // 2
    df_risk = df_risk.head(n_each + n_show % 2)
    df_prot = df_prot.head(n_each)
    top = pd.concat([df_risk, df_prot], ignore_index=True)

    if top.empty:
        return
    top = top.sort_values("mean_delta_mask", ascending=True)

    # Check if we have the visit-shuffle column (v4+)
    has_visit_shuffle = "mean_delta_shuffle_visits" in top.columns
    # Handle old column name for backward compat
    within_col = "mean_delta_shuffle_within" if "mean_delta_shuffle_within" in top.columns else "mean_delta_shuffle"

    y = np.arange(len(top))
    n_bars = 3 if has_visit_shuffle else 2
    bar_h = 0.8 / n_bars

    fig = plt.figure(figsize=(18, max(7, len(top) * 0.75)))
    ax = fig.add_subplot(1, 1, 1)

    # Determine direction: prefer IG, fallback to OR
    def _is_risk(row):
        if "ig_direction" in row.index and not pd.isna(row.get("ig_direction")):
            return row["ig_direction"] > 0
        return row["odds_ratio"] > 1

    # Bar 1: Mask pattern tokens
    mask_colors = ["#c44e52" if _is_risk(r) else "#4c72b0" for _, r in top.iterrows()]
    offset = -(n_bars - 1) * bar_h / 2
    ax.barh(y + offset, top["mean_delta_mask"], height=bar_h,
            color=mask_colors, edgecolor="white")

    # Bar 2: Shuffle visit order (the temporal test)
    if has_visit_shuffle:
        visit_colors = ["#e8a0a2" if _is_risk(r) else "#a0b8d0" for _, r in top.iterrows()]
        ax.barh(y + offset + bar_h, top["mean_delta_shuffle_visits"], height=bar_h,
                color=visit_colors, edgecolor="white")

    # Bar 3: Shuffle within visits (control — should be ~0)
    within_colors = ["#f0c8c8" if _is_risk(r) else "#c8d8e8" for _, r in top.iterrows()]
    last_offset = offset + bar_h * (n_bars - 1)
    ax.barh(y + last_offset, top[within_col], height=bar_h,
            color=within_colors, edgecolor="white")

    labels = []
    _titles = icd_titles or {}
    for _, r in top.iterrows():
        direction = "\u25B2" if _is_risk(r) else "\u25BC"
        pat_label = _fmt_pat(r["pattern"], _titles, max_tok_len=55)
        or_str = f"OR={r['odds_ratio']:.2f}" if not pd.isna(r.get("odds_ratio")) else ""
        labels.append(f"{direction} {pat_label} ({or_str})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean \u0394 probability (baseline \u2212 perturbed)\n"
                  "Positive = pattern supports prediction; Negative = pattern opposes prediction")

    if has_visit_shuffle:
        ax.set_title(
            f"Pattern validation ({split.upper()}): three perturbation tests\n"
            "Mask tokens: does model rely on these codes?\n"
            "Shuffle visits: is model sensitive to cross-visit order?\n"
            "Shuffle within: control (within-visit order is arbitrary)",
            fontsize=9)
    else:
        ax.set_title(
            f"Pattern validation ({split.upper()}): masking vs shuffling\n"
            "\u25B2 = risk patterns, \u25BC = protective patterns",
            fontsize=9)

    # Custom legend showing both perturbation type and risk/protective color
    from matplotlib.patches import Patch
    if has_visit_shuffle:
        legend_handles = [
            Patch(facecolor="#c44e52", label="Mask pattern tokens (\u25B2 risk)"),
            Patch(facecolor="#4c72b0", label="Mask pattern tokens (\u25BC protective)"),
            Patch(facecolor="#e8a0a2", label=f"Shuffle visit order ({n_shuffle_draws} draws, \u25B2 risk)"),
            Patch(facecolor="#a0b8d0", label=f"Shuffle visit order ({n_shuffle_draws} draws, \u25BC protective)"),
            Patch(facecolor="#f0c8c8", label=f"Shuffle within visits ({n_shuffle_draws} draws, \u25B2 risk)"),
            Patch(facecolor="#c8d8e8", label=f"Shuffle within visits ({n_shuffle_draws} draws, \u25BC protective)"),
        ]
    else:
        legend_handles = [
            Patch(facecolor="#c44e52", label="Mask pattern tokens (\u25B2 risk)"),
            Patch(facecolor="#4c72b0", label="Mask pattern tokens (\u25BC protective)"),
            Patch(facecolor="#f0c8c8", label=f"Shuffle within ({n_shuffle_draws} draws, \u25B2 risk)"),
            Patch(facecolor="#c8d8e8", label=f"Shuffle within ({n_shuffle_draws} draws, \u25BC protective)"),
        ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig4_temporal_validation_{split}.png"), dpi=300)
    plt.close(fig)


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
        f"Representative {split} patient — {pick.get('n_visits', '?')} visits, "
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
        f"C: AToP salient tokens — chronological visit order\n"
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

    suffix = f" — {panel_c_label}" if panel_c_label else ""
    fig.suptitle(
        f"Global importance comparison: IG vs GradientSHAP vs AToP ({split.upper()}){suffix}\n"
        "All panels computed on the same test cohort — "
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


def _render_fig6_split(out_dir, icd_titles, ig_global, shap_global, df_pathways,
                       n_show, n_shap_patients, split, panel_c_label=""):
    """Render 3-panel fig6 with Panel C split into risk (top) and protective (bottom).
    
    Panels A and B are identical to _render_fig6. Panel C shows N/2 risk + N/2 protective
    side by side, ensuring both directions are visible even when one dominates by |IG|.
    """
    N = n_show
    half = max(1, N // 2)
    fig, axes = plt.subplots(1, 3, figsize=(32, max(10, N * 0.6)))

    suffix = f" — {panel_c_label}" if panel_c_label else ""
    fig.suptitle(
        f"Global importance comparison: IG vs GradientSHAP vs AToP ({split.upper()}){suffix}\n"
        "Panel C: top risk + top protective patterns (balanced view)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # --- Panel A: Global IG (same as _render_fig6) ---
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

    # --- Panel B: Global SHAP (same as _render_fig6) ---
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

    # --- Panel C: Split risk/protective ---
    ax = axes[2]
    if not df_pathways.empty:
        df_pw = df_pathways.copy()
        df_pw["_abs_ig"] = df_pw["ig_signed_mean"].abs()
        df_pw["_n_blocks"] = df_pw["pattern"].apply(_count_pattern_blocks)

        # Top risk
        df_risk = (df_pw[df_pw["ig_signed_mean"] > 0]
                   .sort_values("_abs_ig", ascending=False)
                   .head(half))
        # Top protective
        df_prot = (df_pw[df_pw["ig_signed_mean"] < 0]
                   .sort_values("_abs_ig", ascending=False)
                   .head(half))
        
        top_paths = pd.concat([df_risk, df_prot]).sort_values(
            "ig_signed_mean", ascending=True)

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
    c_title = f"C: AToP temporal patterns\n(top {half} risk + top {half} protective)"
    if panel_c_label:
        c_title += f" — {panel_c_label}"
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


def fig_supp_conditional_impact(out_dir, df_pathways, icd_titles, n_show=15,
                                 split="test", min_concordant=5,
                                 n_test_patients=None):
    """Supplementary figure: conditional impact score for temporal patterns.

    Ranks patterns by E[IG | present, IG concordant with OR direction],
    highlighting patterns that are rare but powerful when present. This
    adapts the conditional SHAP impact score (Appendix E) to AToP's
    pattern-level IG framework.

    For risk patterns (OR > 1): shows E[IG | present, IG > 0]
    For protective patterns (OR <= 1): shows E[|IG| | present, IG < 0]

    Patterns must have at least min_concordant patients to avoid noisy
    estimates from very small samples.
    """
    if df_pathways is None or df_pathways.empty:
        return
    if "ig_cond_impact" not in df_pathways.columns:
        print("  [supp_cond] No conditional impact scores — skipping")
        return

    df = df_pathways[df_pathways["n_concordant"] >= min_concordant].copy()
    if df.empty:
        print(f"  [supp_cond] No patterns with >= {min_concordant} concordant patients — skipping")
        return

    # Split risk and protective, take top of each
    df_risk = df[df["odds_ratio"] > 1].nlargest(n_show, "ig_cond_impact")
    df_prot = df[df["odds_ratio"] <= 1].nsmallest(n_show, "ig_cond_impact")

    n_risk = len(df_risk)
    n_prot = len(df_prot)
    if n_risk + n_prot == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(24, max(6, max(n_risk, n_prot) * 0.5)),
                              gridspec_kw={"width_ratios": [1, 1]})

    fig.suptitle(
        f"Conditional impact: patterns ranked by E[IG | present, direction-consistent] ({split.upper()})\n"
        f"Highlights rare-but-powerful patterns — restricted to patients where pattern "
        f"pushes prediction in the expected direction\n"
        f"(min {min_concordant} direction-consistent patients required)",
        fontsize=11, fontweight="bold", y=1.02)

    _titles = icd_titles or {}

    # Left: Risk patterns
    ax = axes[0]
    if n_risk > 0:
        df_r = df_risk.sort_values("ig_cond_impact", ascending=True)
        labels = []
        for _, r in df_r.iterrows():
            pat_label = _fmt_pat(r["pattern"], _titles, max_tok_len=50)
            labels.append(f"\u25B2 {pat_label}\n    IG>0 in {r['n_concordant']}/{r['n_present']}, "
                          f"OR={r['odds_ratio']:.2f}")
        ax.barh(range(n_risk), df_r["ig_cond_impact"].values, color="#c44e52",
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_risk))
        ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("E[IG | present, IG > 0]\n\u2192 risk contribution when active")
    ax.set_title("Risk patterns (OR > 1)", fontsize=10)

    # Right: Protective patterns
    ax = axes[1]
    if n_prot > 0:
        df_pr = df_prot.sort_values("ig_cond_impact", ascending=False)
        labels = []
        for _, r in df_pr.iterrows():
            pat_label = _fmt_pat(r["pattern"], _titles, max_tok_len=50)
            labels.append(f"\u25BC {pat_label}\n    IG<0 in {r['n_concordant']}/{r['n_present']}, "
                          f"OR={r['odds_ratio']:.2f}")
        ax.barh(range(n_prot), df_pr["ig_cond_impact"].abs().values, color="#4c72b0",
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_prot))
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No protective patterns met threshold",
                ha="center", va="center", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("|E[IG | present, IG < 0]|\n\u2190 protective contribution when active")
    ax.set_title("Protective patterns (OR \u2264 1)", fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, f"supp_conditional_impact_{split}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [supp_cond] Saved \u2192 {path}")

    # --- Prevalence vs Conditional Impact scatter (reviewer suggestion #6) ---
    if df.empty:
        return
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

    n_test = n_test_patients or 1
    df_scatter = df.copy()
    df_scatter["prevalence"] = df_scatter["n_present"] / n_test
    df_scatter["is_risk"] = df_scatter["odds_ratio"] > 1

    for is_risk, grp in df_scatter.groupby("is_risk"):
        color = "#c44e52" if is_risk else "#4c72b0"
        label = "Risk (OR > 1)" if is_risk else "Protective (OR \u2264 1)"
        ax2.scatter(
            grp["prevalence"], grp["ig_cond_impact"].abs(),
            s=grp["n_concordant"].clip(lower=5) * 3,
            c=color, alpha=0.6, edgecolors="white", linewidth=0.5,
            label=label)

    ax2.set_xlabel(f"Prevalence (fraction of {split} patients)", fontsize=10)
    ax2.set_ylabel("Conditional impact |I_j|", fontsize=10)
    ax2.set_xscale("log")
    ax2.legend(fontsize=9)
    ax2.set_title(
        f"Prevalence vs conditional impact ({split.upper()})\n"
        f"Point size = n direction-consistent patients",
        fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    path2 = os.path.join(out_dir, f"supp_prevalence_vs_impact_{split}.png")
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [supp_prev_impact] Saved \u2192 {path2}")


def fig_supp_jaccard_clusters(out_dir, df_pathways, icd_titles, n_show=15,
                               split="test", jaccard_threshold=0.5, short_names=False,
                               csv_dir=None):
    """Supplementary figure: Jaccard-clustered pattern representatives.

    Groups patterns by carrier-set overlap (Jaccard similarity > threshold),
    then shows one representative per cluster (highest |ig_signed_mean|).
    Annotated with cluster size and member count.
    """
    if df_pathways is None or df_pathways.empty:
        return
    if "_carrier_keys" not in df_pathways.columns:
        print("  [supp_jaccard] No carrier sets available — skipping")
        return

    df = df_pathways[df_pathways["n_present"] >= 3].copy()
    if df.empty:
        return

    # Compute Jaccard and greedily cluster (numpy bit-vector optimized)
    patterns = df.sort_values("ig_signed_mean", key=abs, ascending=False).reset_index(drop=True)

    # Build dense patient index mapping for bit vectors
    all_patients = set()
    carrier_list = []
    size_list = []
    for i, row in patterns.iterrows():
        ck = row["_carrier_keys"]
        if isinstance(ck, set):
            all_patients.update(ck)
            carrier_list.append(ck)
            size_list.append(len(ck))
        else:
            carrier_list.append(set())
            size_list.append(0)

    patient_to_idx = {p: i for i, p in enumerate(sorted(all_patients))}
    n_patients = len(patient_to_idx)

    # Pack carrier sets into bit vectors
    print(f"  [supp_jaccard] Building bit vectors for {len(patterns)} patterns...")
    bitvecs = []
    for carriers in carrier_list:
        bits = np.zeros(n_patients, dtype=np.bool_)
        for p in carriers:
            bits[patient_to_idx[p]] = True
        bitvecs.append(np.packbits(bits))

    clusters = []  # list of (representative_idx, member_indices)
    assigned = set()
    n_total = len(patterns)
    report_interval = max(1, n_total // 10)

    for i in range(n_total):
        if i in assigned:
            continue
        if i > 0 and i % report_interval == 0:
            print(f"  [supp_jaccard] {i}/{n_total} processed, {len(clusters)} clusters...")

        if size_list[i] == 0:
            assigned.add(i)
            clusters.append((i, [i]))
            continue

        bv_i = bitvecs[i]
        sz_i = size_list[i]
        members = [i]

        for j in range(i + 1, n_total):
            if j in assigned:
                continue
            sz_j = size_list[j]
            if sz_j == 0:
                continue

            # Quick reject: upper bound on Jaccard
            upper_bound = min(sz_i, sz_j) / max(sz_i, sz_j)
            if upper_bound < jaccard_threshold:
                continue

            # Exact Jaccard via bit operations
            intersection = int(np.unpackbits(bv_i & bitvecs[j]).sum())
            union = int(np.unpackbits(bv_i | bitvecs[j]).sum())
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard >= jaccard_threshold:
                members.append(j)
                assigned.add(j)

        assigned.add(i)
        clusters.append((i, members))

    # Build representative DataFrame
    rep_rows = []
    for rep_idx, member_idxs in clusters:
        rep = patterns.loc[rep_idx].copy()
        rep["cluster_size"] = len(member_idxs)
        rep["cluster_total_patients"] = len(
            set().union(*(patterns.loc[m, "_carrier_keys"] for m in member_idxs))
        )
        rep_rows.append(rep)

    df_reps = pd.DataFrame(rep_rows)
    if df_reps.empty:
        return

    # Split risk/protective, take top of each
    df_risk = df_reps[df_reps["ig_signed_mean"] > 0].nlargest(n_show, "ig_signed_mean")
    df_prot = df_reps[df_reps["ig_signed_mean"] < 0].nsmallest(n_show, "ig_signed_mean")
    n_risk = len(df_risk)
    n_prot = len(df_prot)
    if n_risk + n_prot == 0:
        return

    _titles = icd_titles or {}
    fig, axes = plt.subplots(1, 2, figsize=(24, max(6, max(n_risk, n_prot) * 0.55)),
                              gridspec_kw={"width_ratios": [1, 1]})

    fig.suptitle(
        f"Jaccard-clustered pattern representatives ({split.upper()})\n"
        f"Patterns grouped by carrier-set overlap (Jaccard ≥ {jaccard_threshold}), "
        f"one representative per cluster\n"
        f"{len(clusters)} clusters from {len(patterns)} patterns",
        fontsize=11, fontweight="bold", y=1.02)

    # Left: Risk
    ax = axes[0]
    if n_risk > 0:
        df_r = df_risk.sort_values("ig_signed_mean", ascending=True)
        labels = []
        for _, r in df_r.iterrows():
            if "pattern_readable" in r.index and pd.notna(r.get("pattern_readable")):
                pat_label = r["pattern_readable"]
            else:
                pat_label = _fmt_pat(r["pattern"], _titles, max_tok_len=45)
            if short_names:
                pat_label = _strip_codes(pat_label)
            labels.append(f"\u25B2 {pat_label}\n    cluster={r['cluster_size']}, "
                          f"patients={r['cluster_total_patients']}, n={r['n_present']}")
        ax.barh(range(n_risk), df_r["ig_signed_mean"].values, color="#c44e52",
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_risk))
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean signed IG(P)\n\u2192 risk contribution")
    ax.set_title("Risk clusters (IG > 0)", fontsize=10)

    # Right: Protective
    ax = axes[1]
    if n_prot > 0:
        df_pr = df_prot.sort_values("ig_signed_mean", ascending=False)
        labels = []
        for _, r in df_pr.iterrows():
            if "pattern_readable" in r.index and pd.notna(r.get("pattern_readable")):
                pat_label = r["pattern_readable"]
            else:
                pat_label = _fmt_pat(r["pattern"], _titles, max_tok_len=45)
            if short_names:
                pat_label = _strip_codes(pat_label)
            labels.append(f"\u25BC {pat_label}\n    cluster={r['cluster_size']}, "
                          f"patients={r['cluster_total_patients']}, n={r['n_present']}")
        ax.barh(range(n_prot), df_pr["ig_signed_mean"].abs().values, color="#4c72b0",
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_prot))
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("|Mean signed IG(P)|\n\u2192 protective contribution")
    ax.set_title("Protective clusters (IG < 0)", fontsize=10)

    fig.tight_layout()
    jt_str = f"_{jaccard_threshold:.2f}".replace(".", "")
    path = os.path.join(out_dir, f"supp_jaccard_clusters_{split}{jt_str}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [supp_jaccard] Saved \u2192 {path}")

    # Save full cluster assignments as CSV (for downstream network analysis)
    cluster_rows = []
    for ci, (rep_idx, member_idxs) in enumerate(clusters):
        rep_pat = patterns.loc[rep_idx, "pattern"]
        rep_ig = patterns.loc[rep_idx, "ig_signed_mean"]
        cluster_patients = set().union(*(patterns.loc[m, "_carrier_keys"] for m in member_idxs))
        for m_idx in member_idxs:
            row = patterns.loc[m_idx]
            cluster_rows.append({
                "cluster_id": ci,
                "is_representative": m_idx == rep_idx,
                "pattern": row["pattern"],
                "ig_signed_mean": row["ig_signed_mean"],
                "n_present": row["n_present"],
                "odds_ratio": row.get("odds_ratio", None),
                "cluster_size": len(member_idxs),
                "cluster_total_patients": len(cluster_patients),
                "cluster_rep_pattern": rep_pat,
                "cluster_rep_ig": rep_ig,
                "cluster_direction": "risk" if rep_ig > 0 else "protective",
            })
    df_clusters = pd.DataFrame(cluster_rows)
    _csv_dir = csv_dir if csv_dir else out_dir
    csv_path = os.path.join(_csv_dir, f"jaccard_clusters_{split}{jt_str}.csv")
    df_clusters.drop(columns=["_carrier_keys"], errors="ignore").to_csv(csv_path, index=False)
    print(f"  [supp_jaccard] Cluster assignments saved \u2192 {csv_path} "
          f"({len(clusters)} clusters, {len(cluster_rows)} pattern-cluster pairs)")


def _compute_panel_c(df_patterns, df_seq, df_ig, mining_method, label=""):
    """Compute Panel C pattern results using occurrence-aggregated attribution.

    For each pattern, scans all test patients in df_ig for occurrences
    where the pattern tokens appear in the correct temporal order (by
    visit_idx). For each patient containing the pattern, computes a
    patient-level score = sum of signed IG at the matched positions
    (first occurrence, greedy left-to-right). The global pattern
    importance is the mean of these patient-level scores over all
    patients who contain the pattern.

    This makes Panel C directly comparable to Panel A (mean IG per token
    over all test patients): both are computed on the same cohort using
    the same IG values, just at different granularity (single-token vs
    multi-token sequential pattern).
    """
    is_episode = (mining_method == "episode")
    pattern_results = []
    if df_patterns.empty or df_ig.empty:
        return pd.DataFrame(pattern_results)

    ig_col = "ig_signed" if "ig_signed" in df_ig.columns else "ig_abs"
    has_visit_idx = "visit_idx" in df_ig.columns
    if not has_visit_idx:
        print(f"  [{label}] WARNING: df_ig missing visit_idx — "
              "cannot verify temporal order, falling back to token co-occurrence only")

    # ── Build per-patient token→[(visit_idx, ig_value)] lookup ───────────
    # Group by patient, then within each patient collect (visit_idx, ig) per token.
    # For occurrence matching we need to know which visit each token appeared in.
    print(f"  [{label}] Building per-patient token occurrence index...")
    patient_tokens = {}  # key → {token_str: [(visit_idx, ig_val), ...]}
    for (pid, hid), grp in df_ig.groupby(["patient_id", "index_hadm_id"]):
        key = (str(pid), str(hid))
        tok_occ = {}
        for _, row in grp.iterrows():
            tok = row["token_str"]
            vidx = int(row["visit_idx"]) if has_visit_idx else 0
            ig_val = float(row[ig_col])
            if tok not in tok_occ:
                tok_occ[tok] = []
            tok_occ[tok].append((vidx, ig_val))
        # Sort each token's occurrences by visit_idx for greedy matching
        for tok in tok_occ:
            tok_occ[tok].sort(key=lambda x: x[0])
        patient_tokens[key] = tok_occ

    n_patients_total = len(patient_tokens)
    prefix = f"    [{label}] " if label else "    "
    print(f"{prefix}{n_patients_total} test patients indexed")

    # ── Score each pattern ───────────────────────────────────────────────
    n_no_match = 0
    for pat_idx, pat_row in df_patterns.iterrows():
        pat_str = pat_row["pattern"]
        orr = pat_row["odds_ratio"]
        sup = pat_row["n_admissions_present"]

        if is_episode:
            ep = _parse_episode_pattern(pat_str)
            # Episode: list of frozensets, each frozenset is a within-visit itemset
            # Temporal order: itemset i must match visit ≤ itemset i+1's visit
            pat_itemsets = [list(s) for s in ep]
        else:
            flat = [p.strip() for p in pat_str.split(" -> ")]
            # Flat pattern: each token is its own singleton "itemset"
            pat_itemsets = [[t] for t in flat]

        all_tokens_needed = set()
        for itemset in pat_itemsets:
            all_tokens_needed.update(itemset)

        patient_scores = []
        n_concordant = 0
        matched_patient_keys = set()

        for key, tok_occ in patient_tokens.items():
            # Quick check: does this patient have all required tokens?
            if not all_tokens_needed.issubset(tok_occ.keys()):
                continue

            # Greedy left-to-right matching: for each itemset in order,
            # find the earliest visit_idx ≥ previous match where all
            # tokens in the itemset are present.
            matched_positions = []  # list of (token, visit_idx, ig_val)
            match_ok = True
            min_visit = -1  # next itemset must be at visit > this (cross-visit)
                            # or >= this (same-visit allowed for episode within-set)

            for itemset_i, itemset in enumerate(pat_itemsets):
                # For episode patterns: all tokens in this itemset must appear
                # in the same visit, at visit_idx > min_visit (strict for cross-visit)
                # For the first itemset, any visit is fine.
                #
                # Strategy: find the earliest visit where all tokens in this
                # itemset are present at visit_idx > min_visit.

                # Collect candidate visits: visits where ALL tokens in this itemset appear
                token_visit_sets = []
                for tok in itemset:
                    visits_for_tok = {v for v, _ in tok_occ[tok] if v > min_visit}
                    token_visit_sets.append(visits_for_tok)

                if not token_visit_sets:
                    match_ok = False
                    break

                # Intersect: visits where all tokens in this itemset co-occur
                common_visits = token_visit_sets[0]
                for vs in token_visit_sets[1:]:
                    common_visits &= vs
                    if not common_visits:
                        break

                if not common_visits:
                    match_ok = False
                    break

                # Pick earliest valid visit
                chosen_visit = min(common_visits)
                min_visit = chosen_visit  # next itemset must be strictly after

                # Collect IG values for each token at this visit
                for tok in itemset:
                    # Find the IG for this token at chosen_visit
                    for v, ig_val in tok_occ[tok]:
                        if v == chosen_visit:
                            matched_positions.append((tok, v, ig_val))
                            break

            if not match_ok or not matched_positions:
                continue

            # Patient-level pattern score: sum of IG at matched positions
            patient_ig_sum = sum(ig_val for _, _, ig_val in matched_positions)
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

    print(f"[fig6] Building global 3-panel importance comparison ({split})...")
    if df_ig.empty:
        print(f"  [fig6] Skipped: no IG data available.")
        return

    ig_col = "ig_signed" if "ig_signed" in df_ig.columns else "ig_abs"

    # ── Panel A: IG on all test patients ─────────────────────────────────
    # Consistent with Panel C: for each token, sum IG per patient (across
    # all salient positions of that token), then average across patients.
    # This makes Panel A and Panel C directly comparable — both report
    # mean patient-level IG contribution.
    print(f"  [fig6] Panel A: aggregating IG (signed, all {split} patients)...")
    ig_filtered = df_ig[~df_ig["token_str"].isin(SPECIAL_TOKENS)].copy()
    tok_patient_counts = ig_filtered.groupby("token_str")["patient_id"].nunique()
    common_tokens = set(tok_patient_counts[tok_patient_counts >= 10].index)
    ig_common = ig_filtered[ig_filtered["token_str"].isin(common_tokens)]
    # Step 1: sum IG per (patient, token) — a patient with SI at 3 positions gets one sum
    ig_per_patient = ig_common.groupby(["token_str", "patient_id"])[ig_col].sum().reset_index()
    # Step 2: mean across patients — each patient contributes equally
    ig_global = ig_per_patient.groupby("token_str")[ig_col].mean()

    # ── Panel B: GradientSHAP ─────────────────────────────────────────────────
    shap_global = pd.Series(dtype=float)
    n_shap_patients = 0
    test_keys = list(tensors_by_key.keys()) if tensors_by_key else []

    if not test_keys:
        print("  [fig6] Panel B: skipping GradientSHAP (no tensors available)")
    else:
        print(f"  [fig6] Panel B: computing GradientSHAP (all {len(test_keys)} patients)...")
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
            # Return (B, 1) — GradientExplainer indexes output with 2D indexing
            probs = torch.sigmoid(logits.view(-1, 1))
            return probs

    if not _skip_shap:
        model.eval()
        shap_accum: Counter = Counter()
        shap_count: Counter = Counter()

        # Check for cached SHAP results
        shap_cache_path = os.path.join(csv_dir, f"shap_global_{split}.csv") if csv_dir else None
        if shap_cache_path and os.path.exists(shap_cache_path):
            print(f"  [fig6] Panel B: loading cached GradientSHAP from {shap_cache_path}")
            _shap_df = pd.read_csv(shap_cache_path)
            shap_global = pd.Series(_shap_df["shap_mean"].values, index=_shap_df["token"].values)
            _skip_shap = True  # skip computation, use cached
        elif shap_cache_path and os.path.exists(shap_cache_path + ".partial"):
            print(f"  [fig6] Panel B: found partial SHAP cache — loading as fallback")
            _shap_df = pd.read_csv(shap_cache_path + ".partial")
            shap_global = pd.Series(_shap_df["shap_mean"].values, index=_shap_df["token"].values)
            # Save as full cache — partial is good enough for global ranking
            _shap_df.to_csv(shap_cache_path, index=False)
            print(f"  [fig6] Panel B: promoted partial ({len(_shap_df)} tokens) to full cache")
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
                print(f"  [fig6] Panel B: cached GradientSHAP → {shap_cache_path}")

    # ── Panel C: occurrence-aggregated pattern attribution ──────────────
    # Check for cached carrier sets — skip expensive recomputation if available
    print("  [fig6] Panel C: computing occurrence-aggregated pattern IG...")
    
    _panel_c_cached = False
    if csv_dir:
        import pickle as _pkl
        _all_pkl = os.path.join(csv_dir, f"carrier_sets_{split}_all.pkl")
        _all_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}_all.csv")
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
                    print(f"  [fig6] Panel C: loaded {len(df_pathways_all)} cached patterns "
                          f"from {_all_csv}")
                    _panel_c_cached = True
                else:
                    print(f"  [fig6] Panel C: cache mismatch ({len(cached_pats)} cached vs "
                          f"{len(current_pats)} current) — recomputing")
            except Exception as e:
                print(f"  [fig6] Panel C: cache load failed ({e}) — recomputing")
    
    if not _panel_c_cached:
        df_pathways_all = _compute_panel_c(df_patterns, df_seq, df_ig, mining_method, label="all")

    # Main figure: multi-token patterns (≥2 tokens, incl. same-visit co-occurrences)
    df_patterns_multi_set = set(_filter_multi_token(df_patterns, min_tokens=2)["pattern"].tolist())
    n_multi = len(df_patterns_multi_set)
    print(f"  [fig6] Panel C multi-token: {n_multi}/{len(df_patterns)} patterns with \u22652 tokens")
    df_pathways_multi = df_pathways_all[df_pathways_all["pattern"].isin(df_patterns_multi_set)].copy()

    # Supplement: cross-visit only (≥2 visit blocks, ≥2 tokens)
    df_patterns_xvisit_set = set(_filter_cross_visit(
        _filter_multi_token(df_patterns, min_tokens=2), min_blocks=2)["pattern"].tolist())
    n_xvisit = len(df_patterns_xvisit_set)
    print(f"  [fig6] Panel C cross-visit: {n_xvisit}/{len(df_patterns)} patterns "
          f"with \u22652 tokens AND \u22652 visit blocks")
    df_pathways_xvisit = df_pathways_all[df_pathways_all["pattern"].isin(df_patterns_xvisit_set)].copy()

    # ── Render: multi-token (main paper figure) ──────────────────────────
    fig = _render_fig6(out_dir, icd_titles, ig_global, shap_global,
                       df_pathways_multi, n_show, len(test_keys), split,
                       panel_c_label="\u22652-token patterns (incl. same-visit)")
    path_main = os.path.join(out_dir, f"fig3_global_comparison_{split}.png")
    fig.savefig(path_main, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig3] Saved (main) \u2192 {path_main}")

    # ── Render: balanced risk/protective (supplement) ─────────────────────
    fig = _render_fig6_split(supp_dir, icd_titles, ig_global, shap_global,
                             df_pathways_multi, n_show, len(test_keys), split,
                             panel_c_label="\u22652-token patterns")
    path_split = os.path.join(supp_dir, f"fig3_global_comparison_{split}_split.png")
    fig.savefig(path_split, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig3] Saved (supplement split) \u2192 {path_split}")

    # ── Render: cross-visit supplement ────────────────────────────────────
    fig = _render_fig6(supp_dir, icd_titles, ig_global, shap_global,
                       df_pathways_xvisit, n_show, len(test_keys), split,
                       panel_c_label="cross-visit patterns only")
    path_supp_xvisit = os.path.join(supp_dir, f"fig3_global_comparison_{split}_xvisit.png")
    fig.savefig(path_supp_xvisit, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig3] Saved (supplement cross-visit) \u2192 {path_supp_xvisit}")

    # ── Render: all patterns (supplement) ─────────────────────────────────
    fig = _render_fig6(supp_dir, icd_titles, ig_global, shap_global,
                       df_pathways_all, n_show, len(test_keys), split,
                       panel_c_label="all patterns (incl. single-token)")
    path_supp = os.path.join(supp_dir, f"fig3_global_comparison_{split}_all.png")
    fig.savefig(path_supp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig3] Saved (supplement all) \u2192 {path_supp}")

    # ── Supplementary: conditional impact (rare-but-powerful patterns) ────
    try:
        fig_supp_conditional_impact(supp_dir, df_pathways_multi, icd_titles,
                                     n_show=n_show, split=split,
                                     n_test_patients=len(test_keys))
    except Exception as e:
        print(f"  [WARN] Conditional impact figure failed: {e}")

    # ── Supplementary: Jaccard-clustered representatives ──────────────────
    try:
        fig_supp_jaccard_clusters(supp_dir, df_pathways_multi, icd_titles,
                                   n_show=n_show, split=split, csv_dir=csv_dir)
    except Exception as e:
        print(f"  [WARN] Jaccard cluster figure failed: {e}")

    # ── CSVs ──────────────────────────────────────────────────────────────
    n_test_total = len(test_keys)
    for lbl, df_pw, sfx in [("all", df_pathways_all, "_all"),
                             ("multi", df_pathways_multi, ""),
                             ("xvisit", df_pathways_xvisit, "_xvisit")]:
        if not df_pw.empty:
            df_out = df_pw.drop(columns=["pat_tokens", "_carrier_keys"], errors="ignore").copy()
            df_out["prevalence"] = df_out["n_present"] / max(n_test_total, 1)
            df_out["pattern_readable"] = df_out["pattern"].apply(
                lambda p: _fmt_pat(p, icd_titles, max_tok_len=80))
            df_out.to_csv(
                os.path.join(csv_dir, f"fig3_pathway_importance_{split}{sfx}.csv"), index=False)

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
                    print(f"  [carriers] Saved {len(carriers)} carrier sets → {pkl_path}")

    all_toks = sorted(set(ig_global.index) | set(shap_global.index))
    tok_rows = []
    n_test_total = len(test_keys) if test_keys else 0
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
        os.path.join(csv_dir, f"fig3_token_importance_{split}.csv"), index=False)

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
        print("  [fig5] No reversed-order data — skipping")
        return
    if csv_dir is None:
        csv_dir = out_dir

    _titles = icd_titles or {}

    # Filter to patterns with both forward and reversed matches
    df_plot = df_rev[(df_rev["n_forward"] > 0) & (df_rev["n_reversed"] > 0)].copy()
    if df_plot.empty:
        print("  [fig5] No patterns with both forward and reversed matches — skipping")
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
    print(f"  [fig5] Saved \u2192 {path}")

    # CSV
    df_rev.to_csv(os.path.join(csv_dir, f"fig5_reversed_order_{split}.csv"), index=False)


# ============================================================================
# SUPPLEMENTAL FIGURES
# ============================================================================


def fig_supp_ig_heatmap_multi_patient(out_dir, df_ig_long, icd_titles,
                                       n_patients=10, n_tokens=15, split="test"):
    if df_ig_long.empty:
        return
    ig_col = "ig_signed" if "ig_signed" in df_ig_long.columns else "ig_abs"

    # Select top 5 readmitted + top 5 non-readmitted by total |IG|
    patient_ig = (df_ig_long.groupby(["patient_id", "index_hadm_id", "readmission"])[ig_col]
                  .apply(lambda x: x.abs().sum()).reset_index(name="_total_ig"))
    n_half = n_patients // 2
    readmitted = patient_ig[patient_ig["readmission"] == 1].nlargest(n_half, "_total_ig")
    not_readmitted = patient_ig[patient_ig["readmission"] == 0].nlargest(n_half, "_total_ig")
    top_patients = pd.concat([readmitted, not_readmitted], ignore_index=True)

    sub = df_ig_long.merge(top_patients[["patient_id", "index_hadm_id"]],
                           on=["patient_id", "index_hadm_id"])
    # Top tokens by mean |signed IG|
    token_importance = sub.groupby("token_str")[ig_col].apply(lambda x: x.abs().mean())
    token_importance = token_importance.sort_values(ascending=False)
    top_tokens = token_importance.head(n_tokens).index.tolist()

    rows, row_labels = [], []
    for _, p in top_patients.iterrows():
        psub = sub[(sub.patient_id == p["patient_id"]) & (sub.index_hadm_id == p["index_hadm_id"])]
        tok_vals = {r["token_str"]: r[ig_col] for _, r in psub.iterrows()}
        rows.append([tok_vals.get(t, 0.0) for t in top_tokens])
        row_labels.append(f"{'✓' if p['readmission'] == 1 else '✗'} P{str(p['patient_id'])[-4:]}")

    mat = np.array(rows)
    col_labels = [_fmt_tok(t, icd_titles) for t in top_tokens]
    # Truncate long labels
    col_labels = [l if len(l) <= 55 else l[:54] + "…" for l in col_labels]

    vmax = max(abs(mat.min()), abs(mat.max())) or 1.0
    fig, ax = plt.subplots(figsize=(max(10, n_tokens * 0.8), max(4, n_patients * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(f"Signed IG heatmap: top patients × top tokens ({split.upper()})\n"
                 f"Top {n_half} readmitted (✓) + top {n_half} non-readmitted (✗)",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="Signed IG\n← protective    risk →", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_supp_ig_heatmap_{split}.png"), dpi=300)
    plt.close(fig)


def fig_supp_ig_stream_heatmap(out_dir, df_ig_long, split="test"):
    if df_ig_long.empty:
        return
    ig_col = "ig_signed" if "ig_signed" in df_ig_long.columns else "ig_abs"
    grouped = (df_ig_long.groupby(["patient_id", "index_hadm_id", "readmission", "stream"])[ig_col]
               .apply(lambda x: x.abs().sum()).reset_index(name="_total"))
    pivot = grouped.pivot_table(index="readmission", columns="stream", values="_total", aggfunc="mean")
    col_order = [c for c in ["C", "P", "D"] if c in pivot.columns]
    if not col_order:
        print("  [WARN] fig_supp_ig_stream_heatmap: no stream columns found, skipping")
        return
    pivot = pivot[col_order]
    col_labels = {"C": "Conditions (C:)", "P": "Procedures (P:)", "D": "Drugs (D:)"}
    pivot.columns = [col_labels.get(c, c) for c in pivot.columns]
    pivot.index = ["Not readmitted (0)" if v == 0 else "Readmitted (1)" for v in pivot.index]

    fig, ax = plt.subplots(figsize=(max(6, len(col_order) * 3), 3))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            color = "white" if val > pivot.values.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, color=color)
    ax.set_title(f"Mean total |IG| per sample by stream and readmission status ({split.upper()})",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="Mean |IG|", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_supp_ig_stream_heatmap_{split}.png"), dpi=300)
    plt.close(fig)


def fig_supp_pattern_admission_heatmap(out_dir, df_patterns, df_seq, icd_titles,
                                        mining_method, df_ig=None, top_k=10,
                                        max_admissions=30, split="test"):
    if df_patterns.empty or df_seq.empty:
        return
    is_episode = (mining_method == "episode")

    # Compute mean signed IG per pattern for ranking
    df_p = df_patterns.copy()
    if df_ig is not None and not df_ig.empty:
        ig_cache = _build_ig_cache(df_ig)

        pat_index = PatternIndex.from_df(df_seq)
        seq_rows = df_seq.reset_index(drop=True)

        ig_vals = []
        for _, pr in df_p.iterrows():
            pat_str = pr["pattern"]
            if is_episode:
                ep = _parse_episode_pattern(pat_str)
                pat_fsets = [frozenset(s) for s in ep]
                pat_tokens = set()
                for s in ep:
                    pat_tokens.update(s)
            else:
                flat = [p.strip() for p in pat_str.split(" -> ")]
                pat_fsets = [frozenset([t]) for t in flat]
                pat_tokens = set(flat)

            matched = pat_index.patients_matching_pattern(pat_fsets, pat_tokens, is_episode)
            stats = _compute_pattern_ig_stats(ig_cache, pat_tokens, matched, seq_rows)
            ig_vals.append(stats["ig_signed_mean"] if stats else 0.0)
        df_p["ig_signed_mean"] = ig_vals
        top = df_p.reindex(df_p["ig_signed_mean"].abs().nlargest(top_k).index)
    else:
        top = df_p.head(top_k)

    patterns = []
    for _, row in top.iterrows():
        pat_str = row["pattern"]
        if is_episode:
            ep = _parse_episode_pattern(pat_str)
            pat_fsets = [frozenset(s) for s in ep]
            pat_tokens = set()
            for s in ep:
                pat_tokens.update(s)
        else:
            flat = [p.strip() for p in pat_str.split(" -> ")]
            pat_fsets = [frozenset([t]) for t in flat]
            pat_tokens = set(flat)
        pat_readable = _fmt_pat(pat_str, icd_titles, max_tok_len=40)
        ig_val = row.get("ig_signed_mean", 0.0)
        patterns.append((pat_fsets, pat_tokens, pat_readable, ig_val))

    df_sorted = df_seq.sort_values("readmission", ascending=False).head(max_admissions)

    # Use PatternIndex for fast matching
    pat_index2 = PatternIndex.from_df(df_sorted.reset_index(drop=True))

    mat = np.zeros((len(patterns), len(df_sorted)))
    for i, (pf, pt, _, _) in enumerate(patterns):
        matched = pat_index2.patients_matching_pattern(pf, pt, is_episode)
        for j in matched:
            mat[i, j] = 1.0

    adm_labels = [f"{'✓' if r['readmission'] == 1 else '✗'}" for _, r in df_sorted.iterrows()]
    pat_labels = [f"{pr} (IG={ig_val:+.3f})" for _, _, pr, ig_val in patterns]

    fig, ax = plt.subplots(figsize=(max(8, len(df_sorted) * 0.3), max(5, len(patterns) * 0.55)))
    ax.imshow(mat, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(len(adm_labels)))
    ax.set_xticklabels(adm_labels, fontsize=6)
    ax.set_xlabel("Samples (✓ = readmitted, ✗ = not)")
    ax.set_yticks(range(len(pat_labels)))
    ax.set_yticklabels(pat_labels, fontsize=7.5)
    ax.set_title(f"Pattern presence across samples ({split.upper()}, top {top_k} by |mean IG mass|)",
                 fontsize=10)
    n_readmit = int(df_sorted["readmission"].sum())
    if 0 < n_readmit < len(df_sorted):
        ax.axvline(n_readmit - 0.5, color="red", linewidth=1.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_supp_pattern_presence_{split}.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def fig_supp_pattern_decomposition(out_dir, df_patterns, df_seq, df_ig, icd_titles,
                                    mining_method, top_k=10, split="test"):
    """Pattern × constituent tokens heatmap, colored by mean signed IG.
    
    Rows = top patterns (by |mean signed IG sum|)
    Columns = union of constituent tokens (aligned so shared tokens line up)
    Cells = mean signed IG of that token across patients where that pattern is present
    """
    if df_patterns.empty or df_seq.empty or df_ig is None or df_ig.empty:
        return
    is_episode = (mining_method == "episode")

    # Raw signed IG cache
    ig_cache = _build_ig_cache(df_ig)

    pat_index = PatternIndex.from_df(df_seq)
    seq_rows = df_seq.reset_index(drop=True)

    # Compute per-token mean IG and overall signed sum per pattern
    pattern_info = []
    for _, pr in df_patterns.iterrows():
        pat_str = pr["pattern"]
        if is_episode:
            ep = _parse_episode_pattern(pat_str)
            pat_fsets = [frozenset(s) for s in ep]
            pat_tokens = set()
            for s in ep:
                pat_tokens.update(s)
        else:
            flat = [p.strip() for p in pat_str.split(" -> ")]
            pat_fsets = [frozenset([t]) for t in flat]
            pat_tokens = set(flat)

        matched = pat_index.patients_matching_pattern(pat_fsets, pat_tokens, is_episode)

        # Per-token mean IG across matched patients
        tok_ig_accum = {t: [] for t in pat_tokens}
        signed_sums = []
        for ri in matched:
            row = seq_rows.iloc[ri]
            key = (str(row["patient_id"]), str(row["index_hadm_id"]))
            tok_ig = ig_cache.get(key, {})
            pat_vals = []
            for t in pat_tokens:
                if t in tok_ig:
                    tok_ig_accum[t].append(tok_ig[t])
                    pat_vals.append(tok_ig[t])
            if pat_vals:
                signed_sums.append(sum(pat_vals))

        tok_mean_ig = {t: float(np.mean(v)) if v else 0.0 for t, v in tok_ig_accum.items()}
        ig_signed_mean = float(np.mean(signed_sums)) if signed_sums else 0.0

        pattern_info.append({
            "pattern": pat_str,
            "pat_tokens": pat_tokens,
            "tok_mean_ig": tok_mean_ig,
            "ig_signed_mean": ig_signed_mean,
            "n_matched": len(matched),
        })

    if not pattern_info:
        return

    # Rank by |mean signed IG sum|, take top_k
    pattern_info.sort(key=lambda x: abs(x["ig_signed_mean"]), reverse=True)
    pattern_info = pattern_info[:top_k]

    # Collect all unique tokens across selected patterns
    all_tokens = []
    seen = set()
    for pi in pattern_info:
        for t in sorted(pi["pat_tokens"]):
            if t not in seen:
                all_tokens.append(t)
                seen.add(t)

    # Build matrix
    mat = np.full((len(pattern_info), len(all_tokens)), np.nan)
    for i, pi in enumerate(pattern_info):
        for j, tok in enumerate(all_tokens):
            if tok in pi["tok_mean_ig"]:
                mat[i, j] = pi["tok_mean_ig"][tok]

    # Labels
    row_labels = [_fmt_pat(pi["pattern"], icd_titles, max_tok_len=45) +
                  f" (n={pi['n_matched']})" for pi in pattern_info]
    col_labels = [_fmt_tok(t, icd_titles) for t in all_tokens]
    col_labels = [l if len(l) <= 40 else l[:39] + "…" for l in col_labels]

    # Plot
    vmax = np.nanmax(np.abs(mat)) or 1.0
    fig, ax = plt.subplots(figsize=(max(8, len(all_tokens) * 1.0),
                                     max(4, len(pattern_info) * 0.6)))

    # Use masked array for NaN cells
    masked_mat = np.ma.masked_invalid(mat)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#f0f0f0")  # light gray for cells where token not in pattern

    im = ax.imshow(masked_mat, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=-vmax, vmax=vmax)

    # Add text values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                val = mat[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title(f"Pattern decomposition: token-level IG within top patterns ({split.upper()})\n"
                 f"Gray = token not in pattern",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="Mean signed IG\n← protective    risk →", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_supp_pattern_decomposition_{split}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


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


def fig_supp_ood_diagnostic(out_dir, df_val, split="test"):
    """Supplementary: OOD diagnostic for perturbation validation.

    Shows distribution of predicted probabilities under each perturbation type
    vs baseline. If perturbed predictions remain in a reasonable range (not
    collapsing to 0/1), perturbation results reflect genuine model reliance
    rather than OOD confusion.
    """
    if df_val is None or df_val.empty:
        return
    # Check if raw probs are available
    if "_base_probs" not in df_val.columns:
        print("  [supp_ood] No raw predictions stored — skipping")
        return

    # Collect all predictions across patterns
    base_all, masked_all, shuffle_within_all, shuffle_visit_all = [], [], [], []
    for _, row in df_val.iterrows():
        bp = row.get("_base_probs")
        mp = row.get("_masked_probs")
        sw = row.get("_shuffled_within_probs_flat")
        sv = row.get("_shuffled_visit_probs_flat")
        if bp is not None and isinstance(bp, list):
            base_all.extend(bp)
        if mp is not None and isinstance(mp, list):
            masked_all.extend(mp)
        if sw is not None and isinstance(sw, list):
            shuffle_within_all.extend(sw)
        if sv is not None and isinstance(sv, list):
            shuffle_visit_all.extend(sv)

    if not base_all:
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    bins = np.linspace(0, 1, 41)

    data_sets = [
        ("Baseline\n(unperturbed)", base_all, "#555555"),
        ("Mask pattern\ntokens with PAD", masked_all, "#c44e52"),
        ("Shuffle visit\nblock order", shuffle_visit_all, "#e8a0a2"),
        ("Shuffle within\nvisits (control)", shuffle_within_all, "#a0b8d0"),
    ]

    for ax, (title, data, color) in zip(axes, data_sets):
        if not data:
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        arr = np.array(data)
        ax.hist(arr, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Predicted P(readmission)")
        ax.axvline(arr.mean(), color="black", linestyle="--", linewidth=1.0,
                   label=f"mean={arr.mean():.3f}")
        # OOD indicators
        pct_extreme = float(((arr < 0.01) | (arr > 0.99)).mean() * 100)
        pct_reasonable = float(((arr >= 0.05) & (arr <= 0.95)).mean() * 100)
        ax.text(0.97, 0.95,
                f"N={len(arr):,}\n"
                f"mean={arr.mean():.3f}\n"
                f"std={arr.std():.3f}\n"
                f"in [0.05,0.95]: {pct_reasonable:.0f}%\n"
                f"extreme (<0.01 or >0.99): {pct_extreme:.1f}%",
                transform=ax.transAxes, fontsize=7.5,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[0].set_ylabel("Count")
    fig.suptitle(
        f"OOD diagnostic ({split.upper()}): prediction distributions under perturbation\n"
        "If perturbed distributions remain well-spread (not collapsing to 0 or 1),\n"
        "perturbation Δŷ reflects model reliance, not out-of-distribution confusion.",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(out_dir, f"supp_ood_diagnostic_{split}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [supp_ood] Saved → {path}")


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
