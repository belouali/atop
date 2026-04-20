#!/usr/bin/env python3
"""
Sensitivity analysis: stability of mined patterns across IG mass thresholds.

For each threshold (e.g., 50%, 80%, 95%), re-selects salient tokens from
existing IG values, rebuilds salient visit blocks, re-mines episode patterns,
and compares the resulting top-K pattern sets via Jaccard similarity.

This addresses the reviewer concern about circularity: if patterns are stable
across thresholds, the specific IG cutoff doesn't drive the results.

Requires: a completed AToP run with saved df_ig and df_seq.

Usage:
  python scripts/sensitivity_ig_threshold.py \
      --run_dir runs/exp01 --mimic_dir /data/mimiciv/hosp \
      --out_dir explanations/ \
      --thresholds 0.50,0.80,0.95 --top_k 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def reselect_salient(df_ig_patient: pd.DataFrame, mass: float):
    """Re-run saliency selection at a given mass threshold for one patient.

    Returns set of salient token strings.
    """
    ig_col = "ig_signed" if "ig_signed" in df_ig_patient.columns else "ig_abs"
    items = df_ig_patient.copy()
    items["_abs"] = items[ig_col].abs()
    items = items.sort_values("_abs", ascending=False)
    total = items["_abs"].sum()
    if total < 1e-12:
        return set()
    cumsum = items["_abs"].cumsum()
    target = mass * total
    n_keep = (cumsum <= target).sum() + 1
    n_keep = min(n_keep, len(items))
    return set(items.iloc[:n_keep]["token_str"].values)


def rebuild_visit_blocks(df_ig_patient: pd.DataFrame, salient_set: set):
    """Rebuild salient visit blocks from df_ig rows filtered to salient tokens.

    Requires visit_idx column.
    """
    has_vi = "visit_idx" in df_ig_patient.columns
    if not has_vi:
        raise ValueError("df_ig must have visit_idx column. Re-run IG computation.")

    visit_to_tokens = {}
    for _, row in df_ig_patient.iterrows():
        tok = row["token_str"]
        if tok not in salient_set:
            continue
        vidx = int(row["visit_idx"])
        if vidx < 0:
            continue
        if vidx not in visit_to_tokens:
            visit_to_tokens[vidx] = set()
        visit_to_tokens[vidx].add(tok)

    blocks = []
    for vidx in sorted(visit_to_tokens.keys()):
        blocks.append(frozenset(visit_to_tokens[vidx]))
    return blocks


def mine_at_threshold(df_ig: pd.DataFrame, mass: float,
                      min_support_frac: float, max_len: int, topn: int):
    """Re-mine patterns at a given IG mass threshold.

    Returns df_patterns (or empty DataFrame).
    """
    from atop.mining.patterns import mine_patterns

    # Rebuild salient visit blocks per patient
    all_salient_flat = []
    all_visit_blocks = []
    all_y = []

    for (pid, hid), grp in df_ig.groupby(["patient_id", "index_hadm_id"]):
        salient_set = reselect_salient(grp, mass)
        blocks = rebuild_visit_blocks(grp, salient_set)
        all_salient_flat.append(sorted(salient_set))
        all_visit_blocks.append(blocks)
        # Use readmission label from df_ig
        all_y.append(int(grp["readmission"].iloc[0]))

    y = np.array(all_y)

    print(f"    Threshold {mass:.0%}: {len(all_visit_blocks)} patients, "
          f"mean {np.mean([len(b) for b in all_visit_blocks]):.1f} blocks/patient")

    df_patterns = mine_patterns(
        seqs_flat=all_salient_flat,
        seqs_visit_blocks=all_visit_blocks,
        y=y,
        method="episode",
        min_support_frac=min_support_frac,
        episode_max_len=max_len,
        episode_topn=topn,
    )
    return df_patterns


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def main():
    parser = argparse.ArgumentParser(description="IG threshold sensitivity analysis")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--mimic_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="",
                        help="AToP output dir (default: run_dir/explain)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"])
    parser.add_argument("--thresholds", type=str, default="0.50,0.80,0.95",
                        help="Comma-separated IG mass thresholds")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top patterns to compare")
    parser.add_argument("--min_support_frac", type=float, default=0.01)
    parser.add_argument("--episode_max_len", type=int, default=4)
    parser.add_argument("--episode_topn", type=int, default=500)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "explain")
    thresholds = [float(t) for t in args.thresholds.split(",")]
    top_k = args.top_k

    # Load saved data
    dat_dir = os.path.join(out_dir, "data")
    ig_path = os.path.join(dat_dir, f"ig_{args.split}.csv")
    if not os.path.exists(ig_path):
        print(f"ERROR: {ig_path} not found. Run the full pipeline first.")
        sys.exit(1)

    print(f"Loading IG data from {ig_path}...")
    df_ig = pd.read_csv(ig_path)

    if "visit_idx" not in df_ig.columns:
        print("ERROR: df_ig missing visit_idx column. Re-run IG computation with updated code.")
        sys.exit(1)

    print(f"  {len(df_ig)} rows, {df_ig['patient_id'].nunique()} patients")
    print(f"  Thresholds: {thresholds}")
    print(f"  Top-K for Jaccard: {top_k}")
    print()

    # Mine at each threshold
    pattern_sets = {}
    for mass in thresholds:
        t0 = time.time()
        print(f"  Mining at {mass:.0%}...")
        df_pats = mine_at_threshold(
            df_ig, mass,
            args.min_support_frac, args.episode_max_len, args.episode_topn,
        )
        elapsed = time.time() - t0
        n = len(df_pats)
        print(f"    → {n} patterns mined in {elapsed:.1f}s")
        # Top K by support
        top = set(df_pats.nlargest(top_k, "n_admissions_present")["pattern"].values)
        pattern_sets[mass] = top

    # Compute pairwise Jaccard
    print(f"\n{'='*60}")
    print(f"Jaccard similarity of top-{top_k} patterns")
    print(f"{'='*60}")

    pairs = list(itertools.combinations(thresholds, 2))
    results = []
    for t1, t2 in pairs:
        j = jaccard(pattern_sets[t1], pattern_sets[t2])
        results.append({"threshold_1": t1, "threshold_2": t2, "jaccard": j})
        print(f"  {t1:.0%} vs {t2:.0%}: Jaccard = {j:.3f}")

    # Overall summary
    all_j = [r["jaccard"] for r in results]
    print(f"\n  Mean Jaccard: {np.mean(all_j):.3f}")
    print(f"  Min  Jaccard: {np.min(all_j):.3f}")

    # Save results
    csv_path = os.path.join(out_dir, "figures", "csv",
                            f"sensitivity_ig_threshold_{args.split}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # Also save the pattern lists
    for mass in thresholds:
        pats = sorted(pattern_sets[mass])
        tag = f"{int(mass*100)}"
        list_path = os.path.join(out_dir, "figures", "csv",
                                 f"sensitivity_patterns_{tag}pct_{args.split}.csv")
        pd.DataFrame({"pattern": pats}).to_csv(list_path, index=False)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
