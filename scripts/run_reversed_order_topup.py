"""run_reversed_order_topup, CPU-only reversed-order analysis for the
manuscript-driven whitelist (union of Figure 4B's top-30 cross-visit
patterns and Table 4's top-5-per-family displayed cross-visit set).

Mirrors the role of ``run_validation_topup.py`` but for the temporal-order
ablation (Figure S1). Reversed-order is pure-pandas IG aggregation from
the cached IG values, no model inference required, so CPU is fine and
the whole run is ~5-10 min for ~50 patterns.

Reads ``<run_dir>/5_analysis/reversed_order_whitelist.txt`` (built by
the build_validation_whitelist sibling helper or an inline one-shot),
loads the pre-mined patterns + sequences + IG cache, runs
``run_reversed_order_analysis`` with that whitelist, and overwrites
``<run_dir>/5_analysis/reversed_order_<split>.csv``.

Idempotent. Re-running on the same whitelist replays the same matching
+ aggregation logic, producing identical output.

Local usage (CPU):
    python scripts/run_reversed_order_topup.py \\
        --run_dir ../runs/mimiciv_readmission_30d \\
        --mimic_dir ../data/mimiciv/3.1/hosp
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.run_layout import RunLayout


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run reversed-order ablation on the manuscript-driven whitelist.")
    p.add_argument("--run_dir", required=True,
                   help="v2 run directory (with 0_inputs/, 1_model/, etc.)")
    p.add_argument("--mimic_dir", required=True,
                   help="MIMIC-IV hosp/ directory (needed by AToPExplainer.from_bundle).")
    p.add_argument("--device", default="cpu",
                   help="cpu / cuda / auto (default cpu; this step does no model inference).")
    p.add_argument("--split", default="test")
    p.add_argument("--whitelist_path", default=None,
                   help="Whitelist file (default: <run_dir>/5_analysis/"
                        "reversed_order_whitelist.txt).")

    # Mining knobs (mirror Notebook 1B so we hit the cached scored CSV)
    p.add_argument("--min_support_frac", type=float, default=0.001)
    p.add_argument("--scoring_min_support_frac", type=float, default=0.0)
    p.add_argument("--episode_max_len", type=int, default=4)
    p.add_argument("--jaccard_dedup", type=float, default=0.0)

    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    whitelist_path = args.whitelist_path or os.path.join(
        layout.analysis_dir, "reversed_order_whitelist.txt")
    rev_csv = os.path.join(layout.analysis_dir, f"reversed_order_{args.split}.csv")

    if not os.path.exists(whitelist_path):
        sys.exit(f"  [ERROR] Missing {whitelist_path}")
    with open(whitelist_path) as f:
        whitelist = [l.strip() for l in f if l.strip()]
    print(f"[1] Whitelist: {len(whitelist)} patterns "
          f"({os.path.basename(whitelist_path)})")

    if os.path.exists(rev_csv):
        print(f"[2] Existing reversed CSV will be overwritten: "
              f"{os.path.basename(rev_csv)}")

    print("\n[3] Loading model bundle (CPU, no inference)...")
    from atop.explainer import AToPExplainer
    explainer = AToPExplainer.from_bundle(
        args.run_dir, mimic_dir=args.mimic_dir, device=args.device)

    # Apply mining-knob overrides to hit the scored cache. episode_min_steps
    # is a post-load filter; default 2 drops single-block patterns but every
    # whitelist entry here is cross-visit (≥2 blocks), so default is fine.
    explainer.config.min_support_frac = args.min_support_frac
    explainer.config.scoring_min_support_frac = args.scoring_min_support_frac
    explainer.config.episode_max_len = args.episode_max_len
    explainer.config.jaccard_dedup = args.jaccard_dedup
    explainer.config.episode_min_steps = 1   # keep everything visible

    explainer.compute_attributions()
    explainer.mine_patterns()

    visible = sum(1 for p in whitelist
                   if p in set(explainer.df_patterns["pattern"].tolist()))
    print(f"\n  [sanity] {visible}/{len(whitelist)} whitelist patterns "
          f"present in df_patterns")

    print(f"\n[4] Running reversed-order analysis on whitelist...")
    from atop.explain.validation import run_reversed_order_analysis

    df_rev = run_reversed_order_analysis(
        explainer.df_patterns,
        explainer.df_test,
        explainer.df_ig_test,
        explainer._mining_method,
        icd_titles=explainer.icd_titles,
        top_k=len(whitelist),
        pattern_whitelist=set(whitelist),
    )

    if df_rev is None or df_rev.empty:
        sys.exit("  [ERROR] reversed-order analysis returned no rows.")

    os.makedirs(layout.analysis_dir, exist_ok=True)
    df_rev.to_csv(rev_csv, index=False)
    print(f"\n[5] Wrote {len(df_rev)} rows to {rev_csv}")

    final_set = set(df_rev["pattern"].tolist())
    missing = [p for p in whitelist if p not in final_set]
    if missing:
        print(f"\n  [WARN] {len(missing)} whitelist patterns missing from results "
              f"(likely 0 forward+reversed matches):")
        for p in missing[:5]:
            print(f"    {p}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    else:
        print(f"\n  All {len(whitelist)} whitelist patterns analyzed.")


if __name__ == "__main__":
    main()
