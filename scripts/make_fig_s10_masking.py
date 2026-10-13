"""make_fig_s10_masking, Supplementary Figure S10: per-pattern masking effects.

Histogram of the masking effect for every perturbed pattern (n = 266),
where each pattern's effect is delta_y = y_original - y_masked averaged over
its test-cohort carriers (validation_test.csv). Bars are colored by the
pattern's attribution direction: the sign recorded at validation time
(ig_direction), or, where that was not stored (0), the sign of the
test-cohort carrier-averaged signed IG in pathway_importance_test*.csv
or in the manuscript Tables 3-4 (ig_value); when
validation_pattern_directions.csv exists (all patterns scored with the
unified rule) it takes precedence. Patterns with no direction are grey.

Replaces the earlier mask-vs-shuffle version (scripts/parked/make_fig_s6.py); the shuffle
controls are parked as backup and no longer reported.

Usage:
    python scripts/make_fig_s10_masking.py --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RISK = "#c44e52"
PROT = "#4c72b0"
NONE = "#bbbbbb"


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure S10 (masking effects)")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--tables-dir", default="../manuscript/jamia_submission/tables")
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/FigureS10_masking_distribution.png")
    args = p.parse_args()

    adir = os.path.join(args.run_dir, "5_analysis")
    val = pd.read_csv(os.path.join(adir, "validation_test.csv"))
    sig = {}
    for f in ["pathway_importance_test_all.csv", "pathway_importance_test.csv",
              "pathway_importance_test_xvisit.csv"]:
        path = os.path.join(adir, f)
        if os.path.exists(path):
            for _, r in pd.read_csv(path).iterrows():
                sig.setdefault(r["pattern"], r["ig_signed_mean"])
    for f in ["Table3_full_pool_clusters.csv", "Table4_cross_visit_clusters.csv"]:
        path = os.path.join(args.tables_dir, f)
        if os.path.exists(path):
            for _, r in pd.read_csv(path).iterrows():
                sig.setdefault(r["pattern"], r["ig_value"])
    # Preferred source: test-cohort carrier-averaged signed IG under the
    # single scoring rule (atop/explain/scoring.py), written by
    # scripts/score_validation_patterns.py for every validated pattern.
    uni = os.path.join(adir, "validation_pattern_directions.csv")
    if os.path.exists(uni):
        u = pd.read_csv(uni)
        for _, r in u.dropna(subset=["sigma_ig_test"]).iterrows():
            sig[r["pattern"]] = r["sigma_ig_test"]
        direction = val["pattern"].map(sig)
    else:
        direction = val["ig_direction"].where(val["ig_direction"] != 0,
                                              val["pattern"].map(sig))
    val["dir"] = np.select([direction > 0, direction < 0], ["risk", "protective"], "none")
    d = val["mean_delta_mask"]
    agree = ((val["dir"] == "risk") & (d > 0)) | ((val["dir"] == "protective") & (d < 0))
    n_risk = int((val["dir"] == "risk").sum())
    n_prot = int((val["dir"] == "protective").sum())
    n_none = int((val["dir"] == "none").sum())
    print(f"n={len(val)} risk={n_risk} protective={n_prot} no-direction={n_none}; "
          f"agreement {int(agree.sum())}/{n_risk + n_prot}")

    bins = np.linspace(d.min(), d.max(), 41)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.hist([d[val["dir"] == "protective"], d[val["dir"] == "none"], d[val["dir"] == "risk"]],
            bins=bins, stacked=True, color=[PROT, NONE, RISK], edgecolor="white", linewidth=0.4,
            label=[f"Protective attribution (n = {n_prot})",
                   f"Direction not recorded (n = {n_none})",
                   f"Risk-supporting attribution (n = {n_risk})"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Masking effect Δŷ = ŷ_original − ŷ_masked "
                  "(mean over carriers)", fontsize=10)
    ax.set_ylabel("Number of patterns", fontsize=10)
    ax.set_title(f"Per-pattern masking effects, n = {len(val)} patterns", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.99, 0.62, f"Direction agrees with attribution:\n{int(agree.sum())} of {n_risk + n_prot} "
            "patterns with a recorded direction", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
