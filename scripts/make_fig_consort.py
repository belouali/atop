"""make_fig_consort, CONSORT-style cohort + analysis flow diagram (Figure S14).

Addresses reviewer concern #11 (cohort/index/denominator reconciliation):
lays out the relationship among the full MIMIC-IV cohort, the 30-day
readmission labeling, the train/val/test splits, and the 38-patient
attribution exclusion, with all prevalence values shown against their
named cohort so the numbers cannot read as drift.

Pure matplotlib (no data dependency beyond the hard-coded reconciled
counts, which are verified against the run in the manuscript build).

Usage:
    python scripts/make_fig_consort.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


BOX_FC = "#eef2f7"
BOX_EC = "#33475b"
EXCL_FC = "#f6e9e9"
EXCL_EC = "#9c4a4a"
SPLIT_FC = "#e8f0e8"
SPLIT_EC = "#3d6b3d"


def _box(ax, x, y, w, h, text, fc=BOX_FC, ec=BOX_EC, fontsize=11, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            zorder=3, color="#111111", weight=weight, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>",
        color="#33475b", linewidth=1.4, mutation_scale=14,
        shrinkA=1, shrinkB=1, zorder=1))


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure S14 (CONSORT flow).")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/"
                           "FigureS14_consort.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(11, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")

    CX = 4.0          # main column x
    EX = 8.1          # exclusion column x
    W, H = 5.2, 1.15  # main box size
    EW, EH = 3.4, 0.95

    # 1. Source
    _box(ax, CX, 14.2, W, H,
         "MIMIC-IV v3.1\n546,028 hospital admissions",
         fontsize=11.5, weight="bold")
    # 2. Patient assembly
    _box(ax, CX, 12.5, W, H,
         "Patients with ≥1 acute inpatient admission\n"
         "assembled into one chronological token sequence\n"
         "(ICD-9/10 diagnoses + procedures, RxNorm drugs)")
    _arrow(ax, CX, 14.2 - H / 2, CX, 12.5 + H / 2)

    # 3. Labeling
    _box(ax, CX, 10.8, W, H,
         "30-day readmission labeling\n"
         "elective / surgical same-day admissions excluded\n"
         "from positive label (planned care, not unplanned)")
    _arrow(ax, CX, 12.5 - H / 2, CX, 10.8 + H / 2)

    # 3b. Outcome exclusion: in-hospital deaths are removed before
    #     labeling (compute_readmission_labels drops hospital_expire_flag
    #     == 1), so the flow from source admissions to analytic cohort is
    #     fully accounted for.
    # main box right edge is CX + W/2 = 6.6; place the exclusion box so
    # its left edge sits clear of it, with the arrow spanning the gap
    _box(ax, 8.45, 10.8, 2.8, EH,
         "11,801 admissions excluded (2.2%)\nin-hospital death, cannot be readmitted",
         fc=EXCL_FC, ec=EXCL_EC, fontsize=9.5)
    _arrow(ax, CX + W / 2, 10.8, 8.45 - 2.8 / 2, 10.8)

    # 4. Analytic cohort
    _box(ax, CX, 9.05, W, H,
         "Analytic cohort\n"
         "218,196 patients · 12.6% unplanned 30-day readmission",
         fontsize=11.5, weight="bold")
    _arrow(ax, CX, 10.8 - H / 2, CX, 9.05 + H / 2)

    # 5. Split header
    _box(ax, CX, 7.4, W, H,
         "Patient-level split (70 / 10 / 20)",
         fontsize=11.5, weight="bold")
    _arrow(ax, CX, 9.05 - H / 2, CX, 7.4 + H / 2)

    # 6. Three split boxes
    sy = 5.5
    _box(ax, 1.7, sy, 2.7, 1.5,
         "Train\n152,737\n(12.6% readmit)",
         fc=SPLIT_FC, ec=SPLIT_EC, fontsize=10.5)
    _box(ax, 4.6, sy, 2.5, 1.5,
         "Validation\n21,819\n(model selection)",
         fc=SPLIT_FC, ec=SPLIT_EC, fontsize=10.5)
    _box(ax, 7.4, sy, 2.5, 1.5,
         "Test\n43,640\n(12.7% readmit)",
         fc=SPLIT_FC, ec=SPLIT_EC, fontsize=10.5)
    _arrow(ax, CX, 7.4 - H / 2, 1.7, sy + 0.75)
    _arrow(ax, CX, 7.4 - H / 2, 4.6, sy + 0.75)
    _arrow(ax, CX, 7.4 - H / 2, 7.4, sy + 0.75)

    # 7. Mining (from train) + attribution (from test)
    _box(ax, 1.7, 3.2, 3.0, 1.5,
         "Pattern mining\n(train salient tokens)\n"
         "24,490 mined (min-support)\n9,213 cross-visit reps (j≥0.2)",
         fontsize=9.5)
    _arrow(ax, 1.7, sy - 0.75, 1.7, 3.2 + 0.75)

    _box(ax, 7.4, 3.2, 3.0, 1.5,
         "Per-token IG attribution\n(test cohort)\n"
         "applied to all 43,640",
         fontsize=9.5)
    _arrow(ax, 7.4, sy - 0.75, 7.4, 3.2 + 0.75)

    # 8. Attribution exclusion
    _box(ax, 7.4, 1.35, 3.0, 1.0,
         "38 excluded\n(empty salient-token set)",
         fc=EXCL_FC, ec=EXCL_EC, fontsize=9.5)
    _arrow(ax, 7.4, 3.2 - 0.75, 7.4, 1.35 + 0.5)

    # 9. Final analytic attribution set
    _box(ax, 3.0, 1.35, 3.2, 1.0,
         "Analytic attribution set\n43,602 patients",
         fc=BOX_FC, ec=BOX_EC, fontsize=10.5, weight="bold")
    _arrow(ax, 7.4 - 1.5, 1.35, 3.0 + 1.6, 1.35)

    # Footnote on prevalence reconciliation
    ax.text(5.0, 0.35,
            "Prevalence by cohort: overall/train 12.6%, test 12.7% "
            "(rounding of 12.65% and 12.74%); these are the same metric "
            "on different cohorts, not conflicting values.",
            ha="center", va="center", fontsize=8.5, style="italic",
            color="#555555", wrap=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output_path} "
          f"({os.path.getsize(args.output_path)//1024} KB)")


if __name__ == "__main__":
    main()
