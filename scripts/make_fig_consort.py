"""make_fig_consort, cohort and analysis flow diagram (Figure S1).

Lays out the relationship among the full MIMIC-IV admissions, the 30-day
readmission labeling, the train/validation/test split, and the 38-patient
attribution exclusion, with every prevalence shown against its named
cohort so the numbers cannot read as drift.

Style follows the JAMA-type flow diagram used for the DepAnx paper: square
boxes with a light grey fill and a thin dark-grey edge, a bold lead line
(the count) over a regular description, thin grey arrows, exclusions in a
right-hand column of the same grey. No colour coding.

Pure matplotlib; the counts are hard-coded from the reconciled run.

Usage:
    python scripts/make_fig_consort.py [--output-path PATH] [--dpi 300]
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
})

FILL, EDGE = "#E9EFF3", "#555555"      # DepAnx Figure 1 constants
LEAD_PT, TEXT_PT = 8.2, 7.8
LINE_H = 0.235                          # axis units per text line (8 pt at this scale)


def _box(ax, x, top, w, lead, text="", pad=0.14):
    """Square box anchored at its top-left corner; returns its bottom y."""
    n_lines = text.count("\n") + 1 if text else 0
    h = pad + LINE_H * 1.25 + (LINE_H * n_lines + 0.06 if text else 0) + pad
    ax.add_patch(FancyBboxPatch((x, top - h), w, h, boxstyle="square,pad=0",
                                fc=FILL, ec=EDGE, lw=0.6, zorder=2))
    ax.text(x + 0.14, top - pad, lead, va="top", ha="left",
            fontsize=LEAD_PT, fontweight="bold", zorder=3)
    if text:
        ax.text(x + 0.14, top - pad - LINE_H * 1.25 - 0.04, text, va="top",
                ha="left", fontsize=TEXT_PT, linespacing=1.4, zorder=3)
    return top - h


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=EDGE,
                                shrinkA=0, shrinkB=0), zorder=1)


def _line(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=EDGE, lw=0.7, zorder=1,
            solid_capstyle="butt")


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure S1 (cohort flow).")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/"
                           "FigureS14_consort.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(9.1, 9.4))   # 0.72 in per axis unit
    ax.axis("off")
    ax.set_xlim(0, 12.7)
    ax.set_ylim(0, 12.0)

    MX, MW = 2.3, 5.6        # main column, centred over the three split columns
    EX, EW = 8.15, 3.7       # exclusion column
    GAP = 0.55               # vertical gap between boxes
    cx = MX + MW / 2

    # 1. Source admissions
    top = 11.9
    b1 = _box(ax, MX, top, MW, "546,028 Hospital admissions",
              "MIMIC-IV v3.1; patients with ≥1 acute inpatient admission")
    # 2. Sequence assembly
    t2 = b1 - GAP
    b2 = _box(ax, MX, t2, MW, "One chronological token sequence per patient",
              "ICD-9/10 diagnoses and procedures, RxNorm drugs,\n"
              "ordered by admission")
    _arrow(ax, cx, b1, cx, t2)
    # 3. Labeling, with the death exclusion branching off the connector
    t3 = b2 - GAP
    b3 = _box(ax, MX, t3, MW, "30-day unplanned readmission labeling",
              "Elective and surgical same-day admissions not counted\n"
              "as unplanned readmissions")
    _arrow(ax, cx, b2, cx, t3)
    # 4. Analytic cohort; exclusion sits on the connector between 3 and 4
    t4 = b3 - GAP - 0.55
    b4 = _box(ax, MX, t4, MW, "218,196 Patients in the analytic cohort",
              "12.6% unplanned 30-day readmission")
    _arrow(ax, cx, b3, cx, t4)
    ymid = (b3 + t4) / 2
    ex_top = ymid + 0.42
    _box(ax, EX, ex_top, EW, "11,801 Admissions excluded (2.2%)",
         "In-hospital death;\ncannot be readmitted")
    _arrow(ax, cx, ymid, EX, ymid)

    # 5. Patient-level split into three columns
    t5 = b4 - GAP - 0.25
    SW = 3.1
    xs = [0.3, 3.55, 6.8]
    leads = ["152,737 Training patients", "21,819 Validation patients",
             "43,640 Test patients"]
    texts = ["70% split; 12.6% readmitted",
             "10% split; model selection",
             "20% split; 12.7% readmitted"]
    bottoms = []
    for x, lead, text in zip(xs, leads, texts):
        bottoms.append(_box(ax, x, t5, SW, lead, text))
    # connector: down from cohort, horizontal bar, three drops
    ybar = (b4 + t5) / 2
    _line(ax, cx, b4, cx, ybar)
    _line(ax, xs[0] + SW / 2, ybar, xs[2] + SW / 2, ybar)
    for x in xs:
        _arrow(ax, x + SW / 2, ybar, x + SW / 2, t5)
    ax.text(cx + 0.12, ybar + 0.07, "Patient-level split, 70 / 10 / 20",
            fontsize=TEXT_PT - 0.6, ha="left", va="bottom", color=EDGE)

    # 6. Mining (training) and attribution (test)
    t6 = min(bottoms) - GAP
    b6a = _box(ax, xs[0], t6, SW, "Pattern mining",
               "On training-cohort salient tokens\n"
               "70,278 episodes mined (pooled)\n"
               "24,490 at whole-cohort support\n"
               "9,213 cross-visit representatives\n(Jaccard ≥ 0.2)")
    _arrow(ax, xs[0] + SW / 2, bottoms[0], xs[0] + SW / 2, t6)
    b6c = _box(ax, xs[2], t6, SW, "Per-token IG attribution",
               "Applied to all 43,640 test patients")
    _arrow(ax, xs[2] + SW / 2, bottoms[2], xs[2] + SW / 2, t6)

    # 7. Attribution exclusion (right of the test column, on the connector)
    #    and the final analytic set spanning the two right-hand columns
    tx = xs[2] + SW / 2
    EX2, EW2 = xs[2] + SW + 0.25, 2.4
    t7 = b6c - GAP - 0.75
    ymid2 = (b6c + t7) / 2
    _box(ax, EX2, ymid2 + 0.42, EW2, "38 Excluded",
         "Empty salient-token set")
    _arrow(ax, tx, ymid2, EX2, ymid2)
    FW = 4.7   # just wide enough for the lead line; centred under the test column
    b7 = _box(ax, tx - FW / 2, t7, FW,
         "43,602 Patients in the analytic attribution set",
         "Carrier-conditional token summaries\nand pattern-level analyses")
    _arrow(ax, tx, b6c, tx, t7)
    ax.set_ylim(b7 - 0.25, 12.0)

    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output_path} "
          f"({os.path.getsize(args.output_path)//1024} KB)")


if __name__ == "__main__":
    main()
