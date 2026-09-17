"""make_fig_s6, Supplementary Figure S6: per-pattern Δŷ distribution.

Two panels: overlaid histograms + side-by-side boxplots of Δŷ across
patterns, comparing mask (AToP) vs shuffle-within-visit and shuffle-across-
visits baselines. Mann-Whitney U tests annotate the boxplot panel.

Usage:
    python scripts/make_fig_s6.py --run_dir ../runs/full_CPD_v1
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.run_layout import RunLayout


RISK      = "#c44e52"    # canonical risk red (matches Figs S1, S3, S5)
SHUFFLE_W = "#a0b8d0"    # light blue, distinct from data palette
SHUFFLE_V = "#cccccc"    # neutral light gray for the weakest baseline


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure S6 (Δŷ distribution)")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/FigureS6_validation_distribution.png")
    p.add_argument("--analysis-dir", default=None,
                   help="Directory holding validation_*.csv (default: <run_dir>/5_analysis)")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    csv_dir = args.analysis_dir or layout.analysis_dir

    val_csv = layout.validation_csv(args.split) if csv_dir == layout.analysis_dir else \
              os.path.join(csv_dir, f"validation_{args.split}.csv")
    if not os.path.exists(val_csv):
        sys.exit(f"  [ERROR] Missing {val_csv}")
    val = pd.read_csv(val_csv)
    print(f"[1] Loaded {len(val):,} validated patterns")

    need = ["mean_delta_mask", "mean_delta_shuffle_within", "mean_delta_shuffle_visits"]
    val = val.dropna(subset=need).copy()
    print(f"    {len(val):,} patterns with full shuffle baselines")

    # One-sided MWU on absolute Δŷ. the claim is "|mask| > |shuffle|"
    # not a two-sided distribution shift.
    u1, p1 = mannwhitneyu(val["mean_delta_mask"].abs(),
                           val["mean_delta_shuffle_within"].abs(),
                           alternative="greater")
    u2, p2 = mannwhitneyu(val["mean_delta_mask"].abs(),
                           val["mean_delta_shuffle_visits"].abs(),
                           alternative="greater")
    print(f"[2] MWU mask vs shuffle-within: p = {_fmt_p(p1)}")
    print(f"    MWU mask vs shuffle-visits: p = {_fmt_p(p2)}")

    # ── Render ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2),
                                    gridspec_kw={"wspace": 0.25})

    bins = np.linspace(val[need].values.min() - 0.01, val[need].values.max() + 0.01, 40)
    for col, color, label in [
        ("mean_delta_shuffle_visits", SHUFFLE_V, "Shuffle across visits"),
        ("mean_delta_shuffle_within", SHUFFLE_W, "Shuffle within visit"),
        ("mean_delta_mask",           RISK,      "Mask (AToP)"),
    ]:
        ax1.hist(val[col], bins=bins, alpha=0.75, color=color, edgecolor="white",
                 linewidth=0.4, label=label)
    ax1.axvline(0, color="black", linewidth=0.6)
    ax1.set_xlabel("\u0394\u0177 = \u0177_original \u2212 \u0177_perturbed  "
                    "(predicted-readmission-probability change)",
                    fontsize=12, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Number of patterns",
                    fontsize=12, fontweight="bold", labelpad=8)
    ax1.set_title(
        "A \u00B7 Per-pattern \u0394\u0177 distribution",
        fontsize=12, loc="left", pad=10,
    )
    ax1.tick_params(labelsize=10)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax1.grid(True, axis="y", color="#eeeeee", linewidth=0.5)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    box_data = [val["mean_delta_shuffle_visits"].values,
                val["mean_delta_shuffle_within"].values,
                val["mean_delta_mask"].values]
    labels = ["Shuffle\nacross visits", "Shuffle\nwithin visit", "Mask\n(AToP)"]
    colors = [SHUFFLE_V, SHUFFLE_W, RISK]
    bp = ax2.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=7))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")

    y_top = max(val[need].max()) + 0.02
    ax2.plot([2, 2, 3, 3], [y_top, y_top + 0.008, y_top + 0.008, y_top],
             color="black", linewidth=0.8)
    ax2.text(2.5, y_top + 0.012, f"p {_fmt_p(p1)}",
             ha="center", va="bottom", fontsize=9)
    ax2.plot([1, 1, 3, 3], [y_top + 0.04, y_top + 0.048, y_top + 0.048, y_top + 0.04],
             color="black", linewidth=0.8)
    ax2.text(2.0, y_top + 0.052, f"p {_fmt_p(p2)}",
             ha="center", va="bottom", fontsize=9)

    ax2.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax2.set_ylabel("\u0394\u0177", fontsize=12, fontweight="bold", labelpad=8)
    ax2.set_title(
        f"B \u00B7 Box-and-whisker \u2014 n = {len(val):,} patterns",
        fontsize=12, loc="left", pad=10,
    )
    ax2.tick_params(labelsize=10)
    ax2.grid(True, axis="y", color="#eeeeee", linewidth=0.5)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    # Effect-size summary box, right side of Panel B
    mask_abs   = val["mean_delta_mask"].abs().mean()
    within_abs = val["mean_delta_shuffle_within"].abs().mean()
    visits_abs = val["mean_delta_shuffle_visits"].abs().mean()
    ax2.text(
        0.97, 0.04,
        f"Mean |\u0394\u0177|:\n"
        f"  Mask           {mask_abs:.4f}\n"
        f"  Shuffle within {within_abs:.4f}  "
        f"({within_abs/mask_abs*100:.0f}% of mask)\n"
        f"  Shuffle visits {visits_abs:.4f}  "
        f"({visits_abs/mask_abs*100:.0f}% of mask)",
        transform=ax2.transAxes, ha="right", va="bottom",
        fontsize=9, family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                   ec="#aaaaaa", lw=0.6),
    )

    # No in-image suptitle; the DOCX caption supplies the figure heading.
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[3] Saved figure: {args.output_path} ({os.path.getsize(args.output_path)//1024} KB)")

    print(f"\nStats to cite:")
    print(f"  mean \u0394\u0177 (mask):            {val['mean_delta_mask'].mean():+.4f}")
    print(f"  mean \u0394\u0177 (shuffle-within):  {val['mean_delta_shuffle_within'].mean():+.4f}")
    print(f"  mean \u0394\u0177 (shuffle-visits):  {val['mean_delta_shuffle_visits'].mean():+.4f}")

    out_csv = os.path.join(csv_dir, "fig_s6_validation_distribution.csv")
    val[["pattern", "pattern_readable", *need]].to_csv(out_csv, index=False)
    print(f"    csv: {out_csv}")


if __name__ == "__main__":
    main()
