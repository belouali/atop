"""make_fig_s2, Supplementary Figure S2: candidate sequences vs support threshold.

Answers the reviewer question "why min_support_frac = 0.001?" using the
mining cache already on disk. Plots the cumulative count of candidate
patterns meeting each support level, stacked by pattern structure, with
cross-hair reference lines + bold central callout at the locked threshold.

Usage:
    python scripts/make_fig_s2.py --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.run_layout import RunLayout


BUCKETS = ["Three+ block cross-visit",
           "Two-block cross-visit",
           "Single-visit compound",
           "Single token"]
BUCKET_COLORS = {
    "Single token":             "#999999",
    "Single-visit compound":    "#55a1b3",
    "Two-block cross-visit":    "#c0392b",
    "Three+ block cross-visit": "#5b1d12",
}


def _bucket(row) -> str:
    n_tok, n_st = int(row["n_tokens"]), int(row["n_steps"])
    if n_tok == 1:
        return "Single token"
    if n_st == 1:
        return "Single-visit compound"
    if n_st == 2:
        return "Two-block cross-visit"
    return "Three+ block cross-visit"


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure S2 (support threshold curve)")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--n_train", type=int, default=152737,
                   help="Training cohort size (for the 0.1% annotation)")
    p.add_argument("--threshold", type=int, default=153,
                   help="Min-support threshold to annotate (default = ceil(0.1%% of n_train))")
    p.add_argument("--jaccard_dedup_count", type=int, default=None,
                   help="Pattern count after Jaccard dedup. If omitted, computed "
                        "automatically from the cached Jaccard reps at j=0.2 on the "
                        "test split (canonical number; matches Table S1).")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/FigureS2_support_threshold.png")
    p.add_argument("--analysis-dir", default=None,
                   help="Directory to write backing CSV (default: <run_dir>/5_analysis)")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    csv_dir = args.analysis_dir or layout.analysis_dir

    # ── Resolve the Jaccard-dedup count (auto-compute if not provided) ──
    if args.jaccard_dedup_count is None:
        from atop.explain.jaccard import get_jaccard_representatives
        rep_set = get_jaccard_representatives(layout.analysis_dir, "test", 0.2)
        if rep_set is None:
            sys.exit("  [ERROR] No cached Jaccard reps at j=0.2 (test). "
                      "Re-run run_atop with --steps validate first, "
                      "or pass --jaccard_dedup_count explicitly.")
        args.jaccard_dedup_count = len(rep_set)
        print(f"  [Jaccard] auto-resolved: {args.jaccard_dedup_count:,} "
               f"representatives at j=0.2")

    # ── Load latest mining cache ─────────────────────────────────────
    pat_files = sorted(
        glob.glob(os.path.join(layout.mining_dir, "all_scored_*.csv")),
        key=os.path.getmtime,
    )
    if not pat_files:
        sys.exit(f"  [ERROR] No mining cache CSV under {layout.mining_dir}/")
    df = pd.read_csv(pat_files[-1])
    print(f"[1] Loaded {len(df):,} patterns from {os.path.basename(pat_files[-1])}")

    df["bucket"] = df.apply(_bucket, axis=1)

    # Pattern-count curve by support threshold (log-spaced grid)
    xs = np.unique(np.round(np.logspace(
        0, np.log10(df["n_admissions_present"].max()), 120
    ))).astype(int)
    counts = {b: [] for b in BUCKETS}
    totals = []
    for x in xs:
        sub = df[df["n_admissions_present"] >= x]
        totals.append(len(sub))
        for b in BUCKETS:
            counts[b].append((sub["bucket"] == b).sum())
    print(f"[2] Rendering curve across {len(xs)} support levels")

    # ── Render ───────────────────────────────────────────────────────
    # Wider-than-tall aspect ratio gives the big highlight annotation
    # central breathing room (style cue from comparable mining-curve figures).
    fig, ax = plt.subplots(figsize=(12, 6.8))

    # Stacked area, absolute counts, RESTRICTED to x >= threshold.
    # The mining cache only contains patterns with support >= 153, so
    # for any x < 153 the count is mechanically 24,490 (= the same
    # mined pool repeatedly). Showing that region as colored bars
    # would falsely imply we measured what's there. instead we mask
    # it and overlay an explicit "not mined" hatch.
    mask = xs >= args.threshold
    xs_visible = xs[mask]
    cum = np.zeros(len(xs_visible))
    for b in BUCKETS:
        y = np.array(counts[b])[mask]
        ax.fill_between(xs_visible, cum, cum + y, color=BUCKET_COLORS[b],
                         alpha=0.90, label=b,
                         edgecolor=BUCKET_COLORS[b], linewidth=1.0)
        cum = cum + y
    totals_visible = np.array(totals)[mask]
    ax.plot(xs_visible, totals_visible, color="black", linewidth=2.4,
             label="All frequent sequences")

    total_at_thr = int(df[df["n_admissions_present"] >= args.threshold].shape[0])

    # ── Hatched "not mined" backdrop over x < threshold ──
    # The mining cache contains no patterns with support < 153, so the
    # composition at lower support is genuinely unknown. The hatched
    # zone visually says "we have no data here" (vs the unmaskéd
    # alternative which would have showed a flat plateau at 24,490
    # that is mechanically the same 24,490 mined patterns repeatedly,
    # not a true observation of lower-support patterns).
    ax.fill_between([1, args.threshold], 1, max(totals) * 1.1,
                     color="#e8e8e8", alpha=0.65,
                     hatch="///", edgecolor="#aaaaaa", linewidth=0.0,
                     zorder=1)

    # ── Cross-hair reference at the locked threshold ──
    HL = "#2c3e50"   # cross-hair color (deep slate)
    ax.axvline(args.threshold, color=HL, linestyle="--", linewidth=2.2,
                alpha=0.9, zorder=2)
    ax.axhline(total_at_thr, color=HL, linestyle="--", linewidth=2.2,
                alpha=0.9, zorder=2)
    # Label the hatched zone (placed inside it, centered vertically).
    ax.text(np.sqrt(1 * args.threshold), np.sqrt(max(totals) * 0.5),
             "Not mined\n(support < threshold)",
             ha="center", va="center", fontsize=10.5,
             color="#555555", style="italic", fontweight="bold",
             zorder=3)

    # ── BIG bold central callout, colored highlight number ──
    # Positioned in the MIDDLE-RIGHT empty region (right of the cross-hair
    # intersection, where the cumulative curve has dropped to near zero
    # on the linear-linear axes). Headline number rendered in a larger
    # orange weight via a separate text() call layered above the box.
    HIGHLIGHT = "#e67e22"   # orange highlight for the headline number
    # Log axes: use multiplicative offsets for callout positioning so
    # the box lands in the empty middle-right area.
    text_x = args.threshold * 10        # ~1500 on the log x-axis
    text_y = total_at_thr * 0.05        # ~1200 on log y
    callout = ax.annotate(
        f"\n  candidate sequences mined at\n"
        f"  min support {args.threshold} "
        f"(0.1% of {args.n_train:,} training patients)\n\n"
        f"  $\\Rightarrow$ {args.jaccard_dedup_count:,} representatives "
        f"after Jaccard dedup (j $\\geq$ 0.2)",
        xy=(args.threshold, total_at_thr),
        xytext=(text_x, text_y),
        fontsize=12.5, ha="left", va="center", color=HL,
        arrowprops=dict(arrowstyle="-", color=HL, lw=0.7,
                         linestyle=":", alpha=0.6,
                         connectionstyle="arc3,rad=-0.2"),
        bbox=dict(boxstyle="round,pad=0.7", fc="#fff5eb", ec=HL, lw=1.5),
    )
    # BIG orange headline number, layered above the box.
    ax.text(text_x * 1.02, text_y * 4.5, f"{total_at_thr:,}",
             fontsize=28, fontweight="bold", color=HIGHLIGHT,
             ha="left", va="center", zorder=5)

    # Log-log axes: the count data spans 1 to ~25K and the support
    # spans 1 to ~30K, both orders of magnitude wide.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, xs.max())
    ax.set_ylim(1, max(totals) * 1.1)
    ax.set_xlabel(
        "Minimum support  (training patients carrying the pattern)",
        fontsize=13, fontweight="bold", labelpad=10,
    )
    ax.set_ylabel(
        "Number of candidate sequences  (count with support ≥ x)",
        fontsize=13, fontweight="bold", labelpad=10,
    )
    ax.tick_params(axis="both", labelsize=11)
    # No in-image title; the DOCX caption supplies it.
    ax.grid(True, which="both", color="#e6e6e6", linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Legend OUTSIDE the right edge of the plot to avoid overlapping
    # the colored bars / cross-hair / callout.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
               fontsize=10.5, framealpha=0.95,
               title="Pattern structure", title_fontsize=11,
               borderaxespad=0)

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(args.output_path) // 1024
    print(f"[3] Saved figure: {args.output_path} ({size_kb} KB)")

    os.makedirs(csv_dir, exist_ok=True)
    pd.DataFrame({
        "min_support": xs,
        "total_patterns": totals,
        **{b: counts[b] for b in BUCKETS},
    }).to_csv(os.path.join(csv_dir, "fig_s2_support_threshold.csv"), index=False)
    print(f"    csv: {os.path.join(csv_dir, 'fig_s2_support_threshold.csv')}")


if __name__ == "__main__":
    main()
