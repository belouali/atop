"""make_fig5, render Figure 5 (perturbation-validation, 2-panel dumbbell).

Mirrors make_fig4.py's style:
  • 2 columns side-by-side, titles ABOVE each column (no empty vertical space)
  • Bold panel letter + colon + plain-English description in the header
  • Math notation Δŷ lives on the x-axis label, above the
    "← protective    risk →" direction arrows
  • Inline 3-decimal Δŷ annotations beside each dumbbell
  • 40/60 bars-to-text split per panel so pattern labels can grow
  • Shared xlim across panels (both report Δŷ, comparability matters)
  • No top legend (red/blue + axis arrows + inline values are self-explanatory)
  • Sign-convention rule lives in the DOCX caption, not on the figure

Panel order matches make_fig4.py:
  • **Panel A (left)**, Full pool (Fig 3 Panel C source):
      no shape minimums, Jaccard-deduplicated. The widest pool, dominated
      by single-visit multi-token co-occurrences with largest |ΣIG|.
  • **Panel B (right)**, Cross-visit pool (Fig 3 Panel D source):
      ≥2 visits AND ≥2 tokens, Jaccard-deduplicated. The temporal-trajectory
      AToP claim.

Data sources:
  • Panel A: panel_c_<split>_j<jt>_short.csv  (full pool)
  • Panel B: panel_d_<split>_j<jt>_short.csv  (cross-visit pool)
  • Validation deltas: 5_analysis/validation_<split>.csv

Both panel CSVs are written by make_fig3.py.

Usage:
    python scripts/make_fig5.py --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.explain.pattern_labels import PatternLabelResolver
from atop.run_layout import RunLayout


# ── Defaults ─────────────────────────────────────────────────────────────
JACCARD_THRESHOLD = 0.2       # matches Fig 3
N_SHOW = 30                   # top-N per panel; matches Fig 3 Panel C/D
FONTSIZE = 16                 # matches make_fig4.py (40/60 split gives room)
DPI = 300

# ── Palette ──────────────────────────────────────────────────────────────
# Convention Δŷ = ŷ_original − ŷ_masked (the standard occlusion / ERASER
# importance convention: importance = drop in output when the feature is
# removed). This matches Tables 3/4, Methods, and the faithfulness
# analysis, and aligns Figure 5's colors and orientation with Figures 3
# and 4:
#   • risk-supporting pattern → masking LOWERS predicted risk → Δŷ > 0 →
#     red, dumbbell points RIGHT.
#   • protective pattern → masking RAISES predicted risk → Δŷ < 0 → blue,
#     dumbbell points LEFT.
COLOR_RISK = "#c44e52"        # red , Δŷ > 0 (risk-supporting; masking lowered risk)
COLOR_PROT = "#4c72b0"        # blue, Δŷ < 0 (protective; masking raised risk)
COLOR_BASE = "#b8b8b8"        # faint gray square at 0 (dumbbell baseline)


# Label abbreviations applied to every pattern label before wrapping.
# Pattern labels are LONG; a few targeted abbreviations keep them
# on one line. Mirrors make_fig4.py. First match wins.
LABEL_ABBREV = [
    ("metastases", "mets"),
    ("Bone mets", "Bone mets"),
    # "Unemployment NOS" → "Unemployment": NOS is redundant noise here
    # (only one Unemployment code in the vocab).
    ("Unemployment NOS", "Unemployment"),
    # "Anxiety disorder" → "Anxiety": "disorder" is redundant after
    # "Anxiety". Applied before any NOS-bearing label, so this also
    # rewrites "Anxiety disorder NOS" → "Anxiety NOS".
    ("Anxiety disorder", "Anxiety"),
]


def _abbreviate_label(lab: str) -> str:
    for src, dst in LABEL_ABBREV:
        if src in lab:
            lab = lab.replace(src, dst)
    return lab


def _wrap_long_label(lab: str, max_chars: int = 70) -> str:
    """Hard-wrap a label longer than max_chars at the last space before the
    limit. Pattern labels can be 80+ chars when they include multi-token
    co-occurrences inside curly braces."""
    if len(lab) <= max_chars:
        return lab
    bp = lab.rfind(" ", 0, max_chars)
    if bp > 20:
        return lab[:bp].rstrip() + "\n  " + lab[bp:].lstrip()
    return lab


def _load_panel_csv(panels_dir: str, panel_letter: str, split: str,
                     jt_pct: int) -> pd.DataFrame | None:
    """Load ``panel_<letter>_<split>_j<pct>_short.csv`` if it exists."""
    path = os.path.join(panels_dir,
                         f"panel_{panel_letter}_{split}_j{jt_pct}_short.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _select_topn_with_validation(panel_df: pd.DataFrame, val: pd.DataFrame,
                                   n: int) -> pd.DataFrame:
    """Take top-N patterns from the Fig 3 panel, join with validation deltas,
    and sort by |Δŷ| so the largest-effect patterns plot at the top.

    The validation CSV already stores Δŷ = ŷ_original − ŷ_masked (the
    standard occlusion/ERASER importance convention: risk-supporting
    patterns have Δŷ > 0 because masking them lowers predicted risk).
    No sign flip is applied, so Figure 5 matches Tables 3/4, the Methods
    definition, the faithfulness analysis, and the red=risk / right=positive
    orientation of Figures 3 and 4."""
    sel = panel_df.merge(
        val[["pattern", "mean_delta_mask", "n_matched_test"]],
        on="pattern", how="left")
    missing = sel["mean_delta_mask"].isna().sum()
    if missing:
        print(f"    [WARN] {missing}/{len(sel)} patterns lack validation, dropped.")
        sel = sel[sel["mean_delta_mask"].notna()].copy()
    if len(sel) > n:
        sel = sel.head(n).copy()
    # Sort by |Δŷ| ASCENDING so the biggest absolute masking effect ends
    # up at the END of the list, plotting at y=N-1 (TOP of the panel).
    # Matches the Figs 3/4 convention of ranking by absolute magnitude.
    sel = sel.assign(_abs=sel["mean_delta_mask"].abs()) \
             .sort_values("_abs", ascending=True) \
             .drop(columns="_abs")
    return sel.reset_index(drop=True)


def _draw_dumbbell_panel(ax, sel: pd.DataFrame, resolver: PatternLabelResolver,
                          fontsize: int, x_lim: tuple[float, float],
                          show_n: bool = False) -> None:
    """Draw one dumbbell panel into ``ax``. Caller controls layout, title,
    and x-axis label (mirrors signed_ig_bars's split of concerns)."""
    # Build labels, abbreviate, wrap, optionally append carrier count.
    # Value annotations are rendered separately as inline ax.text past
    # the dot tip (matching Fig 4 / signed_ig_bars convention).
    labels = []
    for _, r in sel.iterrows():
        lab = resolver.format_pattern(r["pattern"], short_names=True)
        lab = _abbreviate_label(lab)
        lab = _wrap_long_label(lab)
        if show_n:
            lab = f"{lab}  (n={int(r['n_matched_test'])})"
        labels.append(lab)
    deltas = list(sel["mean_delta_mask"])
    N = len(deltas)
    y = list(range(N))

    # Faint reference squares at 0, the "dumbbell" baseline. Larger
    # than before so they read as a clear visual anchor at the same
    # weight as the colored endpoint dots.
    ax.scatter([0] * N, y, marker="s", color=COLOR_BASE, s=55,
               zorder=2, linewidths=0)

    # Dumbbell arrows + circles + inline value annotations PAST the
    # dot tip. Annotation font and color match Fig 4 / signed_ig_bars
    # EXACTLY: fontsize = base - 2, color = "#222222" (single dark
    # gray, not direction-colored). Direction is already conveyed by
    # the dumbbell arrow color.
    fs_annot = fontsize - 2
    xrange = x_lim[1] - x_lim[0]
    for i, d in enumerate(deltas):
        # Color encodes pattern type (matches Figs 3/4):
        #   d > 0 → risk-supporting (masking lowered predicted risk) → red.
        #   d < 0 → protective (masking raised predicted risk) → blue.
        color = COLOR_RISK if d > 0 else COLOR_PROT
        ax.annotate(
            "", xy=(d, i), xytext=(0, i),
            arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.35",
                            color=color, lw=2.8, shrinkA=2, shrinkB=4),
            zorder=3,
        )
        ax.plot(d, i, "o", color=color, markersize=11,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4)
        # Inline value annotation: positioned PAST the dot (away from
        # zero) with 4% of xrange offset, comfortable visible gap
        # between the dot and the text.
        off = xrange * 0.040 * (1 if d >= 0 else -1)
        ha = "left" if d >= 0 else "right"
        ax.text(d + off, i, f"{d:+.3f}",
                fontsize=fs_annot, color="#222222",
                ha=ha, va="center",
                clip_on=False, zorder=5)

    ax.set_yticks(y)
    # No explicit color → matplotlib default black, matches Fig 4 and
    # prints darker than the previous "#1f2937" slate.
    ax.set_yticklabels(labels, fontsize=fontsize)
    # Y-tick labels pushed 50 points LEFT of spine so the negative-side
    # value annotations (with clip_on=False) have room to extend past
    # the spine without overlapping the pattern label text.
    ax.tick_params(axis="y", pad=50, length=0)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.set_xlim(*x_lim)
    ax.set_ylim(-0.8, N - 0.2)  # matches Fig 4 / signed_ig_bars
    # Hide LEFT spine too, with tight xlim, the negative annotations
    # extend past the spine line and the visible spine cuts through
    # the annotation text. Bottom spine is kept for the x-axis ticks.
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render Figure 5 (perturbation-validation dumbbell, 2-panel).")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--jaccard_threshold", type=float, default=JACCARD_THRESHOLD)
    p.add_argument("--n_show", type=int, default=N_SHOW,
                   help="Top-N per panel (default 30, matches Fig 3 Panel C/D).")
    p.add_argument("--fontsize", type=int, default=FONTSIZE)
    p.add_argument("--dpi", type=int, default=DPI)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/Figure5.png")
    p.add_argument("--panels-dir", default=None,
                   help="Directory holding panel_c/d_*_short.csv from "
                        "make_fig3 (default: <run_dir>/7_outputs/panels).")
    p.add_argument("--show-n", action="store_true", default=False,
                   help="Append (n=...) carrier count to each pattern label "
                        "(off by default, keeps labels concise).")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    panels_dir = args.panels_dir or layout.outputs_panels_dir
    jt_pct = int(args.jaccard_threshold * 100)

    # ── Load Fig 3 Panel C + Panel D source CSVs ─────────────────────
    panel_c_df = _load_panel_csv(panels_dir, "c", args.split, jt_pct)
    panel_d_df = _load_panel_csv(panels_dir, "d", args.split, jt_pct)

    if panel_c_df is None:
        sys.exit(f"  [ERROR] panel_c_{args.split}_j{jt_pct}_short.csv missing.\n"
                 f"  Re-run scripts/make_fig3.py to materialize Panel C.")
    if panel_d_df is None:
        sys.exit(f"  [ERROR] panel_d_{args.split}_j{jt_pct}_short.csv missing.\n"
                 f"  Re-run scripts/make_fig3.py to materialize Panel D.")
    print(f"[1] Loaded Fig 3 Panel C source: {len(panel_c_df)} patterns "
          f"(full pool)")
    print(f"[2] Loaded Fig 3 Panel D source: {len(panel_d_df)} patterns "
          f"(cross-visit pool)")

    # ── Load validation deltas ───────────────────────────────────────
    val_csv = layout.validation_csv(args.split)
    if not os.path.exists(val_csv):
        sys.exit(f"  [ERROR] Missing {val_csv}")
    val = pd.read_csv(val_csv)
    print(f"[3] Loaded {len(val):,} validation rows "
          f"({os.path.basename(val_csv)})")

    # ── Join + select top-N each panel ───────────────────────────────
    print("\n[4] Selecting top-N per panel…")
    # Convention: Δŷ = ŷ_original − ŷ_masked
    #   • positive Δŷ = masking LOWERED risk = risk-supporting pattern
    #   • negative Δŷ = masking RAISED risk  = protective pattern
    print(f"  Panel A (full pool / Fig 3 C):")
    sel_a = _select_topn_with_validation(panel_c_df, val, args.n_show)
    print(f"    → {len(sel_a)} patterns "
          f"({(sel_a['mean_delta_mask'] > 0).sum()} risk-supporting, "
          f"{(sel_a['mean_delta_mask'] < 0).sum()} protective)")
    print(f"  Panel B (cross-visit / Fig 3 D):")
    sel_b = _select_topn_with_validation(panel_d_df, val, args.n_show)
    print(f"    → {len(sel_b)} patterns "
          f"({(sel_b['mean_delta_mask'] > 0).sum()} risk-supporting, "
          f"{(sel_b['mean_delta_mask'] < 0).sum()} protective)")

    # ── Compose figure ───────────────────────────────────────────────
    resolver = PatternLabelResolver(args.run_dir, verbose=False)

    # Both panels: SYMMETRIC xlim at (-0.3, +0.3). Same fixed range
    # across both pools so dumbbell lengths are directly comparable
    # by eye. Ticks land at -0.2, 0.0, 0.2 on both.
    xlim_a = (-0.3, 0.3)
    xlim_b = (-0.3, 0.3)
    panel_xlims = [xlim_a, xlim_b]

    # ── Composition matches make_fig4.py EXACTLY ──────────────────────
    # Same panel_w (13.0), same wspace (0.65), same height formula, so
    # Fig 5 reads as a layout sibling of Fig 4, not a wider, more
    # awkward cousin.
    rows_panel = max(len(sel_a), len(sel_b))
    panel_w = 13.0           # matches make_fig4.py
    panel_h = max(10.0, rows_panel * 0.55 + 2.5)
    fig_w = 2 * panel_w + 3.0
    fig_h = panel_h + 0.8
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.65},
    )

    # Subtitles parallel to make_fig4.py: "Pool name: description (qualifier)"
    # Subtitle framing balances two needs:
    #   1. The ARROW visualizes the pattern's contribution to predicted
    #      risk (not the masking action itself), so "contribution" is
    #      the subject and the arrow direction stays intuitive.
    #   2. The reader must understand this is methodologically DIFFERENT
    #      from Figs 3/4 (which use IG attribution). The "validated by
    #      masking" qualifier flags this as the perturbation-validation
    #      figure, the empirical confirmation that IG-attributed
    #      patterns actually move the model's output causally.
    DUM_SUBTITLES = {
        0: "Masking effect of top patterns on predicted readmission risk",
        1: "Masking effect of top cross-visit patterns on predicted readmission risk",
    }
    DUM_HEADS = {0: "A", 1: "B"}
    # X-axis label: Δŷ = ŷ_original − ŷ_masked (standard occlusion/ERASER
    # importance convention, matching Tables 3/4, Methods, the faithfulness
    # analysis, and the red=risk / right=positive orientation of Figs 3/4).
    #   • risk-supporting pattern → masking lowers ŷ → Δŷ > 0 → red, RIGHT
    #   • protective pattern       → masking raises ŷ → Δŷ < 0 → blue, LEFT
    # Double newline adds vertical space between the formula and the
    # direction-arrow line so they don't read as one jammed block.
    DUM_XLABEL = (
        r"$\Delta\hat{y} = \hat{y}_{\mathrm{original}} - \hat{y}_{\mathrm{masked}}$"
        + "\n\n" + r"← protective    risk-supporting →"
    )

    panel_data = [sel_a, sel_b]
    for i, (ax, sel) in enumerate(zip(axes, panel_data)):
        _draw_dumbbell_panel(
            ax, sel, resolver,
            fontsize=args.fontsize, x_lim=panel_xlims[i],
            show_n=args.show_n,
        )
        # xlabel matches Fig 4 exactly: same fontsize + labelpad=30
        # for substantial vertical breathing room between x-tick
        # numbers and the formula+arrows label.
        ax.set_xlabel(DUM_XLABEL, fontsize=args.fontsize, labelpad=30)
        ax.tick_params(axis="x", labelsize=args.fontsize - 1)
        # Force xticks to -0.2, 0.0, 0.2 so both panels show the same
        # three-tick pattern regardless of matplotlib's auto-tick choice.
        ax.set_xticks([-0.2, 0.0, 0.2])

    # ── Bars-to-text ratio: 40/60 split per panel ──────────────────────
    # Mirrors make_fig4.py. Dumbbells area takes 40% of each panel column;
    # y-tick labels take 60%. This gives pattern labels room to be large
    # without losing dumbbell legibility. No tight_layout below, it would
    # undo this manual placement.
    fig.canvas.draw()
    BARS_FRACTION = 0.40
    for ax in axes:
        pos = ax.get_position()
        new_width = pos.width * BARS_FRACTION
        new_x0 = pos.x0 + pos.width * (1 - BARS_FRACTION)
        ax.set_position([new_x0, pos.y0, new_width, pos.height])

    # Column titles drawn via fig.text, left-anchored at the panel's
    # tick-label edge. After the 40/60 split, the tick-label column
    # starts well to the LEFT of the axes, re-measure via tight
    # bounding box (which now includes the wide tick-label area)
    # to anchor precisely. Same logic as make_fig4.py.
    fig.canvas.draw()
    for i, ax in enumerate(axes):
        tbbox = ax.get_tightbbox(fig.canvas.get_renderer())
        tbbox_fig = tbbox.transformed(fig.transFigure.inverted())
        title_x = tbbox_fig.x0
        title_y = tbbox_fig.y1 + 0.020  # matches Fig 4 (bumped from 0.005)
        fig.text(
            title_x, title_y,
            f"$\\bf{{{DUM_HEADS[i]}.}}$  " + DUM_SUBTITLES[i],
            fontsize=16, ha="left", va="bottom", fontweight="bold",
        )

    # NOTE: no tight_layout call, it would undo the manual
    # set_position() above and reset the 40/60 split.
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(args.output_path) // 1024
    print(f"\n[5] Saved figure (2-panel dumbbell): "
          f"{args.output_path} ({size_kb} KB)")

    # Backing CSV, one row per pattern with `panel` column ('A' or 'B').
    export_rows = []
    for letter, sel in [("A", sel_a), ("B", sel_b)]:
        for _, r in sel.iterrows():
            export_rows.append({
                "panel":      letter,
                "pattern":    r["pattern"],
                "label":      resolver.format_pattern(r["pattern"], short_names=True),
                # delta uses Δŷ = ŷ_original − ŷ_masked convention.
                # direction describes the pattern's nature, inferred
                # from the sign of the masking effect.
                "delta":      round(r["mean_delta_mask"], 5),
                "n_carriers": int(r["n_matched_test"]),
                "direction":  "risk" if r["mean_delta_mask"] > 0 else "protective",
            })
    out_csv = os.path.splitext(args.output_path)[0] + ".csv"
    pd.DataFrame(export_rows).to_csv(out_csv, index=False)
    print(f"    csv: {out_csv}")


if __name__ == "__main__":
    main()
