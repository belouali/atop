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



def _render_dumbbells_a4(sel_a, sel_b, resolver, output_path, dpi=300,
                         show_n=False, x_lim=(-0.3, 0.3)):
    """Figure 5 at print size for an A4/Letter page (6.5 in wide).

    Same stacked layout as Figure 4 (A above B, one shared x-axis), with
    dot and arrow sizes scaled to 7 pt labels. Rows within each panel are
    ordered by |delta_y| (largest masking effect at the top).
    """
    fig_w = 6.5
    fs_label, fs_val, fs_tick, fs_xlab, fs_head, fs_sub = 7, 6.3, 7, 7.5, 9, 8
    pitch = 0.126
    n_a, n_b = len(sel_a), len(sel_b)
    head_h, axis_h, gap_h, xlab_h = 0.50, 0.30, 0.30, 0.42
    h_a, h_b = n_a * pitch, n_b * pitch
    fig_h = head_h + h_a + axis_h + gap_h + head_h + h_b + axis_h + xlab_h
    lab_w, right = 3.55 / fig_w, 0.08 / fig_w
    fig = plt.figure(figsize=(fig_w, fig_h))
    y_b = (xlab_h + axis_h) / fig_h
    y_a = y_b + (h_b + head_h + gap_h + axis_h) / fig_h
    ax_a = fig.add_axes([lab_w, y_a, 1 - lab_w - right, h_a / fig_h])
    ax_b = fig.add_axes([lab_w, y_b, 1 - lab_w - right, h_b / fig_h])
    panels = [
        (ax_a, sel_a, "A  All patterns",
         "Masking effect of the Figure 4A patterns, ordered by |\u0394\u0177|"),
        (ax_b, sel_b, "B  Cross-visit patterns",
         "Masking effect of the Figure 4B patterns, ordered by |\u0394\u0177|"),
    ]
    xrange = x_lim[1] - x_lim[0]
    for ax, sel, _, _ in panels:
        labels = []
        for _, r in sel.iterrows():
            lab = _abbreviate_label(resolver.format_pattern(r["pattern"], short_names=True))
            if show_n:
                lab = f"{lab}  (n={int(r['n_matched_test'])})"
            labels.append(lab)
        deltas = list(sel["mean_delta_mask"])
        y = list(range(len(deltas)))
        ax.scatter([0] * len(y), y, marker="s", color=COLOR_BASE, s=7,
                   zorder=2, linewidths=0)
        for i, d in enumerate(deltas):
            color = COLOR_RISK if d > 0 else COLOR_PROT
            ax.annotate("", xy=(d, i), xytext=(0, i),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2",
                                        color=color, lw=1.1, shrinkA=1, shrinkB=2),
                        zorder=3)
            ax.plot(d, i, "o", color=color, markersize=3.6,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
            off = xrange * 0.025 * (1 if d >= 0 else -1)
            ax.text(d + off, i, f"{d:+.3f}", fontsize=fs_val, color="#222222",
                    ha="left" if d >= 0 else "right", va="center", zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=fs_label)
        ax.tick_params(axis="y", pad=3, length=0)
        ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.55)
        ax.set_xlim(*x_lim)
        ax.set_ylim(-0.6, len(y) - 0.4)
        ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        ax.tick_params(axis="x", labelsize=fs_tick, length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_b.set_xlabel(r"$\Delta\hat{y} = \hat{y}_{\mathrm{original}} - \hat{y}_{\mathrm{masked}}$"
                    "     \u2190 protective    risk-supporting \u2192",
                    fontsize=fs_xlab, labelpad=4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, _, head, sub in panels:
        bb = ax.get_position()
        x0 = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0
        fig.text(x0, bb.y1 + 0.27 / fig_h, head, fontsize=fs_head,
                 fontweight="bold", ha="left", va="bottom")
        fig.text(x0, bb.y1 + 0.08 / fig_h, sub, fontsize=fs_sub,
                 ha="left", va="bottom", color="#333333")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)



def _wrap_pattern_label(lab, max_chars=30):
    """Break a long pattern label into two lines, preferring a break after a
    visit arrow, then after an item separator, closest to the middle."""
    import re as _re
    if len(lab) <= max_chars:
        return lab
    mid = len(lab) / 2
    for pattern in (r"\u2192 ", r"(; |, )"):
        cands = [m.end() for m in _re.finditer(pattern, lab)
                 if 4 < m.end() < len(lab) - 4
                 and m.end() <= max_chars + 6 and len(lab) - m.end() <= max_chars + 6]
        if cands:
            best = min(cands, key=lambda i: abs(i - mid))
            return lab[:best].rstrip() + "\n" + lab[best:].lstrip()
    bp = lab.rfind(" ", 0, max_chars)
    return lab[:bp] + "\n" + lab[bp + 1:] if bp > 0 else lab



def _two_column_axes(n_rows, fig_w=6.5, lab_w=1.95, bar_w=1.1, gap=0.2,
                     pitch=0.225, head_h=0.78, foot_h=0.72):
    """Figure-3-style geometry: two columns, labels left of each bar panel."""
    fig_h = head_h + n_rows * pitch + foot_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    bottom, top = foot_h / fig_h, 1 - head_h / fig_h
    x_a = lab_w / fig_w
    x_b = (2 * lab_w + bar_w + gap) / fig_w
    ax_a = fig.add_axes([x_a, bottom, bar_w / fig_w, top - bottom])
    ax_b = fig.add_axes([x_b, bottom, bar_w / fig_w, top - bottom])
    return fig, ax_a, ax_b, top, fig_h


def _render_dumbbells_side(sel_a, sel_b, resolver, output_path, dpi=300, show_n=False, wide=False):
    """Figure 5 in the two-column geometry of Figure 3 (6.5 in wide)."""
    if wide:   # landscape: one-line labels, 9 in wide
        fs_label, fs_val, fs_tick, fs_xlab, fs_head, fs_sub = 7, 6.3, 7, 7.5, 9.5, 8
        geo = dict(fig_w=9.0, lab_w=3.3, bar_w=1.05, gap=0.2, pitch=0.165, head_h=0.72, foot_h=0.66)
        wrap_at = 200
    else:
        fs_label, fs_val, fs_tick, fs_xlab, fs_head, fs_sub = 6.5, 5.8, 6.5, 7, 9, 7.5
        geo = {}
        wrap_at = 30
    n = max(len(sel_a), len(sel_b))
    fig, ax_a, ax_b, top, fig_h = _two_column_axes(n, **geo)
    data_max = max(sel_a["mean_delta_mask"].abs().max(), sel_b["mean_delta_mask"].abs().max())
    lim = data_max * 1.75
    panels = [(ax_a, sel_a, "A  All patterns", "Masking effect of the Figure 4A set, by |\u0394\u0177|"),
              (ax_b, sel_b, "B  Cross-visit patterns", "Masking effect of the Figure 4B set, by |\u0394\u0177|")]
    for ax, sel, _, _ in panels:
        labels = []
        for _, r in sel.iterrows():
            lab = _abbreviate_label(resolver.format_pattern(r["pattern"], short_names=True))
            if show_n:
                lab = f"{lab}  (n={int(r['n_matched_test'])})"
            labels.append(_wrap_pattern_label(lab, wrap_at))
        deltas = list(sel["mean_delta_mask"]); y = list(range(len(deltas)))
        ax.scatter([0] * len(y), y, marker="s", color=COLOR_BASE, s=6, zorder=2, linewidths=0)
        for i, d in enumerate(deltas):
            color = COLOR_RISK if d > 0 else COLOR_PROT
            ax.annotate("", xy=(d, i), xytext=(0, i),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.18",
                                        color=color, lw=1.0, shrinkA=1, shrinkB=2), zorder=3)
            ax.plot(d, i, "o", color=color, markersize=3.2, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
            off = data_max * 0.09 * (1 if d >= 0 else -1)
            ax.text(d + off, i, f"{d:+.3f}", fontsize=fs_val, color="#222222",
                    ha="left" if d >= 0 else "right", va="center", zorder=5)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=fs_label)
        ax.tick_params(axis="y", pad=3, length=0)
        ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.55)
        ax.set_xlim(-lim, lim); ax.set_ylim(-0.6, len(y) - 0.4)
        ax.set_xticks([-0.2, 0.0, 0.2])
        ax.tick_params(axis="x", labelsize=fs_tick, length=3)
        ax.set_xlabel(r"$\Delta\hat{y} = \hat{y}_{\mathrm{original}} - \hat{y}_{\mathrm{masked}}$" + "\n\u2190 protective    risk-supporting \u2192",
                      fontsize=fs_xlab, labelpad=4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, _, head, sub in panels:
        x0 = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0
        fig.text(x0, top + 0.40 / fig_h, head, fontsize=fs_head, fontweight="bold", ha="left", va="bottom")
        fig.text(x0, top + 0.17 / fig_h, sub, fontsize=fs_sub, ha="left", va="bottom", color="#333333")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


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
    p.add_argument("--layout", default="wide", choices=["stacked", "side", "wide"],
                   help="wide: landscape two columns with one-line labels (default); side: portrait two columns with wrapped labels; stacked: A above B.")
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

    if args.layout in ("side", "wide"):
        _render_dumbbells_side(sel_a, sel_b, resolver, args.output_path, args.dpi,
                               show_n=args.show_n, wide=(args.layout == "wide"))
    else:
        _render_dumbbells_a4(sel_a, sel_b, resolver, args.output_path, args.dpi,
                             show_n=args.show_n)
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
