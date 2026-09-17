"""make_fig_s1, render Figure S1 (forward vs reversed temporal order ablation).

Two render modes share the same source data
(``reversed_order_<split>.csv``):

  ``--mode flat``     Single ranked list, the top-30 cross-visit patterns
                       from Figure 4B (top-30 by |E[ΣIG | present]|), one
                       row per pattern, sorted within the figure by
                       |ig_diff|.

  ``--mode grouped``  Bars organized by phenotype family, matching the
                       Table 4 displayed set (top-5 per family + Other
                       risk/protective split). Phenotype labels appear
                       on the left margin; horizontal separators divide
                       families.

The reader cross-walks Figure S1 to Figure 4B (flat mode) or Table 4
(grouped mode) one-to-one.

Usage:
    python scripts/make_fig_s1.py --run_dir ../runs/mimiciv_readmission_30d --mode flat
    python scripts/make_fig_s1.py --run_dir ../runs/mimiciv_readmission_30d --mode grouped
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
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.explain.pattern_labels import PatternLabelResolver
from atop.run_layout import RunLayout


# Forward-order risk/protective + lighter shades for reversed-order
COLOR_FWD_RISK = "#c44e52"
COLOR_FWD_PROT = "#4c72b0"
COLOR_REV_RISK = "#e8a0a2"
COLOR_REV_PROT = "#a0b8d0"


def _load_reversed_csv(csv_dir: str, split: str) -> pd.DataFrame:
    for cand in (
        f"reversed_order_{split}.csv",
        f"fig5_reversed_order_{split}.csv",
        f"reversed_{split}.csv",
    ):
        path = os.path.join(csv_dir, cand)
        if os.path.exists(path):
            return pd.read_csv(path)
    sys.exit(f"  [ERROR] No reversed-order CSV under {csv_dir}/")


def _load_manifest(analysis_dir: str) -> pd.DataFrame:
    path = os.path.join(analysis_dir, "reversed_order_whitelist_manifest.csv")
    if not os.path.exists(path):
        sys.exit(f"  [ERROR] Missing {path}\n"
                  f"  Build it with the build_validation_whitelist sibling helper "
                  f"(or the inline one-shot in the supplement workflow).")
    return pd.read_csv(path)


def _render_panel(ax, df_plot: pd.DataFrame, resolver: PatternLabelResolver,
                   fontsize: int, bar_h: float = 0.38) -> None:
    """Render the forward + reversed paired-bar panel. Caller sets up
    figure size, group separators, and phenotype labels (if any).
    """
    y = np.arange(len(df_plot))
    fwd_colors = [COLOR_FWD_RISK if v > 0 else COLOR_FWD_PROT
                   for v in df_plot["ig_mean_forward"]]
    rev_colors = [COLOR_REV_RISK if v > 0 else COLOR_REV_PROT
                   for v in df_plot["ig_mean_reversed"]]
    ax.barh(y + bar_h / 2, df_plot["ig_mean_forward"], height=bar_h,
             color=fwd_colors, edgecolor="white", alpha=0.95)
    ax.barh(y - bar_h / 2, df_plot["ig_mean_reversed"], height=bar_h,
             color=rev_colors, edgecolor="white", alpha=0.95)

    labels = []
    for _, r in df_plot.iterrows():
        if "pattern" in r.index:
            pat_label = resolver.format_pattern(r["pattern"], short_names=True)
        else:
            pat_label = r.get("pattern_readable", "")
        if len(pat_label) > 80:
            pat_label = pat_label[:77] + "..."
        labels.append(pat_label)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize - 1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel(
        r"$E[\Sigma\mathrm{IG}\,|\,\mathrm{present}]$" + "\n\n"
        + r"$\leftarrow$ protective    risk $\rightarrow$",
        fontsize=fontsize, labelpad=20,
    )
    ax.legend(handles=[
        Patch(facecolor=COLOR_FWD_RISK, label="Forward (risk)"),
        Patch(facecolor=COLOR_REV_RISK, label="Reversed (risk)"),
        Patch(facecolor=COLOR_FWD_PROT, label="Forward (protective)"),
        Patch(facecolor=COLOR_REV_PROT, label="Reversed (protective)"),
    ], fontsize=fontsize - 1, loc="lower right", framealpha=0.95)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _draw_phenotype_bands(fig, ax, df_plot: pd.DataFrame, fontsize: int) -> None:
    """Add phenotype family headers in the far-left figure margin (outside
    the pattern-label column) + horizontal separators between groups.
    df_plot must be pre-sorted by phenotype family with the family
    ordering encoded in row order.

    Labels are placed in FIGURE coordinates so they sit cleanly to the
    left of the long pattern labels, not on top of them. Separators
    are drawn at full figure width using axhline.
    """
    if "phenotype" not in df_plot.columns:
        return
    phen_col = df_plot["phenotype"].fillna("(unclassified)")
    phen_list = list(phen_col)
    n = len(phen_list)

    boundaries = [0]
    for i in range(1, n):
        if phen_list[i] != phen_list[i - 1]:
            boundaries.append(i)
    boundaries.append(n)

    fig.canvas.draw()
    ax_pos = ax.get_position()   # figure-relative bbox of the axes

    # Map each phenotype-group's y-center (in data coords) → figure y coord
    # For barh: y=0 is bottom, y=n-1 is top. Axes data y-range = (-0.5, n-0.5).
    def _y_data_to_fig(y_data: float) -> float:
        # axes spans (ax_pos.y0, ax_pos.y0 + ax_pos.height) in fig coords
        # data y-range we use: -0.5 to n-0.5
        frac = (y_data - (-0.5)) / (n - 0.5 - (-0.5))
        return ax_pos.y0 + frac * ax_pos.height

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start > 0:
            ax.axhline(start - 0.5, color="#444444", linewidth=1.0, alpha=0.7)
        ymid = (start + end - 1) / 2.0
        phen_name = phen_list[start]
        fig.text(0.01, _y_data_to_fig(ymid), phen_name,
                  fontsize=fontsize + 2, fontweight="bold",
                  ha="left", va="center", color="#222222")


def _filter_to_source(df_rev: pd.DataFrame, manifest: pd.DataFrame,
                       source: str) -> pd.DataFrame:
    """Restrict df_rev to patterns from a given source (fig4b or table4),
    attach phenotype + source metadata."""
    if source == "flat":   # Fig 4B
        keep = set(manifest[manifest["in_fig4b"]]["pattern"].astype(str))
    elif source == "grouped":   # Table 4
        keep = set(manifest[manifest["in_table4"]]["pattern"].astype(str))
    else:
        sys.exit(f"  [ERROR] unknown source: {source}")

    df = df_rev[df_rev["pattern"].astype(str).isin(keep)].copy()
    # Attach phenotype from manifest (only meaningful for table 4 / grouped)
    phen_map = dict(zip(manifest["pattern"].astype(str), manifest["phenotype"]))
    df["phenotype"] = df["pattern"].astype(str).map(phen_map).fillna("")
    return df


def _split_other_phenotype(df: pd.DataFrame) -> pd.DataFrame:
    """For the grouped (Table 4) view, split 'Other' into 'Other (risk-supporting)'
    and 'Other (protective)' to mirror the main-paper table organization."""
    out = df.copy()
    is_other = out["phenotype"] == "Other"
    out.loc[is_other & (out["ig_mean_forward"] > 0), "phenotype"] = "Other (risk-supporting)"
    out.loc[is_other & (out["ig_mean_forward"] <= 0), "phenotype"] = "Other (protective)"
    return out


BARS_FRACTION = 0.40   # mirrors make_fig4.py; bars right 40%, labels left 60%


def _compress_bars_area(fig, ax) -> None:
    """Apply Fig 4's BARS_FRACTION=0.40 layout: bars compressed to the
    right 40% of the axes box, leaving the left 60% for long pattern
    labels (and phenotype headers in grouped mode)."""
    fig.canvas.draw()
    pos = ax.get_position()
    new_width = pos.width * BARS_FRACTION
    new_x0 = pos.x0 + pos.width * (1 - BARS_FRACTION)
    ax.set_position([new_x0, pos.y0, new_width, pos.height])


def render_flat(df_rev: pd.DataFrame, manifest: pd.DataFrame,
                 resolver: PatternLabelResolver, out_path: str, dpi: int,
                 fontsize: int = 14) -> None:
    """Flat mode, top-30 from Fig 4B, sorted by |ig_diff|."""
    df = _filter_to_source(df_rev, manifest, "flat")
    df = df[(df["n_forward"] > 0) & (df["n_reversed"] > 0)].copy()
    df["_abs_diff"] = df["ig_diff"].abs()
    df = df.sort_values("_abs_diff", ascending=True)

    # Pattern labels are long; row pitch generous so labels don't crowd.
    panel_w = 18.0
    panel_h = max(10.0, len(df) * 0.55 + 2.5)
    fig, ax = plt.subplots(figsize=(panel_w, panel_h))
    _render_panel(ax, df, resolver, fontsize=fontsize)

    # Pad x-axis so negative-bar tips have room on the left
    xmin, xmax = ax.get_xlim()
    xrange = xmax - xmin
    ax.set_xlim(xmin - xrange * 0.10, xmax + xrange * 0.10)

    _compress_bars_area(fig, ax)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"  [flat] saved {out_path} ({size_kb} KB, {len(df)} patterns)")


def _build_col_y_positions(df_col: pd.DataFrame) -> tuple[list[float], list[tuple[int, str]]]:
    """Compute y positions for each row, leaving 1.0-row vertical gap
    above each phenotype family header. Returns:
      ys          . y position per data row, top-to-bottom in figure
      group_heads . list of (y_position_of_header, phenotype_name) tuples
    """
    if len(df_col) == 0:
        return [], []
    phen_list = list(df_col["phenotype"].fillna("(unclassified)"))
    ys = []
    group_heads = []
    cursor = 0.0
    last_phen = None
    for ph in phen_list:
        if ph != last_phen:
            # Add a header above this group
            group_heads.append((cursor, ph))
            cursor += 1.0   # space for header
            last_phen = ph
        ys.append(cursor)
        cursor += 1.0
    return ys, group_heads


def _render_col(ax, df_col: pd.DataFrame, resolver: PatternLabelResolver,
                 fontsize: int, show_xlabel: bool, show_legend: bool,
                 max_total_rows: float, bar_h: float = 0.38,
                 label_x_frac: float = -1.40) -> None:
    """Render one column of the 3-column grouped layout.
      • Bars: paired forward + reversed, BARS_FRACTION area on right
      • Pattern labels: LEFT-JUSTIFIED in the freed-up left margin
      • Phenotype headers: bold, placed ABOVE each family group with
        a separator line; also LEFT-JUSTIFIED
    All columns share max_total_rows so the y-axis range is identical
    across columns (visual height matches).
    """
    if len(df_col) == 0:
        ax.set_axis_off()
        return

    ys, group_heads = _build_col_y_positions(df_col)
    # Convert to top-down: y=0 should be at TOP of figure (matplotlib
    # barh draws y=0 at bottom by default, so we'll invert the axis
    # rather than reversing ys).
    fwd_colors = [COLOR_FWD_RISK if v > 0 else COLOR_FWD_PROT
                   for v in df_col["ig_mean_forward"]]
    rev_colors = [COLOR_REV_RISK if v > 0 else COLOR_REV_PROT
                   for v in df_col["ig_mean_reversed"]]
    ax.barh([y + bar_h / 2 for y in ys], df_col["ig_mean_forward"],
             height=bar_h, color=fwd_colors, edgecolor="white", alpha=0.95)
    ax.barh([y - bar_h / 2 for y in ys], df_col["ig_mean_reversed"],
             height=bar_h, color=rev_colors, edgecolor="white", alpha=0.95)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    # Suppress default y-tick labels (manual placement below)
    ax.set_yticks(ys)
    ax.set_yticklabels([""] * len(ys))
    # Set shared y-axis range so all columns have the same vertical pitch
    ax.set_ylim(max_total_rows - 0.5, -1.0)   # inverted: top is top

    # Pattern labels (LEFT-justified). Aggressively wrap long patterns
    # to multiple lines at " → " arrows so text fits within the column
    # without colliding with the bars area on the right.
    def _wrap(pat: str, line_max: int = 42) -> str:
        if len(pat) <= line_max:
            return pat
        lines = []
        rest = pat
        while len(rest) > line_max:
            cut = rest.rfind(" → ", 0, line_max)
            if cut <= 0:
                cut = rest.find(" → ", line_max // 2)
            if cut <= 0:
                cut = line_max
            lines.append(rest[:cut + 3].rstrip())
            rest = rest[cut + 3:]
        if rest:
            lines.append(rest)
        return "\n".join(lines)

    for y, (_, r) in zip(ys, df_col.iterrows()):
        pat = resolver.format_pattern(r["pattern"], short_names=True)
        pat = _wrap(pat, line_max=42)
        ax.text(label_x_frac, y, pat,
                 transform=ax.get_yaxis_transform(),
                 ha="left", va="center", fontsize=fontsize, color="#111111")

    # Phenotype headers (LEFT-justified, bold, with separator line below)
    for y_head, ph in group_heads:
        ax.text(label_x_frac, y_head, ph,
                 transform=ax.get_yaxis_transform(),
                 ha="left", va="center",
                 fontsize=fontsize + 2, fontweight="bold", color="#000000")
        # Separator line between header and first row
        ax.axhline(y_head + 0.5, color="#666666", linewidth=0.9, alpha=0.5)

    ax.tick_params(axis="x", labelsize=fontsize - 1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    if show_xlabel:
        ax.set_xlabel(
            r"$E[\Sigma\mathrm{IG}\,|\,\mathrm{present}]$" + "\n"
            + r"$\leftarrow$ protective    risk $\rightarrow$",
            fontsize=fontsize, labelpad=14,
        )
    if show_legend:
        ax.legend(handles=[
            Patch(facecolor=COLOR_FWD_RISK, label="Forward (risk)"),
            Patch(facecolor=COLOR_REV_RISK, label="Reversed (risk)"),
            Patch(facecolor=COLOR_FWD_PROT, label="Forward (protective)"),
            Patch(facecolor=COLOR_REV_PROT, label="Reversed (protective)"),
        ], fontsize=fontsize - 1, loc="lower right", framealpha=0.95)


def render_grouped(df_rev: pd.DataFrame, manifest: pd.DataFrame,
                    resolver: PatternLabelResolver, out_path: str, dpi: int,
                    fontsize: int = 14) -> None:
    """Grouped mode (single column, vertical layout with extra spacing
    between phenotype families). Table 4 displayed set, banded by
    phenotype.
      - Patterns stacked top-to-bottom in a single column
      - Bold phenotype headers above each family
      - Empty row of vertical space between families (visual separation)
      - Horizontal separator line between families
      - Left-justified pattern labels + phenotype headers
    """
    df = _filter_to_source(df_rev, manifest, "grouped")
    df = df[(df["n_forward"] > 0) & (df["n_reversed"] > 0)].copy()
    df = _split_other_phenotype(df)

    PHEN_ORDER = ["Psychiatric", "Oncologic", "Obstetric",
                   "Chest pain workup", "Hepatobiliary", "Substance use",
                   "Other (risk-supporting)", "Other (protective)"]
    df["_pord"] = df["phenotype"].apply(
        lambda p: PHEN_ORDER.index(p) if p in PHEN_ORDER else 99)
    df["_abs_fwd"] = df["ig_mean_forward"].abs()
    df = df.sort_values(["_pord", "_abs_fwd"], ascending=[True, False]).reset_index(drop=True)

    # Build y positions: header row + pattern rows for each family,
    # plus a 1.0-row vertical GAP between families for visual separation.
    GAP_BETWEEN_FAMILIES = 0.4
    HEADER_ROW = 1.0
    ys = []
    group_heads = []   # list of (y_header, phenotype_name)
    phen_list = list(df["phenotype"])
    cursor = 0.0
    last_phen = None
    for ph in phen_list:
        if ph != last_phen:
            if last_phen is not None:
                cursor += GAP_BETWEEN_FAMILIES   # blank gap after previous family
            group_heads.append((cursor, ph))
            cursor += HEADER_ROW
            last_phen = ph
        ys.append(cursor)
        cursor += 1.0
    total_units = cursor

    # X-axis range
    xmin = df[["ig_mean_forward", "ig_mean_reversed"]].min().min()
    xmax = df[["ig_mean_forward", "ig_mean_reversed"]].max().max()
    xrange = xmax - xmin
    xlim = (xmin - xrange * 0.10, xmax + xrange * 0.10)

    fig_w = 18.0
    fig_h = max(11.0, total_units * 0.55 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Bars (paired forward + reversed)
    bar_h = 0.38
    fwd_colors = [COLOR_FWD_RISK if v > 0 else COLOR_FWD_PROT
                   for v in df["ig_mean_forward"]]
    rev_colors = [COLOR_REV_RISK if v > 0 else COLOR_REV_PROT
                   for v in df["ig_mean_reversed"]]
    ax.barh([y + bar_h / 2 for y in ys], df["ig_mean_forward"],
             height=bar_h, color=fwd_colors, edgecolor="white", alpha=0.95)
    ax.barh([y - bar_h / 2 for y in ys], df["ig_mean_reversed"],
             height=bar_h, color=rev_colors, edgecolor="white", alpha=0.95)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    # Suppress default y-ticks; place pattern labels manually left-justified
    ax.set_yticks(ys)
    ax.set_yticklabels([""] * len(ys))
    ax.set_ylim(total_units - 0.5, -1.0)   # invert: top is top

    LABEL_X_FRAC = -1.50   # axes-fraction; tuned for BARS_FRACTION=0.40
    for y, (_, r) in zip(ys, df.iterrows()):
        pat = resolver.format_pattern(r["pattern"], short_names=True)
        if len(pat) > 90:
            pat = pat[:87] + "..."
        ax.text(LABEL_X_FRAC, y, pat,
                 transform=ax.get_yaxis_transform(),
                 ha="left", va="center", fontsize=fontsize, color="#111111")

    # Phenotype headers (bold, left-justified at same x as pattern labels)
    # + separator line right BELOW the header to mark the start of the group
    for y_head, ph in group_heads:
        ax.text(LABEL_X_FRAC, y_head, ph,
                 transform=ax.get_yaxis_transform(),
                 ha="left", va="center",
                 fontsize=fontsize + 3, fontweight="bold", color="#000000")
        ax.axhline(y_head + 0.5, color="#444444", linewidth=1.0, alpha=0.6)

    ax.set_xlim(*xlim)
    ax.set_xlabel(
        r"$E[\Sigma\mathrm{IG}\,|\,\mathrm{present}]$" + "\n"
        + r"$\leftarrow$ protective    risk $\rightarrow$",
        fontsize=fontsize, labelpad=18,
    )
    ax.tick_params(axis="x", labelsize=fontsize - 1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax.legend(handles=[
        Patch(facecolor=COLOR_FWD_RISK, label="Forward (risk)"),
        Patch(facecolor=COLOR_REV_RISK, label="Reversed (risk)"),
        Patch(facecolor=COLOR_FWD_PROT, label="Forward (protective)"),
        Patch(facecolor=COLOR_REV_PROT, label="Reversed (protective)"),
    ], fontsize=fontsize - 1, loc="lower right", framealpha=0.95)

    # Compress bars to right 40% so the left 60% is for labels.
    fig.canvas.draw()
    pos = ax.get_position()
    new_w = pos.width * BARS_FRACTION
    new_x0 = pos.x0 + pos.width * (1 - BARS_FRACTION)
    ax.set_position([new_x0, pos.y0, new_w, pos.height])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"  [grouped] saved {out_path} ({size_kb} KB, {len(df)} patterns, "
           f"7 families with {GAP_BETWEEN_FAMILIES:.0f}-row gaps)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render Figure S1 (reversed-order ablation, two layout modes).")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--mode", choices=["flat", "grouped", "scatter", "topbars", "both"], default="topbars")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fontsize", type=int, default=14)
    p.add_argument("--output-dir",
                    default="../manuscript/jamia_submission/figures/supplements")
    p.add_argument("--analysis-dir", default=None)
    p.add_argument("--short-names-csv",
                    default=os.path.join(os.path.dirname(__file__), "short_names.csv"))
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    csv_dir = args.analysis_dir or layout.analysis_dir
    resolver = PatternLabelResolver(args.run_dir, short_names_csv=args.short_names_csv)

    df_rev = _load_reversed_csv(csv_dir, args.split)
    manifest = _load_manifest(csv_dir)
    print(f"[1] Loaded {len(df_rev)} reversed-order rows, "
           f"{len(manifest)} whitelist entries")

    if args.mode in ("flat", "both"):
        out_flat = os.path.join(args.output_dir,
                                  "FigureS1_reversed_order_flat.png")
        render_flat(df_rev, manifest, resolver, out_flat, args.dpi,
                     fontsize=args.fontsize)

    if args.mode in ("grouped", "both"):
        out_grp = os.path.join(args.output_dir,
                                 "FigureS1_reversed_order_grouped.png")
        render_grouped(df_rev, manifest, resolver, out_grp, args.dpi,
                        fontsize=args.fontsize)

    if args.mode in ("scatter", "both"):
        # Scatter retained as an alternate visualization in the data
        # release; the canonical Figure S1 in the manuscript is the
        # top-30 bar chart (see below).
        out_sca = os.path.join(args.output_dir,
                                 "FigureS1_reversed_order_scatter.png")
        render_scatter(df_rev, manifest, out_sca, args.dpi,
                        fontsize=args.fontsize)

    if args.mode in ("topbars", "both"):
        out_tb = os.path.join(args.output_dir,
                                 "FigureS1_reversed_order.png")
        render_topbars(args.run_dir, resolver, out_tb, args.dpi,
                        fontsize=args.fontsize)


# Phenotype color palette (consistent across grouped + scatter modes)
PHEN_COLORS = {
    "Psychiatric":              "#c44e52",   # red
    "Oncologic":                "#dd8452",   # orange
    "Obstetric":                "#4c72b0",   # blue
    "Chest pain workup":        "#8172b3",   # purple
    "Hepatobiliary":            "#937860",   # brown
    "Substance use":            "#da8bc3",   # pink
    "Other (risk-supporting)":  "#55a868",   # green (risk-side)
    "Other (protective)":       "#64b5cd",   # cyan (protective-side)
}


def render_scatter(df_rev: pd.DataFrame, manifest: pd.DataFrame,
                    out_path: str, dpi: int, fontsize: int = 14) -> None:
    """Single-panel scatter, forward signed ΣIG (x) vs reversed signed ΣIG (y).
    One dot per pattern; colored by phenotype family. The y=x diagonal
    is the "no temporal sensitivity" reference. Temporal sensitivity
    drags dots TOWARD THE Y-AXIS (reversed magnitude shrinks while sign
    is preserved). Quadrant pattern (no sign flips) is visually obvious.
    """
    from scipy.stats import wilcoxon
    df = _filter_to_source(df_rev, manifest, "grouped")
    df = df[(df["n_forward"] > 0) & (df["n_reversed"] > 0)].copy()
    df = _split_other_phenotype(df)

    x = df["ig_mean_forward"].to_numpy()
    y = df["ig_mean_reversed"].to_numpy()
    abs_fwd = np.abs(x)
    abs_rev = np.abs(y)
    attn_pct = 100.0 * (1 - abs_rev / abs_fwd)

    # Headline stats
    pct_below = float((abs_fwd > abs_rev).mean() * 100)
    pct_signed_preserved = float((np.sign(x) == np.sign(y)).mean() * 100)
    mean_attn = float(np.mean(attn_pct))
    median_attn = float(np.median(attn_pct))
    w_stat, w_p = wilcoxon(abs_fwd, abs_rev)

    fig, ax = plt.subplots(figsize=(11, 10))

    # Reference lines
    lim = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.10
    ax.plot([-lim, lim], [-lim, lim], "--", color="#666666", linewidth=1.2,
             label="y = x (no temporal sensitivity)", zorder=1)
    ax.axhline(0, color="#999999", linewidth=0.6, zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.6, zorder=1)

    # Scatter per phenotype
    for phen, color in PHEN_COLORS.items():
        m = df["phenotype"] == phen
        if not m.any():
            continue
        # Marker size proportional to log(n_forward) so heavy patterns stand out
        sizes = 50 + 40 * np.log10(df.loc[m, "n_forward"].clip(lower=1))
        ax.scatter(x[m], y[m], s=sizes, color=color, alpha=0.85,
                    edgecolors="white", linewidths=1.2, label=phen, zorder=3)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(
        r"Forward order: $E[\Sigma\mathrm{IG}\,|\,\mathrm{present}]$",
        fontsize=fontsize + 1, labelpad=10,
    )
    ax.set_ylabel(
        r"Reversed order: $E[\Sigma\mathrm{IG}\,|\,\mathrm{present}]$",
        fontsize=fontsize + 1, labelpad=10,
    )
    ax.tick_params(labelsize=fontsize - 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Headline stats box (top-left, inside (-lim, +lim))
    stats_text = (
        f"n = {len(df)} cross-visit patterns (Table 4)\n"
        f"|forward| > |reversed|: {pct_below:.0f}% ({int((abs_fwd > abs_rev).sum())}/{len(df)})\n"
        f"sign preserved: {pct_signed_preserved:.0f}% ({int((np.sign(x) == np.sign(y)).sum())}/{len(df)})\n"
        f"mean attenuation: {mean_attn:.0f}%\n"
        f"Wilcoxon $p$ = {w_p:.1e}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
             ha="left", va="top", fontsize=fontsize - 1,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                        edgecolor="#cccccc", alpha=0.95))

    # Quadrant annotations
    ax.text(lim * 0.95, lim * 0.05, "Risk-supporting\n(forward & reversed both +)",
             ha="right", va="bottom", fontsize=fontsize - 2, color="#c44e52",
             style="italic", alpha=0.7)
    ax.text(-lim * 0.95, -lim * 0.05, "Protective\n(forward & reversed both −)",
             ha="left", va="top", fontsize=fontsize - 2, color="#4c72b0",
             style="italic", alpha=0.7)

    ax.legend(fontsize=fontsize - 2, loc="lower right", framealpha=0.95,
               title="Phenotype", title_fontsize=fontsize - 1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"  [scatter] saved {out_path} ({size_kb} KB, {len(df)} dots)")


def render_topbars(run_dir: str, resolver: PatternLabelResolver,
                    out_path: str, dpi: int, fontsize: int = 12) -> None:
    """Top-30 by reversal attenuation, paired-bar visualization.
    Visual companion to Table S2. Each row = one pattern with two bars
    (forward ΣIG dark, reversed ΣIG light) and the attenuation %
    annotated at the right edge. Sorted by attenuation descending.
    """
    df_rev = pd.read_csv(os.path.join(run_dir, "5_analysis",
                                         "reversed_order_test.csv"))
    df_rev = df_rev[(df_rev["n_forward"] >= 20)
                     & (df_rev["n_reversed"] >= 20)].copy()
    df_rev["abs_fwd"] = df_rev["ig_mean_forward"].abs()
    df_rev["abs_rev"] = df_rev["ig_mean_reversed"].abs()
    df_rev["attn_pct"] = 100.0 * (1 - df_rev["abs_rev"] / df_rev["abs_fwd"])
    df_rev = df_rev.nlargest(30, "attn_pct").copy()
    df_rev["readable_label"] = df_rev["pattern"].apply(
        lambda p: resolver.format_pattern(p, short_names=True))
    # Top of plot = highest attenuation → matplotlib draws y=0 at bottom,
    # so we reverse and use indices.
    df_rev = df_rev.sort_values("attn_pct", ascending=True).reset_index(drop=True)

    fig_w = 18.0
    fig_h = max(10.0, len(df_rev) * 0.42 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y = np.arange(len(df_rev))
    bar_h = 0.38
    fwd_colors = [COLOR_FWD_RISK if v > 0 else COLOR_FWD_PROT
                   for v in df_rev["ig_mean_forward"]]
    rev_colors = [COLOR_REV_RISK if v > 0 else COLOR_REV_PROT
                   for v in df_rev["ig_mean_reversed"]]
    ax.barh(y + bar_h / 2, df_rev["ig_mean_forward"], height=bar_h,
             color=fwd_colors, edgecolor="white", alpha=0.95)
    ax.barh(y - bar_h / 2, df_rev["ig_mean_reversed"], height=bar_h,
             color=rev_colors, edgecolor="white", alpha=0.95)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    # Y-tick labels: pattern + attenuation%
    ax.set_yticks(y)
    labels = []
    for _, r in df_rev.iterrows():
        pat = r["readable_label"]
        if len(pat) > 75:
            pat = pat[:72] + "..."
        labels.append(pat)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize - 1)

    # Attenuation % annotated to the RIGHT of the longer of the two bars.
    xmin, xmax = ax.get_xlim()
    xrange = xmax - xmin
    text_offset = xrange * 0.02
    for yi, (_, r) in zip(y, df_rev.iterrows()):
        # Place the % at the right end of the chart consistently
        fwd_val = r["ig_mean_forward"]
        x_right = max(fwd_val, r["ig_mean_reversed"]) if fwd_val > 0 else 0
        if fwd_val < 0:
            x_right = 0   # for protective patterns, % goes at zero-line
        ax.text(x_right + text_offset, yi, f"{r['attn_pct']:.0f}%",
                 ha="left", va="center", fontsize=fontsize - 1,
                 fontweight="bold", color="#444444")

    ax.set_xlabel(
        r"$E[\Sigma\mathrm{IG}\,|\,\mathrm{present}]$" + "\n"
        + r"$\leftarrow$ protective    risk $\rightarrow$",
        fontsize=fontsize, labelpad=15,
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Extra right-side room for the attenuation % labels
    ax.set_xlim(xmin - xrange * 0.05, xmax + xrange * 0.12)

    # Legend placed BELOW the x-axis label, horizontally, so it never sits
    # on top of bars (the previous "lower right" inside-axes legend
    # covered the last 2 patterns).
    ax.legend(handles=[
        Patch(facecolor=COLOR_FWD_RISK, label="Forward (risk)"),
        Patch(facecolor=COLOR_REV_RISK, label="Reversed (risk)"),
        Patch(facecolor=COLOR_FWD_PROT, label="Forward (protective)"),
        Patch(facecolor=COLOR_REV_PROT, label="Reversed (protective)"),
    ], fontsize=fontsize - 1, loc="upper center",
       bbox_to_anchor=(0.5, -0.10), ncol=4, framealpha=0.95,
       borderaxespad=0)

    # Apply Fig 4's BARS_FRACTION=0.40 compress so labels get the left 60%.
    fig.canvas.draw()
    pos = ax.get_position()
    new_w = pos.width * BARS_FRACTION
    new_x0 = pos.x0 + pos.width * (1 - BARS_FRACTION)
    ax.set_position([new_x0, pos.y0, new_w, pos.height])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"  [topbars] saved {out_path} ({size_kb} KB, top {len(df_rev)})")


if __name__ == "__main__":
    main()
