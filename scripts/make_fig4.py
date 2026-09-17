"""make_fig4, render Figure 4 (pattern-level attribution, 2-panel bar chart).

Mirrors the new make_fig3.py style:
  • 2 columns side-by-side, titles ABOVE each column (no empty vertical space)
  • Bold panel letter + colon + plain-English description in the header
  • Math notation E[ΣIG | present] lives on the x-axis label, above the
    "← protective    risk →" direction arrows
  • Inline bar value annotations
  • Per-panel xlim (Panel A and Panel B both compute carrier-conditional ΣIG,
    same units, but each panel auto-fits its own data for legibility,
    matching the make_fig3.py convention)
  • No top legend (red/blue + axis label are self-explanatory)

Data sources:
  • Panel A: panel_c_<split>_j<jt>_short.csv  → full pool (no shape minimums)
  • Panel B: panel_d_<split>_j<jt>_short.csv  → cross-visit pool (≥2 visits, ≥2 tokens)

Both CSVs are written by make_fig3.py (which always computes them, even
though the new 2-panel Figure 3 only renders the token-level panels).

Usage:
    python scripts/make_fig4.py --run_dir ../runs/mimiciv_readmission_30d
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

from atop.plots import signed_ig_bars
from atop.run_layout import RunLayout


# ── Defaults ─────────────────────────────────────────────────────────────
JACCARD_THRESHOLD = 0.2
N_SHOW = 30
FONTSIZE = 16     # bumped, with 65% of each panel reserved for text-tick
                  # labels (and only 35% for bars), the labels can grow
                  # significantly without colliding with the bars.
DPI = 300


# Label abbreviations applied to every pattern label before wrapping.
# Pattern labels are LONG, so a few aggressive abbreviations keep them
# on one line. Add new entries as needed; first match wins.
LABEL_ABBREV = [
    ("metastases", "mets"),
    ("Bone mets", "Bone mets"),         # keep capitalization stable
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
    """Hard-wrap a pattern label longer than max_chars at the last space
    before the limit. Pattern labels can be 80+ chars when they include
    multi-token co-occurrences inside curly braces."""
    if len(lab) <= max_chars:
        return lab
    bp = lab.rfind(" ", 0, max_chars)
    if bp > 20:
        return lab[:bp].rstrip() + "\n  " + lab[bp:].lstrip()
    return lab


def _load_panel_csv(panels_dir: str, key: str, split: str,
                     jt: float) -> pd.DataFrame:
    """Load panel_<key>_<split>_j<jt_pct>_short.csv produced by make_fig3."""
    jt_pct = int(jt * 100)
    path = os.path.join(panels_dir, f"panel_{key}_{split}_j{jt_pct}_short.csv")
    if not os.path.exists(path):
        sys.exit(f"  [ERROR] Missing {path}. Run scripts/make_fig3.py first "
                  ", it writes the pattern CSVs Figure 4 consumes.")
    df = pd.read_csv(path)
    print(f"  [load] {os.path.basename(path)}: {len(df)} patterns")
    return df


def _prep_panel(df: pd.DataFrame, n_show: int,
                 show_n: bool = False) -> tuple[list[float], list[str]]:
    """Take top-N by |ig_value|, sort ascending for display, return
    (values, labels) ready for signed_ig_bars. Labels are abbreviated
    (e.g. "metastases" → "mets") and hard-wrapped if too long."""
    df = df.copy()
    df["_abs"] = df["ig_value"].abs()
    top = df.nlargest(n_show, "_abs").sort_values("_abs", ascending=True)
    values = list(top["ig_value"])
    labels = []
    for _, r in top.iterrows():
        lab = _abbreviate_label(str(r["label"]))
        lab = _wrap_long_label(lab)
        if show_n and r.get("n_present", 0) > 0:
            lab = f"{lab}  (n={int(r['n_present'])})"
        labels.append(lab)
    return values, labels


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render Figure 4 (pattern-level attribution, 2-panel bar chart).")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--jaccard_threshold", type=float, default=JACCARD_THRESHOLD)
    p.add_argument("--n_show", type=int, default=N_SHOW,
                   help="Top-N per panel (default 30).")
    p.add_argument("--fontsize", type=int, default=FONTSIZE)
    p.add_argument("--dpi", type=int, default=DPI)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/Figure4.png")
    p.add_argument("--panels-dir", default=None,
                   help="Directory holding panel_c/d_*_short.csv "
                        "(default: <run_dir>/7_outputs/panels).")
    p.add_argument("--show-n", action="store_true", default=False,
                   help="Append (n=...) carrier count to each pattern label "
                        "(off by default, keeps labels concise; carrier counts "
                        "live in Table 3 and TableS2).")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    panels_dir = args.panels_dir or layout.outputs_panels_dir

    print(f"[1] Loading pattern CSVs from {panels_dir}/")
    df_a = _load_panel_csv(panels_dir, "c", args.split, args.jaccard_threshold)
    df_b = _load_panel_csv(panels_dir, "d", args.split, args.jaccard_threshold)

    print(f"[2] Selecting top-{args.n_show} per panel...")
    values_a, labels_a = _prep_panel(df_a, args.n_show, show_n=args.show_n)
    values_b, labels_b = _prep_panel(df_b, args.n_show, show_n=args.show_n)

    # ── Composition mirrors make_fig3.py's 2-panel layout ─────────────
    # Pattern labels are much longer than token labels (Figure 3); each
    # panel needs LOTS of horizontal room for both labels and bars.
    # Row pitch also has to be bigger to accommodate wrapped 2-line
    # labels for long curly-brace patterns.
    rows_panel = max(len(values_a), len(values_b))
    panel_w = 13.0          # wide for long pattern labels
    panel_h = max(10.0, rows_panel * 0.55 + 2.5)
    fig_w = 2 * panel_w + 3.0
    fig_h = panel_h + 0.8
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.65},
    )

    # Subtitles mirror make_fig3.py's "ranking: description" structure.
    # Math notation (E[ΣIG | present]) lives on the x-axis below.
    PAT_SUBTITLES = {
        0: "Top 30 patterns ranked by marginal contribution to model predictions",
        1: "Top 30 cross-visit patterns ranked by marginal contribution to model predictions",
    }
    PAT_HEADS = {0: "A", 1: "B"}
    # X-axis label: same two-line format as Figure 3 (math + direction arrows).
    # Double newline puts visible vertical space between the math
    # formula and the direction-arrow line so they don't read as a
    # single jammed block.
    PAT_XLABELS = {
        0: r"$E[\Sigma\mathrm{IG}\,|\,\mathrm{present}]$" + "\n\n" + r"← protective    risk →",
        1: r"$E[\Sigma\mathrm{IG}\,|\,\mathrm{present}]$" + "\n\n" + r"← protective    risk →",
    }

    panel_data = [
        (values_a, labels_a),
        (values_b, labels_b),
    ]
    for i, (ax, (vals, labs)) in enumerate(zip(axes, panel_data)):
        signed_ig_bars(
            values=vals, labels=labs,
            ax=ax, fontsize=args.fontsize,
            show_value_annotations=True,
        )
        # labelpad=30 (was default ~4) adds substantial vertical
        # breathing room between the x-axis tick numbers and the
        # formula+arrows label text below them.
        ax.set_xlabel(PAT_XLABELS[i], fontsize=args.fontsize, labelpad=30)
        ax.tick_params(axis="x", labelsize=args.fontsize - 1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # Pad the LEFT side of the x-axis range so negative-bar value
        # annotations (rendered to the LEFT of the bar tip) don't
        # overlap the pattern labels in the y-tick column. Right side
        # gets a smaller pad, positive-bar annotations sit just past
        # the tip and never collide with labels.
        xmin, xmax = ax.get_xlim()
        xrange = xmax - xmin
        ax.set_xlim(xmin - xrange * 0.30, xmax + xrange * 0.10)

    # ── Shared x-axis across BOTH panels ───────────────────────────────
    # The legend claims the panels share a scale so cross-pool magnitudes
    # are directly comparable, and the Discussion quantifies that gap. If
    # each panel auto-scaled to its own data the comparison would be
    # visually false: a +0.8 cross-visit bar would render the same length
    # as a +1.5 single-visit bar. Force one symmetric range, driven by the
    # larger pool, onto both axes.
    data_max = max(
        max((abs(v) for v in values_a), default=0.0),
        max((abs(v) for v in values_b), default=0.0),
    )
    shared_lim = data_max * 1.45  # headroom for value annotations
    for ax in axes:
        ax.set_xlim(-shared_lim, shared_lim)
    print(f"  [xaxis] shared range +/-{shared_lim:.2f} "
          f"(max |Sigma IG| = {data_max:.3f}) applied to both panels")

    # ── Bars-to-text ratio: 40/60 split per panel ──────────────────────
    # Bars area takes 40% of each panel column; y-tick labels take 60%.
    # The extra 5% (vs the earlier 35%) gives negative-bar value
    # annotations enough room not to overlap the pattern labels in the
    # tick column. Layout is final after this; tight_layout below would
    # un-do it, so we don't call tight_layout for Figure 4.
    fig.canvas.draw()
    BARS_FRACTION = 0.40
    for ax in axes:
        pos = ax.get_position()
        new_width = pos.width * BARS_FRACTION
        new_x0 = pos.x0 + pos.width * (1 - BARS_FRACTION)
        ax.set_position([new_x0, pos.y0, new_width, pos.height])

    # Column titles drawn via fig.text, left-anchored at the panel's
    # tick-label edge. After the 35/65 split above, the tick-label
    # column starts well to the LEFT of the axes, re-measure via
    # tight bounding box (which now includes the wide tick-label area)
    # to anchor precisely.
    fig.canvas.draw()
    for i, ax in enumerate(axes):
        tbbox = ax.get_tightbbox(fig.canvas.get_renderer())
        tbbox_fig = tbbox.transformed(fig.transFigure.inverted())
        title_x = tbbox_fig.x0
        # Gap between subtitle and panel top, bumped from 0.005 to
        # 0.020 so the subtitle has visible breathing room above the
        # topmost row of pattern labels.
        title_y = tbbox_fig.y1 + 0.020
        fig.text(
            title_x, title_y,
            f"$\\bf{{{PAT_HEADS[i]}.}}$  " + PAT_SUBTITLES[i],
            fontsize=16, ha="left", va="bottom", fontweight="bold",
        )

    # NOTE: no tight_layout call, it would undo the manual
    # set_position() above and reset the 35/65 split.
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(args.output_path) // 1024
    print(f"[3] Saved figure (2-panel pattern-level): "
          f"{args.output_path} ({size_kb} KB)")


if __name__ == "__main__":
    main()
