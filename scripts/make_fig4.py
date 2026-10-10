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


def _wrap_long_label(lab: str, max_chars: int = 200) -> str:
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



def _render_pattern_panels_a4(values_a, labels_a, values_b, labels_b,
                              output_path, dpi=300):
    """Figure 4 at print size for an A4/Letter page (6.5 in wide).

    Panels are stacked (A above B) so long pattern labels fit on one line
    at 7 pt, and both panels share one x-axis range so bar lengths are
    directly comparable across the two pools.
    """
    from matplotlib.ticker import MaxNLocator
    fig_w = 6.5
    fs_label, fs_val, fs_tick, fs_xlab, fs_head, fs_sub = 7, 6.3, 7, 7.5, 9, 8
    pitch = 0.126                      # inches per row
    n_a, n_b = len(values_a), len(values_b)
    head_h, axis_h, gap_h = 0.50, 0.30, 0.30
    xlab_h = 0.42
    h_a, h_b = n_a * pitch, n_b * pitch
    fig_h = head_h + h_a + axis_h + gap_h + head_h + h_b + axis_h + xlab_h
    lab_w, right = 3.25 / fig_w, 0.08 / fig_w
    fig = plt.figure(figsize=(fig_w, fig_h))
    y_b = (xlab_h + axis_h) / fig_h
    y_a = y_b + (h_b + head_h + gap_h + axis_h) / fig_h
    ax_a = fig.add_axes([lab_w, y_a, 1 - lab_w - right, h_a / fig_h])
    ax_b = fig.add_axes([lab_w, y_b, 1 - lab_w - right, h_b / fig_h])
    data_max = max(max(abs(v) for v in values_a), max(abs(v) for v in values_b))
    lim = data_max * 1.32
    panels = [
        (ax_a, values_a, labels_a, "A  All patterns",
         "Top 30 patterns of any shape, ranked by carrier-averaged |\u03a3IG|"),
        (ax_b, values_b, labels_b, "B  Cross-visit patterns",
         "Top 30 patterns spanning \u22652 visits with \u22652 tokens, same ranking"),
    ]
    for ax, vals, labs, _, _ in panels:
        signed_ig_bars(values=vals, labels=labs, ax=ax, fontsize=fs_label,
                       bar_height=0.62)
        pad = data_max * 0.02
        for i, v in enumerate(vals):
            ax.text(v + (pad if v >= 0 else -pad), i, f"{v:+.3f}",
                    ha="left" if v >= 0 else "right", va="center",
                    fontsize=fs_val, color="#222222")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, len(vals) - 0.4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, symmetric=True))
        ax.tick_params(axis="x", labelsize=fs_tick, length=3)
        ax.tick_params(axis="y", length=0, pad=3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax_b.set_xlabel(r"$E[\Sigma\mathrm{IG}\,|\,\mathrm{present}]$"
                    "     \u2190 protective    risk \u2192",
                    fontsize=fs_xlab, labelpad=4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, _, _, head, sub in panels:
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


PATTERN_PANEL_TEXT = [
    ("A  All patterns", "Top 30 of any shape, by carrier-averaged |\u03a3IG|"),
    ("B  Cross-visit patterns", "\u22652 visits, \u22652 tokens; top 30 by |\u03a3IG|"),
]
PATTERN_XLABEL = (r"$E[\Sigma\mathrm{IG}\,|\,\mathrm{present}]$"
                  + "\n\u2190 protective    risk \u2192")


def _render_pattern_panels_side(values_a, labels_a, values_b, labels_b,
                                output_path, dpi=300, wide=False, *,
                                panel_text=None, xlabel=None, geo_override=None):
    """Figure 4 in the two-column geometry of Figure 3 (6.5 in wide).

    ``panel_text`` (two (head, sub) tuples), ``xlabel`` and ``geo_override``
    (keys of ``_two_column_axes``) let the SHAP replication (Figure S12)
    reuse the geometry; the defaults reproduce Figure 4 exactly.
    """
    from matplotlib.ticker import MaxNLocator
    if wide:   # landscape: one-line labels, 9 in wide
        fs_label, fs_val, fs_tick, fs_xlab, fs_head, fs_sub = 7, 6.3, 7, 7.5, 9.5, 8
        geo = dict(fig_w=9.0, lab_w=3.3, bar_w=1.05, gap=0.2, pitch=0.165, head_h=0.72, foot_h=0.66)
        wrap_at = 200
    else:
        fs_label, fs_val, fs_tick, fs_xlab, fs_head, fs_sub = 6.5, 5.8, 6.5, 7, 9, 7.5
        geo = {}
        wrap_at = 30
    if geo_override:
        geo = {**geo, **geo_override}
    (head_a, sub_a), (head_b, sub_b) = panel_text or PATTERN_PANEL_TEXT
    xlabel = xlabel or PATTERN_XLABEL
    n = max(len(values_a), len(values_b))
    fig, ax_a, ax_b, top, fig_h = _two_column_axes(n, **geo)
    data_max = max(max(abs(v) for v in values_a), max(abs(v) for v in values_b))
    lim = data_max * 1.75
    panels = [
        (ax_a, values_a, labels_a, head_a, sub_a),
        (ax_b, values_b, labels_b, head_b, sub_b),
    ]
    for ax, vals, labs, _, _ in panels:
        signed_ig_bars(values=vals, labels=[_wrap_pattern_label(l, wrap_at) for l in labs],
                       ax=ax, fontsize=fs_label, bar_height=0.6)
        pad = data_max * 0.03
        for i, v in enumerate(vals):
            ax.text(v + (pad if v >= 0 else -pad), i, f"{v:+.3f}",
                    ha="left" if v >= 0 else "right", va="center",
                    fontsize=fs_val, color="#222222")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, len(vals) - 0.4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        ax.tick_params(axis="x", labelsize=fs_tick, length=3)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_xlabel(xlabel, fontsize=fs_xlab, labelpad=4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, _, _, head, sub in panels:
        x0 = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0
        fig.text(x0, top + 0.40 / fig_h, head, fontsize=fs_head, fontweight="bold", ha="left", va="bottom")
        fig.text(x0, top + 0.17 / fig_h, sub, fontsize=fs_sub, ha="left", va="bottom", color="#333333")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


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
    p.add_argument("--layout", default="wide", choices=["stacked", "side", "wide"],
                   help="wide: landscape two columns with one-line labels (default); side: portrait two columns with wrapped labels; stacked: A above B.")
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

    if args.layout in ("side", "wide"):
        _render_pattern_panels_side(values_a, labels_a, values_b, labels_b,
                                    args.output_path, args.dpi, wide=(args.layout == "wide"))
    else:
        _render_pattern_panels_a4(values_a, labels_a, values_b, labels_b,
                                  args.output_path, args.dpi)
    size_kb = os.path.getsize(args.output_path) // 1024
    print(f"[3] Saved figure (2-panel pattern-level): "
          f"{args.output_path} ({size_kb} KB)")


if __name__ == "__main__":
    main()
