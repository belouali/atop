"""make_fig_s7_jaccard_sensitivity \u2014 Panel C at 6 Jaccard thresholds (2x3 grid).

Defends the locked j=0.2 choice by showing Panel C at a monotonic sweep of
Jaccard thresholds: (no dedup, 0.8, 0.6, 0.4, 0.2, 0.1). Each sub-panel
shows the top-30 patterns by |\u03A3IG| within the representative set at that
threshold.

Reading the grid left-to-right, top-to-bottom:
  \u2022 at very loose thresholds (no dedup, j=0.8) the top-30 is overwhelmed
    by near-duplicate Live-birth / MDD\u2013SI variants
  \u2022 as the threshold tightens (j=0.6 \u2192 0.4 \u2192 0.2) phenotypes emerge and
    the top-30 increasingly spans distinct clinical trajectories
  \u2022 at j=0.1 (tighter than locked) dedup is even more aggressive but the
    same five phenotypes still fit \u2014 i.e. j=0.2 is a conservative choice
    within the regime where all phenotypes are preserved.

Narrative for the supplement caption:

    Panel C selection is sensitive to the Jaccard deduplication threshold
    used to collapse patterns with overlapping patient carriers. We chose
    j = 0.2 \u2014 aggressive enough that all five clinical phenotypes
    (obstetric, psychiatric, oncologic, chest-pain-workup, hepatic)
    fit in the top-30, but not so aggressive that within-phenotype
    variants are over-collapsed. Looser thresholds (j \u2265 0.5) leave the
    top-30 dominated by near-duplicate obstetric variants; tighter
    thresholds (j = 0.1) preserve the same phenotype coverage as 0.2.

Usage:
    python scripts/make_fig_s7_jaccard_sensitivity.py \
        --run_dir ../runs/full_CPD_v2
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.plots import signed_ig_bars
from atop.explain.pattern_labels import PatternLabelResolver
from atop.explain.jaccard import get_jaccard_representatives
from atop.run_layout import RunLayout


N_SHOW = 30
FONTSIZE = 9      # page-scaled: the old 16 was sized for a 48-inch canvas
                  # that then got downscaled ~6x, leaving ~2 pt on the page


def _bold(s: str) -> str:
    return r"$\bf{" + s.replace(" ", r"\ ").replace(":", r"{:}") + r"}$"


def _select_top_n(df_pw: pd.DataFrame, rep_patterns: set, n: int) -> pd.DataFrame:
    """Top-n by |ig_signed_mean| among the representative set, ordered
    asc by |\u03A3IG| for display (largest magnitude at top of chart)."""
    reps = df_pw[df_pw["pattern"].isin(rep_patterns)].copy()
    reps["_abs"] = reps["ig_signed_mean"].abs()
    return reps.nlargest(n, "_abs").sort_values("_abs", ascending=True).drop(columns="_abs")


def _render_panel(df_sel: pd.DataFrame, resolver: PatternLabelResolver,
                  threshold_label: str, n_reps: int, panel_letter: str,
                  fontsize: int, panel_height: float,
                  panel_fig_width: float, target_bar_inches: float,
                  tmp_dir: str) -> str:
    """Render a single Panel-C-style figure and save to a tmp PNG."""
    labels = [resolver.format_pattern(p, short_names=True)
              for p in df_sel["pattern"]]
    values = list(df_sel["ig_signed_mean"])

    fig, ax = plt.subplots(figsize=(panel_fig_width, panel_height))
    signed_ig_bars(values=values, labels=labels, ax=ax, fontsize=fontsize)

    fs_title = fontsize + 2
    fs_xlabel = fontsize - 1
    sub = f"top-{len(df_sel)} patterns from {n_reps:,} representative{'s' if n_reps != 1 else ''}"
    head = f"{panel_letter}. {threshold_label}"
    ax.set_title(_bold(head) + "\n" + sub, fontsize=fs_title, pad=15)
    ax.set_xlabel("E[\u03A3IG | present]\n\u2190 protective    risk \u2192",
                   fontsize=fs_xlabel)
    ax.tick_params(axis="x", labelsize=fs_xlabel - 1)

    fig.tight_layout()
    # Pin the bar-area width in inches so all 4 panels visually match after stitch.
    pos = ax.get_position()
    target_bar_frac = target_bar_inches / panel_fig_width
    new_right = pos.x1
    new_left = new_right - target_bar_frac
    ax.set_position([new_left, pos.y0, target_bar_frac, pos.height])

    tmp = tempfile.NamedTemporaryFile(prefix=f"fig_s7_{panel_letter}_",
                                       suffix=".png", dir=tmp_dir, delete=False)
    fig.savefig(tmp.name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def main() -> None:
    p = argparse.ArgumentParser(description="Render Fig S7 (Jaccard sensitivity).")
    p.add_argument("--run_dir", default="../runs/full_CPD_v2")
    p.add_argument("--split", default="test")
    p.add_argument("--n_show", type=int, default=N_SHOW)
    p.add_argument("--fontsize", type=int, default=FONTSIZE)
    p.add_argument("--fig_width", type=float, default=13.0,
                   help="Total figure width in inches (divided across 3 columns).")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/FigureS7_jaccard_sensitivity.png")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    csv_dir = layout.analysis_dir

    # Load the multi-token pathway CSV (Panel C source).
    pw_csv = layout.pathway_csv(args.split)
    if not os.path.exists(pw_csv):
        sys.exit(f"  [ERROR] Missing {pw_csv}")
    df_pw_full = pd.read_csv(pw_csv)
    df_pw_with_ck = df_pw_full[df_pw_full["n_present"] >= 3].copy()
    total_patterns = len(df_pw_with_ck)
    print(f"[1] Multi-token patterns (n_present \u2265 3): {total_patterns:,}")

    # ── Assemble representative sets for six panels (2x3 grid) ───────
    # Reading order top-left \u2192 bottom-right follows a monotonic sweep
    # from "no dedup" \u2192 tightest aggressive threshold. Top row shows
    # loose thresholds where near-duplicates dominate; bottom row shows
    # tight thresholds where phenotype coverage emerges.
    print("\n[2] Building representative sets at each threshold\u2026")
    threshold_specs = [
        # (panel letter, display label, threshold_or_None)
        ("A", "No Jaccard dedup",                              None),
        ("B", "Jaccard \u2265 0.8 dedup (near-identity merges only)", 0.8),
        ("C", "Jaccard \u2265 0.6 dedup",                      0.6),
        ("D", "Jaccard \u2265 0.4 dedup",                      0.4),
        ("E", "Jaccard \u2265 0.2 dedup (locked \u2014 main paper)", 0.2),
        ("F", "Jaccard \u2265 0.1 dedup (tighter than locked)", 0.1),
    ]
    rep_sets = []
    for letter, label, thresh in threshold_specs:
        if thresh is None:
            reps = set(df_pw_with_ck["pattern"].tolist())
        else:
            reps = get_jaccard_representatives(csv_dir, args.split, thresh)
            if reps is None:
                sys.exit(f"  [ERROR] Could not obtain reps at j={thresh}")
        print(f"    {letter}: {label:<55s} \u2192 {len(reps):>6,} representatives")
        rep_sets.append((letter, label, reps))

    # ── Render each panel to a tmp PNG ────────────────────────────────
    resolver = PatternLabelResolver(args.run_dir, verbose=False)
    panel_fig_width = args.fig_width / 2  # 2 columns (A,C / E,F)
    # Row pitch drives legibility more than font size does: at ~0.15 in/row
    # the 30 labels nearly touch. 0.26 in/row gives clear separation between
    # features at the cost of a taller panel.
    panel_height = max(5.0, args.n_show * 0.26)
    target_bar_inches = panel_fig_width * 0.40

    # B and D are omitted from the rendered figure: their top-30 sets are
    # near-identical to C, so they cost width without adding information.
    # All six thresholds remain in the backing CSV.
    SHOW = ["A", "C", "E", "F"]
    render_sets = [rs for rs in rep_sets if rs[0] in SHOW]
    print(f"\n[3] Rendering {len(render_sets)} panels (top-{args.n_show} each)\u2026")
    tmp_dir = tempfile.mkdtemp(prefix="fig_s7_")
    tmp_paths = []
    for letter, label, reps in render_sets:
        df_sel = _select_top_n(df_pw_with_ck, reps, args.n_show)
        path = _render_panel(
            df_sel, resolver, label, len(reps), letter,
            args.fontsize, panel_height, panel_fig_width, target_bar_inches,
            tmp_dir)
        tmp_paths.append(path)
        print(f"    {letter}: rendered \u2192 {os.path.basename(path)}")

    # ── Stitch into 2x3 grid ──────────────────────────────────────────
    # Row 1: A B C   Row 2: D E F
    print("\n[4] Stitching composite figure (2x3 grid)\u2026")
    from PIL import Image
    images = [Image.open(p) for p in tmp_paths]

    # Normalize each panel to the same height (tallest wins).
    target_h = max(im.height for im in images)
    resized = []
    for im in images:
        if im.height != target_h:
            new_w = int(im.width * target_h / im.height)
            im = im.resize((new_w, target_h), Image.LANCZOS)
        resized.append(im)

    row_gap_px = 80   # vertical space between rows
    col_gap_px = 40   # horizontal space between columns

    def _build_row(ims):
        total_w = sum(im.width for im in ims) + col_gap_px * (len(ims) - 1)
        row = Image.new("RGB", (total_w, target_h), "white")
        x = 0
        for im in ims:
            row.paste(im, (x, 0))
            x += im.width + col_gap_px
        return row

    row1 = _build_row(resized[:2])
    row2 = _build_row(resized[2:])
    final_w = max(row1.width, row2.width)
    body_h = row1.height + row2.height + row_gap_px

    # Header band with a self-explanatory suptitle stating the pool inclusion
    # criteria. Reviewers should be able to identify what's in this figure
    # without cross-referencing the README.
    from PIL import ImageDraw, ImageFont
    header_h = 180
    title_text = (
        "Jaccard sensitivity \u2014 cross-visit multi-token patterns only "
        "(locked Panel C source)"
    )
    subtitle_text = (
        "Patterns spanning \u22652 visits with \u22652 tokens "
        "(min_steps=2, min_tokens=2). "
        "Excludes single-visit-only and single-token patterns. "
        "Each panel = top-30 by |\u03A3IG| within the Jaccard-deduplicated representative set."
    )
    composite = Image.new("RGB", (final_w, body_h + header_h), "white")
    composite.paste(row1, ((final_w - row1.width) // 2, header_h))
    composite.paste(row2, ((final_w - row2.width) // 2,
                            header_h + row1.height + row_gap_px))

    draw = ImageDraw.Draw(composite)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 56)
        sub_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 36)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    draw.text((final_w // 2, 36), title_text, fill="#111111",
              font=title_font, anchor="mt")
    draw.text((final_w // 2, 110), subtitle_text, fill="#444444",
              font=sub_font, anchor="mt")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    composite.save(args.output_path)
    size_kb = os.path.getsize(args.output_path) // 1024
    print(f"[5] Saved figure: {args.output_path} ({size_kb} KB)")

    # Clean up tmp panel PNGs
    for p in tmp_paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # ── Backing CSV (long form: threshold, rank, pattern, IG) ─────────
    rows = []
    for letter, label, reps in rep_sets:
        df_sel = _select_top_n(df_pw_with_ck, reps, args.n_show).iloc[::-1]  # rank 1 first
        for rank, (_, r) in enumerate(df_sel.iterrows(), start=1):
            rows.append({
                "panel":           letter,
                "threshold_label": label,
                "n_representatives": len(reps),
                "rank":            rank,
                "pattern":         r["pattern"],
                "pattern_readable": resolver.format_pattern(r["pattern"], short_names=True),
                "ig_signed_mean":  r["ig_signed_mean"],
                "n_present":       int(r.get("n_present", 0)),
            })
    out_csv = os.path.splitext(args.output_path)[0] + ".csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"    csv: {out_csv}")


if __name__ == "__main__":
    main()
