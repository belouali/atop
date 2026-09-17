"""make_fig_s11_shap_fig3 \u2014 SHAP-based replication of Fig 3.

Mirrors Fig 3's 4-panel 2\u00d72 layout (atomic top row, compositional bottom
row), but every value comes from cohort KernelSHAP rather than Captum
Integrated Gradients. Demonstrates that AToP's central findings
(carrier-conditioning surfaces rare-but-decisive features; multi-token
pattern aggregation surfaces compositional phenotypes) are properties of
the **aggregation convention**, not the per-instance attribution method
\u2014 the same patterns surface under SHAP.

Inputs (built by ``scripts/build_cohort_shap_cache.py`` then
``scripts/build_shap_aggregations.py``):
  * ``5_analysis/token_importance_shap_<split>.csv`` \u2014 per-token mean
    SHAP across sampled cohort.
  * ``7_outputs/panels/panel_c_shap_<split>_j<jt>_short.csv`` \u2014 full-pool
    patterns by carrier-averaged \u03A3SHAP.
  * ``7_outputs/panels/panel_d_shap_<split>_j<jt>_short.csv`` \u2014 cross-
    visit patterns by carrier-averaged \u03A3SHAP.

The rendering code is structurally identical to scripts/make_fig3.py
(stitched 2x2 grid; pinned axes geometry; ``atomic_split`` xlim mode by
default so Panel A's tiny population-weighted SHAP magnitudes don't
swamp the comparison).

Usage:
    python scripts/make_fig_s11_shap_fig3.py --run_dir ../runs/full_CPD_v2
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
from atop.run_layout import RunLayout


# ── Defaults. readable 2×2 layout balanced against PIL's ~179M-pixel
# decompression-bomb safety limit (final image must fit under it for the
# PDF builder). At 300 dpi, fig_width=42, panel_height=14.5 gives a 4-
# panel stitch around 12600×8700 = 110M pixels, safely under the limit.
# fontsize=24 keeps labels readable when scaled to journal column width.
N_SHOW = 30
JACCARD_THRESHOLD = 0.2
FIG_WIDTH = 42
FONTSIZE = 24


def _bold(s: str) -> str:
    return r"$\bf{" + s.replace(" ", r"\ ").replace(":", r"{:}") + r"}$"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render Fig S11 (SHAP-based replication of Fig 3)")
    p.add_argument("--run_dir", default="../runs/full_CPD_v2")
    p.add_argument("--split", default="test")
    p.add_argument("--n_show", type=int, default=N_SHOW)
    p.add_argument("--jaccard_threshold", type=float, default=JACCARD_THRESHOLD)
    p.add_argument("--xlim_mode", default="atomic_split",
                   choices=["global", "row", "atomic_split", "per_panel"],
                   help="Same semantics as make_fig3.py.")
    p.add_argument("--min_n_present", type=int, default=10,
                   help="Drop tokens/patterns with fewer than N sampled carriers "
                        "before ranking (default 10). KernelSHAP \"mean\" is "
                        "undefined for rare tokens with only 1\u20132 sampled "
                        "carriers \u2014 the SHAP value of a single patient is not "
                        "a population estimate. Set to 1 to disable the filter "
                        "(produces noisy rankings dominated by single-instance "
                        "tokens).")
    p.add_argument("--fig_width", type=float, default=FIG_WIDTH)
    p.add_argument("--fontsize", type=int, default=FONTSIZE)
    p.add_argument("--stitch_gap_px", type=int, default=40)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/"
                           "FigureS11_shap_fig3_variant.png")
    p.add_argument("--analysis-dir", default=None)
    p.add_argument("--panels-dir", default=None)
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    csv_dir = args.analysis_dir or layout.analysis_dir
    panels_dir = args.panels_dir or layout.outputs_panels_dir

    # ── Load the SHAP-aggregated CSVs ───────────────────────────────
    tok_csv = os.path.join(csv_dir, f"token_importance_shap_{args.split}.csv")
    if not os.path.exists(tok_csv):
        sys.exit(f"  [ERROR] Missing {tok_csv}\n  Run "
                 f"scripts/build_shap_aggregations.py first.")
    df_tok = pd.read_csv(tok_csv)
    print(f"[1] SHAP token CSV: {len(df_tok):,} tokens "
          f"({os.path.basename(tok_csv)})")

    jt_pct = int(args.jaccard_threshold * 100)
    panel_c_path = os.path.join(panels_dir,
                                  f"panel_c_shap_{args.split}_j{jt_pct}_short.csv")
    panel_d_path = os.path.join(panels_dir,
                                  f"panel_d_shap_{args.split}_j{jt_pct}_short.csv")
    if not (os.path.exists(panel_c_path) and os.path.exists(panel_d_path)):
        sys.exit(f"  [ERROR] Missing panel CSVs:\n    {panel_c_path}\n    "
                 f"{panel_d_path}\n  Run scripts/build_shap_aggregations.py "
                 f"first.")
    panel_c_df = pd.read_csv(panel_c_path)
    panel_d_df = pd.read_csv(panel_d_path)
    print(f"[2] SHAP Panel C: {len(panel_c_df)} patterns")
    print(f"    SHAP Panel D: {len(panel_d_df)} patterns")

    # ── Resolver (canonical labels) ─────────────────────────────────
    resolver = PatternLabelResolver(args.run_dir, verbose=False)

    # ── Panels A + B (atomic): population-weighted vs carrier ──────
    df_ig_tok = df_tok.dropna(subset=["ig_mean"]).copy()
    # Apply min-carrier filter \u2014 KernelSHAP "mean" with 1-2 sampled carriers
    # is one patient's SHAP value, not a population estimate. Filtering keeps
    # rankings statistically meaningful.
    n_before = len(df_ig_tok)
    df_ig_tok = df_ig_tok[df_ig_tok["n_present"] >= args.min_n_present].copy()
    print(f"  [filter] kept {len(df_ig_tok):,}/{n_before:,} tokens with "
          f"\u2265 {args.min_n_present} sampled carriers")
    df_ig_tok["ig_pop"] = df_ig_tok["ig_mean"] * df_ig_tok["prevalence"]
    df_ig_tok["abs_carrier"] = df_ig_tok["ig_mean"].abs()
    df_ig_tok["abs_pop"]     = df_ig_tok["ig_pop"].abs()

    # Panel A = population-weighted top-N
    df_a = df_ig_tok.nlargest(args.n_show, "abs_pop").sort_values("abs_pop")
    labels_a = [resolver.format_token(t) for t in df_a["token"]]
    values_a = list(df_a["ig_pop"])

    # Panel B = carrier-conditional top-N
    df_b = df_ig_tok.nlargest(args.n_show, "abs_carrier").sort_values("abs_carrier")
    labels_b = [resolver.format_token(t) for t in df_b["token"]]
    values_b = list(df_b["ig_mean"])

    # Panel C / D: apply same min-carrier filter at pattern level.
    n_c_before, n_d_before = len(panel_c_df), len(panel_d_df)
    panel_c_df = panel_c_df[panel_c_df["n_present"] >= args.min_n_present].copy()
    panel_d_df = panel_d_df[panel_d_df["n_present"] >= args.min_n_present].copy()
    print(f"  [filter] kept Panel C: {len(panel_c_df)}/{n_c_before}  "
          f"Panel D: {len(panel_d_df)}/{n_d_before}  patterns "
          f"(\u2265 {args.min_n_present} sampled carriers)")
    # Sort ascending for display
    panel_c_df = panel_c_df.sort_values("ig_value", key=abs, ascending=True)
    panel_d_df = panel_d_df.sort_values("ig_value", key=abs, ascending=True)
    panel_c_labels = [resolver.format_pattern(p, short_names=True)
                       for p in panel_c_df["pattern"]]
    panel_d_labels = [resolver.format_pattern(p, short_names=True)
                       for p in panel_d_df["pattern"]]
    panel_c_values = list(panel_c_df["ig_value"])
    panel_d_values = list(panel_d_df["ig_value"])

    # ── Layout constants (same as make_fig3 stitch) ─────────────────
    panel_height = max(12, args.n_show * 0.50)   # taller per-row → labels breathe
    fs_title = args.fontsize + 2
    fs_xlabel = args.fontsize - 1

    # Title pad to uniform line count
    panel_titles = [
        ("A. Population SHAP (atomic):",
         "Mean SHAP across sampled patients"),
        ("B. Carrier-conditional SHAP (atomic):",
         "Mean SHAP among sampled carriers"),
        ("C. Pattern SHAP (no shape minimums):",
         f"Carrier-averaged \u03A3SHAP, all pattern shapes\nJaccard \u2265 {args.jaccard_threshold} dedup"),
        ("D. Cross-visit pattern SHAP:",
         f"Carrier-averaged \u03A3SHAP, \u22652 visits & \u22652 tokens\nJaccard \u2265 {args.jaccard_threshold} dedup"),
    ]
    max_sub_lines = max(s.count("\n") + 1 for _, s in panel_titles)
    panel_titles = [
        (h, s + ("\n " * (max_sub_lines - s.count("\n") - 1)))
        for h, s in panel_titles
    ]

    panel_specs = [
        dict(values=values_a, labels=labels_a, title=panel_titles[0],
             xlabel="E[SHAP] across sampled patients\n\u2190 protective    risk \u2192"),
        dict(values=values_b, labels=labels_b, title=panel_titles[1],
             xlabel="E[SHAP | present]\n\u2190 protective    risk \u2192"),
        dict(values=panel_c_values, labels=panel_c_labels, title=panel_titles[2],
             xlabel="E[\u03A3SHAP | present]\n\u2190 protective    risk \u2192"),
        dict(values=panel_d_values, labels=panel_d_labels, title=panel_titles[3],
             xlabel="E[\u03A3SHAP | present]\n\u2190 protective    risk \u2192"),
    ]

    # ── X-limit unification (same logic as make_fig3) ───────────────
    def _max_abs(*spec_lists):
        vals = [v for sl in spec_lists for s in sl for v in s["values"]]
        return max(abs(v) for v in vals) if vals else 1.0

    mode = args.xlim_mode
    if mode == "global":
        m = _max_abs(panel_specs)
        panel_xlims = [(-m * 1.05, m * 1.05)] * 4
    elif mode == "row":
        mt = _max_abs(panel_specs[:2]); mb = _max_abs(panel_specs[2:])
        panel_xlims = [(-mt * 1.05, mt * 1.05)] * 2 + \
                      [(-mb * 1.05, mb * 1.05)] * 2
    elif mode == "atomic_split":
        ma = _max_abs(panel_specs[:1])
        m_rest = _max_abs(panel_specs[1:])
        panel_xlims = [(-ma * 1.05, ma * 1.05)] + \
                      [(-m_rest * 1.05, m_rest * 1.05)] * 3
    else:
        panel_xlims = [None] * 4

    # ── Stitched 2x2 render ─────────────────────────────────────────
    from PIL import Image
    panel_fig_width = args.fig_width / 2
    axes_left, axes_right = 0.77, 0.97
    axes_bottom, axes_top = 0.11, 0.84
    axes_w = axes_right - axes_left
    axes_h = axes_top - axes_bottom

    tmp_paths = []
    for i, spec in enumerate(panel_specs):
        fig_i, ax_i = plt.subplots(figsize=(panel_fig_width, panel_height))
        signed_ig_bars(values=spec["values"], labels=spec["labels"],
                       ax=ax_i, fontsize=args.fontsize)
        ax_i.set_xlabel(spec["xlabel"], fontsize=fs_xlabel)
        ax_i.tick_params(axis="x", labelsize=fs_xlabel - 1)
        if panel_xlims[i] is not None:
            ax_i.set_xlim(*panel_xlims[i])
        ax_i.set_position([axes_left, axes_bottom, axes_w, axes_h])

        head, sub = spec["title"]
        head_y = 0.965
        # Derive the title->subtitle gap from the actual type size rather than
        # a fixed figure-fraction. The previous constant (0.018) was ~19 pt on
        # a 15-inch panel while the title sets at fs_title (26 pt), so the
        # subtitle's ascenders collided with the title's descenders.
        line_frac = (fs_title * 1.45 / 72.0) / panel_height
        sub_y = head_y - line_frac * (1 + head.count("\n"))
        fig_i.text(0.5, head_y, head, ha="center", va="top",
                    fontsize=fs_title, fontweight="bold", color="#111111")
        fig_i.text(0.5, sub_y, sub, ha="center", va="top",
                    fontsize=fs_title - 1, color="#444444")

        tmp = tempfile.NamedTemporaryFile(
            prefix=f"fig_s11_{chr(ord('a')+i)}_", suffix=".png", delete=False)
        fig_i.savefig(tmp.name, dpi=args.dpi)
        plt.close(fig_i)
        tmp_paths.append(tmp.name)

    images = [Image.open(p) for p in tmp_paths]
    sizes = {im.size for im in images}
    if len(sizes) != 1:
        print(f"  [WARN] panel sizes differ: {sizes}")
    cell_w, cell_h = images[0].size
    col_gap = args.stitch_gap_px
    row_gap = args.stitch_gap_px
    final_w = 2 * cell_w + col_gap
    final_h = 2 * cell_h + row_gap
    composite = Image.new("RGB", (final_w, final_h), "white")
    composite.paste(images[0], (0, 0))
    composite.paste(images[1], (cell_w + col_gap, 0))
    composite.paste(images[2], (0, cell_h + row_gap))
    composite.paste(images[3], (cell_w + col_gap, cell_h + row_gap))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    composite.save(args.output_path)
    for tp in tmp_paths:
        try: os.unlink(tp)
        except OSError: pass

    size_kb = os.path.getsize(args.output_path) // 1024
    print(f"\n[3] Saved figure: {args.output_path} "
          f"({final_w}\u00d7{final_h} px, {size_kb} KB)")

    # ── Cross-method overlap stat for the caption ────────────────────
    ig_tok_csv = os.path.join(csv_dir, f"token_importance_{args.split}.csv")
    if os.path.exists(ig_tok_csv):
        df_ig = pd.read_csv(ig_tok_csv).dropna(subset=["ig_mean"]).copy()
        df_ig["abs_carrier"] = df_ig["ig_mean"].abs()
        ig_top_b = set(df_ig.nlargest(args.n_show, "abs_carrier")["token"])
        shap_top_b = set(df_b["token"])
        overlap_b = ig_top_b & shap_top_b
        print(f"\n=== Cross-method overlap (Panel B, top-{args.n_show}) ===")
        print(f"  IG carrier-top-{args.n_show} \u2229 SHAP carrier-top-{args.n_show}: "
              f"{len(overlap_b)}/{args.n_show}")


if __name__ == "__main__":
    main()
