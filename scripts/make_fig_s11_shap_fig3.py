"""make_fig_s11_shap_fig3 — Figure S12: KernelSHAP replication of Figures 3 and 4.

Renders the SHAP-based counterparts of the IG token panels (Figure 3 A/B)
and pattern panels (Figure 4 A/B) at print size, reusing the main-text
renderers so the supplementary figure reads like the figures it mirrors.
Every value comes from cohort KernelSHAP rather than Integrated Gradients;
the aggregation conventions (population-weighted vs carrier-conditional
means, carrier-averaged pattern sums, Jaccard deduplication) are the same.

Two files are written because 4 x 30 rows do not fit one page at a legible
type size:

  * ``FigureS12a_shap_tokens.png``   (portrait, 6.5 in wide): panels A and B,
    population-weighted and carrier-conditional mean SHAP per token, drawn
    with Figure 3's print geometry;
  * ``FigureS12b_shap_patterns.png`` (landscape, 9.4 in wide): panels C and
    D, carrier-averaged sum of SHAP for all patterns and for cross-visit
    patterns, drawn with Figure 4's landscape geometry.

Inputs (built by ``scripts/build_cohort_shap_cache.py`` then
``scripts/build_shap_aggregations.py``):
  * ``5_analysis/token_importance_shap_<split>.csv``
  * ``7_outputs/panels/panel_c_shap_<split>_j<jt>_short.csv``
  * ``7_outputs/panels/panel_d_shap_<split>_j<jt>_short.csv``

Usage:
    python scripts/make_fig_s11_shap_fig3.py --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from atop.explain.pattern_labels import PatternLabelResolver
from atop.run_layout import RunLayout
from make_fig3 import _render_token_panels_a4
from make_fig4 import _prep_panel, _render_pattern_panels_side


N_SHOW = 30
JACCARD_THRESHOLD = 0.2
MIN_N_PRESENT = 10
SUPP_DIR = os.path.join(HERE, "..", "..", "manuscript", "jamia_submission",
                        "figures", "supplements")
TOKENS_PNG = "FigureS12a_shap_tokens.png"
PATTERNS_PNG = "FigureS12b_shap_patterns.png"

ARROWS = "\n← protective    risk →"
TOKEN_TEXT = [
    ("A  Population-weighted", "Mean SHAP over all patients",
     r"$E[\mathrm{SHAP}]$" + ARROWS),
    ("B  Carrier-conditional", "Mean SHAP among patients with the token",
     r"$E[\mathrm{SHAP}\,|\,\mathrm{present}]$" + ARROWS),
]
PATTERN_TEXT = [
    ("C  All patterns", "Top 30 of any shape, by carrier-averaged |ΣSHAP|"),
    ("D  Cross-visit patterns", "≥2 visits, ≥2 tokens; top 30 by |ΣSHAP|"),
]
PATTERN_XLABEL = r"$E[\Sigma\mathrm{SHAP}\,|\,\mathrm{present}]$" + ARROWS
# Part 1 keeps Figure 3's type sizes; the label column is a little wider
# (1.5 in) for the longer psychiatric and obstetric token names, and the
# figure is drawn a little shorter (7.7 in) so heading + figure fit one page.
TOKEN_GEOMETRY = dict(lab_w_in=1.5, fig_h=7.7)
# Part 2 keeps Figure 4's type sizes; the label column grows to 3.6 in so the
# longest cross-visit SHAP pattern stays on one line, and the row pitch is
# tightened so figure + caption fit one landscape page.
PATTERN_GEOMETRY = dict(fig_w=9.4, lab_w=3.6, bar_w=1.0, pitch=0.135)


def _top_n_by_abs(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """Top-n rows by |col|, sorted ascending for a bottom-to-top bar chart."""
    top = df.reindex(df[col].abs().nlargest(n).index)
    return top.reindex(top[col].abs().sort_values(ascending=True).index)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render Figure S12 (KernelSHAP replication of Figures 3 and 4)")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--n_show", type=int, default=N_SHOW)
    p.add_argument("--jaccard_threshold", type=float, default=JACCARD_THRESHOLD)
    p.add_argument("--min_n_present", type=int, default=MIN_N_PRESENT,
                   help="Drop tokens/patterns with fewer than N sampled carriers "
                        "before ranking (default 10): a KernelSHAP mean over one "
                        "or two patients is not a population estimate.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-dir", default=SUPP_DIR,
                   help=f"Directory for {TOKENS_PNG} and {PATTERNS_PNG}.")
    p.add_argument("--analysis-dir", default=None)
    p.add_argument("--panels-dir", default=None)
    p.add_argument("--short-names-csv",
                   default=os.path.join(HERE, "short_names.csv"))
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    csv_dir = args.analysis_dir or layout.analysis_dir
    panels_dir = args.panels_dir or layout.outputs_panels_dir

    # ── Token-level SHAP (panels A and B) ───────────────────────────
    tok_csv = os.path.join(csv_dir, f"token_importance_shap_{args.split}.csv")
    if not os.path.exists(tok_csv):
        sys.exit(f"  [ERROR] Missing {tok_csv}\n  Run "
                 f"scripts/build_shap_aggregations.py first.")
    df_tok = pd.read_csv(tok_csv).dropna(subset=["ig_mean"]).copy()
    n_before = len(df_tok)
    df_tok = df_tok[df_tok["n_present"] >= args.min_n_present].copy()
    print(f"[1] SHAP token CSV: kept {len(df_tok):,}/{n_before:,} tokens with "
          f"≥ {args.min_n_present} sampled carriers")
    df_tok["shap_pop"] = df_tok["ig_mean"] * df_tok["prevalence"]

    resolver = PatternLabelResolver(args.run_dir, short_names_csv=args.short_names_csv,
                                    verbose=False)
    df_a = _top_n_by_abs(df_tok, "shap_pop", args.n_show)
    df_b = _top_n_by_abs(df_tok, "ig_mean", args.n_show)
    labels_a = [resolver.format_token(t) for t in df_a["token"]]
    labels_b = [resolver.format_token(t) for t in df_b["token"]]
    values_a = list(df_a["shap_pop"])
    values_b = list(df_b["ig_mean"])

    # ── Pattern-level SHAP (panels C and D) ─────────────────────────
    jt_pct = int(round(args.jaccard_threshold * 100))
    panel_dfs = []
    for key in ("c", "d"):
        path = os.path.join(panels_dir, f"panel_{key}_shap_{args.split}_j{jt_pct}_short.csv")
        if not os.path.exists(path):
            sys.exit(f"  [ERROR] Missing {path}\n  Run "
                     f"scripts/build_shap_aggregations.py first.")
        df = pd.read_csv(path)
        kept = df[df["n_present"] >= args.min_n_present].copy()
        print(f"[2] panel_{key}_shap: kept {len(kept)}/{len(df)} patterns")
        panel_dfs.append(kept)
    values_c, labels_c = _prep_panel(panel_dfs[0], args.n_show)
    values_d, labels_d = _prep_panel(panel_dfs[1], args.n_show)

    # ── Render both parts at print size ─────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_tokens = os.path.join(args.output_dir, TOKENS_PNG)
    out_patterns = os.path.join(args.output_dir, PATTERNS_PNG)
    _render_token_panels_a4(values_a, labels_a, values_b, labels_b, out_tokens,
                            args.dpi, panel_text=TOKEN_TEXT,
                            val_fmts=("{:+.4f}", "{:+.3f}"), **TOKEN_GEOMETRY)
    _render_pattern_panels_side(values_c, labels_c, values_d, labels_d, out_patterns,
                                args.dpi, wide=True, panel_text=PATTERN_TEXT,
                                xlabel=PATTERN_XLABEL, geo_override=PATTERN_GEOMETRY)
    from PIL import Image
    for path in (out_tokens, out_patterns):
        with Image.open(path) as im:
            w, h = im.size
        print(f"[3] Saved {path} ({w}×{h} px = {w/args.dpi:.2f}×{h/args.dpi:.2f} in)")

    # ── Cross-method overlap statistics for the caption ─────────────
    ig_tok_csv = os.path.join(csv_dir, f"token_importance_{args.split}.csv")
    if os.path.exists(ig_tok_csv):
        df_ig = pd.read_csv(ig_tok_csv).dropna(subset=["ig_mean"])
        ig_top_b = set(df_ig.reindex(df_ig["ig_mean"].abs().nlargest(args.n_show).index)["token"])
        print(f"\n=== Cross-method overlap (top-{args.n_show}) ===")
        print(f"  tokens, carrier-conditional: {len(ig_top_b & set(df_b['token']))}/{args.n_show}")
    for key, name, shap_df in (("c", "all patterns", panel_dfs[0]),
                               ("d", "cross-visit patterns", panel_dfs[1])):
        ig_path = os.path.join(panels_dir, f"panel_{key}_{args.split}_j{jt_pct}_short.csv")
        if os.path.exists(ig_path):
            ig_set = set(pd.read_csv(ig_path)["pattern"])
            shap_set = set(_top_n_by_abs(shap_df, "ig_value", args.n_show)["pattern"])
            print(f"  {name}: {len(ig_set & shap_set)}/{args.n_show}")


if __name__ == "__main__":
    main()
