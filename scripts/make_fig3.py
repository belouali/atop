"""make_fig3, render Figure 3 (3-panel global attribution).

Three complementary views of test-set attribution, all sharing the same
horizontal bar grammar (``atop.plots.signed_ig_bars``):

  Panel A, Population-weighted IG  E[IG] = ig_carrier × prevalence
  Panel B, Carrier-conditional IG  E[IG | present]   with n_present
  Panel C, Carrier-conditional ΣIG over Jaccard-deduped temporal patterns

Outputs:
  - ``Figure3.png`` (or wherever ``--output-path`` points)
  - ``panel_a_<split>_j<pct>_short.csv``  (token-level)
  - ``panel_b_<split>_j<pct>_short.csv``  (token-level + n_present)
  - ``panel_c_<split>_j<pct>_short.csv``  (pattern-level, INCLUDES raw
    ``pattern`` column so make_fig4 can join with validation deltas)
  - ``jaccard_reps_<split>_j<pct>.pkl``   (cached on first run; reused after)

Usage:
    python scripts/make_fig3.py --run_dir ../runs/full_CPD_v1
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
from atop.explain.pattern_labels import PatternLabelResolver
from atop.explain.jaccard import get_jaccard_representatives
from atop.explain.cooccurrence import score_all_raw_patterns, cluster_jaccard_inmemory
from atop.run_layout import RunLayout


# ── Defaults (locked in TASKS/_shared_context.md) ────────────────────────
N_SHOW = 15
JACCARD_THRESHOLD = 0.2
PANEL_C_MODE = "top"     # "top" = ranked by |ΣIG|, "balanced" = N/2 risk + N/2 prot
FIG_WIDTH = 54
FONTSIZE = 18
SHOW_N = False           # show "(n=X)" suffix on Panel B/C labels


def _load_or_score_full_pool(layout: RunLayout, split: str,
                              min_carriers: int = 5,
                              verbose: bool = True) -> pd.DataFrame:
    """Load the full-pool scored DataFrame for Panel C, or compute + cache it.

    Cache: ``<analysis_dir>/full_pool_scored_<split>.pkl``. Invalidated when
    the raw mining cache (``raw_episode_*.pkl``) is newer than the cache.
    """
    import pickle
    cache_path = os.path.join(layout.analysis_dir,
                               f"full_pool_scored_{split}.pkl")
    raw_files = sorted([f for f in os.listdir(layout.mining_dir)
                        if f.startswith("raw_episode_")])
    if not raw_files:
        sys.exit(f"  [ERROR] No raw_episode_*.pkl in {layout.mining_dir}")
    raw_path = os.path.join(layout.mining_dir, raw_files[-1])

    if os.path.exists(cache_path) and \
            os.path.getmtime(cache_path) >= os.path.getmtime(raw_path):
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        if verbose:
            print(f"  [Panel C] cached full pool \u2192 {len(df):,} patterns "
                  f"({os.path.basename(cache_path)})")
        return df

    if verbose:
        print(f"  [Panel C] computing full pool from raw cache "
              f"(no minimums; min_carriers={min_carriers}, vectorized \u2014 ~30 s)\u2026")
    with open(raw_path, "rb") as f:
        raw_mining = pickle.load(f)
    with open(layout.sequences_pkl(split), "rb") as f:
        df_seq = pickle.load(f)
    df_ig = pd.read_csv(layout.ig_csv(split))
    df = score_all_raw_patterns(
        raw_mining, df_seq, df_ig,
        min_carriers=min_carriers, min_blocks=1, min_tokens_total=1,
        verbose=verbose,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    if verbose:
        print(f"  [Panel C] cached \u2192 {cache_path}")
    return df


def _select_panel_c_patterns(
    df_pw: pd.DataFrame,
    rep_patterns: set,
    n_show: int,
    mode: str = "top",
) -> pd.DataFrame:
    """Pick the patterns to display in Panel C from the Jaccard reps."""
    df_reps = df_pw[df_pw["pattern"].isin(rep_patterns)].copy()
    df_reps["_abs"] = df_reps["ig_signed_mean"].abs()
    if mode == "top":
        top = df_reps.nlargest(n_show, "_abs").sort_values("_abs", ascending=True)
    else:  # balanced
        half = max(1, n_show // 2)
        df_r = df_reps[df_reps["ig_signed_mean"] > 0].nlargest(half, "ig_signed_mean")
        df_p = df_reps[df_reps["ig_signed_mean"] < 0].nsmallest(half, "ig_signed_mean")
        top = pd.concat([df_r, df_p]).sort_values("ig_signed_mean", key=abs, ascending=True)
    return top.drop(columns=["_abs"])



def _render_token_panels_a4(values_a, labels_a, values_b, labels_b,
                            output_path, dpi=300):
    """Figure 3 at print size for an A4/Letter text block (6.5 in wide).

    Drawn at the size it is printed, so fonts stay legible (7.5 pt labels,
    9 pt titles) instead of being shrunk from a poster-sized canvas. Each
    panel gets a short two-line title anchored over its own label column.
    """
    from matplotlib.ticker import MaxNLocator
    fig_w, fig_h = 6.5, 8.4
    fs_label, fs_val, fs_tick, fs_xlab, fs_head, fs_sub = 7.5, 6.5, 7, 7.5, 9, 8
    lab_w, gap, right = 1.42 / fig_w, 0.30 / fig_w, 0.06 / fig_w
    bar_w = (1 - 2 * lab_w - gap - right) / 2
    top, bottom = 1 - 0.78 / fig_h, 0.80 / fig_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_a = fig.add_axes([lab_w, bottom, bar_w, top - bottom])
    ax_b = fig.add_axes([lab_w + bar_w + gap + lab_w, bottom, bar_w, top - bottom])
    panels = [
        (ax_a, values_a, labels_a, "A  Population-weighted",
         "Mean IG over all patients",
         r"$E[\mathrm{IG}]$" + "\n\u2190 protective    risk \u2192"),
        (ax_b, values_b, labels_b, "B  Carrier-conditional",
         "Mean IG among patients with the token",
         r"$E[\mathrm{IG}\,|\,\mathrm{present}]$" + "\n\u2190 protective    risk \u2192"),
    ]
    for ax, vals, labs, _, _, xlab in panels:
        signed_ig_bars(values=vals, labels=labs, ax=ax, fontsize=fs_label,
                       bar_height=0.62)
        m = max(abs(v) for v in vals)
        pad = m * 0.03
        for i, v in enumerate(vals):
            ax.text(v + (pad if v >= 0 else -pad), i, f"{v:+.3f}",
                    ha="left" if v >= 0 else "right", va="center",
                    fontsize=fs_val, color="#222222")
        ax.set_xlim(-m * 1.6, m * 1.6)
        ax.set_ylim(-0.6, len(vals) - 0.4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        ax.tick_params(axis="x", labelsize=fs_tick, length=3)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_xlabel(xlab, fontsize=fs_xlab, labelpad=4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, _, _, head, sub, _ in panels:
        x0 = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0
        fig.text(x0, top + 0.40 / fig_h, head, fontsize=fs_head,
                 fontweight="bold", ha="left", va="bottom")
        fig.text(x0, top + 0.17 / fig_h, sub, fontsize=fs_sub,
                 ha="left", va="bottom", color="#333333")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure 3 (global 3-panel attribution)")
    p.add_argument("--run_dir", default="../runs/full_CPD_v1")
    p.add_argument("--split", default="test")
    p.add_argument("--n_show", type=int, default=N_SHOW)
    p.add_argument("--jaccard_threshold", type=float, default=JACCARD_THRESHOLD)
    p.add_argument("--panel_c_mode", default=PANEL_C_MODE, choices=["top", "balanced"])
    p.add_argument("--fig_width", type=float, default=FIG_WIDTH)
    p.add_argument("--fontsize", type=int, default=FONTSIZE)
    p.add_argument("--show_n", action="store_true", default=SHOW_N)
    p.add_argument("--bar_squeeze", type=float, default=0.55)
    p.add_argument("--xlim_mode",
                   default="atomic_split",
                   choices=["global", "row", "atomic_split", "per_panel"],
                   help="How panels share x-axis ranges. Default "
                        "\"atomic_split\" = Panel A on its own scale "
                        "(population-weighted IG is \u224850\u00d7 smaller magnitude "
                        "than carrier-conditional metrics; forcing it onto a "
                        "shared axis makes its bars invisible) + B/C/D share "
                        "one xlim so token, full-pool-pattern, and cross-visit-"
                        "pattern aggregations are directly comparable. Other "
                        "modes: \"global\" (all 4 share one xlim \u2014 Panel A "
                        "invisible), \"row\" (top A,B share / bottom C,D share), "
                        "\"per_panel\" (each auto-fitted, legacy).")
    p.add_argument("--no_jaccard", action="store_true",
                   help="Skip Jaccard deduplication for Panel C \u2014 show the "
                        "raw top-N multi-token patterns by |\u03A3IG| directly "
                        "(likely dominated by near-duplicate variants of the "
                        "strongest phenotypes, e.g. many Live-birth patterns).")
    p.add_argument("--stitch", action="store_true",
                   help="Render each panel as its own figure at full width, then "
                        "composite horizontally with PIL. Keeps all three panels "
                        "at identical fontsize (tick labels render within each "
                        "panel's native label area, so nothing collides). Default "
                        "off \u2014 when off, uses the single-figure matplotlib "
                        "subplots layout.")
    p.add_argument("--stitch_gap_px", type=int, default=8,
                   help="Horizontal/vertical gap (px) between panels in "
                        "stitch mode. Default 8 (was 40, left huge "
                        "between-panel whitespace requiring post-hoc "
                        "PNG cropping).")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--render_only", action="store_true",
                   help="Redraw Figure 3 from the saved panel_a/panel_b CSVs "
                        "(no recomputation; panel C/D CSVs are left untouched).")
    p.add_argument("--output-path", default="../manuscript/jamia_submission/figures/Figure3.png")
    p.add_argument("--analysis-dir", default=None,
                   help="Directory holding token_importance_*.csv, pathway_importance_*.csv, "
                        "carrier_sets_*.pkl etc. (default: <run_dir>/5_analysis)")
    p.add_argument("--panels-dir", default=None,
                   help="Directory for panel_a/b/c CSVs (default: <run_dir>/7_outputs/panels)")
    p.add_argument("--short-names-csv",
                   default=os.path.join(os.path.dirname(__file__), "short_names.csv"))
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    csv_dir = args.analysis_dir or layout.analysis_dir
    panels_dir = args.panels_dir or layout.outputs_panels_dir

    if args.render_only:
        jt = int(round(args.jaccard_threshold * 100))
        pa = pd.read_csv(os.path.join(panels_dir, f"panel_a_{args.split}_j{jt}_short.csv"))
        pb = pd.read_csv(os.path.join(panels_dir, f"panel_b_{args.split}_j{jt}_short.csv"))
        _render_token_panels_a4(list(pa["ig_value"]), list(pa["label"]),
                                list(pb["ig_value"]), list(pb["label"]),
                                args.output_path, args.dpi)
        print(f"[render_only] Saved {args.output_path}")
        return

    # ── Label resolver ────────────────────────────────────────────────
    resolver = PatternLabelResolver(args.run_dir, short_names_csv=args.short_names_csv)

    # ── Load token-level CSV (Panels A + B) ───────────────────────────
    tok_csv = layout.token_csv(args.split) if csv_dir == layout.analysis_dir else \
              os.path.join(csv_dir, f"token_importance_{args.split}.csv")
    if not os.path.exists(tok_csv):
        sys.exit(f"  [ERROR] Missing {tok_csv}")
    df_tok = pd.read_csv(tok_csv)
    print(f"[1] Token CSV: {len(df_tok):,} tokens")

    # Estimate n_test from prevalence column on pathway CSV (for the suptitle)
    pw_csv = layout.pathway_csv(args.split) if csv_dir == layout.analysis_dir else \
             os.path.join(csv_dir, f"pathway_importance_{args.split}.csv")
    df_pw_full = pd.read_csv(pw_csv) if os.path.exists(pw_csv) else pd.DataFrame()
    n_test = 0
    if not df_pw_full.empty and "prevalence" in df_pw_full.columns and \
       "n_present" in df_pw_full.columns:
        row0 = df_pw_full.iloc[0]
        if row0["prevalence"] > 0:
            n_test = int(round(row0["n_present"] / row0["prevalence"]))

    # ── Panel A data: population-weighted ─────────────────────────────
    df_ig_tok = df_tok.dropna(subset=["ig_mean"]).copy()
    if "n_present" in df_tok.columns and "prevalence" in df_tok.columns:
        ig_pop_series = pd.Series({
            r["token"]: r["ig_mean"] * r["prevalence"]
            for _, r in df_ig_tok.iterrows()
        })
        tok_n_present = {r["token"]: int(r.get("n_present", 0)) for _, r in df_ig_tok.iterrows()}
        tok_prevalence = {r["token"]: r.get("prevalence", 0) for _, r in df_ig_tok.iterrows()}
    else:
        ig_pop_series = pd.Series(df_ig_tok.set_index("token")["ig_mean"])
        tok_n_present, tok_prevalence = {}, {}
    print(f"[2] Panel A: {len(ig_pop_series):,} tokens")

    # Top-N by |population IG|, then sort ascending for display
    ig_pop_top = ig_pop_series.reindex(
        ig_pop_series.abs().sort_values(ascending=False).head(args.n_show).index
    )
    ig_pop_top = ig_pop_top.reindex(ig_pop_top.abs().sort_values(ascending=True).index)
    labels_a = [resolver.format_token(t) for t in ig_pop_top.index]
    values_a = list(ig_pop_top.values)

    # ── Panel B data: carrier-conditional ────────────────────────────
    ig_carrier_series = pd.Series(df_ig_tok.set_index("token")["ig_mean"])
    ig_carrier_top = ig_carrier_series.reindex(
        ig_carrier_series.abs().sort_values(ascending=False).head(args.n_show).index
    )
    ig_carrier_top = ig_carrier_top.reindex(ig_carrier_top.abs().sort_values(ascending=True).index)
    labels_b = []
    for t in ig_carrier_top.index:
        lbl = resolver.format_token(t)
        n_pr = tok_n_present.get(t, 0)
        if args.show_n and n_pr > 0:
            lbl = f"{lbl}  (n={n_pr})"
        labels_b.append(lbl)
    values_b = list(ig_carrier_top.values)
    print(f"[3] Panel B: {len(ig_carrier_top)} tokens")

    # ── Panel C data: FULL POOL (no shape minimums) + Jaccard dedup ────
    # Panel C demonstrates what the carrier-averaged \u03a3IG ranking selects
    # when no structural constraints are imposed: top-N is dominated by
    # single-visit multi-token co-occurrence (the "raw" winner). This
    # motivates Panel D's locks. See Fig S8 for the full-pool sweep over
    # Jaccard thresholds.
    df_full = _load_or_score_full_pool(layout, args.split)
    if args.no_jaccard:
        rep_patterns_c = set(df_full["pattern"].tolist())
        print(f"  [Panel C] no Jaccard dedup \u2014 {len(rep_patterns_c):,} patterns")
    else:
        rep_patterns_c = cluster_jaccard_inmemory(
            df_full, args.jaccard_threshold, verbose=True)
        print(f"  [Panel C] Jaccard \u2265 {args.jaccard_threshold} \u2192 "
              f"{len(rep_patterns_c):,} representatives")
    panel_c_df = _select_panel_c_patterns(
        df_full, rep_patterns_c, args.n_show, args.panel_c_mode,
    )
    panel_c_labels_raw = [resolver.format_pattern(p, short_names=True)
                          for p in panel_c_df["pattern"]]
    if args.show_n:
        panel_c_labels = [f"{l}  (n={int(n)})" for l, n in
                          zip(panel_c_labels_raw, panel_c_df["n_present"])]
    else:
        panel_c_labels = panel_c_labels_raw
    panel_c_values = list(panel_c_df["ig_signed_mean"])
    print(f"[4] Panel C: {len(panel_c_df)} patterns "
          f"(full pool, top-{args.n_show} by |\u03A3IG|)")

    # ── Panel D data: cross-visit pool only (the locked Panel C source) ──
    # This is what the main paper's locks (`min_steps=2`, `min_tokens=2`)
    # surface: cross-visit temporal trajectories invisible to single-visit
    # attribution. See Fig S7 for the cross-visit-only sweep over Jaccard
    # thresholds.
    if df_pw_full.empty:
        sys.exit("  [ERROR] Cannot render Panel D \u2014 pathway CSV missing")
    df_pw_with_ck = df_pw_full[df_pw_full["n_present"] >= 3].copy()
    if args.no_jaccard:
        rep_patterns_d = set(df_pw_with_ck["pattern"].tolist())
        print(f"  [Panel D] no Jaccard dedup \u2014 {len(rep_patterns_d):,} patterns")
    else:
        rep_patterns_d = get_jaccard_representatives(
            csv_dir, args.split, args.jaccard_threshold)
        if rep_patterns_d is None:
            sys.exit("  [ERROR] Jaccard reps missing for Panel D")
    panel_d_df = _select_panel_c_patterns(
        df_pw_with_ck, rep_patterns_d, args.n_show, args.panel_c_mode,
    )
    panel_d_labels_raw = [resolver.format_pattern(p, short_names=True)
                          for p in panel_d_df["pattern"]]
    if args.show_n:
        panel_d_labels = [f"{l}  (n={int(n)})" for l, n in
                          zip(panel_d_labels_raw, panel_d_df["n_present"])]
    else:
        panel_d_labels = panel_d_labels_raw
    panel_d_values = list(panel_d_df["ig_signed_mean"])
    print(f"[5] Panel D: {len(panel_d_df)} patterns "
          f"(cross-visit pool, top-{args.n_show} by |\u03A3IG|)")

    # ── Shared layout constants ───────────────────────────────────────
    max_rows = max(args.n_show, len(panel_c_df), len(panel_d_df), 10)
    panel_height = max(10, max_rows * 0.45)
    fs_title = args.fontsize + 2
    fs_xlabel = args.fontsize - 1

    # Panel titles. Bold first line (panel role) + regular second line
    # (description). Bolding only the head uses mathtext (``$\bf{...}$``)
    # so set_title renders both weights without overlap.
    def _bold(s):
        return r"$\bf{" + s.replace(" ", r"\ ").replace(":", r"{:}") + r"}$"

    if args.no_jaccard:
        c_dedup = "no Jaccard dedup"
        d_dedup = "no Jaccard dedup"
    else:
        c_dedup = f"Jaccard \u2265 {args.jaccard_threshold} dedup"
        d_dedup = f"Jaccard \u2265 {args.jaccard_threshold} dedup"
    panel_titles = [
        ("A. Population token attribution:",
         "Mean IG across all patients"),
        ("B. Conditional token attribution:",
         "Mean IG among carriers"),
        ("C. Pattern attribution (no shape minimums):",
         f"Carrier-averaged \u03A3IG, all pattern shapes\n{c_dedup}"),
        ("D. Cross-visit temporal pattern attribution:",
         f"Carrier-averaged \u03A3IG, \u22652 visits & \u22652 tokens\n{d_dedup}"),
    ]
    # Force every subtitle to the same line count so all 4 titles take
    # identical vertical space \u2014 keeps the axes y-position aligned across
    # the 2x2 grid (otherwise A/B's 1-line subtitle pushes their bars up
    # relative to C/D's 2-line subtitle).
    max_sub_lines = max(s.count("\n") + 1 for _, s in panel_titles)
    panel_titles = [
        (h, s + ("\n " * (max_sub_lines - s.count("\n") - 1)))
        for h, s in panel_titles
    ]

    def _set_panel_title(ax, head, sub):
        ax.set_title(_bold(head) + "\n" + sub, fontsize=fs_title, pad=15)

    panel_specs = [
        dict(values=values_a, labels=labels_a, title=panel_titles[0],
             xlabel="E[IG] across all patients\n\u2190 protective    risk \u2192"),
        dict(values=values_b, labels=labels_b, title=panel_titles[1],
             xlabel="E[IG | present]\n\u2190 protective    risk \u2192"),
        dict(values=panel_c_values, labels=panel_c_labels, title=panel_titles[2],
             xlabel="E[\u03A3IG | present]\n\u2190 protective    risk \u2192"),
        dict(values=panel_d_values, labels=panel_d_labels, title=panel_titles[3],
             xlabel="E[\u03A3IG | present]\n\u2190 protective    risk \u2192"),
    ]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.stitch:
        # ── Stitch mode: 4 standalone panels composited as a 2x2 grid.
        # Top row    = atomic single-token attribution (A, B)
        # Bottom row = compositional pattern attribution (C, D)
        #
        # Symmetry rules (every one matters \u2014 violating any breaks
        # alignment in the final 2x2):
        #
        #   1. Each panel saved at IDENTICAL canvas dimensions
        #      (panel_fig_width \u00d7 panel_height inches \u00d7 dpi). NO
        #      ``bbox_inches="tight"`` \u2014 that would crop each panel to
        #      a different width based on label length and break the grid.
        #   2. Axes geometry pinned to a fixed (x0, y0, width, height) for
        #      ALL panels \u2014 not derived from per-panel tight_layout, which
        #      varies based on label widths. Bar columns line up
        #      horizontally and vertically across the 2x2.
        #   3. Title rendered via ``fig.text`` centered over the FULL
        #      canvas (not over the narrow right-anchored axes), so the
        #      panel's visual center of mass sits in the middle of the
        #      cell. Padded to the max line count so every panel reserves
        #      identical vertical space for the title.
        #   4. Stitch is a true grid \u2014 no per-row centering, no height
        #      normalization.
        import tempfile
        from PIL import Image

        # Each panel gets half the figure width.
        panel_fig_width = args.fig_width / 2

        # ── Pinned axes geometry (figure-fraction coords) ──────────────
        # Identical for all 4 panels. The bar area is intentionally
        # narrow (~20% of canvas width) because the 30-row pattern labels
        # in C/D need most of the horizontal real estate to render
        # without truncation \u2014 the figure's primary information
        # carrier is the LABELS, not the bar magnitudes.
        # Tightened defaults: was 0.84 / 0.11 / 40 px stitch gap, plus
        # extensive post-hoc PNG cropping via tighten_figure3.py to
        # recover white space. New defaults render a tight figure
        # directly so no post-processing is needed.
        axes_left   = 0.77   # leaves ~77% of canvas for ytick labels
        axes_right  = 0.97   # 3% right margin for rightmost xtick label
        axes_bottom = 0.06   # was 0.11, title sits closer to xlabel
        axes_top    = 0.93   # was 0.84, title hugs canvas top
        axes_w = axes_right - axes_left   # \u2248 0.20 of canvas (~30% of prior)
        axes_h = axes_top - axes_bottom

        # ── X-limit unification across panels ──────────────────────────
        # ``--xlim_mode`` controls how panels share x-axis ranges:
        #   "global"  \u2014 all 4 panels share one xlim (= global max).
        #             Honest "same axis" but Panel A's tiny
        #             population-weighted IG (max \u22480.04) becomes
        #             visually invisible against C's max \u22482.17.
        #   "row"     \u2014 top row (A,B) share, bottom row (C,D) share.
        #             A still small relative to B, but C+D are honestly
        #             comparable (same metric, different pools).
        #   "atomic_split" \u2014 A keeps its own (auto) xlim, B shares with
        #             C+D. Sacrifices A=B comparability to keep A's bars
        #             visible. Best compromise.
        #   "per_panel" \u2014 each panel has its own xlim (legacy).
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
            ma = _max_abs(panel_specs[:1])           # A alone
            m_rest = _max_abs(panel_specs[1:])       # B, C, D shared
            panel_xlims = [(-ma * 1.05, ma * 1.05)] + \
                           [(-m_rest * 1.05, m_rest * 1.05)] * 3
        else:  # per_panel
            panel_xlims = [None] * 4   # signed_ig_bars's auto-fit kept

        for letter, lim in zip("ABCD", panel_xlims):
            if lim is None:
                print(f"  [stitch] xlim Panel {letter}: per-panel auto-fit")
            else:
                print(f"  [stitch] xlim Panel {letter}: "
                      f"({lim[0]:+.3f}, {lim[1]:+.3f})")

        tmp_paths = []
        for i, spec in enumerate(panel_specs):
            fig_i, ax_i = plt.subplots(figsize=(panel_fig_width, panel_height))
            signed_ig_bars(values=spec["values"], labels=spec["labels"],
                           ax=ax_i, fontsize=args.fontsize)
            ax_i.set_xlabel(spec["xlabel"], fontsize=fs_xlabel)
            ax_i.tick_params(axis="x", labelsize=fs_xlabel - 1)

            # Override signed_ig_bars's auto-fit xlim with the unified
            # range chosen by --xlim_mode. None = keep auto-fit.
            if panel_xlims[i] is not None:
                ax_i.set_xlim(*panel_xlims[i])

            # Pin axes geometry. Skip tight_layout entirely \u2014 it'd just
            # introduce per-panel variation we then have to undo.
            ax_i.set_position([axes_left, axes_bottom, axes_w, axes_h])

            # Title centered over the FULL canvas via fig.text, NOT
            # ax.set_title (which centers over the narrow axes \u2192 looks
            # off-center because the axes is right-anchored).
            head, sub = spec["title"]
            head_y = 0.985        # was 0.965, pushes title hard against canvas top
            sub_y  = head_y - 0.018 * (1 + head.count("\n"))
            fig_i.text(0.5, head_y, head, ha="center", va="top",
                        fontsize=fs_title, fontweight="bold", color="#111111")
            fig_i.text(0.5, sub_y, sub, ha="center", va="top",
                        fontsize=fs_title - 1, color="#444444")

            tmp = tempfile.NamedTemporaryFile(
                prefix=f"fig3_panel_{chr(ord('a')+i)}_", suffix=".png", delete=False)
            fig_i.savefig(tmp.name, dpi=args.dpi)
            plt.close(fig_i)
            tmp_paths.append(tmp.name)

        # Sanity: every panel image MUST be identical pixel size.
        images = [Image.open(p) for p in tmp_paths]
        sizes = {im.size for im in images}
        if len(sizes) != 1:
            print(f"  [WARN] panel sizes differ \u2014 stitch will be misaligned: "
                  f"{sizes}")
        cell_w, cell_h = images[0].size

        # True 2x2 grid \u2014 no centering, no resizing.
        col_gap = args.stitch_gap_px
        row_gap = args.stitch_gap_px
        final_w = 2 * cell_w + col_gap
        final_h = 2 * cell_h + row_gap
        composite = Image.new("RGB", (final_w, final_h), "white")
        composite.paste(images[0], (0, 0))                     # A: top-left
        composite.paste(images[1], (cell_w + col_gap, 0))      # B: top-right
        composite.paste(images[2], (0, cell_h + row_gap))      # C: bot-left
        composite.paste(images[3], (cell_w + col_gap, cell_h + row_gap))  # D: bot-right
        composite.save(args.output_path)

        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

        size_kb = os.path.getsize(args.output_path) // 1024
        print(f"[6] Saved figure (stitched 2x2, {final_w}\u00d7{final_h} px): "
              f"{args.output_path} ({size_kb} KB)")

    else:
        # ── 2-panel token-level figure (A + B side-by-side) ────────────
        # Figure 3 in the manuscript text is the TOKEN-LEVEL comparison
        # (Population vs Carrier-conditional). The C/D pattern panels
        # are still computed above so panel_d_*.csv exists for Figure 4
        # to consume, but they're no longer rendered here.
        #
        # Layout: 2 columns, titles ABOVE each column (no empty vertical
        # space), large labels, inline bar value annotations, single
        # legend at the top.
        tok_specs = panel_specs[:2]   # A + B only
        _render_token_panels_a4(tok_specs[0]["values"], tok_specs[0]["labels"],
                                tok_specs[1]["values"], tok_specs[1]["labels"],
                                args.output_path, args.dpi)
        size_kb = os.path.getsize(args.output_path) // 1024
        print(f"[6] Saved figure (2-panel token-level): "
              f"{args.output_path} ({size_kb} KB)")

    # ── Backing CSVs ─────────────────────────────────────────────────
    os.makedirs(panels_dir, exist_ok=True)
    jt_pct = int(args.jaccard_threshold * 100)
    sfx = "_short"

    rows_a = []
    for t in ig_pop_top.index:
        rows_a.append({
            "raw_token": t,
            "label": resolver.format_token(t),
            "ig_value": ig_pop_top[t],
            "n_present": tok_n_present.get(t, 0),
            "prevalence": tok_prevalence.get(t, 0),
        })
    pd.DataFrame(rows_a).to_csv(
        os.path.join(panels_dir, f"panel_a_{args.split}_j{jt_pct}{sfx}.csv"), index=False
    )

    rows_b = []
    for t in ig_carrier_top.index:
        rows_b.append({
            "raw_token": t,
            "label": resolver.format_token(t),
            "ig_value": ig_carrier_top[t],
            "n_present": tok_n_present.get(t, 0),
        })
    pd.DataFrame(rows_b).to_csv(
        os.path.join(panels_dir, f"panel_b_{args.split}_j{jt_pct}{sfx}.csv"), index=False
    )

    # Panel C (full pool, no minimums), keep raw `pattern` column.
    rows_c = []
    for _, r, lbl in zip(range(len(panel_c_df)), panel_c_df.itertuples(), panel_c_labels_raw):
        rows_c.append({
            "pattern": r.pattern,
            "label": lbl,
            "ig_value": r.ig_signed_mean,
            "n_present": int(r.n_present),
        })
    panel_c_path = os.path.join(panels_dir, f"panel_c_{args.split}_j{jt_pct}{sfx}.csv")
    pd.DataFrame(rows_c).to_csv(panel_c_path, index=False)

    # Panel D (cross-visit pool), make_fig4 reads THIS one to align Fig 4's
    # validation dumbbell with Panel D's pattern set (the locked Panel C).
    rows_d = []
    for _, r, lbl in zip(range(len(panel_d_df)), panel_d_df.itertuples(), panel_d_labels_raw):
        rows_d.append({
            "pattern": r.pattern,
            "label": lbl,
            "ig_value": r.ig_signed_mean,
            "n_present": int(r.n_present),
        })
    panel_d_path = os.path.join(panels_dir, f"panel_d_{args.split}_j{jt_pct}{sfx}.csv")
    pd.DataFrame(rows_d).to_csv(panel_d_path, index=False)
    print(f"[7] Backing CSVs in {panels_dir}/")
    print(f"      panel_c_{args.split}_j{jt_pct}{sfx}.csv  (full pool)")
    print(f"      panel_d_{args.split}_j{jt_pct}{sfx}.csv  (cross-visit pool, Fig 4 source)")


if __name__ == "__main__":
    main()
