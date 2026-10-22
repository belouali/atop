"""make_fig_s10_pattern_aggregation_comparison \u2014 carrier vs population-weighted ΣIG, FULL POOL.

Pattern-level analog of Fig S9 (which compared at the token level). Quantifies
the methodological refinement at the pattern aggregation layer: switching
from population-weighted ΣIG (the natural pattern-level extension of the
SHAP / Captum-IG default convention) to carrier-conditional ΣIG (AToP's
choice in Fig 3 Panel C / Panel D) surfaces a set of multi-token
compositional patterns that the standard convention buries deep in the
long tail.

Side-by-side 2-panel comparison on the **full pool** (Fig 3 Panel C source,
Jaccard-deduplicated at j=0.2):

  * **Left panel (carrier-conditional / AToP):** Top-30 patterns by
    ``E[ΣIG | present]``, what Fig 3 Panel C shows.
  * **Right panel (population-weighted / SHAP-equivalent):** Top-30 by
    ``E[ΣIG] = E[ΣIG | present] × prevalence``, the natural pattern-level
    extension of standard global feature importance aggregation.

Each row is annotated with its rank under the OTHER aggregation. Bars with
bold dark outlines are absent from the other panel's top-30.

Headline number for the figure caption: of 30 high-impact patterns surfaced
by carrier-conditional aggregation, **only 1** also appears in the SHAP-
equivalent population-weighted top-30. The other 29 are clinically
meaningful multi-token co-occurrence patterns (e.g.
``{Severe MDD; SI}`` at population-rank ~#543) whose impact-per-occurrence
is large but whose prevalence is too low to survive prevalence-weighting.

The 1/30 overlap (compared to S9's 7/30 for tokens) shows the aggregation
penalty compounds at the pattern level: a multi-token pattern's prevalence
is the intersection of its tokens' prevalences, so prevalence-weighting
penalizes compositional patterns even harder than single tokens.

Usage:
    python paper/jamia2026/figures/make_fig_s10_pattern_aggregation_comparison.py \
        --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from atop.explain.cooccurrence import cluster_jaccard_inmemory
from atop.explain.pattern_labels import PatternLabelResolver
from atop.run_layout import RunLayout


# Visual tokens (matched to S9 for cross-figure consistency).
COLOR_RISK     = "#c44e52"
COLOR_PROT     = "#4c72b0"
COLOR_RISK_DIM = "#e8a0a2"
COLOR_PROT_DIM = "#a0b8d0"
EDGE_FLAG      = "#222222"


def _draw_panel(ax, df, *, magnitude_col: str, label_col: str,
                  rank_col: str, fontsize: int, panel_head: str,
                  panel_sub: str, xlabel: str) -> None:
    n = len(df)
    y = list(range(n))
    vals = df[magnitude_col].tolist()
    labels = df[label_col].tolist()
    ranks = df[rank_col].tolist()

    for i, (v, r) in enumerate(zip(vals, ranks)):
        in_other = (r is not None) and (r <= 30)
        if v >= 0:
            color = COLOR_RISK_DIM if in_other else COLOR_RISK
        else:
            color = COLOR_PROT_DIM if in_other else COLOR_PROT
        edge = EDGE_FLAG if not in_other else "white"
        lw = 1.2 if not in_other else 0.5
        ax.barh(i, v, color=color, edgecolor=edge, linewidth=lw, height=0.78)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.axvline(0, color="black", linewidth=0.8)

    # Right-side rank annotation per row. The bold dark edge on the bar
    # already signals "out of top-30"; the rank number alone is enough.
    pad = max(abs(v) for v in vals) * 0.04
    for i, r in enumerate(ranks):
        ann = "n/a" if r is None else f"#{int(r)}"
        is_out = (r is None) or (r > 30)
        ax.text(ax.get_xlim()[1] - pad, i, ann,
                ha="right", va="center",
                fontsize=fontsize - 2,
                color="#222222" if is_out else "#777777",
                fontweight="bold" if is_out else "normal")

    ax.set_ylim(-0.8, n - 0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    def _bold(s: str) -> str:
        return r"$\bf{" + s.replace(" ", r"\ ").replace(":", r"{:}") + r"}$"
    ax.set_title(_bold(panel_head) + "\n" + panel_sub,
                  fontsize=fontsize + 2, pad=14, color="#222222")
    ax.set_xlabel(xlabel, fontsize=fontsize - 1, labelpad=8)


def _load_full_pool_with_reps(layout: RunLayout, split: str,
                                jaccard_threshold: float):
    """Load full-pool scored DataFrame and j=0.2 Jaccard reps. Caches the
    rep set to ``5_analysis/full_pool_jaccard_reps_test_j<pct>.pkl`` so
    subsequent calls are instant."""
    pool_pkl = os.path.join(layout.analysis_dir,
                             f"full_pool_scored_{split}.pkl")
    if not os.path.exists(pool_pkl):
        sys.exit(f"  [ERROR] Missing {pool_pkl}\n  Run paper/jamia2026/figures/make_fig3.py "
                 f"first to materialize the full-pool scored cache.")
    with open(pool_pkl, "rb") as f:
        df = pickle.load(f)

    jt_pct = int(jaccard_threshold * 100)
    rep_cache = os.path.join(layout.analysis_dir,
                              f"full_pool_jaccard_reps_{split}_j{jt_pct}.pkl")
    if (os.path.exists(rep_cache)
            and os.path.getmtime(rep_cache) >= os.path.getmtime(pool_pkl)):
        with open(rep_cache, "rb") as f:
            reps = pickle.load(f)
        print(f"  [reps] cached \u2192 {len(reps):,} ({os.path.basename(rep_cache)})")
    else:
        print(f"  [reps] computing j={jaccard_threshold} dedup over "
              f"{len(df):,} patterns (~30 s vectorized)\u2026")
        reps = cluster_jaccard_inmemory(df, jaccard_threshold, verbose=False)
        with open(rep_cache, "wb") as f:
            pickle.dump(reps, f)
        print(f"  [reps] cached \u2192 {rep_cache}")
    return df, reps


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render Fig S10 (pattern-level aggregation comparison).")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--jaccard_threshold", type=float, default=0.2,
                   help="Jaccard dedup threshold (default 0.2 = matches Fig 3 Panel C).")
    p.add_argument("--n_show", type=int, default=30)
    p.add_argument("--fontsize", type=int, default=12)
    p.add_argument("--fig_width", type=float, default=28.0,
                   help="Total figure width across both panels.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/"
                           "FigureS10_pattern_aggregation_comparison.png")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)

    # ── Load full pool + Jaccard reps ─────────────────────────────────
    df_full, reps = _load_full_pool_with_reps(
        layout, args.split, args.jaccard_threshold)
    df = df_full[df_full["pattern"].isin(reps)].copy()

    # ── Estimate N (test set size) for prevalence weighting ──────────
    # Use the maximum n_present as an upper bound proxy isn't right \u2014
    # better: pull from sequences_<split>.pkl directly.
    seq_pkl = layout.sequences_pkl(args.split)
    if os.path.exists(seq_pkl):
        with open(seq_pkl, "rb") as f:
            df_seq = pickle.load(f)
        N = len(df_seq)
    else:
        # Fallback: solve from a known token's prevalence in pathway CSV.
        N = int(df["n_present"].max() / 0.20)  # rough estimate; warn
        print(f"  [WARN] sequences_pkl missing; estimated N={N:,}")

    # ── Compute population-weighted ΣIG and absolute magnitudes ───────
    df["ig_pop"]      = df["ig_signed_mean"] * (df["n_present"] / N)
    df["abs_carrier"] = df["ig_signed_mean"].abs()
    df["abs_pop"]     = df["ig_pop"].abs()
    print(f"[1] Full pool reps: {len(df):,} patterns ({N:,} test patients)")

    # Per-pattern rank under each aggregation
    df_full_carrier = df.sort_values("abs_carrier", ascending=False).reset_index(drop=True)
    df_full_carrier["carrier_rank"] = df_full_carrier.index + 1
    df_full_pop     = df.sort_values("abs_pop",     ascending=False).reset_index(drop=True)
    df_full_pop["pop_rank"] = df_full_pop.index + 1
    rank_carrier = dict(zip(df_full_carrier["pattern"], df_full_carrier["carrier_rank"]))
    rank_pop     = dict(zip(df_full_pop["pattern"],     df_full_pop["pop_rank"]))

    # Top-N each panel
    df_b = df_full_carrier.head(args.n_show).copy()
    df_a = df_full_pop.head(args.n_show).copy()
    df_b["pop_rank_other"]     = df_b["pattern"].map(rank_pop)
    df_a["carrier_rank_other"] = df_a["pattern"].map(rank_carrier)

    overlap = set(df_a["pattern"]) & set(df_b["pattern"])
    print(f"[2] overlap: {len(overlap)}/{args.n_show}  |  "
          f"in carrier not pop: {args.n_show - len(overlap)}  |  "
          f"in pop not carrier: {args.n_show - len(overlap)}")

    # ── Format labels via the canonical resolver ──────────────────────
    resolver = PatternLabelResolver(args.run_dir, verbose=False)
    df_b["label_disp"] = df_b["pattern"].apply(
        lambda p: resolver.format_pattern(p, short_names=True))
    df_a["label_disp"] = df_a["pattern"].apply(
        lambda p: resolver.format_pattern(p, short_names=True))

    # Soft-wrap long labels (Pattern C labels can be ~80 chars)
    def _wrap(lab, w=55):
        if len(lab) <= w:
            return lab
        bp = lab.rfind(" ", 0, w)
        return lab[:bp] + "\n  " + lab[bp:].lstrip() if bp > 20 else lab
    df_b["label_disp"] = df_b["label_disp"].apply(_wrap)
    df_a["label_disp"] = df_a["label_disp"].apply(_wrap)

    # Order bottom \u2192 top so topmost = rank #1
    df_b = df_b.sort_values("abs_carrier", ascending=True).reset_index(drop=True)
    df_a = df_a.sort_values("abs_pop",     ascending=True).reset_index(drop=True)

    # ── Render ────────────────────────────────────────────────────────
    panel_height = max(10, args.n_show * 0.45)
    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width, panel_height),
                              gridspec_kw={"wspace": 0.55})

    _draw_panel(
        axes[0], df_b,
        magnitude_col="ig_signed_mean", label_col="label_disp",
        rank_col="pop_rank_other",
        fontsize=args.fontsize,
        panel_head="A. Carrier-conditional \u03A3IG (AToP):",
        panel_sub=f"Top-{args.n_show} patterns by E[\u03A3IG | present].  "
                   f"Annotation = pop-weighted rank "
                   f"(bold + dark edge \u2192 outside population top-{args.n_show}).",
        xlabel="E[\u03A3IG | present]   \u2190 protective    risk \u2192",
    )
    _draw_panel(
        axes[1], df_a,
        magnitude_col="ig_pop", label_col="label_disp",
        rank_col="carrier_rank_other",
        fontsize=args.fontsize,
        panel_head="B. Population-weighted \u03A3IG:",
        panel_sub=f"Top-{args.n_show} by E[\u03A3IG] = E[\u03A3IG | present] \u00d7 prevalence.  "
                   f"Annotation = carrier-conditional rank "
                   f"(bold + dark edge \u2192 outside carrier top-{args.n_show}).",
        xlabel="E[\u03A3IG]   \u2190 protective    risk \u2192",
    )

    fs_title = args.fontsize + 4
    fs_sub   = args.fontsize - 1
    fig.suptitle(
        "Pattern-level aggregation choice has near-total non-overlap "
        "(only " + str(len(overlap)) + " of " + str(args.n_show) + " patterns shared)",
        fontsize=fs_title, fontweight="bold", y=0.995,
    )
    # Subtitle values are computed from the plotted data (a hard-coded
    # rank range and a "dominated by single-token" claim were wrong).
    _is_single = lambda p: not str(p).startswith("{") and "->" not in str(p)
    n_single_b = int(sum(_is_single(p) for p in df_a["pattern"]))
    _ranks = [int(x) for x in df_b["pop_rank_other"].dropna()
              if int(x) > args.n_show]
    _rank_txt = (f"#{min(_ranks):,}\u2013#{max(_ranks):,}" if _ranks else "n/a")
    fig.text(
        0.5, 0.965,
        f"Full pool, j={args.jaccard_threshold} Jaccard reps. The population-weighted "
        f"top-{args.n_show} contains {n_single_b} single-token entries, the rest "
        f"high-prevalence within-visit sets; the {len(_ranks)} carrier-conditional "
        f"patterns absent from it sit at population ranks {_rank_txt}. Bold "
        f"dark-edged bars = absent from the other panel's top-{args.n_show}; "
        f"their rank in the other ranking is shown in bold next to each bar.",
        ha="center", va="top", fontsize=fs_sub, color="#444444",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(args.output_path) // 1024
    print(f"[3] Saved figure: {args.output_path} ({size_kb} KB)")

    # ── Backing CSV ───────────────────────────────────────────────────
    rows = []
    for _, r in df_b.iterrows():
        rows.append({
            "panel": "A_carrier_top",
            "rank_in_panel": int(rank_carrier[r["pattern"]]),
            "pattern": r["pattern"], "label": r["label_disp"].replace("\n", " "),
            "n_present": int(r["n_present"]),
            "prevalence": r["n_present"] / N,
            "ig_carrier_mean": r["ig_signed_mean"],
            "ig_pop_weighted": r["ig_pop"],
            "rank_in_other": int(r["pop_rank_other"]),
            "in_other_top_n": bool(r["pop_rank_other"] <= args.n_show),
        })
    for _, r in df_a.iterrows():
        rows.append({
            "panel": "B_population_top",
            "rank_in_panel": int(rank_pop[r["pattern"]]),
            "pattern": r["pattern"], "label": r["label_disp"].replace("\n", " "),
            "n_present": int(r["n_present"]),
            "prevalence": r["n_present"] / N,
            "ig_carrier_mean": r["ig_signed_mean"],
            "ig_pop_weighted": r["ig_pop"],
            "rank_in_other": int(r["carrier_rank_other"]),
            "in_other_top_n": bool(r["carrier_rank_other"] <= args.n_show),
        })
    out_csv = os.path.splitext(args.output_path)[0] + ".csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"    csv: {out_csv}")

    # Headline numbers
    print(f"\n=== Headline numbers for caption ===")
    print(f"  Top-{args.n_show} overlap:                   {len(overlap)}/{args.n_show}")
    print(f"  Compare to Fig S9 (token level):       7/30 overlap")
    print(f"  \u2192 the aggregation penalty compounds at the pattern level")


if __name__ == "__main__":
    main()
