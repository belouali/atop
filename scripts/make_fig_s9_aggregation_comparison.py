"""make_fig_s9_aggregation_comparison \u2014 carrier-conditional vs. population-weighted IG.

Quantifies the methodological refinement at the heart of AToP: switching
the IG aggregation convention from population-weighted (the SHAP / Captum-
IG default) to carrier-conditional (AToP's choice) surfaces a set of
clinically obvious rare-but-decisive features that the standard convention
buries deep in the long tail of the ranking.

The figure is a side-by-side 2-panel comparison:

  * **Left panel (Panel B equivalent):** Top-30 tokens by carrier-conditional
    ``E[IG | present]`` \u2014 the AToP refinement (Fig 3 Panel B).
  * **Right panel (Panel A equivalent):** Top-30 tokens by population-weighted
    ``E[IG] = E[IG | present] \u00d7 prevalence`` \u2014 mathematically equivalent
    to the standard SHAP ``summary_plot`` global feature importance (Fig 3
    Panel A).

Each row is annotated with its **rank under the OTHER aggregation**. Tokens
whose other-rank is > 30 (i.e., absent from the other panel's top-30) are
visually flagged \u2014 these are the features the alternative convention
misses.

Headline number for the figure caption: of 30 high-impact tokens surfaced
by carrier-conditional aggregation, only ~7 also appear in the SHAP-
equivalent population-weighted top-30. The other 23 are clinically obvious
risk/protective markers whose impact-per-occurrence is large but whose
population prevalence is low enough that the standard convention dilutes
them past visibility.

Usage:
    python scripts/make_fig_s9_aggregation_comparison.py \
        --run_dir ../runs/full_CPD_v2
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


# Visual tokens.
COLOR_RISK     = "#c44e52"
COLOR_PROT     = "#4c72b0"
COLOR_RISK_DIM = "#e8a0a2"   # softer fill when feature ALSO appears in other panel
COLOR_PROT_DIM = "#a0b8d0"
EDGE_FLAG      = "#222222"   # bold edge if NOT in other panel's top-30


def _draw_panel(ax, df, *, magnitude_col: str, label_col: str,
                  rank_col: str, fontsize: int, panel_head: str,
                  panel_sub: str, xlabel: str) -> None:
    """Draw one ranked-bars panel.

    df: top-N rows already sorted bottom-to-top (smallest |val| first so the
        topmost bar visually corresponds to rank #1).
    rank_col: per-row rank under the OTHER aggregation; > 30 \u2192 flag bold.
    """
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
    # already signals "out of top-30"; the rank number alone (no "[out]"
    # text) is enough \u2014 dim+regular for in-top-30, bold+dark for out.
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render Fig S9 (carrier vs population IG aggregation comparison).")
    p.add_argument("--run_dir", default="../runs/full_CPD_v2")
    p.add_argument("--split", default="test")
    p.add_argument("--n_show", type=int, default=30)
    p.add_argument("--fontsize", type=int, default=14)
    p.add_argument("--fig_width", type=float, default=24.0,
                   help="Total figure width across both panels.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/"
                           "FigureS9_aggregation_comparison.png")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)

    # ── Load token IG data ───────────────────────────────────────────
    tok_csv = layout.token_csv(args.split)
    if not os.path.exists(tok_csv):
        sys.exit(f"  [ERROR] Missing {tok_csv}")
    df = pd.read_csv(tok_csv).dropna(subset=["ig_mean", "prevalence"]).copy()
    df["ig_pop"]      = df["ig_mean"] * df["prevalence"]
    df["abs_carrier"] = df["ig_mean"].abs()
    df["abs_pop"]     = df["ig_pop"].abs()
    print(f"[1] {len(df):,} tokens with IG data")

    # ── Compute per-token rank under each aggregation ────────────────
    df_carrier_full = df.sort_values("abs_carrier", ascending=False).reset_index(drop=True)
    df_carrier_full["carrier_rank"] = df_carrier_full.index + 1
    df_pop_full     = df.sort_values("abs_pop",     ascending=False).reset_index(drop=True)
    df_pop_full["pop_rank"] = df_pop_full.index + 1
    rank_carrier = dict(zip(df_carrier_full["token"], df_carrier_full["carrier_rank"]))
    rank_pop     = dict(zip(df_pop_full["token"],     df_pop_full["pop_rank"]))

    # ── Top-N each panel ─────────────────────────────────────────────
    df_b = df_carrier_full.head(args.n_show).copy()  # carrier (Panel B equiv)
    df_a = df_pop_full.head(args.n_show).copy()      # population (Panel A equiv)

    df_b["pop_rank_other"]     = df_b["token"].map(rank_pop)
    df_a["carrier_rank_other"] = df_a["token"].map(rank_carrier)

    overlap = set(df_a["token"]) & set(df_b["token"])
    in_b_not_a = set(df_b["token"]) - set(df_a["token"])
    in_a_not_b = set(df_a["token"]) - set(df_b["token"])
    print(f"[2] overlap: {len(overlap)}/{args.n_show}  |  "
          f"in B not A: {len(in_b_not_a)}  |  in A not B: {len(in_a_not_b)}")

    # ── Format labels (resolver-backed) ──────────────────────────────
    resolver = PatternLabelResolver(args.run_dir, verbose=False)
    df_b["label_disp"] = df_b["token"].apply(resolver.format_token)
    df_a["label_disp"] = df_a["token"].apply(resolver.format_token)

    # Order bottom \u2192 top for display (smallest |val| first \u2192 topmost = rank #1)
    df_b = df_b.sort_values("abs_carrier", ascending=True).reset_index(drop=True)
    df_a = df_a.sort_values("abs_pop",     ascending=True).reset_index(drop=True)

    # ── Render ────────────────────────────────────────────────────────
    panel_height = max(8, args.n_show * 0.42)
    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width, panel_height),
                              gridspec_kw={"wspace": 0.45})

    _draw_panel(
        axes[0], df_b,
        magnitude_col="ig_mean", label_col="label_disp",
        rank_col="pop_rank_other",
        fontsize=args.fontsize,
        panel_head="A. Carrier-conditional IG (AToP refinement):",
        panel_sub=f"Top-{args.n_show} by E[IG | present].  Annotation = pop-weighted rank "
                   f"(bold + dark edge \u2192 outside population top-{args.n_show}).",
        xlabel="E[IG | present]   \u2190 protective    risk \u2192",
    )
    _draw_panel(
        axes[1], df_a,
        magnitude_col="ig_pop", label_col="label_disp",
        rank_col="carrier_rank_other",
        fontsize=args.fontsize,
        panel_head="B. Population-weighted IG:",
        panel_sub=f"Top-{args.n_show} by E[IG] = E[IG | present] \u00d7 prevalence.  "
                   f"Annotation = carrier-conditional rank "
                   f"(bold + dark edge \u2192 outside carrier top-{args.n_show}).",
        xlabel="E[IG]   \u2190 protective    risk \u2192",
    )

    # ── Suptitle ─────────────────────────────────────────────────────
    fs_title = args.fontsize + 4
    fs_sub   = args.fontsize - 1
    fig.suptitle(
        "Aggregation choice changes which features surface as high-impact",
        fontsize=fs_title, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.965,
        f"Of the {args.n_show} top tokens by carrier-conditional aggregation, "
        f"only {len(overlap)} also appear in the population-weighted top-"
        f"{args.n_show}.  Bold dark-edged bars = absent from the other panel's "
        f"top-{args.n_show}; their rank in the other ranking is shown in bold "
        f"next to each bar.",
        ha="center", va="top", fontsize=fs_sub, color="#444444",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(args.output_path) // 1024
    print(f"[3] Saved figure: {args.output_path} ({size_kb} KB)")

    # ── Backing CSV (long format, both rankings + ranks-in-other) ────
    rows = []
    for _, r in df_b.iterrows():
        rows.append({
            "panel": "A_carrier_top",
            "rank_in_panel": int(rank_carrier[r["token"]]),
            "token": r["token"], "label": r["label_disp"],
            "ig_mean_carrier": r["ig_mean"], "prevalence": r["prevalence"],
            "ig_pop_weighted": r["ig_pop"],
            "rank_in_other": int(r["pop_rank_other"]),
            "in_other_top_n": bool(r["pop_rank_other"] <= args.n_show),
        })
    for _, r in df_a.iterrows():
        rows.append({
            "panel": "B_population_top",
            "rank_in_panel": int(rank_pop[r["token"]]),
            "token": r["token"], "label": r["label_disp"],
            "ig_mean_carrier": r["ig_mean"], "prevalence": r["prevalence"],
            "ig_pop_weighted": r["ig_pop"],
            "rank_in_other": int(r["carrier_rank_other"]),
            "in_other_top_n": bool(r["carrier_rank_other"] <= args.n_show),
        })
    out_csv = os.path.splitext(args.output_path)[0] + ".csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"    csv: {out_csv}")

    # ── Headline stats for the figure caption ────────────────────────
    print(f"\n=== Headline numbers for caption ===")
    print(f"  Top-{args.n_show} overlap:                   {len(overlap)}/{args.n_show}")
    print(f"  In carrier top-N, NOT in pop top-N:    {len(in_b_not_a)}")
    print(f"  In pop top-N, NOT in carrier top-N:    {len(in_a_not_b)}")


if __name__ == "__main__":
    main()
