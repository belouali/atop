"""make_fig_s8_full_pool_sensitivity \u2014 FULL pool \u00d7 6 Jaccard thresholds.

Third in the sensitivity sweep series:

  * Fig S7 \u2014 cross-visit pool (min_steps=2)        \u00d7 6 Jaccard thresholds
  * Fig S8 \u2014 combined pool   (min_steps=1)        \u00d7 6 Jaccard thresholds
  * **Fig S8** \u2014 FULL pool (no minimums on visit blocks OR token count)
                \u00d7 6 Jaccard thresholds

Reading the three figures in sequence shows two complementary stories:

  * **Within each figure** (j-sweep): defends the locked j=0.2 dedup
    \u2014 looser thresholds preserve near-duplicates that monopolize the
    top-N; tighter ones over-collapse phenotypes.
  * **Across figures** (S7 \u2192 S8 \u2192 S9): defends the locked min_steps=2 +
    min_tokens=2 \u2014 single-visit multi-token co-occurrence patterns
    monotonically dominate the top-N as the pool inclusion criteria
    relax. Cross-visit narratives only emerge when min_steps=2 is
    enforced (S7).

The full pool here includes every distinct pattern from the raw mining
cache:

  * single-token single-block   ``[{A}]``                 (\u2248 Panel B)
  * single-token multi-block    ``[{A}, {A}, \u2026]``         (token repetition)
  * multi-token  single-block   ``[{A, B, \u2026}]``           (co-occurrence)
  * multi-token  multi-block    ``[{A, \u2026}, {B, \u2026}, \u2026]`` (Panel C shape)

Usage:
    python scripts/make_fig_s8_full_pool_sensitivity.py \
        --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.plots import signed_ig_bars
from atop.explain.pattern_labels import PatternLabelResolver
from atop.explain.cooccurrence import (
    score_all_raw_patterns, cluster_jaccard_inmemory,
)
from atop.run_layout import RunLayout


N_SHOW = 30
FONTSIZE = 9      # page-sized layout: 9 pt prints at 9 pt (was 16 for the
                  # old 48-inch stitched canvas, which then got downscaled)


def _bold(s: str) -> str:
    return r"$\bf{" + s.replace(" ", r"\ ").replace(":", r"{:}") + r"}$"


def _select_top_n(df_pool: pd.DataFrame, rep_patterns: set, n: int) -> pd.DataFrame:
    reps = df_pool[df_pool["pattern"].isin(rep_patterns)].copy()
    reps["_abs"] = reps["ig_signed_mean"].abs()
    return reps.nlargest(n, "_abs").sort_values("_abs", ascending=True).drop(columns="_abs")


def _render_panel(df_sel: pd.DataFrame, resolver: PatternLabelResolver,
                   threshold_label: str, n_reps: int, panel_letter: str,
                   fontsize: int, panel_height: float,
                   panel_fig_width: float, target_bar_inches: float,
                   tmp_dir: str) -> str:
    labels = [resolver.format_pattern(p, short_names=True)
              for p in df_sel["pattern"]]
    values = list(df_sel["ig_signed_mean"])

    fig, ax = plt.subplots(figsize=(panel_fig_width, panel_height))
    signed_ig_bars(values=values, labels=labels, ax=ax, fontsize=fontsize)

    fs_title = fontsize + 2
    fs_xlabel = fontsize - 1
    sub = (f"top-{len(df_sel)} patterns from {n_reps:,} representative"
           f"{'s' if n_reps != 1 else ''} (full pool)")
    head = f"{panel_letter}. {threshold_label}"
    ax.set_title(_bold(head) + "\n" + sub, fontsize=fs_title, pad=15)
    ax.set_xlabel("E[\u03a3IG | present]\n\u2190 protective    risk \u2192",
                   fontsize=fs_xlabel)
    ax.tick_params(axis="x", labelsize=fs_xlabel - 1)

    fig.tight_layout()
    pos = ax.get_position()
    target_bar_frac = target_bar_inches / panel_fig_width
    new_right = pos.x1
    new_left = new_right - target_bar_frac
    ax.set_position([new_left, pos.y0, target_bar_frac, pos.height])

    tmp = tempfile.NamedTemporaryFile(prefix=f"fig_s8_{panel_letter}_",
                                       suffix=".png", dir=tmp_dir, delete=False)
    fig.savefig(tmp.name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render Fig S8 (full-pool Jaccard sensitivity).")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--n_show", type=int, default=N_SHOW)
    p.add_argument("--min_carriers", type=int, default=5)
    p.add_argument("--fontsize", type=int, default=FONTSIZE)
    p.add_argument("--fig_width", type=float, default=7.7,
                   help="Total figure width in inches (page-sized: the "
                        "shared-label layout needs no downscaling).")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/"
                           "FigureS8_full_pool_jaccard_sensitivity.png")
    args = p.parse_args()

    L = RunLayout(args.run_dir)

    # 1. Build the FULL pool from the raw mining cache (no minimums on
    #    blocks or tokens). This is the slow step (~30 s).
    print("[1] Loading raw mining cache + sequences + IG\u2026")
    # Prefer the scored-pool cache written by the main pipeline: rescoring
    # re-reads the full IG table (~173 MB) and repeats work already done.
    cache = os.path.join(L.analysis_dir, f"full_pool_scored_{args.split}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            df_pool = pickle.load(f)
        print(f"\n[2] Loaded cached scored pool: {len(df_pool):,} patterns")
    else:
        raw_files = sorted([f for f in os.listdir(L.mining_dir)
                            if f.startswith("raw_episode_")])
        if not raw_files:
            sys.exit(f"  [ERROR] No raw_episode_*.pkl in {L.mining_dir}")
        with open(os.path.join(L.mining_dir, raw_files[-1]), "rb") as f:
            raw_mining = pickle.load(f)
        with open(L.sequences_pkl(args.split), "rb") as f:
            df_seq = pickle.load(f)
        df_ig = pd.read_csv(L.ig_csv(args.split))
        print("\n[2] Scoring every distinct pattern in the raw cache\u2026")
        df_pool = score_all_raw_patterns(
            raw_mining, df_seq, df_ig,
            min_carriers=args.min_carriers,
            min_blocks=1,
            min_tokens_total=1,
        )
    if df_pool.empty:
        sys.exit("  [ERROR] No patterns scored.")

    # 2. For each of 6 thresholds, get reps.
    print("\n[3] Building representative sets at each threshold\u2026")
    threshold_specs = [
        ("A", "No Jaccard dedup",                              None),
        ("B", "Jaccard \u2265 0.8 (near-identity merges only)",  0.8),
        ("C", "Jaccard \u2265 0.6",                              0.6),
        ("D", "Jaccard \u2265 0.4",                              0.4),
        ("E", "Jaccard \u2265 0.2 (locked \u2014 main paper)",   0.2),
        ("F", "Jaccard \u2265 0.1 (tighter than locked)",        0.1),
    ]
    # The Jaccard sweeps dominate runtime (~40 min for the full pool), and
    # they depend only on the carrier sets, so cache them.
    rs_cache = os.path.join(L.analysis_dir, f"s8_repsets_{args.split}.pkl")
    if os.path.exists(rs_cache) and os.path.getmtime(rs_cache) >= os.path.getmtime(cache):
        with open(rs_cache, "rb") as f:
            rep_sets = pickle.load(f)
        print(f"\n[3] Loaded cached representative sets ({len(rep_sets)} thresholds)")
        for letter, label, reps in rep_sets:
            print(f"    {letter}: {label:<55s} \u2192 {len(reps):>6,} representatives")
        threshold_specs = []
    rep_sets = rep_sets if not threshold_specs else []
    for letter, label, thresh in threshold_specs:
        if thresh is None:
            reps = set(df_pool["pattern"].tolist())
        else:
            reps = cluster_jaccard_inmemory(df_pool, thresh, verbose=False)
        print(f"    {letter}: {label:<55s} \u2192 {len(reps):>6,} representatives")
        rep_sets.append((letter, label, reps))

    if threshold_specs:                     # only write when freshly computed
        with open(rs_cache, "wb") as f:
            pickle.dump(rep_sets, f)
        print(f"    cached representative sets \u2192 {os.path.basename(rs_cache)}")

    # 3. Render: ONE shared label column + narrow bar strips.
    #
    # The previous layout drew six self-contained panels and stitched them,
    # which repeated every pattern label six times. Only 43 distinct patterns
    # appear across the six top-30 sets, so 76% of the drawn text was
    # duplicated - and that redundancy forced a ~42-inch canvas, which is why
    # the type became unreadable once scaled to a page. Sharing one label
    # column collapses the canvas to page size, so the figure prints at its
    # true font size, and it aligns rows across thresholds so ranking
    # stability can be read horizontally.
    #
    # Panels B and D are omitted from the rendered figure: their top-30 sets
    # are identical to C (30/30 overlap), so they add width without adding
    # information. All six remain in the backing CSV.
    resolver = PatternLabelResolver(args.run_dir, verbose=False)
    SHOW = ["A", "C", "E", "F"]
    # Narrow strips cannot hold the full threshold strings; keep them short
    # and put the full wording in the caption.
    SHORT = {"A": "A. No dedup", "B": "B. J \u2265 0.8", "C": "C. J \u2265 0.6",
             "D": "D. J \u2265 0.4", "E": "E. J \u2265 0.2 (locked)",
             "F": "F. J \u2265 0.1"}
    sel = [(l, lab, reps) for (l, lab, reps) in rep_sets if l in SHOW]

    per_panel = {}
    for letter, label, reps in sel:
        d = _select_top_n(df_pool, reps, args.n_show)
        per_panel[letter] = (label, len(reps), d.set_index("pattern"))

    # Row order: union of the displayed sets, ranked by peak |SIG|.
    peak = {}
    for _, (_, _, d) in per_panel.items():
        for pat, r in d.iterrows():
            peak[pat] = max(peak.get(pat, 0.0), abs(r["ig_signed_mean"]))
    rows_order = sorted(peak, key=peak.get, reverse=True)
    n_rows = len(rows_order)
    print(f"\n[4] Shared-label layout: {n_rows} rows x {len(sel)} thresholds "
          f"({sum(len(d) for _,_,d in per_panel.values())} bars)")

    RISK, PROT = "#c0392b", "#2c6fbb"
    vmax = max(abs(df_pool["ig_signed_mean"].loc[
        df_pool["pattern"].isin(rows_order)])) * 1.15
    fig_h = max(6.0, 0.205 * n_rows + 1.4)
    fig = plt.figure(figsize=(args.fig_width, fig_h), dpi=args.dpi)
    gs = fig.add_gridspec(1, len(sel), left=0.40, right=0.985,
                          top=0.905, bottom=0.055, wspace=0.14)

    for ci, letter in enumerate(SHOW):
        label, n_reps, d = per_panel[letter]
        ax = fig.add_subplot(gs[0, ci])
        for ri, pat in enumerate(rows_order):
            y = n_rows - ri
            if pat in d.index:
                v = float(d.loc[pat, "ig_signed_mean"])
                ax.barh(y, v, height=0.72,
                        color=(RISK if v > 0 else PROT), linewidth=0)
            else:
                # absent at this threshold: visible marker, not a silent gap
                ax.plot([0], [y], marker="d", ms=2.6, color="#9a9a9a",
                        zorder=3)
        ax.axvline(0, color="#444444", lw=0.6)
        ax.set_xlim(-vmax, vmax)
        ax.set_ylim(0.4, n_rows + 0.6)
        ax.set_title(f"{SHORT[letter]}\n({n_reps:,} reps)",
                     fontsize=args.fontsize - 1.5, fontweight="bold", pad=6)
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(axis="x", labelsize=args.fontsize - 2.5, pad=1)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", lw=0.3, color="#dddddd")
        ax.set_axisbelow(True)
        if ci == 0:
            for ri, pat in enumerate(rows_order):
                ax.text(-vmax * 1.08, n_rows - ri,
                        resolver.format_pattern(pat, short_names=True),
                        ha="right", va="center", fontsize=args.fontsize - 2.2)

    fig.text(0.5, 0.982,
             "Jaccard sensitivity \u2014 full pattern pool, no shape minimums",
             ha="center", fontsize=args.fontsize + 2, fontweight="bold")
    fig.text(0.5, 0.951,
             "Rows are the union of the displayed top-30 sets (n = %d), ordered "
             "by peak |\u03a3IG|; a grey diamond marks a pattern absent at that "
             "threshold. Panels B and D are omitted (identical to C)." % n_rows,
             ha="center", fontsize=args.fontsize - 2.5, color="#444444")
    fig.text(0.5, 0.016,
             "E[\u03a3 IG | present]      \u2190 protective    risk \u2192",
             ha="center", fontsize=args.fontsize - 1.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color=RISK, lw=5, label="risk-supporting"),
                        Line2D([0], [0], color=PROT, lw=5, label="protective")],
               loc="lower left", bbox_to_anchor=(0.02, 0.0), frameon=False,
               fontsize=args.fontsize - 2.2)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(args.output_path) // 1024
    print(f"[5] Saved figure: {args.output_path} ({size_kb} KB)")

    # 5. Backing CSV.
    rows = []
    for letter, label, reps in rep_sets:
        df_sel = _select_top_n(df_pool, reps, args.n_show).iloc[::-1]
        for rank, (_, r) in enumerate(df_sel.iterrows(), start=1):
            pat = r["pattern"]
            blocks = pat.split(" -> ")
            n_blocks = len(blocks)
            n_tokens = sum(len(b[1:-1].split(", ")) if b.startswith("{") else 1 for b in blocks)
            rows.append({
                "panel":             letter,
                "threshold_label":   label,
                "n_representatives": len(reps),
                "rank":              rank,
                "pattern":           pat,
                "pattern_readable":  resolver.format_pattern(pat, short_names=True),
                "ig_signed_mean":    r["ig_signed_mean"],
                "n_present":         int(r.get("n_present", 0)),
                "n_blocks":          n_blocks,
                "n_tokens_total":    n_tokens,
                "is_single_block":   n_blocks == 1,
                "is_single_token":   n_tokens == 1,
            })
    out_csv = os.path.splitext(args.output_path)[0] + ".csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"    csv: {out_csv}")


if __name__ == "__main__":
    main()
