"""make_fig_s10_reversal_scatter, Supplementary Figure S10.

Forward-order versus reversed-order carrier-averaged ΣIG for every
cross-visit pattern in the reversal pool (at least 20 test patients
matching each ordering), with the Table 4 patterns overlaid. Replaces
the earlier top-30-by-attenuation bar chart, which showed a selection by
the statistic being claimed; this view shows the pool statistics the main
text cites (share attenuated, share with sign preserved).

Inputs (produced by the pipeline and the reversed-order top-up):
  <run_dir>/5_analysis/reversed_order_test.csv
  <run_dir>/5_analysis/reversed_order_whitelist_manifest.csv  (in_table4)

Usage:
    python paper/jamia2026/figures/make_fig_s10_reversal_scatter.py \
        --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_RISK = "#c44e52"
COLOR_PROT = "#4c72b0"
MIN_N = 20


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--output-dir",
                   default="../manuscript/jamia_submission/figures/supplements")
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()

    adir = os.path.join(a.run_dir, "5_analysis")
    rev = pd.read_csv(os.path.join(adir, "reversed_order_test.csv"))
    man = pd.read_csv(os.path.join(adir, "reversed_order_whitelist_manifest.csv"))
    t4 = set(man.loc[man["in_table4"] == True, "pattern"])

    pool = rev[(rev["n_forward"] >= MIN_N) & (rev["n_reversed"] >= MIN_N)].copy()
    x, y = pool["ig_mean_forward"].to_numpy(), pool["ig_mean_reversed"].to_numpy()
    attenuated = np.abs(x) > np.abs(y)
    sign_kept = np.sign(x) == np.sign(y)
    print(f"pool n={len(pool)}  attenuated {attenuated.mean():.1%}  "
          f"sign preserved {sign_kept.mean():.1%}  "
          f"median attenuation {np.median(1 - np.abs(y) / np.abs(x)) * 100:.0f}%")

    rep = rev[rev["pattern"].isin(t4)].copy()
    rep = rep[rep["pattern"] != rep["reversed_pattern"]]   # drop palindromes
    rx, ry = rep["ig_mean_forward"].to_numpy(), rep["ig_mean_reversed"].to_numpy()
    print(f"Table 4 non-palindromic patterns overlaid: {len(rep)}; "
          f"attenuated {(np.abs(rx) > np.abs(ry)).sum()} of {len(rep)}")

    lim = float(max(np.abs(np.r_[x, y, rx, ry]).max() * 1.06, 0.1))
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(6.5, 6.3))
    risk = x > 0
    ax.scatter(x[~risk], y[~risk], s=10, alpha=0.45, color=COLOR_PROT, lw=0,
               label="Pool, protective (forward ΣIG < 0)")
    ax.scatter(x[risk], y[risk], s=10, alpha=0.45, color=COLOR_RISK, lw=0,
               label="Pool, risk-increasing (forward ΣIG > 0)")
    rrisk = rx > 0
    ax.scatter(rx[~rrisk], ry[~rrisk], s=46, facecolor=COLOR_PROT, edgecolor="black",
               lw=0.8, label="Table 4 pattern, protective")
    ax.scatter(rx[rrisk], ry[rrisk], s=46, facecolor=COLOR_RISK, edgecolor="black",
               lw=0.8, label="Table 4 pattern, risk-increasing")
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.45", lw=1,
            label="y = x (no change under reversal)")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Forward order: carrier-averaged ΣIG")
    ax.set_ylabel("Reversed order: carrier-averaged ΣIG")
    ax.text(0.03, 0.97,
            f"Pool: n = {len(pool):,} cross-visit patterns\n"
            f"(≥20 patients matching each ordering)\n"
            f"|ΣIG| lower when reversed: {attenuated.mean() * 100:.0f}%\n"
            f"sign preserved: {sign_kept.mean() * 100:.0f}%",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    os.makedirs(a.output_dir, exist_ok=True)
    out = os.path.join(a.output_dir, "FigureS10_reversal_scatter.png")
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
