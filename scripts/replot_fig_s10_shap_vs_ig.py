"""Replot supplement Figure S10 (per-patient KernelSHAP vs IG agreement) from
the CSVs that make_fig_s3.py wrote, without recomputing SHAP.

Panel A (one exemplar patient) uses independent axis scales because IG is on
the logit scale and KernelSHAP on the probability scale; a shared y = x line
flattened every SHAP value onto zero in the previous rendering. Panel B is
z-scored on both axes, as before.

Usage:
    python scripts/replot_fig_s10_shap_vs_ig.py --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

RISK, PROT = "#c44e52", "#4c72b0"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/FigureS3_shap_vs_ig.png")
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()
    adir = os.path.join(a.run_dir, "5_analysis")
    pa = pd.read_csv(os.path.join(adir, "fig_s3_patient_level.csv"))
    agg = pd.read_csv(os.path.join(adir, "fig_s3_aggregate.csv"))

    r_a, _ = pearsonr(pa["ig_signed"], pa["shap_signed"])
    rho_a, _ = spearmanr(pa["ig_signed"], pa["shap_signed"])
    ig_z = (agg["ig_signed"] - agg["ig_signed"].mean()) / agg["ig_signed"].std()
    sh_z = (agg["shap_signed"] - agg["shap_signed"].mean()) / agg["shap_signed"].std()
    r_b, _ = pearsonr(agg["ig_signed"], agg["shap_signed"])
    rho_b, _ = spearmanr(agg["ig_signed"], agg["shap_signed"])
    jac = []
    for _, g in agg.groupby(["patient_id", "index_hadm_id"]):
        ti = set(g.assign(x=g["ig_signed"].abs()).nlargest(5, "x")["token_str"])
        ts = set(g.assign(x=g["shap_signed"].abs()).nlargest(5, "x")["token_str"])
        if ti | ts:
            jac.append(len(ti & ts) / len(ti | ts))
    n_pat = agg.groupby(["patient_id", "index_hadm_id"]).ngroups
    print(f"Panel A: n={len(pa)} r={r_a:.3f} rho={rho_a:.3f}")
    print(f"Panel B: n={len(agg)} patients={n_pat} r={r_b:.3f} rho={rho_b:.3f} Jaccard@5={np.mean(jac):.2f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"wspace": 0.30})

    cols = [RISK if v > 0 else PROT for v in pa["ig_signed"]]
    axA.scatter(pa["ig_signed"], pa["shap_signed"], s=85, c=cols, edgecolor="black",
                linewidth=0.5, alpha=0.9, zorder=3)
    lx = pa["ig_signed"].abs().max() * 1.2
    ly = pa["shap_signed"].abs().max() * 1.2
    axA.axhline(0, color="#888888", linewidth=0.6, zorder=1)
    axA.axvline(0, color="#888888", linewidth=0.6, zorder=1)
    axA.set_xlim(-lx, lx); axA.set_ylim(-ly, ly)
    lab = pa.assign(x=pa["ig_signed"].abs()).nlargest(6, "x")
    for _, r in lab.iterrows():
        axA.annotate(str(r["token_readable"]), (r["ig_signed"], r["shap_signed"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbbbbb", lw=0.5))
    axA.set_xlabel("Integrated Gradients (signed, logit scale)", fontsize=12, fontweight="bold")
    axA.set_ylabel("KernelSHAP (signed, probability scale)", fontsize=12, fontweight="bold")
    axA.set_title(f"A · Exemplar patient: top {len(pa)} tokens by |IG|\n"
                  f"Pearson r = {r_a:.3f}    Spearman ρ = {rho_a:.3f}",
                  fontsize=12, loc="left", pad=10)
    axA.grid(True, color="#eeeeee", linewidth=0.5)

    colsb = [RISK if v > 0 else PROT for v in agg["ig_signed"]]
    axB.scatter(ig_z, sh_z, s=18, c=colsb, alpha=0.55, zorder=3)
    lim = max(ig_z.abs().max(), sh_z.abs().max()) * 1.05
    axB.plot([-lim, lim], [-lim, lim], color="#888888", linestyle="--", linewidth=1.0,
             label="y = x (z-scored)", zorder=1)
    axB.axhline(0, color="#dddddd", linewidth=0.6); axB.axvline(0, color="#dddddd", linewidth=0.6)
    axB.set_xlim(-lim, lim); axB.set_ylim(-lim, lim)
    axB.set_xlabel("Integrated Gradients (z-score)", fontsize=12, fontweight="bold")
    axB.set_ylabel("KernelSHAP (z-score)", fontsize=12, fontweight="bold")
    axB.set_title(f"B · Aggregate: {n_pat} test patients × top-10\n"
                  f"n = {len(agg)} pairs    Pearson r = {r_b:.3f}    Spearman ρ = {rho_b:.3f}"
                  f"    Per-patient Jaccard@5 = {np.mean(jac):.2f} (mean)",
                  fontsize=12, loc="left", pad=10)
    axB.legend(loc="lower right", fontsize=10)
    axB.grid(True, color="#eeeeee", linewidth=0.5)
    for ax in (axA, axB):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    os.makedirs(os.path.dirname(a.output_path), exist_ok=True)
    fig.savefig(a.output_path, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved {a.output_path} ({os.path.getsize(a.output_path)//1024} KB)")


if __name__ == "__main__":
    main()
