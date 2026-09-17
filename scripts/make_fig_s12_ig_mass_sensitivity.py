"""make_fig_s12_ig_mass_sensitivity, Sensitivity of AToP downstream
analysis to the IG saliency-mass threshold.

The default AToP pipeline marks a token as "salient" if it falls in the
top-cumulative-|IG| set capturing >= 80% of the patient's total |IG|
(``cfg.ig_mass = 0.8`` in atop/config.py, applied in
atop/attribution/saliency.py::_select_salient). The threshold is borrowed
from PCA-style explained-variance conventions; this script empirically
checks robustness of the downstream conclusions to that choice.

For each tau in {0.60, 0.70, 0.80, 0.90, 0.95}:
  1. Re-derive per-token is_salient by re-applying _select_salient with
     the new mass target on the cached IG.
  2. Rebuild salient_visit_blocks per test patient.
  3. Match the EXISTING mined pattern dictionary against the new salient
     blocks via PatternIndex (no re-mining; we hold the candidate space
     fixed and vary only the downstream saliency cutoff).
  4. Compute per-pattern carrier set and signed sigma_IG (sum of token
     ig_signed across carrier patient instances).
  5. Identify top-K risk and protective patterns.

Caveat surfaced in the figure caption: this analysis varies the saliency
cutoff used downstream (matching, ranking) while holding the upstream
mined pattern pool fixed (mined once at tau=0.80 from the training cohort).
A full sensitivity that re-mines at each tau would multiply mining cost by
the number of thresholds; the downstream-only sweep covers the conclusions
the manuscript draws from the mined pool.

Plots a 2x2:
  A. Pattern matching count vs tau (matched on >=1 test patient)
  B. Per-patient salient token count distribution per tau (boxplot)
  C. Top-K risk pattern Jaccard overlap matrix across thresholds
  D. Top-K protective pattern Jaccard overlap matrix across thresholds

Output:
  manuscript/jamia_submission/figures/supplements/FigureS12_ig_mass_sensitivity.png
  runs/<run>/5_analysis/fig_s12_ig_mass_sensitivity.csv (raw counts per tau)

Usage:
    python scripts/make_fig_s12_ig_mass_sensitivity.py --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.explain.matching import PatternIndex
from atop.mining.patterns import _parse_episode_pattern
from atop.run_layout import RunLayout


THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95]
TOP_K = 30
MAX_TOKENS_HARD_CAP = 50  # mirrors cfg.ig_max_tokens


# ── Re-derive is_salient at a new tau (mirrors _select_salient) ─────────
def recompute_salient_per_patient(df_ig: pd.DataFrame, tau: float,
                                    max_tokens: int = MAX_TOKENS_HARD_CAP
                                    ) -> pd.DataFrame:
    """Returns a copy of df_ig with `is_salient` recomputed at threshold tau.
    Mirrors atop/attribution/saliency.py::_select_salient: sort by |IG|
    desc, cumulate until cum/total >= tau, also cap at max_tokens."""
    out = df_ig.copy()
    out["is_salient"] = False
    for (pid, hadm), g in out.groupby(["patient_id", "index_hadm_id"], sort=False):
        igs = g["ig_abs"].values
        tot = igs.sum() + 1e-12
        order = np.argsort(-igs)
        keep = np.zeros(len(g), dtype=bool)
        cum = 0.0
        for j in order:
            keep[j] = True
            cum += igs[j]
            if cum / tot >= tau:
                break
            if max_tokens and keep.sum() >= max_tokens:
                break
        out.loc[g.index, "is_salient"] = keep
    return out


def build_salient_visit_blocks(df_ig_salient: pd.DataFrame) -> pd.DataFrame:
    """For each (pid, hadm), construct salient_visit_blocks = list of
    frozensets, one per visit_idx, of the salient token_strs at that visit."""
    rows = []
    for (pid, hadm), g in df_ig_salient.groupby(["patient_id", "index_hadm_id"], sort=False):
        sal = g[g["is_salient"]]
        n_visits = int(g["visit_idx"].max() + 1) if len(g) else 0
        blocks = []
        for vi in range(n_visits):
            tokens = set(sal[sal["visit_idx"] == vi]["token_str"].tolist())
            if tokens:
                blocks.append(frozenset(tokens))
        rows.append({"patient_id": pid, "index_hadm_id": hadm,
                      "salient_visit_blocks": blocks,
                      "n_salient": int(sal.shape[0])})
    return pd.DataFrame(rows)


def score_patterns_at_tau(df_patterns: pd.DataFrame,
                            df_blocks: pd.DataFrame,
                            df_ig_salient: pd.DataFrame,
                            tau: float) -> pd.DataFrame:
    """Match each mined pattern against the new salient blocks; return
    per-pattern signed sigma_IG (mean across carriers)."""
    print(f"  [tau={tau:.2f}] indexing {len(df_blocks):,} test patients…")
    idx = PatternIndex.from_df(df_blocks)

    # Per-(pid, hadm) → dict {token_str: signed_ig_sum_at_salient_positions}
    by_patient_tok = defaultdict(lambda: defaultdict(float))
    sal = df_ig_salient[df_ig_salient["is_salient"]]
    for _, r in sal.iterrows():
        by_patient_tok[(r["patient_id"], r["index_hadm_id"])][r["token_str"]] \
            += float(r["ig_signed"])

    blocks_by_row = list(zip(df_blocks["patient_id"].tolist(),
                                df_blocks["index_hadm_id"].tolist()))

    print(f"  [tau={tau:.2f}] matching {len(df_patterns):,} patterns…")
    t0 = time.time()
    out_rows = []
    for i, r in df_patterns.iterrows():
        pat_str = r["pattern"]
        ep = _parse_episode_pattern(pat_str)
        ep_fsets = [frozenset(s) for s in ep]
        pat_tokens = set()
        for s in ep:
            pat_tokens.update(s)

        matched_rows = idx.patients_matching_pattern(ep_fsets, pat_tokens, is_episode=True)
        n_match = len(matched_rows)
        if n_match == 0:
            out_rows.append({"pattern": pat_str, "n_match": 0,
                             "sigma_ig_mean": 0.0, "sigma_ig_sum": 0.0})
            continue
        # Per-carrier σIG = sum of (ig_signed at salient positions) for each
        # token in the pattern. Average across carriers for the per-pattern σ.
        sums = []
        for ri in matched_rows:
            key = blocks_by_row[ri]
            tok_map = by_patient_tok.get(key, {})
            s = sum(tok_map.get(t, 0.0) for t in pat_tokens)
            sums.append(s)
        sigma_mean = float(np.mean(sums))
        sigma_sum  = float(np.sum(sums))
        out_rows.append({"pattern": pat_str, "n_match": n_match,
                         "sigma_ig_mean": sigma_mean,
                         "sigma_ig_sum":  sigma_sum})
        if (i + 1) % 5000 == 0:
            print(f"    {i+1:>6,}/{len(df_patterns):,} patterns scored "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"  [tau={tau:.2f}] done in {time.time()-t0:.0f}s")
    return pd.DataFrame(out_rows)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure S12 (IG-mass sensitivity)")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/"
                           "FigureS12_ig_mass_sensitivity.png")
    p.add_argument("--csv-path", default=None)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    csv_path = args.csv_path or os.path.join(layout.analysis_dir,
                                                f"fig_s12_ig_mass_sensitivity.csv")

    print(f"[1] Loading IG ({args.split}) + mined pattern dictionary…")
    df_ig_orig = pd.read_csv(layout.ig_csv(args.split),
                               dtype={"patient_id": str, "index_hadm_id": str})
    print(f"    IG: {len(df_ig_orig):,} rows, "
          f"{df_ig_orig[['patient_id','index_hadm_id']].drop_duplicates().shape[0]:,} patients")

    pat_files = sorted(glob.glob(os.path.join(layout.mining_dir, "all_scored_*.csv")),
                        key=os.path.getmtime)
    df_patterns = pd.read_csv(pat_files[-1])
    print(f"    Patterns: {len(df_patterns):,} (from {os.path.basename(pat_files[-1])})")

    # ── Per-tau sweep ────────────────────────────────────────────────
    summary_rows = []
    top_risk_sets = {}
    top_prot_sets = {}
    salient_counts = {}

    for tau in THRESHOLDS:
        print(f"\n[2] tau = {tau:.2f}")
        df_sal = recompute_salient_per_patient(df_ig_orig, tau)
        n_sal_per_pat = df_sal[df_sal["is_salient"]].groupby(
            ["patient_id", "index_hadm_id"]).size().values
        salient_counts[tau] = n_sal_per_pat
        print(f"  median salient tokens per patient: {int(np.median(n_sal_per_pat))} "
              f"(IQR {int(np.quantile(n_sal_per_pat, 0.25))}–"
              f"{int(np.quantile(n_sal_per_pat, 0.75))})")

        df_blocks = build_salient_visit_blocks(df_sal)
        n_with_blocks = (df_blocks["salient_visit_blocks"].apply(len) > 0).sum()
        print(f"  patients with >=1 salient block: {n_with_blocks:,}/{len(df_blocks):,}")

        df_scored = score_patterns_at_tau(df_patterns, df_blocks, df_sal, tau)
        n_matched_any = int((df_scored["n_match"] > 0).sum())
        n_risk = int((df_scored["sigma_ig_mean"] > 0).sum())
        n_prot = int((df_scored["sigma_ig_mean"] < 0).sum())

        # Top-K by absolute mean σIG, on each side
        risk = df_scored[df_scored["sigma_ig_mean"] > 0].sort_values(
            "sigma_ig_mean", ascending=False).head(args.top_k)
        prot = df_scored[df_scored["sigma_ig_mean"] < 0].sort_values(
            "sigma_ig_mean").head(args.top_k)
        top_risk_sets[tau] = set(risk["pattern"].tolist())
        top_prot_sets[tau] = set(prot["pattern"].tolist())

        summary_rows.append({
            "tau": tau,
            "median_salient_per_patient": int(np.median(n_sal_per_pat)),
            "p25_salient": int(np.quantile(n_sal_per_pat, 0.25)),
            "p75_salient": int(np.quantile(n_sal_per_pat, 0.75)),
            "n_patients_with_blocks": int(n_with_blocks),
            "n_patterns_matched_any":  n_matched_any,
            "n_risk_patterns":  n_risk,
            "n_prot_patterns":  n_prot,
        })

    df_summary = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_summary.to_csv(csv_path, index=False)
    print(f"\n[3] Summary saved → {csv_path}")
    print(df_summary.to_string(index=False))

    # ── Compute Jaccard overlap matrices ──────────────────────────────
    n = len(THRESHOLDS)
    risk_jac = np.zeros((n, n))
    prot_jac = np.zeros((n, n))
    for i, ti in enumerate(THRESHOLDS):
        for j, tj in enumerate(THRESHOLDS):
            risk_jac[i, j] = jaccard(top_risk_sets[ti], top_risk_sets[tj])
            prot_jac[i, j] = jaccard(top_prot_sets[ti], top_prot_sets[tj])

    # ── Render 2x2 figure ─────────────────────────────────────────────
    print("\n[4] Rendering…")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Panel A: pattern count vs tau
    axA = axes[0, 0]
    axA.plot(df_summary["tau"], df_summary["n_patterns_matched_any"],
             "o-", color="#222", lw=1.6, markersize=7, label="Patterns matched")
    axA.plot(df_summary["tau"], df_summary["n_risk_patterns"],
             "o--", color="#ca0020", lw=1.2, markersize=5, label="Risk (σIG > 0)")
    axA.plot(df_summary["tau"], df_summary["n_prot_patterns"],
             "o--", color="#2166ac", lw=1.2, markersize=5, label="Protective (σIG < 0)")
    axA.axvline(0.80, color="#888", linestyle=":", lw=1.0)
    axA.text(0.80, axA.get_ylim()[1] * 0.95, "  default τ=0.80",
             fontsize=8, color="#666", va="top", ha="left")
    axA.set_xlabel("IG saliency mass threshold τ")
    axA.set_ylabel("Number of patterns")
    axA.set_title("A · Patterns surviving downstream matching",
                   loc="left", fontsize=11.5, fontweight="bold")
    axA.legend(fontsize=8.5, loc="best")
    axA.grid(True, color="#eeeeee", lw=0.5)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # Panel B: per-patient salient count distribution
    axB = axes[0, 1]
    box_data = [salient_counts[t] for t in THRESHOLDS]
    bp = axB.boxplot(box_data, positions=THRESHOLDS, widths=0.05,
                       patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#fde0c0")
        patch.set_edgecolor("#222222")
    for el in bp["medians"]:
        el.set_color("#000")
        el.set_linewidth(1.2)
    axB.axvline(0.80, color="#888", linestyle=":", lw=1.0)
    axB.set_xlabel("IG saliency mass threshold τ")
    axB.set_ylabel("Salient tokens per patient")
    axB.set_xlim(0.55, 1.0)
    axB.set_xticks(THRESHOLDS)
    axB.set_xticklabels([f"{t:.2f}" for t in THRESHOLDS])
    axB.set_title("B · Per-patient salient-token count distribution",
                   loc="left", fontsize=11.5, fontweight="bold")
    axB.grid(True, axis="y", color="#eeeeee", lw=0.5)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    # Panel C: top-K risk Jaccard overlap heatmap
    axC = axes[1, 0]
    im = axC.imshow(risk_jac, cmap="Reds", vmin=0, vmax=1, aspect="equal")
    axC.set_xticks(range(n))
    axC.set_yticks(range(n))
    axC.set_xticklabels([f"{t:.2f}" for t in THRESHOLDS])
    axC.set_yticklabels([f"{t:.2f}" for t in THRESHOLDS])
    axC.set_xlabel("τ")
    axC.set_ylabel("τ")
    axC.set_title(f"C · Top-{args.top_k} risk-pattern Jaccard overlap",
                   loc="left", fontsize=11.5, fontweight="bold")
    for i in range(n):
        for j in range(n):
            color = "white" if risk_jac[i, j] > 0.5 else "#222"
            axC.text(j, i, f"{risk_jac[i, j]:.2f}",
                     ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, ax=axC, fraction=0.046, pad=0.04, label="Jaccard")

    # Panel D: top-K protective Jaccard overlap heatmap
    axD = axes[1, 1]
    im = axD.imshow(prot_jac, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    axD.set_xticks(range(n))
    axD.set_yticks(range(n))
    axD.set_xticklabels([f"{t:.2f}" for t in THRESHOLDS])
    axD.set_yticklabels([f"{t:.2f}" for t in THRESHOLDS])
    axD.set_xlabel("τ")
    axD.set_ylabel("τ")
    axD.set_title(f"D · Top-{args.top_k} protective-pattern Jaccard overlap",
                   loc="left", fontsize=11.5, fontweight="bold")
    for i in range(n):
        for j in range(n):
            color = "white" if prot_jac[i, j] > 0.5 else "#222"
            axD.text(j, i, f"{prot_jac[i, j]:.2f}",
                     ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, ax=axD, fraction=0.046, pad=0.04, label="Jaccard")

    # No in-image suptitle prefix; the DOCX caption supplies "Figure S12. ..."
    fig.suptitle(f"Sensitivity to IG saliency-mass threshold τ "
                  f"(test cohort, top-{args.top_k} pattern stability)",
                  fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    sz = os.path.getsize(args.output_path) // 1024
    print(f"[5] Saved figure → {args.output_path} ({sz} KB)")


if __name__ == "__main__":
    main()
