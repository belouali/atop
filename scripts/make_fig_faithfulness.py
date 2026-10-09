"""make_fig_faithfulness, quantitative faithfulness curves for AToP.

Answers reviewer concern #5 (clinical-value claims lack a faithfulness
test). Measures whether AToP's IG-ranked salient tokens are the tokens
the model actually relies on, by progressive masking:

  COMPREHENSIVENESS: remove the top-k IG tokens (highest |IG| first) and
  measure how far the predicted probability moves. If AToP is faithful,
  removing its top-ranked tokens should move the prediction much more
  than removing the same number of randomly chosen salient tokens.

  SUFFICIENCY: keep ONLY the top-k IG tokens (mask everything else) and
  measure how close the prediction stays to the original. If the top-k
  are sufficient, the prediction should be largely recovered by a small k.

Both are contrasted against a matched-random baseline (same number of
tokens removed/kept, drawn at random from the patient's salient set),
which is the ERASER-style faithfulness protocol (DeYoung et al. 2020).

This validates faithfulness-to-the-model (AToP surfaces what the model
uses), NOT clinical correctness (that what the model uses is clinically
right); the latter needs separate clinician review.

Reuses the masking + batched-prediction primitives from
atop.explain.validation.

Usage (CPU is fine on a sampled subset):
    python scripts/make_fig_faithfulness.py --run_dir ../runs/mimiciv_readmission_30d \\
        --mimic_dir ../data/mimiciv/3.1/hosp --sample_n 500 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.config import PAD_IDX
from atop.explain.validation import mask_positions, predict_batch
from atop.models.single_stream_transformer import SingleStreamTransformer
from atop.run_layout import RunLayout


RISK = "#c44e52"     # AToP curve (canonical risk red)
RANDOM = "#999999"   # random baseline (neutral gray)
KEEP = "#4c72b0"     # sufficiency / kept-token blue


def _build_model(layout: RunLayout, device):
    import torch
    with open(layout.config_json) as f:
        cfg = json.load(f)
    state = torch.load(layout.model_pt, map_location="cpu", weights_only=False)
    model = SingleStreamTransformer(
        vocab_size=state["token_emb.weight"].shape[0],
        embedding_dim=state["token_emb.weight"].shape[1],
        max_seq_len=state["pos_emb.weight"].shape[0],
        num_heads=cfg.get("heads", 4), num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def _keep_only_positions(input_ids, keep_positions, salient_positions):
    """Mask every SALIENT position EXCEPT keep_positions (sufficiency).
    Non-salient tokens and special tokens are left untouched so the
    sequence scaffold (visit markers, CLS) stays intact."""
    drop = [p for p in salient_positions if p not in set(keep_positions)]
    return mask_positions(input_ids, drop)


def main() -> None:
    p = argparse.ArgumentParser(description="Render faithfulness curves (comprehensiveness + sufficiency).")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--mimic_dir", default="../data/mimiciv/3.1/hosp")
    p.add_argument("--split", default="test")
    p.add_argument("--sample_n", type=int, default=500,
                   help="Number of test patients to sample (default 500).")
    p.add_argument("--max_k", type=int, default=10,
                   help="Max number of top tokens to remove/keep (default 10).")
    p.add_argument("--n_random_draws", type=int, default=5,
                   help="Random-baseline draws per patient, averaged (default 5).")
    p.add_argument("--min_salient", type=int, default=3,
                   help="Skip patients with fewer than this many salient tokens.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-path",
                   default="../manuscript/jamia_submission/figures/supplements/"
                           "FigureS13_faithfulness.png")
    p.add_argument("--analysis-dir", default=None)
    args = p.parse_args()

    import torch
    rng = np.random.RandomState(args.seed)

    layout = RunLayout(args.run_dir)
    csv_dir = args.analysis_dir or layout.analysis_dir
    device = torch.device(args.device)

    print("[1] Loading model + tensors + IG...", flush=True)
    model = _build_model(layout, device)
    tensors = torch.load(layout.tensors_pt(args.split),
                          map_location="cpu", weights_only=False)
    ig = pd.read_csv(layout.ig_csv(args.split),
                      dtype={"patient_id": str, "index_hadm_id": str})

    # Salient tokens only (the candidates AToP would surface). Rank by |IG|.
    ig_sal = ig[ig["is_salient"]].copy()
    print(f"    {len(ig_sal):,} salient tokens across "
          f"{ig_sal.groupby(['patient_id','index_hadm_id']).ngroups:,} patients")

    # Group salient tokens per patient, keep those with >= min_salient
    groups = []
    for (pid, hadm), g in ig_sal.groupby(["patient_id", "index_hadm_id"]):
        if len(g) >= args.min_salient and (pid, hadm) in tensors:
            groups.append((pid, hadm, g))
    print(f"    {len(groups):,} patients with >= {args.min_salient} salient tokens")

    # Sample
    if len(groups) > args.sample_n:
        idx = rng.choice(len(groups), size=args.sample_n, replace=False)
        groups = [groups[i] for i in idx]
    print(f"[2] Sampled {len(groups)} patients (max_k={args.max_k}, "
          f"{args.n_random_draws} random draws each)")

    K = args.max_k
    # Accumulators: rows = patients, cols = k (1..K). Store Δŷ (drop) for
    # comprehensiveness, and recovered ŷ for sufficiency.
    comp_atop = []      # |ŷ_orig - ŷ_masked_topk|
    comp_rand = []
    suff_atop = []      # |ŷ_orig - ŷ_keep_topk|  (smaller = more sufficient)
    suff_rand = []

    # Build all the masked tensors per patient, batch-predict.
    for gi, (pid, hadm, g) in enumerate(groups):
        base = tensors[(pid, hadm)]
        # Rank salient positions by |IG| descending
        g_sorted = g.sort_values("ig_abs", ascending=False)
        ranked_pos = g_sorted["pos"].tolist()
        salient_pos = g["pos"].tolist()
        n_sal = len(salient_pos)
        kmax = min(K, n_sal)

        # Build tensors to score in one batch:
        # [orig, comp_atop_1..kmax, suff_atop_1..kmax,
        #  comp_rand(draws x kmax), suff_rand(draws x kmax)]
        batch = [base]
        # comprehensiveness AToP: remove top-k
        for k in range(1, kmax + 1):
            batch.append(mask_positions(base, ranked_pos[:k]))
        # sufficiency AToP: keep only top-k
        for k in range(1, kmax + 1):
            batch.append(_keep_only_positions(base, ranked_pos[:k], salient_pos))
        # random draws
        rand_perms = [rng.permutation(salient_pos).tolist()
                      for _ in range(args.n_random_draws)]
        for perm in rand_perms:
            for k in range(1, kmax + 1):
                batch.append(mask_positions(base, perm[:k]))   # comp random
        for perm in rand_perms:
            for k in range(1, kmax + 1):
                batch.append(_keep_only_positions(base, perm[:k], salient_pos))  # suff random

        probs = predict_batch(model, device, batch, batch_size=128)
        y0 = probs[0]
        ptr = 1
        comp_a = probs[ptr:ptr + kmax]; ptr += kmax
        suff_a = probs[ptr:ptr + kmax]; ptr += kmax
        comp_r_draws = []
        for _ in range(args.n_random_draws):
            comp_r_draws.append(probs[ptr:ptr + kmax]); ptr += kmax
        suff_r_draws = []
        for _ in range(args.n_random_draws):
            suff_r_draws.append(probs[ptr:ptr + kmax]); ptr += kmax

        # Pad to K with NaN if patient had < K salient tokens
        def _pad(arr):
            a = np.full(K, np.nan)
            a[:len(arr)] = arr
            return a

        comp_atop.append(_pad(np.abs(y0 - comp_a)))
        suff_atop.append(_pad(np.abs(y0 - suff_a)))
        comp_rand.append(_pad(np.abs(y0 - np.mean(comp_r_draws, axis=0))))
        suff_rand.append(_pad(np.abs(y0 - np.mean(suff_r_draws, axis=0))))

        if (gi + 1) % 50 == 0:
            print(f"    {gi+1}/{len(groups)}", flush=True)

    comp_atop = np.array(comp_atop)
    comp_rand = np.array(comp_rand)
    suff_atop = np.array(suff_atop)
    suff_rand = np.array(suff_rand)

    # ── Aggregate with bootstrap CIs over patients ──
    def _mean_ci(mat, n_boot=1000):
        """Column-wise nanmean + 95% bootstrap CI over rows (patients)."""
        means, los, his = [], [], []
        n = mat.shape[0]
        for k in range(mat.shape[1]):
            col = mat[:, k]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            means.append(col.mean())
            boots = [np.random.RandomState(args.seed + b).choice(col, len(col), replace=True).mean()
                     for b in range(n_boot)]
            los.append(np.percentile(boots, 2.5))
            his.append(np.percentile(boots, 97.5))
        return np.array(means), np.array(los), np.array(his)

    print("[3] Bootstrapping CIs...", flush=True)
    ca_m, ca_lo, ca_hi = _mean_ci(comp_atop)
    cr_m, cr_lo, cr_hi = _mean_ci(comp_rand)
    sa_m, sa_lo, sa_hi = _mean_ci(suff_atop)
    sr_m, sr_lo, sr_hi = _mean_ci(suff_rand)

    ks = np.arange(1, K + 1)

    # ── Render ──
    fig, (axC, axS) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"wspace": 0.25})

    # Panel A: comprehensiveness (higher AToP curve = more faithful)
    axC.plot(ks, ca_m, "-o", color=RISK, lw=2.2, label="AToP top-k removed")
    axC.fill_between(ks, ca_lo, ca_hi, color=RISK, alpha=0.18)
    axC.plot(ks, cr_m, "-s", color=RANDOM, lw=2.0, label="Random k removed")
    axC.fill_between(ks, cr_lo, cr_hi, color=RANDOM, alpha=0.18)
    axC.set_xlabel("Number of salient tokens removed (k)",
                    fontsize=12, fontweight="bold", labelpad=8)
    axC.set_ylabel("|Δŷ|  (prediction change vs original)",
                    fontsize=12, fontweight="bold", labelpad=8)
    axC.set_title("A · Comprehensiveness",
                   fontsize=12.5, loc="left", pad=10, fontweight="bold")
    axC.set_xticks(ks)
    axC.grid(True, color="#eeeeee", lw=0.5)
    axC.legend(loc="upper left", fontsize=10, framealpha=0.95)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)

    # Panel B: sufficiency (lower AToP curve = more sufficient)
    axS.plot(ks, sa_m, "-o", color=KEEP, lw=2.2, label="Keep only AToP top-k")
    axS.fill_between(ks, sa_lo, sa_hi, color=KEEP, alpha=0.18)
    axS.plot(ks, sr_m, "-s", color=RANDOM, lw=2.0, label="Keep only random k")
    axS.fill_between(ks, sr_lo, sr_hi, color=RANDOM, alpha=0.18)
    axS.set_xlabel("Number of salient tokens kept (k)",
                    fontsize=12, fontweight="bold", labelpad=8)
    axS.set_ylabel("|Δŷ|  (prediction change vs original)",
                    fontsize=12, fontweight="bold", labelpad=8)
    axS.set_title("B · Sufficiency",
                   fontsize=12.5, loc="left", pad=10, fontweight="bold")
    axS.set_xticks(ks)
    axS.grid(True, color="#eeeeee", lw=0.5)
    axS.legend(loc="upper right", fontsize=10, framealpha=0.95)
    for s in ("top", "right"):
        axS.spines[s].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[4] Saved {args.output_path} "
          f"({os.path.getsize(args.output_path)//1024} KB)")

    # ── Headline stats for the legend / Results sentence ──
    from scipy.stats import wilcoxon
    # Use the k=5 column (or kmax) for a single headline number
    k_head = min(5, K) - 1
    ca5 = comp_atop[:, k_head]; cr5 = comp_rand[:, k_head]
    mask = ~np.isnan(ca5) & ~np.isnan(cr5)
    w, pval = wilcoxon(ca5[mask], cr5[mask], alternative="greater")
    print("\n=== Headline (k=5) ===")
    print(f"  Comprehensiveness AToP |Δŷ| @k=5: {np.nanmean(ca5):.4f}")
    print(f"  Comprehensiveness random |Δŷ| @k=5: {np.nanmean(cr5):.4f}")
    print(f"  Ratio (AToP / random): {np.nanmean(ca5)/max(np.nanmean(cr5),1e-9):.2f}x")
    print(f"  Wilcoxon AToP > random: p = {pval:.2e}")

    # AUC between curves (comprehensiveness) — area between AToP and random
    auc_diff = np.trapezoid(ca_m - cr_m, ks)
    print(f"  Comprehensiveness AUC gap (AToP - random): {auc_diff:.4f}")

    # ── Backing CSV ──
    out_csv = os.path.join(csv_dir, "fig_faithfulness_curves.csv")
    pd.DataFrame({
        "k": ks,
        "comp_atop_mean": ca_m, "comp_atop_lo": ca_lo, "comp_atop_hi": ca_hi,
        "comp_rand_mean": cr_m, "comp_rand_lo": cr_lo, "comp_rand_hi": cr_hi,
        "suff_atop_mean": sa_m, "suff_atop_lo": sa_lo, "suff_atop_hi": sa_hi,
        "suff_rand_mean": sr_m, "suff_rand_lo": sr_lo, "suff_rand_hi": sr_hi,
    }).to_csv(out_csv, index=False)
    print(f"    csv: {out_csv}")
    print(f"    n_patients = {len(groups)}")


if __name__ == "__main__":
    main()
