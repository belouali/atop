#!/usr/bin/env python3
"""
Compare trained Transformer against LACE baseline.

Usage:
  python scripts/compare_baselines.py \
      --run_dir runs/exp01 \
      --mimic_dir /data/mimiciv/hosp \
      --lace_csv lace_scores_all.csv \
      [--bootstrap --n_boot 1000]

Results saved to: runs/exp01/baselines/
  baseline_comparison.csv
  baseline_comparison_bootstrap.csv (if --bootstrap)
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.config import set_seed, pick_device
from atop.registry import load_bundle
from atop.data.mimic import load_mimic_tables, build_readmission_labels
from atop.data.tokenization import build_patient_sequences
from atop.data.datasets import split_samples_by_patient, MIMICReadmissionDataset, collate_fn
from atop.models.training import evaluate_model
from atop.baselines.lace import load_lace_scores
from atop.utils import auroc_np, pr_auc_np


def parse_args():
    p = argparse.ArgumentParser(description="Compare model vs LACE baseline")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--mimic_dir", type=str, required=True)
    p.add_argument("--lace_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--bootstrap", action="store_true", default=False)
    p.add_argument("--n_boot", type=int, default=1000)
    return p.parse_args()


def _merge_samples_lace(sample_list, lace_df):
    """Vectorized merge of samples with LACE scores."""
    sdf = pd.DataFrame([
        {"subject_id": str(s["patient_id"]),
         "hadm_id": str(s["index_hadm_id"]),
         "label": s["readmit_30d"]}
        for s in sample_list
    ])
    merged = sdf.merge(lace_df, on=["subject_id", "hadm_id"], how="inner")
    return merged


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.run_dir, "baselines")
    os.makedirs(out_dir, exist_ok=True)

    device = pick_device(args.device)
    config, model, vocab, splits = load_bundle(args.run_dir, device)
    set_seed(config.seed)

    # ── Rebuild data + splits ────────────────────────────────────────
    admissions, diagnoses, procedures, prescriptions = load_mimic_tables(args.mimic_dir)
    adm_labels = build_readmission_labels(
        admissions, exclude_elective_readmissions=config.exclude_elective_readmissions)

    samples = build_patient_sequences(
        adm_labels, diagnoses, procedures, prescriptions,
        max_visits=config.max_visits, one_per_patient=config.one_per_patient,
        max_drug_freq=config.max_drug_freq,
        drug_exclude_substrings=config.drug_exclude_substrings or None,
        token_types=config.token_types,
        chronic_filter=config.chronic_filter)

    # Reproduce splits
    train_pids = set(str(p) for p in splits.get("train", []))
    if train_pids:
        test_pids = set(str(p) for p in splits.get("test", []))
        all_split_pids = train_pids | test_pids | set(str(p) for p in splits.get("val", []))
        samples = [s for s in samples if str(s["patient_id"]) in all_split_pids]
        train_samples = [s for s in samples if str(s["patient_id"]) in train_pids]
        test_samples = [s for s in samples if str(s["patient_id"]) in test_pids]
    else:
        if config.max_patients > 0:
            all_pids = list(set(s["patient_id"] for s in samples))
            if len(all_pids) > config.max_patients:
                rng = np.random.RandomState(config.seed)
                keep = set(rng.choice(all_pids, size=config.max_patients, replace=False))
                samples = [s for s in samples if s["patient_id"] in keep]
        train_samples, _, test_samples = split_samples_by_patient(samples, seed=config.seed)

    print(f"[data] train={len(train_samples):,} | test={len(test_samples):,}")

    # ── Transformer test metrics ─────────────────────────────────────
    from torch.utils.data import DataLoader
    test_ds = MIMICReadmissionDataset(test_samples, vocab, config.max_seq_len)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size,
                             shuffle=False, collate_fn=collate_fn)
    test_metrics = evaluate_model(model, test_loader, device)
    y_prob_test = test_metrics.pop("y_prob")
    y_true_test = test_metrics.pop("y_true")
    trans_auroc = test_metrics["auroc"]
    trans_prauc = test_metrics["pr_auc"]
    print(f"[transformer] AUROC={trans_auroc:.4f} | PR-AUC={trans_prauc:.4f}")

    # ── LACE merge ───────────────────────────────────────────────────
    lace_df = load_lace_scores(args.lace_csv)
    train_l = _merge_samples_lace(train_samples, lace_df)
    test_l = _merge_samples_lace(test_samples, lace_df)

    if test_l.empty or train_l.empty:
        print("[LACE] No matching admissions found — check subject_id/hadm_id alignment")
        return

    print(f"[LACE] Matched: train={len(train_l):,} | test={len(test_l):,}")

    y_tr = train_l["label"].values.astype(float)
    y_te = test_l["label"].values.astype(float)

    results = {"model": [], "auroc": [], "pr_auc": []}
    prob_store = {}  # name -> prob array for bootstrap

    # ── 1) Transformer ───────────────────────────────────────────────
    results["model"].append("Transformer")
    results["auroc"].append(trans_auroc)
    results["pr_auc"].append(trans_prauc)

    # ── 2) Raw LACE score ────────────────────────────────────────────
    lace_scores = test_l["lace_score"].values.astype(float)
    lace_raw_auroc = auroc_np(y_te, lace_scores)
    lace_raw_prauc = pr_auc_np(y_te, lace_scores)
    results["model"].append("LACE (raw score)")
    results["auroc"].append(lace_raw_auroc)
    results["pr_auc"].append(lace_raw_prauc)
    prob_store["LACE (raw score)"] = lace_scores
    print(f"[LACE raw]  AUROC={lace_raw_auroc:.4f} | PR-AUC={lace_raw_prauc:.4f}")

    # ── 3) LACE → LR, LACE components → LR ──────────────────────────
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        lr1 = LogisticRegression(max_iter=1000, random_state=42)
        lr1.fit(train_l[["lace_score"]].values, y_tr)
        lace_lr_prob = lr1.predict_proba(test_l[["lace_score"]].values)[:, 1]
        lace_lr_auroc = auroc_np(y_te, lace_lr_prob)
        lace_lr_prauc = pr_auc_np(y_te, lace_lr_prob)
        results["model"].append("LACE → LR")
        results["auroc"].append(lace_lr_auroc)
        results["pr_auc"].append(lace_lr_prauc)
        prob_store["LACE → LR"] = lace_lr_prob
        print(f"[LACE→LR]   AUROC={lace_lr_auroc:.4f} | PR-AUC={lace_lr_prauc:.4f}")

        feat = ["L", "A", "C", "E"]
        if all(f in train_l.columns for f in feat):
            sc = StandardScaler()
            lr2 = LogisticRegression(max_iter=1000, random_state=42)
            lr2.fit(sc.fit_transform(train_l[feat].values.astype(float)), y_tr)
            lace_comp_prob = lr2.predict_proba(
                sc.transform(test_l[feat].values.astype(float)))[:, 1]
            lace_comp_auroc = auroc_np(y_te, lace_comp_prob)
            lace_comp_prauc = pr_auc_np(y_te, lace_comp_prob)
            results["model"].append("LACE components → LR")
            results["auroc"].append(lace_comp_auroc)
            results["pr_auc"].append(lace_comp_prauc)
            prob_store["LACE components → LR"] = lace_comp_prob
            print(f"[LACE comp] AUROC={lace_comp_auroc:.4f} | PR-AUC={lace_comp_prauc:.4f}")
    except Exception as e:
        print(f"[WARN] LR models failed: {e}")

    # ── Save results ─────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(out_dir, "baseline_comparison.csv"), index=False)
    print(f"\n{df_results.to_string(index=False)}")

    # ── Bootstrap CIs (optional) ─────────────────────────────────────
    if args.bootstrap:
        try:
            print(f"\n[bootstrap] Running {args.n_boot} iterations...")

            # Align transformer probs with LACE-matched test samples
            test_sdf = pd.DataFrame([
                {"subject_id": str(s["patient_id"]),
                 "hadm_id": str(s["index_hadm_id"]),
                 "y_prob": float(y_prob_test[i]),
                 "label": s["readmit_30d"]}
                for i, s in enumerate(test_samples)
            ])
            matched = test_sdf.merge(
                lace_df[["subject_id", "hadm_id"]].drop_duplicates(),
                on=["subject_id", "hadm_id"], how="inner")
            trans_probs_matched = matched["y_prob"].values

            if len(trans_probs_matched) != len(y_te):
                print(f"[bootstrap] Skipped — alignment mismatch "
                      f"({len(trans_probs_matched)} vs {len(y_te)})")
            else:
                prob_store["Transformer"] = trans_probs_matched
                rng = np.random.RandomState(42)
                n = len(y_te)
                boot = {name: [] for name in prob_store}

                for _ in range(args.n_boot):
                    idx = rng.randint(0, n, size=n)
                    yb = y_te[idx]
                    if yb.sum() == 0 or yb.sum() == n:
                        continue
                    for name, probs in prob_store.items():
                        boot[name].append(auroc_np(yb, probs[idx]))

                n_valid = len(boot["Transformer"])
                print(f"[bootstrap] {n_valid} valid iterations")

                if n_valid >= 10:
                    ci_rows = []
                    for name in boot:
                        b = np.array(boot[name])
                        ci = np.percentile(b, [2.5, 97.5])
                        point = df_results.loc[
                            df_results["model"] == name, "auroc"].values[0]
                        print(f"  {name:28s} {point:.4f} "
                              f"(95% CI: {ci[0]:.4f}–{ci[1]:.4f})")
                        ci_rows.append({"model": name, "auroc": point,
                                        "ci_lo": ci[0], "ci_hi": ci[1]})

                    trans_b = np.array(boot["Transformer"])
                    for name in boot:
                        if name == "Transformer":
                            continue
                        delta = trans_b - np.array(boot[name])
                        dci = np.percentile(delta, [2.5, 97.5])
                        print(f"  Δ(Trans - {name:20s}) = "
                              f"{dci[0]:+.4f} to {dci[1]:+.4f}")
                        ci_rows.append({
                            "model": f"Δ(Trans - {name})",
                            "auroc": float(np.mean(delta)),
                            "ci_lo": dci[0], "ci_hi": dci[1]})

                    pd.DataFrame(ci_rows).to_csv(
                        os.path.join(out_dir, "baseline_comparison_bootstrap.csv"),
                        index=False)
        except Exception as e:
            print(f"[WARN] Bootstrap failed: {e}")
            traceback.print_exc()

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
