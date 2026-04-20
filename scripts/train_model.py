#!/usr/bin/env python3
"""
Train an AToP Transformer model on MIMIC-IV readmission prediction.

Produces a model bundle in --out_dir containing:
  config.json, model.pt, vocab.pkl, splits.json, metrics.json, summary.json,
  fig1_dataset_performance.png, dataset_summary.csv, test_metrics.csv

Usage:
  python scripts/train_model.py --mimic_dir /data/mimiciv/2.2/hosp --out_dir runs/exp01
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add parent to path so `atop` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.config import AToPConfig, set_seed, pick_device
from atop.data.mimic import load_mimic_tables, build_readmission_labels
from atop.data.tokenization import build_patient_sequences
from atop.data.datasets import (
    build_vocabulary, MIMICReadmissionDataset, collate_fn, split_samples_by_patient,
)
from atop.models.single_stream_transformer import SingleStreamTransformer
from atop.models.training import train_model, evaluate_model
from atop.explain.figures import fig1_dataset_performance, fig_train_curves
from atop.registry import save_bundle, register_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Train AToP Transformer")
    p.add_argument("--mimic_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="",
                   help="Output directory. If empty, auto-generates under --runs_root")
    p.add_argument("--runs_root", type=str, default="runs",
                   help="Parent directory for auto-generated run dirs (default: runs/)")
    # Data
    p.add_argument("--token_types", type=str, default="CPD")
    p.add_argument("--max_visits", type=int, default=20)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--one_per_patient", action="store_true", default=True)
    p.add_argument("--max_drug_freq", type=float, default=0.5)
    p.add_argument("--drug_exclude_substrings", type=str, nargs="*", default=None,
                   help="Substrings to exclude from drug names. "
                        "Pass with no args to use built-in default list, "
                        "or specify custom substrings (e.g., Flush Bag Vial)")
    p.add_argument("--chronic_filter", action="store_true", default=False)
    p.add_argument("--first_occurrence_only", action="store_true", default=False,
                   help="Keep each code (C/P/D) only in the earliest admission where it appears. "
                        "Removes all repeated codes across admissions.")
    p.add_argument("--first_occurrence_drugs_only", action="store_true", default=False,
                   help="Apply first-occurrence filtering only to drug tokens. "
                        "Conditions and procedures are kept at every visit.")
    p.add_argument("--exclude_elective_readmissions", action="store_true", default=False)
    p.add_argument("--harmonize_icd", action="store_true", default=False,
                   help="Map ICD-9 codes to ICD-10 via CMS GEMs (1:1 mappings only)")
    p.add_argument("--icd_exclude_prefixes", type=str, nargs="*", default=None,
                   help="ICD prefixes to exclude (e.g. Z20 Z23). "
                        "Pass with no args to use built-in default list.")
    p.add_argument("--drug_mapping", type=str, default="",
                   help="Path to NDC→RxNorm ingredient CSV (from build_ndc_rxnorm_map.py)")
    p.add_argument("--max_patients", type=int, default=0)
    # Model
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience: stop if val_auroc doesn't improve for N epochs")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--preprocess_only", action="store_true", default=False,
                   help="Only run preprocessing (build + cache samples), then exit. "
                        "Use this on a CPU runtime to prepare the cache before GPU training.")
    p.add_argument("--use_icd_titles", action="store_true", default=True)
    return p.parse_args()


def _generate_run_id(args) -> str:
    """Generate a descriptive run ID from timestamp + key params."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tok = args.token_types
    n = f"{args.max_patients}p" if args.max_patients > 0 else "full"
    return f"{ts}_{tok}_{n}_seed{args.seed}"


def main():
    args = parse_args()

    # Resolve drug exclusion: None = not passed, [] = flag with no args → use default
    from atop.config import DEFAULT_DRUG_EXCLUDE_SUBSTRINGS, DEFAULT_ICD_EXCLUDE_PREFIXES
    if args.drug_exclude_substrings is not None and len(args.drug_exclude_substrings) == 0:
        args.drug_exclude_substrings = DEFAULT_DRUG_EXCLUDE_SUBSTRINGS
        print(f"[drug filter] Using default exclusion list ({len(DEFAULT_DRUG_EXCLUDE_SUBSTRINGS)} substrings)")

    # Resolve ICD exclusion: None = not passed, [] = flag with no args → use default
    if args.icd_exclude_prefixes is not None and len(args.icd_exclude_prefixes) == 0:
        args.icd_exclude_prefixes = DEFAULT_ICD_EXCLUDE_PREFIXES
        print(f"[icd filter] Using default exclusion list ({len(DEFAULT_ICD_EXCLUDE_PREFIXES)} prefixes)")

    # Auto-generate out_dir if not specified
    if not args.out_dir:
        run_id = _generate_run_id(args)
        args.out_dir = os.path.join(args.runs_root, run_id)
        print(f"[run] Auto-generated: {args.out_dir}")

    config = AToPConfig.from_args(args)
    config.mimic_dir = args.mimic_dir

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"[device] {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    admissions, diagnoses, procedures, prescriptions = load_mimic_tables(args.mimic_dir)
    adm_labels = build_readmission_labels(
        admissions, exclude_elective_readmissions=args.exclude_elective_readmissions)
    print(f"[data] Admissions with labels: {len(adm_labels):,}")

    samples = build_patient_sequences(
        adm_labels, diagnoses, procedures, prescriptions,
        max_visits=args.max_visits, one_per_patient=args.one_per_patient,
        max_drug_freq=args.max_drug_freq,
        drug_exclude_substrings=args.drug_exclude_substrings or None,
        token_types=args.token_types,
        chronic_filter=args.chronic_filter,
        first_occurrence_only=args.first_occurrence_only,
        first_occurrence_drugs_only=args.first_occurrence_drugs_only,
        harmonize_icd=args.harmonize_icd,
        icd_exclude_prefixes=args.icd_exclude_prefixes or None,
        drug_mapping=args.drug_mapping,
        cache_dir=os.path.join(args.mimic_dir, ".atop_cache"))

    if args.preprocess_only:
        print(f"\n[preprocess_only] Done. {len(samples):,} samples cached.")
        print(f"[preprocess_only] Switch to GPU runtime and re-run without --preprocess_only to train.")
        return

    if args.max_patients > 0:
        all_pids = list(set(s["patient_id"] for s in samples))
        if len(all_pids) > args.max_patients:
            rng = np.random.RandomState(args.seed)
            keep_pids = set(rng.choice(all_pids, size=args.max_patients, replace=False))
            samples = [s for s in samples if s["patient_id"] in keep_pids]
            print(f"[subsample] {args.max_patients:,} patients → {len(samples):,} samples")

    # ── Split ────────────────────────────────────────────────────────────
    train_samples, val_samples, test_samples = split_samples_by_patient(samples, seed=args.seed)
    print(f"[split] train={len(train_samples):,} | val={len(val_samples):,} | test={len(test_samples):,}")

    vocab = build_vocabulary(train_samples, min_freq=2)
    print(f"[vocab] {len(vocab):,} tokens")

    train_ds = MIMICReadmissionDataset(train_samples, vocab, args.max_seq_len)
    val_ds = MIMICReadmissionDataset(val_samples, vocab, args.max_seq_len)
    test_ds = MIMICReadmissionDataset(test_samples, vocab, args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # ── Summary ──────────────────────────────────────────────────────────
    all_labels = np.array([s["readmit_30d"] for s in samples])
    n_multi = sum(1 for s in samples if s["n_visits"] > 1)
    summary = {
        "n_samples": len(samples),
        "n_unique_patients": len(set(s["patient_id"] for s in samples)),
        "n_readmit": int(all_labels.sum()),
        "n_no_readmit": int((1 - all_labels).sum()),
        "readmission_rate": float(all_labels.mean()),
        "n_multi_visit": n_multi,
        "pct_multi_visit": 100 * n_multi / max(len(samples), 1),
        "vocab_size": len(vocab),
    }
    pd.Series(summary).to_csv(os.path.join(args.out_dir, "dataset_summary.csv"))

    # ── Model ────────────────────────────────────────────────────────────
    model = SingleStreamTransformer(
        vocab_size=len(vocab), embedding_dim=args.embedding_dim,
        max_seq_len=args.max_seq_len, num_heads=args.heads,
        num_layers=args.num_layers, dropout=args.dropout)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Train ────────────────────────────────────────────────────────────
    val_metrics, train_log = train_model(model, train_loader, val_loader, device,
                                         args.epochs, args.lr, args.weight_decay,
                                         patience=args.patience)

    # ── Test ─────────────────────────────────────────────────────────────
    test_metrics = evaluate_model(model, test_loader, device)
    y_true = test_metrics.pop("y_true")
    y_prob = test_metrics.pop("y_prob")
    pd.Series(test_metrics).to_csv(os.path.join(args.out_dir, "test_metrics.csv"))
    print(f"[test] AUROC={test_metrics['auroc']:.4f} | PR-AUC={test_metrics['pr_auc']:.4f}")

    fig1_dataset_performance(args.out_dir, summary, test_metrics, y_true, y_prob)
    fig_train_curves(args.out_dir, train_log)

    # ── Save bundle ──────────────────────────────────────────────────────
    train_ids = [str(s["patient_id"]) for s in train_samples]
    val_ids = [str(s["patient_id"]) for s in val_samples]
    test_ids = [str(s["patient_id"]) for s in test_samples]

    save_bundle(args.out_dir, config, model, vocab,
                train_ids, val_ids, test_ids,
                metrics=test_metrics, summary=summary,
                train_log=train_log)

    # ── Experiment registry ──────────────────────────────────────────────
    runs_root = args.runs_root
    if runs_root == "runs" and os.path.isabs(args.out_dir):
        # Derive from out_dir parent when using absolute paths
        runs_root = os.path.dirname(args.out_dir)
    os.makedirs(runs_root, exist_ok=True)
    register_experiment(runs_root, args.out_dir, config, test_metrics, summary)

    print(f"\n{'='*60}")
    print(f"Model bundle saved to: {args.out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
