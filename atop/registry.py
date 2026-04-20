"""Model bundle save/load + experiment registry."""
from __future__ import annotations

import csv
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from atop.config import AToPConfig
from atop.models.single_stream_transformer import SingleStreamTransformer


def save_bundle(
    run_dir: str,
    config: AToPConfig,
    model: SingleStreamTransformer,
    vocab: Dict[str, int],
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    metrics: Optional[Dict[str, float]] = None,
    summary: Optional[Dict[str, Any]] = None,
    train_log: Optional[List[Dict]] = None,
):
    """
    Save a complete model bundle to run_dir.

    Bundle contents:
        config.json         Full AToPConfig (includes tokenization params)
        tokenization.json   Tokenization-specific subset for quick reference
        model.pt            Model state dict
        vocab.pkl           Token→index vocabulary
        splits.json         Patient ID lists for train/val/test
        metrics.json        Test AUROC/PR-AUC
        summary.json        Dataset summary stats
        train_log.csv       Per-epoch training metrics
    """
    os.makedirs(run_dir, exist_ok=True)

    # Config (full)
    config.save(os.path.join(run_dir, "config.json"))

    # Tokenization subset (for quick reference without loading full config)
    tok_meta = {
        "token_types": config.token_types,
        "max_visits": config.max_visits,
        "max_seq_len": config.max_seq_len,
        "max_drug_freq": config.max_drug_freq,
        "chronic_filter": config.chronic_filter,
        "one_per_patient": config.one_per_patient,
        "exclude_elective_readmissions": config.exclude_elective_readmissions,
    }
    with open(os.path.join(run_dir, "tokenization.json"), "w") as f:
        json.dump(tok_meta, f, indent=2)

    # Model weights
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    # Vocabulary
    with open(os.path.join(run_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    # Splits
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(os.path.join(run_dir, "splits.json"), "w") as f:
        json.dump(splits, f)

    # Metrics
    if metrics:
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Summary
    if summary:
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # Training log
    if train_log:
        import pandas as pd
        pd.DataFrame(train_log).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

    print(f"[bundle] Saved to {run_dir}")


def load_bundle(
    run_dir: str,
    device: torch.device = torch.device("cpu"),
) -> Tuple[AToPConfig, SingleStreamTransformer, Dict[str, int], Dict]:
    """Load a model bundle and return (config, model, vocab, splits)."""
    config = AToPConfig.load(os.path.join(run_dir, "config.json"))

    with open(os.path.join(run_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    model = SingleStreamTransformer(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        max_seq_len=config.max_seq_len,
        num_heads=config.heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model.load_state_dict(torch.load(
        os.path.join(run_dir, "model.pt"), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    splits = {}
    splits_path = os.path.join(run_dir, "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)

    print(f"[bundle] Loaded from {run_dir} (vocab={len(vocab)}, device={device})")
    return config, model, vocab, splits


# ============================================================================
# Experiment registry — auto-generated lab notebook at runs/index.csv
# ============================================================================

def register_experiment(
    runs_root: str,
    run_dir: str,
    config: AToPConfig,
    metrics: Dict[str, float],
    summary: Optional[Dict[str, Any]] = None,
):
    """
    Append a row to runs/index.csv with this experiment's key info.

    Creates the file with headers if it doesn't exist. Each row captures:
    run_id, date, task, token_types, filters, backbone, key hyperparams,
    test AUROC/PR-AUC, and path to the run directory.
    """
    index_path = os.path.join(runs_root, "index.csv")
    exists = os.path.exists(index_path)

    row = {
        "run_id": os.path.basename(run_dir),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "token_types": config.token_types,
        "chronic_filter": config.chronic_filter,
        "max_drug_freq": config.max_drug_freq,
        "exclude_elective": config.exclude_elective_readmissions,
        "backbone": "transformer",
        "embed_dim": config.embedding_dim,
        "layers": config.num_layers,
        "heads": config.heads,
        "epochs": config.epochs,
        "lr": config.lr,
        "seed": config.seed,
        "test_auroc": metrics.get("auroc", ""),
        "test_prauc": metrics.get("pr_auc", ""),
        "n_samples": summary.get("n_samples", "") if summary else "",
        "vocab_size": summary.get("vocab_size", "") if summary else "",
        "path": run_dir,
    }

    with open(index_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[registry] Logged to {index_path}")
