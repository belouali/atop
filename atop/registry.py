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

    Bundle contents (v2 layout):
        0_inputs/config.json         Full AToPConfig
        0_inputs/tokenization.json   Tokenization subset
        0_inputs/vocab.pkl           Token → index map
        0_inputs/splits.json         Train/val/test patient IDs
        1_model/model.pt             Model state dict
        1_model/train_log.csv        Per-epoch training metrics
        2_predictions/metrics.json   Test AUROC/PR-AUC
        2_predictions/summary.json   Dataset summary stats

    Falls back to writing everything to ``run_dir/`` root for legacy callers
    that haven't created the v2 stage folders yet.
    """
    os.makedirs(run_dir, exist_ok=True)

    # ── Resolve target sub-folders (create if missing for first save) ────
    inputs_dir      = os.path.join(run_dir, "0_inputs")
    model_dir       = os.path.join(run_dir, "1_model")
    predictions_dir = os.path.join(run_dir, "2_predictions")
    use_v2 = True
    for d in (inputs_dir, model_dir, predictions_dir):
        os.makedirs(d, exist_ok=True)

    # Config (full) → 0_inputs/
    config.save(os.path.join(inputs_dir, "config.json"))

    # Tokenization subset → 0_inputs/
    tok_meta = {
        "token_types": config.token_types,
        "max_visits": config.max_visits,
        "max_seq_len": config.max_seq_len,
        "max_drug_freq": config.max_drug_freq,
        "chronic_filter": config.chronic_filter,
        "one_per_patient": config.one_per_patient,
        "exclude_elective_readmissions": config.exclude_elective_readmissions,
    }
    with open(os.path.join(inputs_dir, "tokenization.json"), "w") as f:
        json.dump(tok_meta, f, indent=2)

    # Model weights → 1_model/
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

    # Vocabulary → 0_inputs/
    with open(os.path.join(inputs_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    # Splits → 0_inputs/
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(os.path.join(inputs_dir, "splits.json"), "w") as f:
        json.dump(splits, f)

    # Metrics → 2_predictions/
    if metrics:
        with open(os.path.join(predictions_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Summary → 2_predictions/
    if summary:
        with open(os.path.join(predictions_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # Training log → 1_model/
    if train_log:
        import pandas as pd
        pd.DataFrame(train_log).to_csv(os.path.join(model_dir, "train_log.csv"), index=False)

    print(f"[bundle] Saved to {run_dir} (v2 layout)")


def _resolve_bundle_path(run_dir: str, basename: str, *subfolders: str) -> str:
    """Locate a bundle file in either the v2 layout (sub-folder per stage) or
    the legacy layout (everything at run_dir root). Tries each subfolder in
    order, then falls back to ``<run_dir>/<basename>``."""
    for sub in subfolders:
        candidate = os.path.join(run_dir, sub, basename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(run_dir, basename)


def load_bundle(
    run_dir: str,
    device: torch.device = torch.device("cpu"),
) -> Tuple[AToPConfig, SingleStreamTransformer, Dict[str, int], Dict]:
    """Load a model bundle and return (config, model, vocab, splits).

    Layout-aware: prefers v2 sub-folders (``0_inputs/``, ``1_model/``) when
    present; falls back to legacy ``run_dir`` root.
    """
    config_path = _resolve_bundle_path(run_dir, "config.json", "0_inputs")
    vocab_path  = _resolve_bundle_path(run_dir, "vocab.pkl",   "0_inputs")
    splits_path = _resolve_bundle_path(run_dir, "splits.json", "0_inputs")
    model_path  = _resolve_bundle_path(run_dir, "model.pt",    "1_model")

    config = AToPConfig.load(config_path)

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    model = SingleStreamTransformer(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        max_seq_len=config.max_seq_len,
        num_heads=config.heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    splits = {}
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)

    print(f"[bundle] Loaded from {run_dir} (vocab={len(vocab)}, device={device})")
    return config, model, vocab, splits


# ============================================================================
# Experiment registry, auto-generated lab notebook at runs/index.csv
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
