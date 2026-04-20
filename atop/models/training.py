"""Model training and evaluation loops."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from atop.utils import auroc_np, pr_auc_np


def train_model(model, train_loader: DataLoader, val_loader: DataLoader,
                device: torch.device, epochs: int, lr: float, weight_decay: float,
                patience: int = 5,
                ) -> tuple:
    """
    Train with BCE loss, return (best_val_metrics, train_log).

    Early stops if val_auroc doesn't improve for `patience` epochs.
    Always restores the best model weights before returning.

    train_log is a list of dicts with per-epoch metrics, suitable for
    pd.DataFrame(train_log).to_csv("train_log.csv").
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Compute class weight for imbalanced data
    all_labels = []
    for batch in train_loader:
        all_labels.append(batch["label"].numpy())
    all_labels = np.concatenate(all_labels)
    pos_weight = torch.tensor([(len(all_labels) - all_labels.sum()) / max(all_labels.sum(), 1)],
                              dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auroc = 0.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    train_log = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            out = model(ids)
            loss = criterion(out["logits"], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_loss = train_loss / max(n_batches, 1)

        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f} | "
              f"val_auroc={val_metrics['auroc']:.4f} | val_pr_auc={val_metrics['pr_auc']:.4f}")

        train_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_auroc": val_metrics["auroc"],
            "val_pr_auc": val_metrics["pr_auc"],
        })

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  [early stop] No improvement for {patience} epochs. "
                      f"Best val_auroc={best_val_auroc:.4f} at epoch {best_epoch}.")
                break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"  [best model] Restored epoch {best_epoch} (val_auroc={best_val_auroc:.4f})")
    return evaluate_model(model, val_loader, device), train_log



def evaluate_model(model: SingleStreamTransformer, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Compute AUROC and PR-AUC on a loader."""
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            out = model(ids)
            y_prob_all.append(out["y_prob"].cpu().numpy())
            y_true_all.append(batch["label"].numpy())
    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    return {
        "auroc": auroc_np(y_true, y_prob),
        "pr_auc": pr_auc_np(y_true, y_prob),
        "y_true": y_true,
        "y_prob": y_prob,
    }


# ============================================================================
# METRICS (no sklearn dependency)
# ============================================================================
