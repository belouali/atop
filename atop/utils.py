"""Shared utilities: metrics, display helpers, ICD title loading."""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from atop.config import PAD_IDX


# ── Metrics ──────────────────────────────────────────────────────────────

def auroc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def pr_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def roc_curve_np(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve as _roc_curve
    fpr, tpr, _ = _roc_curve(y_true, y_score)
    return fpr, tpr


def pr_curve_np(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import precision_recall_curve as _pr_curve
    prec, rec, _ = _pr_curve(y_true, y_score)
    return rec, prec


# ── ICD title loading ────────────────────────────────────────────────────

def load_icd_titles(mimic_dir: str) -> Dict[Tuple[str, str], str]:
    titles = {}
    for ext in [".csv.gz", ".csv"]:
        path = os.path.join(mimic_dir, "d_icd_diagnoses" + ext)
        if os.path.exists(path):
            df = pd.read_csv(path, dtype=str)
            for _, r in df.iterrows():
                titles[("D", str(r["icd_code"]).strip())] = str(r["long_title"]).strip()[:60]
            break
    for ext in [".csv.gz", ".csv"]:
        path = os.path.join(mimic_dir, "d_icd_procedures" + ext)
        if os.path.exists(path):
            df = pd.read_csv(path, dtype=str)
            for _, r in df.iterrows():
                titles[("P", str(r["icd_code"]).strip())] = str(r["long_title"]).strip()[:60]
            break
    return titles


def load_drug_names(drug_mapping_path: str) -> Dict[str, str]:
    """Load RxCUI → ingredient name mapping from the NDC-to-ingredient CSV.

    Returns dict mapping drug token (e.g. 'D:RX_161') to ingredient name
    (e.g. 'Acetaminophen').
    """
    drug_names = {}
    if not drug_mapping_path or not os.path.exists(drug_mapping_path):
        return drug_names
    df = pd.read_csv(drug_mapping_path, dtype=str).fillna("")
    for _, r in df.iterrows():
        rxcui = r.get("ingredient_rxcui", "")
        name = r.get("ingredient_name", "") or r.get("ingredient_normalized", "")
        if rxcui and name:
            tok = f"D:RX_{rxcui}"
            if tok not in drug_names:
                drug_names[tok] = name.strip().title()
    print(f"[drug names] Loaded {len(drug_names)} RxCUI→name mappings")
    return drug_names


def parse_icd_token(token: str) -> Tuple[str, str]:
    if ":" in token:
        prefix, rest = token.split(":", 1)
        if "_" in rest:
            ver, code = rest.split("_", 1)
            return ver, code
    return "", token


def format_token_readable(tok: str, icd_titles: Dict[Tuple[str, str], str],
                          drug_names: Dict[str, str] = None) -> str:
    if tok.startswith("D:"):
        # Check explicit drug_names first, then module-level registry
        if drug_names and tok in drug_names:
            return f"D:{drug_names[tok]}"
        # Fall back to label_utils registry
        from atop.explain.label_utils import _DRUG_NAMES
        if tok in _DRUG_NAMES:
            return f"D:{_DRUG_NAMES[tok]}"
        drug = tok[2:]
        return f"D:{drug}"
    if ":" in tok:
        prefix = tok[0]
        ver, code = parse_icd_token(tok)
        lookup_key = ("D" if prefix == "C" else "P", code)
        title = icd_titles.get(lookup_key, "")
        if title:
            return f"{prefix}:{code} ({title})"
    return tok


def truncate_label(label: str, max_len: int = 45) -> str:
    return label if len(label) <= max_len else label[:max_len - 1] + "…"


def stream_color(token_str: str) -> str:
    if token_str.startswith("C:"):
        return "#4C72B0"
    elif token_str.startswith("P:"):
        return "#DD8452"
    elif token_str.startswith("D:"):
        return "#55A868"
    return "#999999"


def token_stream(tok: str) -> str:
    if tok.startswith("C:"):
        return "C"
    elif tok.startswith("P:"):
        return "P"
    elif tok.startswith("D:"):
        return "D"
    return "?"


# ── Prediction helper ────────────────────────────────────────────────────

def predict_one(model, device: torch.device, input_ids: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        out = model(input_ids.to(device))
        return float(torch.sigmoid(out["logits"]).cpu().item())
