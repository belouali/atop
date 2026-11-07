"""Vocabulary, encoding, Dataset, collate, and splitting."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from atop.config import PAD_IDX, SPECIAL_TOKENS


def build_vocabulary(samples: List[Dict[str, Any]], min_freq: int = 2) -> Dict[str, int]:
    """Build token→index vocabulary from training samples."""
    counter = Counter()
    for s in samples:
        for tok in s["flat_tokens"]:
            if tok not in SPECIAL_TOKENS:
                counter[tok] += 1

    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    next_idx = len(SPECIAL_TOKENS)
    for tok, cnt in counter.most_common():
        if cnt >= min_freq:
            vocab[tok] = next_idx
            next_idx += 1

    print(f"[vocab] {len(vocab):,} tokens (min_freq={min_freq})")
    return vocab



def encode_sequence(tokens: List[str], vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    """Encode token list to index tensor. Truncate from LEFT (drop oldest visits) but
    always preserve the [CLS] token at position 0. Pad RIGHT with PAD_IDX."""
    ids = [vocab.get(t, PAD_IDX) for t in tokens]
    if len(ids) > max_len:
        # Keep CLS (first token) + last (max_len - 1) tokens of the remainder
        ids = [ids[0]] + ids[-(max_len - 1):]
    ids = ids + [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ============================================================================
# DATASET + DATALOADER
# ============================================================================


class MIMICReadmissionDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], vocab: Dict[str, int], max_seq_len: int):
        self.samples = samples
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        input_ids = encode_sequence(s["flat_tokens"], self.vocab, self.max_seq_len)
        label = torch.tensor(s["readmit_30d"], dtype=torch.float)
        return {
            "input_ids": input_ids,
            "label": label,
            "patient_id": s["patient_id"],
            "index_hadm_id": s["index_hadm_id"],
            "n_visits": s["n_visits"],
            "flat_tokens": s["flat_tokens"],
            "token_visit_idx": s["token_visit_idx"],
        }



def collate_fn(batch):
    """Custom collate: stack tensors, keep metadata as lists."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {
        "input_ids": input_ids,
        "label": labels,
        "patient_id": [b["patient_id"] for b in batch],
        "index_hadm_id": [b["index_hadm_id"] for b in batch],
        "n_visits": [b["n_visits"] for b in batch],
        "flat_tokens": [b["flat_tokens"] for b in batch],
        "token_visit_idx": [b["token_visit_idx"] for b in batch],
    }


# ============================================================================
# MODEL, Single-sequence Transformer
# ============================================================================


def split_samples_by_patient(samples: List[Dict[str, Any]], ratios=(0.7, 0.1, 0.2),
                             seed: int = 42) -> Tuple[List, List, List]:
    """Split samples by patient (all samples for a patient go to same split)."""
    pids = list(set(s["patient_id"] for s in samples))
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)

    n1 = int(len(pids) * ratios[0])
    n2 = int(len(pids) * (ratios[0] + ratios[1]))
    train_pids = set(pids[:n1])
    val_pids = set(pids[n1:n2])
    test_pids = set(pids[n2:])

    train = [s for s in samples if s["patient_id"] in train_pids]
    val = [s for s in samples if s["patient_id"] in val_pids]
    test = [s for s in samples if s["patient_id"] in test_pids]
    return train, val, test


# ============================================================================
# IG PROCESSING
# ============================================================================
