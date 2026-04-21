"""
Regenerate runs/full_CPD/ig_cache/tensors_test.pt without recomputing IG.

The tensors are just encoded input_ids produced by MIMICReadmissionDataset;
they don't depend on model gradients. When the model hash changes (e.g.
after a re-unpack updates model.pt's mtime) explainer.py wipes the IG
cache entirely, but the tensors themselves are still valid for the same
vocabulary/sequences — so we just rebuild them and avoid the ~2 hour
full IG compute on 43k test patients.
"""
from __future__ import annotations
import os, sys, pickle, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.data.datasets import MIMICReadmissionDataset, collate_fn

BUNDLE = "../runs/full_CPD"
OUT    = f"{BUNDLE}/ig_cache/tensors_test.pt"

print("[1] Loading vocab.pkl...")
with open(f"{BUNDLE}/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
print(f"    vocab size: {len(vocab)}")

print("[2] Loading test sequences pkl (precomputed by earlier pipeline run)...")
with open(f"{BUNDLE}/explain/data/sequences_test.pkl", "rb") as f:
    df = pickle.load(f)
print(f"    shape: {df.shape}")

# MIMICReadmissionDataset expects a list of sample dicts with keys
# matching the one-per-patient cache schema. The DataFrame has the
# required columns already — convert rows to dicts the dataset can use.
print("[3] Building sample list for the test dataset...")
samples = []
for _, r in df.iterrows():
    samples.append({
        "patient_id": r["patient_id"],
        "index_hadm_id": r["index_hadm_id"],
        "readmit_30d": int(r["readmission"]),
        "n_visits": int(r["n_visits"]),
        "visit_hadm_ids": [],              # not used by encoder
        "flat_tokens": r["combined_full"],
        "token_visit_idx": [None] + [0] * (len(r["combined_full"]) - 1)
        if isinstance(r["combined_full"], list) else [],
    })
print(f"    built {len(samples):,} samples")

# We don't have max_seq_len directly — get it from config.json
import json
with open(f"{BUNDLE}/config.json") as f:
    cfg = json.load(f)
max_seq_len = int(cfg.get("max_seq_len", 512))
print(f"    max_seq_len: {max_seq_len}")

ds = MIMICReadmissionDataset(samples, vocab, max_seq_len)
loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

print("[4] Iterating dataset to collect input_ids by (pid, hadm_id)...")
tensors_by_key = {}
total = 0
for batch in loader:
    input_ids = batch["input_ids"]          # (B, L)
    B = input_ids.shape[0]
    for b in range(B):
        pid = str(batch["patient_id"][b])
        hadm = str(batch["index_hadm_id"][b])
        tensors_by_key[(pid, hadm)] = input_ids[b].unsqueeze(0).clone()
    total += B
    if total % 4096 == 0:
        print(f"    progress: {total:,}/{len(samples):,}")
print(f"    collected {len(tensors_by_key):,} tensors")

print(f"[5] Saving → {OUT}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save(tensors_by_key, OUT)
print(f"    ✅ saved ({os.path.getsize(OUT)/1024/1024:.1f} MB)")
