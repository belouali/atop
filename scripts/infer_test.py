"""
Run the trained SingleStreamTransformer on cached test tensors and save
per-patient predictions for bootstrap CI computation.

Inputs:
  runs/full_CPD/model.pt                    — state dict
  runs/full_CPD/ig_cache/tensors_test.pt    — dict[(pid, hadm_id) -> (1, 512) LongTensor]
  runs/full_CPD/explain/data/sequences_test.pkl — cohort with readmission label

Output:
  runs/full_CPD/test_predictions.csv        — cols: subject_id, hadm_id, y_true, y_prob
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.models.single_stream_transformer import SingleStreamTransformer

FULL_RUN = "../runs/full_CPD"
OUT_CSV  = f"{FULL_RUN}/test_predictions.csv"

print("[1] Loading model weights...")
state = torch.load(f"{FULL_RUN}/model.pt", map_location="cpu", weights_only=False)
vocab_size = state["token_emb.weight"].shape[0]
embedding_dim = state["token_emb.weight"].shape[1]
max_seq_len = state["pos_emb.weight"].shape[0]
print(f"    vocab={vocab_size}  d={embedding_dim}  L={max_seq_len}")

# Config from runs/full_CPD/config.json: heads=4, num_layers=2, dropout=0.1
model = SingleStreamTransformer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    max_seq_len=max_seq_len,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
)
model.load_state_dict(state)
model.eval()

print("[2] Loading cached test tensors...")
tensors = torch.load(f"{FULL_RUN}/ig_cache/tensors_test.pt", map_location="cpu", weights_only=False)
print(f"    {len(tensors):,} test patients")

print("[3] Loading labels from sequences_test.pkl...")
with open(f"{FULL_RUN}/explain/data/sequences_test.pkl", "rb") as f:
    seqs = pickle.load(f)
label_map = {
    (str(r["patient_id"]), str(r["index_hadm_id"])): float(r["readmission"])
    for _, r in seqs.iterrows()
}
print(f"    {len(label_map):,} labels")

print("[4] Running inference in batches of 128...")
keys = list(tensors.keys())
BATCH = 128
results = []
with torch.no_grad():
    for i in range(0, len(keys), BATCH):
        batch_keys = keys[i:i+BATCH]
        batch_ids  = torch.cat([tensors[k] for k in batch_keys], dim=0)  # (B, 512)
        out = model(batch_ids)
        probs = out["y_prob"].numpy()
        for k, p in zip(batch_keys, probs):
            pid, hid = str(k[0]), str(k[1])
            y = label_map.get((pid, hid))
            results.append({"subject_id": pid, "hadm_id": hid, "y_true": y, "y_prob": float(p)})
        if (i // BATCH) % 50 == 0:
            print(f"    {i + len(batch_keys):,}/{len(keys):,}")

df = pd.DataFrame(results)
df = df.dropna(subset=["y_true"])
df.to_csv(OUT_CSV, index=False)
print(f"[5] Saved {len(df):,} predictions to {OUT_CSV}")

from sklearn.metrics import roc_auc_score, average_precision_score
print(f"    AUROC = {roc_auc_score(df['y_true'], df['y_prob']):.4f}")
print(f"    PR-AUC= {average_precision_score(df['y_true'], df['y_prob']):.4f}")
