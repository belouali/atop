"""build_cohort_shap_cache \u2014 GPU-only cohort-wide KernelSHAP runner.

Builds a per-patient-per-token KernelSHAP cache over a stratified-random
sample of test patients, so the methodological argument from Fig 3 / Fig
S9 / Fig S10 (carrier-conditional vs. population-weighted aggregation)
can be replicated under SHAP instead of Integrated Gradients.

Why this exists: the existing per-patient SHAP cache (used by Fig S3) is
limited to the top-K tokens *by IG* per patient \u2014 circular if you want to
build a SHAP-based ranking. This script computes SHAP for **every unique
non-special token** in each sampled patient's sequence, so downstream
aggregation can mirror the IG aggregation choices in Fig 3 directly
(see scripts/build_shap_aggregations.py).

Output: ``<run_dir>/3_attribution/cohort_shap_<split>.pkl``
  dict keyed by (pid, hadm) \u2192 dict {token_str: signed_shap}

**Single growing cache** (no ``n<sample>`` in filename): all runs against
the same split write to the same file, so ``--sample_n`` is interpreted as
"ensure cache contains AT LEAST N patients." Re-running with a larger N
**extends** the previous cache rather than re-computing. The deterministic
stratified ordering (seeded) means the first N entries are always the same
set across runs; ``--sample_n 1000`` adds 500 new patients to a cache built
at 500.

Resilience: incremental save every ``flush_every`` patients (atomic write
via .tmp + os.replace). Resume by re-running with the same flags \u2014 the
script skips any (pid, hadm) keys already in the cache.

Compute estimate (Colab T4):
  * ~10\u201330 s/patient at ``shap-nsamples=50``
  * 500 patients \u2192 ~1.5 hours
  * 1000 patients \u2192 ~3 hours
  * 5000 patients \u2192 ~15 hours (multi-session)
  * "all" (~43,640 patients) \u2192 ~5\u201310 days, not practically useful
  * Drop ``shap-nsamples`` to 25 if you need it faster (population
    averaging across 500+ patients tolerates noisier per-patient SHAP).

Colab usage (from atop_package/):
    # First run: build 500 patients
    !python scripts/build_cohort_shap_cache.py \\
        --run_dir ../runs/full_CPD_v2 \\
        --mimic_dir ../data/mimiciv/3.1/hosp \\
        --device cuda --sample_n 500

    # Later: extend to 1000 (only computes the 500 new patients)
    !python scripts/build_cohort_shap_cache.py \\
        --run_dir ../runs/full_CPD_v2 \\
        --mimic_dir ../data/mimiciv/3.1/hosp \\
        --device cuda --sample_n 1000

    # All: extend until every test patient is covered
    !python scripts/build_cohort_shap_cache.py \\
        --run_dir ../runs/full_CPD_v2 \\
        --mimic_dir ../data/mimiciv/3.1/hosp \\
        --device cuda --sample_n all
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.config import PAD_IDX, CLS_IDX, VISIT_IDX, SPECIAL_TOKENS
from atop.models.single_stream_transformer import SingleStreamTransformer
from atop.run_layout import RunLayout


def _shap_for_all_tokens(pid: str, hadm: str, *,
                           tensors, sequences, model, device,
                           shap_nsamples: int, chunk: int, seed: int):
    """KernelSHAP on every unique non-special token in this patient's
    sequence. Returns ``{token_str: signed_shap_value}`` averaged across
    all positions where the token appears (so a token occurring twice in
    the sequence contributes one combined SHAP value)."""
    import shap as shap_lib

    key = (pid, hadm)
    if key not in tensors:
        return None
    input_ids_tensor = tensors[key]

    # Pull this patient's sequence to recover token strings + positions.
    seq_row = sequences[(sequences["patient_id"] == pid)
                         & (sequences["index_hadm_id"] == hadm)]
    if len(seq_row) == 0:
        return None
    seq_row = seq_row.iloc[0]

    # Map token positions in the input_ids tensor.
    combined_map = seq_row["combined_map"]   # list of (pos, token_str)
    feats = [(pos, tok) for pos, tok in combined_map
             if tok not in SPECIAL_TOKENS]
    if not feats:
        return None
    K = len(feats)

    base_ids = input_ids_tensor.clone()

    def predict_masked(z):
        batch = []
        for row in z:
            inp = base_ids.clone()
            for j, keep in enumerate(row.tolist()):
                if keep < 0.5:
                    inp[0, feats[j][0]] = PAD_IDX
            batch.append(inp)
        stacked = torch.cat(batch, dim=0)
        all_p = []
        with torch.no_grad():
            for i in range(0, len(stacked), chunk):
                ch = stacked[i:i + chunk].to(device)
                out = model(ch)
                all_p.append(out["y_prob"].cpu().numpy())
        return np.concatenate(all_p, axis=0)

    rng_bg = np.random.RandomState(seed)
    bg = rng_bg.randint(0, 2, size=(10, K)).astype(float)
    x_inst = np.ones((1, K))
    expl = shap_lib.KernelExplainer(predict_masked, bg)
    sv = expl.shap_values(x_inst, nsamples=shap_nsamples, silent=True)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv).reshape(-1)

    # Aggregate: one signed SHAP per UNIQUE token (sum positions).
    out: dict = {}
    for (_, tok), shap_v in zip(feats, sv.tolist()):
        out[tok] = out.get(tok, 0.0) + float(shap_v)
    return out


def _atomic_save(cache: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _stratified_ordering(df_seq: pd.DataFrame, df_pred: pd.DataFrame,
                            seed: int) -> list:
    """**Deterministic** stratified ordering of every test (pid, hadm) key.

    Returns the FULL cohort in an order such that any prefix ``[:N]`` is
    a near-balanced stratified sample of N patients (positive/negative
    readmit ratios match the cohort base rate). Same seed \u2192 same ordering,
    so ``[:500]`` is identical across runs and ``[:1000]`` is a strict
    extension of ``[:500]``. Lets the cache grow incrementally without
    redoing previously-cached work.
    """
    # Predictions CSV uses MIMIC-native column names; rename to match the
    # sequences DataFrame for the join.
    rename_map = {}
    if "subject_id" in df_pred.columns:
        rename_map["subject_id"] = "patient_id"
    if "hadm_id" in df_pred.columns and "index_hadm_id" not in df_pred.columns:
        rename_map["hadm_id"] = "index_hadm_id"
    pred = df_pred.rename(columns=rename_map)
    # IDs come through as floats (e.g. 10000068.0) \u2014 cast via int first to
    # drop the decimal, then to str.
    for col in ("patient_id", "index_hadm_id"):
        pred[col] = pred[col].astype("Int64").astype(str)
    pred = pred[["patient_id", "index_hadm_id", "y_true"]].copy()
    pred["y_true"] = pred["y_true"].astype(int)
    seq = df_seq[["patient_id", "index_hadm_id"]].astype(str).copy()
    joined = seq.merge(pred, on=["patient_id", "index_hadm_id"], how="inner")
    if joined.empty:
        sys.exit(f"  [ERROR] No overlap between sequences and predictions \u2014 "
                 f"check ID format. Seq sample: {seq.iloc[0].to_dict()}; "
                 f"Pred sample: {pred.iloc[0].to_dict()}")

    pos = joined[joined["y_true"] == 1].reset_index(drop=True)
    neg = joined[joined["y_true"] == 0].reset_index(drop=True)
    p_rate = len(pos) / len(joined)
    print(f"  [ordering] cohort: {len(pos):,} pos + {len(neg):,} neg "
          f"(base rate {p_rate:.4f})")

    rng = np.random.RandomState(seed)
    pos_perm = rng.permutation(len(pos)).tolist()
    neg_perm = rng.permutation(len(neg)).tolist()

    # Interleave so that any prefix maintains base-rate stratification.
    # At each position, take a positive iff its share-so-far is below p_rate.
    out = []
    pi = ni = 0
    while pi < len(pos_perm) or ni < len(neg_perm):
        if pi >= len(pos_perm):
            out.append((neg.iloc[neg_perm[ni]]["patient_id"],
                          neg.iloc[neg_perm[ni]]["index_hadm_id"]))
            ni += 1
        elif ni >= len(neg_perm):
            out.append((pos.iloc[pos_perm[pi]]["patient_id"],
                          pos.iloc[pos_perm[pi]]["index_hadm_id"]))
            pi += 1
        else:
            done = pi + ni
            cur_p_share = pi / max(done, 1)
            if cur_p_share < p_rate:
                out.append((pos.iloc[pos_perm[pi]]["patient_id"],
                              pos.iloc[pos_perm[pi]]["index_hadm_id"]))
                pi += 1
            else:
                out.append((neg.iloc[neg_perm[ni]]["patient_id"],
                              neg.iloc[neg_perm[ni]]["index_hadm_id"]))
                ni += 1
    print(f"  [ordering] built deterministic stratified ordering of "
          f"{len(out):,} (pid, hadm) keys")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build cohort KernelSHAP cache (per-patient, per-token).")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--mimic_dir", required=True,
                   help="MIMIC-IV hosp/ directory \u2014 needed to load model "
                        "config + drug names.")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample_n", default="500",
                   help="Target cohort size (default 500). Pass an integer or "
                        "'all' to extend until every test patient is cached. "
                        "Re-running with a larger N extends the existing cache "
                        "rather than recomputing (deterministic stratified "
                        "ordering means the first N entries are stable).")
    p.add_argument("--shap_nsamples", type=int, default=50,
                   help="KernelSHAP samples per patient (default 50; bump "
                        "to 100\u2013200 for tighter per-patient values).")
    p.add_argument("--chunk", type=int, default=64,
                   help="Forward-pass batch size per KernelSHAP coalition.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flush_every", type=int, default=10,
                   help="Save cache to disk every N patients (atomic write).")
    p.add_argument("--cache_path", default=None,
                   help="Cache output path (default: <run_dir>/3_attribution/"
                        "cohort_shap_<split>.pkl \u2014 single growing cache, "
                        "no n-suffix).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be computed, then exit.")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    cache_path = args.cache_path or os.path.join(
        layout.attribution_dir, f"cohort_shap_{args.split}.pkl")

    # ── Load model + tensors + sequences + predictions ───────────────
    print(f"[1] Loading model bundle from {args.run_dir}\u2026")
    with open(layout.vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    with open(layout.config_json) as f:
        cfg = json.load(f)

    state = torch.load(layout.model_pt, map_location="cpu", weights_only=False)
    vocab_size = state["token_emb.weight"].shape[0]
    emb_dim = state["token_emb.weight"].shape[1]
    max_seq_len = state["pos_emb.weight"].shape[0]
    model = SingleStreamTransformer(
        vocab_size=vocab_size, embedding_dim=emb_dim, max_seq_len=max_seq_len,
        num_heads=cfg.get("heads", 4), num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(state)
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    print(f"  vocab={vocab_size}  emb_dim={emb_dim}  device={device}")

    tensors = torch.load(layout.tensors_pt(args.split),
                          map_location="cpu", weights_only=False)
    sequences = pd.read_pickle(layout.sequences_pkl(args.split))
    sequences = sequences.astype({"patient_id": str, "index_hadm_id": str})
    df_pred = pd.read_csv(layout.test_predictions_csv)
    print(f"  tensors={len(tensors):,}  sequences={len(sequences):,}  "
          f"preds={len(df_pred):,}")

    # ── Deterministic stratified ordering of the full cohort ────────
    print(f"\n[2] Building deterministic stratified ordering (seed={args.seed})\u2026")
    full_ordering = _stratified_ordering(sequences, df_pred, args.seed)
    cohort_size = len(full_ordering)

    # Parse --sample_n: integer or 'all'.
    if str(args.sample_n).lower() == "all":
        target_n = cohort_size
    else:
        try:
            target_n = int(args.sample_n)
        except ValueError:
            sys.exit(f"  [ERROR] --sample_n must be an integer or 'all' "
                     f"(got {args.sample_n!r})")
        if target_n > cohort_size:
            print(f"  [WARN] target_n={target_n:,} > cohort size "
                  f"{cohort_size:,}; capping to cohort size.")
            target_n = cohort_size

    sample_keys = full_ordering[:target_n]
    print(f"  target = {target_n:,} of {cohort_size:,} test patients")

    # ── Load existing cache, determine missing patients ──────────────
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"\n[3] Existing cache: {len(cache):,} patients in "
              f"{os.path.basename(cache_path)}")
    else:
        cache = {}
        print(f"\n[3] No existing cache \u2014 starting fresh.")

    cached_in_target = sum(1 for k in sample_keys if k in cache)
    missing = [k for k in sample_keys if k not in cache]
    print(f"  Cached within target window: {cached_in_target:,}/{target_n:,}")
    print(f"  Need GPU compute:            {len(missing):,} patients")
    if cache and cached_in_target < len(cache):
        # Cache contains entries OUTSIDE the deterministic prefix \u2014 either
        # an earlier seed or hand-curated keys. Surface this so the user
        # knows their cache is bigger than the target window indicates.
        extra = len(cache) - cached_in_target
        print(f"  [info] cache holds {extra:,} additional patients outside "
              f"the current target window (different seed?). Those are kept.")

    if not missing:
        print("    Nothing to do. Exiting.")
        return

    if args.dry_run:
        print(f"\n  Dry run \u2014 first 5 keys that would be computed:")
        for k in missing[:5]:
            print(f"    {k}")
        return

    # ── KernelSHAP per missing patient ───────────────────────────────
    print(f"\n[4] Running KernelSHAP for {len(missing)} patients "
          f"(nsamples={args.shap_nsamples}, chunk={args.chunk})\u2026")
    t0 = time.time()
    n_done = 0
    n_skipped = 0
    for i, (pid, hadm) in enumerate(missing, start=1):
        try:
            result = _shap_for_all_tokens(
                pid, hadm,
                tensors=tensors, sequences=sequences, model=model, device=device,
                shap_nsamples=args.shap_nsamples, chunk=args.chunk, seed=args.seed,
            )
        except Exception as e:
            print(f"\n  [WARN] {pid}/{hadm} failed: {e}")
            n_skipped += 1
            continue

        if result is None:
            n_skipped += 1
            continue

        cache[(pid, hadm)] = result
        n_done += 1

        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-3)
        eta_s = (len(missing) - i) / max(rate, 1e-3)
        print(f"\r  [{i:>4}/{len(missing)}]  done={n_done}  skipped={n_skipped}  "
              f"rate={rate:.2f}/s  ETA={eta_s/60:.1f} min     ",
              end="", flush=True)

        if n_done % args.flush_every == 0:
            _atomic_save(cache, cache_path)
    print()

    # ── Final flush ──────────────────────────────────────────────────
    _atomic_save(cache, cache_path)
    print(f"\n[5] Saved cache: {cache_path} ({len(cache)} patients)")
    if n_skipped:
        print(f"  [WARN] {n_skipped} patients skipped (missing tensor / "
              f"sequence / SHAP failure)")


if __name__ == "__main__":
    main()
