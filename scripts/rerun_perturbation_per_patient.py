"""Backup analysis (not reported): per-patient rerun of the 266-pattern
perturbation analysis, keeping absolute per-patient changes. Paths are
relative to a scratch bundle (run/0_inputs, run/1_model,
run/3_attribution/{tensors_test.pt,sequences_test.pkl}) plus
validation_test.csv. See parked_supplement_blocks.py.


Identical to atop.explain.validation.run_validation (same carrier matching,
first 200 carriers in test-cohort order, mask / within-visit shuffle /
visit-block shuffle, 50 draws each), but additionally stores, per carrier:
  d_mask              = y_orig - y_masked
  d_within_mean       = mean over draws of (y_orig - y_shuffled_within)
  d_within_abs        = mean over draws of |y_orig - y_shuffled_within|
  d_visits_mean / d_visits_abs  (same for visit-block shuffle)
Masking is deterministic, so per-pattern mean d_mask must reproduce
validation_test.csv exactly; that check proves the setup matches.
"""
import sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, '.')
from atop.registry import load_bundle
from atop.explain.validation import (mask_positions, shuffle_within_visits,
    shuffle_visit_blocks, predict_batch, find_episode_occurrence)
from atop.explain.validation import _parse_episode_pattern

torch.set_num_threads(max(1, torch.get_num_threads()))
np.random.seed(42); torch.manual_seed(42)
dev = torch.device('cpu')
cfg, model, vocab, splits = load_bundle('run', dev)
vocab_inv = {v: k for k, v in vocab.items()}
T = torch.load('run/3_attribution/tensors_test.pt', weights_only=False)
seq = pd.read_pickle('run/3_attribution/sequences_test.pkl')
val = pd.read_csv('validation_test.csv')
N_DRAWS = cfg.n_shuffle_draws; MAX_ADM = cfg.validate_max_admissions_per_pattern
print(f"patterns={len(val)} draws={N_DRAWS} max_carriers={MAX_ADM}", flush=True)

rows, check = [], []
t0 = time.time()
for pi, pat_str in enumerate(val['pattern']):
    ep = _parse_episode_pattern(pat_str)
    matches = []
    for _, r in seq.iterrows():
        key = (r['patient_id'], r['index_hadm_id'])
        if key not in T:
            continue
        occ = find_episode_occurrence(r['combined_full'], r['combined_map'], ep)
        if occ:
            matches.append((key, occ))
        if len(matches) >= MAX_ADM:
            break
    base_t = [T[k] for k, _ in matches]
    base = predict_batch(model, dev, base_t)
    masked = predict_batch(model, dev, [mask_positions(b, [p for p, _ in o]) for b, (_, o) in zip(base_t, matches)])
    sw = predict_batch(model, dev, [shuffle_within_visits(b, vocab, vocab_inv) for b in base_t for _ in range(N_DRAWS)]).reshape(len(matches), N_DRAWS)
    sv = predict_batch(model, dev, [shuffle_visit_blocks(b) for b in base_t for _ in range(N_DRAWS)]).reshape(len(matches), N_DRAWS)
    dw = base[:, None] - sw; dv = base[:, None] - sv
    for (k, _), b, m, a, c, e, f in zip(matches, base, masked, dw.mean(1), np.abs(dw).mean(1), dv.mean(1), np.abs(dv).mean(1)):
        rows.append(dict(pattern=pat_str, patient_id=k[0], index_hadm_id=k[1], y_orig=b,
                         d_mask=b - m, d_within_mean=a, d_within_abs=c, d_visits_mean=e, d_visits_abs=f))
    check.append((pat_str, float((base - masked).mean()), float(val.loc[pi, 'mean_delta_mask']), len(matches), int(val.loc[pi, 'n_matched_test'])))
    if (pi + 1) % 10 == 0 or pi == 0:
        el = time.time() - t0
        print(f"{pi+1}/{len(val)} patterns, {el/60:.1f} min elapsed, ~{el/(pi+1)*(len(val)-pi-1)/60:.0f} min left", flush=True)
        pd.DataFrame(rows).to_csv('perturbation_per_patient.csv', index=False)

pd.DataFrame(rows).to_csv('perturbation_per_patient.csv', index=False)
c = pd.DataFrame(check, columns=['pattern', 'mask_rerun', 'mask_saved', 'n_rerun', 'n_saved'])
c.to_csv('perturbation_check.csv', index=False)
print("max |mask_rerun - mask_saved| =", (c.mask_rerun - c.mask_saved).abs().max(),
      "| carrier counts identical:", bool((c.n_rerun == c.n_saved).all()))
print("DONE")
