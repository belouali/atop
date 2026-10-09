"""regen_tensors_test_fixed: rebuild ``3_attribution/tensors_<split>.pt`` from
the cached sequences while preserving in-sequence [PAD] placeholders.

Why this script exists (the bug it replaces)
--------------------------------------------
``tensors_<split>.pt`` is the cache ``{(patient_id, index_hadm_id):
LongTensor[1, max_seq_len]}`` of the exact model inputs that integrated
gradients were computed on (``atop.attribution.saliency.
process_split_for_sequences`` with ``store_tensors=True``).  When that cache
had to be regenerated, the earlier helper rebuilt it from ``combined_full``,
the list of decoded non-PAD tokens::

    ids = [vocab[t] for t in combined_full] + [PAD] * (max_seq_len - len(combined_full))

That is wrong whenever the original input held a [PAD] *inside* the
sequence.  ``atop.data.datasets.encode_sequence`` maps every
out-of-vocabulary code to ``PAD_IDX`` (``vocab.get(t, PAD_IDX)``) without
removing it, so a visit ``[VISIT] <oov code> A B`` is encoded as
``2 0 a b`` and the model (and IG) saw ``A`` at position 3, not 2.  The
saliency loop skips PAD ids when it decodes ``combined_full``, so a
left-packed rebuild shifts every token after an OOV code one position to
the left.  In the MIMIC-IV test split this affected 3,201 / 43,640 patients
(7.3%); for them every downstream step that pairs the tensor with the
``pos`` column of ``ig_<split>.csv`` (masking / perturbation validation,
SHAP comparison, faithfulness curves) touched the wrong token.

The fix uses ``combined_map`` instead, which stores the encoded position of
every non-PAD token::

    ids = [PAD_IDX] * max_seq_len
    for pos, tok in combined_map:
        ids[pos] = vocab[tok]

This reproduces the original tensors exactly: every ``pos`` in
``ig_<split>.csv`` again indexes the token it was computed for.

Usage::

    python scripts/regen_tensors_test_fixed.py --run_dir ../runs/<run> [--split test]
        [--max_seq_len 512] [--out PATH] [--check REFERENCE.pt] [--check-ig] [--dry-run]

By default writes ``<run_dir>/3_attribution/tensors_<split>.pt`` (an existing
file is first copied to ``.bak``).  ``--check`` reports how many patients
differ from a reference cache; ``--check-ig`` verifies that every IG row's
``pos`` indexes its own token in the rebuilt tensor.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.config import PAD_IDX, AToPConfig  # noqa: E402
from atop.run_layout import RunLayout        # noqa: E402


def rebuild_tensors(df_seq, vocab: dict, max_seq_len: int) -> dict:
    """``{(str pid, str hadm): LongTensor[1, max_seq_len]}`` from ``combined_map``."""
    tensors = {}
    for row in df_seq.itertuples(index=False):
        ids = [PAD_IDX] * max_seq_len
        for pos, tok in row.combined_map:
            if pos >= max_seq_len:
                raise ValueError(f"position {pos} >= max_seq_len={max_seq_len} "
                                 f"for patient {row.patient_id}/{row.index_hadm_id}")
            if tok not in vocab:
                raise KeyError(f"token {tok!r} (patient {row.patient_id}/"
                               f"{row.index_hadm_id}) is not in vocab")
            ids[pos] = vocab[tok]
        key = (str(row.patient_id), str(row.index_hadm_id))
        tensors[key] = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    return tensors


def _infer_max_seq_len(layout: RunLayout) -> int:
    cfg_path = layout.config_json
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        if "max_seq_len" in cfg:
            return int(cfg["max_seq_len"])
    return AToPConfig().max_seq_len


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--max_seq_len", type=int, default=None,
                   help="defaults to 0_inputs/config.json, else AToPConfig.max_seq_len")
    p.add_argument("--out", default=None,
                   help="output path (default <run_dir>/3_attribution/tensors_<split>.pt)")
    p.add_argument("--check", default=None,
                   help="reference tensors .pt to compare against (reports differing patients)")
    p.add_argument("--check-ig", action="store_true",
                   help="verify every ig_<split>.csv row's pos indexes its token")
    p.add_argument("--dry-run", action="store_true", help="do not write anything")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    max_seq_len = args.max_seq_len or _infer_max_seq_len(layout)
    out_path = args.out or layout.tensors_pt(args.split)

    with open(layout.vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    with open(layout.sequences_pkl(args.split), "rb") as f:
        df_seq = pickle.load(f)
    print(f"[1] {len(df_seq):,} sequences, vocab {len(vocab):,}, max_seq_len {max_seq_len}")

    tensors = rebuild_tensors(df_seq, vocab, max_seq_len)
    n_gap = sum(
        [pos for pos, _ in row.combined_map] != list(range(len(row.combined_map)))
        for row in df_seq.itertuples(index=False))
    print(f"[2] rebuilt {len(tensors):,} tensors; {n_gap:,} ({100 * n_gap / len(tensors):.2f}%) "
          f"contain in-sequence [PAD] placeholders that a combined_full rebuild would drop")

    if args.check:
        ref = torch.load(args.check, map_location="cpu", weights_only=False)
        missing = [k for k in tensors if k not in ref]
        n_diff = sum(1 for k in tensors if k in ref and not torch.equal(ref[k], tensors[k]))
        print(f"[3] vs {args.check}: {n_diff:,} / {len(tensors):,} patients differ, "
              f"{len(missing):,} missing from reference, "
              f"{len(set(ref) - set(tensors)):,} extra in reference")

    if args.check_ig:
        import numpy as np
        import pandas as pd
        ig = pd.read_csv(layout.ig_csv(args.split),
                         usecols=["patient_id", "index_hadm_id", "pos", "token_str"])
        n_bad = 0
        for (pid, hid), g in ig.groupby(["patient_id", "index_hadm_id"], sort=False):
            t = tensors[(str(pid), str(hid))][0].numpy()
            ids = np.array([vocab[tok] for tok in g["token_str"]])
            n_bad += int((t[g["pos"].to_numpy()] != ids).any())
        print(f"[4] IG-position check: {n_bad:,} patients whose IG rows do not index "
              f"their own token (expect 0)")

    if args.dry_run:
        print("[dry-run] nothing written")
        return
    if os.path.exists(out_path):
        bak = out_path + ".bak"
        shutil.copy2(out_path, bak)
        print(f"    existing file backed up to {bak}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(tensors, out_path)
    print(f"[5] wrote {out_path}")


if __name__ == "__main__":
    main()
