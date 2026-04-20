#!/usr/bin/env python3
"""
Standalone LACE index computation for MIMIC-IV.

Computes the LACE readmission risk score for every admission.
Output: lace_scores_all.csv

Usage:
  python scripts/precompute_lace.py --mimic_dir /data/mimiciv/2.2/hosp
  python scripts/precompute_lace.py --mimic_dir /data/mimiciv/2.2/hosp --out lace_scores_all.csv

Then pass to AToP:
  python scripts/run_atop.py --lace_csv lace_scores_all.csv ...
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.data.mimic import load_mimic_tables
from atop.baselines.lace import compute_lace


def main():
    p = argparse.ArgumentParser(
        description="Compute LACE readmission risk scores for all MIMIC-IV admissions")
    p.add_argument("--mimic_dir", type=str, required=True,
                   help="Path to MIMIC-IV hosp/ directory")
    p.add_argument("--out", type=str, default="lace_scores_all.csv",
                   help="Output CSV path (default: lace_scores_all.csv)")
    args = p.parse_args()

    t0 = time.time()
    admissions, diagnoses, _, _ = load_mimic_tables(args.mimic_dir)
    df = compute_lace(admissions, diagnoses, mimic_dir=args.mimic_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df):,} scores -> {args.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
