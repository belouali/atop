"""make_table2, Table 2 (model performance: AToP vs LACE baselines).

Thin wrapper around ``scripts/compare_baselines.py`` (which is already
parameterized). Computes AUROC + PR-AUC for AToP, LACE LR, and LACE
XGBoost on the test split. Writes
``<run_dir>/baselines/baseline_comparison.csv``.

Usage:
    python scripts/make_table2.py --run_dir ../runs/full_CPD_v1 \
                                  --mimic_dir ../data/mimiciv/3.1/hosp
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Compute Table 2 (AToP vs LACE baselines)")
    p.add_argument("--run_dir", default="../runs/full_CPD_v1")
    p.add_argument("--mimic_dir", default="../data/mimiciv/3.1/hosp")
    p.add_argument("--lace_csv", default="lace_scores_all.csv",
                   help="Filename relative to run_dir (default: lace_scores_all.csv)")
    p.add_argument("--bootstrap", action="store_true",
                   help="Also run paired bootstrap CIs (slower)")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()

    here = os.path.dirname(__file__)
    cmd = [
        sys.executable, os.path.join(here, "compare_baselines.py"),
        "--run_dir", args.run_dir,
        "--mimic_dir", args.mimic_dir,
        "--lace_csv", args.lace_csv,
    ]
    if args.bootstrap:
        cmd += ["--bootstrap", "--n_boot", str(args.n_boot)]

    print(f"[run] {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
