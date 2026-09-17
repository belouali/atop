"""make_table_s1, Table S1 (paired bootstrap 95% CIs for AUROC comparisons).

Thin wrapper around ``scripts/table_s1_auroc_bootstrap.py``, passes
``--run_dir``, ``--mimic_dir``, and ``--n-boot`` via env vars (which the
engine reads on import) so the engine doesn't need its own argparse.

Compares AToP Transformer vs LACE LR vs LACE XGBoost on the test set,
1,000 paired bootstrap resamples (~30 s + bootstrap time).

Outputs:
  - <run_dir>/6_baselines/auroc_bootstrap.csv         (per-comparison stats)
  - <run_dir>/6_baselines/auroc_bootstrap_table.md    (paste-ready Markdown)

Usage:
    python scripts/make_table_s1.py --run_dir ../runs/full_CPD_v1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Compute Table S1 (bootstrap CIs)")
    p.add_argument("--run_dir", default="../runs/full_CPD_v1")
    p.add_argument("--mimic_dir", default="../data/mimiciv/3.1/hosp")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    here = os.path.dirname(__file__)
    src = os.path.join(here, "table_s1_auroc_bootstrap.py")
    if not os.path.exists(src):
        sys.exit(f"  [ERROR] Source script missing: {src}")

    env = os.environ.copy()
    env["ATOP_RUN_DIR"] = args.run_dir
    env["ATOP_MIMIC_DIR"] = args.mimic_dir
    env["ATOP_N_BOOT"] = str(args.n_boot)

    print(f"[1] Running Table S1 bootstrap "
          f"(run_dir={args.run_dir}, n_boot={args.n_boot})...")
    sys.exit(subprocess.run([sys.executable, src], env=env, cwd=here).returncode)


if __name__ == "__main__":
    main()
