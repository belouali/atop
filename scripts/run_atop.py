#!/usr/bin/env python3
"""
Run the AToP analysis pipeline on a trained model bundle.

Computes IG attribution (cached), mines temporal patterns (cached), and runs
analysis steps (mining global attribution data, perturbation validation,
reversed-order ablation). Writes data CSVs to ``5_analysis/`` and run-dir
diagnostic figures to ``7_outputs/figures/``.

**Manuscript figures and tables are rendered separately by ``scripts/make_*.py``**
from the data this pipeline produces.

Usage:
    python scripts/run_atop.py --run_dir runs/mimiciv_readmission_30d \\
        --mimic_dir ../data/mimiciv/3.1/hosp \\
        --steps mine_global,validate
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.run_layout import RunLayout


class TeeLogger:
    """Duplicate stdout/stderr to both console and a log file."""

    def __init__(self, log_path: str):
        self.log_file = open(log_path, "w", buffering=1)  # line-buffered
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.log_path = log_path

    def start(self):
        sys.stdout = self
        sys.stderr = _StderrTee(self.log_file, self.stderr)
        self.log_file.write(f"=== AToP Run Log ===\n")
        self.log_file.write(f"Started: {datetime.now().isoformat()}\n")
        self.log_file.write(f"Command: {' '.join(sys.argv)}\n")
        self.log_file.write(f"{'='*60}\n\n")

    def write(self, msg):
        self.stdout.write(msg)
        self.log_file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def stop(self):
        self.log_file.write(f"\n{'='*60}\n")
        self.log_file.write(f"Finished: {datetime.now().isoformat()}\n")
        self.log_file.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr


class _StderrTee:
    """Tee stderr to both console and log file."""

    def __init__(self, log_file, stderr):
        self.log_file = log_file
        self.stderr = stderr

    def write(self, msg):
        self.stderr.write(msg)
        self.log_file.write(f"[STDERR] {msg}")

    def flush(self):
        self.stderr.flush()
        self.log_file.flush()


def parse_args():
    p = argparse.ArgumentParser(
        description="Run AToP analysis pipeline (IG → mining → analysis steps)"
    )
    # ── Required ─────────────────────────────────────────────────────
    p.add_argument("--run_dir", type=str, required=True,
                   help="Path to v2 run directory (from train_model.py)")
    p.add_argument("--mimic_dir", type=str, required=True,
                   help="Path to MIMIC-IV hosp/ directory")
    p.add_argument("--device", type=str, default="auto",
                   help="cuda / cpu / auto")

    # ── Step selector ────────────────────────────────────────────────
    p.add_argument("--steps", type=str, default="",
                   help="Comma-separated analysis steps to run (default: all). "
                        "Valid: patient_ig, mine_global, validate, reversed_order, "
                        "tables, data.")
    p.add_argument("--n_show", type=int, default=-1,
                   help="Number of items per panel (default: 15)")
    p.add_argument("--n_exemplars", type=int, default=3,
                   help="Number of patient exemplars for patient_ig step (default: 3)")
    p.add_argument("--validate_top_k", type=int, default=-1,
                   help="Number of risk + protective patterns for validate step "
                        "(default: 15 each)")

    # ── Mining knobs (locked at specific values for JAMIA submission, but
    # exposed so reviewer-response sensitivity analyses can override) ──
    p.add_argument("--episode_min_steps", type=int, default=-1,
                   help="Minimum visit blocks per pattern (locked: 2). -1 = use config.")
    p.add_argument("--episode_max_len", type=int, default=-1,
                   help="Maximum pattern length in visit blocks (locked: 4). -1 = use config.")
    p.add_argument("--min_support_frac", type=float, default=-1,
                   help="Mining min-support fraction per group (locked: 0.001). "
                        "-1 = use config.")
    p.add_argument("--scoring_min_support_frac", type=float, default=-99,
                   help="Scoring min-support fraction (locked: 0 disables). "
                        "-99 = use config.")
    p.add_argument("--jaccard_dedup", type=float, default=-1,
                   help="Mining-time Jaccard dedup threshold (locked: 0.0 disables; "
                        "display-time dedup happens separately at j=0.2). -1 = use config.")
    return p.parse_args()


def main():
    args = parse_args()
    layout = RunLayout(args.run_dir)
    layout.mkdirs()  # ensures 0_..7_ stage dirs exist

    # Log into the run dir's diagnostics folder (v2-aware path)
    log_dir = os.path.join(layout.outputs_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")
    logger = TeeLogger(log_path)
    logger.start()

    t0 = time.time()

    try:
        from atop.explainer import AToPExplainer

        explainer = AToPExplainer.from_bundle(
            args.run_dir, mimic_dir=args.mimic_dir, device=args.device)

        # Apply CLI overrides if given
        if args.episode_min_steps >= 0:
            explainer.config.episode_min_steps = args.episode_min_steps
        if args.episode_max_len >= 0:
            explainer.config.episode_max_len = args.episode_max_len
        if args.min_support_frac >= 0:
            explainer.config.min_support_frac = args.min_support_frac
        if args.scoring_min_support_frac > -99:
            explainer.config.scoring_min_support_frac = args.scoring_min_support_frac
        if args.jaccard_dedup >= 0:
            explainer.config.jaccard_dedup = args.jaccard_dedup
        if args.n_show >= 0:
            explainer.config.n_show = args.n_show
        if args.validate_top_k >= 0:
            explainer.config.validate_top_k = args.validate_top_k

        print(f"[config] run_dir = {args.run_dir}")
        print(f"[config] device  = {args.device}")
        print(f"[config] log     = {log_path}")
        print()

        explainer.compute_attributions()
        explainer.mine_patterns()

        steps = [s.strip() for s in args.steps.split(",") if s.strip()] or None
        # report() is v2-aware: writes to 5_analysis/, 7_outputs/figures/, etc.
        # via RunLayout. The legacy out_dir argument is unused in v2 mode.
        explainer.report(out_dir=layout.run_dir, steps=steps,
                         n_exemplars=args.n_exemplars)

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed/60:.1f} min.")
        print(f"Log: {log_path}")

    except Exception as e:
        import traceback
        print(f"\n[FATAL] {e}")
        traceback.print_exc()

    finally:
        logger.stop()


if __name__ == "__main__":
    main()
