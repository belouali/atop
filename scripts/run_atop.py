#!/usr/bin/env python3
"""
Run the AToP explanation pipeline on a trained model bundle.

Computes IG attributions, mines temporal patterns, validates pattern reliance,
and generates all figures and tables.

All output is logged to a timestamped file in the output directory.

Usage:
  python scripts/run_atop.py --run_dir runs/exp01 --mimic_dir /data/mimiciv/hosp
  python scripts/run_atop.py --run_dir runs/exp01 --mimic_dir /data/mimiciv/hosp \
      --out_dir explanations/ --mining_method episode
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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
    p = argparse.ArgumentParser(description="Run AToP explanation pipeline")
    p.add_argument("--run_dir", type=str, required=True,
                   help="Path to trained model bundle (from train_model.py)")
    p.add_argument("--mimic_dir", type=str, required=True,
                   help="Path to MIMIC-IV hosp/ directory")
    p.add_argument("--out_dir", type=str, default="",
                   help="Output directory (default: run_dir/explain)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--mining_method", type=str, default="",
                   help="Override mining method (episode/prefixspan/ngram)")
    p.add_argument("--ig_batch_size", type=int, default=0,
                   help="Override IG batch size (0 = use config default)")
    p.add_argument("--ig_max_train_samples", type=int, default=-1,
                   help="Override max train samples for IG (-1 = use config, 0 = all)")
    p.add_argument("--episode_topn", type=int, default=-1,
                   help="Override max patterns per group (-1 = use config, 0 = no limit)")
    p.add_argument("--episode_min_steps", type=int, default=-1,
                   help="Minimum number of visit blocks in a pattern "
                        "(1=all, 2=cross-visit, 4=full chain). -1 = use config.")
    p.add_argument("--episode_max_len", type=int, default=-1,
                   help="Maximum pattern length in visit blocks (default: 4). -1 = use config.")
    p.add_argument("--cap_by_or_per_length", type=int, default=-1,
                   help="After OR scoring, keep top N per pattern length by |log OR| "
                        "(N/2 risk + N/2 protective). 0 = disabled. -1 = use config.")
    p.add_argument("--cap_metric", type=str, default="",
                   help="Metric for per-length capping: 'or' (|log OR|) or "
                        "'prev_diff' (|prevalence difference|). Default: or")
    p.add_argument("--min_support_frac", type=float, default=-1,
                   help="Override min support fraction (-1 = use config, e.g. 0.01)")
    p.add_argument("--scoring_min_support_frac", type=float, default=-99,
                   help="Scoring-level min support fraction. -1 = same as mining frac (default behavior). "
                        "0 = no scoring filter. >0 = fraction of full cohort. -99 = use config.")
    p.add_argument("--jaccard_dedup", type=float, default=-1,
                   help="Jaccard threshold for pre-scoring pattern deduplication. "
                        "0 = disabled. -1 = use config.")
    p.add_argument("--jaccard_rep", type=str, default="",
                   help="Representative selection strategy for Jaccard clusters: "
                        "support, n_tokens, n_steps (default: support)")
    p.add_argument("--figures", type=str, default="",
                   help="Comma-separated list of figures to generate (default: all). "
                        "Valid: fig2,fig3,fig4,fig5,supp_heatmap,supp_decomposition,"
                        "supp_ood,tables,data")
    p.add_argument("--n_show", type=int, default=-1,
                   help="Number of items per panel in figures (default: 15)")
    p.add_argument("--n_exemplars", type=int, default=3,
                   help="Number of diverse patient exemplars for fig2 (default: 3)")
    p.add_argument("--validate_top_k", type=int, default=-1,
                   help="Number of risk + protective patterns to validate in fig4 (default: 15 each)")
    p.add_argument("--attention_n_patterns", type=int, default=-1,
                   help="Number of patterns to test for attention flow (default: 100)")
    p.add_argument("--attention_max_patients", type=int, default=-1,
                   help="Max patients sampled per pattern for attention flow (default: 200)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.run_dir, "explain")
    os.makedirs(out_dir, exist_ok=True)

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"run_{timestamp}.log")
    logger = TeeLogger(log_path)
    logger.start()

    t0 = time.time()

    try:
        from atop.explainer import AToPExplainer

        explainer = AToPExplainer.from_bundle(
            args.run_dir, mimic_dir=args.mimic_dir, device=args.device)

        # Apply overrides if given
        if args.ig_batch_size > 0:
            explainer.config.ig_batch_size = args.ig_batch_size
        if args.ig_max_train_samples >= 0:
            explainer.config.ig_max_train_samples = args.ig_max_train_samples
        if args.episode_topn >= 0:
            explainer.config.episode_topn = args.episode_topn
        if args.episode_min_steps >= 0:
            explainer.config.episode_min_steps = args.episode_min_steps
        if args.episode_max_len >= 0:
            explainer.config.episode_max_len = args.episode_max_len
        if args.cap_by_or_per_length >= 0:
            explainer.config.cap_by_or_per_length = args.cap_by_or_per_length
        if args.cap_metric:
            explainer.config.cap_metric = args.cap_metric
        if args.min_support_frac >= 0:
            explainer.config.min_support_frac = args.min_support_frac
        if args.scoring_min_support_frac > -99:
            explainer.config.scoring_min_support_frac = args.scoring_min_support_frac
        if args.jaccard_dedup >= 0:
            explainer.config.jaccard_dedup = args.jaccard_dedup
        if args.jaccard_rep:
            explainer.config.jaccard_rep = args.jaccard_rep
        if args.n_show >= 0:
            explainer.config.n_show = args.n_show
        if args.validate_top_k >= 0:
            explainer.config.validate_top_k = args.validate_top_k
        if args.attention_n_patterns >= 0:
            explainer.config.attention_n_patterns = args.attention_n_patterns
        if args.attention_max_patients >= 0:
            explainer.config.attention_max_patients = args.attention_max_patients

        print(f"[config] run_dir={args.run_dir}")
        print(f"[config] out_dir={out_dir}")
        print(f"[config] mining_method={args.mining_method or 'default'}")
        print(f"[config] device={args.device}")
        print()

        explainer.compute_attributions()
        explainer.mine_patterns(method=args.mining_method or None)

        fig_list = [f.strip() for f in args.figures.split(",") if f.strip()] or None
        explainer.report(out_dir=out_dir, figures=fig_list, n_exemplars=args.n_exemplars)

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed/60:.1f} min. Results in: {out_dir}")
        print(f"Log saved to: {log_path}")

    except Exception as e:
        import traceback
        print(f"\n[FATAL] {e}")
        traceback.print_exc()

    finally:
        logger.stop()


if __name__ == "__main__":
    main()
