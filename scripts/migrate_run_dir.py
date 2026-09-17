"""migrate_run_dir, reorganize a legacy run directory into the v2 layout.

Maps every legacy file to its new home under :class:`atop.run_layout.RunLayout`
(numbered pipeline stages 0_inputs through 7_outputs) and renames the few
files where the new directory makes the old prefix redundant
(``fig3_pathway_importance_*.csv`` → ``pathway_importance_*.csv`` etc.).

Operates **in place** by default, uses ``os.rename`` on the same filesystem,
so the move is atomic and no extra disk space is needed.

Usage:
    python scripts/migrate_run_dir.py --run_dir ../runs/full_CPD_v0
    python scripts/migrate_run_dir.py --run_dir ../runs/full_CPD_v1 --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.run_layout import RunLayout


def _is_already_migrated(run_dir: str) -> bool:
    """Heuristic: if 0_inputs/ and 5_analysis/ exist, treat as migrated."""
    return (os.path.isdir(os.path.join(run_dir, "0_inputs")) and
            os.path.isdir(os.path.join(run_dir, "5_analysis")))


def _safe_move(src: str, dst: str, *, dry_run: bool, log: list) -> None:
    if not os.path.exists(src):
        return
    if os.path.exists(dst):
        log.append(f"  [skip] {dst} exists; leaving {src} in place")
        return
    log.append(f"  {os.path.relpath(src)} → {os.path.relpath(dst)}")
    if dry_run:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.rename(src, dst)


def _move_glob(pattern: str, dst_dir: str, *, rename_fn=None, dry_run: bool, log: list) -> None:
    """Move every file matching ``pattern`` into ``dst_dir``; optionally rename
    via ``rename_fn(basename) -> new_basename``."""
    for src in glob.glob(pattern):
        if os.path.isdir(src):
            continue
        base = os.path.basename(src)
        new_base = rename_fn(base) if rename_fn else base
        dst = os.path.join(dst_dir, new_base)
        _safe_move(src, dst, dry_run=dry_run, log=log)


def _strip_fig_prefix(name: str) -> str:
    """`fig3_pathway_importance_test.csv` → `pathway_importance_test.csv`,
    `fig5_reversed_order_test.csv` → `reversed_order_test.csv`. Everything
    else returned unchanged."""
    for prefix in ("fig3_", "fig5_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def migrate(run_dir: str, *, dry_run: bool = False) -> list:
    """Run the migration. Returns the move log."""
    if _is_already_migrated(run_dir):
        return [f"  Run dir already in v2 layout: {run_dir}"]

    layout = RunLayout(run_dir)
    if not dry_run:
        layout.mkdirs()

    log: list = []

    # ── 0_inputs ────────────────────────────────────────────────────
    for f in ("config.json", "splits.json", "vocab.pkl", "tokenization.json",
              "icd_titles.json"):
        _safe_move(os.path.join(run_dir, f),
                   os.path.join(layout.inputs_dir, f),
                   dry_run=dry_run, log=log)

    # ── 1_model ─────────────────────────────────────────────────────
    for f in ("model.pt", "train_log.csv"):
        _safe_move(os.path.join(run_dir, f),
                   os.path.join(layout.model_dir, f),
                   dry_run=dry_run, log=log)

    # ── 2_predictions ───────────────────────────────────────────────
    for f in ("test_predictions.csv", "test_metrics.csv", "metrics.json",
              "summary.json", "dataset_summary.csv"):
        _safe_move(os.path.join(run_dir, f),
                   os.path.join(layout.predictions_dir, f),
                   dry_run=dry_run, log=log)

    # ── 3_attribution (flatten ig_cache/) ───────────────────────────
    _move_glob(os.path.join(run_dir, "ig_cache", "*"),
               layout.attribution_dir, dry_run=dry_run, log=log)
    if os.path.isdir(os.path.join(run_dir, "ig_cache")) and not dry_run:
        try:
            os.rmdir(os.path.join(run_dir, "ig_cache"))
        except OSError:
            pass

    # ── 4_mining (flatten mining_cache/) ────────────────────────────
    _move_glob(os.path.join(run_dir, "mining_cache", "*"),
               layout.mining_dir, dry_run=dry_run, log=log)
    if os.path.isdir(os.path.join(run_dir, "mining_cache")) and not dry_run:
        try:
            os.rmdir(os.path.join(run_dir, "mining_cache"))
        except OSError:
            pass

    # ── 5_analysis (from explain/figures/csv/ + explain/exemplars.json) ─
    explain_dir = os.path.join(run_dir, "explain")
    explain_csv_dir = os.path.join(explain_dir, "figures", "csv")

    _safe_move(os.path.join(explain_dir, "exemplars.json"),
               layout.exemplars_json, dry_run=dry_run, log=log)

    if os.path.isdir(explain_csv_dir):
        # Move analysis-data CSVs/PKLs with the fig3_/fig5_ prefix stripped
        for src in sorted(glob.glob(os.path.join(explain_csv_dir, "*"))):
            if os.path.isdir(src):
                continue
            base = os.path.basename(src)
            # Skip backup files
            if "_backup" in base:
                continue
            # Skip Fig S2/S3/S5/S6 source CSVs from the legacy supp_fig_s*.py
            #, these were stale anyway; the new make_fig_s*.py write into the
            # new locations themselves on next run.
            if base.startswith(("supp_fig_s", "fig_s")):
                continue
            new_base = _strip_fig_prefix(base)
            _safe_move(src, os.path.join(layout.analysis_dir, new_base),
                       dry_run=dry_run, log=log)

    # ── 6_baselines ─────────────────────────────────────────────────
    _safe_move(os.path.join(run_dir, "lace_scores_all.csv"),
               layout.lace_scores_csv, dry_run=dry_run, log=log)
    _move_glob(os.path.join(run_dir, "baselines", "*"),
               layout.baselines_dir, dry_run=dry_run, log=log)
    if os.path.isdir(os.path.join(run_dir, "baselines")) and not dry_run:
        try:
            os.rmdir(os.path.join(run_dir, "baselines"))
        except OSError:
            pass

    # ── 7_outputs/diagnostics (root-level diagnostic PNGs) ──────────
    for f in ("fig1_dataset_performance.png", "fig_train_curves.png"):
        _safe_move(os.path.join(run_dir, f),
                   os.path.join(layout.outputs_diagnostics_dir, f),
                   dry_run=dry_run, log=log)

    # ── 7_outputs/panels (panel_a/b/c + name_mappings) ──────────────
    explain_main_dir = os.path.join(explain_dir, "figures", "main")
    if os.path.isdir(explain_main_dir):
        for src in sorted(glob.glob(os.path.join(explain_main_dir, "panel_*.csv"))):
            base = os.path.basename(src)
            _safe_move(src, os.path.join(layout.outputs_panels_dir, base),
                       dry_run=dry_run, log=log)
        for src in sorted(glob.glob(os.path.join(explain_main_dir, "name_mappings_*.csv"))):
            base = os.path.basename(src)
            _safe_move(src, os.path.join(layout.outputs_panels_dir, base),
                       dry_run=dry_run, log=log)

        # ── 7_outputs/figures (run-dir copies of fig PNGs) ──────────
        for src in sorted(glob.glob(os.path.join(explain_main_dir, "*.png"))):
            base = os.path.basename(src)
            _safe_move(src, os.path.join(layout.outputs_figures_dir, base),
                       dry_run=dry_run, log=log)
        # Also any fig4_dumbbell csv / sources
        for src in sorted(glob.glob(os.path.join(explain_main_dir, "fig4_dumbbell*.csv"))):
            base = os.path.basename(src)
            _safe_move(src, os.path.join(layout.outputs_figures_dir, base),
                       dry_run=dry_run, log=log)
        for src in sorted(glob.glob(os.path.join(explain_main_dir, "attention_flow*.csv"))):
            base = os.path.basename(src)
            _safe_move(src, os.path.join(layout.outputs_figures_dir, base),
                       dry_run=dry_run, log=log)

    # ── Try to clean up the now-empty explain/ tree ─────────────────
    if not dry_run:
        for sub in ("explain/figures/csv", "explain/figures/main",
                    "explain/figures/supplement", "explain/figures",
                    "explain/data", "explain"):
            d = os.path.join(run_dir, sub)
            if os.path.isdir(d):
                try:
                    if not os.listdir(d):
                        os.rmdir(d)
                        log.append(f"  [rmdir] {sub}/")
                except OSError:
                    pass

    return log


def main() -> None:
    p = argparse.ArgumentParser(description="Migrate a legacy run dir to the v2 layout")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would move; don't actually rename anything")
    args = p.parse_args()

    if not os.path.isdir(args.run_dir):
        sys.exit(f"  [ERROR] Not a directory: {args.run_dir}")

    print(f"Migrating: {args.run_dir} {'(DRY RUN)' if args.dry_run else ''}")
    print("-" * 70)
    log = migrate(args.run_dir, dry_run=args.dry_run)
    for line in log:
        print(line)
    print("-" * 70)
    print(f"  {len(log)} moves" + ("  (dry run \u2014 no files changed)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
