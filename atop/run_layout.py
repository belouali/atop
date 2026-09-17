"""Canonical run-directory layout (April 2026 v2).

Every run directory follows the same numbered-stage structure so the pipeline
order is self-documenting and so deletion / archival decisions can be made by
folder rather than by file:

    runs/<run_name>/
    ├── 0_inputs/        config.json, splits.json, vocab.pkl, tokenization.json,
    │                    icd_titles.json
    ├── 1_model/         model.pt, train_log.csv
    ├── 2_predictions/   test_predictions.csv, test_metrics.csv, metrics.json,
    │                    summary.json, dataset_summary.csv
    ├── 3_attribution/   ig_<split>.csv, sequences_<split>.pkl, tensors_<split>.pt
    │                    (and SHAP outputs when produced)
    ├── 4_mining/        all_scored_<hash>.csv, raw_episode_*.pkl
    ├── 5_analysis/      exemplars.json, pathway_importance_<split>.csv,
    │                    token_importance_<split>.csv, carrier_sets_<split>.pkl,
    │                    jaccard_reps_<split>_j<pct>.pkl, validation_<split>.csv,
    │                    reversed_order_<split>.csv
    ├── 6_baselines/     lace_scores_all.csv, lace_baselines.csv,
    │                    auroc_bootstrap.{csv,md}
    └── 7_outputs/       figures/, panels/, tables/, diagnostics/

`make_*.py` scripts construct paths via :class:`RunLayout` so a future layout
change touches only this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# ── Stage folder names ──────────────────────────────────────────────
INPUTS = "0_inputs"
MODEL = "1_model"
PREDICTIONS = "2_predictions"
ATTRIBUTION = "3_attribution"
MINING = "4_mining"
ANALYSIS = "5_analysis"
BASELINES = "6_baselines"
OUTPUTS = "7_outputs"

ALL_STAGE_DIRS = [INPUTS, MODEL, PREDICTIONS, ATTRIBUTION, MINING, ANALYSIS,
                  BASELINES, OUTPUTS]


@dataclass(frozen=True)
class RunLayout:
    """Resolves canonical paths under ``run_dir``.

    All path properties / methods are pure, no I/O. Use :meth:`mkdirs` to
    create the stage directories and the ``7_outputs`` sub-folders.
    """

    run_dir: str

    # ── Stage directories ───────────────────────────────────────────
    @property
    def inputs_dir(self) -> str:        return os.path.join(self.run_dir, INPUTS)
    @property
    def model_dir(self) -> str:         return os.path.join(self.run_dir, MODEL)
    @property
    def predictions_dir(self) -> str:   return os.path.join(self.run_dir, PREDICTIONS)
    @property
    def attribution_dir(self) -> str:   return os.path.join(self.run_dir, ATTRIBUTION)
    @property
    def mining_dir(self) -> str:        return os.path.join(self.run_dir, MINING)
    @property
    def analysis_dir(self) -> str:      return os.path.join(self.run_dir, ANALYSIS)
    @property
    def baselines_dir(self) -> str:     return os.path.join(self.run_dir, BASELINES)
    @property
    def outputs_dir(self) -> str:       return os.path.join(self.run_dir, OUTPUTS)

    # ── 7_outputs sub-folders ───────────────────────────────────────
    @property
    def outputs_figures_dir(self) -> str:
        return os.path.join(self.outputs_dir, "figures")
    @property
    def outputs_panels_dir(self) -> str:
        return os.path.join(self.outputs_dir, "panels")
    @property
    def outputs_tables_dir(self) -> str:
        return os.path.join(self.outputs_dir, "tables")
    @property
    def outputs_diagnostics_dir(self) -> str:
        return os.path.join(self.outputs_dir, "diagnostics")

    # ── Single-file convenience properties ──────────────────────────
    @property
    def config_json(self) -> str:           return os.path.join(self.inputs_dir, "config.json")
    @property
    def splits_json(self) -> str:           return os.path.join(self.inputs_dir, "splits.json")
    @property
    def vocab_pkl(self) -> str:             return os.path.join(self.inputs_dir, "vocab.pkl")
    @property
    def tokenization_json(self) -> str:     return os.path.join(self.inputs_dir, "tokenization.json")
    @property
    def icd_titles_json(self) -> str:       return os.path.join(self.inputs_dir, "icd_titles.json")

    @property
    def model_pt(self) -> str:              return os.path.join(self.model_dir, "model.pt")
    @property
    def train_log_csv(self) -> str:         return os.path.join(self.model_dir, "train_log.csv")

    @property
    def test_predictions_csv(self) -> str:  return os.path.join(self.predictions_dir, "test_predictions.csv")
    @property
    def test_metrics_csv(self) -> str:      return os.path.join(self.predictions_dir, "test_metrics.csv")
    @property
    def metrics_json(self) -> str:          return os.path.join(self.predictions_dir, "metrics.json")
    @property
    def summary_json(self) -> str:          return os.path.join(self.predictions_dir, "summary.json")
    @property
    def dataset_summary_csv(self) -> str:   return os.path.join(self.predictions_dir, "dataset_summary.csv")

    @property
    def lace_scores_csv(self) -> str:       return os.path.join(self.baselines_dir, "lace_scores_all.csv")

    @property
    def exemplars_json(self) -> str:        return os.path.join(self.analysis_dir, "exemplars.json")

    # ── Split-aware patterned helpers ───────────────────────────────
    def ig_csv(self, split: str) -> str:
        return os.path.join(self.attribution_dir, f"ig_{split}.csv")

    def sequences_pkl(self, split: str) -> str:
        return os.path.join(self.attribution_dir, f"sequences_{split}.pkl")

    def tensors_pt(self, split: str) -> str:
        return os.path.join(self.attribution_dir, f"tensors_{split}.pt")

    @property
    def shap_cache_pkl(self) -> str:
        """Per-patient KernelSHAP cache (populated by ``make_fig_s3``)."""
        return os.path.join(self.attribution_dir, "shap_cache.pkl")

    def pathway_csv(self, split: str, suffix: str = "") -> str:
        """``pathway_importance_<split><suffix>.csv``  (suffix e.g. ``_all``, ``_xvisit``)."""
        return os.path.join(self.analysis_dir, f"pathway_importance_{split}{suffix}.csv")

    def token_csv(self, split: str) -> str:
        return os.path.join(self.analysis_dir, f"token_importance_{split}.csv")

    def carrier_sets_pkl(self, split: str, suffix: str = "") -> str:
        return os.path.join(self.analysis_dir, f"carrier_sets_{split}{suffix}.pkl")

    def jaccard_reps_pkl(self, split: str, jaccard_threshold: float) -> str:
        return os.path.join(self.analysis_dir,
                            f"jaccard_reps_{split}_j{int(jaccard_threshold * 100)}.pkl")

    def validation_csv(self, split: str) -> str:
        return os.path.join(self.analysis_dir, f"validation_{split}.csv")

    def reversed_order_csv(self, split: str) -> str:
        return os.path.join(self.analysis_dir, f"reversed_order_{split}.csv")

    def panel_csv(self, panel: str, split: str, jaccard_threshold: float,
                  short: bool = True) -> str:
        sfx = "_short" if short else ""
        return os.path.join(self.outputs_panels_dir,
                            f"panel_{panel}_{split}_j{int(jaccard_threshold * 100)}{sfx}.csv")

    def table_csv(self, name: str) -> str:
        return os.path.join(self.outputs_tables_dir, f"{name}.csv")

    def diagnostic_png(self, name: str) -> str:
        return os.path.join(self.outputs_diagnostics_dir, f"{name}.png")

    # ── House-keeping ───────────────────────────────────────────────
    def mkdirs(self) -> None:
        """Create every stage directory + ``7_outputs`` sub-folders."""
        for d in (self.inputs_dir, self.model_dir, self.predictions_dir,
                  self.attribution_dir, self.mining_dir, self.analysis_dir,
                  self.baselines_dir, self.outputs_dir,
                  self.outputs_figures_dir, self.outputs_panels_dir,
                  self.outputs_tables_dir, self.outputs_diagnostics_dir):
            os.makedirs(d, exist_ok=True)


# ── Runtime / cache diagnostic helper for notebooks ───────────────────
def _gpu_info() -> str:
    """Return ``"<name>, <vram> MiB"`` or ``"no GPU"``."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or "no GPU"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "no GPU"


def _cpu_ram_gb() -> float:
    """Cross-platform total RAM in GB."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, AttributeError):
        try:
            import psutil
            return psutil.virtual_memory().total / 1e9
        except ImportError:
            return 0.0


def runtime_check(
    layout: "RunLayout",
    *,
    needs: dict,
    min_ram_gb: float = 8.0,
) -> None:
    """Print GPU + RAM + cache status for the current notebook stage.

    Args
    ----
    layout: ``RunLayout`` for the current run dir.
    needs:  dict mapping ``"<short label>"`` → ``(<absolute path>, <"gpu"|"cpu"|"both">)``
            describing each artifact this notebook produces and whether
            (re-)producing it requires GPU. Used to decide whether the
            current runtime is sufficient.
    min_ram_gb: warn if available RAM is below this.
    """
    gpu = _gpu_info()
    ram = _cpu_ram_gb()
    has_gpu = gpu != "no GPU"

    print(f"Runtime:  {gpu}")
    print(f"CPU RAM:  {ram:.1f} GB")
    print(f"Run dir:  {layout.run_dir}\n")

    print("Cache status (\u2713 = present, will skip):")
    needs_gpu_for = []
    for name, (path, kind) in needs.items():
        present = os.path.exists(path)
        mark = "\u2713" if present else "\u2717"
        print(f"  {mark}  {name:<30s} ({kind})")
        if not present and kind in ("gpu", "both"):
            needs_gpu_for.append(name)

    print()
    if needs_gpu_for and not has_gpu:
        print(f"\u26A0  GPU runtime needed to (re-)compute: {', '.join(needs_gpu_for)}")
        print("   \u2192 Switch to a Colab GPU runtime before continuing.")
    elif needs_gpu_for:
        print(f"\u2713  GPU available; will compute: {', '.join(needs_gpu_for)}")
    else:
        print("\u2713  Everything cached \u2014 no GPU needed for this notebook.")
        if has_gpu:
            print("   (You can downgrade to a CPU runtime to save Compute Units.)")

    if ram < min_ram_gb:
        print(f"\n\u26A0  Available RAM ({ram:.1f} GB) < recommended {min_ram_gb} GB. "
              f"Consider High-RAM tier.")
