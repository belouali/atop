"""Configuration, constants, and utility functions."""
from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import torch

# ── Global constants ─────────────────────────────────────────────────────
PAD_IDX = 0
CLS_IDX = 1
VISIT_IDX = 2
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[VISIT]"]  # ordered: PAD=0, CLS=1, VISIT=2

# Default non-informative drug substrings to exclude
DEFAULT_DRUG_EXCLUDE_SUBSTRINGS = [
    # IV fluids / delivery vehicles
    "Flush", "Bag", "Vial", "Syringe", "Sterile Water",
    "D5W", "D5 ", " NS", "LR ",
    "Sodium Chloride 0.9%", "0.9% Sodium Chloride",
    "Iso-Osmotic Sodium", "Mini Bag",
    # Supplements / vitamins
    "Multivitamins", "Vitamin D", "Calcium Carbonate", "Calcium Gluconate",
    "Neutra-Phos", "Potassium Phosphate", "Thiamine",
    # Routine ward meds
    "Chlorhexidine", "Nystatin Oral", "Simethicone",
    "Aluminum-Magnesium", "Milk of Magnesia",
    "Lidocaine Jelly", "DiphenhydrAMINE",
    # Vaccines
    "Influenza Vaccine", "PNEUMOcoccal", "Pneumococcal",
    # Sliding-scale insulin
    "Insulin Human Regular",
]

# RxCUI-based exclude list: non-informative drugs that survive RxNorm mapping.
# These are given to nearly all hospitalized patients regardless of condition.
# Format: D:RX_{rxcui} — matched exactly against RxNorm-mapped drug tokens.
DEFAULT_DRUG_EXCLUDE_RXCUIS = [
    # IV fluids / delivery vehicles
    "D:RX_9853",   # sodium chloride (saline) — 90.7%
    "D:RX_82003",  # dextrose (D5W / glucose) — 56.1%
    "D:RX_9863",   # potassium chloride (KCl) — 60.6%
    # Routine ward meds
    "D:RX_161",    # acetaminophen (Tylenol) — 81.2%
    "D:RX_5224",   # heparin (DVT prophylaxis) — 66.4%
    "D:RX_4850",   # ondansetron (Zofran, anti-nausea) — 53.9%
    "D:RX_114202", # docusate (stool softener) — 33.2%
    "D:RX_8591",   # propofol (anesthesia) — 35.9%
    "D:RX_26225",  # magnesium (supplement) — 41.4%
    # Supplements / vitamins (RxNorm equivalents)
    "D:RX_11124",  # calcium (various forms) — 19.0%
]

# Default ICD prefixes to exclude (non-informative administrative / screening codes)
DEFAULT_ICD_EXCLUDE_PREFIXES = [
    # COVID screening
    "Z20",     # Contact with and (suspected) exposure to communicable diseases
    # General screening
    "Z11", "Z12", "Z13",  # Screening for various conditions
    # Vaccination
    "Z23",     # Encounter for immunization
    # Administrative encounters
    "Z00",     # Encounter for general examination
    "Z02",     # Encounter for administrative examination
    "Z76",     # Persons encountering health services in other circumstances
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: torch.use_deterministic_algorithms(True) can cause errors
    # with some ops (scatter, index_put). We set it to warn-only mode.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass  # older PyTorch versions don't support warn_only


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


@dataclass
class AToPConfig:
    """All configuration for an AToP run — saved as config.json in the model bundle."""
    # Data
    mimic_dir: str = ""
    token_types: str = "CPD"
    max_visits: int = 20
    max_seq_len: int = 512
    one_per_patient: bool = True
    max_drug_freq: float = 0.5
    drug_exclude_substrings: List[str] = field(default_factory=list)
    chronic_filter: bool = False
    first_occurrence_only: bool = False  # keep each code only in earliest admission
    first_occurrence_drugs_only: bool = False  # first-occurrence for drugs only (C/P unchanged)
    exclude_elective_readmissions: bool = False
    harmonize_icd: bool = False
    icd_exclude_prefixes: List[str] = field(default_factory=list)
    drug_mapping: str = ""  # Path to NDC→RxNorm ingredient CSV (from build_ndc_rxnorm_map.py)
    max_patients: int = 0

    # Model
    embedding_dim: int = 128
    heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-5
    seed: int = 42

    # IG / Saliency
    ig_n_steps: int = 30
    ig_batch_size: int = 4
    ig_mass: float = 0.8
    ig_max_tokens: int = 50
    ig_max_train_samples: int = 0

    # Mining
    mining_method: str = "episode"
    min_support_frac: float = 0.02
    mine_on_trainval: bool = False
    ngram_min_len: int = 2
    ngram_max_len: int = 4
    prefixspan_min_len: int = 2
    prefixspan_topn: int = 500
    episode_max_len: int = 4
    episode_topn: int = 0
    episode_min_steps: int = 1  # minimum number of visit blocks in a pattern (1=all, 2=cross-visit only)
    cap_by_or_per_length: int = 0  # 0 = disabled; >0 = keep top N per length by |log OR|
    cap_metric: str = "or"  # "or" = |log OR|, "prev_diff" = |prevalence difference|
    jaccard_dedup: float = 0.0  # 0 = disabled; >0 = Jaccard threshold for pre-scoring deduplication
    jaccard_rep: str = "support"  # representative selection: support, n_tokens, n_steps

    # SHAP
    shap_max_features: int = 15
    shap_nsamples: int = 200

    # Validation
    validate_top_k: int = 15
    validate_max_admissions_per_pattern: int = 200
    n_shuffle_draws: int = 50

    # Attention flow
    attention_n_patterns: int = 100  # number of patterns to test (top N/2 risk + N/2 protective)
    attention_max_patients: int = 200  # max patients sampled per pattern

    # Display
    n_show: int = 15  # number of items per panel in figures

    # Paths
    out_dir: str = "atop_results"
    lace_csv: str = ""
    device: str = "auto"
    use_icd_titles: bool = True

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AToPConfig":
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def from_args(cls, args) -> "AToPConfig":
        """Create from argparse Namespace."""
        d = {k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        return cls(**d)
