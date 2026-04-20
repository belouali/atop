#!/usr/bin/env python3
"""
Preprocess MIMIC-IV data and cache the result.

Run this on a CPU runtime (free) to build and cache the preprocessed samples.
Then run train_model.py on a GPU runtime — it will load from cache in seconds.

Usage:
    python scripts/preprocess.py \
        --mimic_dir /path/to/mimiciv/hosp/ \
        --token_types CPD \
        --harmonize_icd \
        --icd_exclude_prefixes \
        --max_drug_freq 0.25 \
        --drug_exclude_substrings \
        --drug_mapping /path/to/ndc_to_ingredient.csv \
        --exclude_elective_readmissions \
        --first_occurrence_drugs_only \
        --chronic_filter
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.config import (
    DEFAULT_DRUG_EXCLUDE_SUBSTRINGS,
    DEFAULT_ICD_EXCLUDE_PREFIXES,
)
from atop.data.mimic import load_mimic_tables, build_readmission_labels
from atop.data.tokenization import build_patient_sequences


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess MIMIC-IV data and cache samples")
    p.add_argument("--mimic_dir", type=str, required=True,
                   help="Path to MIMIC-IV hosp/ directory")

    # Tokenization parameters
    p.add_argument("--token_types", type=str, default="CPD",
                   help="Which token streams to include: C=conditions, P=procedures, D=drugs")
    p.add_argument("--max_visits", type=int, default=20)
    p.add_argument("--one_per_patient", action="store_true", default=False)
    p.add_argument("--harmonize_icd", action="store_true", default=False,
                   help="Map ICD-9 codes to ICD-10 using CMS GEMs")
    p.add_argument("--icd_exclude_prefixes", nargs="*", default=None,
                   help="ICD prefixes to exclude (pass flag with no args for defaults)")

    # Drug filtering
    p.add_argument("--max_drug_freq", type=float, default=0.0,
                   help="Exclude drugs appearing in more than this fraction of admissions")
    p.add_argument("--drug_exclude_substrings", nargs="*", default=None,
                   help="Drug name substrings to exclude (pass flag with no args for defaults)")
    p.add_argument("--drug_mapping", type=str, default="",
                   help="Path to NDC-to-ingredient CSV from build_ndc_rxnorm_map.py")

    # Condition/drug occurrence filters
    p.add_argument("--chronic_filter", action="store_true", default=False,
                   help="Keep chronic dx only at first occurrence or when seq_num=1")
    p.add_argument("--first_occurrence_only", action="store_true", default=False,
                   help="Keep each code (C/P/D) only in earliest admission")
    p.add_argument("--first_occurrence_drugs_only", action="store_true", default=False,
                   help="Apply first-occurrence filtering only to drug tokens")

    # Label definition
    p.add_argument("--exclude_elective_readmissions", action="store_true", default=False,
                   help="Exclude elective/planned readmissions from positive labels")

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve defaults
    if args.drug_exclude_substrings is not None and len(args.drug_exclude_substrings) == 0:
        args.drug_exclude_substrings = DEFAULT_DRUG_EXCLUDE_SUBSTRINGS
        print(f"[drug filter] Using default exclusion list ({len(DEFAULT_DRUG_EXCLUDE_SUBSTRINGS)} substrings)")

    if args.icd_exclude_prefixes is not None and len(args.icd_exclude_prefixes) == 0:
        args.icd_exclude_prefixes = DEFAULT_ICD_EXCLUDE_PREFIXES
        print(f"[icd filter] Using default exclusion list ({len(DEFAULT_ICD_EXCLUDE_PREFIXES)} prefixes)")

    # ── Load tables ──────────────────────────────────────────────────────
    admissions, diagnoses, procedures, prescriptions = load_mimic_tables(args.mimic_dir)
    adm_labels = build_readmission_labels(
        admissions, exclude_elective_readmissions=args.exclude_elective_readmissions)
    print(f"[data] Admissions with labels: {len(adm_labels):,}")

    # ── Build + cache ────────────────────────────────────────────────────
    cache_dir = os.path.join(args.mimic_dir, ".atop_cache")
    samples = build_patient_sequences(
        adm_labels, diagnoses, procedures, prescriptions,
        max_visits=args.max_visits,
        one_per_patient=args.one_per_patient,
        max_drug_freq=args.max_drug_freq,
        drug_exclude_substrings=args.drug_exclude_substrings or None,
        token_types=args.token_types,
        chronic_filter=args.chronic_filter,
        first_occurrence_only=args.first_occurrence_only,
        first_occurrence_drugs_only=args.first_occurrence_drugs_only,
        harmonize_icd=args.harmonize_icd,
        icd_exclude_prefixes=args.icd_exclude_prefixes or None,
        drug_mapping=args.drug_mapping,
        cache_dir=cache_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    n_patients = len(set(s["patient_id"] for s in samples))
    n_multi = sum(1 for s in samples if s["n_visits"] > 1)
    n_readmit = sum(1 for s in samples if s["readmit_30d"] == 1)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete")
    print(f"  Samples:     {len(samples):,}")
    print(f"  Patients:    {n_patients:,}")
    print(f"  Multi-visit: {n_multi:,} ({100*n_multi/len(samples):.1f}%)")
    print(f"  Readmitted:  {n_readmit:,} ({100*n_readmit/len(samples):.1f}%)")
    print(f"  Cache dir:   {cache_dir}")
    print(f"{'='*60}")
    print(f"\nNow run train_model.py with the same data params on a GPU runtime.")
    print(f"It will load from cache in seconds.")


if __name__ == "__main__":
    main()
