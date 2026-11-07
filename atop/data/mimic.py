"""MIMIC-IV data loading and label construction."""
from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


def load_mimic_tables(mimic_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four core hosp tables from MIMIC-IV."""
    print("[data] Loading MIMIC-IV tables...")

    def _read(fname):
        for ext in [".csv.gz", ".csv"]:
            path = os.path.join(mimic_dir, fname + ext)
            if os.path.exists(path):
                return pd.read_csv(path, low_memory=False)
        raise FileNotFoundError(f"Cannot find {fname}.csv or {fname}.csv.gz in {mimic_dir}")

    admissions = _read("admissions")
    admissions["subject_id"] = admissions["subject_id"].astype(str)
    admissions["hadm_id"] = admissions["hadm_id"].astype(str)
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

    diagnoses = _read("diagnoses_icd")
    diagnoses["subject_id"] = diagnoses["subject_id"].astype(str)
    diagnoses["hadm_id"] = diagnoses["hadm_id"].astype(str)

    procedures = _read("procedures_icd")
    procedures["subject_id"] = procedures["subject_id"].astype(str)
    procedures["hadm_id"] = procedures["hadm_id"].astype(str)

    prescriptions = _read("prescriptions")
    prescriptions["subject_id"] = prescriptions["subject_id"].astype(str)
    prescriptions["hadm_id"] = prescriptions["hadm_id"].astype(str)
    if "starttime" in prescriptions.columns:
        prescriptions["starttime"] = pd.to_datetime(prescriptions["starttime"], errors="coerce")

    print(f"  admissions: {len(admissions):,}")
    print(f"  diagnoses:  {len(diagnoses):,}")
    print(f"  procedures: {len(procedures):,}")
    print(f"  prescriptions: {len(prescriptions):,}")
    return admissions, diagnoses, procedures, prescriptions



def build_readmission_labels(admissions: pd.DataFrame,
                             exclude_elective_readmissions: bool = False) -> pd.DataFrame:
    """
    For each admission, compute whether the patient was readmitted within 30 days.
    Returns a DataFrame with subject_id, hadm_id, admittime, dischtime, readmit_30d.
    Excludes in-hospital deaths (no chance to be readmitted).

    If exclude_elective_readmissions=True, only unplanned readmissions count as
    positive labels. Elective/scheduled readmissions (e.g. planned chemo, staged
    surgery) are not counted. Uses the admission_type column of the *next* admission.
    """
    adm = admissions.copy()
    # Exclude in-hospital deaths
    if "hospital_expire_flag" in adm.columns:
        adm = adm[adm["hospital_expire_flag"] == 0].copy()

    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    # For each admission, check if next admission by same patient is within 30 days
    adm["next_admittime"] = adm.groupby("subject_id")["admittime"].shift(-1)
    adm["days_to_next"] = (adm["next_admittime"] - adm["dischtime"]).dt.total_seconds() / 86400.0

    within_30 = (adm["days_to_next"] >= 0) & (adm["days_to_next"] <= 30)

    if exclude_elective_readmissions and "admission_type" in adm.columns:
        # Check the admission_type of the NEXT admission
        adm["next_admission_type"] = adm.groupby("subject_id")["admission_type"].shift(-1)
        # MIMIC-IV admission_type values include:
        #   ELECTIVE, SURGICAL SAME DAY ADMISSION, EU OBSERVATION,
        #   EW EMER., URGENT, OBSERVATION ADMIT, DIRECT EMER., AMBULANCE, etc.
        elective_types = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}
        next_is_elective = adm["next_admission_type"].str.upper().isin(elective_types)
        adm["readmit_30d"] = (within_30 & ~next_is_elective).astype(int)
        n_elective_excluded = int((within_30 & next_is_elective).sum())
        print(f"[labels] Excluded {n_elective_excluded:,} elective readmissions from positive labels")
    else:
        adm["readmit_30d"] = within_30.astype(int)

    # Last admission per patient: label = 0 (no readmission observed)
    adm["readmit_30d"] = adm["readmit_30d"].fillna(0).astype(int)

    return adm[["subject_id", "hadm_id", "admittime", "dischtime", "readmit_30d"]].copy()

