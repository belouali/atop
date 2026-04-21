"""
Compute Table 1 (cohort characteristics) using the FULL 218,196-patient cohort
— train + val + test. Reproduces the model's index-admission rule:
  - ≥2 admissions: penultimate (nth n-2, 0-indexed)
  - 1 admission:   that single admission (nth 0, label=0)
matching atop/data/tokenization.py:753-754 and exclude_elective_readmissions=True
from runs/full_CPD/config.json.

Outputs demographic/LACE/visit-count stats to stdout so values can be dropped
straight into the manuscript Table 1.
"""
from __future__ import annotations
import os, sys, json, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.data.mimic import build_readmission_labels

FULL_RUN  = "../runs/full_CPD"
MIMIC_DIR = "../data/mimiciv/3.1/hosp"
MAX_VISITS = 20  # from config.json


# ── Load splits + config for validation ───────────────────────────────────
with open(f"{FULL_RUN}/splits.json") as f:
    splits = json.load(f)
all_pids = set(str(p) for p in (splits["train"] + splits["val"] + splits["test"]))
print(f"[splits]   train={len(splits['train']):,}  val={len(splits['val']):,}  "
      f"test={len(splits['test']):,}  total={len(all_pids):,}")

with open(f"{FULL_RUN}/summary.json") as f:
    summary = json.load(f)
print(f"[summary]  n_samples={summary['n_samples']:,}  "
      f"n_readmit={summary['n_readmit']:,}  "
      f"n_no_readmit={summary['n_no_readmit']:,}")


# ── Load MIMIC tables ─────────────────────────────────────────────────────
print("\n[1] Loading admissions + patients...")
admissions = pd.read_csv(
    f"{MIMIC_DIR}/admissions.csv.gz",
    dtype={"subject_id": str, "hadm_id": str},
    parse_dates=["admittime", "dischtime"],
)
patients = pd.read_csv(
    f"{MIMIC_DIR}/patients.csv.gz",
    dtype={"subject_id": str},
    usecols=["subject_id", "gender", "anchor_age"],
)
print(f"    admissions={len(admissions):,}  patients={len(patients):,}")


# ── Build readmission labels per-admission (exclude_elective_readmissions=True) ──
print("\n[2] Building readmission labels (exclude_elective_readmissions=True)...")
labels_df = build_readmission_labels(admissions, exclude_elective_readmissions=True)
print(f"    labels rows: {len(labels_df):,}")


# ── Apply the model's one_per_patient rule ─────────────────────────────────
print("\n[3] Selecting index admission per patient (penultimate if ≥2, else first)...")
# Keep only our splits' patients
labels_df = labels_df[labels_df["subject_id"].isin(all_pids)].copy()
labels_df = labels_df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
labels_df["pos_in_pt"] = labels_df.groupby("subject_id").cumcount()
labels_df["n_in_pt"]   = labels_df.groupby("subject_id")["admittime"].transform("count")

# index_position per the rule (idx_pos = n-2 if n>=2 else 0)
labels_df["idx_pos"] = np.where(labels_df["n_in_pt"] >= 2,
                                labels_df["n_in_pt"] - 2, 0)
index_df = labels_df[labels_df["pos_in_pt"] == labels_df["idx_pos"]].copy()
# n_visits = min(idx_pos + 1, MAX_VISITS)
index_df["n_visits"] = np.minimum(index_df["idx_pos"] + 1, MAX_VISITS)
print(f"    selected index admissions: {len(index_df):,}")
print(f"    readmit rate: {index_df['readmit_30d'].mean():.4f}")


# ── Merge demographics + admission type ────────────────────────────────────
print("\n[4] Merging demographics + admission type...")
cohort = index_df.merge(
    patients[["subject_id", "gender", "anchor_age"]],
    on="subject_id", how="left",
)
cohort = cohort.merge(
    admissions[["subject_id", "hadm_id", "admission_type"]],
    on=["subject_id", "hadm_id"], how="left",
)
cohort["age"]    = pd.to_numeric(cohort["anchor_age"], errors="coerce")
cohort["female"] = (cohort["gender"] == "F").astype(float)
# JAMIA Table 1 "Emergency/Urgent" — MIMIC admission_type values that are
# non-elective and non-observation count as emergency/urgent. Anything NOT
# {ELECTIVE, SURGICAL SAME DAY ADMISSION} is emergent.
elective_types = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}
cohort["is_emergency"] = (~cohort["admission_type"].str.upper().isin(elective_types)).astype(float)


# ── Merge LACE components ──────────────────────────────────────────────────
print("\n[5] Merging LACE components...")
lace = pd.read_csv(
    f"{FULL_RUN}/lace_scores_all.csv",
    dtype={"subject_id": str, "hadm_id": str},
)
cohort = cohort.merge(
    lace[["subject_id", "hadm_id", "los_days", "cci", "ed_visits_6mo",
          "A", "L", "C", "E", "lace_score"]],
    on=["subject_id", "hadm_id"], how="left",
)


# ── Report stats by readmission status ─────────────────────────────────────
def med_iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "n/a"
    return f"{s.median():.0f} ({s.quantile(0.25):.0f}–{s.quantile(0.75):.0f})"

def med_iqr_float(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "n/a"
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}–{s.quantile(0.75):.1f})"

def n_pct(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "n/a"
    return f"{int(s.sum()):,} ({s.mean()*100:.1f}%)"

print("\n" + "=" * 72)
print("TABLE 1 — COHORT CHARACTERISTICS (FULL 218,196)")
print("=" * 72)
n = len(cohort)
nr = int(cohort["readmit_30d"].sum())
nnr = n - nr
print(f"\n  Overall N:             {n:,}")
print(f"  Readmitted:            {nr:,}  ({nr/n*100:.1f}%)")
print(f"  Not readmitted:        {nnr:,}  ({nnr/n*100:.1f}%)")

r  = cohort[cohort["readmit_30d"] == 1]
nr_df = cohort[cohort["readmit_30d"] == 0]


def row(name, fn, col, use_float=False):
    label_format = med_iqr_float if use_float else med_iqr
    if fn == "med_iqr":
        formatter = label_format
    elif fn == "n_pct":
        formatter = n_pct
    else:
        raise ValueError(fn)
    print(f"\n  {name}")
    print(f"    Overall:         {formatter(cohort[col])}")
    print(f"    Readmitted:      {formatter(r[col])}")
    print(f"    Not readmitted:  {formatter(nr_df[col])}")


row("Age, median (IQR)",                         "med_iqr", "age")
row("Female sex, n (%)",                         "n_pct",   "female")
row("Emergency/Urgent, n (%)",                   "n_pct",   "is_emergency")
row("Length of stay (days), median (IQR)",       "med_iqr", "los_days", use_float=True)
row("Charlson Comorbidity Index, median (IQR)",  "med_iqr", "cci")
row("ED visits (prior 6 months), median (IQR)",  "med_iqr", "ed_visits_6mo")
row("Prior admissions (n_visits), median (IQR)", "med_iqr", "n_visits")
row("LACE score, median (IQR)",                  "med_iqr", "lace_score")

print("\n" + "=" * 72)
print("CROSS-CHECK: train + val + test counts should equal summary.json")
print("=" * 72)
print(f"  expected (summary.json):  n_samples={summary['n_samples']:,}  "
      f"n_readmit={summary['n_readmit']:,}  n_no_readmit={summary['n_no_readmit']:,}")
print(f"  computed here:            n={n:,}  n_readmit={nr:,}  n_no_readmit={nnr:,}")
delta_n  = abs(n - summary["n_samples"])
delta_r  = abs(nr - summary["n_readmit"])
if delta_n <= 5 and delta_r <= 5:
    print(f"  ✅ matches summary.json within rounding tolerance")
else:
    print(f"  ⚠️  mismatch: Δn={delta_n}, Δreadmit={delta_r}")
