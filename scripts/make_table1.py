"""make_table1, Table 1 (cohort characteristics) on the full train+val+test cohort.

Reproduces the model's index-admission rule:
  - ≥2 admissions: penultimate (n-2, 0-indexed)
  - 1 admission:   that single admission (label=0)

Output: stratified statistics (Overall / Readmitted / Not readmitted / SMD)
written to <run_dir>/7_outputs/tables/table1_cohort.csv and copied to the
manuscript tables directory.

Standardized mean difference (SMD) is reported instead of p-values because
with n ≈ 218k everything is statistically significant; SMD is sample-size
invariant. Convention: SMD > 0.1 is considered imbalanced.

Variables reported:
  Demographics: age, female sex, race/ethnicity, insurance
  Acute encounter: admission type, length of stay
  Comorbidity / history: Charlson, prior ED visits, prior admissions
  Composite risk score: LACE

Usage:
    python scripts/make_table1.py --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.data.mimic import build_readmission_labels
from atop.run_layout import RunLayout


MAX_VISITS = 20  # from config.json


# ── Categorical mappings ────────────────────────────────────────────────
def map_race(r) -> str:
    """Collapse MIMIC's granular race values into 5 broad groups."""
    if pd.isna(r):
        return "Other/Unknown"
    r = str(r).upper()
    if "WHITE" in r or r == "PORTUGUESE":
        return "White"
    if "BLACK" in r or "AFRICAN" in r:
        return "Black"
    if "HISPANIC" in r or "LATIN" in r:
        return "Hispanic or Latino"
    if "ASIAN" in r:
        return "Asian"
    return "Other/Unknown"


def map_insurance(i) -> str:
    """Collapse MIMIC insurance to Medicare/Medicaid/Private/Other."""
    if pd.isna(i):
        return "Other/Unknown"
    i = str(i).strip()
    if i == "Medicare":
        return "Medicare"
    if i == "Medicaid":
        return "Medicaid"
    if i == "Private":
        return "Private"
    return "Other/Unknown"


def map_admission_type(a) -> str:
    """Collapse MIMIC admission_type into 4 buckets."""
    if pd.isna(a):
        return "Other"
    a = str(a).upper()
    if "EMER" in a:
        return "Emergency"
    if "URGENT" in a:
        return "Urgent"
    if "OBSERV" in a:
        return "Observation"
    if "ELECT" in a or "SURGICAL SAME DAY" in a:
        return "Elective"
    return "Other"


def bin_ed_visits(n) -> str:
    if pd.isna(n):
        return "0"
    n = int(n)
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    return "≥2"


def bin_n_visits(n) -> str:
    """Bin total admissions (incl. index): 1 / 2 / ≥3."""
    if pd.isna(n):
        return "1"
    n = int(n)
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "≥3"


# ── Formatting helpers ──────────────────────────────────────────────────
def med_iqr(s, fmt=".0f") -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "n/a"
    return f"{s.median():{fmt}} ({s.quantile(0.25):{fmt}}–{s.quantile(0.75):{fmt}})"


def n_pct_from_count(n: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{n:,} ({n / total * 100:.1f}%)"


def n_pct(s) -> str:
    """For a binary indicator series."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "n/a"
    return n_pct_from_count(int(s.sum()), len(s))


# ── SMD helpers ─────────────────────────────────────────────────────────
def smd_continuous(x_r: pd.Series, x_nr: pd.Series) -> float:
    """SMD = |mean_R − mean_NR| / pooled SD."""
    x_r = pd.to_numeric(x_r, errors="coerce").dropna()
    x_nr = pd.to_numeric(x_nr, errors="coerce").dropna()
    if len(x_r) < 2 or len(x_nr) < 2:
        return float("nan")
    v_r = x_r.var(ddof=1)
    v_nr = x_nr.var(ddof=1)
    pooled_sd = ((v_r + v_nr) / 2.0) ** 0.5
    if pooled_sd == 0:
        return 0.0
    return abs(x_r.mean() - x_nr.mean()) / pooled_sd


def smd_binary(p_r: float, p_nr: float) -> float:
    """SMD for two proportions."""
    var_pool = (p_r * (1 - p_r) + p_nr * (1 - p_nr)) / 2.0
    if var_pool == 0:
        return 0.0
    return abs(p_r - p_nr) / (var_pool ** 0.5)


def fmt_smd(v: float) -> str:
    if pd.isna(v):
        return ""
    return f"{v:.3f}"


# ── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Compute Table 1 (cohort characteristics)")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--mimic_dir", default="../data/mimiciv/3.1/hosp")
    p.add_argument("--max-visits", type=int, default=MAX_VISITS)
    p.add_argument("--output-csv", default=None,
                   help="CSV destination (default: <run_dir>/7_outputs/tables/table1_cohort.csv)")
    p.add_argument("--manuscript-csv", default=None,
                   help="Optional second copy destination for manuscript pipeline.")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    out_csv = args.output_csv or layout.table_csv("table1_cohort")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # ── Splits + summary ─────────────────────────────────────────────
    with open(layout.splits_json) as f:
        splits = json.load(f)
    all_pids = set(str(p) for p in (splits["train"] + splits["val"] + splits["test"]))
    print(f"[splits]   train={len(splits['train']):,}  val={len(splits['val']):,}  "
          f"test={len(splits['test']):,}  total={len(all_pids):,}")

    with open(layout.summary_json) as f:
        summary = json.load(f)
    print(f"[summary]  n_samples={summary['n_samples']:,}  "
          f"n_readmit={summary['n_readmit']:,}  n_no_readmit={summary['n_no_readmit']:,}")

    # ── MIMIC tables ─────────────────────────────────────────────────
    print("\n[1] Loading admissions + patients...")
    admissions = pd.read_csv(
        os.path.join(args.mimic_dir, "admissions.csv.gz"),
        dtype={"subject_id": str, "hadm_id": str},
        parse_dates=["admittime", "dischtime"],
    )
    patients = pd.read_csv(
        os.path.join(args.mimic_dir, "patients.csv.gz"),
        dtype={"subject_id": str},
        usecols=["subject_id", "gender", "anchor_age"],
    )
    print(f"    admissions={len(admissions):,}  patients={len(patients):,}")

    # ── Readmission labels per admission ──────────────────────────────
    print("\n[2] Building readmission labels (exclude_elective_readmissions=True)...")
    labels_df = build_readmission_labels(admissions, exclude_elective_readmissions=True)

    # ── Index-admission rule ─────────────────────────────────────────
    labels_df = labels_df[labels_df["subject_id"].isin(all_pids)].copy()
    labels_df = labels_df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    labels_df["pos_in_pt"] = labels_df.groupby("subject_id").cumcount()
    labels_df["n_in_pt"] = labels_df.groupby("subject_id")["admittime"].transform("count")
    labels_df["idx_pos"] = np.where(labels_df["n_in_pt"] >= 2,
                                    labels_df["n_in_pt"] - 2, 0)
    index_df = labels_df[labels_df["pos_in_pt"] == labels_df["idx_pos"]].copy()
    index_df["n_visits"] = np.minimum(index_df["idx_pos"] + 1, args.max_visits)
    print(f"\n[3] Selected index admissions: {len(index_df):,}  "
          f"readmit rate={index_df['readmit_30d'].mean():.4f}")

    # ── Demographics + admission type + insurance + race ─────────────
    print("\n[4] Merging demographics + admission type + insurance + race...")
    cohort = index_df.merge(
        patients[["subject_id", "gender", "anchor_age"]],
        on="subject_id", how="left",
    )
    cohort = cohort.merge(
        admissions[["subject_id", "hadm_id", "admission_type",
                    "race", "insurance"]],
        on=["subject_id", "hadm_id"], how="left",
    )
    cohort["age"] = pd.to_numeric(cohort["anchor_age"], errors="coerce")
    cohort["female"] = (cohort["gender"] == "F").astype(float)

    cohort["race_group"] = cohort["race"].map(map_race)
    cohort["insurance_group"] = cohort["insurance"].map(map_insurance)
    cohort["adm_type_group"] = cohort["admission_type"].map(map_admission_type)

    # ── LACE components ──────────────────────────────────────────────
    print("\n[5] Merging LACE components...")
    lace_path = layout.lace_scores_csv
    if not os.path.exists(lace_path):
        sys.exit(f"  [ERROR] Missing {lace_path}, run scripts/precompute_lace.py first")
    lace = pd.read_csv(lace_path, dtype={"subject_id": str, "hadm_id": str})
    cohort = cohort.merge(
        lace[["subject_id", "hadm_id", "los_days", "cci", "ed_visits_6mo",
              "A", "L", "C", "E", "lace_score"]],
        on=["subject_id", "hadm_id"], how="left",
    )
    cohort["ed_visit_bin"] = cohort["ed_visits_6mo"].map(bin_ed_visits)
    cohort["n_visits_bin"] = cohort["n_visits"].map(bin_n_visits)

    # ── Stratified counts ────────────────────────────────────────────
    n_total = len(cohort)
    nr = int(cohort["readmit_30d"].sum())
    nnr = n_total - nr
    r_df = cohort[cohort["readmit_30d"] == 1]
    nr_df = cohort[cohort["readmit_30d"] == 0]

    print("\n" + "=" * 72)
    print("TABLE 1, COHORT CHARACTERISTICS")
    print("=" * 72)
    print(f"\n  Overall N:             {n_total:,}")
    print(f"  Readmitted:            {nr:,}  ({nr / n_total * 100:.1f}%)")
    print(f"  Not readmitted:        {nnr:,}  ({nnr / n_total * 100:.1f}%)")

    # ── Row builders ─────────────────────────────────────────────────
    rows = []

    def cont_row(name: str, col: str, fmt: str = ".0f") -> None:
        ov = med_iqr(cohort[col], fmt)
        ra = med_iqr(r_df[col], fmt)
        nra = med_iqr(nr_df[col], fmt)
        s = smd_continuous(r_df[col], nr_df[col])
        rows.append({"variable": name, "overall": ov, "readmitted": ra,
                     "not_readmitted": nra, "smd": fmt_smd(s)})

    def bin_row(name: str, col: str) -> None:
        ov = n_pct(cohort[col])
        ra = n_pct(r_df[col])
        nra = n_pct(nr_df[col])
        p_r = pd.to_numeric(r_df[col], errors="coerce").dropna().mean()
        p_nr = pd.to_numeric(nr_df[col], errors="coerce").dropna().mean()
        s = smd_binary(p_r, p_nr)
        rows.append({"variable": name, "overall": ov, "readmitted": ra,
                     "not_readmitted": nra, "smd": fmt_smd(s)})

    def section_header(name: str) -> None:
        rows.append({"variable": name, "overall": "", "readmitted": "",
                     "not_readmitted": "", "smd": ""})

    def cat_subrow(label: str, col: str, value: str) -> None:
        is_cat_overall = (cohort[col] == value)
        is_cat_r = (r_df[col] == value)
        is_cat_nr = (nr_df[col] == value)
        ov = n_pct_from_count(int(is_cat_overall.sum()), n_total)
        ra = n_pct_from_count(int(is_cat_r.sum()), len(r_df))
        nra = n_pct_from_count(int(is_cat_nr.sum()), len(nr_df))
        p_r = is_cat_r.mean()
        p_nr = is_cat_nr.mean()
        s = smd_binary(p_r, p_nr)
        rows.append({"variable": f"  {label}", "overall": ov, "readmitted": ra,
                     "not_readmitted": nra, "smd": fmt_smd(s)})

    # ── Build table rows ─────────────────────────────────────────────
    cont_row("Age, median (IQR)", "age")
    bin_row("Female sex, n (%)", "female")

    section_header("Race/ethnicity, n (%)")
    for cat in ["White", "Black", "Hispanic or Latino", "Asian", "Other/Unknown"]:
        cat_subrow(cat, "race_group", cat)

    section_header("Insurance, n (%)")
    for cat in ["Medicare", "Medicaid", "Private", "Other/Unknown"]:
        cat_subrow(cat, "insurance_group", cat)

    section_header("Admission type, n (%)")
    for cat in ["Emergency", "Urgent", "Observation", "Elective", "Other"]:
        # Skip "Other" if zero in cohort
        if (cohort["adm_type_group"] == cat).sum() == 0:
            continue
        cat_subrow(cat, "adm_type_group", cat)

    cont_row("Length of stay (days), median (IQR)", "los_days", ".1f")
    cont_row("Charlson Comorbidity Index, median (IQR)", "cci")

    section_header("ED visits (prior 6 mo), n (%)")
    for cat in ["0", "1", "≥2"]:
        cat_subrow(cat, "ed_visit_bin", cat)

    section_header("Number of admissions (incl. index), n (%)")
    for cat in ["1", "2", "≥3"]:
        cat_subrow(cat, "n_visits_bin", cat)

    cont_row("LACE score, median (IQR)", "lace_score")

    # ── Print to console ─────────────────────────────────────────────
    print()
    fmt_str = "  {:<48}  {:<22}  {:<22}  {:<22}  {:<6}"
    print(fmt_str.format("Variable", "Overall", "Readmitted", "Not readmitted", "SMD"))
    print("  " + "-" * 130)
    for r in rows:
        print(fmt_str.format(r["variable"], r["overall"], r["readmitted"],
                             r["not_readmitted"], r["smd"]))

    # ── Save CSV ─────────────────────────────────────────────────────
    out_df = pd.DataFrame(rows, columns=["variable", "overall", "readmitted",
                                           "not_readmitted", "smd"])
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

    # ── Sidecar metadata CSV with N counts for column headers ────────
    meta_csv = os.path.splitext(out_csv)[0] + "_meta.csv"
    pd.DataFrame([{
        "n_total": n_total,
        "n_readmitted": nr,
        "n_not_readmitted": nnr,
        "pct_readmitted": round(nr / n_total * 100, 2),
        "pct_not_readmitted": round(nnr / n_total * 100, 2),
    }]).to_csv(meta_csv, index=False)
    print(f"Saved metadata CSV: {meta_csv}")

    # ── Copy to manuscript directory if requested ───────────────────
    if args.manuscript_csv:
        os.makedirs(os.path.dirname(args.manuscript_csv), exist_ok=True)
        out_df.to_csv(args.manuscript_csv, index=False)
        # Also copy metadata
        meta_manuscript = os.path.splitext(args.manuscript_csv)[0] + "_meta.csv"
        pd.DataFrame([{
            "n_total": n_total,
            "n_readmitted": nr,
            "n_not_readmitted": nnr,
            "pct_readmitted": round(nr / n_total * 100, 2),
            "pct_not_readmitted": round(nnr / n_total * 100, 2),
        }]).to_csv(meta_manuscript, index=False)
        print(f"Also saved to manuscript: {args.manuscript_csv}")
        print(f"Also saved metadata to: {meta_manuscript}")

    # ── Cross-check ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CROSS-CHECK against summary.json")
    print("=" * 72)
    print(f"  expected: n={summary['n_samples']:,} readmit={summary['n_readmit']:,}")
    print(f"  computed: n={n_total:,} readmit={nr:,}")
    if abs(n_total - summary["n_samples"]) <= 5 and abs(nr - summary["n_readmit"]) <= 5:
        print("  ✓ matches within rounding tolerance")
    else:
        print(f"  ⚠ mismatch: Δn={abs(n_total - summary['n_samples'])}, "
              f"Δreadmit={abs(nr - summary['n_readmit'])}")


if __name__ == "__main__":
    main()
