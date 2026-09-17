"""
Supplementary Table S1, Paired bootstrap 95% CIs for AUROC comparisons.

Compares AToP Transformer vs LACE logistic regression vs LACE XGBoost on
the test set, with 1,000 paired bootstrap resamples. Emits:
  - runs/full_CPD/baselines/auroc_bootstrap.csv        (per-comparison stats)
  - runs/full_CPD/baselines/auroc_bootstrap_table.md   (paste-ready Markdown)

The AToP predictions come from the cached test_predictions.csv. LACE LR
and LACE XGBoost are retrained in-script on the same cohort used in
reproduce_lace_baselines.py (≈30 s combined) so per-patient predictions
are guaranteed aligned with the AToP predictions by subject_id+hadm_id.

Run from atop_package/:
    python scripts/table_s1_auroc_bootstrap.py
"""
from __future__ import annotations
import os, sys, json, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.run_layout import RunLayout

# Resolve from env so make_table_s1 / a CLI caller can override without
# touching this file.
BUNDLE = os.environ.get("ATOP_RUN_DIR", "../runs/mimiciv_readmission_30d")
MIMIC_DIR = os.environ.get("ATOP_MIMIC_DIR", "../data/mimiciv/3.1/hosp")
N_BOOT = int(os.environ.get("ATOP_N_BOOT", "1000"))
SEED = 42

_layout = RunLayout(BUNDLE)
SEQ_DIR = _layout.attribution_dir          # was BUNDLE/explain/data
OUT_DIR = _layout.baselines_dir            # was BUNDLE/baselines
OUT_CSV = os.path.join(OUT_DIR, "auroc_bootstrap.csv")
OUT_MD = os.path.join(OUT_DIR, "auroc_bootstrap_table.md")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["los_days", "cci", "ed_visits_6mo", "A", "age", "female", "prior_admissions"]


# ── Load AToP test predictions ─────────────────────────────────────────────
print("[1] Loading AToP test predictions...")
atop = pd.read_csv(
    _layout.test_predictions_csv,
    dtype={"subject_id": str, "hadm_id": str},
)
atop = atop.rename(columns={"y_prob": "atop_prob"})
print(f"    AToP: {len(atop):,} patients")


# ── Build LACE feature frame (same cohort construction as reproduce_baselines) ─
print("[2] Loading LACE cohort inputs...")

def load_cohort(split: str) -> pd.DataFrame:
    with open(f"{SEQ_DIR}/sequences_{split}.pkl", "rb") as f:
        df = pickle.load(f)
    out = df[["patient_id", "index_hadm_id", "readmission"]].copy()
    out = out.rename(columns={
        "patient_id": "subject_id",
        "index_hadm_id": "hadm_id",
        "readmission": "readmit_30d",
    })
    out["subject_id"] = out["subject_id"].astype(str)
    out["hadm_id"] = out["hadm_id"].astype(str)
    out["readmit_30d"] = out["readmit_30d"].astype(float)
    return out


train_cohort = load_cohort("train")
test_cohort = load_cohort("test")
print(f"    train={len(train_cohort):,}  test={len(test_cohort):,}")

print("[3] Loading LACE scores + demographics...")
lace = pd.read_csv(
    _layout.lace_scores_csv,
    dtype={"subject_id": str, "hadm_id": str},
)
patients = pd.read_csv(
    f"{MIMIC_DIR}/patients.csv.gz",
    dtype={"subject_id": str},
    usecols=["subject_id", "gender", "anchor_age"],
)
admissions = pd.read_csv(
    f"{MIMIC_DIR}/admissions.csv.gz",
    dtype={"subject_id": str, "hadm_id": str},
    parse_dates=["admittime"],
)
admissions_sorted = admissions.sort_values(["subject_id", "admittime"])
admissions_sorted["prior_admissions"] = admissions_sorted.groupby("subject_id").cumcount()
prior_adm = admissions_sorted[["subject_id", "hadm_id", "prior_admissions"]]


def merge_features(cohort: pd.DataFrame) -> pd.DataFrame:
    df = cohort.merge(
        lace[["subject_id", "hadm_id", "los_days", "cci", "ed_visits_6mo", "A"]],
        on=["subject_id", "hadm_id"], how="inner",
    )
    df = df.merge(patients[["subject_id", "gender", "anchor_age"]],
                  on="subject_id", how="left")
    df = df.merge(prior_adm[["subject_id", "hadm_id", "prior_admissions"]],
                  on=["subject_id", "hadm_id"], how="left")
    df["age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["female"] = (df["gender"] == "F").astype(float)
    df = df.dropna(subset=FEATURES)
    df = df[np.isfinite(df[FEATURES].to_numpy().astype(float)).all(axis=1)]
    return df


train_df = merge_features(train_cohort)
test_df = merge_features(test_cohort)
print(f"    train merged: {len(train_df):,}  test merged: {len(test_df):,}")

X_train = train_df[FEATURES].values.astype(float)
y_train = train_df["readmit_30d"].values.astype(float)
X_test = test_df[FEATURES].values.astype(float)
y_test = test_df["readmit_30d"].values.astype(float)


# ── Train LACE baselines and collect per-patient predictions ──────────────
print("[4] Training LACE logistic regression...")
sc = StandardScaler()
lr = LogisticRegression(max_iter=1000, random_state=SEED)
lr.fit(sc.fit_transform(X_train), y_train)
lace_lr_prob = lr.predict_proba(sc.transform(X_test))[:, 1]

print("[5] Training LACE XGBoost...")
scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
xgb = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos, random_state=SEED,
    eval_metric="logloss", verbosity=0,
)
xgb.fit(X_train, y_train)
lace_xgb_prob = xgb.predict_proba(X_test)[:, 1]


lace_preds = test_df[["subject_id", "hadm_id", "readmit_30d"]].copy()
lace_preds["lace_lr_prob"] = lace_lr_prob
lace_preds["lace_xgb_prob"] = lace_xgb_prob


# ── Align all three prediction sets ────────────────────────────────────────
print("[6] Aligning AToP + LACE LR + LACE XGB on common (subject_id, hadm_id)...")
aligned = atop.merge(lace_preds, on=["subject_id", "hadm_id"], how="inner")
print(f"    aligned cohort: {len(aligned):,} patients")
# Sanity check labels agree
label_match = (aligned["y_true"].astype(float) == aligned["readmit_30d"].astype(float)).all()
print(f"    label consistency: {'✅' if label_match else '⚠️ mismatch'}")

y = aligned["y_true"].values.astype(float)
p_atop = aligned["atop_prob"].values
p_lr = aligned["lace_lr_prob"].values
p_xgb = aligned["lace_xgb_prob"].values


def auroc_safe(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


print(f"\nPoint AUROCs:")
print(f"  AToP Transformer: {auroc_safe(y, p_atop):.4f}")
print(f"  LACE XGBoost:    {auroc_safe(y, p_xgb):.4f}")
print(f"  LACE LR:         {auroc_safe(y, p_lr):.4f}")


# ── Paired bootstrap ───────────────────────────────────────────────────────
print(f"\n[7] Paired bootstrap ({N_BOOT} resamples, seed={SEED})...")
rng = np.random.RandomState(SEED)
n = len(y)

d_atop_xgb, d_atop_lr, d_xgb_lr = [], [], []
auroc_atop, auroc_xgb, auroc_lr = [], [], []
for _ in range(N_BOOT):
    idx = rng.randint(0, n, size=n)
    yb = y[idx]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    a_a = roc_auc_score(yb, p_atop[idx])
    a_x = roc_auc_score(yb, p_xgb[idx])
    a_l = roc_auc_score(yb, p_lr[idx])
    auroc_atop.append(a_a)
    auroc_xgb.append(a_x)
    auroc_lr.append(a_l)
    d_atop_xgb.append(a_a - a_x)
    d_atop_lr.append(a_a - a_l)
    d_xgb_lr.append(a_x - a_l)


def summary(vals, name):
    arr = np.array(vals)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    # Two-sided p: fraction of resamples where delta crosses zero
    # (approximation, valid for deltas only)
    if name.startswith("d_"):
        pval = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
    else:
        pval = float("nan")
    return {
        "name": name,
        "point": float(np.mean(arr)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": float(pval) if pval == pval else None,
    }


rows = [
    summary(auroc_atop, "AUROC: AToP Transformer"),
    summary(auroc_xgb, "AUROC: LACE XGBoost"),
    summary(auroc_lr, "AUROC: LACE logistic regression"),
    summary(d_atop_xgb, "d_ΔAUROC: AToP − LACE XGBoost"),
    summary(d_atop_lr, "d_ΔAUROC: AToP − LACE LR"),
    summary(d_xgb_lr, "d_ΔAUROC: LACE XGBoost − LACE LR"),
]
df_rows = pd.DataFrame(rows)
df_rows.to_csv(OUT_CSV, index=False)
print(f"\n[8] Saved CSV → {OUT_CSV}")


# ── Markdown table ─────────────────────────────────────────────────────────
def fmt_row(r):
    point = r["point"]
    lo, hi = r["ci_lo"], r["ci_hi"]
    if r["p_value"] is None:
        return (f"| {r['name'].replace('AUROC: ', '')} "
                f"| {point:.4f} | ({lo:.4f}, {hi:.4f}) |, |")
    p = r["p_value"]
    pstr = "< 0.001" if p < 0.001 else f"{p:.3f}"
    return (f"| {r['name'].replace('d_ΔAUROC: ', '')} "
            f"| {point:+.4f} | ({lo:+.4f}, {hi:+.4f}) | {pstr} |")


lines = [
    "# Table S1, Bootstrap 95% CIs for AUROC comparisons",
    "",
    f"Paired bootstrap (n = {N_BOOT:,} resamples, seed = {SEED}) on the "
    f"aligned test cohort (n = {len(aligned):,}).",
    "",
    "## Individual model AUROCs",
    "",
    "| Model | AUROC | 95% CI |",
    "|---|---|---|",
    fmt_row(rows[0]).replace(" |,", "").rstrip("|").rstrip() + " |",
    fmt_row(rows[1]).replace(" |,", "").rstrip("|").rstrip() + " |",
    fmt_row(rows[2]).replace(" |,", "").rstrip("|").rstrip() + " |",
    "",
    "## Pairwise ΔAUROC (positive = first model better)",
    "",
    "| Comparison | ΔAUROC | 95% CI | p (two-sided) |",
    "|---|---|---|---|",
    fmt_row(rows[3]),
    fmt_row(rows[4]),
    fmt_row(rows[5]),
]
md = "\n".join(lines)
with open(OUT_MD, "w") as f:
    f.write(md + "\n")
print(f"    Markdown → {OUT_MD}")

print("\n" + "=" * 66)
print(md)
print("=" * 66)
