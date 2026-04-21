"""
Reproduce LACE+ LR and XGBoost baselines for AToP paper.

Features (7): los_days, cci, ed_visits_6mo, A (acuity), age, female, prior_admissions
Cohort: matches the Transformer model exactly. (patient_id, index_hadm_id, readmission)
tuples are pulled from runs/full_CPD/explain/data/sequences_{train,test}.pkl,
which the model produced during training/evaluation. This guarantees the same
patients and same index admissions as the paper's AUROC 0.766 run.

Expected results (from Colab session):
  LACE+ LR:      AUROC 0.657, PR-AUC ~0.24
  LACE+ XGBoost: AUROC 0.729, PR-AUC ~0.32

Output saved to: runs/full_CPD/baselines/
"""
from __future__ import annotations
import os, sys, json, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FULL_RUN  = "../runs/full_CPD"
MIMIC_DIR = "../data/mimiciv/3.1/hosp"
SEQ_DIR   = f"{FULL_RUN}/explain/data"
OUT_DIR   = f"{FULL_RUN}/baselines"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["los_days", "cci", "ed_visits_6mo", "A", "age", "female", "prior_admissions"]


def load_cohort(split: str) -> pd.DataFrame:
    """Load (patient_id, index_hadm_id, readmission) from cached sequences."""
    with open(f"{SEQ_DIR}/sequences_{split}.pkl", "rb") as f:
        df = pickle.load(f)
    cols = ["patient_id", "index_hadm_id", "readmission"]
    out = df[cols].copy()
    out = out.rename(columns={"patient_id": "subject_id",
                              "index_hadm_id": "hadm_id",
                              "readmission": "readmit_30d"})
    out["subject_id"] = out["subject_id"].astype(str)
    out["hadm_id"]    = out["hadm_id"].astype(str)
    out["readmit_30d"] = out["readmit_30d"].astype(float)
    return out


# ── Load the model's exact cohort ──────────────────────────────────────────
print("[1] Loading model's cohort from sequences_*.pkl...")
train_cohort = load_cohort("train")
test_cohort  = load_cohort("test")
print(f"    train: {len(train_cohort):,}  readmit rate {train_cohort['readmit_30d'].mean():.4f}")
print(f"    test : {len(test_cohort):,}  readmit rate {test_cohort['readmit_30d'].mean():.4f}")

# ── Load LACE scores ───────────────────────────────────────────────────────
print("[2] Loading LACE scores...")
lace = pd.read_csv(f"{FULL_RUN}/lace_scores_all.csv",
                   dtype={"subject_id": str, "hadm_id": str})
print(f"    {len(lace):,} rows")

# ── Load MIMIC demographics + build prior_admissions ───────────────────────
print("[3] Loading MIMIC demographics...")
patients = pd.read_csv(f"{MIMIC_DIR}/patients.csv.gz",
                       dtype={"subject_id": str},
                       usecols=["subject_id", "gender", "anchor_age"])
admissions = pd.read_csv(f"{MIMIC_DIR}/admissions.csv.gz",
                         dtype={"subject_id": str, "hadm_id": str},
                         parse_dates=["admittime"])

# prior_admissions = number of this patient's admissions strictly before this one
admissions_sorted = admissions.sort_values(["subject_id", "admittime"])
admissions_sorted["prior_admissions"] = admissions_sorted.groupby("subject_id").cumcount()
prior_adm = admissions_sorted[["subject_id", "hadm_id", "prior_admissions"]]


def merge_features(cohort: pd.DataFrame) -> pd.DataFrame:
    df = cohort.merge(
        lace[["subject_id", "hadm_id", "los_days", "cci", "ed_visits_6mo", "A"]],
        on=["subject_id", "hadm_id"], how="inner"
    )
    df = df.merge(patients[["subject_id", "gender", "anchor_age"]],
                  on="subject_id", how="left")
    df = df.merge(prior_adm[["subject_id", "hadm_id", "prior_admissions"]],
                  on=["subject_id", "hadm_id"], how="left")
    df["age"]    = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["female"] = (df["gender"] == "F").astype(float)
    df = df.dropna(subset=FEATURES)
    # guard against non-finite los_days (occasional bad timestamps)
    df = df[np.isfinite(df[FEATURES].to_numpy().astype(float)).all(axis=1)]
    return df


print("[4] Merging features for train and test...")
train_df = merge_features(train_cohort)
test_df  = merge_features(test_cohort)
print(f"    train: {len(train_df):,} / {len(train_cohort):,} matched")
print(f"    test : {len(test_df):,} / {len(test_cohort):,} matched")
print(f"    train readmit rate: {train_df['readmit_30d'].mean():.4f}")
print(f"    test  readmit rate: {test_df['readmit_30d'].mean():.4f}")

X_train = train_df[FEATURES].values.astype(float)
y_train = train_df["readmit_30d"].values.astype(float)
X_test  = test_df[FEATURES].values.astype(float)
y_test  = test_df["readmit_30d"].values.astype(float)

# ── Logistic Regression ────────────────────────────────────────────────────
print("\n[5] LACE+ Logistic Regression...")
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_prob  = lr.predict_proba(X_test_s)[:, 1]
lr_auroc = roc_auc_score(y_test, lr_prob)
lr_prauc = average_precision_score(y_test, lr_prob)
print(f"    AUROC={lr_auroc:.4f}  PR-AUC={lr_prauc:.4f}  (paper: 0.657 / ~0.24)")
if abs(lr_auroc - 0.657) > 0.01:
    print(f"    ⚠️  AUROC differs from expected 0.657 by {abs(lr_auroc-0.657):.4f}")
else:
    print(f"    ✅ Matches expected 0.657 within tolerance")

# ── XGBoost ────────────────────────────────────────────────────────────────
print("\n[6] LACE+ XGBoost...")
scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)
xgb.fit(X_train, y_train)
xgb_prob  = xgb.predict_proba(X_test)[:, 1]
xgb_auroc = roc_auc_score(y_test, xgb_prob)
xgb_prauc = average_precision_score(y_test, xgb_prob)
print(f"    AUROC={xgb_auroc:.4f}  PR-AUC={xgb_prauc:.4f}  (paper: 0.729 / ~0.32)")
if abs(xgb_auroc - 0.729) > 0.01:
    print(f"    ⚠️  AUROC differs from expected 0.729 by {abs(xgb_auroc-0.729):.4f}")
else:
    print(f"    ✅ Matches expected 0.729 within tolerance")

# ── Bootstrap CIs ──────────────────────────────────────────────────────────
print("\n[7] Bootstrap CIs (n=1000)...")
rng = np.random.RandomState(42)
n = len(y_test)
lr_boot, xgb_boot, lr_prb, xgb_prb = [], [], [], []
for _ in range(1000):
    idx = rng.randint(0, n, size=n)
    yb = y_test[idx]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    lr_boot.append(roc_auc_score(yb, lr_prob[idx]))
    xgb_boot.append(roc_auc_score(yb, xgb_prob[idx]))
    lr_prb.append(average_precision_score(yb, lr_prob[idx]))
    xgb_prb.append(average_precision_score(yb, xgb_prob[idx]))

lr_ci  = np.percentile(lr_boot,  [2.5, 97.5])
xgb_ci = np.percentile(xgb_boot, [2.5, 97.5])
lr_prauc_ci  = np.percentile(lr_prb,  [2.5, 97.5])
xgb_prauc_ci = np.percentile(xgb_prb, [2.5, 97.5])
print(f"    LR  AUROC: {lr_auroc:.4f}  (95% CI: {lr_ci[0]:.4f}–{lr_ci[1]:.4f})")
print(f"    XGB AUROC: {xgb_auroc:.4f}  (95% CI: {xgb_ci[0]:.4f}–{xgb_ci[1]:.4f})")

# ── Save results ───────────────────────────────────────────────────────────
print("\n[8] Saving results...")
results = pd.DataFrame([
    {"model": "LACE+ LR",
     "features": ",".join(FEATURES),
     "n_train": len(train_df), "n_test": len(test_df),
     "auroc": lr_auroc, "pr_auc": lr_prauc,
     "auroc_ci_lo": lr_ci[0], "auroc_ci_hi": lr_ci[1],
     "pr_auc_ci_lo": lr_prauc_ci[0], "pr_auc_ci_hi": lr_prauc_ci[1]},
    {"model": "LACE+ XGBoost",
     "features": ",".join(FEATURES),
     "n_train": len(train_df), "n_test": len(test_df),
     "auroc": xgb_auroc, "pr_auc": xgb_prauc,
     "auroc_ci_lo": xgb_ci[0], "auroc_ci_hi": xgb_ci[1],
     "pr_auc_ci_lo": xgb_prauc_ci[0], "pr_auc_ci_hi": xgb_prauc_ci[1]},
])
out_path = f"{OUT_DIR}/lace_plus_baselines.csv"
results.to_csv(out_path, index=False)
print(f"    Saved to: {out_path}")
print(f"\n{results.to_string(index=False)}")

print("\n" + "=" * 60)
print("COPY INTO PAPER:")
print(f"  LACE+ LR:      AUROC {lr_auroc:.2f} (95% CI {lr_ci[0]:.2f}–{lr_ci[1]:.2f}), "
      f"PR-AUC {lr_prauc:.2f}")
print(f"  LACE+ XGBoost: AUROC {xgb_auroc:.2f} (95% CI {xgb_ci[0]:.2f}–{xgb_ci[1]:.2f}), "
      f"PR-AUC {xgb_prauc:.2f}")
print("=" * 60)
