"""
Extract all [val] placeholder values for AToP_paper_v6.docx
from cached CSVs in runs/full_CPD/explain/
"""
import os, sys, glob, json, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FULL_RUN  = "../runs/full_CPD"
MIMIC_DIR = "../data/mimiciv/3.1/hosp"
SEQ_DIR   = f"{FULL_RUN}/explain/data"
CSV_DIR   = f"{FULL_RUN}/explain/figures/csv"
MAIN_DIR  = f"{FULL_RUN}/explain/figures/main"
TABLE_DIR = f"{FULL_RUN}/explain/tables"
MINE_DIR  = f"{FULL_RUN}/mining_cache"

print("=" * 70)
print("AToP [val] PLACEHOLDER EXTRACTION")
print("=" * 70)

# ── Summary stats ─────────────────────────────────────────────────────────
with open(f"{FULL_RUN}/summary.json") as f:
    summary = json.load(f)
with open(f"{FULL_RUN}/metrics.json") as f:
    metrics = json.load(f)
with open(f"{FULL_RUN}/config.json") as f:
    cfg = json.load(f)

print("\n── MODEL PERFORMANCE ──")
print(f"  AUROC:   {metrics['auroc']:.4f}")
print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")

# ── Mining / attribution operational config ───────────────────────────────
print("\n── MINING & ATTRIBUTION CONFIG (from runs/full_CPD/config.json) ──")
mining_keys = [
    # Attribution
    "ig_n_steps", "ig_mass", "ig_max_tokens", "ig_batch_size",
    # Mining method + support
    "mining_method", "min_support_frac", "mine_on_trainval",
    # Episode mining
    "episode_max_len", "episode_min_steps", "episode_topn",
    # N-gram / PrefixSpan (reference only if used)
    "ngram_min_len", "ngram_max_len",
    "prefixspan_min_len", "prefixspan_topn",
    # Dedup / capping
    "jaccard_dedup", "jaccard_rep", "cap_by_or_per_length", "cap_metric",
    # Validation
    "validate_top_k", "validate_max_admissions_per_pattern", "n_shuffle_draws",
    # Token construction
    "token_types", "max_visits", "max_seq_len", "one_per_patient",
    "exclude_elective_readmissions", "harmonize_icd", "chronic_filter",
    "first_occurrence_only", "first_occurrence_drugs_only", "max_drug_freq",
]
for k in mining_keys:
    if k in cfg:
        v = cfg[k]
        print(f"  {k:<40} {v}")

# ── TABLE 1: Cohort characteristics ───────────────────────────────────────
print("\n── TABLE 1: COHORT CHARACTERISTICS ──")
t1 = pd.read_csv(f"{TABLE_DIR}/table1_population.csv")
for _, row in t1.iterrows():
    print(f"  {row['Characteristic']:<45} {row['Value']}")

n_total   = summary['n_samples']
n_readmit = summary['n_readmit']
n_no      = summary['n_no_readmit']
print(f"\n  Readmitted (n):         {n_readmit:,}")
print(f"  Not Readmitted (n):     {n_no:,}")
print(f"  Total (N):              {n_total:,}")

# ── Model cohort from cached sequences (correct cohort, matches Transformer run)
print("\n  [Loading model cohort from sequences_*.pkl]")
def load_seq_cohort(split):
    with open(f"{SEQ_DIR}/sequences_{split}.pkl", "rb") as f:
        df = pickle.load(f)
    return df[["patient_id", "index_hadm_id", "readmission"]].rename(columns={
        "patient_id": "subject_id", "index_hadm_id": "hadm_id", "readmission": "readmit_30d"
    }).astype({"subject_id": str, "hadm_id": str})

cohort = pd.concat([load_seq_cohort("train"), load_seq_cohort("test")], ignore_index=True)
print(f"  Cohort patients (train+test): {len(cohort):,} "
      f"(val={summary['n_samples']-len(cohort):,} excluded: no cached sequences)")

# Merge demographics
patients = pd.read_csv(f"{MIMIC_DIR}/patients.csv.gz", dtype=str)
admissions = pd.read_csv(f"{MIMIC_DIR}/admissions.csv.gz", dtype=str,
                         parse_dates=['admittime', 'dischtime'])
cohort = cohort.merge(patients[['subject_id', 'gender', 'anchor_age']],
                     on='subject_id', how='left')
cohort['anchor_age'] = pd.to_numeric(cohort['anchor_age'], errors='coerce')


def med_iqr(series):
    s = series.dropna()
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}–{s.quantile(0.75):.1f})"

def n_pct(series):
    s = series.dropna()
    return f"{int(s.sum())} ({s.mean()*100:.1f}%)"

n_r  = int(cohort['readmit_30d'].sum())
n_nr = len(cohort) - n_r
r    = cohort[cohort['readmit_30d'] == 1]
nr   = cohort[cohort['readmit_30d'] == 0]

print(f"\n  Readmitted (n):          {n_r:,}   → fill TABLE 1 header")
print(f"  Not Readmitted (n):      {n_nr:,}   → fill TABLE 1 header")
print(f"\n  Age, median (IQR):")
print(f"    Overall:   {med_iqr(cohort['anchor_age'])}")
print(f"    Readmit:   {med_iqr(r['anchor_age'])}")
print(f"    No readm:  {med_iqr(nr['anchor_age'])}")

cohort['is_female'] = (cohort['gender'] == 'F').astype(float)
print(f"\n  Female sex, n (%):")
print(f"    Overall:   {n_pct(cohort['is_female'])}")
print(f"    Readmit:   {n_pct(cohort[cohort['readmit_30d']==1]['is_female'])}")
print(f"    No readm:  {n_pct(cohort[cohort['readmit_30d']==0]['is_female'])}")

adm_types = admissions[['hadm_id', 'admission_type']].drop_duplicates()
cohort2 = cohort.merge(adm_types, on='hadm_id', how='left')
cohort2['is_emergency'] = cohort2['admission_type'].str.contains(
    'EMERGENCY|URGENT', case=False, na=False).astype(float)
print(f"\n  Emergency/Urgent, n (%):")
print(f"    Overall:   {n_pct(cohort2['is_emergency'])}")
print(f"    Readmit:   {n_pct(cohort2[cohort2['readmit_30d']==1]['is_emergency'])}")
print(f"    No readm:  {n_pct(cohort2[cohort2['readmit_30d']==0]['is_emergency'])}")

# ── LACE elements ──────────────────────────────────────────────────────────
lace_path = f"{MIMIC_DIR}/lace_scores_all.csv"
if not os.path.exists(lace_path):
    lace_path = f"{FULL_RUN}/lace_scores_all.csv"
if os.path.exists(lace_path):
    lace = pd.read_csv(lace_path, dtype={'subject_id': str, 'hadm_id': str})
    print(f"\n  [LACE] rows loaded: {len(lace)}, columns: {lace.columns.tolist()}")
    lace_cohort = cohort[['subject_id', 'hadm_id', 'readmit_30d']].merge(
        lace, on=['subject_id', 'hadm_id'], how='inner')
    print(f"  Matched {len(lace_cohort)} of {len(cohort)} cohort patients to LACE scores")
    lace_r  = lace_cohort[lace_cohort['readmit_30d'] == 1]
    lace_nr = lace_cohort[lace_cohort['readmit_30d'] == 0]
    for col in lace.columns:
        if col in ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'readmit_30d']:
            continue
        try:
            s  = pd.to_numeric(lace_cohort[col], errors='coerce')
            sr = pd.to_numeric(lace_r[col], errors='coerce')
            sn = pd.to_numeric(lace_nr[col], errors='coerce')
            if s.notna().sum() > 0:
                print(f"\n  {col}:")
                print(f"    Overall:   {med_iqr(s)}")
                print(f"    Readmit:   {med_iqr(sr)}")
                print(f"    No readm:  {med_iqr(sn)}")
        except Exception:
            pass
else:
    print(f"  [LACE] file not found")

# ── TABLE 2: Mining summary ────────────────────────────────────────────────
print("\n── TABLE 2: MINING SUMMARY ──")

t1_dict = dict(zip(t1['Characteristic'], t1['Value']))
mean_seq  = t1_dict.get('Sequence length, mean (SD)', '[not found]')
mean_sal  = t1_dict.get('IG-salient seq length, mean (SD)', '[not found]')
print(f"  Mean sequence length (tokens):   {mean_seq}")
print(f"  Mean salient tokens (80% IG):    {mean_sal}")

try:
    full_mean = float(str(mean_seq).split(' ')[0])
    sal_mean  = float(str(mean_sal).split(' ')[0])
    ratio = sal_mean / full_mean * 100
    print(f"  Compression ratio:               {ratio:.1f}%")
except Exception:
    print("  Compression ratio:               [could not compute]")

mine_files = glob.glob(f"{MINE_DIR}/all_scored_*.csv")
if mine_files:
    mine_file = sorted(mine_files)[-1]
    scored = pd.read_csv(mine_file)
    print(f"\n  [Mining file: {os.path.basename(mine_file)}]")
    print(f"  Total scored patterns:           {len(scored)}")
    multi = scored[scored['pattern'].str.contains(' -> ', na=False)]
    single = scored[~scored['pattern'].str.contains(' -> ', na=False)]
    print(f"  Single-token patterns:           {len(single)}")
    print(f"  Cross-visit patterns (≥2 steps): {len(multi)}")
    print(f"  Columns available: {scored.columns.tolist()}")
else:
    print(f"  [Mining] No scored CSV found in {MINE_DIR}")

pathway_file = f"{CSV_DIR}/fig3_pathway_importance_test.csv"
# Fallback: panel_c contains the deduplicated pathway list used for fig 3
if not os.path.exists(pathway_file):
    alt = f"{MAIN_DIR}/panel_c_test_j20_short.csv"
    if os.path.exists(alt):
        pathway_file = alt
if os.path.exists(pathway_file):
    pw = pd.read_csv(pathway_file)
    print(f"\n  [Pathway file: {os.path.relpath(pathway_file, FULL_RUN)}]")
    print(f"  Columns: {pw.columns.tolist()}")
    pat_col = 'pattern' if 'pattern' in pw.columns else ('pattern_readable' if 'pattern_readable' in pw.columns else None)
    ig_col  = next((c for c in ['ig_signed_mean', 'sig_ig_mean', 'mean_ig_signed', 'signed_ig', 'original_sig_ig'] if c in pw.columns), None)
    if pat_col:
        pw_multi = pw[pw[pat_col].astype(str).str.contains(' -> ', na=False)]
    else:
        pw_multi = pd.DataFrame()
    print(f"\n  After Jaccard dedup (j>=0.2):")
    print(f"    Total pathway patterns:        {len(pw)}")
    print(f"    Cross-visit patterns:          {len(pw_multi)}")
    if ig_col:
        pw_risk = pw[pw[ig_col] > 0]
        pw_prot = pw[pw[ig_col] < 0]
        print(f"    Risk patterns (ΣIG > 0):       {len(pw_risk)}  (col={ig_col})")
        print(f"    Protective patterns (ΣIG < 0): {len(pw_prot)}")

# ── TABLE 3: Phenotype clusters ────────────────────────────────────────────
print("\n── TABLE 3: PHENOTYPE CLUSTER VALUES ──")

dumbbell_file = f"{MAIN_DIR}/fig4_dumbbell_test_j20_short.csv"
if os.path.exists(dumbbell_file):
    db = pd.read_csv(dumbbell_file)
    print(f"  [Dumbbell file: {os.path.basename(dumbbell_file)}]")
    print(f"  Columns: {db.columns.tolist()}")
    label_col = next((c for c in ['label', 'pattern_readable', 'pattern'] if c in db.columns), None)
    sig_col   = next((c for c in ['original_sig_ig', 'sig_ig', 'ig_signed_mean'] if c in db.columns), None)
    delta_col = next((c for c in ['delta', 'delta_prob', 'delta_y'] if c in db.columns), None)
    imp_col   = next((c for c in ['imputed', 'is_imputed'] if c in db.columns), None)
    print(f"\n  {'Label':<50} {'ΣIG':>10} {'Δŷ':>10} {'Imputed':>8}")
    print(f"  {'-'*80}")
    for _, r in db.iterrows():
        lbl = str(r.get(label_col, ''))[:48]
        sig = f"{r[sig_col]:.4f}"   if sig_col   and pd.notna(r[sig_col])   else '[?]'
        dlt = f"{r[delta_col]:+.4f}" if delta_col and pd.notna(r[delta_col]) else '[?]'
        imp = str(r[imp_col])        if imp_col   and pd.notna(r[imp_col])   else ''
        print(f"  {lbl:<50} {sig:>10} {dlt:>10} {imp:>8}")
else:
    print(f"  [Dumbbell CSV not found]: {dumbbell_file}")

# Pattern carriers
if os.path.exists(pathway_file):
    pw = pd.read_csv(pathway_file)
    n_col  = next((c for c in ['n_present', 'n_carriers', 'n_pos', 'support'] if c in pw.columns), None)
    lbl_col = 'pattern_readable' if 'pattern_readable' in pw.columns else ('label' if 'label' in pw.columns else 'pattern')
    ig_col = next((c for c in ['ig_signed_mean', 'sig_ig_mean', 'mean_ig_signed', 'original_sig_ig'] if c in pw.columns), None)
    if n_col:
        print(f"\n  Top 20 patterns by {ig_col or 'order'}  (n carriers column: {n_col}):")
        print(f"  {'Pattern':<60} {'n':>8}")
        print(f"  {'-'*70}")
        ordered = pw.sort_values(ig_col, ascending=False) if ig_col else pw
        for _, r in ordered.head(20).iterrows():
            lbl = str(r[lbl_col])[:58]
            print(f"  {lbl:<60} {int(r[n_col]):>8}")

# ── Bootstrap CIs on model predictions ─────────────────────────────────────
print("\n── BOOTSTRAP CIs on model AUROC ──")
pred_csv = f"{FULL_RUN}/test_predictions.csv"
if os.path.exists(pred_csv):
    pp = pd.read_csv(pred_csv)
    print(f"  [Loaded] {pred_csv}  shape={pp.shape}  cols={pp.columns.tolist()}")
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_true = pp['y_true'].values
    y_score = pp['y_prob'].values if 'y_prob' in pp.columns else pp['y_score'].values
    rng = np.random.RandomState(42)
    n_boot = 1000
    aurocs, praucs = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yb = y_true[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        aurocs.append(roc_auc_score(yb, y_score[idx]))
        praucs.append(average_precision_score(yb, y_score[idx]))
    print(f"  AUROC:  {np.mean(aurocs):.4f}  (95% CI: {np.percentile(aurocs,2.5):.4f}–{np.percentile(aurocs,97.5):.4f})")
    print(f"  PR-AUC: {np.mean(praucs):.4f}  (95% CI: {np.percentile(praucs,2.5):.4f}–{np.percentile(praucs,97.5):.4f})")
else:
    print(f"  [skip] {pred_csv} not found — run scripts/infer_test.py first to generate predictions")

print("\n" + "=" * 70)
print("DONE — copy values above into AToP_paper_v6.docx")
print("=" * 70)
