"""make_table2_mining_summary, Table 2 (AToP pattern mining summary).

Recomputes the per-stage AToP pipeline statistics from the current run-dir
caches. Single-column structure (mining stats are split-specific by section,
not by column, the v1 train/test layout produced confusing empty cells):

  Section                              | Value
  -------------------------------------+--------
  Attribution (training cohort)        |
    Patients with computed IG
    Mean sequence length (tokens)
    Mean salient tokens (80% IG mass)
    Compression ratio (full / salient)
  Pattern mining (training cohort)     |
    Episodes mined (post-min-support)
    Cross-visit subset (>=2 blocks)
    Full-pool patterns scored on test
  Jaccard deduplication (j = 0.20)     |
    Cross-visit representatives
    Full-pool representatives
  Pattern evaluation (test cohort)     |
    Risk reps (sigma-IG > 0)
    Protective reps (sigma-IG < 0)
    Validated patterns (masking)
    Cohort SHAP cache size

Output: ``manuscript/jamia_submission/tables/Table2_mining_summary.csv``

Usage:
    python scripts/make_table2_mining_summary.py --run_dir ../runs/full_CPD_v2
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.run_layout import RunLayout
from atop.explain.jaccard import get_jaccard_representatives


SPECIAL = {"[CLS]", "[VISIT]", "[PAD]"}


def _attribution_stats(layout: RunLayout, split: str) -> dict:
    """Cohort attribution stats. Returns counts, mean (SD) seq length, mean
    (SD) salient token count, and compression ratio (full / salient)."""
    with open(layout.sequences_pkl(split), "rb") as f:
        df_seq = pickle.load(f)
    seq_lens = df_seq["combined_full"].apply(
        lambda x: sum(1 for t in x if t not in SPECIAL))

    df_ig = pd.read_csv(layout.ig_csv(split),
                          dtype={"patient_id": str, "index_hadm_id": str})
    n_salient_per_patient = []
    for (_, _), g in df_ig.groupby(["patient_id", "index_hadm_id"]):
        v = g["ig_abs"].sort_values(ascending=False).values
        if len(v) == 0 or v.sum() == 0:
            continue
        cum = 0.0
        target = 0.8 * v.sum()
        for i, x in enumerate(v, start=1):
            cum += x
            if cum >= target:
                n_salient_per_patient.append(i)
                break

    sal = pd.Series(n_salient_per_patient) if n_salient_per_patient else pd.Series()
    return {
        "n_patients":   int(len(df_seq)),
        "mean_full":    float(seq_lens.mean()),
        "std_full":     float(seq_lens.std()),
        "mean_salient": float(sal.mean()) if len(sal) else float("nan"),
        "std_salient":  float(sal.std()) if len(sal) else float("nan"),
        "compression":  float(seq_lens.mean() / sal.mean()) if len(sal) and sal.mean() else float("nan"),
    }


def _categorize_pattern(pat: str) -> str:
    """Classify a pattern by shape:
       - single_token: 1 token in 1 visit block, e.g. 'A'
       - single_visit_cooccurrence: >=2 tokens within 1 visit block, e.g. '{A; B}'
       - cross_visit: >=2 visit blocks, e.g. 'A -> B' or '{A; B} -> {C}'
    """
    blocks = pat.split(" -> ")
    if len(blocks) > 1:
        return "cross_visit"
    return "single_visit_cooccurrence" if blocks[0].strip().startswith("{") \
           else "single_token"


def _mining_counts(layout: RunLayout, split: str,
                    jaccard_threshold: float = 0.2) -> dict:
    """Pattern counts at each pipeline stage, with shape-category breakdown."""
    scored_files = sorted([f for f in os.listdir(layout.mining_dir)
                           if f.startswith("all_scored_") and f.endswith(".csv")])
    if not scored_files:
        sys.exit(f"  [ERROR] No all_scored_*.csv in {layout.mining_dir}")
    df_scored = pd.read_csv(os.path.join(layout.mining_dir, scored_files[-1]))
    n_total = int(len(df_scored))

    # Shape-category breakdown of the mined pool, supports the manuscript
    # observation that single-visit co-occurrence dominates without locks.
    df_scored["_cat"] = df_scored["pattern"].apply(_categorize_pattern)
    cat_counts = df_scored["_cat"].value_counts().to_dict()
    n_single_token = int(cat_counts.get("single_token", 0))
    n_single_visit_cooc = int(cat_counts.get("single_visit_cooccurrence", 0))
    n_cross_visit = int(cat_counts.get("cross_visit", 0))

    # Full-pool count (every distinct pattern in the raw mining cache scored
    # against the test cohort; built by ``score_all_raw_patterns``)
    full_pool_pkl = os.path.join(layout.analysis_dir,
                                   f"full_pool_scored_{split}.pkl")
    if os.path.exists(full_pool_pkl):
        with open(full_pool_pkl, "rb") as f:
            df_full = pickle.load(f)
        n_full_pool = int(len(df_full))
    else:
        n_full_pool = None

    reps_xv = get_jaccard_representatives(layout.analysis_dir, split, jaccard_threshold)
    n_dedup_xv = len(reps_xv) if reps_xv is not None else None

    # Cross-visit candidate patterns scored on test (pre-dedup), counted with
    # the same filter get_jaccard_representatives applies (has carrier set and
    # n_present >= 3). This is the no-dedup count Figure S7 sweeps.
    n_xvisit_scored = None
    _carr_pkl = os.path.join(layout.analysis_dir, f"carrier_sets_{split}.pkl")
    if os.path.exists(layout.pathway_csv(split)) and os.path.exists(_carr_pkl):
        _dfx = pd.read_csv(layout.pathway_csv(split))
        with open(_carr_pkl, "rb") as f:
            _carr = pickle.load(f)
        _dfx["_ck"] = _dfx["pattern"].map(_carr)
        n_xvisit_scored = int(
            len(_dfx[_dfx["_ck"].notna() & (_dfx["n_present"] >= 3)]))

    # Full-pool reps (the new Panel C source)
    fp_reps_pkl = os.path.join(
        layout.analysis_dir,
        f"full_pool_jaccard_reps_{split}_j{int(jaccard_threshold * 100)}.pkl")
    if os.path.exists(fp_reps_pkl):
        with open(fp_reps_pkl, "rb") as f:
            n_dedup_fp = len(pickle.load(f))
    else:
        n_dedup_fp = None

    if reps_xv is not None and os.path.exists(layout.pathway_csv(split)):
        df_pw = pd.read_csv(layout.pathway_csv(split))
        df_dd = df_pw[df_pw["pattern"].isin(reps_xv)]
        n_risk = int((df_dd["ig_signed_mean"] > 0).sum())
        n_prot = int((df_dd["ig_signed_mean"] < 0).sum())
    else:
        n_risk = n_prot = None

    val_csv = layout.validation_csv(split)
    if os.path.exists(val_csv):
        n_validated = int(len(pd.read_csv(val_csv)))
    else:
        n_validated = None

    cohort_shap_pkl = os.path.join(layout.attribution_dir,
                                     f"cohort_shap_{split}.pkl")
    if os.path.exists(cohort_shap_pkl):
        with open(cohort_shap_pkl, "rb") as f:
            n_shap_patients = len(pickle.load(f))
    else:
        n_shap_patients = None

    return {
        "patterns_min_support":      n_total,
        "n_single_token":            n_single_token,
        "n_single_visit_cooc":       n_single_visit_cooc,
        "n_cross_visit":             n_cross_visit,
        "full_pool_patterns":        n_full_pool,
        "cross_visit_scored":        n_xvisit_scored,
        "patterns_dedup_xvisit":     n_dedup_xv,
        "patterns_dedup_full_pool":  n_dedup_fp,
        "n_risk":                    n_risk,
        "n_protective":              n_prot,
        "n_validated":               n_validated,
        "n_shap_patients":           n_shap_patients,
    }


def _fmt_mean_sd(mean: float, std: float, decimals: int = 1) -> str:
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ({std:.{decimals}f})"


def _fmt_int(n: int | None) -> str:
    return f"{n:,}" if n is not None else "n/a"


def main() -> None:
    p = argparse.ArgumentParser(description="Compute Table 2 (mining summary)")
    p.add_argument("--run_dir", default="../runs/full_CPD_v2")
    p.add_argument("--jaccard_threshold", type=float, default=0.2)
    p.add_argument("--output_csv", default=None,
                   help="Default: manuscript/jamia_submission/tables/"
                        "Table2_mining_summary.csv")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    out_csv = args.output_csv or os.path.join(
        os.path.dirname(__file__), "..", "..",
        "manuscript", "jamia_submission", "tables",
        "Table2_mining_summary.csv")

    print("[1] Attribution stats (training cohort)…")
    attr = _attribution_stats(layout, "train")
    for k, v in attr.items():
        print(f"    {k}: {v}")

    print("\n[2] Mining counts…")
    mining = _mining_counts(layout, "test", args.jaccard_threshold)
    for k, v in mining.items():
        print(f"    {k}: {v}")

    # ── Build CSV in single-value-column format ─────────────────────
    # Curated for manuscript narrative. Drops internal pipeline counts the
    # reader does not need (full-pool intermediate counts, SHAP cache size,
    # patient counts redundant with Table 1). No cross-references to other
    # figures or tables.
    jstr = f"{args.jaccard_threshold:.2f}"
    mine_sec = "Frequent-episode mining (training cohort, min-support 0.001)"
    score_sec = "Candidate patterns scored on test cohort"
    rep_sec = f"Representatives after Jaccard dedup (j = {jstr}, test carriers)"
    rows = [
        # section, metric, value
        ["Attribution (training cohort)",
         "Sequence length, tokens, mean (SD)",
         _fmt_mean_sd(attr["mean_full"], attr["std_full"])],
        ["Attribution (training cohort)",
         "Salient tokens, 80% IG mass, mean (SD)",
         _fmt_mean_sd(attr["mean_salient"], attr["std_salient"])],
        ["Attribution (training cohort)",
         "Token compression ratio (full / salient)",
         f"{attr['compression']:.1f}"],

        [mine_sec, "Episodes mined",
         _fmt_int(mining["patterns_min_support"])],
        [mine_sec, "    Single-token patterns",
         _fmt_int(mining["n_single_token"])],
        [mine_sec, "    Single-visit co-occurrence patterns",
         _fmt_int(mining["n_single_visit_cooc"])],
        [mine_sec, "    Cross-visit patterns",
         _fmt_int(mining["n_cross_visit"])],

        [score_sec, "Full pool (all shapes)",
         _fmt_int(mining["full_pool_patterns"])],
        [score_sec, "    Cross-visit subset",
         _fmt_int(mining["cross_visit_scored"])],

        [rep_sec, "Full pool (all shapes)",
         _fmt_int(mining["patterns_dedup_full_pool"])],
        [rep_sec, "Cross-visit (locked reporting dictionary)",
         _fmt_int(mining["patterns_dedup_xvisit"])],
        [rep_sec, "    Risk, ΣIG > 0",
         _fmt_int(mining["n_risk"])],
        [rep_sec, "    Protective, ΣIG < 0",
         _fmt_int(mining["n_protective"])],
    ]
    df_out = pd.DataFrame(rows, columns=["section", "metric", "value"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[3] Saved → {out_csv}\n")
    # Pretty-print with section dividers
    last = None
    for _, r in df_out.iterrows():
        if r["section"] != last:
            print(f" , {r['section']}")
            last = r["section"]
        print(f"      {r['metric']:<55s}  {r['value']:>14s}")


if __name__ == "__main__":
    main()
