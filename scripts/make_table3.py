"""make_table3 \u2014 Table 3 (top-15 phenotype clusters table).

Joins one of Fig 3's pattern panels with Fig 4's masking-validation deltas.

Default semantics (``--panel d``):
  Loads ``panel_d_<split>_j<pct>_short.csv`` (cross-visit pool, j=0.2 dedup)
  \u2014 i.e. the canonical AToP claim. This preserves Table 3's original
  meaning under the new 4-panel Fig 3 layout (where what used to be
  "Panel C" is now "Panel D"). Joins with the matching subset of Fig 4
  (rows where ``panel == "A"``).

Alternative (``--panel c``):
  Loads ``panel_c_<split>_j{pct}_short.csv`` (full pool, no shape minimums)
  and joins with Fig 4 ``panel == "B"``. Use this for a Table S2-style
  contrast highlighting the single-visit co-occurrence patterns that
  dominate when no temporal lock is enforced.

Output: ``<run_dir>/7_outputs/tables/table3_phenotype_clusters.csv`` plus
a printed text table.

Dependencies: scripts/make_fig3.py and scripts/make_fig4.py must run first
(they produce the panel CSVs and ``Figure4.csv`` this script joins on).

Usage:
    python scripts/make_table3.py --run_dir ../runs/mimiciv_readmission_30d
    python scripts/make_table3.py --run_dir ../runs/mimiciv_readmission_30d --panel c \\
        --output-csv ../manuscript/jamia_submission/tables/TableS2_full_pool_clusters.csv
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.run_layout import RunLayout


# In the new 2-panel Fig 4: panel "A" = cross-visit (= Fig 3 Panel D),
# panel "B" = full pool (= Fig 3 Panel C).
PANEL_TO_FIG4_LETTER = {"d": "A", "c": "B"}


# ── Phenotype categorization ─────────────────────────────────────────
# Lightweight rule-based grouping from the readable label. Preserves the
# v1 manuscript's grouping (Psychiatric / Oncologic / Obstetric / Substance
# use / Other) so the table still leads with phenotype family. Deliberately
# simple and inspectable; clinical reviewers can re-categorize trivially.
PHENOTYPE_RULES = [
    # Order matters: first match wins. Sequence chosen so the more
    # CLINICALLY SPECIFIC phenotype wins when categories could overlap.
    # "MDD + alcohol" → Psychiatric. Cancer dominates if present (e.g.,
    # pancreatic cancer with biliary obstruction → Oncologic, not
    # Hepatobiliary), since malignancy is the headline diagnosis.
    ("Psychiatric",
     {"MDD", "Severe MDD", "Recurrent severe MDD", "Depression NOS",
      "Mood disorder NOS", "Bipolar", "Manic", "PTSD", "Anxiety",
      "Schizo", "schizo",       # NEW: lowercase catches "Paranoid schizophrenia"
      "Psychosis", "Delusional", "SI", "Suicide",
      "Hx self-harm", "self-harm", "Risperidone", "Olanzapine",
      "Fluoxetine", "Mood disorder", "Brief psychotic"}),
    ("Oncologic",
     {"Cancer", "metastases", "Bone met", "Pancreatic", "Liver met",
      "Lung met", "Mal. pleural", "Ascites", "Chemo", "Cancer pain",
      "Cancer anemia",
      # NEW: catch "Malignant neoplasm of X" / "Secondary malignant
      # neoplasm of X" labels that Cancer/Pancreatic substrings missed.
      "Malignant neoplasm", "malignant neoplasm",
      "neoplasm", "Neoplasm",
      "Antineoplastic", "antineoplastic"}),
    # Hepatobiliary: pure bile-duct / liver / gallbladder disease (no
    # cancer co-occurrence, those go to Oncologic above).
    ("Hepatobiliary",
     {"bile duct", "Bile duct", "biliary", "Biliary", "ERCP",
      "cholangitis", "Cholangitis", "cholecystitis", "Cholecystitis",
      "Cholestasis", "cholestasis", "cirrhosis", "Cirrhosis",
      "Hepatitis", "hepatitis", "Liver failure", "liver failure",
      "Gallbladder", "gallbladder", "Gallstone", "gallstone",
      # NEW: pancreatitis (without cancer co-occurrence)
      "Pancreatitis", "pancreatitis"}),
    ("Obstetric",
     {"Live birth", "C-section", "Dibucaine", "Perineal", "gestation",
      "pregnancy", "Pregnancy", "Oxytocin", "OB lac", "Bupivacaine",
      # NEW: maternal / gestational / preterm-labor wordings + amniotic
      "Maternal", "maternal",
      "Gestational", "gestational",
      "preterm labor", "Preterm labor",
      "Amniotic", "amniotic",
      "multigravida", "Multigravida"}),
    ("Substance use",
     {"Alcohol intox", "Alcohol abuse", "Cannabis", "Benzodiazepine",
      "Acetamin OD", "SSRI OD", "Analgesic OD", "self-poisoning"}),
    # Chest pain workup: chest-pain-centered diagnostic evaluations.
    # Tested AFTER Substance use so that a pattern such as
    # {Alcohol abuse; Resp measurements} is not labelled a chest-pain
    # workup on the strength of a diagnostic-test token alone.
    ("Chest pain workup",
     {"Chest pain NOS", "ECG", "Chest X-ray", "Resp measurements",
      "Atypical chest pain"}),
    # Anything else falls into "Other"
]


def _classify_phenotype(label: str) -> str:
    for name, keywords in PHENOTYPE_RULES:
        for kw in keywords:
            if kw in label:
                return name
    return "Other"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute Table 3 (top-N phenotype clusters)")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--jaccard_threshold", type=float, default=0.2)
    p.add_argument("--panel", default="d", choices=["c", "d"],
                   help="Which Fig 3 panel to source: \"d\" (cross-visit, default \u2014 "
                        "the canonical AToP claim, preserves original Table 3 "
                        "semantics) or \"c\" (full pool, no shape minimums).")
    p.add_argument("--top_n", type=int, default=30,
                   help="Top-N rows by |\u03a3IG| to include (default 30 \u2014 "
                        "matches Fig 3 Panel D and Fig 4 Panel A row counts).")
    p.add_argument("--panels-dir", default=None,
                   help="Default: <run_dir>/7_outputs/panels")
    p.add_argument("--fig4-csv",
                   default="../manuscript/jamia_submission/figures/Figure4.csv",
                   help="The Figure4.csv emitted alongside Figure4.png.")
    p.add_argument("--output-csv", default=None,
                   help="Default: <run_dir>/7_outputs/tables/"
                        "table3_phenotype_clusters.csv")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    panels_dir = args.panels_dir or layout.outputs_panels_dir
    out_csv = args.output_csv or layout.table_csv("table3_phenotype_clusters")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    jt_pct = int(args.jaccard_threshold * 100)
    fig4_letter = PANEL_TO_FIG4_LETTER[args.panel]

    # ── Load source pool (panel CSV or upstream cache, depending on top_n) ──
    panel_path = os.path.join(panels_dir,
                               f"panel_{args.panel}_{args.split}_j{jt_pct}_short.csv")
    if not os.path.exists(panel_path):
        sys.exit(f"  [ERROR] Missing {panel_path}\n  Run scripts/make_fig3.py first.")
    panel_csv = pd.read_csv(panel_path)
    pool_label = "cross-visit pool" if args.panel == "d" else "full pool"

    # If top_n fits in the panel CSV, use it directly. Otherwise pull from
    # the upstream Jaccard-deduplicated cache (full_pool_scored for Panel C,
    # pathway_csv for Panel D) so we have enough patterns.
    if args.top_n <= len(panel_csv):
        pp = panel_csv.copy()
        print(f"[1] Loaded {len(pp)} patterns from {os.path.basename(panel_path)} "
              f"({pool_label})")
    else:
        # Resolve labels via the same canonical resolver Fig 3 uses, so the
        # readable strings match between the figure-bound and table-only rows.
        from atop.explain.pattern_labels import PatternLabelResolver
        resolver = PatternLabelResolver(args.run_dir, verbose=False)

        if args.panel == "c":
            # Full-pool scored cache + Jaccard reps cache
            pool_pkl = os.path.join(layout.analysis_dir,
                                      f"full_pool_scored_{args.split}.pkl")
            reps_pkl = os.path.join(layout.analysis_dir,
                                      f"full_pool_jaccard_reps_{args.split}_j{jt_pct}.pkl")
            if not (os.path.exists(pool_pkl) and os.path.exists(reps_pkl)):
                sys.exit(f"  [ERROR] Missing {pool_pkl} or {reps_pkl}.\n"
                         f"  Re-run scripts/make_fig3.py with --no_jaccard "
                         f"first to materialize the full-pool reps cache.")
            with open(pool_pkl, "rb") as f:
                df_full = pickle.load(f)
            with open(reps_pkl, "rb") as f:
                rep_set = pickle.load(f)
            df_reps = df_full[df_full["pattern"].isin(rep_set)].copy()
        else:
            # Cross-visit: pathway_csv + cross-visit Jaccard reps
            from atop.explain.jaccard import get_jaccard_representatives
            df_pw = pd.read_csv(layout.pathway_csv(args.split))
            df_pw = df_pw[df_pw["n_present"] >= 3].copy()
            rep_set = get_jaccard_representatives(layout.analysis_dir,
                                                    args.split, args.jaccard_threshold)
            if rep_set is None:
                sys.exit("  [ERROR] cross-visit Jaccard reps cache missing")
            df_reps = df_pw[df_pw["pattern"].isin(rep_set)].copy()

        # Build pp DataFrame in panel-CSV schema (pattern, label, ig_value, n_present)
        pp = pd.DataFrame({
            "pattern":   df_reps["pattern"],
            "label":     df_reps["pattern"].apply(
                lambda p: resolver.format_pattern(p, short_names=True)),
            "ig_value":  df_reps["ig_signed_mean"],
            "n_present": df_reps["n_present"],
        })
        print(f"[1] Loaded {len(pp):,} patterns from upstream {pool_label} "
              f"(needed top-{args.top_n}; panel CSV had only {len(panel_csv)} rows)")

    # ── Δŷ source: validation_test.csv (covers everything we validated;
    #    the validation top-up workflow ensured top-50 of each pool has data).
    val_csv = layout.validation_csv(args.split)
    if not os.path.exists(val_csv):
        sys.exit(f"  [ERROR] Missing {val_csv}")
    val = pd.read_csv(val_csv)
    db = val[["pattern", "mean_delta_mask", "n_matched_test"]].copy()
    db = db.rename(columns={"mean_delta_mask": "delta",
                              "n_matched_test": "n_carriers"})
    db["direction"] = db["delta"].apply(
        lambda d: "risk" if d > 0 else ("protective" if d < 0 else "neutral"))
    print(f"[2] Loaded {len(db):,} validation rows from "
          f"{os.path.basename(val_csv)}")

    # ── Join on raw pattern key ──────────────────────────────────────
    merged = pp.merge(
        db[["pattern", "delta", "n_carriers", "direction"]],
        on="pattern", how="left",
    )
    n_missing = merged["delta"].isna().sum()
    if n_missing:
        print(f"  [WARN] {n_missing}/{len(merged)} patterns have no validation "
              f"data \u2014 they'll appear with delta='n/a'.")

    # Trim to top-N by |ΣIG|, then group by phenotype family.
    ig_col = "ig_value"
    merged["_abs"] = merged[ig_col].abs()
    merged = merged.sort_values("_abs", ascending=False).drop(columns=["_abs"])
    if args.top_n and args.top_n < len(merged):
        merged = merged.head(args.top_n).reset_index(drop=True)
        print(f"[3] Trimmed to top-{args.top_n} by |\u03a3IG|.")

    # Load carrier sets so we can compute UNIQUE patient counts per
    # phenotype family (a patient can carry multiple patterns within the
    # family; naive sum overcounts; the right metric is the union of carrier
    # sets across the family's patterns).
    #
    # Two sources are merged because neither covers all pattern shapes:
    #   * ``carrier_sets_<split>.pkl`` covers cross-visit patterns
    #     (Panel D source, ~39,612 keys on the v2 test split)
    #   * ``full_pool_scored_<split>.pkl`` covers all full-pool patterns
    #     including single-visit co-occurrence (Panel C source, with a
    #     min-carriers filter, ~47,818 keys)
    # Whichever source has the pattern, we use; cross-visit takes priority
    # when both have it (carrier_sets_test.pkl is built from sequence
    # matching, slightly more accurate than the cooccurrence cache).
    carrier_sets: dict = {}
    cs_pkl = layout.carrier_sets_pkl(args.split)
    if os.path.exists(cs_pkl):
        with open(cs_pkl, "rb") as f:
            carrier_sets.update(pickle.load(f))
        print(f"[carriers] +{len(carrier_sets):,} from "
              f"{os.path.basename(cs_pkl)}")
    pool_pkl = os.path.join(layout.analysis_dir,
                              f"full_pool_scored_{args.split}.pkl")
    if os.path.exists(pool_pkl):
        with open(pool_pkl, "rb") as f:
            df_pool = pickle.load(f)
        n_added = 0
        for pat, cs in zip(df_pool["pattern"], df_pool["_carrier_keys"]):
            if pat not in carrier_sets:
                carrier_sets[pat] = cs
                n_added += 1
        print(f"[carriers] +{n_added:,} new keys from "
              f"{os.path.basename(pool_pkl)}  "
              f"(total: {len(carrier_sets):,})")
    if not carrier_sets:
        print(f"  [WARN] No carrier-set cache found; per-family unique "
              f"patient counts will be unavailable.")

    # Phenotype family classification + sort by total |ΣIG| within each
    # group so the most-impactful family appears first.
    merged["phenotype"] = merged["label"].apply(_classify_phenotype)
    pheno_order = (merged.groupby("phenotype")["ig_value"]
                   .apply(lambda s: s.abs().sum())
                   .sort_values(ascending=False).index.tolist())
    merged["_pheno_rank"] = merged["phenotype"].apply(pheno_order.index)
    merged["_abs_ig"] = merged["ig_value"].abs()
    merged = (merged.sort_values(["_pheno_rank", "_abs_ig"],
                                   ascending=[True, False])
              .drop(columns=["_pheno_rank", "_abs_ig"])
              .reset_index(drop=True))

    # ── Print phenotype-grouped table ────────────────────────────────────
    sigma_ig = "\u03a3IG"; delta_y = "\u0394\u0177"
    print(f"\n{'Phenotype':<14} {'Pattern':<55} {sigma_ig:>7} "
          f"{'n':>5} {delta_y:>7}  Dir.")
    print("-" * 100)
    last_pheno = None
    for _, r in merged.iterrows():
        delta_v = r.get("delta")
        delta_str = f"{delta_v:+.3f}" if pd.notna(delta_v) else "   n/a"
        direction = r.get("direction") or "n/a"
        pheno_show = r["phenotype"] if r["phenotype"] != last_pheno else ""
        last_pheno = r["phenotype"]
        print(f"{pheno_show:<14} {str(r['label'])[:54]:<55} "
              f"{r[ig_col]:>+7.3f} {int(r['n_present']):>5} "
              f"{delta_str:>7}  {direction}")
    print()
    print(f"  {'Phenotype family':<18} {'patterns':>9}  {'unique patients (test)':>24}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*22}")
    pheno_unique_carriers: dict = {}
    grand_union: set = set()
    for pheno in merged["phenotype"].drop_duplicates():
        sub = merged[merged["phenotype"] == pheno]
        union: set = set()
        for pat in sub["pattern"]:
            union.update(carrier_sets.get(pat, set()))
        pheno_unique_carriers[pheno] = len(union)
        grand_union.update(union)
        print(f"  {pheno:<18} {len(sub):>9}  {len(union):>22,}")
    sum_naive = sum(pheno_unique_carriers.values())
    n_overlap = sum_naive - len(grand_union)
    print(f"  {'-'*18}  {'-'*8}  {'-'*22}")
    print(f"  {'TOTAL (union)':<18} {len(merged):>9}  {len(grand_union):>22,}")
    if n_overlap:
        pct = 100 * n_overlap / sum_naive
        print(f"  inter-family overlap: {n_overlap:,} patients "
              f"({pct:.1f}% of family-sum, {100*n_overlap/len(grand_union):.2f}% "
              f"of true total)")
    else:
        print(f"  inter-family overlap: 0 patients (families are disjoint)")

    # Stash the count map alongside the CSV. Append a TOTAL row with the
    # exact cross-family union count so DOCX builders can display it.
    # Also compute per-family readmit % + relative risk (RR vs test
    # cohort baseline) so the manuscript can show clinically meaningful
    # phenotype-level risk markers instead of just bare carrier counts.
    preds_path = os.path.join(args.run_dir, "2_predictions",
                                "test_predictions.csv")
    sub_label, baseline = {}, None
    if os.path.exists(preds_path):
        preds = pd.read_csv(preds_path, dtype={"subject_id": str})
        sub_label = dict(zip(preds["subject_id"].astype(str),
                              preds["y_true"]))
        baseline = float(preds["y_true"].mean()) if len(preds) else None

    pheno_carrier_subjs: dict = {}
    for pheno in merged["phenotype"].drop_duplicates():
        sub = merged[merged["phenotype"] == pheno]
        subj_union: set = set()
        for pat in sub["pattern"]:
            cs = carrier_sets.get(pat, set())
            subj_union.update(str(s) if not isinstance(s, tuple) else str(s[0])
                                for s in cs)
        pheno_carrier_subjs[pheno] = subj_union

    def _readmit_stats(subjs):
        labels = [sub_label[s] for s in subjs if s in sub_label]
        if not labels or baseline is None:
            return None, None
        pct = sum(labels) / len(labels) * 100.0
        rr = (pct / 100.0) / baseline if baseline > 0 else None
        return pct, rr

    family_csv = os.path.splitext(out_csv)[0] + "_family_totals.csv"
    rows_ft = []
    for p in merged["phenotype"].drop_duplicates():
        subjs = pheno_carrier_subjs.get(p, set())
        pct, rr = _readmit_stats(subjs)
        rows_ft.append({
            "phenotype": p,
            "n_patterns": int((merged["phenotype"] == p).sum()),
            "n_unique_patients": pheno_unique_carriers.get(p, 0),
            "pct_readmitted": round(pct, 1) if pct is not None else None,
            "relative_risk": round(rr, 2) if rr is not None else None,
        })
    # TOTAL row
    grand_subjs: set = set()
    for s in pheno_carrier_subjs.values():
        grand_subjs.update(s)
    pct_tot, rr_tot = _readmit_stats(grand_subjs)
    rows_ft.append({
        "phenotype": "TOTAL (exact union)",
        "n_patterns": int(len(merged)),
        "n_unique_patients": int(len(grand_union)),
        "pct_readmitted": round(pct_tot, 1) if pct_tot is not None else None,
        "relative_risk": round(rr_tot, 2) if rr_tot is not None else None,
    })
    pd.DataFrame(rows_ft).to_csv(family_csv, index=False)
    print(f"  family totals → {family_csv}")
    if baseline is not None:
        print(f"  test-cohort baseline readmit rate: {baseline*100:.2f}%")
        for r in rows_ft:
            if r["pct_readmitted"] is not None:
                print(f"    {r['phenotype']:<28} N={r['n_unique_patients']:>6,}  "
                      f"readmit={r['pct_readmitted']:>5.1f}%  "
                      f"RR={r['relative_risk']:>4.2f}")

    # ── Per-pattern readmit % + RR (vs test-cohort baseline) ─────────
    # For each pattern, look up its carrier set, count how many were
    # readmitted at their index admission, and divide by baseline rate.
    # This is the clinically-meaningful per-pattern signal (NOT the
    # aggregate across 5 cherry-picked patterns, which is misleading).
    if baseline is not None:
        merged["pct_readmitted"] = merged["pattern"].apply(
            lambda p: (lambda subs: (sum(sub_label[s] for s in subs if s in sub_label)
                                       / max(1, sum(1 for s in subs if s in sub_label)))
                                      * 100.0 if subs else None)(
                set(str(s) if not isinstance(s, tuple) else str(s[0])
                     for s in carrier_sets.get(p, set()))))
        merged["relative_risk"] = merged["pct_readmitted"].apply(
            lambda pct: round((pct / 100.0) / baseline, 2)
                         if pct is not None and pd.notna(pct) and baseline > 0
                         else None)
        merged["pct_readmitted"] = merged["pct_readmitted"].apply(
            lambda v: round(v, 1) if v is not None and pd.notna(v) else None)

    # ── Write CSV ────────────────────────────────────────────────────
    out_cols = ["phenotype", "pattern", "label", ig_col, "n_present",
                "delta", "n_carriers", "direction",
                "pct_readmitted", "relative_risk"]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")


if __name__ == "__main__":
    main()
