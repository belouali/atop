"""build_tables_only_docx, quick-review document with only the tables.

Generates ``AToP_JAMIA_tables_only.docx`` containing all 4 main-paper
tables plus Table S2, with their captions and nothing else (no prose,
no figures). Convenient for table-focused review without scrolling
through the whole manuscript.

All table builders mirror the corresponding code in
``build_revised_main_docx.py`` and ``build_revised_supplement_docx.py``
so the output here is byte-for-byte the same content the manuscript
contains; only the surrounding prose is stripped.

Usage:
    python scripts/build_tables_only_docx.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _docx_helpers import (
    new_document, title, heading, para,
    table_title, table_legend, make_table, page_break,
)


REPO_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUBMISSION_DIR = os.path.join(REPO_ROOT, "..", "manuscript", "jamia_submission")
TABLES_DIR     = os.path.join(SUBMISSION_DIR, "tables")
OUT_PATH       = os.path.join(SUBMISSION_DIR, "AToP_JAMIA_tables_only.docx")


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(TABLES_DIR, name))


def _pheno_label_with_count(pheno: str, totals_path: str) -> str:
    """Return just the phenotype name. Aggregate N / readmit% / RR
    across the cherry-picked 5 patterns would be misleading — these are
    not clinical population counts but unions of arbitrary pattern
    carrier sets. Per-pattern readmit% + RR live in the row columns
    instead (totals_path is kept in the signature for API compatibility
    but is no longer read)."""
    return pheno


# ── NaN-safe formatting helpers ──
def _fmt_num(v, fmt: str, default: str = "n/a") -> str:
    if pd.isna(v):
        return default
    return f"{v:{fmt}}"


def _fmt_int(v, default: str = "n/a") -> str:
    if pd.isna(v):
        return default
    return f"{int(v):,}"


def _fmt_str(v, default: str = "") -> str:
    if pd.isna(v):
        return default
    return str(v)

# ── Split "Other" into protective / risk for display ──
def _split_other(df_in, top_n=5, phen_order=None):
    df = df_in[df_in["delta"].notna()].copy()
    df["_abs"] = df["ig_value"].abs()
    df["render_phenotype"] = df.apply(
        lambda r: ("Other (risk-supporting)" if r["ig_value"] > 0
                   else "Other (protective)") if r["phenotype"] == "Other"
                  else r["phenotype"],
        axis=1,
    )
    display_order = list(phen_order or [])
    if "Other" in display_order:
        idx = display_order.index("Other")
        display_order = (display_order[:idx]
                          + ["Other (risk-supporting)", "Other (protective)"]
                          + display_order[idx + 1:])
    df["_pord"] = df["render_phenotype"].apply(
        lambda p: display_order.index(p) if p in display_order else 99)
    df = (df.sort_values(["_pord", "_abs"], ascending=[True, False])
            .groupby("render_phenotype", sort=False, group_keys=False)
            .head(top_n)
            .reset_index(drop=True))
    return df


PHENO_ORDER = ["Psychiatric", "Oncologic", "Obstetric/perinatal",
                "Hepatobiliary", "Substance use", "Cardiopulmonary",
                "Chest pain workup", "Other"]


def build() -> None:
    doc = new_document()
    title(doc, "AToP JAMIA submission, tables for quick review")
    para(doc,
         "This document contains all main-paper tables (Tables 1-4) and "
         "Table S2 from the supplement, with their captions, and nothing "
         "else. Generated from the same CSVs the manuscript uses, so "
         "content here matches the manuscript exactly.",
         italic=True)

    # ── Table 1 ──
    df1 = _read_csv("Table1_cohort.csv")
    df1_meta = _read_csv("Table1_cohort_meta.csv").iloc[0]
    n_tot = int(df1_meta["n_total"])
    n_r = int(df1_meta["n_readmitted"])
    n_nr = int(df1_meta["n_not_readmitted"])
    pct_r = float(df1_meta["pct_readmitted"])
    pct_nr = float(df1_meta["pct_not_readmitted"])
    table_title(doc, "Table 1", "Cohort characteristics.")
    make_table(doc,
                header_row=[
                    "Variable",
                    f"Overall (N={n_tot:,})",
                    f"Readmitted (N={n_r:,}, {pct_r:.1f}%)",
                    f"Not readmitted (N={n_nr:,}, {pct_nr:.1f}%)",
                    "SMD",
                ],
                data_rows=[[_fmt_str(r["variable"]),
                             _fmt_str(r["overall"]),
                             _fmt_str(r["readmitted"]),
                             _fmt_str(r["not_readmitted"]),
                             _fmt_str(r.get("smd", ""))]
                            for _, r in df1.iterrows()],
                col_widths_in=[2.12, 1.31, 1.31, 1.37, 0.50],
                font_size_pt=9)
    table_legend(doc,
        abbreviations=(
            "SMD = standardized mean difference between readmitted and "
            "not-readmitted groups; IQR = interquartile range; "
            "CCI = Charlson Comorbidity Index; ED = Emergency Department; "
            "LACE = Length-of-stay + Acuity-of-admission + CCI + "
            "ED-visits-in-past-6-months risk score; "
            "n (%) = count and percentage of the column total."
        ),
        note=(
            f"MIMIC-IV v3.1 cohort, N = {n_tot:,} patients with at least "
            f"one acute admission; unplanned 30-day readmission rate "
            f"{pct_r:.1f}%. Demographic and acute-encounter values are "
            "reported per index admission; longitudinal covariates "
            "(CCI, prior ED visits, number of admissions) are computed "
            "per patient history up to and including the index admission. "
            "Race, insurance, and admission type are collapsed from the "
            "raw MIMIC-IV categorical values into the broader groups "
            "shown. SMD > 0.1 indicates a meaningful between-group "
            "imbalance (sample-size invariant alternative to p-values, "
            "which are uninformative at this N)."
        ),
    )
    page_break(doc)

    # ── Table 2 (Model performance, promoted from old Table S2) ──
    df2 = _read_csv("Table2_model_performance.csv")
    table_title(doc, "Table 2",
                  "Predictive performance: Transformer vs LACE-covariate baselines.")
    make_table(doc,
                header_row=list(df2.columns),
                data_rows=[[_fmt_str(v) for v in row]
                            for row in df2.itertuples(index=False)],
                col_widths_in=[1.85, 1.40, 1.40, 1.40, 0.55],
                font_size_pt=9)
    table_legend(doc,
        abbreviations=(
            "AUROC = area under the receiver operating characteristic "
            "curve; PR-AUC = area under the precision-recall curve; "
            "CI = confidence interval; ΔAUROC = pairwise AUROC "
            "difference; LACE covariates = the four LACE components plus structured covariates "
            "(length of stay, Charlson comorbidity index, ED visits in "
            "past 6 months, age, sex, prior admissions); LR = logistic "
            "regression; XGBoost = extreme gradient boosting."
        ),
        note=(
            "Test cohort N = 43,640 patients; outcome prevalence = "
            "12.7% (a random classifier achieves PR-AUC ≈ 0.127). "
            "All 95% CIs computed by a paired bootstrap with "
            "**B = 1,000 resamples** (seed = 42) at the patient level, "
            "using the **percentile method** (2.5th, 97.5th). p-values "
            "are two-sided from the bootstrap distribution of ΔAUROC; "
            "values reported as “<0.001” when no resample crosses zero. "
            "The two baselines also differ from each other: LACE "
            "XGBoost vs LACE LR, ΔAUROC = +0.069 (+0.063, +0.076), "
            "p < 0.001. Pattern mining pipeline counts are detailed in "
            "Table S2."
        ),
    )
    page_break(doc)

    # ── Table 3 ──
    df3 = _read_csv("Table3_full_pool_clusters.csv")
    df3 = _split_other(df3, top_n=5, phen_order=PHENO_ORDER)
    table_title(doc, "Table 3",
                  "Top patterns by phenotype family, all patterns.")
    _t3_totals = os.path.join(TABLES_DIR,
                                "Table3_full_pool_clusters_family_totals.csv")
    _rows_t3 = []
    _last_pheno = None
    for _, r in df3.iterrows():
        pheno = _fmt_str(r["render_phenotype"])
        pheno_show = (_pheno_label_with_count(pheno, _t3_totals)
                      if pheno != _last_pheno else "")
        _last_pheno = pheno
        _rows_t3.append([
            pheno_show,
            _fmt_str(r["label"]),
            _fmt_num(r["ig_value"], "+.3f"),
            _fmt_int(r["n_present"]),
            _fmt_num(r["delta"], "+.3f"),
        ])
    make_table(doc,
                header_row=["Phenotype", "Pattern (representative)",
                             "ΣIG", "n", "Δŷ"],
                data_rows=_rows_t3,
                col_widths_in=[1.5, 2.85, 0.6, 0.45, 0.6],
                font_size_pt=9)
    table_legend(doc,
        abbreviations=(
            "ΣIG = test-set carrier-averaged signed sum of "
            "constituent-token IG values (logit space); "
            "Δŷ = mean perturbation effect on predicted readmission "
            "probability after masking the pattern's tokens "
            "(ŷ_masked − ŷ_original, probability space); "
            "n = number of test-cohort carriers of the pattern; "
            "IG = integrated gradients. Crude per-pattern readmission "
            "rates and cohort-baseline relative risks are not shown; see "
            "Supplementary Table S3 for visit-matched, within-cohort "
            "adjusted odds ratios."
        ),
        note=(
            "Patterns are Jaccard ≥ 0.2 deduplicated, **organized "
            "by phenotype family**; **top 5 per family** ranked by "
            "|ΣIG|. Drawn from the all patterns (no shape "
            "minimums). All displayed patterns have completed "
            "perturbation validation (mean Δŷ from masking). The "
            "cross-visit subset is reported in Table 4."
        ),
    )
    page_break(doc)

    # ── Table 4 ──
    df4 = _read_csv("Table4_cross_visit_clusters.csv")
    df4 = _split_other(df4, top_n=5, phen_order=PHENO_ORDER)
    table_title(doc, "Table 4",
                  "Representative cross-visit pattern phenotypes.")
    _t4_totals = os.path.join(TABLES_DIR,
                                "Table4_cross_visit_clusters_family_totals.csv")
    _rows_t4 = []
    _last_pheno = None
    for _, r in df4.iterrows():
        pheno = _fmt_str(r["render_phenotype"])
        pheno_show = (_pheno_label_with_count(pheno, _t4_totals)
                      if pheno != _last_pheno else "")
        _last_pheno = pheno
        _rows_t4.append([
            pheno_show,
            _fmt_str(r["label"]),
            _fmt_num(r["ig_value"], "+.3f"),
            _fmt_int(r["n_present"]),
            _fmt_num(r["delta"], "+.3f"),
        ])
    make_table(doc,
                header_row=["Phenotype", "Pattern (representative)",
                             "ΣIG", "n", "Δŷ"],
                data_rows=_rows_t4,
                col_widths_in=[1.5, 2.85, 0.6, 0.45, 0.6],
                font_size_pt=9)
    table_legend(doc,
        abbreviations=(
            "Same as Table 3: ΣIG = carrier-averaged signed sum of "
            "constituent-token IG values (logit space); "
            "Δŷ = mean perturbation effect on predicted readmission "
            "probability after masking (probability space); "
            "n = number of test-cohort carriers of the pattern; "
            "IG = integrated gradients. As in Table 3, crude readmission "
            "rates and cohort-baseline relative risks are not shown; see "
            "Supplementary Table S3 for visit-matched adjusted odds ratios."
        ),
        note=(
            "Same structure as Table 3 but drawn from the **cross-"
            "visit pool** (≥2 visits, ≥2 tokens; Jaccard ≥ 0.2 dedup, "
            "top 5 per family by |ΣIG|). The cross-visit lock isolates "
            "**temporal-trajectory evidence** across admissions. These "
            "patterns carry ~3× smaller |ΣIG| than Table 3, the cost "
            "the cross-visit lock pays for longitudinal interpretability. "
            "All displayed patterns have completed perturbation "
            "validation (mean Δŷ from masking)."
        ),
    )
    page_break(doc)

    # ── Table S2 (Mining summary, demoted from old Table 2) ──
    heading(doc, "Supplement, Table S2", level=1)
    df_s1 = _read_csv("Table2_mining_summary.csv")
    table_title(doc, "Table S2", "AToP pattern mining pipeline summary.")
    _rows_s1 = []
    _last_s1 = None
    for _, r in df_s1.iterrows():
        if r["section"] != _last_s1:
            _rows_s1.append([_fmt_str(r["section"]), ""])
            _last_s1 = r["section"]
        _rows_s1.append([f"    {_fmt_str(r['metric'])}", _fmt_str(r["value"])])
    make_table(doc,
                header_row=["Statistic", "Count"],
                data_rows=_rows_s1,
                col_widths_in=[4.5, 1.8],
                font_size_pt=9)
    table_legend(doc,
        abbreviations=(
            "IG = integrated gradients; ΣIG = sum of token IG values "
            "for a pattern; SD = standard deviation."
        ),
        note=(
            "Statistics computed over training-cohort admissions; values "
            "shown as mean (SD) where applicable. **Token compression "
            "ratio** = (mean full sequence length) / (mean salient "
            "subsequence length after the 80% IG-mass cutoff), per "
            "patient. Pattern counts are cumulative through each filter "
            "stage at minimum-support 0.001 per class, with Jaccard "
            "deduplication ≥ 0.2 on test-set carrier sets. **Risk** "
            "patterns: ΣIG > 0 (push prediction toward readmission). "
            "**Protective** patterns: ΣIG < 0 (push prediction away "
            "from readmission). Model performance (AUROC, PR-AUC, "
            "ΔAUROC vs baselines) is reported in the main-paper "
            "Table 2."
        ),
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    doc.save(OUT_PATH)
    print(f"✓ Wrote {OUT_PATH}")
    print(f"  Tables: {len(doc.tables)}")


if __name__ == "__main__":
    build()
