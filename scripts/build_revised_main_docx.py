"""build_revised_main_docx, produce AToP_JAMIA_submission.v2.docx

Comprehensive rebuild of the main manuscript reflecting the post-redesign state:

  * 4-panel 2\u00d72 Fig 3 (atomic top row, compositional bottom row)
  * 2-panel Fig 4 (cross-visit + all patterns dumbbells)
  * Carrier-conditional vs population-weighted aggregation as the
    headline methodological framing (D0)
  * Cross-method validation via cohort SHAP (Fig S11, 25/30 token agreement)
  * Updated Tables 2 (baselines) and 3 (cross-visit phenotype clusters)
  * R1\u2013R7 quantitative findings + D0\u2013D4 discussion entries from
    MANUSCRIPT_OBSERVATIONS.md

All data values are read from the current CSVs in
``manuscript/jamia_submission/tables/`` so the document is self-consistent
with the actually-shipped artifacts. Figures are referenced as captions
+ alt text only; embed the PNGs manually in Word.

Usage:
    python scripts/build_revised_main_docx.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _docx_helpers import (
    new_document, title, heading, para, equation, figure_placeholder,
    table_caption, table_title, table_legend, make_table,
    page_break, hrule,
)


# ── NaN-safe formatting helpers (avoid leaking literal "nan" into cells) ──
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

# ── Phenotype rendering helper: split "Other" into protective / risk ──
# "Other" is a catch-all; showing top-5 by |IG| skews to one direction.
# Split into two display sub-phenotypes so readers see both halves.
def _split_other(df_in, top_n=5, phen_order=None):
    df = df_in[df_in["delta"].notna()].copy()
    df["_abs"] = df["ig_value"].abs()
    # render_phenotype: same as phenotype EXCEPT split Other by sign
    df["render_phenotype"] = df.apply(
        lambda r: ("Other (risk-supporting)" if r["ig_value"] > 0
                   else "Other (protective)") if r["phenotype"] == "Other"
                  else r["phenotype"],
        axis=1,
    )
    display_order = list(phen_order or [])
    # Ensure Other splits appear at the end if phen_order included "Other"
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


SUBMISSION_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                               "manuscript", "jamia_submission")
TABLES_DIR     = os.path.join(SUBMISSION_DIR, "tables")
OUT_PATH       = os.path.join(SUBMISSION_DIR, "AToP_JAMIA_submission.docx")
FIGURES_DIR    = os.path.join(SUBMISSION_DIR, "figures")

# Image paths for embedded figures (None / missing → placeholder rendered).
# Figure 1 stays as placeholder because it is a hand-edited PowerPoint
# (.pptx); user pastes it manually in Word from the deck.
IMAGE_PATHS = {
    "Figure 1": None,
    "Figure 2": os.path.join(FIGURES_DIR, "Figure2.png"),
    "Figure 3": os.path.join(FIGURES_DIR, "Figure3.png"),  # token-level
    "Figure 4": os.path.join(FIGURES_DIR, "Figure4.png"),  # pattern-level
    "Figure 5": os.path.join(FIGURES_DIR, "Figure5.png"),  # validation dumbbell
}


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(TABLES_DIR, name))

def _pheno_label_with_count(pheno: str, totals_path: str) -> str:
    """Return just the phenotype name. Aggregate N / readmit% / RR
    across the cherry-picked 5 patterns would be misleading. these are
    not clinical population counts but unions of arbitrary pattern
    carrier sets. Per-pattern readmit% + RR live in the row columns
    instead (totals_path is kept in the signature for API compatibility
    but is no longer read)."""
    return pheno


def build() -> None:
    doc = new_document()

    # ── Title block ──────────────────────────────────────────────────
    title(doc, "AToP: An Explainability Framework for Transformer-Based "
               "EHR Predictions Using Attribution-Guided Temporal Pattern "
               "Mining")
    para(doc, "Anas Belouali, PhD\u00b9, Hadi Kharrazi, MD, PhD\u00b9\u00b2")
    para(doc, "\u00b9Biomedical Informatics and Data Science, Johns Hopkins "
              "University School of Medicine, Baltimore, MD, USA")
    para(doc, "\u00b2Center for Population Health Information Technology, "
              "Johns Hopkins Bloomberg School of Public Health, Baltimore, MD, USA")
    # Corresponding-author block. JAMIA's title-page checklist lists name,
    # postal address, email and telephone, but published articles print only
    # name, address and email - the telephone is an editorial-office field
    # and is supplied in the cover letter instead.
    para(doc, "**Corresponding author:** Anas Belouali, Biomedical "
              "Informatics and Data Science, Johns Hopkins University "
              "School of Medicine, 2024 East Monument Street, Suite "
              "1-200, Baltimore, MD 21205, USA. "
              "Email: anas.belouali@georgetown.edu. "
              "ORCID: 0000-0002-2780-2500.")
    # Up to five index keywords (JAMIA title-page requirement).
    para(doc, "**Keywords:** explainable artificial intelligence; "
              "electronic health records; feature attribution; temporal "
              "pattern mining; hospital readmission")
    # Word-count statement (JAMIA title-page requirement). JAMIA Research
    # and Applications caps the body at 4,000 words and the structured
    # abstract at 250.
    para(doc, "**Word count:** Abstract 249 (limit 250); Body "
              "(Background through Conclusion, excluding title page, "
              "abstract, references, tables and figure legends) "
              "approximately 3,940 (limit 4,000).")
    hrule(doc)

    # ── Abstract ─────────────────────────────────────────────────────
    heading(doc, "Abstract", level=1)
    para(doc, "**Objective:** To develop and evaluate AToP (Attribution "
              "to Temporal Patterns), a post-hoc framework that converts "
              "per-token Integrated Gradients (IG) from a trained "
              "Transformer into temporal pattern narratives.")
    para(doc, "**Materials and Methods:** AToP was applied to a single-"
              "stream Transformer trained on MIMIC-IV v3.1 (218,196 "
              "patients; 12.6% unplanned 30-day readmission). It mines "
              "frequent episodes from attribution-salient record "
              "segments, aggregates attribution over each pattern's "
              "carriers, and collapses near-duplicates by carrier "
              "overlap. Validation used masking, shuffle and reversed-"
              "order controls, and KernelSHAP replication "
              "(test N = 43,602).")
    para(doc, "**Results:** The Transformer achieved test AUROC 0.77 and "
              "PR-AUC 0.34, exceeding LACE logistic regression (0.66) "
              "and XGBoost (0.73) on richer inputs (not a feature-matched "
              "comparison). The three views differed. Population-mean IG "
              "was dominated by high-prevalence codes with small per-"
              "encounter impact. Carrier-averaged IG, which ranks tokens "
              "by their attribution when present, surfaced rare-but-"
              "decisive psychiatric markers: psychosis, self-harm "
              "overdose, suicidal ideation (SI), severe major depressive "
              "disorder (MDD). Pattern-level attribution revealed cross-"
              "visit trajectories invisible to token-level views: higher-"
              "risk psychiatric sequences ({MDD, SI} \u2192 SI) and "
              "oncologic progressions (cancer pain \u2192 cancer anemia), "
              "alongside lower-risk obstetric and chest-pain-workup "
              "trajectories.")
    para(doc, "**Discussion:** Conditioning on carriers surfaces rare "
              "signals hidden by population averaging, and aggregating "
              "along discovered trajectories represents phenotypes "
              "single-feature views cannot. Pattern values are "
              "attribution allocations, not identified pattern effects. "
              "KernelSHAP agreed at the token level (25/30) but only "
              "partially on cross-visit patterns (7/30).")
    para(doc, "**Conclusion:** AToP converts token-level attribution into "
              "temporal narratives spanning cross-encounter trajectories, "
              "demonstrated for a Transformer readmission model on "
              "MIMIC-IV. Clinician alignment and generalization beyond "
              "this setting remain future work.")

    page_break(doc)

    # ── Background and Significance ───────────────────────────────────
    heading(doc, "Background and Significance", level=1)
    para(doc, "Transformer architectures [1] trained end-to-end on "
              "longitudinal electronic health record (EHR) data, including "
              "BEHRT [2], Med-BERT [3], and successor models, have "
              "produced substantial gains in clinical prediction "
              "performance across a range of tasks including hospital "
              "readmission, in-hospital mortality, and acute "
              "deterioration. These gains are accompanied by an "
              "interpretability gap: the same architectural features "
              "that enable Transformers to capture long-range "
              "dependencies across heterogeneous clinical events also "
              "obscure how those events combine to produce a prediction.")
    para(doc, "Clinicians, regulators, and institutional review "
              "processes increasingly require that model outputs be "
              "accompanied by explanations that are both locally "
              "faithful (which inputs drove a prediction) and globally "
              "coherent (the reasoning patterns the model has learned). "
              "Existing post-hoc methods such as SHAP [5], IG [6] and "
              "DeepLIFT [7], with axiomatic extensions [8] and surveyed "
              "in [9], summarize model behaviour as per-token importance "
              "scores and are typically applied to EHR sequences that "
              "way [10]; they produce flat rankings that do not encode "
              "temporal or combinatorial structure. Attention weights in "
              "recurrent EHR models such as RETAIN [11] are also not "
              "necessarily faithful explanations [12], a claim that "
              "remains contested [13]. These methods are by construction "
              "per-token: they assign importance to individual events, "
              "not to sequences of events.")
    para(doc, "The fundamental limitation of per-token attribution is "
              "that clinical reasoning is inherently temporal: a heart "
              "failure diagnosis followed by a chronic kidney disease "
              "admission is clinically distinct from the same two "
              "diagnoses appearing without a temporal relationship. A "
              "separate literature addresses temporal structure through "
              "sequential pattern mining [14] and pattern-based EHR "
              "phenotyping [15], which discover recurrent trajectories "
              "in event sequences. These two literatures (attribution and pattern mining) have remained "
              "largely disconnected at the EHR-sequence level.")
    para(doc, "We present AToP (Attribution to Temporal Patterns), a "
              "post-hoc explainability framework that connects per-token "
              "attribution to sequential pattern mining for longitudinal "
              "EHR. Our contribution is not a new attribution algorithm; "
              "in this EHR-sequence setting we (i) evaluate both the "
              "population-weighted estimand E[score] and the carrier-"
              "conditional estimand E[score | present], and "
              "(ii) extend attribution from individual tokens to data-"
              "discovered multi-token temporal patterns. We evaluate AToP "
              "on unplanned 30-day hospital readmission using MIMIC-IV "
              "v3.1 [16], a large critical care database that provides "
              "the longitudinal coverage needed for cross-visit pattern "
              "discovery.")

    # ── Objective ────────────────────────────────────────────────────
    heading(doc, "Objective", level=1)
    para(doc, "To develop and evaluate AToP, a post-hoc framework that "
              "converts salient token attributions from a trained "
              "Transformer into structured temporal-pattern explanations.")

    # ── Materials and Methods ─────────────────────────────────────────
    heading(doc, "Materials and Methods", level=1)

    heading(doc, "Data and prediction task.", level=2)
    para(doc, "We used MIMIC-IV v3.1 [16], comprising 218,196 patients "
              "with an unplanned 30-day hospital readmission rate of "
              "12.6%. The prediction target was unplanned 30-day readmission "
              "following the index admission, defined as the patient's "
              "penultimate hospitalization, with all prior admissions "
              "serving as longitudinal history. Each patient's hospital "
              "admissions were assembled "
              "into a single chronological token sequence containing "
              "ICD-9 and ICD-10 diagnoses, ICD-9 and ICD-10 procedures, "
              "and prescription drugs. Visit boundaries were marked "
              "with explicit [VISIT] tokens.")
    para(doc, "**Token retention.** Chronic diagnoses (CCSR Z-codes "
              "and Charlson chronic categories) were retained only at "
              "first appearance or when recorded as primary diagnosis "
              "(`seq_num = 1`); drugs were retained only at first "
              "prescription, encoding treatment-initiation events "
              "rather than ongoing medication lists.")
    para(doc, "**Preprocessing.** Prescription drugs were mapped to "
              "RxNorm ingredient concepts via a two-stage NDC-then-name "
              "pipeline against the RxNav API; tokens below training "
              "frequency min_freq = 2 were dropped. The final 24,005-"
              "token vocabulary (diagnoses, procedures, RxNorm-ingredient "
              "drugs, and [VISIT]/[CLS]/[PAD]) is shared in a single "
              "embedding space across all clinical streams. Per-stage "
              "mapping and vocabulary-composition counts are in "
              "Supplementary Methods.")
    para(doc, "**Elective-readmission exclusion.** A readmission was "
              "excluded from the positive label if the *next* "
              "admission's `admission_type` was ELECTIVE or SURGICAL "
              "SAME DAY ADMISSION (planned chemotherapy, staged "
              "surgery). EMERGENCY, URGENT, and OBSERVATION admissions "
              "remained eligible, leaving 27,508 unplanned 30-day "
              "readmissions in the analytic cohort.")

    heading(doc, "Single-stream Transformer.", level=2)
    para(doc, "A single-stream Transformer encoder [1] (24,005-token vocabulary, "
              "3.5M parameters, 4 attention heads, 2 encoder layers, "
              "embedding dimension 128) was trained with binary cross-entropy "
              "loss and early stopping on validation AUROC. All clinical "
              "token types (diagnoses, procedures, drugs) and the [VISIT] / "
              "[CLS] / [PAD] special tokens share one embedding space, "
              "allowing the attention mechanism to model cross-stream "
              "interactions without architectural priors. The pre-classifier "
              "[CLS] token output is passed through a sigmoid-activated "
              "linear classifier producing the readmission probability \u0177. "
              "The model is trained on 152,737 patients (train), with 21,819 "
              "validation and 43,640 test patients held out. The analytic "
              "test cohort used for IG attribution and all downstream AToP "
              "analyses is N = 43,602 (38 patients excluded for empty "
              "salient-token sets after attribution). The full cohort, "
              "labeling, splits, and this exclusion are laid out in the "
              "CONSORT-style flow of Figure S14.")

    heading(doc, "AToP explanation framework.", level=2)
    para(doc, "We propose AToP as a post-hoc explanation framework that "
              "operates on the trained model in five steps (Figure 1): "
              "(1) compute per-token attribution via Integrated Gradients; "
              "(2) retain the salient tokens (those contributing 80% of the "
              "absolute attribution mass per patient, capped at 50 tokens); "
              "(3) mine episodes from the salient tokens of training-cohort "
              "patients using frequent-episode mining, yielding single-token, "
              "within-visit co-occurrence, and cross-visit patterns "
              "(cross-visit locks are applied at the reporting stage where "
              "noted); "
              "(4) aggregate attributions either at the token level "
              "(carrier-conditional, see below) or at the pattern level "
              "(carrier-averaged \u03a3IG over the data-discovered multi-token "
              "patterns); and "
              "(5) apply Jaccard deduplication on test-cohort carrier sets "
              "to collapse near-redundant patterns into one representative "
              "per phenotype cluster. "
              "The 80% cutoff in step (2) is a tractability heuristic; "
              "sensitivity to \u03c4 is shown in Figure S12.")
    para(doc, "**IG output scale.** IG was integrated against the "
              "model's pre-sigmoid logit, so attributions are in logit "
              "space and sum to the logit shift from the all-[PAD] "
              "reference (Captum completeness [17]); the reference "
              "baseline and its implied risk are detailed in "
              "Supplementary Methods.")
    para(doc, "**Notation.** Throughout, scoreⱼ denotes a per-instance "
              "attribution summary for feature j: signed IG for a token, "
              "or signed Σ IG over the constituent tokens of a "
              "multi-token pattern. Token-level rankings report absolute "
              "magnitudes so bar lengths are comparable across risk and "
              "protective features; pattern-level rankings report signed "
              "Σ IG so direction is preserved. Non-carriers contribute "
              "0 in population-weighted estimands. Full notation is in "
              "Table S5.")

    heading(doc, "Aggregation conventions: from per-instance attribution "
                  "to global importance.", level=2)
    para(doc, "Per-instance attribution methods (SHAP, IG, DeepLIFT, "
              "gradient\u00d7input) produce signed per-feature attributions for "
              "individual predictions. The conventional global ranking is "
              "the mean absolute attribution (1/N) \u03a3\u1d62 |attr\u1d62(j)|, the "
              "population-weighted estimand, which for sparse-binary "
              "tokens reduces to prevalence(j) \u00d7 E[|attr| | present(j)] "
              "and is what SHAP `summary_plot` [5] and Captum-IG defaults "
              "[6] compute. We use carrier to mean a patient who "
              "exhibits the scored feature (a token, or all "
              "constituent tokens of a pattern). AToP relaxes the "
              "prevalence-weighted convention in two stages: (i) at "
              "the token level, the carrier-conditional mean "
              "(mean attribution among carriers) surfaces "
              "rare-but-decisive features that prevalence "
              "weighting dilutes past visibility; (ii) at the pattern "
              "level, the carrier-averaged \u03a3 IG "
              "(mean summed attribution across the pattern's "
              "tokens, averaged over carriers of the whole "
              "pattern) surfaces "
              "compositional phenotypes single-feature views "
              "cannot represent. Token estimand → Figure 3B; "
              "pattern Σ IG → Figure 4 (Panel A: all patterns; "
              "Panel B: cross-visit only).")
    para(doc, "**Estimands.** Setting attribution for non-carriers to zero "
              "(sparse-binary tokens, sensible baseline) gives the "
              "law-of-total-expectation identity")
    equation(doc, "E[|attr|(j)] = P(present(j)) \u00d7 E[|attr|(j) | present(j)]",
             number=2)
    para(doc, "The left-hand side is the population-weighted estimand "
              "(Figure 3A); the right-hand conditional factor is AToP's "
              "carrier-conditional estimand (Figures 3B, 4). The two are "
              "well-defined and not interchangeable: rare features with "
              "large per-carrier impact are visible only in the conditional "
              "view, and high-prevalence features with small per-carrier "
              "impact only in the population view. AToP's preference for "
              "the conditional estimand matches the clinical-narrative goal "
              "of explaining individual predictions, not a claim that "
              "population weighting is wrong.")

    heading(doc, "Pattern mining and cross-visit reporting locks.", level=2)
    para(doc, "Frequent-episode mining was applied to the salient-token "
              "subsequences of training-cohort patients only. Candidate "
              "patterns grow by adding a co-occurring token within an "
              "admission block or appending a token in a later block, "
              "yielding three structural pattern types: single-token, "
              "within-visit co-occurrence, and cross-visit sequences. "
              "Mining used a minimum support fraction of 0.001 per class "
              "and a maximum pattern length of 4 visit blocks. The "
              "mined-pattern count as a function of this threshold, "
              "with the locked threshold annotated, is shown in "
              "Figure S2.")
    para(doc, "The full unrestricted pool is used by the per-patient "
              "explanations (Figure 2) and the all-patterns global ranking "
              "(Figure 4A, Table 3). For the cross-visit-only views "
              "(Figure 4B, Table 4), two reporting locks are applied after "
              "mining: (i) min_steps = 2 (\u22652 visit blocks) and "
              "(ii) min_tokens = 2 (\u22652 tokens). Without these "
              "reporting locks, the carrier-averaged \u03a3IG ranking is "
              "dominated by single-visit co-occurrence patterns and "
              "cross-visit patterns never reach the top-30 (Figure S8), "
              "motivating the cross-visit-only reporting view as a "
              "complement, not a replacement, of the all-patterns view. "
              "All test-set ranking and reporting in this paper is "
              "IG-based (carrier-conditional E[\u03a3IG | present]).")

    heading(doc, "Jaccard deduplication.", level=2)
    para(doc, "Co-occurring clinical conditions generate near-duplicate "
              "patterns that describe the same underlying phenotype. We "
              "computed pairwise Jaccard similarity on test-set carrier "
              "sets and applied greedy clustering at threshold "
              "Jaccard \u2265 0.2 (any two patterns whose carrier sets "
              "overlap by \u226520% are collapsed into one cluster, retaining "
              "the higher-magnitude representative). Sensitivity to the "
              "Jaccard threshold is shown in Figures S7\u2013S8.")
    para(doc, "Patterns are mined exclusively on the training cohort; "
              "deduplication uses test-cohort carrier sets only to "
              "summarize which mined patterns co-occur in the reported "
              "test-cohort landscape, with no model parameter, threshold, "
              "or score tuned on test outcomes (Supplementary Methods).")

    heading(doc, "Validation: token masking, shuffling, reversed-order, "
                  "and cross-method replication.", level=2)
    para(doc, "Four sensitivity analyses were run. (1) **Token "
              "masking:** matched token positions for each top pattern "
              "replaced with PAD; \u0394\u0177 = \u0177_original \u2212 \u0177_masked. "
              "(2) **Within-visit shuffle:** positions permuted within "
              "each admission. (3) **Cross-visit shuffle:** visit "
              "blocks permuted. (4) Reversed-order ablation "
              "(Figure S1; per-pattern numerical results in Tables S2 "
              "and S3): mean signed \u03a3 IG of forward- vs reverse-"
              "matching patients compared. Per-pattern \u0394\u0177 distributions "
              "for masking and the two shuffle controls are shown in "
              "Figure S6. Cross-method replication used KernelSHAP at 50 "
              "coalitions per patient (Figure S11); patient-level IG vs "
              "KernelSHAP agreement is shown in Figure S3. Results "
              "below.")
    para(doc, "Clinical review. The senior author (HK), a physician-"
              "informatician, reviewed the surfaced patterns, phenotype-"
              "family assignments and exemplars for clinical "
              "plausibility. This was an internal, non-blinded review "
              "rather than a clinical utility study [18, 19]; a "
              "prospective clinician evaluation remains necessary and is "
              "out of scope here.")

    # ── Results ──────────────────────────────────────────────────────
    heading(doc, "Results", level=1)

    heading(doc, "Predictive performance.", level=2)
    para(doc, "The Transformer achieved test AUROC 0.766 (95% CI: "
              "0.759\u20130.773) and PR-AUC 0.342, a higher score than a "
              "logistic regression classifier on LACE-derived features [20] "
              "(AUROC 0.659, 95% CI: 0.652\u20130.667) and an XGBoost "
              "classifier on the same features (AUROC 0.728, 95% CI: "
              "0.721\u20130.736). Because the Transformer uses the full "
              "token sequence whereas the baselines use LACE-derived "
              "covariates, this is not a feature-matched architecture "
              "comparison (Limitations). "
              "Pairwise paired-bootstrap \u0394AUROC differences (1,000 "
              "resamples on the aligned test cohort) confirmed all "
              "pairwise differences as statistically significant at p < "
              "0.001 (Table 2).")

    # ── Table 1 ──────
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

    # ── Table 2 (Model performance vs baselines) ──────────
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
            "B = 1,000 resamples (seed = 42) at the patient level, "
            "using the percentile method (2.5th, 97.5th). p-values "
            "are two-sided from the bootstrap distribution of ΔAUROC; "
            "values reported as “<0.001” when no resample crosses zero. "
            "The two baselines also differ from each other: LACE "
            "XGBoost vs LACE LR, ΔAUROC = +0.069 (+0.063, +0.076), "
            "p < 0.001. Pattern mining pipeline counts are detailed in "
            "Table S1."
        ),
    )

    heading(doc, "Single-patient explanation.", level=2)
    para(doc, "To illustrate the per-patient output we show one test "
              "patient with a multi-visit psychiatric trajectory, chosen "
              "because it exercises every stage of the pipeline (multiple "
              "visit blocks, both risk-supporting and protective patterns). "
              "It is an illustrative case, not a performance claim, and is "
              "not representative of the full cohort. AToP "
              "produces a ranked temporal narrative (Figure 2): the salient "
              "tokens (80% IG mass) are grouped by visit block, matched "
              "against the training-mined pattern dictionary, and scored "
              "by summed signed IG. Each pattern in the ranked output is "
              "fully traceable back to the specific tokens, visit blocks, "
              "and IG values that produced its score. Alternate exemplars "
              "spanning psychiatric (true positive), obstetric (true "
              "negative), and oncology (disagreement / false positive) "
              "cases are shown in Figures S4A–C.")
    para(doc, "Pattern attribution is an allocation, not a unique effect. "
              "Matched patterns can share underlying token positions. To "
              "produce an additive waterfall (Figure 2B) without "
              "double-counting, matched patterns are assigned "
              "non-overlapping token positions under a deterministic "
              "rule; residual salient and non-salient positions complete "
              "the additive decomposition, so bar lengths sum to total "
              "\u03a3 IG. This is "
              "a bookkeeping decomposition under a fixed rule rather than "
              "identification of unique pattern-level effects: alternative "
              "allocation rules would shift IG mass between overlapping "
              "patterns, so the per-pattern bar magnitudes should be read "
              "as rule-dependent allocations, not rule-invariant effects.")
    # JAMIA submission: figures uploaded as separate files, NOT embedded
    # inline. Full Figure 2 legend + alt text appears in the
    # "Figure Legends" section at the end of the manuscript (after
    # References). The body text already cites Figure 2 by number above.

    heading(doc, "Token-level global attribution: carrier-conditioning "
                  "surfaces rare-but-decisive features.",
            level=2)
    para(doc, "Figure 3 compares the two atomic-token aggregation "
              "conventions on the test set. Panel A shows the "
              "population-weighted ranking E[IG] = E[IG | present] \u00d7 "
              "prevalence, mathematically equivalent to the standard SHAP "
              "`summary_plot` global feature importance. Panel B shows "
              "AToP's carrier-conditional ranking E[IG | present]. "
              "Panel A's top tokens are dominated by high-prevalence "
              "codes whose individual impact per patient is small (MDD "
              "single ep. NOS, Live birth, Chest pain NOS). Panel B's "
              "top tokens are rare-but-high-attribution (Psychoses NOS, "
              "Acetaminophen overdose / self-harm, Severe MDD, Bipolar "
              "manic episode).")
    para(doc, "The two rankings have only 7 of 30 tokens in common. "
              "This divergence is expected by construction: dropping the "
              "prevalence factor from the identity above necessarily "
              "elevates rare, high-per-carrier features, and we report it "
              "to quantify how large the re-ranking is for this task "
              "rather than to claim a new phenomenon. Carrier-"
              "conditioning surfaces 23 high-impact features that the "
              "population-weighted ranking places past rank 100 "
              "(Figure S9), clinically interpretable psychiatric and "
              "obstetric markers whose prevalence is too small to survive "
              "population weighting.")

    heading(doc, "Pattern-level global attribution: cross-visit patterns "
                  "surface a distinct phenotype family", level=2)
    para(doc, "Figure 4 compares two pattern-level rankings, both using "
              "AToP's carrier-averaged aggregation. Panel A ranks the "
              "all patterns: all three pattern shapes mined "
              "from the salient subsequences are eligible: single-"
              "token patterns (e.g., MDD single ep. NOS), single-visit "
              "co-occurrence patterns (e.g., {Severe MDD; SI}), and "
              "multi-visit patterns (e.g., {MDD; SI} \u2192 {Recurrent "
              "severe MDD; SI}). Panel B ranks the cross-visit "
              "patterns only subset (multi-visit patterns with "
              "\u22652 visit blocks and \u22652 tokens; the canonical AToP "
              "output). The all-patterns ranking is dominated by "
              "single-visit co-occurrence patterns; the cross-visit "
              "ranking surfaces multi-visit psychiatric, oncologic and "
              "obstetric trajectories (Figure 4, Tables 3\u20134).")
    para(doc, "Two empirical observations frame Figure 4. First, the "
              "top-50 representatives of the two rankings have near-"
              "zero overlap (1 of 50): restricting to cross-visit patterns surfaces a "
              "structurally distinct family, not a refinement. Second, "
              "single-visit co-occurrence patterns carry larger raw "
              "magnitudes ({Severe MDD; SI}: \u03a3 IG = +1.51, \u0394\u0177 = +0.27) "
              "than cross-visit patterns ({MDD; SI} \u2192 {Recurrent severe "
              "MDD; SI}: \u03a3 IG = +0.80, \u0394\u0177 = +0.06). Cross-visit patterns "
              "are not bigger movers but they summarize attribution "
              "concentrated on tokens in temporally extended "
              "configurations (Jaccard and shape-minimum sensitivity "
              "in Figures S7, S8).")
    # Figure 4. separate file (see Figure Legends at end of manuscript).

    heading(doc, "Phenotype clusters and validation effects.",
             level=2)
    para(doc, "Table 3 organizes the all-patterns pattern "
              "representatives (Figure 4A) by phenotype family, showing "
              "the top 5 per family ranked by |ΣIG|. Table 4 applies "
              "the cross-visit lock (≥2 visits, ≥2 tokens; Figure 4B) "
              "to surface temporal-trajectory patterns the model relies "
              "on across admissions, the headline AToP claim about "
              "multi-admission evidence.")
    para(doc, "Phenotype families are deterministic presentational "
              "groupings and do not affect mining, attribution or "
              "ranking; the full classification rules are given in the "
              "supplement.")

    # ── Table 3 ─────
    df3 = _read_csv("Table3_full_pool_clusters.csv")
    # Phenotype-organized: keep top 5 patterns per phenotype family
    # by |\u03a3 IG|. Order families by total cohort prevalence
    # (psychiatric first, then oncologic, then obstetric, then chest-
    # pain workup).
    PHENO_ORDER = ["Psychiatric", "Oncologic", "Obstetric",
                    "Chest pain workup"]
    # Restrict to the four named phenotype families; drop
    # Hepatobiliary, Substance use, and Other so the table presents
    # as a clean phenotype-oriented summary of the four headline
    # categories (each with top 5 by |Σ IG|).
    df3 = df3[df3["phenotype"].isin(PHENO_ORDER)].copy()
    # Apply Other-split logic: "Other" becomes two sub-rows
    # ("Other (risk-supporting)" and "Other (protective)"), each
    # showing top-5 by |IG|. All other families: standard top-5 by |IG|.
    df3 = _split_other(df3, top_n=5, phen_order=PHENO_ORDER)
    table_title(doc, "Table 3",
                  "Top patterns by phenotype family, all patterns.")
    _t3_totals = os.path.join(TABLES_DIR, "Table3_full_pool_clusters_family_totals.csv")
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
            "(ŷ_original − ŷ_masked, probability space, so risk-"
            "supporting patterns have Δŷ > 0); "
            "n = number of test-cohort carriers of the pattern; "
            "IG = integrated gradients. Crude readmission rates and "
            "relative risks are deliberately not shown: carriers of "
            "cross-visit patterns are by construction multi-visit "
            "patients, whose baseline readmission rate (27.8%) differs "
            "sharply from the whole-cohort rate (12.7%), so an "
            "unadjusted relative risk is confounded by visit count. "
            "Visit-matched adjusted odds ratios are in Table S6. "
            "Pattern-label glossary (applies to Tables 3 and 4 and "
            "Figures 4 and 5): NOS = not otherwise specified; "
            "SI = suicidal ideation; OD = overdose; MDD = major "
            "depressive disorder; PTSD = post-traumatic stress "
            "disorder; AKI = acute kidney injury; GERD = "
            "gastro-esophageal reflux disease; AFib = atrial "
            "fibrillation; CVC = central venous catheter; "
            "ERCP = endoscopic retrograde cholangiopancreatography; "
            "CHF = congestive heart failure; ep. = episode; "
            "Hx = history of; → denotes temporal order across visits, "
            "{ } denotes co-occurrence within a visit."
        ),
        note=(
            "Patterns are Jaccard ≥ 0.2 deduplicated, organized "
            "by phenotype family; top 5 per family ranked by "
            "|ΣIG|. Drawn from the all patterns (no shape "
            "minimums): single-token, single-visit co-occurrence, "
            "and cross-visit patterns. Phenotype family is from a "
            "rule-based classification on the constituent tokens. All "
            "displayed patterns have completed perturbation validation "
            "(mean Δŷ from masking). The cross-visit subset of these "
            "patterns is reported in Table 4 (Figure 4B), the headline "
            "AToP claim about temporal-trajectory evidence. "
            "Carrier counts (n) vary widely; patterns with small n "
            "(roughly n < 20) are based on few carriers and should be "
            "read as hypothesis-generating rather than established "
            "phenotype findings."
        ),
    )

    # \u2500\u2500 Bridge paragraph + Table 4 (all patterns, phenotype-organized) \u2500
    para(doc, "Table 3 (all patterns) and Table 4 (cross-visit) "
              "tell complementary stories about how phenotype evidence "
              "is organized across the patient timeline. Psychiatric "
              "and oncologic phenotypes appear in both views, "
              "unfolding across admissions and within them. Obstetric "
              "and chest-pain workup phenotypes appear primarily in "
              "Table 3: these care episodes are predominantly single-"
              "encounter (delivery + procedures + medications in one "
              "stay; ED chest-pain workup with ECG, chest X-ray, and "
              "respiratory measurements in one visit), so the cross-"
              "visit lock filters most of them out.")

    # \u2500\u2500 Table 4 (all patterns, phenotype-organized) \u2500\u2500
    df4 = _read_csv("Table4_cross_visit_clusters.csv")
    df4 = df4[df4["phenotype"].isin(PHENO_ORDER)].copy()
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
            "Abbreviations, the pattern-label glossary, the rationale for "
            "omitting crude readmission rates and relative risks, and the "
            "small-n caution are all as given for Table 3."
        ),
        note=(
            "Same structure as Table 3 but drawn from the cross-visit "
            "pool (≥2 visits, ≥2 tokens; Jaccard ≥ 0.2 dedup, top 5 "
            "per family by |ΣIG|). The cross-visit lock filters out "
            "single-visit co-occurrence patterns, isolating "
            "temporal-trajectory evidence the model relies on across "
            "admissions. These patterns carry ~2.4× smaller |ΣIG| than "
            "Table 3 (mean |ΣIG| across the top-30 representatives of "
            "each pool), the effect-size cost the cross-visit lock pays "
            "for longitudinal interpretability."
        ),
    )

    # Figure 5. separate file (see Figure Legends at end of manuscript).

    heading(doc, "Attribution direction vs observed outcome.", level=2)
    para(doc, "Attribution direction is a statement about the model; "
              "epidemiologic association is a statement about the cohort. "
              "Cross-visit patterns can only be carried by patients with "
              "at least two admissions, and those patients are readmitted "
              "far more often than the cohort overall (27.8% vs 12.7%), so "
              "whole-cohort comparison makes nearly every cross-visit "
              "pattern look risk-increasing as an artifact of visit count. "
              "Restricting to patients with two or more visits and "
              "adjusting for the number of visits, attribution direction "
              "and outcome direction agree for 25 of the 30 displayed "
              "cross-visit patterns, and repeat cesarean delivery inverts "
              "from an unadjusted relative risk above 1 to an adjusted "
              "odds ratio of 0.83, consistent with the protective "
              "attribution the model assigns (Table S6; Supplementary "
              "Results). These associations are descriptive and are not "
              "a test of attribution faithfulness.")

    heading(doc, "Faithfulness of the attribution ranking.", level=2)
    para(doc, "Using an ERASER-style [21] progressive-masking protocol "
              "[22] on 500 random test patients, masking AToP's top-5 "
              "tokens changed predicted risk more than masking matched "
              "random salient tokens (mean |\u0394\u0177| 0.078 vs 0.042, "
              "1.9x larger; Wilcoxon signed-rank p < 0.001), with "
              "comprehensiveness and sufficiency curves above random at "
              "every k (Figure S13; Supplementary Results). This confirms "
              "faithfulness to the model, not clinical correctness.")

    heading(doc, "Temporal-order validation.", level=2)
    para(doc, "Because a pattern's ΣIG is an order-invariant sum over its "
              "constituent tokens, the claim that the model uses event "
              "order, not merely co-occurrence, requires a dedicated "
              "order-sensitive test. For each displayed cross-visit "
              "pattern (Table 4) we identified test-cohort patients "
              "matching the forward token order and patients matching the "
              "reversed order, and compared the carrier-averaged signed "
              "ΣIG (Figure S1; per-pattern values in Tables S2 and S3). "
              "Of the 30 displayed cross-visit patterns, 3 are symmetric "
              "(palindromic block sequences for which reversal returns "
              "the same pattern) and are excluded as non-informative; "
              "among the remaining 27, reversing token order reduced the "
              "carrier-averaged |ΣIG| in 26 (96%), and sign was preserved "
              "for all 30. Among the 26 patterns that do attenuate, mean "
              "attenuation is 32% (median 28%); one pattern showed a "
              "small increase under reversal. As an independent control, "
              "permuting the order of whole visit blocks produced only "
              "~6% of the masking effect (Figure S6).")
    para(doc, "Two caveats bound this. The 96% figure describes curated "
              "patterns selected for high |ΣIG|; across the broader "
              "floored sweep of 1,714 cross-visit patterns with at least "
              "20 forward- and 20 reverse-order matches (Figure S1, "
              "Table S2), reversal reduces |ΣIG| in 54%, so order "
              "sensitivity is concentrated in the high-attribution "
              "patterns AToP surfaces rather than uniform across mined "
              "sequences. What holds broadly is sign preservation: 95% "
              "of swept patterns keep their direction under reversal. "
              "Table S2 lists the 30 most strongly attenuating patterns "
              "(attenuation > 84%), where sign is preserved in 15, flips "
              "in 11, and falls to exactly zero in 4 \u2014 that tail is "
              "precisely where reversed-order |ΣIG| approaches zero. With "
              "the block-shuffle control, these analyses indicate the "
              "model's cross-visit attributions depend on event sequence "
              "rather than co-occurrence alone.")

    heading(doc, "Cross-method validation.", level=2)
    para(doc, "To assess whether AToP's findings are properties of the "
              "aggregation convention rather than of the attribution "
              "method, we replicated Figures 3 and 4 with KernelSHAP "
              "substituted for Integrated Gradients on the full test "
              "cohort. Token-level carrier-conditional top-30 rankings "
              "agree on 25 of 30 features. At the pattern level agreement "
              "is 21 of 30 for the all-patterns ranking and 7 of 30 for "
              "the cross-visit ranking; the divergence reflects how the "
              "two methods score sums over composite features, "
              "KernelSHAP redistributing interaction credit while IG sums "
              "per-token contributions additively, rather than a "
              "contradiction of AToP. Disagreement between explanation "
              "methods applied to the same model is itself a "
              "well-documented and largely unresolved problem [23] "
              "(Figure S11; Supplementary Results).")

    heading(doc, "Discussion", level=1)
    para(doc, "**What is new here.** AToP's contribution is not a new "
              "attribution or mining algorithm, but their composition: "
              "carrier-conditional aggregation exposes rare "
              "high-attribution features that the standard convention of "
              "mean absolute attribution across all instances [24] "
              "dilutes past visibility, while attribution-guided temporal "
              "mining changes the explanatory unit from isolated tokens "
              "to multi-visit patterns. Carrier-conditional importance is "
              "not population importance: it quantifies attribution "
              "intensity among patients who exhibit a token or pattern, "
              "whereas the population estimand additionally weights by "
              "prevalence. Token-level findings replicate under "
              "KernelSHAP (25/30, Figure S11), indicating the convention "
              "rather than the algorithm is the operative driver, whereas "
              "pattern-level rankings remain method-dependent (7/30) and "
              "the IG variant is reported as canonical.")
    para(doc, "The cross-visit restriction (min_steps = 2, min_tokens = "
              "2) is a design choice: the all-pattern and cross-visit "
              "rankings answer different questions and their top-50 sets "
              "overlap in 1 of 50. Single-visit patterns carry larger "
              "attribution magnitude (~2.4× Σ IG), whereas cross-visit "
              "patterns provide the longitudinal structure that motivates "
              "the framework. Because a pattern's Σ IG is an "
              "order-invariant sum, temporal interpretation rests on the "
              "reversal and block-shuffle controls rather than on Σ IG "
              "alone.")
    para(doc, "A consistent observation across every aggregation level "
              "is the proliferation of \"unspecified\" (NOS) coding "
              "at the highest-impact ranks (MDD single ep. NOS, Anxiety "
              "NOS, Bipolar NOS, AKI NOS in Figure 3B; {Psychosis NOS; "
              "SI} \u2192 Psychosis NOS and recurrent {MDD single ep. NOS; "
              "SI} sequences in Figure 4). We highlight this as a "
              "hypothesis-generating observation: persistent "
              "diagnostic ambiguity that does not resolve on follow-up "
              "is one possible explanation, but documentation "
              "conventions, billing workflows, and coder behavior are "
              "alternatives. Distinguishing among these requires "
              "service-line-stratified analysis, adjustment for coding "
              "density, and clinician review (all out of scope here) "
              "but natural follow-ups before AToP outputs are used to "
              "trigger clinical workflow changes.")
    para(doc, "Placing multiple clinical streams (diagnoses, procedures, "
              "drugs) in a single embedding space enables AToP to "
              "discover cross-stream temporal patterns invisible to "
              "single-stream methods.")
    para(doc, "We deliberately use the term temporal pattern narratives "
              "rather than clinical pathways to reflect that AToP "
              "discovers sequential structures in model-attributed tokens, "
              "not causal disease pathways. With observational EHR data, "
              "any temporal correlation can reflect the underlying disease "
              "process, the clinical workflow that generates the codes, or "
              "an interaction between the two. AToP's outputs should be "
              "treated as model-behavior summaries to be validated against "
              "domain knowledge, not as discovered causal mechanisms.")
    heading(doc, "Limitations.", level=2)
    para(doc, "AToP is post-hoc: it explains a model that remains a "
              "black box, and for high-stakes decisions an "
              "interpretable-by-design model has been argued preferable "
              "to any post-hoc account of an opaque one [25]. We do not "
              "contest that; AToP is for settings where a sequence model "
              "is already in use. We also evaluate faithfulness to the "
              "model rather than usefulness to a decision-maker, which "
              "are distinct targets [26].")
    para(doc, "Further limitations bound our claims. (i) Evaluation uses "
              "one dataset, task, attribution method and architecture; "
              "portability is not demonstrated. (ii) Perturbation tests "
              "create out-of-distribution inputs, which the shuffle and "
              "reversed-order controls mitigate but do not eliminate. "
              "Our faithfulness evaluation is also partly circular in "
              "pattern space: the patterns submitted to masking are the "
              "ones IG ranked highest, so Figure 5 tests whether the "
              "IG-selected subset behaves as predicted, not whether IG "
              "selected the most influential patterns available. A "
              "non-circular test would mask a ranking-independent sample "
              "of the pattern space; we did not run this, and it is the "
              "natural next step. At the token level the "
              "comprehensiveness and sufficiency curves (Figure S13) do "
              "benchmark against random selection, so the concern is "
              "specific to pattern space. (iii) Pattern-level rankings "
              "remain partly attribution-method dependent (7/30 "
              "cross-visit agreement with SHAP). (iv) Patterns with "
              "small carrier counts (n < 20) are flagged discovery-only; "
              "n is shown for every pattern. (v) Deployment-utility "
              "measures and subgroup analysis are out of scope. That "
              "omission is consequential rather than merely incomplete: "
              "several surfaced patterns involve socioeconomic and "
              "behavioural-health tokens whose attribution may differ "
              "systematically across subgroups, so a carrier-conditional "
              "fairness audit is a priority before any deployment use. "
              "(vi) The 80% saliency-mass threshold is a tractability "
              "heuristic, stable across \u03c4 \u2208 [0.60, 0.95] "
              "(Figure S12) but without a principled rule. (vii) The "
              "Table 2 performance gap reflects feature richness and "
              "architecture jointly; feature-matched ablations are left "
              "to future work.")

    # ── Conclusion ──────────────────────────────────────────────────
    heading(doc, "Conclusion", level=1)
    para(doc, "We presented AToP, a post-hoc framework that derives "
              "temporally structured clinical narratives from per-"
              "token attributions of a Transformer on longitudinal "
              "EHR data. Its contribution is a refinement of the "
              "aggregation convention shared by SHAP, IG, and related "
              "methods: carrier-conditional token aggregation surfaces "
              "rare-but-high-attribution features that prevalence "
              "weighting hides; pattern-level \u03a3 IG aggregation "
              "surfaces multi-visit phenotypes that single-feature "
              "attribution cannot represent. KernelSHAP replication "
              "shows token-level carrier-conditional findings (25/30 "
              "agreement) reflect the aggregation convention rather "
              "than the attribution method; pattern-level rankings are "
              "attribution-method dependent and the IG version is "
              "canonical. AToP was evaluated on a single Transformer, "
              "task, and dataset; empirical portability to other "
              "encoders, methods, tasks, and datasets remains to be "
              "demonstrated.")

    # ── Required statements (between Conclusion and References) ─────
    # JAMIA Author Instructions require these for every Original
    # Research submission. Edit the placeholders below before final
    # submission.
    heading(doc, "Author Contributions", level=1)
    para(doc, "Contributions are reported using the CRediT contributor "
              "roles taxonomy.")
    para(doc, "**Anas Belouali:** Conceptualization; Methodology; "
              "Software; Formal analysis; Investigation; Data curation; "
              "Validation; Visualization; Writing – original draft.")
    para(doc, "**Hadi Kharrazi:** Conceptualization; Supervision; "
              "Project administration; Validation (clinical "
              "plausibility review); Writing – review and editing.")
    para(doc, "Both authors approved the final version of the manuscript "
              "and agree to be accountable for all aspects of the work.")

    heading(doc, "Acknowledgments", level=1)
    para(doc, "The authors thank the MIMIC-IV investigators at the "
              "Massachusetts Institute of Technology Laboratory for "
              "Computational Physiology and Beth Israel Deaconess "
              "Medical Center for curating and making the dataset "
              "publicly available through PhysioNet.")

    heading(doc, "Funding", level=1)
    para(doc, "This work received no specific grant from any funding "
              "agency in the public, commercial, or not-for-profit "
              "sectors.")

    heading(doc, "Conflict of Interest", level=1)
    para(doc, "The authors declare no competing interests relevant to "
              "this work.")

    heading(doc, "Data Availability", level=1)
    para(doc, "The MIMIC-IV v3.1 dataset used in this study is publicly "
              "available through PhysioNet "
              "(https://physionet.org/content/mimiciv/3.1/) following "
              "completion of CITI Data or Specimens Only Research "
              "training and a signed Data Use Agreement. The AToP "
              "framework code is publicly available at "
              "https://github.com/belouali/atop (release v1.0.0, the "
              "version used for this work), including the five-step "
              "pipeline, phenotype classifier, masking-validation "
              "pipeline, preprocessing and RxNorm mapping, model "
              "configuration and training seeds, and all attribution, "
              "mining and plotting scripts. All analytic outputs (CSVs, "
              "figures) referenced in the manuscript can be regenerated "
              "from MIMIC-IV v3.1 using these scripts. No patient-level "
              "data is included in the repository.")

    # ── References ──────────────────────────────────────────────────
    heading(doc, "References", level=1)
    refs = [
        # [1] Transformer architecture
        "Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. "
            "Advances in Neural Information Processing Systems. "
            "2017;30:5998–6008.",
        # [2] BEHRT
        "Li Y, Rao S, Solares JRA, et al. BEHRT: Transformer for Electronic "
            "Health Records. Scientific Reports. 2020;10(1):7155.",
        # [3] Med-BERT
        "Rasmy L, Xiang Y, Xie Z, Tao C, Zhi D. Med-BERT: pretrained "
            "contextualized embeddings on large-scale structured electronic "
            "health records for disease prediction. NPJ Digital Medicine. "
            "2021;4(1):86.",
        # [4] Readmission-prediction task grounding
        "Kansagara D, Englander H, Salanitro A, et al. Risk prediction "
            "models for hospital readmission: a systematic review. JAMA. "
            "2011;306(15):1688–1698.",
        # [5] SHAP
        "Lundberg SM, Lee SI. A unified approach to interpreting model "
            "predictions. Advances in Neural Information Processing "
            "Systems. 2017;30:4765–4774.",
        # [6] Integrated Gradients
        "Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep "
            "networks. Proceedings of the 34th International Conference on "
            "Machine Learning. 2017;70:3319\u20133328.",
        # [7] DeepLIFT
        "Shrikumar A, Greenside P, Kundaje A. Learning important features "
            "through propagating activation differences. Proceedings of the "
            "34th International Conference on Machine Learning. "
            "2017;70:3145–3153.",
        # [8] Expected Gradients
        "Erion G, Janizek JD, Sturmfels P, Lundberg SM, Lee SI. Improving "
            "performance of deep learning models with axiomatic attribution "
            "priors and expected gradients. Nat Mach Intell. "
            "2021;3:620–631.",
        # [9] Attribution survey
        "Samek W, Montavon G, Lapuschkin S, Anders CJ, Müller KR. "
            "Explaining deep neural networks and beyond: a review of "
            "methods and applications. Proceedings of the IEEE. "
            "2021;109(3):247–278.",
        # [10] Per-token attribution applied to EHR sequences (XGBoost+SHAP)
        "Lauritsen SM, Kalor ME, Kongsgaard EL, et al. Early detection of "
            "sepsis utilizing deep learning on electronic health record "
            "event sequences. Artificial Intelligence in Medicine. 2020;104:101820.",
        # [11] RETAIN
        "Choi E, Bahadori MT, Kulas JA, Schuetz A, Stewart WF, Sun J. "
            "RETAIN: an interpretable predictive model for healthcare using "
            "reverse time attention mechanism. Advances in Neural "
            "Information Processing Systems. 2016;29:3504–3512.",
        # [12] Attention is not Explanation
        "Jain S, Wallace BC. Attention is not explanation. Proceedings of "
            "the 2019 Conference of the North American Chapter of the "
            "Association for Computational Linguistics. 2019:3543–3556.",
        # [13] Counterpoint to [14]
        "Wiegreffe S, Pinter Y. Attention is not not explanation. "
            "Proceedings of the 2019 Conference on Empirical Methods in "
            "Natural Language Processing. 2019:11–20.",
        # [14] Frequent-episode mining
        "Mannila H, Toivonen H, Verkamo AI. Discovery of frequent episodes "
            "in event sequences. Data Mining and Knowledge Discovery. "
            "1997;1(3):259\u2013289.",
        # [15] Pattern-based EHR phenotyping
        "Hripcsak G, Albers DJ. Next-generation phenotyping of electronic "
            "health records. Journal of the American Medical Informatics "
            "Association. 2013;20(1):117\u2013121.",
        # [16] MIMIC-IV
        "Johnson AEW, Bulgarelli L, Pollard TJ, et al. MIMIC-IV, a freely "
            "accessible electronic health record dataset. Scientific Data. "
            "2023;10(1):1.",
        # [17] Captum library
        "Kokhlikyan N, Miglani V, Martin M, et al. Captum: A unified and "
            "generic model interpretability library for PyTorch. arXiv:2009.07896, 2020.",
        # [18] What clinicians need from explanations
        "Tonekaboni S, Joshi S, McCradden MD, Goldenberg A. What "
            "clinicians want: contextualizing explainable machine learning "
            "for clinical end use. Proceedings of Machine Learning for "
            "Healthcare. 2019;106:359–380.",
        # [19] Clinical-utility caveat for post-hoc explanation
        "Ghassemi M, Oakden-Rayner L, Beam AL. The false hope of current "
            "approaches to explainable artificial intelligence in health "
            "care. Lancet Digit Health. 2021;3(11):e745–e750.",
        # [20] LACE
        "van Walraven C, Dhalla IA, Bell C, et al. Derivation and "
            "validation of an index to predict early death or unplanned "
            "readmission after discharge from hospital to the community. "
            "CMAJ. 2010;182(6):551\u2013557.",
        # [21] ERASER benchmark (named in the Validation and Discussion text)
        "DeYoung J, Jain S, Rajani NF, et al. ERASER: A benchmark to "
            "evaluate rationalized NLP models. Proceedings of the 58th "
            "Annual Meeting of the Association for Computational "
            "Linguistics. 2020:4443–4458.",
        # [22] Saliency-map sanity checks
        "Adebayo J, Gilmer J, Muelly M, et al. Sanity checks for saliency "
            "maps. Advances in Neural Information Processing Systems. "
            "2018;31:9505–9515.",
        # [23] Disagreement between explanation methods
        "Krishna S, Han T, Gu A, Wu S, Jabbari S, Lakkaraju H. The "
            "disagreement problem in explainable machine learning: a "
            "practitioner's perspective. Transactions on Machine Learning "
            "Research. 2024.",
        # [24] TreeExplainer / global from local
        "Lundberg SM, et al. From local explanations to global understanding "
            "with explainable AI for trees. Nature Machine Intelligence. "
            "2020;2(1):56\u201367.",
        # [25] Interpretable-by-design counterposition
        "Rudin C. Stop explaining black box machine learning models for "
            "high stakes decisions and use interpretable models instead. "
            "Nat Mach Intell. 2019;1:206–215.",
        # [26] Interpretability evaluation framing
        "Doshi-Velez F, Kim B. Towards a rigorous science of interpretable "
            "machine learning. arXiv:1702.08608, 2017.",
    ]
    for i, r in enumerate(refs, 1):
        para(doc, f"[{i}] {r}")

    # ── Figure legends section (separate page) ─────────────────────
    page_break(doc)
    heading(doc, "Figure Legends", level=1)
    para(doc, "**Figure 1.** AToP framework overview. Five-step pipeline: "
              "(1) per-token Integrated Gradients on the trained "
              "Transformer; (2) salient-token selection (the tokens "
              "contributing 80% of per-patient attribution mass); "
              "(3) frequent-episode mining from training-cohort salient "
              "subsequences, with cross-visit locks (\u22652 visits, "
              "\u22652 tokens) applied at reporting; (4) carrier-conditional "
              "token-level and pattern-level \u03a3IG aggregation; and "
              "(5) Jaccard \u2265 0.2 deduplication on test-set carrier "
              "sets, retaining the highest-magnitude representative per "
              "phenotype cluster.")
    para(doc, "Alt text: Schematic flow diagram with five sequential steps "
              "linking the trained Transformer to per-token attributions, "
              "salient-token selection, frequent-episode mining, "
              "carrier-conditional aggregation, and Jaccard-based phenotype "
              "deduplication.")

    # Figure 2
    para(doc, "**Figure 2.** Single-patient AToP explanation. "
              "(A) Admission timeline: each admission shows the salient "
              "tokens it contributed to Panel B's patterns (red \u25b2 "
              "risk-pushing, blue \u25bc protective), plus up to two "
              "additional high-|IG| salient tokens in grey that no named "
              "pattern captured. The index admission (prediction point) is "
              "the last circle; the x-axis is real elapsed days; the "
              "observed outcome is the labelled diamond. (B) Allocated "
              "signed \u03a3 IG (logit space). Each bar is the marginal "
              "\u03a3 IG over (position, token) pairs newly claimed by one "
              "matched pattern under the deterministic non-overlapping "
              "allocation rule; Other-salient and Non-salient bars close "
              "the gap to total \u03a3 IG. Per-token IG values are not "
              "printed because under that rule they do not sum "
              "arithmetically to the bars. Prefixes give visit reach "
              "(V1\u2192V5 cross-visit; V5: single-visit); braces denote "
              "within-admission co-occurrence, arrows cross-admission "
              "progression.")
    para(doc, "Alt text: Two-panel figure. Panel A shows a horizontal visit "
              "timeline with one box per admission, each containing the top "
              "salient tokens for that visit with signed IG values, and a "
              "real-days axis below. Panel B shows a horizontal SHAP-style "
              "chevron waterfall of pattern-level \u0394logit contributions "
              "ordered chronologically, with an other-salient bar and a "
              "non-salient bar at the bottom.")

    # Figure 3 (token-level, was old Panels A+B)
    para(doc, "**Figure 3.** Global token-level attribution on the test "
              "set: population-weighted vs carrier-conditional. "
              "(A) Population mean over all patients, non-carriers scored "
              "0. (B) Mean among carriers only. Both panels show the "
              "top-30 tokens by absolute magnitude. The panels are on "
              "different x-axis scales (Panel A roughly \u00b10.04, Panel B "
              "roughly \u00b11.5): the largest population-weighted magnitude "
              "is 0.040 against 1.139 carrier-conditional, a 28-fold "
              "difference. The population-weighted estimand collapses rare "
              "features that the carrier-conditional estimand preserves.")
    para(doc, "Alt text: Two-panel horizontal bar chart. Left panel "
              "shows population-weighted token attribution; right panel "
              "shows carrier-conditional token attribution. Bars are "
              "red for risk-increasing and blue for protective.")

    # Figure 4 (pattern-level, was old Panels C+D)
    para(doc, "**Figure 4.** Global pattern-level attribution on the test "
              "set: all patterns vs cross-visit only. (A) Carrier-"
              "conditional \u03a3 IG, all patterns (single-token, single-visit "
              "co-occurrence, and cross-visit combined). (B) Carrier-"
              "conditional \u03a3 IG, cross-visit only (\u22652 visit blocks, "
              "\u22652 tokens; the canonical AToP output). Both panels show "
              "top-30 representatives by |\u03a3 IG| after Jaccard \u2265 0.2 "
              "deduplication on carrier-set overlap, scored with the "
              "identical position-level allocation operator and drawn on "
              "one shared x-axis, so bar lengths are directly comparable "
              "across panels: the shorter bars in Panel B are the "
              "~2.4-fold magnitude gap reported in the Results, not a "
              "difference in scale.")
    para(doc, "Alt text: Two-panel horizontal bar chart of pattern-"
              "level signed \u03a3 IG. Left panel: all patterns. "
              "Right panel: cross-visit patterns only. Bars "
              "are red for risk-increasing and blue for protective.")

    # Figure 5 (perturbation dumbbell)
    para(doc, "**Figure 5.** Perturbation validation. (A) Top-30 "
              "all-pattern representatives; (B) top-30 cross-visit "
              "representatives (\u22652 visits, \u22652 tokens); ordering "
              "mirrors Figure 4. Each dumbbell is the effect of masking a "
              "pattern's tokens on predicted readmission probability, "
              "averaged over its test-set carriers. "
              "\u0394\u0177 = \u0177_original \u2212 \u0177_masked, as in Tables 3 "
              "and 4: \u0394\u0177 > 0 (red, right) means masking lowered "
              "predicted risk, so the pattern is risk-supporting; "
              "\u0394\u0177 < 0 (blue, left) means protective. Both panels "
              "share an x-axis (\u00b10.30), ranked by |\u0394\u0177|. Masking "
              "is whitelist-driven, each pattern having been selected by "
              "its IG ranking, so Figure 5 confirms the IG-flagged subset "
              "rather than screening the full pattern space.")
    para(doc, "Alt text: Two-panel dumbbell figure. Each row shows a "
              "colored arrow from zero to the masking effect. Red "
              "dumbbells on the right = risk-supporting patterns "
              "(masking lowered predicted risk, \u0394\u0177 > 0); blue "
              "dumbbells on the left = protective patterns (masking "
              "raised predicted risk, \u0394\u0177 < 0). Left panel = all "
              "patterns (no shape minimums). Right panel = cross-visit "
              "patterns (\u22652 visits, \u22652 tokens).")

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    doc.save(OUT_PATH)
    n_para = len(doc.paragraphs)
    n_table = len(doc.tables)
    print(f"\u2713 Wrote {OUT_PATH}")
    print(f"  Paragraphs: {n_para},  Tables: {n_table}")


if __name__ == "__main__":
    build()
