"""build_revised_main_docx, produce AToP_JAMIA_submission.v2.docx

Comprehensive rebuild of the main manuscript reflecting the post-redesign state:

  * 4-panel 2\u00d72 Fig 3 (atomic top row, compositional bottom row)
  * 2-panel Fig 4 (cross-visit + all patterns dumbbells)
  * Carrier-conditional vs population-weighted aggregation as the
    headline methodological framing (D0)
  * Cross-method validation via cohort SHAP (Figures S12 and S13, 25/30 token agreement)
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
def _split_other(df_in, top_n=5, phen_order=None, split_other=True):
    df = df_in[df_in["delta"].notna()].copy()
    df["_abs"] = df["ig_value"].abs()
    if not split_other:
        df["render_phenotype"] = df["phenotype"]
        order = list(phen_order or [])
        df["_pord"] = df["phenotype"].apply(lambda p: order.index(p) if p in order else 99)
        return (df.sort_values(["_pord", "_abs"], ascending=[True, False])
                  .groupby("phenotype", sort=False, group_keys=False)
                  .head(top_n).reset_index(drop=True))
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
    para(doc, "**Word count:** Abstract 250; Body 3,933.")
    hrule(doc)

    # ── Abstract ─────────────────────────────────────────────────────
    heading(doc, "Abstract", level=1)
    para(doc, "**Objective:** To develop and evaluate AToP (Attribution "
              "to Temporal Patterns), a post-hoc explainability framework "
              "that converts per-token Integrated Gradients (IG) from a "
              "Transformer into structured temporal pattern narratives.")
    para(doc, "**Materials and Methods:** AToP was applied to a single-"
              "stream Transformer trained on MIMIC-IV v3.1 (218,196 "
              "patients; 12.6% unplanned 30-day readmission). It mines "
              "frequent episodes from attribution-salient segments, "
              "aggregates attribution among patients whose records "
              "contain each token or pattern, and collapses near-"
              "duplicates by patient overlap. Validation used masking, "
              "reversed-order analysis, token-level faithfulness tests, "
              "and KernelSHAP comparison in the test cohort.")
    para(doc, "**Results:** The Transformer achieved AUROC 0.77 and "
              "PR-AUC 0.34. Population-weighted attribution emphasized "
              "high-prevalence codes with small per-patient impact. "
              "Carrier-conditional attribution, the mean among patients "
              "with the token, surfaced rare high-attribution psychiatric "
              "markers including psychosis, self-harm overdose, suicidal "
              "ideation (SI), and severe major depressive disorder (MDD). "
              "Pattern-level attribution identified within-visit "
              "co-occurrences and cross-visit trajectories not captured "
              "by token rankings: risk-increasing psychiatric sequences "
              "({MDD, SI} \u2192 SI) and oncologic progressions (cancer "
              "anemia \u2192 ascites), alongside protective obstetric and "
              "chest-pain-workup patterns.")
    para(doc, "**Discussion:** Carrier-conditional aggregation shifts "
              "attention from common codes to rare, highly attributed "
              "markers. At the token level, prominent signals were "
              "psychiatric crisis codes; across visits, patterns included "
              "recurrent depression with suicidal ideation and cancer "
              "anemia followed by ascites. Masking patterns shifted "
              "predictions in the attributed direction, and reversing "
              "visit order often attenuated attribution. Pattern values "
              "remain attribution allocations, and cross-visit rankings "
              "were method dependent.")
    para(doc, "**Conclusion:** AToP turns a Transformer\u2019s attributions "
              "into clinically legible trajectories, showing which "
              "events, in which order, may drive its readmission "
              "predictions.")

    page_break(doc)

    # ── Background and Significance ───────────────────────────────────
    # [A]-[F] are provisional citation labels kept until the reference
    # audit: [A] longitudinal-EHR AI review; [B] structured-EHR
    # foundation-model review; [C] EHRSHOT/CLMBR-T; [D] MOTOR; [E]
    # Sequential Pattern Transformer; [F] Belouali et al., BioData Mining
    # 2026 (doi:10.1186/s13040-026-00608-3, verified). Renumber all
    # references by first appearance once Results and Discussion are final.
    heading(doc, "Background and Significance", level=1)
    para(doc, "Transformer-based models have increasingly been used to "
              "learn directly from longitudinal electronic health record "
              "(EHR) data, from architectures such as BEHRT [1] and "
              "Med-BERT [2] to more recent longitudinal EHR foundation "
              "models [3,4]. These approaches have been applied across "
              "diverse clinical prediction tasks [5,6]. However, the same "
              "capability to model dependencies across heterogeneous "
              "clinical events creates an interpretability gap: how those "
              "events combine over time to produce a prediction remains "
              "difficult to discern.")
    para(doc, "As predictive models move toward clinical use, "
              "clinicians, regulators, and institutional review processes "
              "increasingly emphasize explanations that are locally "
              "faithful, showing which inputs drove an individual "
              "prediction, and globally coherent, showing recurring "
              "patterns in model behavior across patients. Existing "
              "post-hoc attribution methods, including SHAP [7], IG [8], "
              "and DeepLIFT [9], summarize model behavior as token-level "
              "importance scores and, when applied to longitudinal EHR "
              "data, typically produce rankings of individual events "
              "according to their average importance [10–12]. These flat "
              "rankings do not encode temporal or combinatorial "
              "structure. Attention-based models such as RETAIN [13] "
              "offer a partial alternative by reflecting temporal order "
              "through visit- and feature-level importance. However, "
              "the faithfulness of attention weights as explanations "
              "remains debated [14,15].")
    para(doc, "The lack of temporal and combinatorial structure is "
              "especially consequential in longitudinal clinical data, "
              "where the same events may carry different meaning "
              "depending on whether they occur together within an "
              "encounter or sequentially across encounters. Temporal "
              "structure has also been studied through sequential "
              "pattern mining [16], pattern-based EHR phenotyping [17], "
              "and temporal condition-pattern mining in sparse EHR data "
              "[18]. More recently, sequential pattern mining has been "
              "applied to longitudinal healthcare data to identify "
              "clinically relevant diagnostic trajectories [19]. Recent "
              "work has combined sequential pattern mining with "
              "Transformer-based prediction and post-hoc attribution [20], "
              "but attribution-based explanation and temporal pattern "
              "mining have remained largely disconnected in longitudinal "
              "EHR modeling.")
    para(doc, "We present AToP (Attribution to Temporal Patterns), a "
              "post-hoc explainability framework that connects token-"
              "level attribution to sequential pattern mining for "
              "longitudinal EHR. AToP extends token-level attribution to "
              "temporal patterns identified through sequential pattern "
              "mining and distinguishes population-weighted from carrier-"
              "conditional attribution. We evaluate AToP on unplanned "
              "30-day hospital readmission, a prediction task for which "
              "conventional risk models have historically shown modest "
              "discrimination [21], using MIMIC-IV v3.1 [22], which "
              "provides the longitudinal coverage needed for cross-visit "
              "pattern discovery.")

    # ── Objective ────────────────────────────────────────────────────
    heading(doc, "Objective", level=1)
    para(doc, "To develop and evaluate AToP, a post-hoc explainability "
              "framework that uses attribution-guided sequential pattern "
              "mining to transform token-level attribution into "
              "structured temporal patterns in longitudinal EHR data.")

    # ── Materials and Methods ─────────────────────────────────────────
    # House rule: Methods cite no Results or supplementary tables and
    # figures (Figure 1, the framework schematic, is the exception).
    heading(doc, "Materials and Methods", level=1)

    heading(doc, "Data and prediction task", level=2)
    para(doc, 'The study cohort included 218,196 patients from MIMIC-IV '
              'v3.1 [22], with an unplanned 30-day hospital readmission '
              'rate of 12.6%. The prediction target was unplanned 30-day '
              'readmission after the index admission, defined as the '
              'patient’s penultimate hospitalization. All earlier '
              'admissions were used as longitudinal history. For patients'
              ' with a single admission, that admission served as the '
              'index.')
    para(doc, 'Each patient’s longitudinal record through the index '
              'admission was represented as a single chronological token '
              'sequence containing ICD-9 and ICD-10 diagnoses and '
              'procedures, and prescription drugs. ICD-9 codes were '
              'mapped to ICD-10 using the CMS General Equivalence '
              'Mappings [23], retaining only unambiguous one-to-one '
              'mappings; codes without such a mapping were kept in their '
              'original ICD-9 form. Visit boundaries were marked with '
              'explicit [VISIT] tokens.')
    para(doc, 'Chronic conditions such as endocrine, circulatory, '
              'respiratory, renal, gastrointestinal, psychiatric, '
              'musculoskeletal, hepatic, and neurological conditions were'
              ' retained only at first appearance or when recorded as the'
              ' primary diagnosis. A chronic code therefore re-enters the'
              ' sequence only when it is the reason for a later '
              'admission, so that discoverable trajectories reflect '
              'clinically active disease rather than carried-forward '
              'problem lists. Drugs were retained only at first '
              'prescription, encoding treatment-initiation events rather '
              'than ongoing medication lists. Drugs administered in more '
              'than 25% of admissions were excluded to prevent routine '
              'treatments and solutions given in inpatient care from '
              'dominating the mined patterns.')
    para(doc, 'Prescription drugs were mapped to RxNorm ingredient '
              'concepts using the RxNav API, first by National Drug Code '
              '(NDC) and then by drug name when an NDC match was '
              'unavailable. Tokens occurring fewer than twice in the '
              'training set were excluded. The final vocabulary contained'
              ' 24,005 tokens spanning diagnoses, procedures, RxNorm '
              'ingredient-level medications, and [VISIT], [CLS], and '
              '[PAD] tokens. Mapping and vocabulary details are given in '
              'the Supplementary Methods.')
    para(doc, 'Admissions with an in-hospital death (11,801, 2.2%) were '
              'excluded before labeling because patients cannot be '
              'readmitted after death; consequently, an index admission '
              'followed within 30 days only by a readmission ending in '
              'death was labeled as not readmitted (709 of 218,196 index '
              'admissions, 0.32%). A readmission was excluded from the '
              'positive outcome when the subsequent admission was coded '
              'as elective or surgical same-day admission. Emergency, '
              'urgent, and observation admissions remained eligible, '
              'yielding 27,508 unplanned 30-day readmissions.')

    heading(doc, "Single-stream Transformer", level=2)
    para(doc, "A single-stream Transformer encoder [24] with a 24,005-"
              "token vocabulary, 3.5 million parameters, four attention "
              "heads, two encoder layers, and 128-dimensional embeddings "
              "was trained using binary cross-entropy loss with early "
              "stopping based on validation AUROC.")
    para(doc, "Diagnoses, procedures, medications, and the [VISIT], "
              "[CLS], and [PAD] special tokens were represented in a "
              "shared embedding space, allowing the model to learn "
              "interactions across token types without type-specific "
              "architectural constraints. The [CLS] representation was "
              "passed through a linear layer with sigmoid activation to "
              "produce the predicted readmission probability.")
    para(doc, "Patients were split at the patient level into training "
              "(70%; n=152,737), validation (10%; n=21,819), and test "
              "(20%; n=43,640) cohorts. Of the 43,640 patients in the "
              "test cohort, 43,602 were included in IG attribution and "
              "downstream AToP analyses; 38 were excluded because "
              "attribution yielded no salient-token set.")

    heading(doc, "Baselines and performance evaluation", level=2)
    para(doc, 'We compared the Transformer with two baselines trained on '
              'seven structured predictors derived from the LACE index '
              '[25]: the four LACE components (length of stay, emergent '
              'admission, Charlson comorbidity index, and emergency '
              'department visits in the preceding six months), entered as'
              ' raw values rather than LACE points, plus age, sex, and '
              'number of prior admissions. Logistic regression used '
              'standardized predictors with L2 regularization. XGBoost '
              'used 200 trees, a maximum depth of 4, a learning rate of '
              '0.05, and positive-class weighting for class imbalance. '
              'Both baselines were trained on the training cohort and '
              'evaluated on the test cohort, with fixed rather than tuned'
              ' hyperparameters.')
    para(doc, "Discrimination was summarized by the area under the "
              "receiver operating characteristic curve (AUROC) and the "
              "area under the precision-recall curve (PR-AUC). "
              "Differences between models were estimated with a paired "
              "bootstrap of test-cohort patients (1,000 resamples), "
              "reporting percentile 95% confidence intervals and two-"
              "sided p values computed as the proportion of resamples in "
              "which the difference crossed zero.")

    heading(doc, "AToP explanation framework", level=2)
    para(doc, "AToP is a post-hoc explanation framework that operates on "
              "the trained model in five steps (Figure 1): (1) compute "
              "token-level attribution using Integrated Gradients; "
              "(2) retain salient tokens contributing 80% of the absolute "
              "attribution mass per patient, capped at 50 tokens; "
              "(3) apply frequent-episode mining to salient tokens from "
              "the training cohort to identify single-token events, "
              "within-visit co-occurrences, and cross-visit sequences; "
              "(4) summarize attribution at the token and pattern levels "
              "using carrier-conditional aggregation; and (5) apply "
              "Jaccard-based deduplication to test-cohort patient sets to "
              "collapse near-redundant patterns into representative "
              "patterns.")
    para(doc, "Integrated Gradients was calculated using the model output "
              "before conversion to a readmission probability. The "
              "resulting attribution values are therefore on the logit "
              "scale. By the completeness property, the token "
              "attributions sum approximately to the difference between "
              "the patient’s model output and the output for an "
              "all-[PAD] reference sequence [8,26]. Details of the "
              "reference sequence and approximation error are provided in "
              "the Supplementary Methods.")
    para(doc, "For each patient, a token’s attribution score was its "
              "signed IG value, and a multi-token pattern’s "
              "attribution score was the sum of the signed IG values of "
              "its constituent tokens. Both token and pattern rankings "
              "were based on absolute attribution magnitude, with the "
              "sign retained to indicate whether the token or pattern "
              "increased or decreased predicted risk.")
    para(doc, 'We define a carrier as a patient whose record contains the'
              ' token or pattern being evaluated. At the token level, '
              'carrier-conditional attribution was calculated as the mean'
              ' signed IG among patients whose records contained the '
              'token.')
    para(doc, "The population-weighted attribution for token j was "
              "calculated as the mean signed IG across all patients:")
    equation(doc, "Aⱼᵖᵒᵖ = (1/N) Σᵢ IGᵢⱼ",
             number=1)
    para(doc, "Tokens were ranked by the absolute magnitude of "
              "Aⱼᵖᵒᵖ, while its sign indicated whether the "
              "token increased or decreased predicted risk. Because "
              "patients whose records did not contain token j contributed "
              "zero attribution for that token, Equation (1) can also be "
              "written as:")
    equation(doc, "Aⱼᵖᵒᵖ = P(j present) × "
                  "E[IGⱼ | j present]",
             number=2)
    para(doc, "Equation (2) shows that population-weighted attribution "
              "reflects both how often a token occurs and how strongly it "
              "is attributed when present. Carrier-conditional "
              "attribution isolates the latter quantity.")
    para(doc, "At the pattern level, each carrier's score was obtained by"
              " matching the pattern to the patient's record step by "
              'step: each step was assigned the earliest visit containing'
              ' all of its tokens, with later steps required to fall in '
              'strictly later visits. The score summed the signed IG of '
              "each step's tokens in the assigned visit, and scores were "
              'averaged across carriers to give the carrier-averaged ΣIG,'
              ' ranked by |ΣIG| with its sign retained. The per-patient '
              'explanation display uses a separate non-overlapping '
              'allocation of salient positions (Supplementary Methods).')
    para(doc, "Frequent-episode mining [16] was applied only to salient-"
              "token subsequences from the training cohort. Patterns were "
              "constructed by either adding a co-occurring token within "
              "the same visit or appending a token from a later visit, "
              "yielding three pattern types: single-token events, within-"
              "visit co-occurrences, and cross-visit sequences. Mining "
              "used a minimum support fraction of 0.001 within each "
              "outcome class and a maximum length of four visit blocks; "
              "the episodes mined in the two classes were pooled.")
    para(doc, "Every pooled episode with at least five test-cohort "
              "carriers entered the all-patterns ranking, and every pooled "
              "episode spanning at least two visits with at least two "
              "tokens and at least three test-cohort carriers entered the "
              "cross-visit ranking; these reporting criteria did not "
              "constrain mining. Per-patient explanations matched against "
              "the subset of pooled episodes that also reached support "
              "0.001 across the whole training cohort.")
    para(doc, "To reduce near-duplicate patterns, we compared the sets of "
              "test-cohort patients whose records contained each pattern "
              "using Jaccard similarity. Patterns with Jaccard similarity "
              "≥0.20 were grouped using greedy clustering, and the "
              "pattern with the largest absolute attribution magnitude "
              "within each group was retained as the representative.")
    para(doc, 'Test-cohort patient sets were used only to deduplicate '
              'patterns already mined in the training cohort and did not '
              'influence training, mining thresholds, or attribution '
              'scores.')

    heading(doc, "Parameter choices and sensitivity analyses", level=2)
    para(doc, 'AToP parameters were specified as design or tractability '
              'constraints rather than selected by optimizing test-set '
              'outcomes. The exception was the 0.25 maximum '
              'drug-prevalence threshold, which was selected by '
              'inspecting patterns mined from the training cohort, '
              'without using outcome labels or test data. The 0.80 '
              'saliency-mass threshold is a tractability heuristic, and '
              'sensitivity was assessed across thresholds from 0.60 to '
              '0.95; the 50-token cap was a safeguard that was rarely '
              'binding. Integrated Gradients used 30 integration steps, '
              'with completeness error characterized in the Supplementary'
              ' Methods. All parameter values are listed in Table S1.')

    heading(doc, "Validation analyses", level=2)
    para(doc, "We evaluated AToP using token masking, reversed-order "
              "analysis between and within patients, "
              "token-level faithfulness analysis, and cross-method "
              "comparison with KernelSHAP.")
    para(doc, 'Masking was applied to the reported patterns and to '
              'additional high-ranked risk-supporting and protective '
              'patterns selected by absolute ΣIG across pipeline '
              'validation runs, 266 unique patterns in total. For token '
              'masking, token positions corresponding to each pattern '
              'were replaced with [PAD], and the change in predicted '
              'readmission probability was calculated as Δŷ = ŷ_original '
              '− ŷ_masked.')
    para(doc, 'For the reversed-order analysis, we compared '
              'carrier-averaged ΣIG between patients whose records '
              'matched each pattern in the forward order and those whose '
              'records matched the reversed order; because these are '
              'different patients, this measures an association between '
              'event order and attribution. As a within-patient test, for'
              ' up to 80 carriers of each reported cross-visit pattern we'
              " reversed the order of the pattern's matched visits within"
              " the patient's own record, recomputed IG, and rescored the"
              ' pattern.')
    para(doc, 'We applied an ERASER-style progressive-masking protocol '
              '[27] to 500 randomly sampled test patients with at least '
              'three salient tokens. For k = 1 to 10, comprehensiveness '
              'was the absolute change in predicted probability after '
              'replacing the k salient tokens with the largest |IG| by '
              '[PAD], and sufficiency was the change after retaining only'
              ' those k tokens and replacing the patient’s remaining '
              'salient tokens by [PAD]; non-salient and special tokens '
              'were left unchanged. Each was compared with a '
              'matched-random baseline that masked or retained the same '
              'number of tokens drawn at random from the patient’s '
              'salient set, averaged over five draws. Confidence '
              'intervals were obtained by bootstrapping patients (1,000 '
              'resamples), and comprehensiveness at k = 5 was compared '
              'between IG and random selection using a one-sided Wilcoxon'
              ' signed-rank test.')
    para(doc, "As a cross-method comparison, attribution was recomputed "
              "using KernelSHAP with approximately 50 coalitions per "
              "patient in the full analytic test cohort. The same "
              "carrier-conditional token and pattern aggregation "
              "conventions were then applied to compare the resulting "
              "global rankings with those obtained using IG.")

    heading(doc, "Attribution direction and observed outcome", level=2)
    para(doc, "To compare attribution direction with observed readmission "
              "association, we restricted the test cohort to patients "
              "with at least two admissions and, for each reported cross-"
              "visit pattern, fit a logistic regression of readmission on "
              "pattern carriage and the log-transformed number of visits. "
              "We report the adjusted odds ratio for pattern carriage "
              "with its 95% Wald confidence interval. These analyses were "
              "descriptive and were not used as tests of attribution "
              "faithfulness.")

    # ── Results ──────────────────────────────────────────────────────
    heading(doc, "Results", level=1)

    heading(doc, "Predictive performance", level=2)
    para(doc, 'Cohort derivation is shown in Figure S1, and cohort '
              'characteristics are given in Table 1. The Transformer '
              'achieved an AUROC of 0.766 (95% CI, 0.759–0.773) and a '
              'PR-AUC of 0.342, compared with 0.728 (95% CI, 0.721–0.736)'
              ' for XGBoost and 0.659 (95% CI, 0.652–0.667) for logistic '
              'regression on LACE-derived covariates [25]; both '
              'differences from the Transformer were statistically '
              'significant (p < 0.001; Table 2). Because the models used '
              'different inputs, these comparisons do not isolate the '
              'effect of architecture.')

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
            "p < 0.001."
        ),
    )

    heading(doc, "Single-patient explanation", level=2)
    para(doc, 'To illustrate the per-patient output, Figure 2 shows one '
              'test patient with a multi-visit oncologic trajectory: five'
              ' admissions over 911 days, readmitted 2 days after the '
              'index discharge, with a predicted readmission probability '
              'of 0.93. The case was chosen to illustrate every stage of '
              'the pipeline and is not representative of the cohort. The '
              'largest allocation is a cross-visit pattern, cancer pain '
              'at the second admission followed by co-occurring cancer '
              'anemia and cancer pain at the index admission (+0.83 of a '
              'total ΣIG of +0.97), alongside smaller protective patterns'
              ' such as thiamine followed by cancer anemia (−0.08). '
              'Alternate exemplars (psychiatric true positive, obstetric '
              'true negative, oncology false positive) are shown in '
              'Figures S2A–C.')
    para(doc, 'Pattern attribution is an allocation, not a unique effect:'
              ' matched patterns can share token positions, so the '
              'waterfall (Figure 2B) assigns each salient position to one'
              ' pattern under a fixed priority rule (Supplementary '
              'Methods), and bar lengths are rule-dependent allocations.')
    # JAMIA submission: figures uploaded as separate files, NOT embedded
    # inline. Full Figure 2 legend + alt text appears in the
    # "Figure Legends" section at the end of the manuscript (after
    # References). The body text already cites Figure 2 by number above.

    heading(doc, "Token-level global attribution: carrier-conditioning "
                  "surfaces rare-but-decisive features",
            level=2)
    para(doc, 'Figure 3 compares the two atomic-token aggregation '
              'conventions on the test set. Panel A shows the '
              'population-weighted ranking (Equation 1). Panel B shows '
              "AToP's carrier-conditional ranking E[IG | present]. Panel "
              "A's top tokens are dominated by high-prevalence codes "
              'whose individual impact per patient is small (MDD single '
              'ep. NOS, i.e. not otherwise specified, Live birth, Chest '
              "pain NOS). Panel B's top tokens are "
              'rare-but-high-attribution (Psychoses NOS, Acetaminophen '
              'overdose / self-harm, Severe MDD, Bipolar manic episode).')
    para(doc, 'The rankings overlap for only 7 of 30 tokens; the other '
              '23 carrier-conditional top-30 tokens fall outside the '
              'population-weighted top 30, and 21 of them rank below '
              '100 (Figure S3).')

    heading(doc, "Pattern-level global attribution: cross-visit patterns "
                  "surface a distinct phenotype family", level=2)
    para(doc, 'Pattern mining counts at each pipeline stage are given in '
              'Table S2. Figure 4 compares two pattern-level rankings '
              'based on carrier-averaged ΣIG: all patterns (Panel A) and '
              'cross-visit patterns only (Panel B). The all-patterns '
              'ranking is dominated by single-visit co-occurrences such '
              'as {Severe MDD; SI}, whereas the cross-visit ranking '
              'surfaces multi-visit psychiatric, oncologic, and obstetric'
              ' trajectories such as {MDD; SI} → {Recurrent severe MDD; '
              'SI} (Tables 3–4).')
    para(doc, 'The top-50 sets of the two rankings overlap in only 1 of '
              '50. Single-visit patterns have about 2.4-fold larger '
              'attribution magnitudes, whereas cross-visit patterns '
              'capture temporally extended configurations (Figures '
              'S4–S5). Population weighting penalizes multi-token '
              'patterns most, because a pattern cannot be more prevalent '
              'than its rarest token; its top-30 retains only 6 of the '
              'carrier-conditional 30 (Figure S6). Attribution is only '
              'partially explained by prevalence and crude relative risk '
              '(Figure S7).')
    # Figure 4. separate file (see Figure Legends at end of manuscript).

    heading(doc, "Phenotype clusters and masking effects",
             level=2)
    para(doc, 'Table 3 groups the all-patterns representatives by '
              'phenotype family (top 5 per family by |ΣIG|), and Table 4 '
              'does the same for cross-visit patterns. Masking each '
              "displayed pattern's tokens moved the prediction in the "
              'expected direction (Figure 5), as it did for all 266 '
              'perturbed patterns (Figure S8).')
    para(doc, 'Phenotype families are presentational groupings defined by'
              ' keyword rules (Supplementary Methods) and do not affect '
              'mining, attribution, or ranking.')

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
            "Visit-matched adjusted odds ratios are in Table S3. "
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
            "Top 5 patterns per phenotype family by |ΣIG|, after "
            "Jaccard ≥ 0.2 deduplication, drawn from the all-patterns "
            "pool (single-token, single-visit co-occurrence and "
            "cross-visit shapes; no shape minimums)."
        ),
    )

    # \u2500\u2500 Bridge paragraph + Table 4 (all patterns, phenotype-organized) \u2500
    para(doc, 'Psychiatric and oncologic phenotypes appear in both '
              'tables, within and across admissions. Obstetric and '
              'chest-pain workup phenotypes appear mainly in Table 3 '
              'because these care episodes are usually completed within '
              'one encounter, so the cross-visit criteria exclude most of'
              ' them.')

    # \u2500\u2500 Table 4 (all patterns, phenotype-organized) \u2500\u2500
    df4 = _read_csv("Table4_cross_visit_clusters.csv")
    # Table 4 keeps six families (the four of Table 3 plus Hepatobiliary and
    # Other, Other not split by sign): these 30 patterns are the set analysed
    # in the reversal and adjusted-odds-ratio tables (Tables S3 and S5).
    PHENO_ORDER_T4 = PHENO_ORDER + ["Hepatobiliary", "Other"]
    df4 = df4[df4["phenotype"].isin(PHENO_ORDER_T4)].copy()
    df4 = _split_other(df4, top_n=5, phen_order=PHENO_ORDER_T4, split_other=False)
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
            "As given for Table 3, including the pattern-label glossary, "
            "the rationale for omitting crude readmission rates and "
            "relative risks, and the small-n caution."
        ),
        note=(
            "As Table 3 but drawn from the cross-visit pool "
            "(≥2 visits, ≥2 tokens), with two further families "
            "(Hepatobiliary and Other) for 30 patterns in total."
        ),
    )

    # Figure 5. separate file (see Figure Legends at end of manuscript).

    heading(doc, "Attribution direction vs observed outcome", level=2)
    para(doc, 'Cross-visit patterns can only occur in patients with at '
              'least two admissions, who are readmitted more often than '
              'the cohort overall (27.8% vs 12.7%), so unadjusted '
              'comparisons make nearly every cross-visit pattern look '
              'risk-increasing. After restriction and adjustment for '
              'visit count, attribution direction and outcome direction '
              'agreed for 25 of the 30 displayed cross-visit patterns, '
              'and repeat cesarean delivery inverted from an unadjusted '
              'relative risk above 1 to an adjusted odds ratio of 0.83, '
              'consistent with its protective attribution (Table S3; '
              'Supplementary Results).')

    heading(doc, "Faithfulness of the attribution ranking", level=2)
    para(doc, 'Masking the five tokens with the largest |IG| changed '
              'predicted risk more than masking matched random salient '
              'tokens (mean |Δŷ| 0.079 vs 0.042; Wilcoxon signed-rank p <'
              ' 0.001), and comprehensiveness and sufficiency exceeded '
              'the random baseline at every k (Figure S9; Supplementary '
              'Results). These results indicate faithfulness to the '
              'model, not clinical correctness.')

    heading(doc, "Temporal-order analysis", level=2)
    para(doc, 'Of the 30 cross-visit patterns in Table 4, 3 are '
              'palindromic. Among the remaining 27, carriers of the '
              'reversed order had lower carrier-averaged |ΣIG| than '
              'carriers of the forward order for 26 (96%; mean '
              'attenuation 32%, median 28%), with the same sign of the '
              'mean in both groups for all 30 (Figure S10; Tables S4 and '
              'S5). Because the two groups are different patients, this '
              'is an association between event order and attribution that'
              ' may also reflect other differences between them.')
    para(doc, 'Across 1,714 cross-visit patterns with at least 20 '
              'forward- and 20 reverse-order matches, reversal reduced '
              '|ΣIG| in 54% and preserved sign in 95% (Figure S10; Table '
              'S4), so the association is concentrated in the '
              'high-attribution patterns AToP surfaces. In a '
              'within-patient visit-reordering analysis, reversing the '
              "order of the matched visits inside each carrier's own "
              'record reduced |ΣIG| for 21 of 27 non-palindromic patterns'
              ' (mean attenuation 14%, median 17%) and for 59% of the 722'
              ' distinct carriers (patient-level one-sided Wilcoxon p < '
              '0.001; 9 of 27 patterns nominally significant, 4 after '
              'Holm correction; Table S6). The effect was smaller than in'
              ' the between-patient comparison and provides additional '
              'evidence that attribution is sensitive to visit order; '
              'because whole visits are moved, the analysis does not '
              "isolate the order of the pattern's own events.")

    heading(doc, "Cross-method comparison", level=2)
    para(doc, 'With KernelSHAP in place of IG on the full test cohort, '
              'per-patient agreement is shown in Figure S11, and '
              'carrier-conditional token top-30 rankings agreed on 25 of '
              '30 features (Figure S12). At the pattern level, agreement '
              'was 21 of 30 for the all-patterns ranking and 7 of 30 for '
              'the cross-visit ranking (Figure S13; Supplementary '
              'Results). A likely contributor is that '
              'KernelSHAP redistributes interaction credit, whereas IG '
              'sums per-token contributions additively. Disagreement '
              'between explanation methods applied to the same model is a'
              ' well-documented and largely unresolved problem [28].')

    heading(doc, "Discussion", level=1)
    para(doc, '**What is new here.** AToP combines carrier-conditional '
              'aggregation with attribution-guided temporal mining, '
              'shifting explanation from isolated tokens to multi-visit '
              'patterns rather than only re-weighting population-averaged'
              ' token importance, the global summary commonly reported '
              'with SHAP [29]. Token-level findings were largely '
              'consistent under KernelSHAP (25/30; Figure S12), whereas '
              'cross-visit pattern rankings remain attribution-method '
              'dependent (7/30); we therefore report the IG '
              'implementation as canonical.')
    para(doc, 'The cross-visit restriction is a design choice: the '
              'all-pattern and cross-visit rankings answer different '
              'questions and their top-50 sets overlap in 1 of 50. '
              'Single-visit patterns carry larger attribution magnitude '
              '(~2.4× Σ IG), whereas cross-visit patterns provide the '
              'longitudinal structure that motivates the framework. '
              "Because a pattern's Σ IG is an order-invariant sum, "
              'temporal interpretation rests on the reversal analysis '
              'rather than on Σ IG alone.')
    para(doc, 'A consistent observation across every aggregation level is'
              ' the proliferation of "unspecified" (NOS) coding at the '
              'highest-impact ranks (MDD single ep. NOS, Anxiety NOS, '
              'Bipolar NOS, AKI NOS in Figure 3B; {Psychosis NOS; SI} → '
              'Psychosis NOS and recurrent {MDD single ep. NOS; SI} '
              'sequences in Figure 4). We highlight this as a '
              'hypothesis-generating observation: persistent diagnostic '
              'ambiguity that does not resolve on follow-up is one '
              'possible explanation, but documentation conventions, '
              'billing workflows, and coder behavior are alternatives. '
              'Distinguishing among these requires service-line '
              'stratification, adjustment for coding density, and '
              'clinician review, which are natural follow-ups before AToP'
              ' outputs inform workflow changes.')
    para(doc, "We deliberately use the term temporal pattern narratives "
              "rather than clinical pathways to reflect that AToP "
              "discovers sequential structures in model-attributed tokens, "
              "not causal disease pathways. With observational EHR data, "
              "any temporal correlation can reflect the underlying disease "
              "process, the clinical workflow that generates the codes, or "
              "an interaction between the two. AToP's outputs should be "
              "treated as model-behavior summaries to be validated against "
              "domain knowledge, not as discovered causal mechanisms.")
    heading(doc, "Limitations", level=2)
    para(doc, "AToP is post-hoc: it explains a model that remains a "
              "black box, and for high-stakes decisions an "
              "interpretable-by-design model has been argued preferable "
              "to any post-hoc account of an opaque one [30]. We do not "
              "contest that; AToP is for settings where a sequence model "
              "is already in use. We also evaluate faithfulness to the "
              "model rather than usefulness to a decision-maker, which "
              "are distinct targets [31].")
    para(doc, 'Further limitations bound our claims. (i) Evaluation uses '
              'one dataset, task, attribution method and architecture; '
              'portability is not demonstrated. (ii) Masking creates '
              'out-of-distribution inputs [32], so masking effects may '
              'partly reflect deletion itself; the between-patient '
              'reversal analysis compares observed records without '
              'perturbing them, whereas the within-patient test reorders '
              'visits and shares this caveat. Our pattern-level '
              'faithfulness evaluation is also partly circular: the '
              'masked patterns are the ones IG ranked highest, so Figure '
              '5 tests whether the IG-selected subset behaves as '
              'predicted, not whether IG selected the most influential '
              'patterns; a ranking-independent sample of the pattern '
              'space is the natural next step. At the token level, the '
              'comprehensiveness and sufficiency curves (Figure S9) '
              'benchmark against random selection. (iii) Pattern-level '
              'rankings remain partly attribution-method dependent (7/30 '
              'cross-visit agreement with SHAP). (iv) Patterns with small'
              ' carrier counts (n < 20) are flagged discovery-only; n is '
              'shown for every pattern. (v) Deployment-utility measures '
              'and subgroup analysis are out of scope; several surfaced '
              'patterns involve socioeconomic and behavioural-health '
              'tokens whose attribution may differ across subgroups, so a'
              ' carrier-conditional fairness audit should precede any '
              'deployment use [33]. (vi) The 80% saliency-mass threshold '
              'is a tractability heuristic; risk-pattern rankings are '
              'stable across τ ∈ [0.60, 0.95] (top-30 overlap ≥ 0.67), '
              'protective-pattern rankings less so (0.20 between the '
              'extremes; Figure S14), and the threshold has no principled'
              ' rule. (vii) The Table 2 performance gap reflects feature '
              'richness and architecture jointly; feature-matched '
              'ablations are left to future work. (viii) In-hospital '
              'deaths were excluded before labeling, so readmissions that'
              ' ended in death were not counted as readmissions (0.32% of'
              ' index admissions; 144 in the test cohort), and death was '
              'not modeled as a competing outcome.')

    # ── Conclusion ──────────────────────────────────────────────────
    heading(doc, "Conclusion", level=1)
    para(doc, 'We presented AToP, a post-hoc framework that derives '
              'temporally structured clinical narratives from per-token '
              'attributions of a Transformer on longitudinal EHR data. '
              'Its contribution lies in how per-instance attributions '
              'from IG and related methods are aggregated: '
              'carrier-conditional token aggregation surfaces '
              'rare-but-high-attribution features that prevalence '
              'weighting hides; pattern-level ΣIG aggregation surfaces '
              'multi-visit phenotypes that single-feature attribution '
              'cannot represent. Comparison with KernelSHAP indicates '
              'that the token-level carrier-conditional findings (25/30 '
              'agreement) are not specific to IG; pattern-level rankings '
              'are attribution-method dependent, and the IG version is '
              'canonical. AToP was evaluated on a single Transformer, '
              'task, and dataset; portability remains to be demonstrated.')

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
              "Project administration; Validation; Writing – review "
              "and editing.")
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
              "https://github.com/belouali/atop (release v1.0.2, the "
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
        # [1]
        'Li Y, Rao S, Solares JRA, et al. BEHRT: Transformer for Electronic Health Records. Scientific Reports. 2020;10(1):7155.',
        # [2]
        'Rasmy L, Xiang Y, Xie Z, Tao C, Zhi D. Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. NPJ Digital Medicine. 2021;4(1):86.',
        # [3]
        'Wornow M, Thapa R, Steinberg E, Fries JA, Shah NH. EHRSHOT: an EHR benchmark for few-shot evaluation of foundation models. Advances in Neural Information Processing Systems (Datasets and Benchmarks Track). 2023;36.',
        # [4]
        'Steinberg E, Fries JA, Xu Y, Shah NH. MOTOR: a time-to-event foundation model for structured medical records. International Conference on Learning Representations (ICLR). 2024.',
        # [5]
        'Amirahmadi A, Ohlsson M, Etminani K. Deep learning prediction models based on EHR trajectories: a systematic review. Journal of Biomedical Informatics. 2023;144:104430.',
        # [6]
        'Wornow M, Xu Y, Thapa R, et al. The shaky foundations of large language models and foundation models for electronic health records. npj Digital Medicine. 2023;6(1):135.',
        # [7]
        'Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems. 2017;30:4765–4774.',
        # [8]
        'Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. Proceedings of the 34th International Conference on Machine Learning. 2017;70:3319–3328.',
        # [9]
        'Shrikumar A, Greenside P, Kundaje A. Learning important features through propagating activation differences. Proceedings of the 34th International Conference on Machine Learning. 2017;70:3145–3153.',
        # [10]
        'Bopche R, Gustad LT, Afset JE, et al. In-hospital mortality, readmission, and prolonged length of stay risk prediction leveraging historical electronic patient records. JAMIA Open. 2024;7(3):ooae074.',
        # [11]
        'Rong R, Gu Z, Lai H, et al. A deep learning model for clinical outcome prediction using longitudinal inpatient electronic health records. JAMIA Open. 2025;8(2):ooaf026.',
        # [12]
        'Placido D, Yuan B, Hjaltelin JX, et al. A deep learning algorithm to predict risk of pancreatic cancer from disease trajectories. Nature Medicine. 2023;29(5):1113–1122.',
        # [13]
        'Choi E, Bahadori MT, Kulas JA, Schuetz A, Stewart WF, Sun J. RETAIN: an interpretable predictive model for healthcare using reverse time attention mechanism. Advances in Neural Information Processing Systems. 2016;29:3504–3512.',
        # [14]
        'Jain S, Wallace BC. Attention is not explanation. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics. 2019:3543–3556.',
        # [15]
        'Wiegreffe S, Pinter Y. Attention is not not explanation. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. 2019:11–20.',
        # [16]
        'Mannila H, Toivonen H, Verkamo AI. Discovery of frequent episodes in event sequences. Data Mining and Knowledge Discovery. 1997;1(3):259–289.',
        # [17]
        'Hripcsak G, Albers DJ. Next-generation phenotyping of electronic health records. Journal of the American Medical Informatics Association. 2013;20(1):117–121.',
        # [18]
        'Campbell EA, Bass EJ, Masino AJ. Temporal condition pattern mining in large, sparse electronic health record data: a case study in characterizing pediatric asthma. Journal of the American Medical Informatics Association. 2020;27(4):558–566.',
        # [19]
        'Belouali A, Kitchen C, Haroz E, et al. Identifying clinical diagnostic trajectories associated with suicide death using temporal sequence mining of linked claims and mortality data. BioData Mining. 2026. doi:10.1186/s13040-026-00608-3.',
        # [20]
        'Assadi Shalmani M, Khani M, Taleban A, et al. Sequential pattern transformer (SPT): a generative and interpretable framework for predicting disease trajectories. Neural Computing and Applications. 2026;38(3):28.',
        # [21]
        'Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission: a systematic review. JAMA. 2011;306(15):1688–1698.',
        # [22]
        'Johnson AEW, Bulgarelli L, Pollard TJ, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data. 2023;10(1):1.',
        # [23]
        'Centers for Medicare & Medicaid Services. Diagnosis Code Set General Equivalence Mappings: ICD-10-CM to ICD-9-CM and ICD-9-CM to ICD-10-CM. Baltimore, MD: Centers for Medicare & Medicaid Services. https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-10-cm-icd-10-pcs-gem-archive',
        # [24]
        'Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Advances in Neural Information Processing Systems. 2017;30:5998–6008.',
        # [25]
        'van Walraven C, Dhalla IA, Bell C, et al. Derivation and validation of an index to predict early death or unplanned readmission after discharge from hospital to the community. CMAJ. 2010;182(6):551–557.',
        # [26]
        'Kokhlikyan N, Miglani V, Martin M, et al. Captum: A unified and generic model interpretability library for PyTorch. arXiv:2009.07896, 2020.',
        # [27]
        'DeYoung J, Jain S, Rajani NF, et al. ERASER: A benchmark to evaluate rationalized NLP models. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020:4443–4458.',
        # [28]
        "Krishna S, Han T, Gu A, Wu S, Jabbari S, Lakkaraju H. The disagreement problem in explainable machine learning: a practitioner's perspective. Transactions on Machine Learning Research. 2024.",
        # [29]
        'Lundberg SM, et al. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence. 2020;2(1):56–67.',
        # [30]
        'Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nat Mach Intell. 2019;1:206–215.',
        # [31]
        'Doshi-Velez F, Kim B. Towards a rigorous science of interpretable machine learning. arXiv:1702.08608, 2017.',
        # [32]
        'Hooker S, Erhan D, Kindermans PJ, Kim B. A benchmark for interpretability methods in deep neural networks. Advances in Neural Information Processing Systems. 2019;32:9737–9748.',
        # [33]
        'Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019;366(6464):447–453.',
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
              "printed because they do not sum arithmetically to the "
              "bars. Prefixes give visit reach "
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
              "top-30 tokens by absolute magnitude. The panels use "
              "different x-axis scales: the largest population-weighted "
              "magnitude is 0.040, against 1.139 for carrier-conditional "
              "attribution, a 28-fold difference.")
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
              "deduplication on carrier-set overlap. Panels share one "
              "x-axis.")
    para(doc, "Alt text: Two-panel horizontal bar chart of pattern-"
              "level signed \u03a3 IG. Left panel: all patterns. "
              "Right panel: cross-visit patterns only. Bars "
              "are red for risk-increasing and blue for protective.")

    # Figure 5 (perturbation dumbbell)
    para(doc, "**Figure 5.** Perturbation validation. (A) The 30 "
              "all-pattern representatives of Figure 4A; (B) the 30 "
              "cross-visit representatives of Figure 4B (\u22652 visits, "
              "\u22652 tokens). Within each panel, rows are ordered by "
              "|\u0394\u0177|. Each dumbbell is the effect of masking a "
              "pattern's tokens on predicted readmission probability, "
              "averaged over its test-set carriers. "
              "\u0394\u0177 = \u0177_original \u2212 \u0177_masked, as in Tables 3 "
              "and 4: \u0394\u0177 > 0 (red, right) means masking lowered "
              "predicted risk, so the pattern is risk-supporting; "
              "\u0394\u0177 < 0 (blue, left) means protective. Panels share "
              "one x-axis. Masking "
              "is whitelist-driven, each pattern having been selected by "
              "its IG ranking, so Figure 5 tests the IG-flagged subset "
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
