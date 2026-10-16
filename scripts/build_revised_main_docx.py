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
    para(doc, "**Word count:** Abstract 237; Body 3,969.")
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
    para(doc, '**Discussion:** Carrier-conditional aggregation and '
              'cross-visit pattern analysis provide complementary '
              'explanations, revealing signals obscured by '
              'prevalence and longitudinal patterns overshadowed by '
              'within-visit co-occurrences. Masking and visit-order '
              'reversal supported correspondence with model '
              'behavior, but pattern attributions do not represent '
              'unique effects or establish clinical validity. '
              'Cross-visit rankings were sensitive to the '
              'attribution method.')
    para(doc, '**Conclusion:** AToP turns a Transformer’s '
              'attributions into clinically legible trajectories, '
              'showing which events, in which order, contribute to '
              'its readmission predictions.')

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
    para(doc, 'The lack of temporal and combinatorial structure is '
              'especially consequential in longitudinal clinical '
              'data, where the same events may carry different '
              'meaning depending on whether they occur together '
              'within an encounter or sequentially across '
              'encounters. Prior work has applied sequential pattern '
              'mining to longitudinal EHR and claims data to study '
              'disease progression, adverse drug events, treatment '
              'pathways, and diagnostic trajectories [16–21]. '
              'Related approaches have also examined temporal '
              'patterns in EHR phenotyping [22]. Sequential pattern '
              'mining has also been used to construct inputs for '
              'Transformer-based disease trajectory prediction, with '
              'SHAP subsequently applied to explain individual '
              'predictions [23].')
    para(doc, 'We present AToP (Attribution to Temporal Patterns), a '
              'post-hoc explainability framework that uses '
              'token-level attribution from a trained Transformer to '
              'guide temporal pattern discovery. AToP aggregates '
              'attribution across patients whose records contain '
              'each pattern and distinguishes population-weighted '
              'from carrier-conditional attribution. We evaluate '
              'AToP on unplanned 30-day hospital readmission, a '
              'prediction task for which conventional risk models '
              'have historically shown modest discrimination [24], '
              'using MIMIC-IV v3.1 [25], which provides the '
              'longitudinal coverage needed for cross-visit pattern '
              'discovery.')

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
              'v3.1 [25], with an unplanned 30-day hospital readmission '
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
              'Mappings [26], retaining only unambiguous one-to-one '
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
    para(doc, 'A Transformer encoder [27] with a 24,005-token vocabulary,'
              ' 3.5 million parameters, four attention heads, two encoder'
              ' layers, and 128-dimensional embeddings was trained using '
              'binary cross-entropy loss with early stopping based on '
              'validation AUROC.')
    para(doc, 'Diagnoses, procedures, medications, and the [VISIT], '
              '[CLS], and [PAD] special tokens were represented in a '
              'shared embedding space, allowing the model to learn '
              'interactions across token types without type-specific '
              'architectural constraints. The [CLS] representation was '
              'passed through a linear layer with sigmoid activation to '
              'produce the predicted readmission probability.')
    para(doc, 'Patients were split at the patient level into training '
              '(70%; n=152,737), validation (10%; n=21,819), and test '
              '(20%; n=43,640) cohorts. Of the 43,640 patients in the '
              'test cohort, 43,602 were included in IG attribution and '
              'downstream AToP analyses; 38 were excluded because '
              'attribution yielded no salient-token set.')

    heading(doc, "Baselines and performance evaluation", level=2)
    para(doc, 'We compared the Transformer with two baselines trained on '
              'seven structured predictors: the four components of the '
              'LACE readmission index [28] (length of stay, emergent '
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
    para(doc, 'Discrimination was summarized by the area under the '
              'receiver operating characteristic curve (AUROC) and the '
              'area under the precision-recall curve (PR-AUC). '
              'Differences between models were estimated with a paired '
              'bootstrap of test-cohort patients (1,000 resamples), '
              'reporting percentile 95% confidence intervals and '
              'two-sided p values.')

    heading(doc, "AToP explanation framework", level=2)
    para(doc, 'AToP is a post-hoc explanation framework that operates on '
              'the trained model in five steps (Figure 1): (1) compute '
              'token-level attribution using Integrated Gradients; (2) '
              'select salient tokens based on their cumulative share of '
              'each patient’s absolute attribution mass; (3) apply '
              'frequent-episode mining to salient tokens from the '
              'training cohort to identify single-token events, '
              'within-visit co-occurrences, and cross-visit sequences; '
              '(4) summarize attribution at the token and pattern levels '
              'using carrier-conditional aggregation; and (5) apply '
              'Jaccard-based deduplication using patient overlap to '
              'collapse near-redundant patterns into representative '
              'patterns.')
    para(doc, 'Integrated Gradients was calculated using the model output'
              ' before conversion to a readmission probability. The '
              'resulting attribution values are therefore on the logit '
              'scale. By the completeness property, the token '
              'attributions sum approximately to the difference between '
              'the patient’s model output and the output for an all-[PAD]'
              ' reference sequence [8,29]. Details of the reference '
              'sequence and approximation error are provided in the '
              'Supplementary Methods.')
    para(doc, 'For each patient, a token’s attribution score was its '
              'signed IG value, and a multi-token pattern’s attribution '
              'score was the sum of the signed IG values of its '
              'constituent tokens. Both token and pattern rankings were '
              'based on absolute attribution magnitude, with the sign '
              'retained to indicate whether the token or pattern '
              'increased or decreased predicted risk.')
    para(doc, 'We define a carrier as a patient whose record contains the'
              ' token or pattern being evaluated. At the token level, '
              'carrier-conditional attribution was calculated as the mean'
              ' signed IG among patients whose records contained the '
              'token.')
    para(doc, 'The population-weighted attribution for token j was '
              'calculated as the mean signed IG across all patients:')
    equation(doc, "Aⱼᵖᵒᵖ = (1/N) Σᵢ IGᵢⱼ",
             number=1)
    para(doc, 'Tokens were ranked by the absolute magnitude of Aⱼᵖᵒᵖ, '
              'while its sign indicated whether the token increased or '
              'decreased predicted risk. Because patients whose records '
              'did not contain token j contributed zero attribution for '
              'that token, Equation (1) can also be written as:')
    equation(doc, "Aⱼᵖᵒᵖ = P(j present) × "
                  "E[IGⱼ | j present]",
             number=2)
    para(doc, 'Equation (2) shows that population-weighted attribution '
              'reflects both how often a token occurs and how strongly it'
              ' is attributed when present. Carrier-conditional '
              'attribution isolates the latter quantity.')
    para(doc, "At the pattern level, each carrier's score was obtained by"
              " matching the pattern to the patient's record step by "
              'step: each step was assigned the earliest visit containing'
              ' all its tokens, with later steps required to occur in '
              'strictly later visits. For each carrier, the score was the'
              " sum of signed IG values for the pattern's tokens in the "
              'matched visits. Scores were then averaged across carriers '
              'to obtain carrier-averaged ΣIG. Patterns were ranked by '
              'the absolute magnitude of their carrier-averaged ΣIG, with'
              ' the sign retained to indicate the direction of '
              'attribution. For the per-patient display, each salient '
              'token occurrence was counted only once to avoid '
              'double-counting across overlapping patterns (Supplementary'
              ' Methods).')
    para(doc, 'Frequent-episode mining [16] was implemented using a '
              'level-wise candidate-generation algorithm and applied only'
              ' to salient-token subsequences from the training cohort. '
              'Patterns were constructed by either adding a co-occurring '
              'token within the same visit or appending a token from a '
              'later visit, yielding three pattern types: single-token '
              'events, within-visit co-occurrences, and cross-visit '
              'sequences. Mining was conducted separately in each outcome'
              ' class, using a minimum support fraction of 0.001 and a '
              'maximum of four tokens per pattern. The resulting patterns'
              ' were then pooled into a single set of distinct patterns.')
    para(doc, 'The pooled patterns formed the candidate set for global '
              'rankings and per-patient explanations, with eligibility '
              'criteria described below.')
    para(doc, 'To reduce near-duplicate patterns, we compared the sets of'
              ' test-cohort patients whose records contained each pattern'
              ' using Jaccard similarity. Patterns with Jaccard '
              'similarity ≥0.20 were grouped using greedy clustering, and'
              ' the pattern with the largest absolute attribution '
              'magnitude within each group was retained as the '
              'representative.')
    para(doc, 'Test-cohort carrier counts were used to determine '
              'reporting eligibility, and carrier-set overlap was used '
              'for deduplication. Neither influenced model training or '
              'pattern discovery.')

    heading(doc, "Parameter choices and sensitivity analyses", level=2)
    para(doc, 'AToP parameters were specified as design or tractability '
              'constraints rather than selected by optimizing test-set '
              'outcomes. The exception was the 0.25 maximum '
              'drug-prevalence threshold, which was selected by '
              'inspecting patterns mined from the training cohort, '
              'without using outcome labels or test data.')
    para(doc, 'After mining, separate eligibility criteria were applied '
              'to global rankings and per-patient explanations. The '
              'all-patterns ranking required at least five test-cohort '
              'carriers, while the cross-visit ranking required at least '
              'three carriers and patterns spanning at least two visits '
              'and two tokens. Minimum carrier counts excluded patterns '
              'represented by very few patients, while the cross-visit '
              'requirements restricted the analysis to longitudinal '
              'sequences. Per-patient explanations used the subset of '
              'pooled patterns observed in at least 0.1% of the full '
              'training cohort.')
    para(doc, 'To focus pattern mining on tokens with the largest '
              'attributions while limiting the search space, we '
              'retained, for each patient, the smallest set of '
              'tokens accounting for 80% of total absolute IG, '
              'capped at 50 tokens. We assessed sensitivity to the '
              'attribution-mass threshold across values from 0.60 to '
              '0.95. Additional implementation details are provided '
              'in the Supplementary Methods; parameter settings and '
              'their selection rationale are summarized in Table S1.')

    heading(doc, "Validation analyses", level=2)
    para(doc, 'We used pattern masking to assess whether the identified '
              'patterns influenced model predictions. We examined the '
              'role of temporal order by comparing attribution for '
              'forward and reversed patterns across patients and '
              'reversing matched visits within patients. We also tested '
              'whether highly attributed tokens influenced predictions '
              'more than randomly selected salient tokens, and whether '
              'IG-based rankings were consistent with KernelSHAP.')
    para(doc, 'Masking was applied to the reported patterns and '
              'additional high-ranked patterns with positive or negative '
              'attribution, selected by absolute ΣIG. For each pattern, '
              'the corresponding token positions were replaced with '
              '[PAD], and the change in predicted readmission probability'
              ' was calculated as Δŷ = ŷ_original − ŷ_masked.')
    para(doc, 'We examined whether pattern attribution was sensitive '
              'to temporal order using two approaches. First, we '
              'compared attribution between patients whose records '
              'contained a pattern in its original order and those '
              'whose records contained the reversed sequence. We '
              'performed this comparison for the reported '
              'cross-visit patterns and for a broader set of '
              'patterns with at least 100 test-cohort carriers, '
              'retaining those with at least 20 patients matching '
              'each ordering. Second, we examined temporal order '
              'within patients by reversing the order of matched '
              'visits in their records, recomputing IG, and '
              'comparing pattern attribution before and after '
              'reversal.')
    para(doc, 'We assessed token-level attribution faithfulness by '
              'examining how model predictions changed when highly '
              'attributed tokens were progressively removed or retained '
              '[30]. The analysis included 500 randomly sampled test '
              'patients with at least three salient tokens.')
    para(doc, 'Comprehensiveness was measured as the absolute change in '
              'predicted readmission probability after masking the top k '
              'salient tokens ranked by |IG|. Sufficiency was measured as'
              ' the absolute change after retaining those k tokens and '
              'masking the remaining salient tokens. We varied k, the '
              'number of selected tokens, from 1 to 10. Non-salient and '
              'special tokens were left unchanged in both analyses. For '
              'each k, only patients with at least k salient tokens were '
              'included.')
    para(doc, 'We compared both measures with a random baseline that '
              'selected the same number of salient tokens. Confidence '
              'intervals were estimated by bootstrapping patients. '
              'Comprehensiveness at k = 5 was compared between IG and '
              'random selection using a one-sided Wilcoxon signed-rank '
              'test.')
    para(doc, 'To assess whether the attribution findings depended on IG,'
              ' we repeated the analysis using KernelSHAP in the analytic'
              ' test cohort. We compared the rankings produced by both '
              'methods for population-weighted and carrier-conditional '
              'token attribution, as well as for all-pattern and '
              'cross-visit pattern attribution. The same aggregation '
              'procedures were applied to both methods.')

    # ── Results ──
    heading(doc, "Results", level=1)

    heading(doc, "Predictive performance", level=2)
    para(doc, 'Cohort derivation is shown in Figure S1, and cohort '
              'characteristics are given in Table 1. The Transformer '
              'achieved an AUROC of 0.766 (95% CI, 0.759–0.773) and '
              'a PR-AUC of 0.342. The XGBoost and logistic '
              'regression baselines, using conventional readmission '
              'predictors, achieved AUROCs of 0.728 and 0.659, '
              'respectively, with PR-AUCs of 0.312 and 0.240. The '
              'Transformer achieved significantly higher AUROC than '
              'both LACE-based models (both p < 0.001; Table 2). '
              'Because the models used different inputs, these '
              'comparisons do not isolate the effect of architecture.')

    # ── Table 1 ──────
    df1 = _read_csv("Table1_cohort.csv")
    df1_meta = _read_csv("Table1_cohort_meta.csv").iloc[0]
    n_tot = int(df1_meta["n_total"])
    n_r = int(df1_meta["n_readmitted"])
    n_nr = int(df1_meta["n_not_readmitted"])
    pct_r = float(df1_meta["pct_readmitted"])
    pct_nr = float(df1_meta["pct_not_readmitted"])
    table_title(doc, "Table 1", "Cohort characteristics by unplanned 30-day readmission status.")
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
            "CCI = Charlson Comorbidity Index; ED = emergency "
            "department; IQR = interquartile range; LACE = Length of "
            "stay, Acuity of admission, Charlson Comorbidity Index, "
            "and Emergency department visits; SMD = standardized mean "
            "difference."
        ),
        note=(
            "Longitudinal measures (CCI, prior ED visits, and number "
            "of admissions) were calculated using patient history "
            "through the index admission."
        ),
    )

    # ── Table 2 (Model performance vs baselines) ──────────
    df2 = _read_csv("Table2_model_performance.csv")
    table_title(doc, "Table 2",
                  "Predictive performance of the Transformer and baseline models.")
    _t2_headers = {"\u0394AUROC vs Transformer (95% CI)":
                   "\u0394AUROC (baseline \u2212 Transformer), 95% CI",
                   "p": "P value"}
    make_table(doc,
                header_row=[_t2_headers.get(c, c) for c in df2.columns],
                data_rows=[[_fmt_str(v) for v in row]
                            for row in df2.itertuples(index=False)],
                col_widths_in=[1.85, 1.40, 1.40, 1.40, 0.55],
                font_size_pt=9)
    table_legend(doc,
        abbreviations=(
            "AUROC = area under the receiver operating characteristic "
            "curve; PR-AUC = area under the precision-recall curve; CI "
            "= confidence interval; LACE = Length of stay, Acuity of "
            "admission, Charlson Comorbidity Index, and Emergency "
            "department visits."
        ),
        note=(
            "Both baseline models used the four LACE components plus "
            "age, sex, and prior admissions. Performance was evaluated "
            "in the test cohort (N = 43,640). Confidence intervals and "
            "P values were calculated using 1,000 patient-level "
            "bootstrap resamples."
        ),
    )

    heading(doc, "Single-patient explanation", level=2)
    para(doc, 'Figure 2 illustrates an AToP explanation for a test '
              'patient with an oncologic trajectory spanning five '
              'admissions over 911 days. The patient was readmitted '
              'two days after the index discharge, with a predicted '
              'readmission probability of 0.93. The largest '
              'attributed contribution was a cross-visit pattern '
              'consisting of cancer pain at the second admission '
              'followed by co-occurring cancer anemia and cancer '
              'pain at the index admission (+0.83 of the total ΣIG '
              'of +0.97). Smaller protective contributions included '
              'thiamine followed by cancer anemia (−0.08). The case '
              'was selected to illustrate the explanation rather '
              'than represent the cohort. Additional examples '
              'involving psychiatric, obstetric, and oncologic '
              'patients are presented in Figures S2A–C.')
    para(doc, 'Because patterns can overlap, each salient token '
              'position is assigned to only one pattern in the '
              'attribution waterfall (panel B of Figure 2), using a '
              'fixed priority rule (Supplementary Methods). The '
              'resulting contributions are allocations rather than '
              'independent pattern effects.')
    # JAMIA submission: figures uploaded as separate files, NOT embedded
    # inline. Full Figure 2 legend + alt text appears in the
    # "Figure Legends" section at the end of the manuscript (after
    # References). The body text already cites Figure 2 by number above.

    heading(doc, 'Token-level global attribution: '
                 'carrier-conditioning highlights rare, highly '
                 'attributed features', level=2)
    para(doc, 'Figure 3 compares population-weighted and '
              'carrier-conditional token attribution in the test '
              'cohort. The population-weighted ranking (Equation 1; '
              'Figure 3A) was dominated by common codes with modest '
              'attribution among carriers, including MDD single '
              'episode NOS (not otherwise specified), live birth, '
              'and chest pain NOS. In contrast, the '
              'carrier-conditional ranking (the conditional term in '
              'Equation 2; Figure 3B) highlighted less prevalent '
              'codes with high attribution among carriers, including '
              'psychosis NOS, acetaminophen overdose/self-harm, '
              'severe MDD, and bipolar manic episode.')
    para(doc, 'Only 7 tokens appeared in both top-30 rankings. Of '
              'the 23 tokens unique to the carrier-conditional '
              'ranking, 21 ranked below 100 in the '
              'population-weighted ranking (Figure S3).')

    heading(doc, 'Pattern-level global attribution: cross-visit '
                 'rankings highlight longitudinal patterns', level=2)
    para(doc, 'Pattern mining counts are reported in Table S2. '
              'Figure 4 compares carrier-averaged ΣIG rankings '
              'across all pattern types (Figure 4A) and cross-visit '
              'patterns only (Figure 4B). The all-patterns ranking '
              'was dominated by within-visit co-occurrences, such as '
              '{Severe MDD; SI}. In contrast, the cross-visit '
              'ranking highlighted psychiatric, oncologic, and '
              'obstetric trajectories, including {MDD; SI} → '
              '{Recurrent severe MDD; SI} (Tables 3–4).')
    para(doc, 'The two rankings showed almost no overlap, sharing '
              'only one pattern among their top 50; rankings at '
              'other Jaccard thresholds are shown in Figures S4–S5. '
              'The top-30 all-pattern representatives had '
              'approximately 2.4-fold larger mean |ΣIG| than the '
              'top-30 cross-visit representatives (1.16 vs 0.48). '
              'When ranked by population-weighted rather than '
              'carrier-averaged ΣIG, only 6 of the top 30 patterns '
              'overlapped (Figure S6).')
    para(doc, 'To assess whether pattern attribution simply '
              'reflected prevalence or observed readmission risk, we '
              'compared attribution magnitudes with pattern '
              'prevalence and relative risk (Figure S7). '
              'Patterns with similar prevalence or relative risk '
              'showed substantial differences in attribution '
              'magnitude.')
    # Figure 4. separate file (see Figure Legends at end of manuscript).

    heading(doc, 'Clinical groups and masking effects', level=2)
    para(doc, 'Tables 3 and 4 present the top five patterns by '
              'absolute carrier-averaged ΣIG within each clinical '
              'group, for all patterns and cross-visit patterns, '
              'respectively. Masking the tokens in each displayed '
              'pattern changed predicted readmission probability in '
              'the direction indicated by its attribution (Figure '
              '5), as it did for all 266 perturbed patterns (Figure '
              'S8).')
    para(doc, 'To examine whether pattern attributions aligned with '
              'observed readmission risk, we estimated odds ratios '
              'for the 30 reported cross-visit patterns, adjusting '
              'for the number of admissions. The direction of model '
              'attribution agreed with the direction of the odds '
              'ratio for 25 of these patterns (Table S3). These '
              'associations were descriptive and did not establish '
              'attribution faithfulness.')

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
                  "Top attribution patterns by clinical group.")
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
                header_row=["Clinical group", "Pattern", "ΣIG", "n", "Δŷ"],
                data_rows=_rows_t3,
                col_widths_in=[1.5, 2.85, 0.6, 0.45, 0.6],
                font_size_pt=9)
    table_legend(doc,
        abbreviations=(
            "IG = Integrated Gradients; ΣIG = carrier-averaged sum of "
            "token attributions (logit scale); n = number of "
            "test-cohort patients containing the pattern; Δŷ = mean "
            "decrease in predicted readmission probability when the "
            "pattern is masked (ŷ_original − ŷ_masked); MDD = major "
            "depressive disorder; SI = suicidal ideation; NOS = not "
            "otherwise specified; Hx = history of; uncomp. = "
            "uncomplicated; intox = intoxication; ep. = episode; Resp "
            "= respiratory; C-section = cesarean delivery."
        ),
        note=(
            "The five patterns with the largest absolute ΣIG are shown "
            "for each clinical group. Positive attribution indicates a "
            "contribution toward higher predicted readmission "
            "probability, and negative attribution indicates a "
            "contribution toward lower predicted probability. Braces { "
            "} indicate co-occurring events within a visit, while "
            "arrows (→) indicate event order across visits. Clinical "
            "groups are presentational groupings assigned by keyword "
            "rules (Supplementary Methods) and do not affect mining, "
            "attribution, or ranking."
        ),
    )

    # \u2500\u2500 Bridge paragraph + Table 4 (all patterns, phenotype-organized) \u2500

    # \u2500\u2500 Table 4 (all patterns, phenotype-organized) \u2500\u2500
    df4 = _read_csv("Table4_cross_visit_clusters.csv")
    # Table 4 keeps six families (the four of Table 3 plus Hepatobiliary and
    # Other, Other not split by sign): these 30 patterns are the set analysed
    # in the reversal and adjusted-odds-ratio tables (Tables S3 and S5).
    PHENO_ORDER_T4 = PHENO_ORDER + ["Hepatobiliary", "Other"]
    df4 = df4[df4["phenotype"].isin(PHENO_ORDER_T4)].copy()
    df4 = _split_other(df4, top_n=5, phen_order=PHENO_ORDER_T4, split_other=False)
    table_title(doc, "Table 4",
                  "Top cross-visit attribution patterns by clinical group.")
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
                header_row=["Clinical group", "Pattern", "ΣIG", "n", "Δŷ"],
                data_rows=_rows_t4,
                col_widths_in=[1.5, 2.85, 0.6, 0.45, 0.6],
                font_size_pt=9)
    table_legend(doc,
        abbreviations=(
            "ΣIG, n, Δŷ, and other abbreviations are as defined in "
            "Table 3; CVC = central venous catheter; ERCP = endoscopic "
            "retrograde cholangiopancreatography; GERD = "
            "gastroesophageal reflux disease; NEC = not elsewhere "
            "classified; Ag = antigen."
        ),
        note=(
            "The five patterns with the largest absolute ΣIG are shown "
            "for each clinical group. All patterns span at least two "
            "visits."
        ),
    )

    # Figure 5. separate file (see Figure Legends at end of manuscript).

    heading(doc, "Faithfulness of the attribution ranking", level=2)
    para(doc, 'To assess whether highly attributed tokens had '
              'greater influence on model predictions, we compared '
              'IG-ranked tokens with randomly selected salient '
              'tokens. Masking the five highest-ranked tokens '
              "produced a larger mean absolute change in the model's "
              'predicted readmission probability than masking five '
              'randomly selected salient tokens (0.079 vs 0.056; '
              'Wilcoxon signed-rank p < 0.001).')
    para(doc, 'Across selections of 1 to 10 tokens, masking the '
              'highest-attributed tokens produced larger prediction '
              'changes than masking randomly selected salient '
              'tokens, and retaining them better preserved the '
              'original predictions, with non-overlapping 95% '
              'bootstrap intervals at every k; only k = 5 was '
              'formally tested (Figure S9). These results indicate '
              'faithfulness to the model, not clinical correctness.')

    heading(doc, "Temporal-order analysis", level=2)
    para(doc, 'Among the 30 cross-visit patterns in Table 4, three '
              'had the same sequence in both directions and were '
              'excluded. Of the remaining 27, attribution magnitudes '
              'were lower for the reversed sequence in 26 (96%; mean '
              'attenuation, 32%; Table S5). In the broader set of '
              '1,714 patterns, however, attribution magnitudes '
              'decreased in only 54%, while attribution direction '
              'remained unchanged in 95% (Figure S10; Table S4). '
              'Because these comparisons used naturally occurring '
              'sequences, differences could also reflect patient '
              'characteristics.')
    para(doc, 'In the within-patient analysis, reversing the order '
              'of matched visits reduced attribution magnitude for '
              '21 of 27 patterns (mean attenuation, 14%). Among 722 '
              'patients, 59% showed a reduction, with a '
              'statistically significant overall difference (p < '
              '0.001; Table S6).')

    heading(doc, "Cross-method comparison", level=2)
    para(doc, 'To assess whether the attribution findings were '
              'consistent across methods, we repeated the analysis '
              'using KernelSHAP instead of IG; per-patient agreement '
              'is shown in Figure S11. The two methods shared 25 of '
              'their top 30 carrier-conditional tokens (Figure S12). '
              'At the pattern level, they shared 21 of the top 30 '
              'patterns across all pattern types, but only 7 of the '
              'top 30 cross-visit patterns (Figure S13). These '
              'findings indicate substantial agreement at the token '
              'level but greater sensitivity to the attribution '
              'method for cross-visit pattern rankings, consistent '
              'with the documented disagreement between explanation '
              'methods applied to the same model [31].')

    heading(doc, "Discussion", level=1)
    para(doc, '**What is new here.** AToP combines carrier-conditional '
              'aggregation with attribution-guided temporal mining, '
              'shifting explanation from isolated tokens to multi-visit '
              'patterns rather than only re-weighting population-averaged'
              ' token importance, the global summary commonly reported '
              'with SHAP [32]. Token-level findings were largely '
              'consistent under KernelSHAP (25/30; Figure S12), whereas '
              'cross-visit pattern rankings remain attribution-method '
              'dependent (7/30); we therefore report the IG '
              'implementation as canonical.')
    para(doc, 'The cross-visit restriction is a design choice: the '
              'all-pattern and cross-visit rankings answer different '
              'questions and their top-50 sets overlap in 1 of 50. '
              'Single-visit patterns carry larger attribution magnitude '
              '(about 2.4-fold among the top-30 representatives), whereas cross-visit patterns provide the '
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
              "to any post-hoc account of an opaque one [33]. We do not "
              "contest that; AToP is for settings where a sequence model "
              "is already in use. We also evaluate faithfulness to the "
              "model rather than usefulness to a decision-maker, which "
              "are distinct targets [34].")
    para(doc, 'Further limitations bound our claims. (i) Evaluation uses '
              'one dataset, task, attribution method and architecture; '
              'portability is not demonstrated. (ii) Masking creates '
              'out-of-distribution inputs [35], so masking effects may '
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
              'deployment use [36]. (vi) The 80% saliency-mass threshold '
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
              "https://github.com/belouali/atop, including the five-step "
              "pipeline, clinical-group classifier, masking-validation "
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
        'Campbell EA, Bass EJ, Masino AJ. Temporal condition pattern mining in large, sparse electronic health record data: a case study in characterizing pediatric asthma. Journal of the American Medical Informatics Association. 2020;27(4):558–566.',
        # [18]
        'Jensen AB, Moseley PL, Oprea TI, et al. Temporal disease trajectories condensed from population-wide registry data covering 6.2 million patients. Nature Communications. 2014;5:4022.',
        # [19]
        'Davazdahemami B, Delen D. Examining the effect of prescription sequence on developing adverse drug reactions: the case of renal failure in diabetic patients. International Journal of Medical Informatics. 2019;125:62–70.',
        # [20]
        'Wright AP, Wright AT, McCoy AB, Sittig DF. The use of sequential pattern mining to predict next prescribed medications. Journal of Biomedical Informatics. 2015;53:73–80.',
        # [21]
        'Belouali A, Kitchen C, Haroz E, et al. Identifying clinical diagnostic trajectories associated with suicide death using temporal sequence mining of linked claims and mortality data. BioData Mining. 2026. doi:10.1186/s13040-026-00608-3.',
        # [22]
        'Hripcsak G, Albers DJ. Next-generation phenotyping of electronic health records. Journal of the American Medical Informatics Association. 2013;20(1):117–121.',
        # [23]
        'Assadi Shalmani M, Khani M, Taleban A, et al. Sequential pattern transformer (SPT): a generative and interpretable framework for predicting disease trajectories. Neural Computing and Applications. 2026;38(3):28.',
        # [24]
        'Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission: a systematic review. JAMA. 2011;306(15):1688–1698.',
        # [25]
        'Johnson AEW, Bulgarelli L, Pollard TJ, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data. 2023;10(1):1.',
        # [26]
        'Centers for Medicare & Medicaid Services. Diagnosis Code Set General Equivalence Mappings: ICD-10-CM to ICD-9-CM and ICD-9-CM to ICD-10-CM. Baltimore, MD: Centers for Medicare & Medicaid Services. https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-10-cm-icd-10-pcs-gem-archive',
        # [27]
        'Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Advances in Neural Information Processing Systems. 2017;30:5998–6008.',
        # [28]
        'van Walraven C, Dhalla IA, Bell C, et al. Derivation and validation of an index to predict early death or unplanned readmission after discharge from hospital to the community. CMAJ. 2010;182(6):551–557.',
        # [29]
        'Kokhlikyan N, Miglani V, Martin M, et al. Captum: A unified and generic model interpretability library for PyTorch. arXiv:2009.07896, 2020.',
        # [30]
        'DeYoung J, Jain S, Rajani NF, et al. ERASER: A benchmark to evaluate rationalized NLP models. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020:4443–4458.',
        # [31]
        "Krishna S, Han T, Gu A, Wu S, Jabbari S, Lakkaraju H. The disagreement problem in explainable machine learning: a practitioner's perspective. Transactions on Machine Learning Research. 2024.",
        # [32]
        'Lundberg SM, et al. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence. 2020;2(1):56–67.',
        # [33]
        'Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nat Mach Intell. 2019;1:206–215.',
        # [34]
        'Doshi-Velez F, Kim B. Towards a rigorous science of interpretable machine learning. arXiv:1702.08608, 2017.',
        # [35]
        'Hooker S, Erhan D, Kindermans PJ, Kim B. A benchmark for interpretability methods in deep neural networks. Advances in Neural Information Processing Systems. 2019;32:9737–9748.',
        # [36]
        'Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019;366(6464):447–453.',
    ]
    for i, r in enumerate(refs, 1):
        para(doc, f"[{i}] {r}")

    # ── Figure legends section (separate page) ─────────────────────
    page_break(doc)
    heading(doc, "Figure Legends", level=1)
    para(doc, '**Figure 1.** AToP framework overview. A trained '
              'Transformer predicts readmission from longitudinal '
              'EHR sequences. AToP uses Integrated Gradients to '
              'identify salient tokens, mines recurring within-visit '
              'and cross-visit patterns from those tokens, and '
              'aggregates attribution among patients carrying each '
              'pattern. Patterns with overlapping patient '
              'populations are grouped using Jaccard similarity, '
              'retaining representative patterns for interpretation. '
              'Pattern values and the predicted probability shown '
              'are illustrative. MDD = major depressive disorder; SI '
              '= suicidal ideation; PTSD = post-traumatic stress '
              'disorder; ECG = electrocardiogram; AKI = acute kidney '
              'injury; ep. = episode.')
    para(doc, 'Alt text: Diagram showing longitudinal EHR data '
              'processed by a Transformer and the five-step AToP '
              'pipeline, producing token-level and temporal-pattern '
              'explanations alongside predicted readmission risk.')

    # Figure 2
    para(doc, '**Figure 2.** Single-patient AToP explanation. (A) '
              'Timeline of five admissions showing selected '
              'attribution-salient clinical events and observed '
              'readmission. Red upward triangles indicate '
              'risk-increasing attribution, and blue downward '
              'triangles indicate protective attribution. (B) Signed '
              'IG attribution (logit scale) allocated across '
              'within-visit and cross-visit patterns, with remaining '
              'attribution shown as other salient and non-salient '
              'tokens. Row prefixes give the admissions each pattern '
              'spans (V1→V2 cross-visit; V5 single-visit). Shared '
              'token positions were assigned to a single pattern '
              'using a fixed priority rule (Supplementary Methods); '
              'the bars represent attribution allocations rather '
              'than independent pattern effects. NOS = not otherwise '
              'specified; AKI = acute kidney injury; Fam Hx = family '
              'history; malig. = malignant; Resect. = resection; R = '
              'right.')
    para(doc, 'Alt text: Two-panel explanation for a readmitted '
              'patient with five hospital admissions. Panel A shows '
              'a timeline of selected clinical events, with red and '
              'blue markers indicating attribution direction. Panel '
              'B shows cumulative signed attribution contributions, '
              'dominated by a cross-visit cancer-pain pattern, '
              'followed by smaller positive and negative '
              'contributions from other patterns and remaining '
              'tokens.')

    # Figure 3 (token-level, was old Panels A+B)
    para(doc, '**Figure 3.** Token-level attribution: '
              'population-weighted versus carrier-conditional. (A) '
              'Mean signed Integrated Gradients (IG) attribution '
              'across all test-cohort patients, with non-carriers '
              'contributing zero. (B) Mean signed IG attribution '
              'among patients carrying each token. Each panel shows '
              'the 30 tokens with the largest absolute mean '
              'attribution. Red bars indicate contributions toward '
              'higher predicted readmission risk, and blue bars '
              'indicate contributions toward lower predicted risk. '
              'Horizontal-axis scales differ between panels. MDD = '
              'major depressive disorder; ep. = episode; NOS = not '
              'otherwise specified; SI = suicidal ideation; uncomp. '
              '= uncomplicated; intox = intoxication; Resp = '
              'respiratory; GERD = gastroesophageal reflux disease; '
              'AKI = acute kidney injury; AFib = atrial '
              'fibrillation; OD = overdose; Acetamin = '
              'acetaminophen; SSRI = selective serotonin reuptake '
              'inhibitor; R = right; L = left.')
    para(doc, 'Alt text: Two horizontal bar charts comparing '
              'token-attribution rankings. Panel A shows '
              'population-weighted attribution, with common events '
              'such as major depressive disorder and live birth '
              'among the highest-ranked tokens. Panel B shows '
              'carrier-conditional attribution, with psychosis, '
              'overdose associated with self-harm, and severe '
              'depression among the highest-ranked tokens. Positive '
              'attribution is red, and negative attribution is blue.')

    # Figure 4 (pattern-level, was old Panels C+D)
    para(doc, '**Figure 4.** Pattern-level attribution: all patterns '
              'versus cross-visit patterns. (A) The 30 '
              'highest-ranked patterns across single-token events, '
              'within-visit co-occurrences, and cross-visit '
              'sequences. (B) The 30 highest-ranked cross-visit '
              'patterns, restricted to sequences spanning at least '
              'two visits and containing at least two tokens. '
              'Rankings use absolute carrier-averaged ΣIG after '
              'Jaccard-based deduplication. Red bars indicate '
              'contributions toward higher predicted readmission '
              'risk, and blue bars indicate contributions toward '
              'lower predicted risk. Both panels use the same '
              'horizontal-axis scale. MDD = major depressive '
              'disorder; ep. = episode; NOS = not otherwise '
              'specified; SI = suicidal ideation; Hx = history of; '
              'uncomp. = uncomplicated; intox = intoxication; Resp = '
              'respiratory; OD = overdose; Acetamin = acetaminophen; '
              'C-section = cesarean delivery; Ag = antigen; PTSD = '
              'post-traumatic stress disorder; mets = metastases.')
    para(doc, 'Alt text: Two horizontal bar charts comparing the 30 '
              'highest-ranked attribution patterns. Panel A is '
              'dominated by within-visit psychiatric combinations, '
              'particularly depression and suicidal ideation. Panel '
              'B shows cross-visit sequences, including recurrent '
              'psychiatric events and oncologic progressions. Red '
              'bars indicate positive attribution and blue bars '
              'negative attribution.')

    # Figure 5 (perturbation dumbbell)
    para(doc, '**Figure 5.** Masking effects of top-ranked '
              'attribution patterns. (A) The 30 highest-ranked '
              'all-pattern representatives from Figure 4A. (B) The '
              '30 highest-ranked cross-visit representatives from '
              'Figure 4B. For each pattern, Δŷ = ŷ_original − '
              'ŷ_masked represents the mean change in predicted '
              'readmission probability after masking its tokens '
              'among test-cohort carriers. Positive values (red) '
              'indicate that masking lowered predicted risk, while '
              'negative values (blue) indicate that masking '
              'increased predicted risk. Patterns are ordered by '
              'absolute Δŷ within each panel. Both panels use the '
              'same horizontal-axis scale. Abbreviations as in '
              'Figure 4.')
    para(doc, 'Alt text: Two-panel chart showing changes in '
              'predicted readmission probability after masking '
              'highly attributed patterns. Panel A displays patterns '
              'of all types, with prominent positive masking effects '
              'for psychiatric combinations. Panel B displays '
              'cross-visit sequences, including positive effects for '
              'psychiatric and oncologic patterns and negative '
              'effects for obstetric and chest-pain patterns. Red '
              'markers indicate positive masking effects and blue '
              'markers indicate negative effects.')

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    doc.save(OUT_PATH)
    n_para = len(doc.paragraphs)
    n_table = len(doc.tables)
    print(f"\u2713 Wrote {OUT_PATH}")
    print(f"  Paragraphs: {n_para},  Tables: {n_table}")


if __name__ == "__main__":
    build()
