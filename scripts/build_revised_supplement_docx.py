"""build_revised_supplement_docx, produce AToP_JAMIA_supplement.v2.docx

Comprehensive rebuild of the supplementary material reflecting the
post-redesign figure inventory:

  * Existing supplements (S1\u2013S6) retained with description corrections
  * NEW: S7 (cross-visit Jaccard sensitivity)
  * NEW: S8 (full-pool Jaccard sensitivity, justifies the locks)
  * NEW: S9 (token-level aggregation comparison: carrier vs population)
  * NEW: S10 (pattern-level aggregation comparison: full pool)
  * NEW: S11 (SHAP-based Fig 3 replication on full test cohort)
  * Existing Table S2 retained (AUROC bootstrap CIs)
  * NEW: Table S4 (full-pool phenotype clusters as contrast to Table 3)

Usage:
    python scripts/build_revised_supplement_docx.py
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _docx_helpers import (
    new_document, title, heading, para, figure_placeholder, figure_landscape,
    table_caption, table_title, table_legend, make_table,
    page_break, hrule,
)


# ── NaN-safe formatting helpers ──
def _fmt_num(v, fmt: str, default: str = "n/a") -> str:
    if pd.isna(v):
        return default
    return format(float(v), fmt)


def _fmt_str(v, default: str = "") -> str:
    if pd.isna(v):
        return default
    return str(v)


def _fmt_int(v, default: str = "n/a") -> str:
    if pd.isna(v):
        return default
    return f"{int(v)}"


SUBMISSION_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                               "manuscript", "jamia_submission")
TABLES_DIR     = os.path.join(SUBMISSION_DIR, "tables")
OUT_PATH       = os.path.join(SUBMISSION_DIR, "AToP_JAMIA_supplement.docx")
SUPPLEMENTS_DIR = os.path.join(SUBMISSION_DIR, "figures", "supplements")

IMAGE_PATHS = {
    "Figure S10":  os.path.join(SUPPLEMENTS_DIR, "FigureS1_reversed_order.png"),
    "Figure S11":  os.path.join(SUPPLEMENTS_DIR, "FigureS3_shap_vs_ig.png"),  # may not exist; placeholder fallback
    "Figure S2A": os.path.join(SUPPLEMENTS_DIR, "FigureS4A_alt_tp.png"),
    "Figure S2B": os.path.join(SUPPLEMENTS_DIR, "FigureS4B_confident_tn.png"),
    "Figure S2C": os.path.join(SUPPLEMENTS_DIR, "FigureS4C_disagreement.png"),
    "Figure S7":  os.path.join(SUPPLEMENTS_DIR, "FigureS5_prevalence_vs_impact.png"),
    "Figure S8": os.path.join(SUPPLEMENTS_DIR, "FigureS10_masking_distribution.png"),
    "Figure S4":  os.path.join(SUPPLEMENTS_DIR, "FigureS7_jaccard_sensitivity.png"),
    "Figure S5":  os.path.join(SUPPLEMENTS_DIR, "FigureS8_full_pool_jaccard_sensitivity.png"),
    "Figure S3":  os.path.join(SUPPLEMENTS_DIR, "FigureS9_aggregation_comparison.png"),
    "Figure S6": os.path.join(SUPPLEMENTS_DIR, "FigureS10_pattern_aggregation_comparison.png"),
    "Figure S12": os.path.join(SUPPLEMENTS_DIR, "FigureS12_shap_tokens.png"),
    "Figure S13": os.path.join(SUPPLEMENTS_DIR, "FigureS13_shap_patterns.png"),
    "Figure S14": os.path.join(SUPPLEMENTS_DIR, "FigureS14_ig_mass_sensitivity.png"),
    "Figure S9": os.path.join(SUPPLEMENTS_DIR, "FigureS13_faithfulness.png"),
    "Figure S1": os.path.join(SUPPLEMENTS_DIR, "FigureS14_consort.png"),
}


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(TABLES_DIR, name))

def _pheno_label_with_count(pheno: str, totals_path: str) -> str:
    """Return e.g. 'Psychiatric (n=369)' if totals CSV exists, else just the name."""
    import pandas as pd
    if not os.path.exists(totals_path):
        return pheno
    df = pd.read_csv(totals_path)
    row = df[df["phenotype"] == pheno]
    if len(row) == 0:
        return pheno
    n = int(row.iloc[0]["n_unique_patients"])
    return f"{pheno} (n={n:,})"


# ── Front page ───────────────────────────────────────────────────────
# The contents listing is GENERATED from the headings the document
# actually contains, then moved to the front. It used to be a second
# hand-maintained list beside a separate "Contents" section, and the two
# drifted: they disagreed on wording for most items, and on the number of
# patterns in Table S5 (the table's own title said 11, the contents said
# 30, the CSV has 30). One generated list cannot disagree with itself.
H1_PT, H2_PT = 14.0, 12.0


def _is_heading(par, pt) -> bool:
    r = par.runs[0] if par.runs else None
    return bool(r and r.bold and r.font.size and r.font.size.pt == pt)


def _collect_items(doc):
    """Section names and figure/table titles, in document order."""
    items = []
    for par in doc.paragraphs:
        t = par.text.strip()
        if not t:
            continue
        if t in ("Supplementary Methods", "Supplementary Results") \
                and _is_heading(par, H1_PT):
            items.append(t)
        elif re.match(r"^Figure S\d+[A-C]?\. ", t) and _is_heading(par, H2_PT):
            items.append("Fig. " + t.split(" ", 1)[1])
        elif re.match(r"^Table S\d+\. ", t) and par.runs and par.runs[0].bold \
                and not _is_heading(par, H2_PT):
            items.append(t.rstrip("."))
    return items


def _prepend_front_page(doc) -> None:
    """Build the front page from the collected items, then move it to the
    top of the document."""
    items = _collect_items(doc)
    mark = len(doc.paragraphs)

    title(doc, "Supplementary Material")
    para(doc, "Anas Belouali, PhD, Hadi Kharrazi, MD, PhD")
    para(doc, "")
    para(doc, "**AToP: An Explainability Framework for Transformer-Based "
              "EHR Predictions Using Attribution-Guided Temporal Pattern "
              "Mining**")
    para(doc, "Submitted to Journal of the American Medical Informatics "
              "Association")
    para(doc, "")
    for it in items:
        para(doc, it)
    para(doc, "")
    para(doc, "*This supplementary material has been provided by the "
              "authors to give readers additional information about "
              "their work.*")
    page_break(doc)

    body = doc.element.body
    first = body[0]
    for par in doc.paragraphs[mark:]:
        el = par._p
        body.remove(el)
        first.addprevious(el)
    print(f"  front page: {len(items)} items generated from document headings")


def build() -> None:
    doc = new_document()

    # ── Extended Methods (relocated from main text for word budget) ──
    heading(doc, "Supplementary Methods", level=1)

    heading(doc, "Integrated Gradients reference baseline", level=2)
    para(doc, "Integrated Gradients was integrated against the model's "
              'pre-sigmoid logit z(x) with the all-[PAD] embedding as the'
              ' reference, at 30 path-integration steps (31 equally '
              'spaced points). Because the reference sequence keeps each '
              "patient's padding mask and position embeddings, its output"
              ' is patient-specific: across 300 randomly sampled test '
              'patients the reference logit z(x_baseline) had mean −0.27 '
              'and SD 0.38 (implied reference probability median 0.43, '
              'interquartile range 0.38 to 0.51) and correlated with '
              'sequence length (Spearman ρ = 0.76). The completeness '
              'residual, z(x) − z(x_baseline) − ΣIG, had a median '
              'absolute value of 0.012 logits (90th percentile 0.032), or'
              ' 1.8% of |z(x) − z(x_baseline)| (90th percentile 7.3%); '
              '100 steps would reduce the median to 0.5%. AToP reporting '
              'uses relative ΣIG rankings and within-model masking '
              'contrasts, so the conclusions do not depend on the '
              'absolute reference level. Masking effects Δŷ are '
              'differences of predicted probabilities, so recalibration '
              'of the model would change their magnitude but not their '
              'sign.')

    heading(doc, "Pattern scoring and position allocation", level=2)
    para(doc, 'Carrier-averaged ΣIG, used for all pattern rankings and '
              'tables, follows one rule for every pattern shape. A '
              "patient's record is indexed as, for each token, the visits"
              " in which it occurs. The pattern's steps are matched "
              'greedily from left to right: step k is assigned the '
              'earliest visit, strictly later than the visit assigned to '
              'step k − 1, in which all of its tokens occur; a patient '
              "with no such assignment is not a carrier. The patient's "
              "score is the sum, over steps and over each step's tokens, "
              "of the signed IG of the token's first occurrence in the "
              'assigned visit. A token that appears in two steps '
              'therefore contributes once per step, occurrences outside '
              'the assigned visits do not contribute, and all tokens are '
              'used, not only salient ones. Patterns are scored '
              'independently, so two patterns that share tokens both '
              'receive credit for them.')
    para(doc, 'The per-patient explanation display (Figure 2) instead '
              "partitions the patient's salient positions among the "
              'matched patterns. Patterns from the mined dictionary are '
              "matched to the patient's salient visit blocks, and each "
              'match claims every salient occurrence of its tokens in its'
              ' matched visits. Matches are then processed in order of '
              'earliest matched visit, then number of visit blocks (more '
              'first), then |ΣIG| (larger first); each match keeps only '
              'the positions not claimed by an earlier match and is '
              'dropped if none remain. The waterfall bars are the IG sums'
              ' over the kept positions, so they add up to the '
              'salient-token ΣIG, with the non-salient positions shown as'
              ' one remaining bar. Alternative priority orders would move'
              ' IG mass between overlapping patterns, which is why the '
              'bars are described as allocations.')

    heading(doc, "Phenotype family assignment", level=2)
    para(doc, "Phenotype families are assigned from each pattern's "
              'readable label by substring matching against keyword '
              'lists, applied in a fixed order in which the first '
              'matching family wins: Psychiatric (MDD, Depression NOS, '
              'Mood disorder, Bipolar, Manic, PTSD, Anxiety, schizo-, '
              'Psychosis, Delusional, SI, Suicide, self-harm, Brief '
              'psychotic, Risperidone, Olanzapine, Fluoxetine); Oncologic'
              ' (Cancer, metastases, Bone met, Pancreatic, Liver met, '
              'Lung met, Mal. pleural, Ascites, Chemo, neoplasm, '
              'Antineoplastic); Hepatobiliary (bile duct, biliary, ERCP, '
              'cholangitis, cholecystitis, cholestasis, cirrhosis, '
              'hepatitis, liver failure, gallbladder, gallstone, '
              'pancreatitis); Obstetric (Live birth, C-section, '
              'Dibucaine, Perineal, gestation, pregnancy, Oxytocin, OB '
              'lac, Bupivacaine, maternal, gestational, preterm labor, '
              'amniotic, multigravida); Chest pain workup (Chest pain '
              'NOS, Atypical chest pain, ECG, Chest X-ray, Resp '
              'measurements); Substance use (Alcohol intox, Alcohol '
              'abuse, Cannabis, Benzodiazepine, Acetamin OD, SSRI OD, '
              'Analgesic OD, self-poisoning); any other label is Other. '
              'Upper- and lower-case variants are included for several '
              'keywords. The order places malignancy above hepatobiliary '
              'disease and psychiatric diagnoses above substance use when'
              ' both occur in one pattern. Families are presentational '
              'groupings only and do not affect mining, attribution, or '
              'ranking; Table 3 shows the first four families listed '
              'above, and Table 4 additionally shows Hepatobiliary and '
              'Other. The lists are defined in make_table3.py in the '
              'released code.')

    heading(doc, "Preprocessing and vocabulary composition", level=2)
    para(doc, "ICD-9 diagnosis and procedure codes were mapped to ICD-10 "
              "with the CMS forward General Equivalence Mappings, keeping "
              "only entries flagged as unambiguous (no_map = 0, "
              "combination = 0); codes without such an entry were kept "
              "in ICD-9. "
              "Prescription drugs were mapped to RxNorm ingredient "
              "concepts (TTY = IN) via a two-stage pipeline against the "
              "RxNav REST API: (1) NDC codes (zero-padded to 11 digits) "
              "were resolved to their RxCUI and then to the ingredient "
              "concept; (2) for NDC misses and rows without an NDC, the "
              "drug name was matched to the closest ingredient concept. "
              "The resulting "
              "NDC-to-ingredient lookup covers 8,195 distinct NDC codes; "
              "records mapping to neither stage were dropped. Tokens with "
              "training-set frequency below min_freq = 2 were dropped "
              "(folded into [PAD] at encode time). The final 24,005-token "
              "vocabulary comprises 14,960 diagnosis tokens (ICD-9/10), "
              "7,877 procedure tokens (ICD-9/10), 1,165 RxNorm-ingredient "
              "drug tokens, and 3 special tokens ([VISIT]/[CLS]/[PAD]), "
              "shared in a single embedding space across all streams.")

    heading(doc, "Test-set use in deduplication", level=2)
    para(doc, "Patterns are mined exclusively on the training cohort. "
              "Jaccard deduplication and representative selection use "
              "test-cohort carrier sets because the main figures and "
              "tables report the test-cohort attribution landscape, and "
              "two patterns are redundant for that report precisely when "
              "they are carried by the same test patients. No model "
              "parameter, threshold, or pattern score is tuned on test "
              "outcomes; the test set is used only to summarize which "
              "mined patterns co-occur in the same patients. Deduplicating "
              "instead on training or validation carrier sets yields the "
              "same representative families, since the carrier-overlap "
              "structure is a property of clinical co-occurrence rather "
              "than of a particular split.")

    page_break(doc)

    # ── Supplementary Results ───────────────────────────────────────
    # Three validation analyses relocated here from the main Results to
    # meet the JAMIA 4,000-word body limit. The main text retains a
    # condensed statement of each headline number and points here.
    heading(doc, "Validation settings", level=2)
    para(doc, 'Masking validation covered the patterns displayed in '
              'Tables 3 and 4 and additional high-ranked risk-supporting '
              'and protective patterns selected by absolute ΣIG across '
              "the pipeline's validation runs, 266 unique patterns in "
              'total; each pattern was masked in up to 200 test-cohort '
              'carriers, taken in cohort order. The within-patient '
              'reversal analysis used up to 80 carriers per displayed '
              'cross-visit pattern, again the first in cohort order. The '
              'token-level faithfulness analysis drew its random baseline'
              ' five times per patient and reported the mean absolute '
              'change over the draws, the same statistic as the IG '
              'curves; 95% confidence intervals were obtained by '
              'bootstrapping patients with 1,000 resamples (random seed '
              '42). Because each k requires at least k salient tokens, '
              'the number of contributing patients falls from 500 at k ≤ '
              '3 to 463, 423, 395, 369, 338, 298 and 277 at k = 4 to 10. '
              'KernelSHAP used approximately 50 coalition samples per '
              'patient over the full analytic test cohort (43,602 '
              'patients); the per-patient comparison in Figure S11 used '
              '200.')
    para(doc, 'The two global rankings use different minimum carrier '
              'counts, set independently when each pool was implemented: '
              'the all-patterns pool was scored with a minimum of 5 '
              'test-cohort carriers, and the cross-visit dictionary was '
              'built with a minimum of 3. Neither value was optimized or '
              'tested for sensitivity. The minimum displayed carrier '
              'count was five in the all-patterns ranking and four in the'
              ' cross-visit ranking.')

    heading(doc, "Attribution direction and observed readmission", level=2)
    para(doc, 'To examine whether model-attributed cross-visit patterns '
              'were associated with observed readmission, we conducted a '
              'separate descriptive analysis. We restricted the test '
              'cohort to patients with at least two admissions, since '
              'cross-visit patterns require multiple visits. For each '
              'displayed cross-visit pattern, we fitted a logistic '
              'regression with readmission as the outcome and pattern '
              'carriage and the log-transformed number of visits as '
              'predictors, and report the visit-adjusted odds ratio for '
              'carriage with its 95% Wald confidence interval (Table S3).'
              ' No matching was performed. These associations were not '
              'used to evaluate attribution faithfulness.')

    heading(doc, "Supplementary Results", level=1)

    heading(doc, "Attribution direction vs observed outcome", level=2)
    para(doc, "Attribution direction is a statement about the model, and "
              "epidemiologic association is a statement about the cohort; "
              "the two need not agree, but a systematic disagreement would "
              "be worth explaining. Comparing them requires care, because "
              "cross-visit patterns can only be carried by patients with "
              "at least two admissions, and those patients are readmitted "
              "far more often than the cohort overall (27.8% vs 12.7%). "
              "Against the whole-cohort baseline nearly every cross-visit "
              "pattern therefore looks risk-increasing, including patterns "
              "the model treats as protective; this is an artifact of "
              "visit count, not a contradiction. Restricting to patients "
              "with two or more visits and adjusting for the number of "
              "visits, attribution direction and outcome direction agree "
              "for 25 of the 30 displayed cross-visit patterns: 15 of 19 "
              "risk-supporting patterns have an adjusted odds ratio above "
              "1, and 10 of 11 protective patterns have an adjusted odds "
              "ratio below 1 (Table S3). The obstetric case "
              "illustrates the correction: repeat cesarean delivery "
              "carries an unadjusted relative risk above 1 against the "
              "whole cohort, but an adjusted odds ratio of 0.83 within the "
              "multi-visit population, consistent with the protective "
              "attribution the model assigns. We report these odds ratios "
              "as descriptive context only. They are associational, "
              "several intervals are wide, and agreement between "
              "attribution and outcome is neither necessary for the "
              "attribution to be faithful to the model nor sufficient to "
              "establish that it is.")

    heading(doc, "Faithfulness of the attribution ranking", level=2)
    para(doc, "To test whether AToP's IG ranking surfaces the tokens the "
              "model actually relies on, we applied an ERASER-style "
              "progressive-masking protocol on a random sample of 500 "
              "test patients, contrasting AToP's top-k tokens against a "
              "matched-random baseline (Figure S9). Removing AToP's "
              "top-5 tokens changed the predicted readmission probability "
              "by a mean |\u0394\u0177| of 0.079, "
              "versus 0.056 for removing the same number of randomly "
              "chosen salient tokens (1.4 times larger; Wilcoxon signed-rank "
              "p < 0.001), with AToP's comprehensiveness curve above "
              "random at every k. Keeping only AToP's top-k tokens "
              "recovered the original prediction more completely than "
              "keeping random tokens (sufficiency; Figure S9B). These "
              "results indicate faithfulness to the model: the IG ranking "
              "identifies tokens that drive the prediction. They do not test "
              "whether those tokens are clinically correct, which we "
              "address qualitatively through the exemplars and leave to "
              "prospective clinician review.")

    heading(doc, "Cross-method comparison with KernelSHAP", level=2)
    para(doc, "To assess whether AToP's findings are properties of the "
              "aggregation convention rather than the choice of "
              "attribution method, we recomputed the rankings of Figures 3 "
              "and 4 with KernelSHAP in place of Integrated Gradients on the "
              "full test cohort (N = 43,602 patients with valid "
              "sequences, ~50 KernelSHAP coalitions per patient). "
              "The cohort-level cross-method comparison is "
              "reported in Figure S12 (token level) and Figure S13 "
              "(pattern level).")
    para(doc, "At the token level (Figure 3B equivalent), IG and "
              "SHAP carrier-conditional top-30 rankings agree on 25 of "
              "30 features (tokens with at least 10 sampled carriers). "
              "All fifteen tokens in SHAP's top-15 also appear in IG's "
              "top-30, twelve of them in IG's top-15 (Figure S12). The "
              "two methods identify essentially the same set of "
              "high-impact carrier-conditional features, with "
              "rank-shuffling between adjacent positions.")
    para(doc, "At the pattern level, agreement is 21 of 30 for the "
              "all-patterns ranking and 7 of 30 for the cross-visit-only "
              "ranking (Figure S13). The two methods therefore agree substantially on "
              "the all-patterns ranking but diverge on the cross-visit "
              "ranking. Disagreement between explanation methods applied "
              "to the same model is a well-documented and largely "
              "unresolved problem, so we treat the divergence as "
              "informative about the methods rather than decisive about "
              "the patterns; it reflects how the methods "
              "score multi-token patterns. A likely contributor is that "
              "KernelSHAP's interaction-aware "
              "coalition sampling dampens high-K sums via Shapley "
              "redistribution; IG's path-integral semantics sum per-token "
              "contributions additively, so K-token \u03a3 IG scales roughly "
              "linearly with K. Consistent with this, IG's cross-visit "
              "top-30 is weighted toward longer patterns than SHAP's "
              "(both span 2\u20134 tokens; mean K = 3.2 for IG vs 2.5 for "
              "SHAP). The token-level 25/30 agreement indicates that both "
              "methods identify largely the same high-impact features; "
              "the pattern-level differences reflect how the methods "
              "score sums over composite features.")

    page_break(doc)

    # ── Supplement Figures ──────────────────────────────────────────
    heading(doc, "Supplementary Figures", level=1)

    heading(doc, "Figure S1. Cohort and analysis flow", level=2)
    figure_placeholder(doc, "Figure S1",
        "CONSORT-style flow of the cohort and analysis. From MIMIC-IV "
        "v3.1 (546,028 hospital admissions), patients with at least one "
        "acute inpatient admission are assembled into one chronological "
        "token sequence (ICD-9/10 diagnoses and procedures, RxNorm "
        "drugs). In-hospital deaths (11,801 admissions, 2.2%) are "
        "excluded before labeling, since "
        "those patients cannot be readmitted. The 30-day readmission "
        "label excludes elective and "
        "surgical same-day admissions from the positive class (planned "
        "care, not unplanned readmission), yielding the analytic cohort "
        "of 218,196 patients with a 12.6% unplanned 30-day readmission "
        "rate. A patient-level 70/10/20 split gives train (152,737; "
        "12.6% readmit), validation (21,819), and test (43,640; 12.7% "
        "readmit). Pattern mining runs on training-cohort salient tokens "
        "(70,278 distinct episodes mined, 24,490 of them at whole-cohort "
        "support; 9,213 cross-visit representatives at j ≥ 0.2); "
        "per-token IG attribution is computed for all 43,640 test "
        "patients, of which 38 are excluded for an empty salient-token "
        "set, leaving the analytic attribution set of 43,602 patients "
        "used for all downstream AToP analyses. The two prevalence "
        "values (overall/train 12.6%, test 12.7%) are the same metric "
        "on different cohorts (rounding of 12.65% and 12.74%), not "
        "conflicting numbers.",
        image_path=IMAGE_PATHS.get("Figure S1"))

    page_break(doc)
    heading(doc, "Figure S2A. Alternate exemplar: psychiatric phenotype "
                  "(true positive)", level=2)
    figure_placeholder(doc, "Figure S2A",
        "Two-panel AToP narrative for a true-positive test patient with "
        "a psychiatric phenotype (recurrent psychosis NOS across "
        "consecutive admissions). Same format as main-paper Figure 2: "
        "(A) admission timeline showing the salient tokens each "
        "admission contributed to the Panel B patterns, marked by sign, "
        "plus up to two additional high-|ΣIG| salient tokens per "
        "admission in grey, with the index admission outlined; "
        "(B) allocated "
        "signed \u03a3 IG decomposition under the non-overlapping "
        "position-level allocation rule. Bars labeled V\u2099\u2192V\u2098 "
        "are cross-visit pattern matches; V\u2098: are single-visit. "
        "Examples in S4a and S4b were chosen to span phenotype families "
        "rather than to validate predictive accuracy.",
        image_path=IMAGE_PATHS.get("Figure S2A"))

    page_break(doc)
    heading(doc, "Figure S2B. Alternate exemplar: obstetric true negative",
             level=2)
    figure_placeholder(doc, "Figure S2B",
        "Two-panel AToP narrative for a true-negative test patient with "
        "an obstetric phenotype (uncomplicated delivery with current "
        "obstetric laceration repair). Same format as Figure 2 and S2a. "
        "The salient subsequence comprises obstetric procedure tokens; "
        "AToP surfaces the corresponding protective obstetric trajectory "
        "in the allocated \u03a3 IG panel and the model assigns a low "
        "predicted readmission probability.",
        image_path=IMAGE_PATHS.get("Figure S2B"))

    page_break(doc)
    heading(doc, "Figure S2C. Disagreement exemplar: oncology false "
                  "positive", level=2)
    figure_placeholder(doc, "Figure S2C",
        "Two-panel AToP narrative for a false-positive test patient: the "
        "model predicted readmission with high confidence (ŷ = 0.93); the "
        "patient did not readmit within 30 days. The patient has a "
        "metastatic-oncology trajectory across four pre-index visits "
        "(bone metastases at V1 and V3; cancer pain and cancer anemia at "
        "V4; chemotherapy-supportive medications). The prediction was "
        "driven primarily by the cross-visit Bone metastases → Bone "
        "metastases pattern (+0.240 allocated ΣIG) and a long tail of "
        "single-visit oncology tokens (+0.969 combined). Same two-panel "
        "format as Figure 2, S2A, and S2B. The explanation shows which "
        "patterns drove the high-confidence prediction: the elevated risk "
        "is attributed to cancer history rather than to tokens of acute "
        "deterioration.",
        image_path=IMAGE_PATHS.get("Figure S2C"))

    page_break(doc)
    heading(doc, "Figure S3. Token-level aggregation comparison", level=2)
    figure_placeholder(doc, "Figure S3",
        "Side-by-side comparison of the two atomic-token aggregation "
        "conventions on the test cohort. (A) Top-30 tokens by carrier-"
        "conditional E[IG | present] (AToP refinement). (B) Top-30 by "
        "population-weighted E[IG] = E[IG | present] × prevalence, the "
        "mean signed IG over all patients with non-carriers contributing "
        "zero. Each row is annotated with its rank "
        "under the alternative aggregation; bars with bold dark outlines "
        "are absent from the other panel's top-30. Of 30 top tokens by "
        "carrier-conditional aggregation, only 7 also appear in the "
        "population-weighted top-30 (n = 6,605 tokens scored). The 23 "
        "features that fall out are clinically interpretable markers "
        "buried deep in the population-weighted ranking: Acetaminophen overdose "
        "self-harm (population #356, carrier #2), Severe MDD (#271 vs "
        "#5), Bipolar manic episode (#166 vs #6), Recurrent severe MDD "
        "(#68 vs #7), and SSRI overdose self-harm (#799 vs #11), among "
        "others.",
        image_path=IMAGE_PATHS.get("Figure S3"))

    page_break(doc)
    heading(doc, "Figure S4. Sensitivity to Jaccard threshold, "
                  "cross-visit pool", level=2)
    figure_placeholder(doc, "Figure S4",
        "Cross-visit pattern pool (≥2 visits, ≥2 tokens) at four Jaccard "
        "deduplication thresholds. Panel A: no dedup (39,611 "
        "representatives). Panel C: j ≥ 0.6 (33,605). Panel E: j ≥ 0.2 "
        "(9,213, locked). Panel F: j ≥ 0.1 (2,075). Thresholds j ≥ 0.8 "
        "(38,074) and j ≥ 0.4 (24,606) are omitted; their top-30 sets are "
        "near-identical to j ≥ 0.6. Each panel shows the top-30 "
        "cross-visit patterns by |ΣIG| within the deduplicated set. "
        "Unlike Figure S5, the panels differ substantially from one "
        "another (A and E share only 8 of 30 patterns), which is itself "
        "the result: on the cross-visit pool the threshold materially "
        "changes what reaches the top. Loose thresholds leave the top-30 "
        "dominated by near-duplicate MDD/SI and obstetric variants; all "
        "six phenotype families surface independently by j ≥ 0.2. At "
        "j ≥ 0.1 that coverage survives in a much smaller pool (2,075), "
        "at the cost of more near-duplicate variants in the top-30.",
        image_path=IMAGE_PATHS.get("Figure S4"))

    page_break(doc)
    heading(doc, "Figure S5. Sensitivity to Jaccard threshold, all "
                  "patterns combined", level=2)
    figure_placeholder(doc, "Figure S5",
        "Jaccard-threshold sweep on the all-patterns combined pool "
        "(single-token, single-visit co-occurrence, and cross-visit "
        "patterns together). Representative counts: 70,275 (no dedup), "
        "61,059 (j ≥ 0.8), 53,623 (j ≥ 0.6), 40,869 (j ≥ 0.4), "
        "17,891 (j ≥ 0.2, locked), and 5,501 (j ≥ 0.1). Four are "
        "displayed (no dedup, 0.6, the locked 0.2, and 0.1); j ≥ 0.8 and "
        "j ≥ 0.4 are omitted, their top-30 sets being identical to "
        "j ≥ 0.6. Each column shows the top-30 patterns by |ΣIG| "
        "within that deduplicated representative set, drawn against a "
        "single shared label column so rows align across thresholds; a "
        "grey diamond marks a pattern absent from that threshold's "
        "top-30. Across every threshold, "
        "the top-30 is invariably dominated by single-visit co-"
        "occurrence patterns; single-token and cross-visit patterns "
        "never reach the top. This is the empirical justification for "
        "AToP's cross-visit and multi-token constraints (min_steps = 2, "
        "min_tokens = 2): without them AToP would recover only "
        "single-admission comorbidity rather than the multi-visit "
        "phenotype trajectories that motivate the framework.",
        image_path=IMAGE_PATHS.get("Figure S5"))

    page_break(doc)
    heading(doc, "Figure S6. Pattern-level aggregation comparison", level=2)
    figure_placeholder(doc, "Figure S6",
        "Pattern-level aggregation comparison across all patterns (single-"
        "token, single-visit co-occurrence, and cross-visit combined), "
        "Jaccard j = 0.2 (17,891 representatives). (A) Top-30 by carrier-"
        "averaged ΣIG. (B) Top-30 by population-weighted ΣIG "
        "(= carrier-averaged ΣIG × prevalence). Only 6 of 30 patterns "
        "overlap between the two rankings (vs 7 / 30 at the token "
        "level in Figure S3). Multi-token patterns are penalized "
        "harder by prevalence weighting because a pattern's prevalence "
        "is the intersection of its tokens' prevalences. The "
        "population-weighted top-30 is dominated by single-token "
        "features (MDD single ep. NOS, Chest pain NOS, Alcohol abuse, "
        "Ibuprofen, Thiamine, Psychosis NOS, etc.); AToP's multi-token "
        "compositional patterns sit at population ranks #3–#2058, so "
        "the aggregation penalty compounds at the pattern level.",
        image_path=IMAGE_PATHS.get("Figure S6"))

    page_break(doc)
    heading(doc, "Figure S7. Pattern prevalence and relative risk vs "
                  "attribution impact", level=2)
    figure_placeholder(doc, "Figure S7",
        "AToP's carrier-averaged |ΣIG| is partially independent of "
        "classical effect-size measures, surfacing patterns that "
        "prevalence-only or RR-only ranking would miss. n = 5,047 cross-"
        "visit patterns. Dots are red for risk (ΣIG > 0), blue for "
        "protective; dot area ∝ log(n test carriers). (A) Prevalence "
        "(% of test patients carrying the pattern, log scale) vs "
        "carrier-averaged |ΣIG|. The long left tail of rare-but-decisive "
        "patterns motivates AToP's carrier-conditional aggregation. The "
        "top-5 rare-decisive risk and top-5 protective patterns are "
        "marked with empty rings and listed in the upper-left textbox. "
        "(B) log₁₀(Relative Risk vs cohort baseline) vs |ΣIG|. Here RR "
        "is computed on the training cohort (baseline 30-day readmission "
        "rate 12.6%), the cohort the patterns were mined from. RR is "
        "unadjusted and, for cross-visit patterns, confounded by visit "
        "count; visit-adjusted odds ratios for the displayed cross-visit "
        "patterns are given in Table S3. Dashed vertical "
        "line marks RR = 1 (baseline). Spearman ρ (|ΣIG| vs "
        "log RR) = −0.13: patterns with comparable RR span a wide range of "
        "|ΣIG|, so effect size alone does not predict a pattern's "
        "attribution impact.",
        image_path=IMAGE_PATHS.get("Figure S7"))

    page_break(doc)
    heading(doc, "Figure S8. Per-pattern masking effects", level=2)
    figure_placeholder(doc, "Figure S8",
        "Masking effect \u0394\u0177 = \u0177_original \u2212 \u0177_masked for each "
        "of the 266 perturbed patterns, averaged over the pattern's "
        "test-cohort carriers. Bars are colored by the pattern's attribution "
        "direction, taken from the test-cohort carrier-averaged \u03a3IG "
        "under the scoring rule of the Supplementary Methods: red, "
        "risk-supporting (\u03a3IG > 0); blue, protective (\u03a3IG < 0). Masking "
        "moved predictions in the direction of the pattern's attribution for "
        "all 266 patterns (168 risk-supporting, 98 protective). Masking creates out-of-distribution inputs, so it is a "
        "proxy test; the direct token-level faithfulness test is the "
        "comprehensiveness and sufficiency analysis in Figure S9.",
        image_path=IMAGE_PATHS.get("Figure S8"))

    page_break(doc)
    heading(doc, "Figure S9. Quantitative faithfulness of AToP-ranked "
                  "tokens", level=2)
    figure_placeholder(doc, "Figure S9",
        "Faithfulness of AToP's IG-based token ranking, measured by "
        "progressive masking on a random sample of 500 test patients "
        "(≥3 salient tokens each), contrasted against a matched-random "
        "baseline (same number of tokens, drawn at random from the "
        "patient's salient set; mean absolute change over 5 draws). Each k "
        "includes only patients with at least k salient tokens (500 at "
        "k ≤ 3, falling to 277 at k = 10). |Δŷ| is the "
        "absolute change in predicted readmission probability vs the "
        "unmasked sequence; shaded bands are 95% bootstrap CIs over "
        "patients. (A) Comprehensiveness: removing the top-k tokens by "
        "|IG|. Removing AToP's top-5 tokens moves the prediction by a "
        "mean |Δŷ| = 0.079 versus 0.056 for random removal (1.4× larger; "
        "Wilcoxon p < 0.001), and AToP's curve lies above random with "
        "non-overlapping CIs at every k. (B) Sufficiency: keeping only "
        "the top-k tokens and masking the patient's remaining salient tokens. AToP's curve lies below "
        "random, i.e. the top-k AToP tokens recover the original "
        "prediction more completely than the same number of random "
        "tokens. Together these indicate that the IG ranking identifies "
        "tokens the model relies on (faithfulness to the model). This "
        "does not test clinical correctness of those tokens, which "
        "requires separate clinician review.",
        image_path=IMAGE_PATHS.get("Figure S9"))

    page_break(doc)
    heading(doc, "Figure S10. Top 30 cross-visit patterns by reversal "
                  "attenuation", level=2)
    figure_placeholder(doc, "Figure S10",
        "Top 30 cross-visit patterns ranked by reversal attenuation. For "
        "each pattern, the darker bar shows the carrier-averaged signed "
        "ΣIG for test-cohort patients matching the forward token order; "
        "the lighter bar shows the same statistic for the reversed token "
        "order. The annotated percentage = 100 × (1 − |ΣIG_rev| / "
        "|ΣIG_fwd|) is the attenuation. Red = risk-supporting (forward "
        "ΣIG > 0), blue = protective. The 30 patterns shown were selected "
        "from the 1,714-pattern floored pool (n forward ≥ 20 and "
        "n reversed ≥ 20). Across this pool 95% of patterns preserve "
        "sign; among the 30 highest-attenuation patterns shown here, "
        "attenuation ranges 84%-99%, and 15 preserve sign while 11 flip "
        "and 4 attenuate to exactly zero. This tail behaves differently "
        "from the pool by construction: these are the patterns whose "
        "reversed-order ΣIG has fallen closest to zero, so it is where "
        "the residual can vanish or change sign. The strongest temporal "
        "sensitivities include Prochlorperazine → Ibuprofen (99%), "
        "Anxiety disorder NOS → Insulin Lispro (99%), and {AKI NOS; "
        "Thiamine} → Polyethylene Glycol (98%). Per-pattern numerical "
        "results are in Table S4, with a phenotype-organized cross-walk "
        "to the main-paper Table 4 patterns in Table S5.",
        image_path=IMAGE_PATHS.get("Figure S10"))

    page_break(doc)
    heading(doc, "Figure S11. Per-patient SHAP vs IG agreement", level=2)
    figure_placeholder(doc, "Figure S11",
        "Per-patient agreement between Integrated Gradients (IG) and "
        "KernelSHAP attribution methods. (A) Exemplar patient (true "
        "positive, recurrent psychosis): per-token signed IG vs signed "
        "KernelSHAP for top 20 tokens by |IG|; 6 most extreme tokens "
        "labeled. Pearson r = 0.52, Spearman ρ = 0.43. (B) Aggregate "
        "scatter pooling top 10 tokens (by |IG|) from 100 stratified-"
        "random test patients, z-scored on both axes so per-method scale "
        "differences do not obscure rank/sign agreement. n = 935 (patient, "
        "token) pairs. Pearson r = 0.67, Spearman ρ = 0.45. Per-patient "
        "Jaccard@5 = 0.57 (mean), indicating substantial salient-token "
        "rank agreement. Red = risk-supporting (IG > 0), blue = protective; "
        "dashed line = y = x. KernelSHAP here used 200 coalition samples "
        "per patient on the top-K-by-IG token subset for high-precision "
        "per-patient values; this differs from the cohort-level comparison "
        "in Figures S12 and S13, which used ~50 coalitions per patient to keep the "
        "full 43,602-patient run tractable. Cohort-level cross-method "
        "overlap statistics are reported in Figures S12 and S13.",
        image_path=IMAGE_PATHS.get("Figure S11"))

    page_break(doc)
    heading(doc, "Figure S12. Token-level attribution with KernelSHAP",
             level=2)
    figure_placeholder(doc, "Figure S12",
        "KernelSHAP counterpart of Figure 3 on the full test cohort "
        "(N = 43,602 patients with valid sequences): population-weighted "
        "(A) and carrier-conditional (B) mean SHAP per token, top 30 by "
        "absolute value, with the same aggregation conventions as the IG "
        "ranking; tokens with fewer than 10 sampled carriers were excluded "
        "before ranking. The carrier-conditional top-30 shares 25 of 30 "
        "tokens with the IG top-30, so the token-level findings are not "
        "specific to IG.",
        image_path=IMAGE_PATHS.get("Figure S12"))

    # Figure S13 is drawn at landscape print size: figure_landscape opens a
    # landscape section (heading, image, caption on one page) and returns to
    # portrait on a new page, so Figure S14 needs no page break.
    figure_landscape(doc, "Figure S13",
        "Figure S13. Pattern-level attribution with KernelSHAP",
        "KernelSHAP counterpart of Figure 4 on the full test cohort: "
        "carrier-averaged \u03a3SHAP for all patterns (A) and for cross-visit "
        "patterns (B), top 30 by absolute value after Jaccard \u2265 0.2 "
        "deduplication; patterns with fewer than 10 sampled carriers were "
        "excluded before ranking. Agreement with the IG rankings is 21 of "
        "30 for all patterns and 7 of 30 for cross-visit patterns. The "
        "lower cross-visit agreement points to a difference in how the "
        "methods score multi-token compositional patterns: a likely "
        "contributor is that KernelSHAP's coalition sampling redistributes "
        "interaction credit, whereas IG sums per-token contributions "
        "additively, so multi-token sums scale with K. The divergence is "
        "concentrated in the ordering of cross-visit patterns.",
        image_path=IMAGE_PATHS.get("Figure S13"))

    heading(doc, "Figure S14. Sensitivity to the IG saliency-mass threshold",
             level=2)
    figure_placeholder(doc, "Figure S14",
        "Four-panel 2×2 sensitivity analysis of the AToP downstream pipeline "
        "to the IG saliency-mass threshold τ. The default τ = 0.80 (per-"
        "patient cumulative |IG| mass cap) is varied across τ ∈ {0.60, "
        "0.70, 0.80, 0.90, 0.95}; the mined pattern dictionary is re-"
        "matched at each setting. (A) Total patterns matched on ≥1 test "
        "patient is essentially flat across τ (24,350 → 24,487; < 1% "
        "variation). (B) Per-patient salient-token count scales linearly "
        "with τ (median 5 → 9 → 15), consistent with the patient-adaptive "
        "property of cumulative-mass thresholding. (C) Top-30 risk-pattern "
        "Jaccard overlap matrix: τ = 0.80 shares 0.82 with τ = 0.70 and "
        "0.76 with τ = 0.90; extremes (0.60 vs 0.95) overlap at 0.67. "
        "(D) Top-30 protective-pattern overlap is lower (0.62-0.71 "
        "adjacent; 0.20 extremes), reflecting noisier protective-side "
        "ranking. Top risk patterns and their ΣIG ranking are robust to the "
        "choice of saliency-mass threshold τ; the protective side is more "
        "sensitive, consistent with panel D.",
        image_path=IMAGE_PATHS.get("Figure S14"))

    page_break(doc)

    # ── Supplement tables ──────────────────────────────────────────
    heading(doc, "Supplementary Tables", level=1)

    # ── Table S1: parameter / hyperparameter table ─────────────────
    page_break(doc)
    df_s4 = _read_csv("TableS4_parameters.csv")
    table_title(doc, "Table S1",
                 "AToP parameters and hyperparameters: values, prespecified-"
                 "vs-tuned status, and rationale.")
    s4_rows = [[_fmt_str(r[c]) for c in df_s4.columns] for _, r in df_s4.iterrows()]
    make_table(doc,
                header_row=list(df_s4.columns),
                data_rows=s4_rows,
                col_widths_in=[1.7, 1.0, 1.2, 3.0],
                font_size_pt=8)
    table_legend(doc,
        abbreviations=(
            "IG = integrated gradients; ΣIG = sum of token IG values "
            "for a pattern; τ = IG saliency-mass threshold; j = Jaccard "
            "carrier-set overlap threshold."
        ),
        note=(
            "Prespecified = fixed before the analysis from "
            "tractability or design considerations; Design choice = a "
            "deliberate methodological convention (not tuned on "
            "results); Tuned by inspection = set by examining "
            "training-cohort output, without reference to any outcome "
            "label or to the test set. No parameter was selected by "
            "optimizing a test-set outcome. Sensitivity sweeps for the saliency-mass "
            "threshold and the Jaccard threshold are in Figures S14 and "
            "S4\u2013S5, respectively."
        ),
    )

    page_break(doc)
    # ── New Table S2: AToP pattern mining pipeline summary ──
    # (Moved from main-paper Table 2 in this revision; current main-paper
    # Table 2 is the model-performance comparison instead. The bootstrap-
    # AUROC CIs that used to be Table S2 are now subsumed into the
    # main-paper Table 2.)
    df_s1 = _read_csv("Table2_mining_summary.csv")
    table_title(doc, "Table S2", "AToP pattern mining pipeline summary.")
    _rows = []
    _last = None
    for _, r in df_s1.iterrows():
        if r["section"] != _last:
            _rows.append([_fmt_str(r["section"]), ""])
            _last = r["section"]
        _rows.append([f"    {_fmt_str(r['metric'])}", _fmt_str(r["value"])])
    make_table(doc,
                header_row=["Statistic", "Count"],
                data_rows=_rows,
                col_widths_in=[4.5, 1.8],
                font_size_pt=9)
    table_legend(doc,
        abbreviations=(
            "IG = integrated gradients; ΣIG = sum of token IG values "
            "for a pattern; SD = standard deviation."
        ),
        note=(
            "Token compression ratio = (mean full sequence length) / "
            "(mean salient subsequence length after the 80% IG-mass "
            "cutoff), per patient. The count blocks are distinct stages, "
            "not a running total. Episodes are mined separately in "
            "readmitted and non-readmitted training patients at support "
            "0.001 of each class (20 and 134 patients) and pooled; the "
            "pooled set is the candidate pool scored on the test cohort "
            "for the global rankings (at least 5 test carriers for the "
            "all-patterns view, at least 3 for the cross-visit view). The "
            "24,490 pooled episodes that also reach support 0.001 of the "
            "whole training cohort (153 patients) when re-matched across "
            "both classes form the dictionary used for per-patient "
            "explanations; the pool is larger because rare episodes "
            "enriched in readmitted patients clear the per-class "
            "threshold but not the whole-cohort one. Representatives are "
            "the two pools after Jaccard deduplication at j ≥ 0.2 on "
            "test carrier sets. The 9,213-pattern cross-visit "
            "representative set is the locked dictionary used for all "
            "cross-visit reporting (Figures 4-5, Table 4); Figures S4 and "
            "S5 sweep the Jaccard threshold for the cross-visit and "
            "full pools respectively. Risk patterns: ΣIG > 0 (push "
            "prediction toward readmission); protective: ΣIG < 0. Model "
            "performance (AUROC, PR-AUC, ΔAUROC vs baselines) is reported "
            "in the main-paper Table 2."
        ),
    )

    page_break(doc)
    # ── Table S3: visit-adjusted odds ratios ────────────────
    page_break(doc)
    df_s6 = _read_csv("TableS6_adjusted_or.csv")
    table_title(doc, "Table S3",
                 "Visit-adjusted odds ratios for the "
                 "cross-visit patterns displayed in main-paper Table 4.")
    s6_rows = []
    _last = None
    for _, r in df_s6.iterrows():
        ph = _fmt_str(r["phenotype"])
        s6_rows.append([
            ph if ph != _last else "",
            _fmt_str(r["label"]),
            f"{r['ig_value']:+.2f}",
            _fmt_str(r["direction"]),
            _fmt_int(r["n_carriers_multivisit"]),
            f"{r['carrier_readmit_pct']:.1f}%" if pd.notna(r["carrier_readmit_pct"]) else "n/a",
            f"{r['matched_baseline_pct']:.1f}%" if pd.notna(r["matched_baseline_pct"]) else "n/a",
            _fmt_str(r["adj_OR"]),
            _fmt_str(r["ci95"]),
        ])
        _last = ph
    make_table(doc,
                header_row=["Phenotype", "Pattern (representative)", "ΣIG",
                             "Direction", "n", "Carrier readmit %",
                             "Non-carrier readmit %", "Adj. OR", "95% CI"],
                data_rows=s6_rows,
                col_widths_in=[0.95, 1.95, 0.45, 0.62, 0.35, 0.62, 0.68, 0.5, 0.78],
                font_size_pt=7)
    table_legend(doc,
        abbreviations=(
            "ΣIG = carrier-averaged signed sum of constituent-token IG "
            "values (logit space); Direction = sign of ΣIG (risk = "
            "ΣIG > 0, protective = ΣIG < 0); n = pattern carriers within "
            "the multi-visit population; Carrier readmit % = observed "
            "30-day readmission rate among those carriers; Non-carrier "
            "readmit % = readmission rate among non-carriers in the same "
            "population; Adj. OR = adjusted odds ratio for carriage; "
            "CI = confidence interval."
        ),
        note=(
            "This table replaces the crude relative risk shown in earlier "
            "drafts of Tables 3 and 4. Cross-visit patterns can only be "
            "carried by patients with at least two admissions, and such "
            "patients are readmitted far more often than the cohort as a "
            "whole (27.8% vs 12.7%), so a relative risk computed against "
            "the whole-cohort baseline is confounded by visit count and "
            "systematically overstates risk, most visibly by making "
            "protective patterns appear risk-increasing. Here the "
            "population is restricted to patients with two or more visits "
            "and we fit logit P(readmission) = b0 + b1(carrier) + "
            "b2 log(n visits) by maximum likelihood, reporting exp(b1) "
            "with a 95% Wald interval. Under this visit-adjusted "
            "adjustment the attribution direction and the epidemiologic "
            "direction agree for 25 of 30 displayed patterns (15 of 19 "
            "risk-supporting patterns have OR > 1; 10 of 11 protective "
            "patterns have OR < 1), compared with near-zero agreement for "
            "protective patterns under the unadjusted cohort-baseline "
            "relative risk. Intervals are wide for patterns with few "
            "carriers and these rows are descriptive; no multiplicity "
            "correction is applied, and the odds ratios are associational "
            "summaries of the cohort, not causal effect estimates and not "
            "a validation of the attribution method."
        ),
    )

    page_break(doc)
    # ── Table S4: top 30 by reversal attenuation (full mined pool) ──
    page_break(doc)
    df_s2 = _read_csv("TableS2_reversed_order_top30.csv")
    table_title(doc, "Table S4",
                 "Top 30 cross-visit patterns ranked by reversal attenuation, "
                 "selected from the 1,714-pattern floored pool "
                 "(n forward-matched ≥ 20 and n reverse-matched ≥ 20), the "
                 "same pool Figure S10 displays.")
    s2_rows = []
    for _, r in df_s2.iterrows():
        s2_rows.append([
            _fmt_str(r["readable_label"]),
            _fmt_int(r["n_blocks"]),
            _fmt_str(r["phenotype"], default="(unclassified)"),
            _fmt_str(r["source"], default="Topup pool"),
            _fmt_int(r["n_forward"]),
            _fmt_int(r["n_reversed"]),
            f"{r['ig_mean_forward']:+.3f}" if pd.notna(r["ig_mean_forward"]) else "n/a",
            f"{r['ig_mean_reversed']:+.3f}" if pd.notna(r["ig_mean_reversed"]) else "n/a",
            f"{r['attenuation_pct']:.1f}%" if pd.notna(r["attenuation_pct"]) else "n/a",
        ])
    make_table(doc,
                header_row=["Pattern (representative)", "#blk", "Phenotype",
                             "Source", "n fwd", "n rev",
                             "ΣIG fwd", "ΣIG rev", "Attn"],
                data_rows=s2_rows,
                col_widths_in=[2.20, 0.40, 0.95, 0.85, 0.45, 0.45, 0.55, 0.55, 0.50],
                font_size_pt=8)
    table_legend(doc,
        abbreviations=(
            "#blk = number of visit blocks in the pattern (≥2 by "
            "construction; the reversed-order test contrasts forward vs "
            "full reversal. For ≥3-block patterns, intermediate "
            "permutations are not exhaustively tested). "
            "Phenotype = main-paper classification family if the "
            "pattern appeared in Table 4 or Figure 4B; otherwise "
            "blank (the topup pool was not constrained to the seven "
            "manuscript phenotype keywords). "
            "Source = Table 4 (main-paper displayed table); "
            "Fig 4B (main-paper figure); Topup pool (added for this "
            "supplement-only top-N analysis). "
            "n fwd / n rev = test-cohort patients matching the forward / "
            "reverse token order. "
            "ΣIG fwd / rev = carrier-averaged signed ΣIG for each match "
            "group. "
            "Attn (attenuation) = 100 × (1 − |ΣIG rev| / |ΣIG fwd|); "
            "positive values indicate reversal weakens the model's "
            "attribution. "
            "Rows sorted by descending attenuation."
        ),
        note=(
            "Selection: the reversed-order analysis evaluated 3,447 "
            "candidate cross-visit patterns (cross-visit representatives "
            "with test-cohort prevalence n ≥ 100, drawn from the "
            "40,327 scored cross-visit patterns), of which 3,429 had "
            "valid forward and reversed matches. The ranking and display "
            "pool (this table and Figure S10) comprises the 1,714 patterns "
            "with both n_forward ≥ 20 and n_reversed ≥ 20, so the "
            "attenuation ratio (|ΣIG rev| / |ΣIG fwd|) reflects model "
            "behavior on sufficiently supported orderings rather than "
            "match-count sparsity. Without these floors the top of the "
            "ranking is dominated by patterns whose forward or reverse "
            "ordering rarely occurs (e.g., match count 1-3, ΣIG ≈ 0). "
            "Displayed patterns are phenotype-classified with the same "
            "keyword list used for main-paper Tables 3-4. "
            "Table S5 lists the 30 displayed Table-4 cross-visit patterns "
            "specifically (cross-walk to "
            "the main paper). Full per-pattern "
            "data for all 3,447 tested patterns is provided in the "
            "supplementary data release."
        ),
    )

    page_break(doc)
    # ── Table S5: Table-4 cross-walk (11 cross-visit patterns) ─────
    page_break(doc)
    df_s3 = _read_csv("TableS3_reversed_order_table4.csv")
    table_title(doc, "Table S5",
                 "Per-pattern reversed temporal order results for the 30 "
                 "cross-visit patterns from main-paper Table 4 "
                 "(direct cross-walk).")
    s3_rows = []
    for _, r in df_s3.iterrows():
        s3_rows.append([
            _fmt_str(r["phenotype"]),
            _fmt_int(r["n_blocks"]),
            _fmt_str(r["readable_label"]),
            _fmt_int(r["n_forward"]),
            _fmt_int(r["n_reversed"]),
            f"{r['ig_mean_forward']:+.3f}" if pd.notna(r["ig_mean_forward"]) else "n/a",
            f"{r['ig_mean_reversed']:+.3f}" if pd.notna(r["ig_mean_reversed"]) else "n/a",
            f"{r['attenuation_pct']:.1f}%" if pd.notna(r["attenuation_pct"]) else "n/a",
        ])
    make_table(doc,
                header_row=["Phenotype", "#blk", "Pattern (representative)",
                             "n fwd", "n rev", "ΣIG fwd", "ΣIG rev", "Attn"],
                data_rows=s3_rows,
                col_widths_in=[1.05, 0.40, 2.40, 0.55, 0.55, 0.65, 0.65, 0.55],
                font_size_pt=8)
    table_legend(doc,
        abbreviations=(
            "Same column definitions as Table S4 (Pattern, #blk, n fwd, "
            "n rev, ΣIG fwd / rev, Attn). Patterns sorted by phenotype "
            "family in the main-paper Table 4 ordering, and within each "
            "family by descending |ΣIG fwd|."
        ),
        note=(
            "Aggregate result for this displayed set: of the 30 patterns, "
            "3 are symmetric (palindromic block sequences for which "
            "reversal returns the same pattern, attenuation 0 by "
            "construction) and are excluded as non-informative. Among the "
            "remaining 27 asymmetric patterns, 26 (96%) "
            "show |ΣIG forward| > |ΣIG reversed|; all 30 preserve sign; "
            "among those 26, mean attenuation is 32% and median 28% "
            "(Wilcoxon signed-rank p < 0.001). One pattern shows a small "
            "increase under reversal. n forward and ΣIG forward are "
            "computed over the forward-order-matching carrier subset, so "
            "they differ from the all-carrier n and ΣIG in main-paper "
            "Table 4. Several patterns "
            "have small reverse-order support (n reversed as low as 2), so "
            "their individual attenuation is descriptive; the population "
            "temporal-order statistic is taken from the floored sweep "
            "(Figure S10, Table S4), not these curated rows."
        ),
    )

    # ── Table S6: within-patient reversal of the Table 4 patterns ──────
    page_break(doc)
    df_s6w = _read_csv("TableS6_within_patient_reversal.csv")
    table_title(doc, "Table S6",
                "Within-patient reversal of the 30 cross-visit patterns in "
                "main-paper Table 4.")
    make_table(doc,
               header_row=["Family", "Pattern", "n", "\u03a3IG obs.",
                           "\u03a3IG rev.", "Atten. %", "Fell %",
                           "Sign %", "p"],
               data_rows=[[_fmt_str(r["Phenotype"]), _fmt_str(r["Pattern"]),
                           _fmt_int(r["n carriers"]),
                           _fmt_num(r["Mean \u03a3IG, observed order"], "+.3f"),
                           _fmt_num(r["Mean \u03a3IG, visits reversed"], "+.3f"),
                           _fmt_num(r["Attenuation (%)"], ".1f"),
                           _fmt_num(r["Carriers with reduced |\u03a3IG| (%)"], ".1f"),
                           _fmt_num(r["Sign preserved (%)"], ".0f"),
                           _fmt_str(r["Wilcoxon p"])]
                          for _, r in df_s6w.iterrows()],
               col_widths_in=[0.8, 2.3, 0.35, 0.55, 0.55, 0.5, 0.5, 0.45, 0.5],
               font_size_pt=8)
    table_legend(doc,
        abbreviations=("Family = phenotype family; n = carriers tested (first 80 test-cohort carriers "
                       "in cohort order); \u03a3IG observed / reversed = carrier-"
                       "averaged pattern score (obs.) on the patient's record as "
                       "observed and after reversing the order of the "
                       "pattern's matched visits within that record; Atten. = "
                       "1 \u2212 |\u03a3IG rev.| / |\u03a3IG obs.|; Fell = "
                       "share of carriers whose |\u03a3IG| fell; Sign = share "
                       "whose score kept its sign; p = one-sided Wilcoxon "
                       "signed-rank test of |\u03a3IG observed| > |\u03a3IG reversed| "
                       "over carriers."),
        note=('Unlike Table S5, which compares different patients '
              'carrying the forward and reversed orders, this analysis '
              "reorders the matched visits inside each carrier's own "
              'record and recomputes IG; all other visits and the '
              'contents of every visit are unchanged. Because whole '
              'visits are moved, the other events in those visits move '
              'with the pattern tokens, so the analysis tests sensitivity'
              ' to visit order rather than isolating the order of the '
              "pattern's own events. For the three palindromic patterns "
              'the pattern reads the same in either direction and any '
              'change reflects the other visit contents, so they are '
              'excluded from the summary counts. Pooled over the 27 '
              'non-palindromic patterns (925 carrier-pattern pairs from '
              '722 distinct patients, 166 of whom carry more than one '
              'pattern), |ΣIG| fell for 60% of pairs and sign was '
              "preserved for 97%; a patient-level test on each patient's "
              'mean over their patterns gives one-sided Wilcoxon p < '
              '0.001 (n = 722). Per-pattern p values are nominal; 4 of 27'
              ' remain below 0.05 after Holm correction.'))

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    _prepend_front_page(doc)
    doc.save(OUT_PATH)
    n_para = len(doc.paragraphs)
    n_table = len(doc.tables)
    print(f"\u2713 Wrote {OUT_PATH}")
    print(f"  Paragraphs: {n_para},  Tables: {n_table}")


if __name__ == "__main__":
    build()
