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
    page_break, hrule, set_body_line_spacing,
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
    "Figure S9":  os.path.join(SUPPLEMENTS_DIR, "FigureS10_reversal_scatter.png"),
    "Figure S10":  os.path.join(SUPPLEMENTS_DIR, "FigureS3_shap_vs_ig.png"),  # may not exist; placeholder fallback
    "Figure S2A": os.path.join(SUPPLEMENTS_DIR, "FigureS4A_alt_tp.png"),
    "Figure S2B": os.path.join(SUPPLEMENTS_DIR, "FigureS4B_confident_tn.png"),
    "Figure S2C": os.path.join(SUPPLEMENTS_DIR, "FigureS4C_disagreement.png"),
    "Figure S6":  os.path.join(SUPPLEMENTS_DIR, "FigureS5_prevalence_vs_impact.png"),
    "Figure S7": os.path.join(SUPPLEMENTS_DIR, "FigureS10_masking_distribution.png"),
    "Figure S3":  os.path.join(SUPPLEMENTS_DIR, "FigureS7_jaccard_sensitivity.png"),
    "Figure S4":  os.path.join(SUPPLEMENTS_DIR, "FigureS8_full_pool_jaccard_sensitivity.png"),
    "Figure S5": os.path.join(SUPPLEMENTS_DIR, "FigureS10_pattern_aggregation_comparison.png"),
    "Figure S11": os.path.join(SUPPLEMENTS_DIR, "FigureS12_shap_tokens.png"),
    "Figure S12": os.path.join(SUPPLEMENTS_DIR, "FigureS13_shap_patterns.png"),
    "Figure S13": os.path.join(SUPPLEMENTS_DIR, "FigureS14_ig_mass_sensitivity.png"),
    "Figure S8": os.path.join(SUPPLEMENTS_DIR, "FigureS13_faithfulness.png"),
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
# patterns in Table S4 (the table's own title said 11, the contents said
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


def add_table_s2(doc) -> None:
    """Table S2 (mining pipeline summary) with title and legend. Shared with build_tables_only_docx.py."""
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
                header_row=["Statistic", "Value"],
                data_rows=_rows,
                col_widths_in=[4.5, 1.8],
                font_size_pt=9)
    table_legend(doc,
        abbreviations=(
            "IG = Integrated Gradients; ΣIG = sum of token IG "
            "attributions; SD = standard deviation."
        ),
        note=(
            "Episodes were mined separately in readmitted and "
            "non-readmitted training patients (support 0.001 of each "
            "class, 20 and 134 patients) and pooled. The per-patient "
            "explanation dictionary is the subset also reaching "
            "support 0.001 of the whole training cohort (153 "
            "patients); the pool is larger because rare episodes "
            "enriched in readmitted patients clear the per-class "
            "threshold but not the whole-cohort one. Global rankings "
            "used the full pooled set. Token compression ratio is the "
            "mean full sequence length divided by the mean salient "
            "subsequence length per patient. Counts represent different stages of the pipeline rather than a running total. Test-carrier counts are full-record carriers."
        ),
    )

def build() -> None:
    set_body_line_spacing(1.5)   # supplement: 1.5 spacing (main stays 2.0)
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
              '100 steps would reduce the median to 0.5%.')
    para(doc, 'Per-patient explanations and all pattern analyses use '
              'the attribution on clinical tokens only; attribution '
              'on the [CLS] and [VISIT] positions is not displayed. '
              'Across the same 300 test patients, these structural '
              'positions carried a median 46% of the total absolute '
              'attribution (above 25% in 69% of patients), with no '
              'consistent sign ([CLS] negative on average; [VISIT] '
              'positive in 68% of patients). The displayed '
              'clinical-token sums are therefore partial sums; the '
              'full identity, including the reference logit and the '
              'structural-token share, is reported for each example '
              'in the legends of Figures 2 and S2A to S2C.')

    heading(doc, "Pattern scoring and position allocation", level=2)
    para(doc, 'Carrier-averaged ΣIG, used for all pattern rankings '
              'and tables, follows one rule for every pattern shape. '
              "A patient's record is indexed as, for each token, the "
              "visits in which it occurs. The pattern's steps are "
              'matched greedily from left to right: step k is '
              'assigned the earliest visit, strictly later than the '
              'visit assigned to step k − 1, in which all of its '
              'tokens occur; a patient with no such assignment is '
              "not a carrier. The patient's score is the sum, over "
              "steps and over each step's tokens, of the signed IG "
              "of the token's first occurrence in the assigned "
              'visit. A token that appears in two steps therefore '
              'contributes once per step, occurrences outside the '
              'assigned visits do not contribute, and all tokens are '
              'used, not only salient ones. Patterns are scored '
              'independently, so two patterns that share tokens both '
              'receive credit for them. Because mining used salient '
              'subsequences whereas carriers are matched on full '
              'records, a carrier need not be matchable on its '
              'salient tokens alone. Among the 1,230 carrier-pattern '
              'pairs of the 30 patterns in Table 4, every pattern '
              'token was salient somewhere in the record in 836 '
              '(68%); restricting carriers to those patients '
              'preserved the sign of all 30 carrier-averaged ΣIG '
              'values (Spearman ρ of |ΣIG| 0.97; median change +4%).')
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

    heading(doc, "Clinical group assignment", level=2)
    para(doc, "Clinical groups are assigned from each pattern's "
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
              ' both occur in one pattern. The keyword rules were '
              'developed from recurring clinical themes observed among '
              'the discovered patterns and were applied consistently to '
              'organize patterns for presentation. Clinical group '
              'assignments did not influence pattern discovery, '
              'attribution, or ranking. Table 3 displays psychiatric, '
              'oncologic, obstetric, and chest-pain groups. Table 4 '
              'additionally includes hepatobiliary and other patterns. '
              'The lists are defined in make_table3.py in the released '
              'code.')

    heading(doc, "Preprocessing and vocabulary composition", level=2)
    para(doc, 'ICD-9 diagnosis and procedure codes were mapped to '
              'ICD-10 with the CMS General Equivalence Mappings '
              '[28]: after removing no-map and combination entries, '
              'only ICD-9 codes with a single ICD-10 target were '
              'converted, which harmonized 80.4% of ICD-9 diagnosis '
              'rows (6,719 of 9,143 codes) and 14.2% of ICD-9 '
              'procedure rows (255 of 2,557 codes); the remaining '
              'codes were kept in ICD-9 (1,863 diagnosis and 1,807 '
              'procedure tokens in the vocabulary). Chronic '
              'conditions, defined by a fixed ICD-prefix list '
              'released with the code (metabolic, cardiovascular, '
              'respiratory, renal, gastrointestinal, neurological, '
              'psychiatric and substance-use disorders, '
              'malignancies, HIV, and status or history Z codes), '
              'were retained at a visit only when recorded as the '
              'primary diagnosis (sequence number 1) or at their '
              "first appearance in the patient's record; acute codes "
              'were always kept. Prescription drugs were mapped to '
              'RxNorm ingredient concepts through the RxNav API, '
              'queried once in March 2026: NDC codes were resolved '
              'to ingredient concepts, and rows without a resolvable '
              'NDC were matched on the first word of the order name, '
              'so combination products map to their first '
              'ingredient; the lookup resolves 4,137 NDCs and 6,177 '
              'drug-name strings to 1,478 ingredients. Each '
              'ingredient was retained only in the first admission '
              'in which it was prescribed, one token per ingredient '
              'per admission. Ingredients prescribed in more than '
              '25% of admissions, ten near-universal ingredients '
              '(for example sodium chloride, dextrose, '
              'acetaminophen, heparin) and eight administrative or '
              'screening ICD prefixes were removed by fixed lists. '
              'Tokens with training-set frequency below 2 were '
              'folded into [PAD] at encoding. The final 24,005-token '
              'vocabulary comprises 14,960 diagnosis tokens '
              '(ICD-9/10), 7,877 procedure tokens (ICD-9/10), 1,165 '
              'RxNorm-ingredient drug tokens, and 3 special tokens '
              '([VISIT]/[CLS]/[PAD]), shared in a single embedding '
              'space across all streams.')

    heading(doc, "Test-set use in deduplication", level=2)
    para(doc, 'Patterns were mined exclusively from the training '
              'cohort. Test-cohort carrier sets were used for '
              'Jaccard deduplication, and test-cohort attributions '
              'were used to select representatives for the reported '
              'rankings. Neither procedure used test outcomes or '
              'altered the trained model.')

    page_break(doc)

    heading(doc, "Validation settings", level=2)
    para(doc, 'Masking validation covered the patterns displayed in '
              'Tables 3 and 4 and additional high-ranked '
              'risk-supporting and protective patterns selected by '
              "absolute ΣIG across the pipeline's validation runs, "
              '266 unique patterns in total; each pattern was masked '
              'in up to 200 test-cohort carriers, taken in cohort '
              'order. The within-patient reversal analysis used up '
              'to 80 carriers per displayed cross-visit pattern, '
              'again the first in cohort order. The token-level '
              'faithfulness analysis drew its random baseline five '
              'times per patient and reported the mean absolute '
              'change over the draws, the same statistic as the IG '
              'curves; 95% confidence intervals were obtained by '
              'bootstrapping patients with 1,000 resamples (random '
              'seed 42). KernelSHAP treated each patient’s '
              'non-special tokens as features, replaced dropped '
              'tokens with [PAD], and used as background ten random '
              'masks of the patient’s own tokens, each token '
              'retained with probability 0.5; approximately 50 '
              'coalition samples were drawn per patient over the '
              'full analytic test cohort (43,602 patients), and the '
              'per-patient comparison in Figure S10 used 200. '
              'KernelSHAP explained the predicted probability, '
              'whereas IG explained the logit, so the two methods '
              'differ in reference distribution and output scale. '
              'Tokens absent from a record are not features; in the '
              'population-weighted aggregation they were assigned '
              'zero, so Equation 2 of the main text applies to the '
              'reported SHAP summaries as it does to IG. The shap '
              'library’s default l1 regularization, which retains 10 '
              'features per explanation, capped each patient at 10 '
              'non-zero token values: 71% of the 1,327,660 (patient, '
              'token) values in the cohort run were exactly zero, '
              'concentrated in records with more than 10 tokens (36% '
              'at 11 to 20 tokens, 66% at 21 to 40, and 83% or more '
              'above 40). Pattern scores under SHAP summed the '
              'values of a pattern’s unique constituent tokens for '
              'each carrier, whereas IG scored tokens at the matched '
              'visit positions. Pattern rankings under KernelSHAP '
              'used the Jaccard groups of the IG analysis, formed '
              'once from test-cohort carrier sets swept in '
              'descending |ΣIG| order, and their representatives '
              'were not reselected by SHAP magnitude. Fifty-eight '
              'token values from 25 of the 43,602 patients exceeded '
              '10 in absolute value, a numerical failure of the '
              'kernel regression; these values were excluded from '
              'the token summaries, and carriers whose constituent '
              'tokens included such a value were excluded from the '
              'affected pattern sums. The analysis used shap version '
              '0.49.1.')
    para(doc, 'The reversed-order comparison scored 3,447 candidate '
              'cross-visit patterns: those of Table 4 and Figure 4B, '
              'plus every multi-visit pattern of the full pool with '
              'at least 100 test-cohort carriers, drawn from the '
              '40,327 multi-visit patterns of that pool. Of these, '
              '3,429 had valid matches in both orderings and 1,714 '
              'had at least 20 patients matching each ordering; the '
              'pool statistics in the Results and Figure S9 refer to '
              'these 1,714. The 40,327 multi-visit patterns include '
              '718 sequences of a single repeated token; the 39,611 '
              'cross-visit candidates in Table S2 additionally '
              'require two distinct tokens and admit 3 or more '
              'carriers, which adds 2 patterns. Both orderings were '
              "matched on patients' attribution-salient visit "
              'blocks, the token universe used for mining, so the '
              'carrier counts in Table S4 are subsets of the '
              'full-record counts in Table 4.')

    heading(doc, "Attribution direction and observed readmission", level=2)
    para(doc, 'To examine whether model-attributed cross-visit '
              'patterns were associated with observed readmission, '
              'we conducted a separate descriptive analysis. We '
              'restricted the test cohort to patients with at least '
              'two admissions, since cross-visit patterns require multiple visits and such patients are readmitted far more often than the cohort overall (27.8% vs 12.7%). For each reported cross-visit '
              'pattern, we fitted a logistic regression with '
              'readmission as the outcome and pattern presence and '
              'the log-transformed number of admissions as '
              'predictors. We report the adjusted odds ratio for '
              'pattern presence with its 95% Wald confidence '
              'interval (Table S3). No matching was performed. These '
              'associations were not used to evaluate attribution '
              'faithfulness.')

    heading(doc, "Supplementary Figures", level=1)

    heading(doc, "Figure S1. Cohort and analysis flow", level=2)
    figure_placeholder(doc, "Figure S1",
        "CONSORT-style flow of the cohort and analysis. From MIMIC-IV "
        "v3.1 (546,028 hospital admissions), each patient's hospital "
        "encounters of any admission type are assembled into one "
        "chronological token sequence (ICD-9/10 diagnoses and "
        "procedures, RxNorm drugs). In-hospital deaths (11,801 "
        "admissions, 2.2%) are excluded before labeling, since those "
        "patients cannot be readmitted. The 30-day readmission label "
        "excludes elective and surgical same-day admissions from the "
        "positive class (planned care, not unplanned readmission), "
        "yielding the analytic cohort of 218,196 patients with a 12.6% "
        "unplanned 30-day readmission rate. A patient-level 70/10/20 "
        "split gives train (152,737; 12.6% readmit), validation "
        "(21,819), and test (43,640; 12.7% readmit). Pattern mining "
        "runs on training-cohort salient tokens (70,278 distinct "
        "episodes mined, 24,490 of them at whole-cohort support); "
        "per-token IG attribution is computed for all 43,640 test "
        "patients, of which 38 are excluded for an empty salient-token "
        "set, leaving the analytic attribution set of 43,602 patients, "
        "on whose carrier sets patterns are scored and deduplicated "
        "(9,213 cross-visit representatives at j ≥ 0.2) and validated. "
        "The two prevalence values (overall/train 12.6%, test 12.7%) "
        "are the same metric on different cohorts (12.61% and 12.74% "
        "before rounding), not conflicting numbers.",
        image_path=IMAGE_PATHS.get("Figure S1"))

    page_break(doc)
    heading(doc, "Figure S2A. Additional example: psychiatric presentation (true positive)", level=2)
    figure_placeholder(doc, "Figure S2A",
        "Two-panel AToP narrative for a true-positive test patient with "
        "a psychiatric clinical presentation (recurrent psychosis NOS across "
        "consecutive admissions). Same format as main-paper Figure 2: "
        "(A) admission timeline showing the salient tokens each "
        "admission contributed to the Panel B patterns, marked by sign, "
        "plus up to two additional high-|ΣIG| salient tokens per "
        "admission in grey, with the index admission outlined; "
        "(B) allocated "
        "signed \u03a3 IG decomposition under the non-overlapping "
        "position-level allocation rule. Bars labeled V\u2099\u2192V\u2098 "
        "are cross-visit pattern matches; V\u2098: are within-visit. "
        "Figures S2A and S2B were chosen to span clinical groups rather than to validate predictive accuracy. The displayed ΣIG (+1.973) sums the clinical tokens; attribution on the [CLS] and [VISIT] positions (+0.733) and the reference logit (−0.220) complete the identity with the patient's logit (+2.533) up to a residual of +0.046.",
        image_path=IMAGE_PATHS.get("Figure S2A"))

    page_break(doc)
    heading(doc, "Figure S2B. Additional example: obstetric true negative",
             level=2)
    figure_placeholder(doc, "Figure S2B",
        "Two-panel AToP narrative for a true-negative test patient with "
        "an obstetric clinical presentation (uncomplicated delivery with current "
        "obstetric laceration repair). Same format as Figures 2 and S2A. "
        "The salient subsequence comprises obstetric procedure tokens; "
        "AToP surfaces the corresponding protective obstetric trajectory "
        "in the allocated \u03a3 IG panel and the model assigns a low predicted readmission probability. The displayed ΣIG (−2.145) sums the clinical tokens; attribution on the [CLS] and [VISIT] positions (+1.062) and the reference logit (−0.198) complete the identity with the patient's logit (−1.300) up to a residual of −0.019.",
        image_path=IMAGE_PATHS.get("Figure S2B"))

    page_break(doc)
    heading(doc, "Figure S2C. Disagreement example: oncology false "
                  "positive", level=2)
    figure_placeholder(doc, "Figure S2C",
        "Two-panel AToP narrative for a false-positive test patient: the "
        "model predicted readmission with high confidence (ŷ = 0.93); the "
        "patient did not readmit within 30 days. The patient has a "
        "metastatic-oncology trajectory across four pre-index visits "
        "(bone metastases at V1 and V3; cancer pain and cancer anemia at "
        "V4; chemotherapy-supportive medications). The prediction was "
        "driven primarily by the cross-visit Bone metastases → Bone "
        "metastases pattern (+0.240 allocated ΣIG) and a larger combined contribution from 13 other salient tokens (+0.969). Same two-panel "
        "format as Figure 2, S2A, and S2B. The explanation shows which patterns drove the high-confidence prediction. The displayed ΣIG (+1.061) sums the clinical tokens; attribution on the [CLS] and [VISIT] positions (+1.280) and the reference logit (+0.145) complete the identity with the patient's logit (+2.534) up to a residual of +0.048.",
        image_path=IMAGE_PATHS.get("Figure S2C"))

    page_break(doc)
    heading(doc, "Figure S3. Sensitivity to Jaccard threshold, "
                  "cross-visit pool", level=2)
    figure_placeholder(doc, "Figure S3",
        "Cross-visit pattern pool (≥2 visits, ≥2 distinct tokens) at four Jaccard deduplication thresholds. Panel A: no dedup (39,611 representatives). Panel B: j ≥ 0.6 (33,605). Panel C: j ≥ 0.2 (9,213, used in the main analyses). Panel D: j ≥ 0.1 (2,075). Thresholds j ≥ 0.8 "
        "(38,074) and j ≥ 0.4 (24,606) are omitted; their top-30 sets are "
        "near-identical to j ≥ 0.6. Each panel shows the top-30 "
        "cross-visit patterns by |ΣIG| within the deduplicated set. "
        "Unlike Figure S4, the panels differ substantially from one "
        "another (A and C share only 8 of 30 patterns), which is itself "
        "the result: on the cross-visit pool the threshold materially "
        "changes what reaches the top. Loose thresholds leave the top-30 "
        "dominated by near-duplicate MDD/SI and obstetric variants; all "
        "six clinical groups surface independently by j ≥ 0.2. At "
                "j ≥ 0.1 that coverage survives in a much smaller pool "
        "(2,075), with oncology sequences filling more of the top-30.",
        image_path=IMAGE_PATHS.get("Figure S3"))

    page_break(doc)
    heading(doc, "Figure S4. Sensitivity to Jaccard threshold, all "
                  "patterns combined", level=2)
    figure_placeholder(doc, "Figure S4",
        "Jaccard-threshold sweep on the all-patterns combined pool "
        "(single-token, within-visit co-occurrence, and cross-visit "
        "patterns together). Representative counts: 70,275 (no dedup), "
        "61,059 (j ≥ 0.8), 53,623 (j ≥ 0.6), 40,869 (j ≥ 0.4), "
        "17,891 (j ≥ 0.2, used in the main analyses), and 5,501 (j ≥ 0.1). Four are displayed (A: no dedup, B: 0.6, C: 0.2, D: 0.1); j ≥ 0.8 and "
        "j ≥ 0.4 are omitted, their top-30 sets being identical to "
        "j ≥ 0.6. Each column shows the top-30 patterns by |ΣIG| "
        "within that deduplicated representative set, drawn against a "
        "single shared label column so rows align across thresholds; a "
        "grey diamond marks a pattern absent from that threshold's "
        "top-30. Across every threshold, "
        "the top-30 is dominated by within-visit co-occurrence patterns; no cross-visit pattern enters any top-30, and single-token patterns appear at most twice per threshold. Without the cross-visit restriction, the unrestricted ranking therefore obscures the cross-visit patterns that motivate the framework.",
        image_path=IMAGE_PATHS.get("Figure S4"))

    page_break(doc)
    heading(doc, "Figure S5. Pattern-level aggregation comparison", level=2)
    figure_placeholder(doc, "Figure S5",
        "Pattern-level aggregation comparison on the all-patterns pool "
        "(single-token, within-visit co-occurrence and cross-visit "
        "patterns combined; Jaccard j = 0.2, 17,891 representatives). "
        "(A) Top 30 by carrier-averaged ΣIG. (B) Top 30 by "
        "population-weighted mean IG. Only 6 of 30 patterns are "
        "shared. The population-weighted top 30 is led by single-token "
        "features (10 of its 30 entries; the remainder are "
        "high-prevalence within-visit sets, mostly obstetric bundles), "
        "whereas the carrier-conditional top 30 consists almost "
        "entirely of within-visit co-occurrence sets; its 24 patterns absent from the population-weighted top 30 sit at population ranks 36 to 3,911. No cross-visit sequence enters either top "
        "30. Bold, dark-edged bars mark patterns absent from the other "
        "panel's top 30, with their rank in the other ranking shown "
        "beside the bar.",
        image_path=IMAGE_PATHS.get("Figure S5"))

    page_break(doc)
    heading(doc, "Figure S6. Pattern prevalence and relative risk vs "
                  "attribution magnitude", level=2)
    figure_placeholder(doc, "Figure S6",
        "Pattern prevalence and relative risk versus attribution "
        "magnitude for n = 5,047 cross-visit patterns: the cross-visit "
        "members of the 24,490-pattern per-patient explanation "
        "dictionary (Table S2) that received test-cohort IG scores. "
        "This pool is not Jaccard-deduplicated and is the set for "
        "which training-cohort readmission counts exist, so it differs "
        "from the 9,213 cross-visit representatives of Figure 4B and "
        "Table 4 (718 patterns shared). Dots are red for risk (ΣIG > "
        "0) and blue for protective patterns; dot area is proportional "
        "to log(n test carriers). (A) Carrier-averaged |ΣIG| against "
        "test-cohort prevalence (log scale); the five risk and five "
        "protective patterns with the largest |ΣIG| at low prevalence "
        "are ringed and listed below the panels. (B) |ΣIG| against "
        "log10 relative risk computed in the training cohort (baseline "
        "12.65%), Spearman ρ = −0.13: patterns with similar relative "
        "risk span a wide range of attribution magnitude. Test-cohort "
        "adjusted odds ratios for the reported patterns are given in "
        "Table S3.",
        image_path=IMAGE_PATHS.get("Figure S6"))

    page_break(doc)
    heading(doc, "Figure S7. Per-pattern masking effects", level=2)
    figure_placeholder(doc, "Figure S7",
        "Masking effect \u0394\u0177 = \u0177_original \u2212 \u0177_masked for each "
        "of the 266 perturbed patterns, averaged over the pattern's "
        "test-cohort carriers. Bars are colored by the pattern's attribution "
        "direction, taken from the test-cohort carrier-averaged \u03a3IG "
        "under the scoring rule of the Supplementary Methods: red, "
        "risk-supporting (\u03a3IG > 0); blue, protective (\u03a3IG < 0). Masking "
        "moved predictions in the direction of the pattern's attribution for "
        "all 266 patterns (168 risk-supporting, 98 protective). Masking creates out-of-distribution inputs, so it is a "
        "proxy test; the direct token-level faithfulness test is the "
        "comprehensiveness and sufficiency analysis in Figure S8.",
        image_path=IMAGE_PATHS.get("Figure S7"))

    page_break(doc)
    heading(doc, "Figure S8. Quantitative faithfulness of AToP-ranked "
                  "tokens", level=2)
    figure_placeholder(doc, "Figure S8",
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
        image_path=IMAGE_PATHS.get("Figure S8"))

    page_break(doc)
    heading(doc, "Figure S9. Attribution under forward and reversed "
                  "temporal order across the cross-visit pool", level=2)
    figure_placeholder(doc, "Figure S9",
                "Attribution under forward and reversed temporal order. Small points represent 1,714 cross-visit patterns with at least 20 test-cohort patients matching each ordering on their attribution-salient visit blocks. Outlined points "
        "additionally show the 27 evaluable patterns reported in Table "
        "4, regardless of their forward- and reversed-order carrier "
        "counts; the remaining 3 read the same in either direction and "
        "were excluded. The horizontal and vertical axes show "
        "carrier-averaged signed ΣIG among patients matching the "
        "forward and reversed orders, respectively; the two carrier "
        "groups may overlap and need not be identical. Red marks "
        "risk-increasing patterns (forward ΣIG > 0) and blue "
        "protective patterns. The dashed line indicates equal "
        "attribution under both orderings. Across the 1,714-pattern "
        "pool, 54% of patterns showed reduced attribution magnitude "
        "under reversal, and 95% preserved attribution direction.",
        alt="Scatter plot of reversed-order against forward-order "
            "attribution for 1,714 cross-visit patterns, clustered along "
            "the diagonal, with the Table 4 patterns highlighted as "
            "larger outlined points mostly below the diagonal.",
        image_path=IMAGE_PATHS.get("Figure S9"))

    page_break(doc)
    heading(doc, "Figure S10. Per-patient SHAP vs IG agreement", level=2)
    figure_placeholder(doc, "Figure S10",
        "Per-patient agreement between Integrated Gradients (IG) and "
        "KernelSHAP attribution methods. (A) Example patient (true "
        "positive, recurrent psychosis): per-token signed IG vs signed "
        "KernelSHAP for the patient's 18 highest-|IG| tokens; the 6 most extreme "
        "are labeled, and the axes are scaled independently because IG is on "
        "the logit scale and KernelSHAP on the probability scale. Pearson r = "
        "0.52, Spearman ρ = 0.43. (B) Aggregate "
        "scatter pooling top 10 tokens (by |IG|) from 100 stratified-"
        "random test patients, z-scored on both axes so per-method scale "
        "differences do not obscure rank/sign agreement. n = 935 (patient, "
        "token) pairs. Pearson r = 0.67, Spearman ρ = 0.45. Per-patient "
        "Jaccard@5 = 0.57 (mean), a top-five token-set overlap. Red = risk-supporting (IG > 0), blue = protective; "
        "dashed line = y = x. KernelSHAP here used 200 coalition samples "
        "per patient on the top-K-by-IG token subset, using a larger sampling budget; this differs from the cohort-level comparison "
        "in Figures S11 and S12, which used ~50 coalitions per patient to keep the "
        "full 43,602-patient run tractable. Cohort-level cross-method "
        "overlap statistics are reported in Figures S11 and S12.",
        image_path=IMAGE_PATHS.get("Figure S10"))

    page_break(doc)
    heading(doc, "Figure S11. Token-level attribution with KernelSHAP",
             level=2)
    figure_placeholder(doc, "Figure S11",
        "KernelSHAP counterpart of Figure 3 on the full test cohort "
        "(N = 43,602 patients with valid sequences): population-weighted "
        "(A) and carrier-conditional (B) mean SHAP per token, top 30 by "
        "absolute value, with the same aggregation conventions as the IG "
        "ranking; tokens with fewer than 10 sampled carriers were excluded "
        "before ranking. The carrier-conditional top-30 shares 25 of 30 tokens with the IG top-30.",
        image_path=IMAGE_PATHS.get("Figure S11"))

    # Figure S12 is drawn at landscape print size: figure_landscape opens a
    # landscape section (heading, image, caption on one page) and returns to
    # portrait on a new page, so Figure S13 needs no page break.
    figure_landscape(doc, "Figure S12",
        "Figure S12. Pattern-level attribution with KernelSHAP",
        "KernelSHAP counterpart of Figure 4 on the full test cohort: "
        "carrier-averaged \u03a3SHAP for all patterns (A) and for cross-visit "
        "patterns (B), top 30 by absolute value after Jaccard \u2265 0.2 "
        "deduplication; patterns with fewer than 10 sampled carriers were "
        "excluded before ranking. Agreement with the IG rankings is 21 of "
                "30 for all patterns and 7 of 30 for cross-visit patterns. The "
        "methods differed in their reference distributions, output "
        "scales, attribution sparsity, and pattern-scoring procedures "
        "(Supplementary Methods), so the lower cross-visit agreement "
        "cannot be attributed to the attribution method alone.",
        image_path=IMAGE_PATHS.get("Figure S12"))

    heading(doc, "Figure S13. Sensitivity to the IG saliency-mass threshold",
             level=2)
    figure_placeholder(doc, "Figure S13",
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
                "ranking. Top-30 risk-pattern membership was relatively stable "
        "across saliency-mass thresholds, whereas protective-pattern "
        "membership was more sensitive.",
        image_path=IMAGE_PATHS.get("Figure S13"))

    # ── Supplement tables (page-break-before on the heading avoids an
    #    empty break paragraph that can leave a blank page) ───────────
    heading(doc, "Supplementary Tables", level=1)
    doc.paragraphs[-1].paragraph_format.page_break_before = True

    # ── Table S1: parameter / hyperparameter table ─────────────────
    df_s4 = _read_csv("TableS4_parameters.csv")
    table_title(doc, "Table S1", "AToP parameter and evaluation settings.")
    s4_rows = [[_fmt_str(r[c]) for c in df_s4.columns] for _, r in df_s4.iterrows()]
    make_table(doc,
                header_row=list(df_s4.columns),
                data_rows=s4_rows,
                col_widths_in=[1.9, 1.3, 1.1, 2.6],
                font_size_pt=8)
    table_legend(doc,
        abbreviations=(
            "IG = Integrated Gradients; τ = saliency-mass threshold; j "
            "= Jaccard similarity threshold; CI = confidence interval."
        ),
        note=(
            "Prespecified settings were fixed before the corresponding "
            "analyses. Design choice denotes a methodological "
            "convention rather than a numerical setting. Selected by "
            "inspection denotes a value chosen by examining "
            "training-cohort output without outcome labels or test "
            "data; the maximum drug-prevalence threshold was the only "
            "such parameter. No parameter was selected by optimizing "
            "test-set outcomes."
        ),
    )

    page_break(doc)
    add_table_s2(doc)

    page_break(doc)
    # ── Table S3: adjusted odds ratios (number of admissions) ──
    df_s6 = _read_csv("TableS6_adjusted_or.csv")
    table_title(doc, "Table S3",
                 "Association of cross-visit patterns with readmission, "
                 "adjusting for the number of admissions.")
    s6_rows = []
    _last = None
    for _, r in df_s6.iterrows():
        ph = _fmt_str(r["phenotype"])
        s6_rows.append([
            ph if ph != _last else "",
            _fmt_str(r["label"]),
            f"{r['ig_value']:+.2f}",
            _fmt_int(r["n_carriers_multivisit"]),
            f"{r['carrier_readmit_pct']:.1f}%" if pd.notna(r["carrier_readmit_pct"]) else "n/a",
            f"{r['matched_baseline_pct']:.1f}%" if pd.notna(r["matched_baseline_pct"]) else "n/a",
            _fmt_str(r["adj_OR"]),
            _fmt_str(r["ci95"]),
        ])
        _last = ph
    make_table(doc,
                header_row=["Clinical group", "Pattern (representative)", "ΣIG",
                             "n", "Carrier readmit %",
                             "Non-carrier readmit %", "Adj. OR", "95% CI"],
                data_rows=s6_rows,
                col_widths_in=[0.95, 2.55, 0.45, 0.35, 0.62, 0.68, 0.5, 0.78],
                font_size_pt=7)
    table_legend(doc,
        abbreviations=(
            "ΣIG = carrier-averaged signed sum of constituent-token IG "
            "values (logit space; positive = risk-increasing, negative = "
            "protective); n = pattern carriers within "
            "the multi-visit population; Carrier readmit % = observed "
            "30-day readmission rate among those carriers; Non-carrier "
            "readmit % = readmission rate among non-carriers in the same "
            "population; Adj. OR = adjusted odds ratio for carriage; "
            "CI = confidence interval."
        ),
        note=(
            "Cross-visit patterns require at least two admissions, and "
            "the analysis was restricted to patients meeting this "
            "criterion. Adjusted odds ratios were estimated separately "
            "for each pattern using logistic regression with readmission "
            "as the outcome and pattern presence and the log-transformed "
            "number of admissions as predictors. Confidence intervals are "
            "95% Wald intervals. The odds ratios describe associations "
            "rather than causal effects, and no multiplicity correction "
            "was applied; 20 of the 30 intervals include 1."
        ),
    )

    page_break(doc)
    # ── Table S4: Table-4 cross-walk (11 cross-visit patterns) ─────
    df_s3 = _read_csv("TableS3_reversed_order_table4.csv")
    table_title(doc, "Table S4",
                 "Forward- and reversed-order attribution for the 30 "
                 "cross-visit patterns in Table 4.")
    s3_rows = []
    _last_ph = None
    for _, r in df_s3.iterrows():
        _ph = _fmt_str(r["phenotype"])
        s3_rows.append([
            _ph if _ph != _last_ph else "",
            _fmt_int(r["n_blocks"]),
            _fmt_str(r["readable_label"]),
            _fmt_int(r["n_forward"]),
            _fmt_int(r["n_reversed"]),
            _fmt_int(r["n_both"]),
            f"{r['ig_mean_forward']:+.3f}" if pd.notna(r["ig_mean_forward"]) else "n/a",
            f"{r['ig_mean_reversed']:+.3f}" if pd.notna(r["ig_mean_reversed"]) else "n/a",
            f"{r['attenuation_pct']:.1f}%" if pd.notna(r["attenuation_pct"]) else "n/a",
        ])
        _last_ph = _ph
    make_table(doc,
                header_row=["Clinical group", "#blk", "Pattern (representative)",
                             "n fwd", "n rev", "n both", "ΣIG fwd", "ΣIG rev",
                             "Attn"],
                data_rows=s3_rows,
                col_widths_in=[1.05, 0.40, 1.95, 0.5, 0.5, 0.5, 0.65, 0.65, 0.55],
                font_size_pt=8)
    table_legend(doc,
        abbreviations=(
            "#blk = number of visit blocks; n fwd and n rev = numbers of "
            "patients matching the forward and reversed orders, "
            "respectively; n both = patients matching both orders; "
            "ΣIG fwd and ΣIG rev = carrier-averaged signed attribution "
            "for each ordering; Attn = 100 × (1 − |ΣIG rev| / |ΣIG fwd|). "
            "Patterns sorted by clinical group in the main-paper Table 4 "
            "ordering, and within each group by descending |ΣIG fwd|."
        ),
        note=(
            "Aggregate result for this displayed set: of the 30 patterns, "
            "3 are symmetric (palindromic block sequences for which "
            "reversal returns the same pattern, attenuation 0 by "
            "construction) and are excluded as non-informative. Among the "
            "remaining 27 asymmetric patterns, 26 (96%) "
            "show |ΣIG forward| > |ΣIG reversed|; all 30 preserve sign; "
                        "across the 27, mean attenuation is 30% and median 27% "
            "(one-sided Wilcoxon signed-rank test of |ΣIG forward| > "
            "|ΣIG reversed| over the 27 patterns, p < 0.001). One "
            "pattern shows a small "
            "increase under reversal. n forward, n reversed, n both and the ΣIG values in this table are computed on each patient's attribution-salient visit blocks (the tokens covering 80% of absolute IG mass, at most 50 per patient, grouped by visit; the token universe used for mining), whereas n and ΣIG in main-paper Table 4 are computed on the complete record. Every salient-block match is also a full-record carrier, so the n forward patients are a subset of the Table 4 carriers (for example 148 of 201 for {MDD single ep. NOS; SI} → SI), and ΣIG forward differs from the Table 4 value mainly because it averages over that subset. The forward and reversed groups are not disjoint: "
            "patients counted under n both contribute to both means. "
            "Several patterns "
            "have small reverse-order support (n reversed as low as 2), so "
            "their individual attenuation is descriptive. Broader "
            "pattern-level reversal results are presented in Figure S9."
        ),
    )

    # ── Table S5: within-patient reversal of the Table 4 patterns ──────
    page_break(doc)
    df_s6w = _read_csv("TableS6_within_patient_reversal.csv")
    table_title(doc, "Table S5",
                "Within-patient temporal-order reversal for the 30 "
                "cross-visit patterns in Table 4.")
    make_table(doc,
               header_row=["Clinical group", "Pattern", "n", "\u03a3IG obs.",
                           "\u03a3IG rev.", "Atten. %", "Fell %",
                           "Sign %", "p"],
               data_rows=[[(_fmt_str(r["Phenotype"]) if _fmt_str(r["Phenotype"]) != _prev else ""), _fmt_str(r["Pattern"]),
                           _fmt_int(r["n carriers"]),
                           _fmt_num(r["Mean \u03a3IG, observed order"], "+.3f"),
                           _fmt_num(r["Mean \u03a3IG, visits reversed"], "+.3f"),
                           _fmt_num(r["Attenuation (%)"], ".1f"),
                           _fmt_num(r["Carriers with reduced |\u03a3IG| (%)"], ".1f"),
                           _fmt_num(r["Sign preserved (%)"], ".0f"),
                           _fmt_str(r["Wilcoxon p"])]
                          for _prev, (_, r) in zip([None] + [_fmt_str(x) for x in df_s6w["Phenotype"].tolist()[:-1]], df_s6w.iterrows())],
               col_widths_in=[0.8, 2.3, 0.35, 0.55, 0.55, 0.5, 0.5, 0.45, 0.5],
               font_size_pt=8)
    table_legend(doc,
        abbreviations=("Clinical group = as in main-paper Table 4; n = carriers tested (first 80 test-cohort carriers "
                       "in cohort order); \u03a3IG observed / reversed = carrier-"
                       "averaged pattern score (obs.) on the patient's record as "
                       "observed and after reversing the order of the "
                       "pattern's matched visits within that record; Atten. = "
                       "1 \u2212 |\u03a3IG rev.| / |\u03a3IG obs.|; Fell = "
                       "share of carriers whose |\u03a3IG| fell; Sign = share "
                       "whose score kept its sign; p = one-sided Wilcoxon "
                       "signed-rank test of |\u03a3IG observed| > |\u03a3IG reversed| "
                       "over carriers."),
        note=('Unlike Table S4, which compares the carriers matching '
              'the forward order with those matching the reversed order '
              '(groups that can share patients), this analysis '
              "reorders the matched visits inside each carrier's own "
              'record and recomputes IG; all other visits and the '
              'contents of every visit are unchanged. Because whole '
              'visits are moved, the other events in those visits move '
              'with the pattern tokens, so the analysis tests sensitivity'
              ' to visit order rather than isolating the order of the '
              "pattern's own events. For the three palindromic patterns "
              'the pattern reads the same in either direction and any '
              'change reflects the other visit contents, so they are '
              'excluded from the summary counts. Across 925 '
              'patient-pattern pairs from 722 distinct patients (166 of '
              'whom carry more than one pattern), attribution magnitude '
              'decreased in 60% of pairs, and 97% preserved attribution '
              'direction. At the patient level, 59% showed a reduction in '
              "their mean |ΣIG| over their patterns; the patient-level "
              'one-sided Wilcoxon test on those means was significant '
              '(p < 0.001, n = 722). Per-pattern p values are nominal; 4 '
              'of 27 remain below 0.05 after Holm correction.'))

    # ── Table S6: Transformer architecture and training settings ─────
    page_break(doc)
    df_s7 = _read_csv("TableS6_hyperparameters.csv")
    table_title(doc, "Table S6", "Transformer architecture and training settings.")
    _rows6, _last6 = [], None
    for _, r in df_s7.iterrows():
        _sec = _fmt_str(r["Section"]) if r["Section"] != _last6 else ""
        _last6 = r["Section"]
        _rows6.append([_sec, _fmt_str(r["Parameter"]), _fmt_str(r["Value"])])
    make_table(doc, header_row=["Section", "Parameter", "Value"], data_rows=_rows6,
               col_widths_in=[0.9, 1.9, 3.7], font_size_pt=8, bold_first_col=True)
    table_legend(doc,
        abbreviations=("AUROC = area under the receiver operating characteristic curve; "
                       "PR-AUC = area under the precision-recall curve; OOV = out of vocabulary."),
        note=("Values not stored in the run configuration (early-stopping patience, "
              "feed-forward activation, optimizer constants) are the training script's "
              "defaults. A single training run (seed 42) was performed; hardware and "
              "library versions were not recorded."))

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
