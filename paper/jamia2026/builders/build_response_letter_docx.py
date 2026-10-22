"""Build the point-by-point response to the pre-submission reviews of the
AToP manuscript (manuscript/jamia_submission/revision/AToP_response_to_reviews.docx).

The 41 numbered points follow the blind-review crosscheck of 2026-10-17 and
the tracker in manuscript/jamia_submission/revision/AToP_review_tracker.md.
Section A covers the corrections made in the current build (commit 9c8f7ef);
Section B gives the responses to design choices we retain, each with the
small manuscript change proposed alongside it; Section C lists the points
that await the planned analyses and is a placeholder until those are done.

Usage:
    python paper/jamia2026/builders/build_response_letter_docx.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _docx_helpers import heading, new_document, para, set_body_line_spacing  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "manuscript",
                        "jamia_submission", "revision",
                        "AToP_response_to_reviews.docx")


def item(doc, number: str, title: str, point: str, response: str, changes: str) -> None:
    para(doc, f"**{number}. {title}**")
    para(doc, point, italic=True)
    para(doc, f"**Response.** {response}")
    para(doc, f"**Changes.** {changes}")
    para(doc, "")


def build() -> None:
    set_body_line_spacing(1.5)
    doc = new_document()
    heading(doc, "Response to reviews: AToP (Attribution to Temporal Patterns)", level=1)
    para(doc, "Draft prepared from four independent pre-submission reviews, "
              "consolidated into 41 numbered points. Numbering follows the "
              "consolidated list. Section A lists corrections already made in "
              "the current manuscript; Section B gives our response on points "
              "where we retain the design and names the clarification added "
              "for each; a closing note states which requests are left to "
              "future work. Numbers quoted here were recomputed from "
              "the saved analysis outputs and the raw MIMIC-IV tables.")
    para(doc, "We thank the reviewers for an unusually thorough reading. The "
              "corrections below improve the accuracy of several descriptions "
              "and the reproducibility of the pipeline; none changes a reported "
              "result.")
    para(doc, "")

    # ── Section A ────────────────────────────────────────────────────
    heading(doc, "A. Corrections made in the revised manuscript", level=2)

    item(doc, "3", "Table 1 admission count",
         "The row 'Number of admissions (incl. index)' is ambiguous next to 12,316 readmitted patients with one admission.",
         "The row counts admissions through the index admission, so a value of 1 means no admission before the index, not a single lifetime admission; the 12,316 readmitted patients in that stratum were readmitted after it.",
         "Table 1 row relabelled 'Number of admissions through the index (incl. index)'.")

    item(doc, "5 and 6", "Post-discharge deaths, same-day returns and transfers",
         "How deaths after discharge, 0-day returns and transfers were handled is not stated.",
         "A readmission was any later encounter in the admissions table beginning within 0 to 30 days of the index discharge; encounters were not merged and no minimum interval was imposed. Only in-hospital deaths were removed before labeling; post-discharge mortality was not used in labeling, so a patient who died within 30 days without a counted readmission is a negative. We reproduced the label rule from the raw tables and matched all 43,640 test labels.",
         "Methods, 'Data and prediction task': the window, the no-merging rule, the post-discharge mortality statement, and 'direct' added to the eligible admission types.")

    item(doc, "8", "Exemplar attribution sums do not reconcile with the reference logit",
         "Figure 2 shows a predicted probability of 0.93 and a total attribution of +0.971, implying a reference logit far above the reported mean.",
         "The reviewer's arithmetic is right and the cause is a reporting gap, not a computational one. The displayed total is the sum over clinical tokens; attribution on the [CLS] and [VISIT] positions is not shown. For the Figure 2 patient the logit is +2.641, the all-[PAD] reference logit +0.302, the clinical-token sum +0.971, the structural-token sum +1.310, and the residual +0.058. Across 300 test patients the structural positions carry a median 46% of total absolute attribution, with no consistent sign. We now state this where the completeness property is introduced, and the figure legends carry the full identity.",
         "Methods, 'AToP explanation framework' (completeness sentence); Figure 2 legend; Figures S2A to S2C legends; Supplementary Methods, 'Integrated Gradients reference baseline'.")

    item(doc, "12", "Outcome-stratified mining",
         "Mining separately within each outcome class should be stated as an outcome-dependent step.",
         "It is stated in the main Methods ('Mining was conducted separately in each outcome class, using a minimum support fraction of 0.001') and the class-specific supports (20 and 134 carriers) are in Table S2.",
         "None.")

    item(doc, "13 and 16", "Carrier matching on full records versus salient subsequences; Table 4 versus Table S4 counts",
         "Mining used salient subsequences but carriers are matched on full records; the same pattern has different carrier counts in Table 4 and Table S4.",
         "Both counts are correct and come from two token universes. Table 4 and Figure 4 match patterns on the complete record, so n is every test patient whose record contains the pattern. Table S4 and Figure S9 match on each patient's attribution-salient visit blocks, the universe used for mining; every salient-block match is also a full-record carrier, so the Table S4 patients are a subset (148 of 201 for {MDD single ep. NOS; SI} → SI), and the attribution difference is mainly a subset effect (+0.648 under the full-record rule on those 148 patients, +0.552 over all 201). Among the 1,230 carrier-pattern pairs of the 30 Table 4 patterns, every pattern token was salient somewhere in the record in 836 (68%); restricting carriers to those patients preserved the sign of all 30 carrier-averaged scores (Spearman ρ of magnitudes 0.97).",
         "Methods, 'AToP explanation framework' (full-record matching named); Methods, 'Validation analyses' (reversal matched on salient visit blocks); Table S4 note rewritten; Figure S9 legend; Supplementary Methods, pattern scoring and validation settings; Table S2 note.")

    item(doc, "14", "Prominence of the selected reversal result",
         "The 26 of 27 reversal result for the reported patterns is given more weight than the 54% result in the broader pool.",
         "Both results are reported side by side in Results and Table S5, including the Holm-corrected count. The Discussion now states the two together.",
         "Discussion, first paragraph: 'for most reported cross-visit patterns but only about half of the broader pattern pool'.")

    item(doc, "17 and 30", "Positional encoding, truncation and training details",
         "The positional scheme, truncation rule and training hyperparameters are not reported, and the prediction head is misdescribed.",
         "We agree, and one description was wrong: the head is a two-layer feed-forward block (128 to 64 to 1 units), not a single linear layer. The model adds learned absolute position embeddings to token embeddings with no visit-level or time embedding; records are the index admission plus up to 19 prior admissions, left-truncated to 512 tokens (7 of 218,196 records affected); the loss is class-weighted binary cross-entropy. A full table of architecture and training settings, read from the saved checkpoint and run configuration, is now in the supplement.",
         "Methods, 'Single-stream Transformer'; new Table S6.")

    item(doc, "18", "Temporal order versus elapsed time",
         "'Temporal' may be read as incorporating time intervals.",
         "Patterns encode the order of admissions only; gaps between admissions are not modelled.",
         "Methods, 'Data and prediction task': one sentence.")

    item(doc, "20", "Masking sample size",
         "Masking used up to 200 carriers per pattern, whereas the tables show the full carrier count.",
         "Correct. Masking effects were computed over the first 200 test-cohort carriers in cohort order; n in Tables 3 and 4 is the full carrier count. Seven of the twenty Table 3 patterns exceed 200 carriers.",
         "Methods, 'Validation analyses'; Table 3 note (inherited by Table 4); Figure 5 legend.")

    item(doc, "22", "KernelSHAP as an independent check",
         "The KernelSHAP comparison differs from the IG pipeline in reference distribution, output scale, sparsity, representative selection and pattern scoring, so it is not an independent replication.",
         "We agree and do not present it as one. The implementation differences are listed in the Supplementary Methods, the limitation states that the 7 of 30 cross-visit agreement does not isolate the attribution method, and the comparison is now introduced as exploratory.",
         "Results, 'Cross-method comparison': 'as an exploratory cross-implementation comparison'.")

    item(doc, "23", "Figure S10A",
         "All KernelSHAP values appear on the zero line although the caption reports r = 0.52.",
         "The values were drawn on a shared y = x axis although IG is on the logit scale and KernelSHAP on the probability scale, which flattened the SHAP values. The panel is replotted from the saved values with independent axes; the statistics are unchanged (r = 0.524, ρ = 0.434 for the example patient; r = 0.674, ρ = 0.453 and Jaccard@5 = 0.57 for the aggregate).",
         "Figure S10 regenerated; caption states the axis scaling and the patient's 18 tokens.")

    item(doc, "26", "Greedy Jaccard deduplication under-specified",
         "The processing order, comparison rule and representative choice are not stated.",
         "Patterns are processed in descending order of absolute carrier-averaged ΣIG; each unassigned pattern seeds a group and absorbs every lower-ranked unassigned pattern whose Jaccard similarity to the seed is at least 0.20; the seed is the representative and is therefore the largest-magnitude member. We re-ran the procedure and reproduced the 9,213 cross-visit representatives exactly; no two patterns tie on magnitude in either pool, so no tie rule is needed.",
         "Methods, 'AToP explanation framework' (deduplication sentences); Supplementary Methods, validation settings.")

    item(doc, "27", "Discovery-only patterns not marked",
         "Patterns with fewer than 20 carriers are said to be flagged but are not.",
         "They now are.",
         "Tables 3 and 4: dagger on n below 20, with a note line.")

    item(doc, "28", "Saliency-threshold sensitivity re-matches rather than re-mines",
         "Figure S13 varies τ with the mined dictionary fixed.",
         "Correct; the caption already says the dictionary is re-matched at each τ, and the limitation now says so too.",
         "Limitation (iii): 'with the mined dictionary fixed'.")

    item(doc, "31", "Baseline comparison wording",
         "The limitation refers to AToP where the Transformer is meant.",
         "Corrected.",
         "Limitation (iv).")

    item(doc, "32", "Preprocessing and mapping detail",
         "The chronic-code rule, primary-diagnosis source, prescription rule, GEM criteria and RxNorm mapping need operational definitions.",
         "The supplement now gives them: chronic status by a fixed ICD-prefix list released with the code, kept at sequence number 1 or first appearance; each RxNorm ingredient kept at its first recorded prescription; GEM conversion restricted to ICD-9 codes with a single ICD-10 target (80.4% of ICD-9 diagnosis rows and 14.2% of ICD-9 procedure rows converted; 1,863 diagnosis and 1,807 procedure ICD-9 tokens retained); RxNav queried once in March 2026 with first-word matching for rows without a resolvable NDC (4,137 NDCs and 6,177 drug-name strings to 1,478 ingredients); the 25% prevalence cap and the fixed exclusion lists.",
         "Methods, 'Data and prediction task' ('first recorded prescription', 'prescribed'); Supplementary Methods, 'Preprocessing and vocabulary composition' rewritten.")

    item(doc, "33", "Bootstrap p values below the attainable resolution",
         "p < 0.001 cannot arise from 1,000 resamples.",
         "Correct. The paired bootstrap now uses the add-one convention; with no resample reversing the sign of either difference, the two-sided value is 2/1,001 and is reported as p = 0.002. Point estimates and intervals are unchanged. The three remaining 'p < 0.001' values are Wilcoxon signed-rank tests.",
         "Results, 'Predictive performance'; Table 2 and its note; bootstrap script.")

    item(doc, "34", "Inconsistencies in Figures S1, S5 and S6",
         "The S1 caption misstates the prevalence rounding and places deduplication in the mining step; S5 and S6 captions and drawings disagree.",
         "All confirmed. The overall rate is 12.61%; representatives are selected on test-cohort carrier sets, now shown at the scoring step; S5's subtitle is computed from the plotted data (6 of 30 shared; 10 single-token entries in the population-weighted top 30; 24 carrier-conditional patterns absent from it at population ranks 36 to 3,911); S6's annotation box hid 7 of its 10 ringed patterns and the high-magnitude tail, so the list now sits below the panels; its caption defines the 5,047-pattern pool (dictionary cross-visit patterns with test scores, not deduplicated, 718 shared with the 9,213 representatives) and states that its relative risks are training-cohort values. The named pattern is drawn at +0.552, matching Table 4.",
         "Figures S1, S5, S6 regenerated from existing outputs; captions rewritten; Table 3 legend 'mean change'.")

    item(doc, "36", "Clinical-group assignment by keyword precedence",
         "{Alcohol abuse; Resp measurements} is labelled a chest-pain workup because a test keyword matches first.",
         "The precedence is corrected so that substance-use keywords are tested before workup keywords. One displayed row moves (Table 3, to Substance use) and the chest-pain group's fifth row is the next masked pattern; one non-displayed row in the Table 4 source moves. Groups remain presentational and do not affect mining, attribution or ranking, as the table note states.",
         "Tables 3 and 4; grouping script.")

    item(doc, "40", "Ethics statement",
         "No ethics or data-governance statement is given.",
         "Added.",
         "New 'Ethics Statement' section before Data Availability.")

    # ── Section B ────────────────────────────────────────────────────
    heading(doc, "B. Design choices retained, with the clarification added for each", level=2)

    item(doc, "1", "Readmissions ending in death and deaths after discharge",
         "A readmission that ends in death is still a readmission; labelling such cases as non-events, and ignoring deaths after discharge, may bias the outcome. Relabelling or a competing-risk formulation is requested.",
         "The reviewer is correct about the rule as implemented. Admissions ending in in-hospital death were removed from the admissions table before index selection and labelling. This is the CMS index criterion ('patient is alive upon discharge', with the rationale that patients who die during the initial hospitalization cannot be readmitted) applied as a row filter, and it is the cohort step used in MIMIC readmission studies, for example ClinicalBERT on MIMIC-III ('in-hospital death precludes readmission, so admissions with deaths are removed') and recent MIMIC-IV readmission cohorts that exclude in-hospital deaths before labelling. Applied to the whole table, the filter also removes a return admission that ended in death, whereas the CMS measure counts such a return as a readmission; the consequence in our data is that an index admission followed only by a readmission ending in death was labelled negative (709 of 218,196 index admissions, 0.32%). The Methods state the rule and this consequence, and post-discharge mortality was not used in labelling, which limitation (viii) records as the absence of a competing-outcome formulation. To size the effect we re-scored the existing test predictions with those readmissions counted as events: 149 labels change (0.34% of the test cohort, 2.7% of positives), AUROC moves from 0.766 to 0.762 and PR-AUC from 0.342 to 0.341. We have not corrected the training labels. The model was trained on the stated definition, so a correction would require retraining and re-deriving every attribution, ranking and perturbation result for a change affecting 0.3% of labels, and the explanation framework, which is the object of the study, does not use the labels. We state this as a decision rather than as future work.",
         "None beyond the Methods statement listed under points 5 and 6; the sensitivity figures above are offered for the response and can be added to the Supplementary Methods if the editor prefers.")

    item(doc, "2", "Penultimate-admission index; single-admission patients",
         "Defining the index as the penultimate admission uses future information, and single-admission patients are negative by construction.",
         "The design takes one index per patient, which avoids within-patient correlation and gives every index the longest available history; the object of study is the explanation of a trained model, not a deployable risk score, and we say so. The reviewer's reading of Table 1 conflates admissions through the index with lifetime admissions, but the structural negatives are real: 119,874 patients (54.9%) have a single eligible hospitalization and are negative by construction (training 83,940; validation 12,050; test 23,884), and the readmission rate among patients with two or more admissions is 28.0% (28.1%, 26.9% and 28.1% by split).",
         "Proposed: two sentences in Methods, 'Data and prediction task', giving these counts; one clause in the scope limitations.")

    item(doc, "4", "Observation-type index admissions and same-day returns",
         "Observation stays and returns within a day may not represent readmissions in the usual sense; a sensitivity analysis excluding them is requested.",
         "The outcome is any unplanned return to hospital care recorded in the admissions table within 0 to 30 days of the index discharge, with elective and same-day surgical returns excluded as planned; encounters were not merged and no minimum interval was imposed, as the Methods now state. This definition is deliberately broader than the inpatient-only CMS measure: observation encounters are part of each patient's longitudinal record in MIMIC-IV, and a sequence model that reads that record should be explained on the same encounter set it was trained on. Observation and emergency returns within a day are therefore returns under our definition, not artefacts of it. We did not restrict the endpoint to inpatient status because that would change the prediction task rather than test the explanation framework, and the definition is stated so that readers can judge the endpoint.",
         "None beyond the Methods statement listed under points 5 and 6.")

    item(doc, "7", "All-[PAD] reference is patient-specific and far from the base rate",
         "The reference logit implies a probability near 0.43 and correlates with sequence length, so 'protective' means lower than an empty sequence of the same length.",
         "The all-[PAD] reference is the standard baseline for token models. It is explicitly specified, its distribution is reported in the Supplementary Methods, completeness is checked, and every attribution in the paper is interpreted relative to it, which the Methods now state in defining the sign. We keep it as a design choice rather than claim that the results are independent of it: the patterns tested were selected with this reference and the reversal analyses compare reference-relative attributions, whereas the masking effects are changes in the model's own output that do not involve the reference. Whether an alternative reference would change the rankings is an open robustness question, which is why the closing note lists it among the analyses not undertaken here. The per-patient identities added under point 8 make the reference explicit for each example (0.575 for the Figure 2 patient). The distribution reported in the Supplementary Methods comes from 300 randomly sampled test patients (reference probability median 0.43, interquartile range 0.38 to 0.51; Spearman 0.76 with sequence length), which characterises the reference well enough that a full-cohort recomputation would not change the description.",
         "Proposed: one sentence on sign semantics moved into Methods, 'AToP explanation framework'.")

    item(doc, "9", "Number of prior admissions is absorbed into the reference",
         "The strongest tabulated predictor never appears in an explanation.",
         "Visit count is structural rather than a token, so no token-level attribution on this input would surface it; the carrier-conditional view scores tokens only. This is a property of token-level explanation of this input representation, and we now state it.",
         "Proposed: one Discussion sentence.")

    item(doc, "10", "Pattern score is an additive sum",
         "Summed token attribution cannot show that a combination or order matters beyond its tokens, and longer patterns rank higher mechanically.",
         "The score is, by design, an allocation of additive token attribution to a temporal template; the claim about order rests on the reversal analyses, and the manuscript says so. We have reworded the one Discussion sentence that could be read as an interaction claim.",
         "Done in the current build: Discussion, 'organizes attributed events into recurring combinations and sequences' wording.")

    item(doc, "11", "No unguided-mining comparator; interaction-attribution methods not discussed",
         "A control without attribution guidance and a comparison with interaction attribution are missing.",
         "Attribution guidance is the design principle of AToP rather than an optimisation to be ablated: the question asked is which combinations and sequences of events the model used, so the search is confined to the events the model attributed, which also makes episode mining tractable (the salient subsequence is 2.2-fold shorter than the record). Mining the full record answers a different question, which events are frequent, and a comparison between the two would evaluate frequency mining rather than the explanation. We have therefore not run an unguided comparator and make no claim about what it would return. Interaction attribution methods assign pairwise credit and are complementary; the Background now says so.",
         "Proposed: one Background sentence citing Integrated Hessians and Shapley interaction indices as complementary.")

    item(doc, "15", "Within-patient reversal moves whole visits",
         "Reversal confounds order with recency and with co-occurring events in the moved visits.",
         "Acknowledged. The between-patient analysis is the complement and involves no perturbation; the limitation already names the shared caveat.",
         "Proposed: 'including recency' added to limitation (i).")

    item(doc, "17 (baseline)", "No bag-of-codes baseline on the same vocabulary",
         "Without a baseline on the same tokens, the gain over LACE models cannot be attributed to sequence modelling.",
         "The manuscript does not attribute the gain to sequence modelling. Table 2 labels the comparators as LACE-feature models, and limitation (iv) states that the gap reflects feature richness and architecture jointly. Whether the model uses order is tested directly by the reversal analyses, which change visit order while holding content fixed within the same model; that is the evidence the paper relies on for the ordering claims.",
         "None.")

    item(doc, "19", "Masking to [PAD] equals the IG reference",
         "Direction agreement under masking is expected by construction, and only direction is reported.",
         "Completeness fixes the total over all tokens, not the direction of a change when a subset is removed, so the agreement is informative but not the strongest evidence. The magnitude relationship, which the manuscript did not report, is strong: across the 266 masked patterns the change in predicted probability tracks the pattern's attribution with Spearman ρ = 0.94 (0.79 among the 168 risk-supporting and 0.92 among the 98 protective patterns), slope 0.15 probability units per logit unit.",
         "Proposed: one Results sentence after the direction result, from existing outputs.")

    item(doc, "21", "Token-level faithfulness effect is small and compared only with random tokens",
         "Masking the top five tokens moves predictions by 0.079 versus 0.056 for random salient tokens.",
         "Random salient tokens are the standard comparator for comprehensiveness and sufficiency, and the paired bootstrap intervals do not overlap at any k. The effects are within-model differences in the model's own output, so their interpretation does not depend on the probability scale being calibrated.",
         "None.")

    item(doc, "24 and 25", "Uncertainty of pattern scores and selection of the top 30",
         "The carrier-averaged scores carry no uncertainty, and a top 30 selected by magnitude on the same data overstates the scores and may not be a stable set.",
         "The carrier-averaged scores in Tables 3 and 4 are descriptive summaries of the model's attributions over the patients in the test cohort who carry each pattern. They are sample means, and their variability with the carriers sampled is largest where carrier counts are small; for that reason n is shown for every row and rows below 20 carriers are marked as exploratory. The rankings are not presented as estimates of population quantities, and selecting the largest means from a large pool can favour noisy rows, which is one reason the behavioural claims rest on the masking and reversal analyses rather than on rank position. For a fixed model, reference and step count the per-patient attribution itself is deterministic up to the approximation error bounded by the completeness check, so the uncertainty in question is that of the cohort summary, not of the attribution method. The framework is a procedure for probing a sequence model, and the manuscript shows that the resulting ranking depends on the pattern pool, the scoring rule, the deduplication threshold, the saliency threshold and the attribution method (Results, 'Pattern discovery'; Figures S3 to S5, S10 and S13; Limitations ii and iii). We do not present any single top 30 as a definitive set, and the claims about model behaviour rest on the masking and reversal analyses rather than on the ranking.",
         "None.")

    item(doc, "29", "Calibration",
         "Only discrimination is reported; calibration is not assessed.",
         "Not assessed. The model was trained with a class-weighted loss (positive weight 6.91), so its output probabilities are shifted upward by construction and are not intended as risk estimates; the mean prediction in the test cohort is 0.39 against an event rate of 0.13. The study uses discrimination only to establish that the model is worth explaining and makes no deployment or risk-stratification claim. All attribution is computed on the logit, and the masking effects (Δŷ) are changes in the model's own output with and without a perturbation; they are internal measures of what the model responds to and are not presented, and should not be read, as changes in calibrated clinical risk.",
         "None.")

    item(doc, "35", "'Clinically legible' without clinician review; care-process patterns",
         "No clinician assessed the patterns, and several top patterns are care-process sequences.",
         "Clinician evaluation is the natural next study and we claim no clinical validity; the Discussion already attributes some surfaced patterns to documentation and workflow.",
         "Abstract conclusion: 'clinically legible trajectories' replaced by 'visit-anchored temporal pattern narratives'; a care-process example (chest pain with chest radiography and respiratory measurements) named in the Discussion paragraph on unspecified codes.")

    item(doc, "37", "'Protective' implies clinical protection",
         "Negative attribution is not protection.",
         "We keep the term, defined at first use as negative attribution relative to the reference, as every figure legend does.",
         "Proposed: definition at first use in Results.")

    item(doc, "38", "Agreement with observed risk in Table S3 is descriptive",
         "Several intervals cross 1 and no multiplicity correction is applied.",
         "The table is presented as directional agreement, not as validation of clinical risk; its note says no multiplicity correction was applied.",
         "Proposed: the count of intervals crossing 1 added to the Table S3 note.")

    item(doc, "39", "Narrative coverage beyond the exemplars",
         "The three exemplars do not show how much of a typical patient's attribution the pattern narrative captures.",
         "The narrative is an explanation format, and every exemplar displays the attribution that no pattern captured: the Figure 2 legend states that remaining attribution is shown as other salient and non-salient tokens, and Figure S2C was included as the partial case, with +0.240 allocated to the named pattern against +0.969 from thirteen other salient tokens. The allocation rule is given in the Supplementary Methods and the Results state that bars are allocations rather than independent effects. The behavioural claims rest on the masking and reversal analyses, not on narrative coverage.",
         "None.")

    # ── Closing note ─────────────────────────────────────────────────
    heading(doc, "Closing note", level=2)
    para(doc, "Requests that require retraining or a full attribution rerun (an alternative IG reference with all figures re-derived, a matched KernelSHAP rerun, permutation nulls for the within-patient test, a shuffled-visit Transformer, an unguided mining comparator) are not planned for this revision and will be stated as future work. The outcome-label question under point 1 is not deferred in this way: the rule, its consequence and its measured effect on discrimination are stated, and the decision not to retrain is taken on that evidence.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}  (paragraphs: {len(doc.paragraphs)})")


if __name__ == "__main__":
    build()
