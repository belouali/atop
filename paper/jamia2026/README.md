# JAMIA 2026 submission: exhibit and document builders

This folder holds the code that turns the outputs of the AToP pipeline into the
figures, tables and documents of the JAMIA submission. It is separate from the
installable library (`atop/`) and from the pipeline entry points (`scripts/`).
Nothing here is needed to use the library.

```
paper/jamia2026/
  builders/   Word and PDF builders and their shared helper
  figures/    figure scripts, replot script, assets/, short_names.csv
  tables/     table scripts
```

## Conventions

- Run every command from the repository root (the `atop_package/` folder).
  The figure and table scripts take `--run_dir`, and their defaults for the
  run directory and for the output folders start with `../`, so they resolve
  against the project folder one level above the repository (`runs/` and
  `manuscript/`). The paper's run directory is `../runs/mimiciv_readmission_30d`.
- The builders locate everything from their own file location, so they run
  from any working directory. They read tables and figures from
  `manuscript/jamia_submission/` and write the documents back into it.
- Neither `runs/`, `data/` nor `manuscript/` is part of the repository.
- Several scripts have a stale default run directory (`../runs/full_CPD_v1`).
  Always pass `--run_dir ../runs/mimiciv_readmission_30d` explicitly.

## Build order

### 1. Pipeline outputs (from `scripts/`)

The exhibits read the run directory written by the pipeline steps in the top
level README: `preprocess.py`, `train_model.py`, `run_atop.py`,
`select_exemplars.py` (exemplar patients for Figures 2 and S2),
`score_validation_patterns.py` (directions for Figure S7),
`build_validation_whitelist.py`, `run_validation_topup.py` and
`run_reversed_order_topup.py`, plus the baselines scripts for Table 2 and the two
KernelSHAP scripts for Figures S11 and S12.

### 2. Main figures

| Exhibit | Command | Notes |
|---|---|---|
| Figure 1 | `python paper/jamia2026/figures/make_fig1.py` | Illustrative schematic. The submitted `Figure1.png` is kept as it is, so do not regenerate it for the submission. |
| Figure 2 | `python paper/jamia2026/figures/make_fig2.py --run_dir ../runs/mimiciv_readmission_30d` | Needs `exemplars.json` from `select_exemplars.py`. |
| Figure 3 | `python paper/jamia2026/figures/make_fig3.py --run_dir ../runs/mimiciv_readmission_30d` | Also writes the panel CSVs and pattern pools that Figures 4 and 5, Tables 3 and 4 and several supplementary figures read. Run it first. |
| Figure 4 | `python paper/jamia2026/figures/make_fig4.py --run_dir ../runs/mimiciv_readmission_30d` | Needs the Figure 3 panel CSVs. Writes `Figure4.csv` beside the PNG. |
| Figure 5 | `python paper/jamia2026/figures/make_fig5.py --run_dir ../runs/mimiciv_readmission_30d` | Needs the Figure 3 panel CSVs and the validation top-up. |

### 3. Main tables

| Exhibit | Command | Notes |
|---|---|---|
| Table 1 | `python paper/jamia2026/tables/make_table1.py --run_dir ../runs/mimiciv_readmission_30d --mimic_dir <mimic hosp dir> --manuscript-csv ../manuscript/jamia_submission/tables/Table1_cohort.csv` | Needs `lace_scores_all.csv` from `scripts/precompute_lace.py`. Also writes the `_meta.csv` file. |
| Table 2, model performance | `python paper/jamia2026/tables/make_table2.py --run_dir ../runs/mimiciv_readmission_30d --mimic_dir <mimic hosp dir>` | Wrapper around `scripts/compare_baselines.py`. Writes `baseline_comparison.csv` in the run directory. |
| Table 2, mining summary | `python paper/jamia2026/tables/make_table2_mining_summary.py --run_dir ../runs/mimiciv_readmission_30d` | Writes `Table2_mining_summary.csv` into the manuscript tables folder. |
| Tables 3 and 4 | `python paper/jamia2026/tables/make_table3.py --run_dir ../runs/mimiciv_readmission_30d --panel d --output-csv ../manuscript/jamia_submission/tables/Table4_cross_visit_clusters.csv`, and the same with `--panel c` and `Table3_full_pool_clusters.csv` | Needs Figures 3 and 4 first. |

`build_validation_whitelist.py` reads the Table 3 and Table 4 CSVs, and its
validation top-up supplies the masking values shown in Figure 5. Producing the
final versions of these exhibits therefore takes two passes through Figure 3,
Tables 3 and 4, the whitelist, the top-up, and Figures 4 and 5.

The builders also read `Table2_model_performance.csv`, `TableS4_parameters.csv`
and the three `TableS6_*.csv` files. No script in this repository writes them.
They are maintained in `manuscript/jamia_submission/tables/`.

### 4. Supplementary figures and tables

| Exhibit | Command (all take `--run_dir ../runs/mimiciv_readmission_30d` unless noted) |
|---|---|
| Figure S1, cohort flow | `python paper/jamia2026/figures/make_fig_consort.py` (no run directory) |
| Figures S2A to S2C, further exemplars | `python paper/jamia2026/figures/make_fig_s4.py` (calls `make_fig2.py`) |
| Figure S3, Jaccard sweep, cross-visit pool | `python paper/jamia2026/figures/make_fig_s7_jaccard_sensitivity.py` |
| Figure S4, Jaccard sweep, full pool | `python paper/jamia2026/figures/make_fig_s8_full_pool_sensitivity.py` |
| Figure S5, carrier versus population-weighted | `python paper/jamia2026/figures/make_fig_s10_pattern_aggregation_comparison.py` |
| Figure S6, prevalence versus impact | `python paper/jamia2026/figures/make_fig_s5.py` |
| Figure S7, masking distribution | `python paper/jamia2026/figures/make_fig_s10_masking.py` |
| Figure S8, faithfulness curves | `python paper/jamia2026/figures/make_fig_faithfulness.py` (needs `--mimic_dir`) |
| Figure S9, forward versus reversed order | `python paper/jamia2026/figures/make_fig_s10_reversal_scatter.py` |
| Figure S10, KernelSHAP versus IG | `python paper/jamia2026/figures/make_fig_s3.py` (GPU, writes the two `fig_s3_*.csv` files), then `python paper/jamia2026/figures/replot_fig_s10_shap_vs_ig.py` (replots from those CSVs) |
| Figures S11 and S12, KernelSHAP replication | `python paper/jamia2026/figures/make_fig_s11_shap_fig3.py` (needs `scripts/build_cohort_shap_cache.py` then `scripts/build_shap_aggregations.py`) |
| Figure S13, IG mass sensitivity | `python paper/jamia2026/figures/make_fig_s12_ig_mass_sensitivity.py` |
| Table S1, AUROC bootstrap | `python paper/jamia2026/tables/table_s1_auroc_bootstrap.py` (needs `scripts/infer_test.py` and `scripts/reproduce_lace_baselines.py` outputs) |
| Table S3, reversed-order table | `python paper/jamia2026/tables/generate_reversed_order_tables.py` (after `scripts/run_reversed_order_topup.py`; also writes Table S2 CSV) |

The figure numbers in the left column are the current numbers in the
supplement. Several PNG file names on disk still carry earlier numbers, for
example `FigureS10_pattern_aggregation_comparison.png` is Figure S5. The map
from figure number to file name is `IMAGE_PATHS` in
`builders/build_revised_supplement_docx.py`.

`figures/short_names.csv` holds the short token labels used by the figure
scripts and by `tables/generate_reversed_order_tables.py`.

### 5. Documents

Run these in this order. Each reads the tables and figures above and the
documents built before it.

| Step | Command | Writes (in `manuscript/jamia_submission/`) |
|---|---|---|
| 1 | `python paper/jamia2026/builders/build_revised_main_docx.py` | `AToP_JAMIA_submission.docx` |
| 2 | `python paper/jamia2026/builders/build_revised_supplement_docx.py` | `AToP_JAMIA_supplement.docx` |
| 3 | `python paper/jamia2026/builders/build_tables_only_docx.py` | `AToP_JAMIA_tables_only.docx` (imports table functions from steps 1 and 2) |
| 4 | `python paper/jamia2026/builders/build_merged_docx.py` | `AToP_merged.docx` (reads the step 1 and 2 documents) |
| 5 | `python paper/jamia2026/builders/build_response_letter_docx.py` | `revision/AToP_response_to_reviews.docx` |
| 6 | `python paper/jamia2026/builders/build_cover_letter_docx.py` | `AToP_JAMIA_cover_letter.docx` |
| 7 | `python paper/jamia2026/builders/build_self_contained_paper_pdf.py` | `AToP_self_contained.pdf` (reads the step 1 and 2 documents) |
| 8 | `python paper/jamia2026/builders/build_self_contained_paper_pdf.py --preprint` | `AToP_arxiv_preprint.pdf` |

`builders/_docx_helpers.py` is the shared helper imported by the Word builders.
The builders need python-docx, reportlab, pandas, matplotlib and pypdf.

## Checking a rebuild

A rebuild is expected to reproduce the submitted documents exactly in content.
Compare paragraph texts, table counts and, for the main and merged documents,
the counts of bold and superscript runs. For the PDFs compare page counts and
extracted text. Byte equality is not expected because of document timestamps.
