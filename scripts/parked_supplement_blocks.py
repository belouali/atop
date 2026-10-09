"""Parked supplementary exhibits (backup, not built).

Removed from build_revised_supplement_docx.py because no analysis in the
manuscript depends on them. Labels are the pre-renumbering ones. To
restore, paste a block back into build() in the supplement builder and
renumber. Images and data are in figures/supplements/backup/ and
tables/backup/.

- Figure S2: mined-pattern count vs minimum-support threshold
  (counts are already in the pipeline-summary table).
- Table S5: notation (every symbol is defined in the main Methods).
- Shuffle controls (within-visit and visit-block permutation): removed from
  the paper. Within-visit order is not claimed by AToP, and the per-pattern
  signed averages understated per-patient effects (risk moved up and down
  in similar shares of carriers). A per-patient rerun (all 266 patterns,
  50 draws, same carriers; masking reproduced to <1e-7) gave mean |dy| per
  patient of 0.075 (mask), 0.049 (within-visit, 66%) and 0.029 (visit
  blocks, 38%); because shuffled inputs are out of distribution, the
  reversal analysis carries the order claim. Script:
  scripts/rerun_perturbation_per_patient.py; per-patient values in
  runs/mimiciv_readmission_30d/5_analysis/backup/ (patient-level, never
  commit). The mask-vs-shuffle figure is figures/supplements/backup/.
  Note: the shuffle runs and the first per-patient rerun used the earlier
  tensors_test.pt, which had been regenerated without the in-sequence [PAD]
  placeholders for out-of-vocabulary codes (positions shifted for 7.3% of
  test patients). All reported analyses were recomputed on correctly
  rebuilt inputs; the parked shuffle numbers were not.
"""

def parked_blocks(doc, IMAGE_PATHS, _read_csv):  # pragma: no cover
    # S2
    page_break(doc)
    heading(doc, "Figure S2. Mined-pattern count vs minimum-support "
                  "threshold", level=2)
    figure_placeholder(doc, "Figure S2",
        "Cumulative count of mined episode patterns surviving each minimum-"
        "support threshold, on log-log axes, stacked by pattern structure "
        "(Single token, Single-visit compound, Two-block cross-visit, Three+ "
        "block cross-visit). The dashed cross-hair marks the locked AToP "
        "support threshold (x = 153 = 0.1% of 152,737 training patients; "
        "y = 24,490 candidate patterns at that threshold). The hatched gray "
        "zone at x < 153 is unmined: the mining cache contains no patterns "
        "below the threshold, so composition at lower support is not "
        "observed. At the locked threshold the training cohort yields 24,490 "
        "frequent episodes (Table S1). These are scored against the test "
        "cohort and deduplicated at j ≥ 0.2 to 17,891 full-pool "
        "representatives, of which the 9,213-pattern cross-visit subset is "
        "the locked reporting dictionary. The steep drop above "
        "the threshold visualizes the trade-off: raising support drops "
        "clinically rare cross-visit patterns first, leaving the pool "
        "progressively dominated by single-token and within-visit patterns.",
        image_path=IMAGE_PATHS.get("Figure S2"))

    # ── Table S5: notation table ───────────────────────────────────
    page_break(doc)
    df_s5 = _read_csv("TableS5_notation.csv")
    table_title(doc, "Table S5",
                 "Notation and terminology used throughout the manuscript.")
    s5_rows = [[_fmt_str(r[c]) for c in df_s5.columns] for _, r in df_s5.iterrows()]
    make_table(doc,
                header_row=list(df_s5.columns),
                data_rows=s5_rows,
                col_widths_in=[2.2, 4.7],
                font_size_pt=8)
    table_legend(doc,
        abbreviations="",
        note=(
            "Signed quantities preserve risk (>0) vs protective (<0) "
            "direction; absolute quantities (|IG|) are used only for "
            "magnitude ranking. Logit-space quantities (IG, ΣIG, z) are "
            "not directly additive with probability-space quantities "
            "(ŷ, Δŷ)."
        ),
    )

