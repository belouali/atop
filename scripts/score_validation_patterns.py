"""score_validation_patterns: carrier-averaged signed IG for every validated pattern.

Scores each pattern in ``5_analysis/validation_<split>.csv`` on the full
test cohort with the single rule in ``atop/explain/scoring.py`` and writes
``5_analysis/validation_pattern_directions.csv`` (pattern, sigma_ig_test,
n_carriers_test). ``make_fig_s10_masking.py`` uses it to colour the masking
histogram by attribution direction.

Usage:
    python scripts/score_validation_patterns.py --run_dir ../runs/mimiciv_readmission_30d
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.explain.scoring import (build_patient_token_occurrences, build_presence_index,
                                  episode_to_itemsets, score_episode_over_cohort)
from atop.explain.validation import _parse_episode_pattern
from atop.run_layout import RunLayout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    args = p.parse_args()
    layout = RunLayout(args.run_dir)
    val = pd.read_csv(layout.validation_csv(args.split))
    ig = pd.read_csv(layout.ig_csv(args.split), dtype={"patient_id": str, "index_hadm_id": str},
                     usecols=["patient_id", "index_hadm_id", "visit_idx", "token_str", "ig_signed"])
    pt = build_patient_token_occurrences(ig)
    idx = build_presence_index(pt)
    rows = []
    for pat in val["pattern"]:
        res = score_episode_over_cohort(episode_to_itemsets(_parse_episode_pattern(pat)), pt, idx)
        scores, carriers = res if res else ([], [])
        rows.append(dict(pattern=pat, sigma_ig_test=float(np.mean(scores)) if scores else np.nan,
                         n_carriers_test=len(carriers)))
    out = os.path.join(layout.analysis_dir, f"validation_pattern_directions.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} patterns)")


if __name__ == "__main__":
    main()
