"""make_fig_s4, Supplementary Figure S4: alternate exemplars.

Renders the same 3-panel Fig 2 layout for two additional exemplars from
``exemplars.json``, written as separate PNGs (S4a + S4b) to the supplements
folder:

  S4a, ``alt_tp``        (different clinical phenotype from main_tp)
  S4b, ``confident_tn``  (confident true negative, protective story)

Internally it just calls ``make_fig2.main()`` twice with different
``--exemplar-key`` and ``--output-path`` overrides.

Usage:
    python scripts/make_fig_s4.py --run_dir ../runs/full_CPD_v1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Render Figure S4 (alt_tp + confident_tn)")
    p.add_argument("--run_dir", default="../runs/full_CPD_v1")
    p.add_argument("--mimic_dir", default="../data/mimiciv/3.1/hosp")
    p.add_argument("--output-dir",
                   default="../manuscript/jamia_submission/figures/supplements")
    p.add_argument("--keys", default="alt_tp,confident_tn",
                   help="Comma-separated exemplar keys to render "
                        "(filename suffix derived from the key)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig2_script = os.path.join(os.path.dirname(__file__), "make_fig2.py")

    # Map exemplar_key → submission filename
    file_map = {
        "alt_tp":       "FigureS4A_alt_tp.png",
        "confident_tn": "FigureS4B_confident_tn.png",
        "complex_tp":   "FigureS4_complex_tp.png",
        "disagreement": "FigureS4C_disagreement.png",
    }

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    for key in keys:
        if key not in file_map:
            print(f"  [WARN] Unknown exemplar key '{key}' \u2014 skipping")
            continue
        out = os.path.join(args.output_dir, file_map[key])
        cmd = [
            sys.executable, fig2_script,
            "--run_dir", args.run_dir,
            "--mimic_dir", args.mimic_dir,
            "--exemplar-key", key,
            "--output-path", out,
            "--no-id-suffix",   # build_revised_supplement_docx looks
                                # for the exact filename without a
                                # patient-id suffix.
        ]
        print(f"\n[{'='*60}]")
        print(f"  Rendering {key} \u2192 {out}")
        print(f"[{'='*60}]")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"  [ERROR] make_fig2 failed for {key}")

    print(f"\nDone \u2014 rendered {len(keys)} exemplar(s) to {args.output_dir}/")


if __name__ == "__main__":
    main()
