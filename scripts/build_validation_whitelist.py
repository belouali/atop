"""build_validation_whitelist, derive the validation whitelist from the
manuscript tables that will actually display Δŷ.

Single source of truth: the patterns that appear in Tables 3 and 4 of the
manuscript are the ones that need masking-validation Δŷ values. This
script reads those table CSVs, applies the same phenotype-organized
top-5-per-family filter that ``build_revised_main_docx.py`` uses, and
emits the union as ``validation_whitelist.txt``.

The downstream GPU step (``run_validation_topup.py``) then validates
exactly the patterns that the manuscript will display, no over-
validation (wasted GPU on patterns the reader never sees) and no
under-validation (NaN Δŷ for displayed patterns).

If the manuscript's table-shape logic ever changes (more phenotypes,
different per-family count, different |ΣIG| top-N source), update both
this script and the build script in lockstep so they remain
synchronized.

Legacy ``--top_n`` mode is preserved for back-compat: pass
``--legacy_topn N`` to use the original "top-N from each pool" logic
instead of the manuscript-derived logic.

Usage (canonical, manuscript-driven):
    python scripts/build_validation_whitelist.py --run_dir ../runs/mimiciv_readmission_30d

Usage (legacy):
    python scripts/build_validation_whitelist.py --run_dir ../runs/mimiciv_readmission_30d \\
        --legacy_topn 50

After running this, run the GPU validation top-up:
    python scripts/run_validation_topup.py --run_dir ../runs/mimiciv_readmission_30d \\
        --mimic_dir ../data/mimiciv/3.1/hosp --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.explain.cooccurrence import (
    score_all_raw_patterns, cluster_jaccard_inmemory,
)
from atop.explain.jaccard import get_jaccard_representatives
from atop.run_layout import RunLayout


# Must match build_revised_main_docx.py and build_revised_supplement_docx.py.
# The four families displayed in Tables 3 and 4 of the main paper.
MANUSCRIPT_PHENOTYPES = ["Psychiatric", "Oncologic", "Obstetric",
                          "Chest pain workup",
                          "Hepatobiliary", "Substance use", "Other"]
TOP_PER_FAMILY = 5


def _topn_reps(df_pool: pd.DataFrame, rep_set: set, n: int) -> list[str]:
    """Top-N patterns by |ig_signed_mean| from the rep subset."""
    sub = df_pool[df_pool["pattern"].isin(rep_set)].copy()
    sub["_abs"] = sub["ig_signed_mean"].abs()
    return sub.nlargest(n, "_abs")["pattern"].tolist()


def _load_or_score_full_pool(layout: RunLayout, split: str,
                              min_carriers: int = 5) -> pd.DataFrame:
    """Same caching logic as ``make_fig3.py``: cache to
    ``<analysis_dir>/full_pool_scored_<split>.pkl`` and reuse if newer
    than the raw mining cache."""
    import pickle
    cache_path = os.path.join(layout.analysis_dir,
                               f"full_pool_scored_{split}.pkl")
    raw_files = sorted([f for f in os.listdir(layout.mining_dir)
                        if f.startswith("raw_episode_")])
    if not raw_files:
        sys.exit(f"  [ERROR] No raw_episode_*.pkl in {layout.mining_dir}")
    raw_path = os.path.join(layout.mining_dir, raw_files[-1])
    if (os.path.exists(cache_path)
            and os.path.getmtime(cache_path) >= os.path.getmtime(raw_path)):
        with open(cache_path, "rb") as f:
            df = pickle.load(f)
        return df
    print(f"  [full pool] computing from raw cache "
          f"(min_carriers={min_carriers})…")
    with open(raw_path, "rb") as f:
        raw_mining = pickle.load(f)
    with open(layout.sequences_pkl(split), "rb") as f:
        df_seq = pickle.load(f)
    df_ig = pd.read_csv(layout.ig_csv(split))
    df = score_all_raw_patterns(
        raw_mining, df_seq, df_ig,
        min_carriers=min_carriers, min_blocks=1, min_tokens_total=1,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def _patterns_from_manuscript_table(csv_path: str) -> list[str]:
    """Apply the phenotype-organized top-5-per-family filter that the
    main-docx builder uses. For the catch-all "Other" phenotype, take
    top-5 risk-supporting (ΣIG > 0) AND top-5 protective (ΣIG < 0)
    SEPARATELY — since "Other" is a heterogeneous bucket and a single
    top-5-by-|IG| selection skews to whichever direction has bigger
    absolute values.
    Returns the canonical 'pattern' keys (not the readable labels).
    """
    if not os.path.exists(csv_path):
        sys.exit(f"  [ERROR] Missing {csv_path}\n"
                  f"  Run scripts/make_table3.py first to produce it.")
    df = pd.read_csv(csv_path)
    if "pattern" not in df.columns:
        sys.exit(f"  [ERROR] {csv_path} has no 'pattern' column "
                  "(needed as canonical key for validation lookup).")
    df = df[df["phenotype"].isin(MANUSCRIPT_PHENOTYPES)].copy()
    df = df.assign(_abs=df["ig_value"].abs())

    # Non-Other families: standard top-5 by |IG|
    non_other = df[df["phenotype"] != "Other"].copy()
    non_other["_pord"] = non_other["phenotype"].apply(
        lambda p: MANUSCRIPT_PHENOTYPES.index(p))
    non_other = (non_other
                  .sort_values(["_pord", "_abs"], ascending=[True, False])
                  .groupby("phenotype", sort=False, group_keys=False)
                  .head(TOP_PER_FAMILY))

    # Other: top-5 risk + top-5 protective
    other_risk = (df[(df["phenotype"] == "Other") & (df["ig_value"] > 0)]
                   .sort_values("_abs", ascending=False)
                   .head(TOP_PER_FAMILY))
    other_prot = (df[(df["phenotype"] == "Other") & (df["ig_value"] < 0)]
                   .sort_values("_abs", ascending=False)
                   .head(TOP_PER_FAMILY))

    keys = (non_other["pattern"].tolist()
             + other_risk["pattern"].tolist()
             + other_prot["pattern"].tolist())
    # Drop duplicates while preserving first-occurrence order.
    seen = set()
    return [p for p in keys if not (p in seen or seen.add(p))]


def manuscript_driven(args: argparse.Namespace, out_path: str) -> None:
    """Derive the whitelist from the two manuscript table CSVs."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    tables_dir = os.path.join(repo_root, "manuscript",
                                "jamia_submission", "tables")
    table3_path = os.path.join(tables_dir, "Table4_cross_visit_clusters.csv")
    table4_path = os.path.join(tables_dir, "Table3_full_pool_clusters.csv")

    print(f"[1] Table 3 (cross-visit, Figure 4B):  {table3_path}")
    t3 = _patterns_from_manuscript_table(table3_path)
    print(f"    → {len(t3)} patterns "
          f"({len(MANUSCRIPT_PHENOTYPES)} families × top-{TOP_PER_FAMILY})")

    print(f"\n[2] Table 4 (all patterns, Figure 4A): {table4_path}")
    t4 = _patterns_from_manuscript_table(table4_path)
    print(f"    → {len(t4)} patterns "
          f"({len(MANUSCRIPT_PHENOTYPES)} families × top-{TOP_PER_FAMILY})")

    union = list(dict.fromkeys(t3 + t4))  # preserves order, dedupes
    overlap = len(t3) + len(t4) - len(union)
    print(f"\n[3] Union: {len(union)} unique patterns "
          f"({overlap} appear in both tables)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for pat in union:
            f.write(pat + "\n")
    print(f"    → {out_path}")

    manifest_path = os.path.splitext(out_path)[0] + "_manifest.csv"
    pd.DataFrame([{
        "pattern":     pat,
        "in_table3":   pat in set(t3),
        "in_table4":   pat in set(t4),
        "table3_rank": t3.index(pat) + 1 if pat in t3 else None,
        "table4_rank": t4.index(pat) + 1 if pat in t4 else None,
    } for pat in union]).to_csv(manifest_path, index=False)
    print(f"    → {manifest_path}")


def legacy_topn(args: argparse.Namespace, out_path: str) -> None:
    """Original behavior: top-N from each pool. Useful when you need to
    validate beyond what's currently surfaced in the manuscript (e.g.
    you're about to bump TOP_PER_FAMILY or add a phenotype family)."""
    layout = RunLayout(args.run_dir)
    n = args.legacy_topn

    print(f"[legacy] Cross-visit pool (Panel D)…")
    pw_csv = layout.pathway_csv(args.split)
    df_pw = pd.read_csv(pw_csv)
    df_pw = df_pw[df_pw["n_present"] >= 3].copy()
    rep_d = get_jaccard_representatives(
        layout.analysis_dir, args.split, args.jaccard_threshold)
    if rep_d is None:
        sys.exit("  [ERROR] Cross-visit Jaccard reps missing")
    panel_d_top = _topn_reps(df_pw, rep_d, n)
    print(f"    → {len(panel_d_top)} patterns")

    print(f"\n[legacy] Full pool (Panel C)…")
    df_full = _load_or_score_full_pool(layout, args.split,
                                         min_carriers=args.min_carriers)
    rep_c = cluster_jaccard_inmemory(df_full, args.jaccard_threshold,
                                       verbose=False)
    panel_c_top = _topn_reps(df_full, rep_c, n)
    print(f"    → {len(panel_c_top)} patterns")

    union = list(dict.fromkeys(panel_d_top + panel_c_top))
    overlap = len(panel_d_top) + len(panel_c_top) - len(union)
    print(f"\n[legacy] Union: {len(union)} unique patterns "
          f"({overlap} overlap)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for pat in union:
            f.write(pat + "\n")
    print(f"    → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the validation whitelist for masking Δŷ.")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--jaccard_threshold", type=float, default=0.2,
                    help="Used by --legacy_topn mode.")
    p.add_argument("--min_carriers", type=int, default=5,
                    help="Used by --legacy_topn mode.")
    p.add_argument("--legacy_topn", type=int, default=None,
                    help="Use original top-N-per-pool logic instead of "
                         "manuscript-derived. Pass an int (e.g. 50).")
    p.add_argument("--out_path", default=None,
                    help="Whitelist output path (default: "
                         "<run_dir>/5_analysis/validation_whitelist.txt).")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    out_path = args.out_path or os.path.join(
        layout.analysis_dir, "validation_whitelist.txt")

    if args.legacy_topn:
        legacy_topn(args, out_path)
    else:
        manuscript_driven(args, out_path)


if __name__ == "__main__":
    main()
