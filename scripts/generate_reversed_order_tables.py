"""Generate Table S2 (top 30 by attenuation across the full mined pool)
and Table S3 (the 35 Table-4 cross-visit patterns, organized by phenotype)
from the cached reversed_order_test.csv + whitelist manifest.

Run AFTER scripts/run_reversed_order_topup.py finishes. Outputs:
  manuscript/jamia_submission/tables/TableS2_reversed_order_top30.csv
  manuscript/jamia_submission/tables/TableS3_reversed_order_table4.csv
"""
from __future__ import annotations
import os, sys, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atop_package"))

# Import the resolver from atop_package
ATOP_ROOT = os.path.dirname(os.path.abspath(__file__))   # atop_package/scripts
ATOP_PKG = os.path.dirname(ATOP_ROOT)                     # atop_package
sys.path.insert(0, ATOP_PKG)
from atop.explain.pattern_labels import PatternLabelResolver


RUN_DIR = "../runs/full_CPD_v2"
TABLES_OUT = "../manuscript/jamia_submission/tables"

PHEN_ORDER = ["Psychiatric", "Oncologic", "Obstetric", "Chest pain workup",
               "Hepatobiliary", "Substance use", "Other"]


def _load_and_score():
    df = pd.read_csv(os.path.join(RUN_DIR, "5_analysis", "reversed_order_test.csv"))
    m = pd.read_csv(os.path.join(RUN_DIR, "5_analysis", "reversed_order_whitelist_manifest.csv"))

    df = df.merge(
        m[["pattern", "in_fig4b", "in_table4", "phenotype", "in_topup"]],
        on="pattern", how="left",
    )
    df = df[(df["n_forward"] > 0) & (df["n_reversed"] > 0)].copy()

    df["abs_fwd"] = df["ig_mean_forward"].abs()
    df["abs_rev"] = df["ig_mean_reversed"].abs()
    df["attenuation_pct"] = 100.0 * (1 - df["abs_rev"] / df["abs_fwd"])
    df["n_blocks"] = df["pattern"].apply(lambda p: str(p).count(" -> ") + 1)

    resolver = PatternLabelResolver(
        RUN_DIR, short_names_csv=os.path.join(ATOP_ROOT, "short_names.csv"))
    df["readable_label"] = df["pattern"].apply(
        lambda p: resolver.format_pattern(p, short_names=True))

    # Source attribution
    def _src(r):
        srcs = []
        if r["in_fig4b"]: srcs.append("Fig 4B")
        if r["in_table4"]: srcs.append("Table 4")
        if r["in_topup"] and not srcs: srcs.append("Topup pool")
        return ", ".join(srcs) if srcs else ""
    df["source"] = df.apply(_src, axis=1)
    return df


def write_table_s2(df: pd.DataFrame) -> None:
    """Top 30 by attenuation across the full pool, FILTERED to require
    BOTH forward and reversed match counts ≥ 20 (the same floored pool
    Figure S1 displays). Without both floors, the top of the attenuation
    ranking is dominated by patterns whose forward or reverse ordering
    barely occurs in the data (match count ≤ 3 with |ig| ≈ 0), which
    measures data sparsity rather than model temporal sensitivity."""
    MIN_N = 20
    eligible = df[(df["n_forward"] >= MIN_N)
                  & (df["n_reversed"] >= MIN_N)].copy()
    top = eligible.nlargest(30, "attenuation_pct").copy()
    out_cols = ["readable_label", "n_blocks", "phenotype", "source",
                 "n_forward", "n_reversed",
                 "ig_mean_forward", "ig_mean_reversed", "attenuation_pct"]
    top = top[out_cols].copy()
    for c in ("ig_mean_forward", "ig_mean_reversed"):
        top[c] = top[c].round(3)
    top["attenuation_pct"] = top["attenuation_pct"].round(1)
    out_path = os.path.join(TABLES_OUT, "TableS2_reversed_order_top30.csv")
    os.makedirs(TABLES_OUT, exist_ok=True)
    top.to_csv(out_path, index=False)
    print(f"  → {out_path} ({len(top)} rows, filter n_fwd & n_rev ≥ {MIN_N})")
    print(f"    Eligible pool (n_fwd ≥ {MIN_N}, n_rev ≥ {MIN_N}): "
           f"{len(eligible):,} of {len(df):,}")
    print(f"    Attenuation range in top 30: "
           f"{top['attenuation_pct'].min():.1f}% to "
           f"{top['attenuation_pct'].max():.1f}%")
    print(f"    Phenotype balance: "
           f"{top['phenotype'].fillna('(unclassified)').value_counts().to_dict()}")
    print(f"    Block-count distribution: "
           f"{top['n_blocks'].value_counts().sort_index().to_dict()}")
    print(f"    n_reversed range in top 30: {top['n_reversed'].min()} to "
           f"{top['n_reversed'].max()}")


def write_table_s3(df: pd.DataFrame) -> None:
    """The 35 Table-4 cross-visit patterns, organized by phenotype."""
    t4 = df[df["in_table4"]].copy()
    t4["_pord"] = t4["phenotype"].apply(
        lambda p: PHEN_ORDER.index(p) if p in PHEN_ORDER else 99)
    t4 = t4.sort_values(["_pord", "abs_fwd"], ascending=[True, False])

    out_cols = ["phenotype", "n_blocks", "readable_label",
                 "n_forward", "n_reversed",
                 "ig_mean_forward", "ig_mean_reversed", "attenuation_pct"]
    t4 = t4[out_cols].copy()
    for c in ("ig_mean_forward", "ig_mean_reversed"):
        t4[c] = t4[c].round(3)
    t4["attenuation_pct"] = t4["attenuation_pct"].round(1)

    out_path = os.path.join(TABLES_OUT, "TableS3_reversed_order_table4.csv")
    t4.to_csv(out_path, index=False)
    print(f"  → {out_path} ({len(t4)} rows, 35 Table-4 cross-walk)")


def main():
    df = _load_and_score()
    print(f"Loaded {len(df)} valid reversed-order rows "
           f"(of {(df['n_forward'] > 0).sum() + (df['n_reversed'] <= 0).sum()} attempted)")
    print(f"  Attenuation distribution: min={df['attenuation_pct'].min():.1f}%, "
           f"median={df['attenuation_pct'].median():.1f}%, "
           f"max={df['attenuation_pct'].max():.1f}%")
    print()
    write_table_s2(df)
    print()
    write_table_s3(df)


if __name__ == "__main__":
    main()
