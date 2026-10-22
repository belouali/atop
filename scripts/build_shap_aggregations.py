"""build_shap_aggregations \u2014 SHAP-based token + pattern aggregation CSVs.

Reads the per-patient KernelSHAP cache produced by
``scripts/build_cohort_shap_cache.py`` and aggregates it into the same
CSV schemas Fig 3 / Fig S11 consume:

  * ``token_importance_shap_<split>.csv`` \u2014 per-token mean SHAP
    (carrier-conditional) + sample prevalence + n_present. Mirrors the IG
    schema in ``token_importance_<split>.csv``.
  * ``panel_c_shap_<split>_j<jt>_short.csv`` \u2014 full-pool patterns scored
    by carrier-averaged \u03A3SHAP (sum of constituent-token SHAPs across
    sampled carriers). Jaccard-deduplicated at the same threshold as
    Fig 3 Panel C.
  * ``panel_d_shap_<split>_j<jt>_short.csv`` \u2014 cross-visit patterns
    (\u22652 visits & \u22652 tokens) scored the same way. Mirrors Panel D.

These outputs let ``make_fig_s11_shap_fig3.py`` render a SHAP-based variant
of Fig 3 without changing the figure renderer's grammar \u2014 the input CSVs
have the same columns the canonical pipeline expects.

Method: for every (pid, hadm) in the SHAP cache, we have ``{token: signed
shap_value}``. To aggregate at the pattern level we multiply pattern
membership (carrier set from the IG-side caches) by per-pattern
constituent-token SHAP sums.

CPU-only. Fast (~30 s).

Usage:
    python scripts/build_shap_aggregations.py \\
        --run_dir ../runs/mimiciv_readmission_30d \\
        --shap_cache ../runs/mimiciv_readmission_30d/3_attribution/cohort_shap_test_n500.pkl
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.explain.cooccurrence import cluster_jaccard_inmemory
from atop.explain.jaccard import get_jaccard_representatives
from atop.explain.pattern_labels import PatternLabelResolver
from atop.run_layout import RunLayout


def _resolve_cache_path(layout, split: str, user_supplied: str) -> str:
    """Find a cohort SHAP cache. Tries (in order): caller's --shap_cache,
    the new single-file default ``cohort_shap_<split>.pkl``, then any
    legacy ``cohort_shap_<split>_n*.pkl`` (picks the largest by size).
    Returns the resolved path or exits."""
    candidates = []
    if user_supplied:
        candidates.append(user_supplied)
    candidates.append(os.path.join(layout.attribution_dir,
                                     f"cohort_shap_{split}.pkl"))
    legacy = sorted(
        glob.glob(os.path.join(layout.attribution_dir,
                                 f"cohort_shap_{split}_n*.pkl")),
        key=os.path.getsize, reverse=True,
    )
    candidates.extend(legacy)

    for p in candidates:
        if p and os.path.exists(p):
            return p

    sys.exit(
        f"  [ERROR] No cohort SHAP cache found. Tried:\n"
        + "\n".join(f"    \u2022 {c}" for c in candidates if c) +
        f"\n  Run scripts/build_cohort_shap_cache.py first (Step 1 in "
        f"notebooks/AToP_1B3_shap_cohort.ipynb \u2014 the GPU step, not the "
        f"dry-run).")


def _load_shap_cache(path: str) -> dict:
    with open(path, "rb") as f:
        cache = pickle.load(f)
    print(f"[1] Loaded SHAP cache: {len(cache):,} patients "
          f"({os.path.basename(path)})")
    return cache


def _aggregate_token_level(cache: dict, outlier_threshold: float = 10.0) -> pd.DataFrame:
    """Per-token: mean signed SHAP across sampled carriers, sample prevalence,
    n_present. Schema mirrors `5_analysis/token_importance_<split>.csv`.

    **Outlier filter:** KernelSHAP with insufficient ``nsamples`` for high-K
    patients (K=number of tokens) produces numerically unstable SHAP values
    (10\u00b9\u00b3-10\u00b9\u2075 range vs the bounded [-1,+1] expected from a sigmoid model).
    Per-patient-token entries with ``|SHAP| > outlier_threshold`` are
    excluded from the aggregation. With ``threshold=10`` and a sigmoid-output
    model (whose total prediction delta is bounded by 1), this filter only
    catches the underdetermined-regression artifacts \u2014 a real per-token
    SHAP value rarely exceeds \u00b15 even for highly influential tokens.
    """
    N_sample = len(cache)
    by_tok: dict = {}
    n_outliers_dropped = 0
    n_kept = 0
    for (_, _), shap_dict in cache.items():
        for tok, val in shap_dict.items():
            if abs(val) > outlier_threshold:
                n_outliers_dropped += 1
                continue
            by_tok.setdefault(tok, []).append(val)
            n_kept += 1

    if n_outliers_dropped:
        pct = 100 * n_outliers_dropped / max(n_outliers_dropped + n_kept, 1)
        print(f"  [outlier filter] dropped {n_outliers_dropped:,} per-patient-"
              f"token entries with |SHAP| > {outlier_threshold} "
              f"({pct:.3f}% of total), kept {n_kept:,}")

    rows = []
    for tok, vals in by_tok.items():
        n = len(vals)
        rows.append({
            "token": tok,
            "ig_mean":   sum(vals) / n,    # carrier-conditional E[SHAP|present]
            "ig_abs":    sum(abs(v) for v in vals) / n,
            "n_present": n,
            "prevalence": n / N_sample,
        })
    df = pd.DataFrame(rows)
    print(f"[2] Token-level: {len(df):,} unique tokens "
          f"(N_sample={N_sample:,})")
    return df


def _aggregate_pattern_level(cache: dict, sample_pids: set,
                              df_pw_full: pd.DataFrame,
                              carrier_sets: dict,
                              N_sample: int,
                              outlier_threshold: float = 10.0) -> pd.DataFrame:
    """Re-score each pattern under SHAP: for every sampled patient who is a
    carrier, sum constituent-token SHAPs; average across sampled carriers.

    Returns a DataFrame with the SAME shape as
    ``5_analysis/pathway_importance_<split>.csv`` so the existing Fig 3
    panel selection / Jaccard logic can consume it unchanged.

    **Outlier filter:** Per-patient pattern \u03A3SHAP is dropped if any
    constituent-token SHAP is unstable (``|SHAP| > outlier_threshold``).
    Otherwise a single high-K-patient artifact pollutes every pattern
    that contains the artifact's token.
    """
    rows = []
    n_patterns_with_sampled_carriers = 0
    total_carrier_drops = 0
    total_carrier_kept = 0
    for _, pat_row in df_pw_full.iterrows():
        pat = pat_row["pattern"]
        carriers = carrier_sets.get(pat, frozenset())
        if not carriers:
            continue
        sampled_carriers = [k for k in carriers if k in sample_pids]
        if not sampled_carriers:
            continue
        n_patterns_with_sampled_carriers += 1

        # Pattern \u2192 set of constituent tokens (union across blocks).
        constituent_tokens = set()
        for step in pat.split(" -> "):
            step = step.strip()
            if step.startswith("{") and step.endswith("}"):
                for t in step[1:-1].split(","):
                    constituent_tokens.add(t.strip())
            else:
                constituent_tokens.add(step)

        per_carrier_sums = []
        for k in sampled_carriers:
            shap_d = cache[k]
            # Only sum stable SHAP values; if ANY constituent token has a
            # numerically unstable per-patient SHAP, drop this carrier from
            # the aggregation entirely (better than letting a 10\u00b9\u2074-magnitude
            # value dominate the pattern's mean).
            tok_vals = [shap_d.get(t, 0.0) for t in constituent_tokens]
            if any(abs(v) > outlier_threshold for v in tok_vals):
                total_carrier_drops += 1
                continue
            per_carrier_sums.append(sum(tok_vals))
            total_carrier_kept += 1

        if not per_carrier_sums:
            continue
        n = len(per_carrier_sums)
        rows.append({
            "pattern": pat,
            "ig_signed_mean": sum(per_carrier_sums) / n,
            "n_present": n,
            "prevalence": n / N_sample,
        })

    if total_carrier_drops:
        pct = 100 * total_carrier_drops / max(total_carrier_drops + total_carrier_kept, 1)
        print(f"  [outlier filter] dropped {total_carrier_drops:,} per-pattern-"
              f"carrier sums with any constituent |SHAP| > {outlier_threshold} "
              f"({pct:.3f}%)")

    df = pd.DataFrame(rows)
    print(f"[3] Pattern-level: {len(df):,} patterns scored "
          f"({n_patterns_with_sampled_carriers:,} had \u22651 sampled carrier)")
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate cohort SHAP cache into Fig 3-shaped CSVs.")
    p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d")
    p.add_argument("--split", default="test")
    p.add_argument("--shap_cache", default=None,
                   help="Path to cohort SHAP cache (default: "
                        "<run_dir>/3_attribution/cohort_shap_<split>.pkl \u2014 "
                        "single growing cache).")
    p.add_argument("--jaccard_threshold", type=float, default=0.2)
    p.add_argument("--n_show", type=int, default=30,
                   help="Top-N to keep per Panel C / D CSV.")
    p.add_argument("--min_n_present", type=int, default=10,
                   help="Min sampled carriers required for a pattern to be "
                        "considered for the top-N panel CSVs (default 10). "
                        "KernelSHAP \"mean\" is undefined for patterns with "
                        "1\u20132 sampled carriers \u2014 the SHAP value of a single "
                        "patient is not a population estimate.")
    p.add_argument("--analysis_dir", default=None,
                   help="Output dir for token/pattern CSVs (default: "
                        "<run_dir>/5_analysis).")
    p.add_argument("--panels_dir", default=None,
                   help="Output dir for panel CSVs (default: "
                        "<run_dir>/7_outputs/panels).")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    analysis_dir = args.analysis_dir or layout.analysis_dir
    panels_dir = args.panels_dir or layout.outputs_panels_dir
    os.makedirs(panels_dir, exist_ok=True)

    # Resolve cache path: honor --shap_cache, fall back to the new default
    # filename, then to any legacy ``_n*.pkl`` cache. Robust against
    # stale notebook variables pointing at the old per-N filename.
    cache_path = _resolve_cache_path(layout, args.split, args.shap_cache)
    cache = _load_shap_cache(cache_path)
    sample_pids = set(cache.keys())
    N_sample = len(cache)

    # ── Token-level (Panels A + B for Fig S11) ───────────────────────
    df_tok = _aggregate_token_level(cache)
    tok_path = os.path.join(analysis_dir,
                              f"token_importance_shap_{args.split}.csv")
    df_tok.to_csv(tok_path, index=False)
    print(f"  \u2192 {tok_path}")

    # ── Pattern-level (Panels C + D for Fig S11) ─────────────────────
    # Load full-pool scored patterns + their carrier sets (built by Fig 3
    # pipeline; cached on disk).
    pool_pkl = os.path.join(analysis_dir,
                              f"full_pool_scored_{args.split}.pkl")
    if not os.path.exists(pool_pkl):
        sys.exit(f"  [ERROR] Missing {pool_pkl}\n  Run paper/jamia2026/figures/make_fig3.py "
                 f"first to materialize the full-pool scored cache.")
    with open(pool_pkl, "rb") as f:
        df_full = pickle.load(f)

    # Carrier sets: built by ``score_all_raw_patterns`` and cached in the
    # ``_carrier_keys`` column \u2014 set of (patient_id, index_hadm_id) tuples
    # (both as strings).
    if "_carrier_keys" not in df_full.columns:
        sys.exit(f"  [ERROR] full_pool_scored_<split>.pkl missing "
                 f"'_carrier_keys' column \u2014 re-run paper/jamia2026/figures/make_fig3.py "
                 f"to refresh the cache.")
    carrier_sets = dict(zip(df_full["pattern"], df_full["_carrier_keys"]))

    df_shap_full = _aggregate_pattern_level(
        cache, sample_pids, df_full, carrier_sets, N_sample)

    # ── Panel C source: full pool, j=0.2 dedup ───────────────────────
    print(f"\n[4] Panel C (full pool) Jaccard reps\u2026")
    rep_c = cluster_jaccard_inmemory(df_full, args.jaccard_threshold,
                                       verbose=False)
    print(f"  j={args.jaccard_threshold} \u2192 {len(rep_c):,} reps "
          f"(carrier sets independent of SHAP/IG choice)")
    df_shap_c_reps = df_shap_full[df_shap_full["pattern"].isin(rep_c)].copy()
    # Filter: require enough sampled carriers for a meaningful mean.
    n_c_before = len(df_shap_c_reps)
    df_shap_c_reps = df_shap_c_reps[df_shap_c_reps["n_present"] >= args.min_n_present].copy()
    print(f"  filtered to \u2265 {args.min_n_present} sampled carriers: "
          f"{len(df_shap_c_reps):,}/{n_c_before:,} reps")
    df_shap_c_reps["_abs"] = df_shap_c_reps["ig_signed_mean"].abs()
    df_shap_c_top = df_shap_c_reps.nlargest(args.n_show, "_abs").drop(columns="_abs")

    # ── Panel D source: cross-visit pool, j=0.2 dedup ────────────────
    print(f"\n[5] Panel D (cross-visit pool) Jaccard reps\u2026")
    pw_csv_path = layout.pathway_csv(args.split)
    df_pw = pd.read_csv(pw_csv_path)
    df_pw = df_pw[df_pw["n_present"] >= 3]
    rep_d = get_jaccard_representatives(analysis_dir, args.split,
                                          args.jaccard_threshold)
    if rep_d is None:
        sys.exit("  [ERROR] cross-visit Jaccard reps missing")

    df_shap_d_pool = df_shap_full[df_shap_full["pattern"].isin(set(df_pw["pattern"]))].copy()
    df_shap_d_reps = df_shap_d_pool[df_shap_d_pool["pattern"].isin(rep_d)].copy()
    # Filter: require enough sampled carriers
    n_d_before = len(df_shap_d_reps)
    df_shap_d_reps = df_shap_d_reps[df_shap_d_reps["n_present"] >= args.min_n_present].copy()
    print(f"  cross-visit + sampled-carrier patterns: {len(df_shap_d_pool):,}")
    print(f"  filtered to \u2265 {args.min_n_present} sampled carriers: "
          f"{len(df_shap_d_reps):,}/{n_d_before:,} reps")
    df_shap_d_reps["_abs"] = df_shap_d_reps["ig_signed_mean"].abs()
    df_shap_d_top = df_shap_d_reps.nlargest(args.n_show, "_abs").drop(columns="_abs")

    # ── Format labels via canonical resolver (same as Fig 3) ─────────
    resolver = PatternLabelResolver(args.run_dir, verbose=False)

    def _emit_panel(df_top: pd.DataFrame, letter: str) -> str:
        df_out = df_top.copy()
        df_out["label"] = df_out["pattern"].apply(
            lambda p: resolver.format_pattern(p, short_names=True))
        df_out = df_out.rename(columns={"ig_signed_mean": "ig_value"})
        out_cols = ["pattern", "label", "ig_value", "n_present"]
        out_path = os.path.join(
            panels_dir,
            f"panel_{letter}_shap_{args.split}_j{int(args.jaccard_threshold * 100)}_short.csv")
        df_out[out_cols].to_csv(out_path, index=False)
        return out_path

    panel_c_path = _emit_panel(df_shap_c_top, "c")
    panel_d_path = _emit_panel(df_shap_d_top, "d")
    print(f"\n[6] Panel CSVs:")
    print(f"  \u2192 {panel_c_path}  ({len(df_shap_c_top)} patterns)")
    print(f"  \u2192 {panel_d_path}  ({len(df_shap_d_top)} patterns)")

    # Preview uses the outer DataFrames \u2014 columns are still named
    # ``ig_signed_mean`` here; the rename to ``ig_value`` happens inside
    # ``_emit_panel`` on a local copy.
    print(f"\n=== Headline preview ===")
    print(f"Panel C top-5 by carrier \u03A3SHAP:")
    for _, r in df_shap_c_top.nlargest(5, "ig_signed_mean").iterrows():
        lab = resolver.format_pattern(r["pattern"], short_names=True)
        print(f"  {r['ig_signed_mean']:+.4f}  n={int(r['n_present']):>3}  {lab[:60]}")
    print(f"\nPanel D top-5 by carrier \u03A3SHAP:")
    for _, r in df_shap_d_top.nlargest(5, "ig_signed_mean").iterrows():
        lab = resolver.format_pattern(r["pattern"], short_names=True)
        print(f"  {r['ig_signed_mean']:+.4f}  n={int(r['n_present']):>3}  {lab[:60]}")


if __name__ == "__main__":
    main()
