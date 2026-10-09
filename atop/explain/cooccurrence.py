"""Full-pool pattern scoring over the raw mining cache (all shapes, min_steps=1).

The main paper locks mining at ``episode_min_steps=2`` (cross-visit only).
This module scores EVERY distinct pattern in the raw mining cache (single
tokens, same-visit co-occurrences and cross-visit episodes alike) with the
single per-patient rule in :mod:`atop.explain.scoring`, i.e. exactly the rule
``figures._compute_panel_c`` applies to the cross-visit pool.  It feeds
``scripts/make_fig3.py`` (full pool, Jaccard representatives, Table 3) and
``scripts/build_shap_aggregations.py``.

No torch needed — pure pandas / numpy / scipy.sparse operations on cached data.
"""
from __future__ import annotations

import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd

from atop.explain.scoring import (
    build_patient_token_occurrences, build_presence_index,
    episode_to_itemsets, score_episode_over_cohort,
)


def _format_episode(ep) -> str:
    """``[frozenset, ...]`` -> ``"A -> {B, C}"`` (the pool's pattern-string format)."""
    return " -> ".join(
        ("{" + ", ".join(sorted(b)) + "}") if len(b) > 1 else next(iter(b))
        for b in ep
    )


def score_single_block_cooccurrence(raw_mining: dict,
                                     df_seq: pd.DataFrame,
                                     df_ig: pd.DataFrame,
                                     min_carriers: int = 5,
                                     verbose: bool = True) -> pd.DataFrame:
    """Score every single-visit co-occurrence pattern (one itemset, ≥2 tokens).

    Thin wrapper around :func:`score_all_raw_patterns` restricted to one-block
    patterns, so same-visit co-occurrences are matched and scored by the same
    rule as every other shape (see :mod:`atop.explain.scoring`).

    Returns columns: ``pattern, ig_signed_mean, n_present, _carrier_keys``
    matching the schema used by ``pathway_importance_<split>.csv``.

    ``raw_mining`` is the dict in ``raw_episode_*.pkl`` with keys
    ``freq_case`` and ``freq_ctrl`` (each a list of
    ``(support, episode, carrier_indices)``).
    """
    single = {
        cohort: [(sup, ep, carriers) for sup, ep, carriers in raw_mining[cohort]
                 if len(ep) == 1 and len(ep[0]) >= 2]
        for cohort in ("freq_case", "freq_ctrl")
    }
    if verbose:
        n_distinct = len({frozenset(ep[0]) for c in single.values() for _, ep, _ in c})
        print(f"  [combined] {n_distinct:,} distinct single-block co-occurrence patterns")
    return score_all_raw_patterns(single, df_seq, df_ig, min_carriers=min_carriers,
                                  min_blocks=1, min_tokens_total=2, verbose=verbose)


def load_combined_pool(run_dir: str, split: str = "test",
                        min_carriers: int = 5,
                        verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Combine multi-block (cross-visit) + single-block co-occurrence pools.

    Returns:
        df_combined : DataFrame with columns
            ``pattern, ig_signed_mean, n_present, _carrier_keys``
            covering both single-block and multi-block patterns.
        carrier_keys_dict : dict ``pattern → set`` for use with the Jaccard
            sweep (the existing ``get_jaccard_representatives`` reads this
            from a per-cohort pickle; we expose the in-memory dict here so
            the supplements can pass it through).
    """
    from atop.run_layout import RunLayout
    L = RunLayout(run_dir)

    # 1. Multi-block pool from the existing pathway CSV + carriers pickle.
    pw_csv = L.pathway_csv(split)
    pkl_path = os.path.join(L.analysis_dir, f"carrier_sets_{split}.pkl")
    if not (os.path.exists(pw_csv) and os.path.exists(pkl_path)):
        raise FileNotFoundError(
            f"Need {pw_csv} and {pkl_path} — run make_fig3.py first.")
    df_multi = pd.read_csv(pw_csv)
    df_multi = df_multi[df_multi["n_present"] >= min_carriers].copy()
    with open(pkl_path, "rb") as f:
        multi_carriers = pickle.load(f)
    # Attach carrier keys to the multi-block DataFrame.
    df_multi["_carrier_keys"] = df_multi["pattern"].map(multi_carriers)
    df_multi = df_multi[df_multi["_carrier_keys"].notna()].copy()
    df_multi = df_multi[["pattern", "ig_signed_mean", "n_present", "_carrier_keys"]]
    if verbose:
        print(f"  [combined] multi-block pool: {len(df_multi):,} patterns")

    # 2. Single-block pool from the raw mining cache.
    raw_files = sorted([f for f in os.listdir(L.mining_dir)
                        if f.startswith("raw_episode_")])
    if not raw_files:
        raise FileNotFoundError(f"No raw_episode_*.pkl in {L.mining_dir}")
    with open(os.path.join(L.mining_dir, raw_files[-1]), "rb") as f:
        raw_mining = pickle.load(f)
    with open(L.sequences_pkl(split), "rb") as f:
        df_seq = pickle.load(f)
    df_ig = pd.read_csv(L.ig_csv(split))

    df_single = score_single_block_cooccurrence(
        raw_mining, df_seq, df_ig, min_carriers=min_carriers, verbose=verbose)

    # 3. Concat (single-block patterns added to multi-block pool).
    df_combined = pd.concat([df_multi, df_single], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset="pattern", keep="first")
    if verbose:
        n_single = (~df_combined["pattern"].str.contains(" -> ", regex=False)).sum()
        n_multi = len(df_combined) - n_single
        print(f"  [combined] final pool: {len(df_combined):,} total "
              f"({n_single:,} single-block + {n_multi:,} multi-block)")

    # Build the unified carrier_keys dict.
    carrier_keys = {row["pattern"]: row["_carrier_keys"]
                     for _, row in df_combined.iterrows()
                     if isinstance(row["_carrier_keys"], set)}
    return df_combined, carrier_keys


def score_all_raw_patterns(raw_mining: dict,
                            df_seq: pd.DataFrame,
                            df_ig: pd.DataFrame,
                            min_carriers: int = 5,
                            min_blocks: int = 1,
                            min_tokens_total: int = 1,
                            verbose: bool = True) -> pd.DataFrame:
    """Score EVERY distinct pattern in the raw mining cache with the single rule.

    Handles every shape uniformly:
      * single-token single-block    ``[{A}]``                 (≈ Panel B token)
      * single-token multi-block     ``[{A}, {A}, …]``         (temporal repetition)
      * multi-token single-block     ``[{A, B, …}]``           (co-occurrence)
      * multi-token multi-block      ``[{A, …}, {B, …}, …]`` (Panel C shape)

    Carriers and scores come from :mod:`atop.explain.scoring`, the same code
    ``figures._compute_panel_c`` uses: a patient carries a pattern when the
    greedy earliest-visit match over their full IG record (every token
    position the model saw) succeeds, and the patient's score is the sum of
    the signed IG of each step token at its matched visit.  ``ig_signed_mean``
    is the mean over carriers, ``n_present`` the number of carriers and
    ``_carrier_keys`` the set of ``(patient_id, index_hadm_id)`` string tuples.

    Args
    ----
    raw_mining:         dict with ``freq_case`` / ``freq_ctrl`` lists of
                        ``(support, episode, carrier_indices)``
    df_seq:             accepted for API compatibility only; presence is
                        read from ``df_ig``
    df_ig:              ``ig_<split>.csv`` (needs ``visit_idx`` for temporal order)
    min_blocks:         require the pattern to have at least this many visit blocks
    min_tokens_total:   require sum-of-token-occurrences ≥ this (``[{A,B}]`` = 2)
    min_carriers:       drop patterns with fewer than this many carriers
    """
    # 1. Enumerate unique patterns from both cohorts.
    episodes_keyed: dict = {}  # key (tuple of frozensets) → {"ep", "n_case", "n_ctrl"}
    for cohort, count_key in (("freq_case", "n_case"), ("freq_ctrl", "n_ctrl")):
        for sup, ep, _carriers in raw_mining[cohort]:
            if len(ep) < min_blocks:
                continue
            n_tokens = sum(len(b) for b in ep)
            if n_tokens < min_tokens_total:
                continue
            key = tuple(frozenset(b) for b in ep)
            entry = episodes_keyed.setdefault(key, {"ep": [frozenset(b) for b in ep],
                                                    "n_case": 0, "n_ctrl": 0})
            entry[count_key] = sup
    if verbose:
        print(f"  [raw-scorer] {len(episodes_keyed):,} distinct patterns in raw cache "
              f"(min_blocks={min_blocks}, min_tokens_total={min_tokens_total})")

    # 2. Per-patient occurrence index + presence matrix (shared with Panel C).
    patient_tokens = build_patient_token_occurrences(df_ig)
    index = build_presence_index(patient_tokens)
    if verbose:
        print(f"  [raw-scorer] {len(patient_tokens):,} patients with IG rows")
        print(f"  [raw-scorer] sparse presence: {index.presence.shape}, "
              f"{index.presence.nnz:,} nnz")

    # 3. Score each pattern.
    scored = []
    total = len(episodes_keyed)
    progress_step = max(1000, total // 10)
    for pi, (key, entry) in enumerate(episodes_keyed.items()):
        ep = entry["ep"]
        pat_itemsets = episode_to_itemsets(ep)
        if not any(pat_itemsets):
            continue  # pattern without tokens
        result = score_episode_over_cohort(pat_itemsets, patient_tokens, index)
        if result is None:
            continue  # some token absent from this split entirely
        ig_sums, carriers = result
        if len(carriers) < min_carriers:
            continue
        mean_sig = float(np.mean(ig_sums)) if ig_sums else 0.0
        scored.append({
            "pattern":         _format_episode(ep),
            "ig_signed_mean":  mean_sig,
            "n_present":       len(carriers),
            "_carrier_keys":   set(carriers),
        })
        if verbose and (pi + 1) % progress_step == 0:
            print(f"    [raw-scorer] {pi+1}/{total} patterns scored, "
                  f"{len(scored)} with ≥{min_carriers} carriers")

    df = pd.DataFrame(scored)
    if verbose:
        print(f"  [raw-scorer] final: {len(df):,} patterns ≥{min_carriers} carriers")
    return df


def cluster_jaccard_inmemory(df_pool: pd.DataFrame, jaccard_threshold: float,
                              verbose: bool = True) -> set:
    """Run the vectorized Jaccard sweep on an in-memory carrier-set dict.

    Wrapper around the same algorithm in ``atop.explain.jaccard`` but
    operating on the combined pool's in-memory ``_carrier_keys`` column,
    not on a disk pickle. Returns the set of representative pattern strings.
    """
    from atop.explain.jaccard import _greedy_jaccard_clusters
    df = df_pool[df_pool["_carrier_keys"].apply(
        lambda c: isinstance(c, set) and len(c) >= 3)].copy()
    df = df.rename(columns={"_carrier_keys": "_ck"})
    if df.empty:
        return set()
    if verbose:
        print(f"    [jaccard] sweep input: {len(df):,} patterns at j={jaccard_threshold}")
    return _greedy_jaccard_clusters(df, jaccard_threshold)
