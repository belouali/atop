"""Combined min_steps=1 pattern pool: single-block co-occurrence + multi-block.

The main paper locks mining at ``episode_min_steps=2`` (cross-visit only).
This module recovers single-block co-occurrence patterns from the raw
mining cache, scores them with the same carrier-averaged \u03a3IG aggregation
as Panel C, and merges them with the existing multi-block pathway pool to
produce the "what Panel C would look like at min_steps=1" view used by
``scripts/make_fig_s8_*.py`` and ``scripts/make_fig_s9_*.py``.

No torch needed \u2014 pure pandas / numpy operations on cached data.
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd


_SPECIAL_TOKENS = frozenset({"[CLS]", "[VISIT]", "[PAD]"})


def _full_visit_blocks(combined_full) -> list:
    """Reconstruct full per-visit token blocks (list of frozensets) from the
    complete token sequence. A new block begins at each ``[VISIT]`` marker;
    ``[CLS]``/``[VISIT]``/``[PAD]`` are dropped. This scores patterns over ALL
    patients who exhibit a pattern (full-presence carriers), not only those
    where the pattern's tokens fell in the salient (80% IG-mass) subset.
    """
    blocks = []
    cur = None
    for t in combined_full or []:
        if t == "[VISIT]":
            cur = []
            blocks.append(cur)
        elif t in _SPECIAL_TOKENS:
            continue
        elif cur is not None:
            cur.append(t)
    return [frozenset(b) for b in blocks if b]


def _build_ig_per_patient_token(df_ig: pd.DataFrame) -> dict:
    """patient key (pid, hadm) \u2192 dict{token \u2192 sum_of_signed_IG}."""
    cache: dict = {}
    ig_col = "ig_signed" if "ig_signed" in df_ig.columns else "ig_abs"
    ig_sum = (df_ig.groupby(["patient_id", "index_hadm_id", "token_str"])[ig_col]
                    .sum())
    for (pid, hid, tok), val in ig_sum.items():
        key = (str(pid), str(hid))
        cache.setdefault(key, {})[tok] = float(val)
    return cache


def _build_token_to_patient_visits(df_seq: pd.DataFrame) -> dict:
    """token \u2192 dict{patient_key: set(visit_indices_with_token)}."""
    out: dict = defaultdict(lambda: defaultdict(set))
    for _, row in df_seq.iterrows():
        key = (str(row["patient_id"]), str(row["index_hadm_id"]))
        svb = row.get("salient_visit_blocks", [])
        if not svb:
            continue
        for vi, block in enumerate(svb):
            for t in block:
                out[t][key].add(vi)
    return out


def score_single_block_cooccurrence(raw_mining: dict,
                                     df_seq: pd.DataFrame,
                                     df_ig: pd.DataFrame,
                                     min_carriers: int = 5,
                                     verbose: bool = True) -> pd.DataFrame:
    """Score every single-block co-occurrence pattern from the raw cache.

    Returns columns: ``pattern, ig_signed_mean, n_present, _carrier_keys``
    matching the schema used by ``pathway_importance_<split>.csv``.

    ``raw_mining`` is the dict in ``raw_episode_*.pkl`` with keys
    ``freq_case`` and ``freq_ctrl`` (each a list of
    ``(support, episode, carrier_indices)``).
    """
    # 1. Collect distinct co-occurrence sets across both cohorts.
    raw_combined: dict = {}
    for cohort in ("freq_case", "freq_ctrl"):
        for sup, ep, _carriers in raw_mining[cohort]:
            if len(ep) != 1:
                continue
            block = ep[0]
            if len(block) < 2:
                continue
            raw_combined[frozenset(block)] = True
    if verbose:
        print(f"  [combined] {len(raw_combined):,} distinct single-block co-occurrence patterns")

    # 2. Build per-patient token index (only need this once per call).
    tok2pv = _build_token_to_patient_visits(df_seq)
    if verbose:
        print(f"  [combined] reverse index: {len(tok2pv):,} tokens \u2192 patient-visits")

    # 3. Per-token IG aggregation per patient.
    ig_cache = _build_ig_per_patient_token(df_ig)
    if verbose:
        print(f"  [combined] IG cache: {len(ig_cache):,} patients")

    # 4. Score each co-occurrence pattern.
    rows = []
    for tok_set in raw_combined:
        tokens = sorted(tok_set)
        # Patients having ALL tokens (somewhere)
        per_token_patients = [set(tok2pv[t].keys()) for t in tokens]
        if not all(per_token_patients):
            continue
        cand = set.intersection(*per_token_patients)
        # Restrict to patients sharing at least one common visit across all tokens
        carriers = set()
        for pkey in cand:
            common_visits = set.intersection(*[tok2pv[t][pkey] for t in tokens])
            if common_visits:
                carriers.add(pkey)
        if len(carriers) < min_carriers:
            continue
        per_carrier = []
        for pkey in carriers:
            tig = ig_cache.get(pkey, {})
            per_carrier.append(sum(tig.get(t, 0.0) for t in tokens))
        if not per_carrier:
            continue
        rows.append({
            "pattern":         "{" + ", ".join(tokens) + "}",
            "ig_signed_mean":  float(np.mean(per_carrier)),
            "n_present":       len(carriers),
            "_carrier_keys":   carriers,
        })
    df = pd.DataFrame(rows)
    if verbose:
        print(f"  [combined] scored {len(df):,} single-block patterns "
              f"(\u2265{min_carriers} carriers)")
    return df


def load_combined_pool(run_dir: str, split: str = "test",
                        min_carriers: int = 5,
                        verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Combine multi-block (cross-visit) + single-block co-occurrence pools.

    Returns:
        df_combined : DataFrame with columns
            ``pattern, ig_signed_mean, n_present, _carrier_keys``
            covering both single-block and multi-block patterns.
        carrier_keys_dict : dict ``pattern \u2192 set`` for use with the Jaccard
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
            f"Need {pw_csv} and {pkl_path} \u2014 run make_fig3.py first.")
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
    """Score EVERY distinct pattern in the raw mining cache.

    Handles every shape uniformly:
      * single-token single-block    ``[{A}]``                 (\u2248 Panel B token)
      * single-token multi-block     ``[{A}, {A}, \u2026]``         (temporal repetition)
      * multi-token single-block     ``[{A, B, \u2026}]``           (co-occurrence, S8)
      * multi-token multi-block      ``[{A, \u2026}, {B, \u2026}, \u2026]`` (Panel C shape)

    Uses the vectorized sparse-presence carrier lookup (same approach as
    ``_compute_panel_c`` in figures.py) followed by a per-carrier greedy
    episode-order match. Returns a DataFrame with the Panel-C schema:
    ``pattern, ig_signed_mean, n_present, _carrier_keys``.

    Args
    ----
    min_blocks:         require the pattern to have at least this many visit blocks
    min_tokens_total:   require sum-of-token-occurrences \u2265 this (``[{A,B}]`` = 2)
    min_carriers:       drop patterns with fewer than this many test-split carriers
    """
    from scipy.sparse import csr_matrix
    from atop.mining.patterns import _is_subpattern

    # 1. Enumerate unique patterns from both cohorts.
    seen: set = set()
    episodes_keyed: dict = {}  # key (tuple of frozensets) \u2192 (episode_list, n_case, n_ctrl)
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
            seen.add(key)
    if verbose:
        print(f"  [raw-scorer] {len(episodes_keyed):,} distinct patterns in raw cache "
              f"(min_blocks={min_blocks}, min_tokens_total={min_tokens_total})")

    # 2. Build per-patient IG cache + per-patient visit-blocks list.
    ig_cache = _build_ig_per_patient_token(df_ig)
    patient_blocks: dict = {}  # key \u2192 list[frozenset(tokens)]
    all_patients = []
    for _, row in df_seq.iterrows():
        key = (str(row["patient_id"]), str(row["index_hadm_id"]))
        # Full-presence carriers: build blocks from the complete sequence, not
        # only the salient (80% IG-mass) tokens.
        svb = _full_visit_blocks(row.get("combined_full", []))
        if not svb:
            patient_blocks[key] = []
        else:
            patient_blocks[key] = [frozenset(b) for b in svb]
        all_patients.append(key)
    if verbose:
        print(f"  [raw-scorer] {len(all_patients):,} patients, {len(ig_cache):,} with IG rows")

    # 3. Build sparse presence matrix (patients \u00d7 tokens) for fast carrier filtering.
    patient_idx = {k: i for i, k in enumerate(all_patients)}
    tok_idx: dict = {}
    rows, cols = [], []
    for k, blocks in patient_blocks.items():
        i = patient_idx[k]
        # Binary presence: each token counts once per patient regardless of how many
        # visit blocks it appears in. Appending per (block, token) made duplicate
        # (patient, token) entries sum into occurrence counts, so the candidate
        # filter (per_row_count == n_pattern_tokens) wrongly excluded patients with
        # a pattern token repeated across visits — dropping true carriers.
        for t in (set().union(*blocks) if blocks else ()):
            j = tok_idx.get(t)
            if j is None:
                j = len(tok_idx)
                tok_idx[t] = j
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.int8)
    presence = csr_matrix((data, (rows, cols)),
                          shape=(len(all_patients), len(tok_idx)), dtype=np.int32)
    if verbose:
        print(f"  [raw-scorer] sparse presence: {presence.shape}, {presence.nnz:,} nnz")

    # 4. Score each pattern.
    scored = []
    total = len(episodes_keyed)
    progress_step = max(1000, total // 10)
    for pi, (key, entry) in enumerate(episodes_keyed.items()):
        ep = entry["ep"]
        pat_tokens = set().union(*ep) if ep else set()
        # Carriers must have ALL pattern tokens in their presence
        needed_cols = [tok_idx[t] for t in pat_tokens if t in tok_idx]
        if len(needed_cols) != len(pat_tokens):
            continue  # some token absent from test split entirely
        if not needed_cols:
            continue
        sub = presence[:, needed_cols]
        per_row_count = np.asarray(sub.sum(axis=1)).ravel()
        cand_rows = np.where(per_row_count == len(needed_cols))[0]
        if cand_rows.size == 0:
            continue
        # Temporal match within candidates
        carriers = set()
        ig_sums = []
        for ri in cand_rows:
            pkey = all_patients[ri]
            blocks = patient_blocks[pkey]
            if not blocks:
                continue
            if _is_subpattern(ep, blocks):
                carriers.add(pkey)
                tig = ig_cache.get(pkey, {})
                ig_sums.append(sum(tig.get(t, 0.0) for t in pat_tokens))
        if len(carriers) < min_carriers:
            continue
        mean_sig = float(np.mean(ig_sums)) if ig_sums else 0.0
        pat_str = " -> ".join(
            ("{" + ", ".join(sorted(b)) + "}") if len(b) > 1 else next(iter(b))
            for b in ep
        )
        scored.append({
            "pattern":         pat_str,
            "ig_signed_mean":  mean_sig,
            "n_present":       len(carriers),
            "_carrier_keys":   carriers,
        })
        if verbose and (pi + 1) % progress_step == 0:
            print(f"    [raw-scorer] {pi+1}/{total} patterns scored, "
                  f"{len(scored)} with \u2265{min_carriers} carriers")

    df = pd.DataFrame(scored)
    if verbose:
        print(f"  [raw-scorer] final: {len(df):,} patterns \u2265{min_carriers} carriers")
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
