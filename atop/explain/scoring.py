"""The single per-patient pattern matcher and scorer.

This module holds the ONE rule by which a mined pattern is (a) decided to be
present in a patient's record and (b) given a per-patient attribution score.
Both pattern pools use it:

* ``atop.explain.figures._compute_panel_c``            cross-visit pool
  (Fig 4B, Table 4, ``pathway_importance_<split>.csv``)
* ``atop.explain.cooccurrence.score_all_raw_patterns``  full pool
  (Fig 4A, Table 3, ``full_pool_scored_<split>.pkl``)

The rule
--------
1. **Per-patient occurrence index.**  Every row of the IG table
   (``ig_<split>.csv``: one row per non-special token position the model
   saw) is indexed by patient and token::

       tok_occ[token] = [(visit_idx, ig_signed), ...]    # sorted by visit_idx

   The sort is stable, so two occurrences of one token in the same visit
   keep their IG-row (= sequence position) order.  Because the whole IG
   table is used, presence is "full presence": a patient can carry a
   pattern even when its tokens were not in the salient subset used for
   mining.

2. **Greedy earliest-visit match.**  A pattern is an ordered list of
   itemsets (one per visit step).  Steps are matched left to right: step
   *i* is assigned the earliest visit, strictly later than the visit
   assigned to step *i-1* (any visit for the first step), in which **all**
   of the step's tokens occur.  If some step cannot be placed the patient
   is not a carrier.

3. **Score.**  The patient's score is the sum, over steps and over each
   step's tokens, of the signed IG of that token at its matched visit
   (first IG row of the token in that visit).  A pattern's
   ``ig_signed_mean`` is the mean of this score over its carriers and
   ``n_present`` is the number of carriers.

No torch needed: pure pandas / numpy / scipy.sparse on cached data.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PatientKey = Tuple[str, str]                 # (patient_id, index_hadm_id), both as str
TokOcc = Dict[str, List[Tuple[int, float]]]  # token -> [(visit_idx, ig_signed), ...]


def ig_value_column(df_ig: pd.DataFrame) -> str:
    """Signed IG when the table has it, else |IG| (legacy tables)."""
    return "ig_signed" if "ig_signed" in df_ig.columns else "ig_abs"


def build_patient_token_occurrences(df_ig: pd.DataFrame) -> Dict[PatientKey, TokOcc]:
    """Step 1 of the rule: ``{(pid, hadm): {token: [(visit_idx, ig), ...]}}``.

    Patients are keyed by ``(str(patient_id), str(index_hadm_id))``; the
    dict is in ``groupby`` key order (sorted), which the presence index
    relies on for a stable patient ordering.  If ``df_ig`` has no
    ``visit_idx`` column every row is treated as visit 0, i.e. the rule
    degrades to same-visit co-occurrence (callers may warn).
    """
    ig_col = ig_value_column(df_ig)
    has_visit_idx = "visit_idx" in df_ig.columns
    patient_tokens: Dict[PatientKey, TokOcc] = {}
    for (pid, hid), grp in df_ig.groupby(["patient_id", "index_hadm_id"]):
        toks = grp["token_str"].tolist()
        igs = grp[ig_col].tolist()
        vis = grp["visit_idx"].tolist() if has_visit_idx else [0] * len(toks)
        tok_occ: TokOcc = {}
        for tok, v, g in zip(toks, vis, igs):
            tok_occ.setdefault(tok, []).append((int(v), float(g)))
        for occ in tok_occ.values():
            occ.sort(key=lambda x: x[0])   # stable: same-visit ties keep row order
        patient_tokens[(str(pid), str(hid))] = tok_occ
    return patient_tokens


class PresenceIndex(NamedTuple):
    """Sparse ``patients x tokens`` 0/1 matrix for fast candidate filtering."""
    patient_keys: List[PatientKey]   # row -> patient key (order of ``patient_tokens``)
    tok_to_col: Dict[str, int]       # token -> column
    presence: object                 # scipy.sparse.csr_matrix, int8


def build_presence_index(patient_tokens: Dict[PatientKey, TokOcc]) -> PresenceIndex:
    """Build the presence matrix once per cohort (vectorized carrier pre-filter)."""
    from scipy.sparse import csr_matrix

    patient_keys = list(patient_tokens.keys())
    pat_key_to_row = {k: i for i, k in enumerate(patient_keys)}
    all_tokens = sorted({t for tok_occ in patient_tokens.values() for t in tok_occ})
    tok_to_col = {t: j for j, t in enumerate(all_tokens)}
    rows, cols = [], []
    for k, tok_occ in patient_tokens.items():
        r = pat_key_to_row[k]
        for tok in tok_occ:
            rows.append(r)
            cols.append(tok_to_col[tok])
    presence = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(len(patient_keys), len(all_tokens)),
    )
    return PresenceIndex(patient_keys, tok_to_col, presence)


def candidate_rows(index: PresenceIndex, tokens_needed: Iterable[str]) -> Optional[np.ndarray]:
    """Rows of the patients whose record contains every token in ``tokens_needed``.

    Returns ``None`` when some token never occurs in the cohort (no patient
    can carry the pattern), so callers can tell "unmatchable" from "no carrier".
    """
    tokens_needed = set(tokens_needed)
    needed_cols = [index.tok_to_col[t] for t in tokens_needed if t in index.tok_to_col]
    if len(needed_cols) != len(tokens_needed):
        return None
    sub = index.presence[:, needed_cols]                        # n_patients x n_needed
    per_patient_count = np.asarray(sub.sum(axis=1)).ravel()
    return np.where(per_patient_count == len(needed_cols))[0]


def episode_to_itemsets(episode: Sequence[Iterable[str]]) -> List[List[str]]:
    """``[frozenset, ...]`` (one per visit step) -> ``[[token, ...], ...]``.

    Both pools go through this so that a pattern's tokens are visited (and
    their IG summed) in the same order whichever pool scores it.
    """
    return [list(s) for s in episode]


def match_and_score_episode(pat_itemsets: Sequence[Sequence[str]],
                            tok_occ: TokOcc) -> Tuple[bool, float]:
    """Steps 2-3 of the rule for one patient.

    Returns ``(is_carrier, score)``; ``score`` is 0.0 when not a carrier.
    """
    matched_positions = []   # (token, visit_idx, ig_val)
    min_visit = -1           # next step must sit at a visit strictly later than this
    for itemset in pat_itemsets:
        # Visits (later than the previous step) in which each token occurs.
        token_visit_sets = []
        for tok in itemset:
            token_visit_sets.append({v for v, _ in tok_occ.get(tok, ()) if v > min_visit})
        if not token_visit_sets:
            return False, 0.0
        # Visits in which ALL of the step's tokens co-occur.
        common_visits = token_visit_sets[0]
        for vs in token_visit_sets[1:]:
            common_visits &= vs
            if not common_visits:
                break
        if not common_visits:
            return False, 0.0
        chosen_visit = min(common_visits)     # earliest valid visit
        min_visit = chosen_visit
        # Signed IG of each step token at that visit (first occurrence there).
        for tok in itemset:
            for v, ig_val in tok_occ.get(tok, ()):
                if v == chosen_visit:
                    matched_positions.append((tok, v, ig_val))
                    break
    if not matched_positions:
        return False, 0.0
    return True, sum(ig_val for _, _, ig_val in matched_positions)


def score_episode_over_cohort(pat_itemsets: Sequence[Sequence[str]],
                              patient_tokens: Dict[PatientKey, TokOcc],
                              index: PresenceIndex,
                              ) -> Optional[Tuple[List[float], List[PatientKey]]]:
    """Apply the rule to every patient of the cohort.

    Returns ``(scores, carrier_keys)`` aligned and in patient-row order, or
    ``None`` when a pattern token never occurs in the cohort.
    """
    tokens_needed: set = set()
    for itemset in pat_itemsets:
        tokens_needed.update(itemset)
    cand = candidate_rows(index, tokens_needed)
    if cand is None:
        return None
    scores: List[float] = []
    carriers: List[PatientKey] = []
    for r in cand:
        key = index.patient_keys[r]
        is_carrier, score = match_and_score_episode(pat_itemsets, patient_tokens[key])
        if is_carrier:
            scores.append(score)
            carriers.append(key)
    return scores, carriers
