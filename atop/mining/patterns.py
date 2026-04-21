"""Pattern mining: episode, n-gram, PrefixSpan + scoring."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def _is_subsequence(pattern: tuple, sequence: list) -> bool:
    """Check if flat pattern is a non-contiguous subsequence of flat sequence (two-pointer)."""
    if not pattern:
        return True
    j = 0
    for tok in sequence:
        if tok == pattern[j]:
            j += 1
            if j == len(pattern):
                return True
    return False


def _is_subpattern(pattern: list, sequence: list) -> bool:
    """
    Check if an itemset-sequence pattern is a sub-pattern of a visit-block sequence.
    pattern: List[frozenset] — each frozenset is an itemset that must match a visit
    sequence: List[frozenset] — the patient's visit blocks
    Each itemset in pattern must be a subset of some itemset in sequence, in order.
    """
    seq_pos = 0
    for pset in pattern:
        found = False
        while seq_pos < len(sequence):
            if pset.issubset(sequence[seq_pos]):
                seq_pos += 1
                found = True
                break
            seq_pos += 1
        if not found:
            return False
    return True


def _match_subpattern(pattern: list, sequence: list):
    """
    Like _is_subpattern but returns the matched block indices.

    Uses greedy earliest-match alignment: scans left-to-right and assigns each
    pattern itemset to the first valid sequence block. When multiple valid
    instantiations exist (e.g., A appears in blocks 0 and 2; B in blocks 3 and 5),
    the earliest alignment is selected (blocks 0→3). This is a deterministic rule
    ensuring reproducibility; alternative rules (e.g., best-IG instantiation)
    may affect individual patient scores but do not materially change global
    pattern rankings.

    Args:
        pattern:  List[frozenset] — ordered itemsets to match
        sequence: List[frozenset] — patient's visit blocks

    Returns:
        List of (pattern_itemset_idx, sequence_block_idx) tuples, or None if no match.
    """
    seq_pos = 0
    matched = []
    for pi, pset in enumerate(pattern):
        found = False
        while seq_pos < len(sequence):
            if pset.issubset(sequence[seq_pos]):
                matched.append((pi, seq_pos))
                seq_pos += 1
                found = True
                break
            seq_pos += 1
        if not found:
            return None
    return matched


# --- Episode mining (itemset-sequence) ---


def mine_episodes(db: List[List[frozenset]], min_support: int,
                  max_len: int = 4, topn: int = 500,
                  min_pattern_items: int = 2) -> List[Tuple[int, list, set]]:
    """
    Mine frequent itemset-sequence patterns from visit-block database.

    db: List[List[frozenset]] — each entry is a patient's visit sequence
    min_support: minimum number of sequences containing the pattern
    max_len: max total items across all itemsets in the pattern
    topn: max patterns to return (by support), 0=no limit
    min_pattern_items: minimum total items across all itemsets

    Returns: List of (support, pattern, patient_indices) tuples.
    """
    from collections import defaultdict
    from atop.config import SPECIAL_TOKENS

    # Safety: strip any special tokens that may have leaked into visit blocks
    clean_db = []
    for seq in db:
        clean_seq = []
        for itemset in seq:
            cleaned = frozenset(t for t in itemset if t not in SPECIAL_TOKENS)
            if cleaned:
                clean_seq.append(cleaned)
        clean_db.append(clean_seq)
    db = clean_db

    item_to_seqs = defaultdict(set)
    for si, seq in enumerate(db):
        for itemset in seq:
            for item in itemset:
                item_to_seqs[item].add(si)

    freq_items = sorted([item for item, seqs in item_to_seqs.items()
                         if len(seqs) >= min_support])

    all_frequent = []

    # Level 1: frequent single-item patterns [{item}]
    current_level = []
    for item in freq_items:
        pat = [frozenset([item])]
        sup = len(item_to_seqs[item])
        if sup >= min_support:
            if 1 >= min_pattern_items:
                all_frequent.append((sup, pat, item_to_seqs[item].copy()))
            current_level.append((pat, item_to_seqs[item].copy()))

    print(f"  [episode] Level 1: {len(current_level)} frequent items")

    level = 1
    while current_level and level < max_len:
        level += 1
        candidates = {}

        for pat, pat_seqs in current_level:
            for item in freq_items:
                item_seqs = item_to_seqs[item]
                overlap = pat_seqs & item_seqs
                if len(overlap) < min_support:
                    continue

                # Extension A: add item to last itemset (same-visit co-occurrence)
                if item not in pat[-1]:
                    new_last = pat[-1] | frozenset([item])
                    new_pat = pat[:-1] + [new_last]
                    key = tuple(tuple(sorted(s)) for s in new_pat)
                    if key not in candidates:
                        candidates[key] = (new_pat, overlap.copy())

                # Extension B: append as new singleton itemset (cross-visit)
                new_pat = pat + [frozenset([item])]
                total_items = sum(len(s) for s in new_pat)
                if total_items <= max_len:
                    key = tuple(tuple(sorted(s)) for s in new_pat)
                    if key not in candidates:
                        candidates[key] = (new_pat, overlap.copy())

        next_level = []
        for key, (cand_pat, cand_seqs) in candidates.items():
            total_items = sum(len(s) for s in cand_pat)
            if total_items > max_len:
                continue
            sup = sum(1 for si in cand_seqs if _is_subpattern(cand_pat, db[si]))
            if sup >= min_support:
                exact_seqs = {si for si in cand_seqs if _is_subpattern(cand_pat, db[si])}
                if total_items >= min_pattern_items:
                    all_frequent.append((sup, cand_pat, exact_seqs))
                next_level.append((cand_pat, exact_seqs))

        print(f"  [episode] Level {level}: {len(next_level)} frequent from {len(candidates)} candidates")
        current_level = next_level

    all_frequent.sort(key=lambda x: -x[0])
    if topn > 0 and len(all_frequent) > topn:
        all_frequent = all_frequent[:topn]

    return all_frequent



def _format_episode_pattern(pat: list) -> str:
    """Format an itemset-sequence pattern for display."""
    parts = []
    for itemset in pat:
        items_sorted = sorted(itemset)
        if len(items_sorted) == 1:
            parts.append(items_sorted[0])
        else:
            parts.append("{" + ", ".join(items_sorted) + "}")
    return " -> ".join(parts)



def _parse_episode_pattern(pat_str: str) -> list:
    """Parse a formatted episode pattern string back to list of frozensets."""
    parts = pat_str.split(" -> ")
    result = []
    for p in parts:
        p = p.strip()
        if p.startswith("{") and p.endswith("}"):
            items = [x.strip() for x in p[1:-1].split(",")]
            result.append(frozenset(items))
        else:
            result.append(frozenset([p]))
    return result


# --- N-gram mining (flat, contiguous) ---


def _admission_pattern_sets_ngram(seqs: List[List[str]], nmin: int, nmax: int) -> List[set]:
    result = []
    for seq in seqs:
        L = len(seq)
        seen = set()
        for n in range(nmin, nmax + 1):
            if L < n:
                continue
            for i in range(L - n + 1):
                seen.add(tuple(seq[i:i + n]))
        result.append(seen)
    return result


# --- PrefixSpan mining (flat, non-contiguous) ---


def _admission_pattern_sets_prefixspan(seqs: List[List[str]], min_support_abs: int,
                                       min_len: int, topn: int = 0):
    from prefixspan import PrefixSpan
    ps = PrefixSpan(seqs)
    raw = ps.frequent(min_support_abs)
    patterns = [(sup, tuple(pat)) for sup, pat in raw if len(pat) >= min_len]
    if topn > 0 and len(patterns) > topn:
        patterns.sort(key=lambda x: x[0], reverse=True)
        patterns = patterns[:topn]
    pat_set = {pat for _, pat in patterns}
    result = []
    for seq in seqs:
        present = set()
        for pat in pat_set:
            if _is_subsequence(pat, seq):
                present.add(pat)
        result.append(present)
    return result, patterns


# --- Scoring ---


def _score_patterns_admission_level(
    admission_sets: List[set], y: np.ndarray, all_patterns: set, min_support_abs: int
) -> pd.DataFrame:
    """Score patterns using vectorized sparse matrix operations.
    
    Instead of iterating patterns and checking set membership per patient,
    builds a sparse (patients × patterns) presence matrix upfront, then
    computes all 2×2 tables in one vectorized pass.
    """
    from scipy.stats import chi2_contingency
    from scipy import sparse

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    n_patients = len(admission_sets)
    y_bool = (y == 1)

    # Map patterns to indices
    pat_list = list(all_patterns)
    pat_to_idx = {p: i for i, p in enumerate(pat_list)}
    n_pats = len(pat_list)

    # Build sparse presence matrix: rows=patients, cols=patterns
    print(f"  [scoring] Building sparse presence matrix ({n_patients} patients × {n_pats} patterns)...")
    row_idx = []
    col_idx = []
    for pi, pset in enumerate(admission_sets):
        for pat in pset:
            if pat in pat_to_idx:
                row_idx.append(pi)
                col_idx.append(pat_to_idx[pat])

    presence = sparse.csr_matrix(
        (np.ones(len(row_idx), dtype=np.bool_), (row_idx, col_idx)),
        shape=(n_patients, n_pats))

    # Vectorized counts
    support = np.array(presence.sum(axis=0)).ravel()  # n_present per pattern
    # Cases present: sum of presence where y==1
    cases_present = np.array(presence[y_bool].sum(axis=0)).ravel()    # a
    ctrls_present = support - cases_present                            # c
    cases_absent = n_pos - cases_present                               # b
    ctrls_absent = n_neg - ctrls_present                               # d

    print(f"  [scoring] Computing OR, chi2, p-values for {n_pats} patterns...")
    rows = []
    n_scored = 0
    report_interval = max(1, n_pats // 10)
    for i, pat in enumerate(pat_list):
        if i > 0 and i % report_interval == 0:
            print(f"    {i}/{n_pats} scored...")

        n_present = int(support[i])
        if n_present < min_support_abs:
            continue

        a = int(cases_present[i])
        b = int(cases_absent[i])
        c = int(ctrls_present[i])
        d = int(ctrls_absent[i])
        or_est = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        log_or = float(np.log(or_est)) if or_est > 0 else 0.0

        try:
            table = np.array([[a, b], [c, d]])
            chi2, pval, _, _ = chi2_contingency(table, correction=True)
        except Exception:
            chi2, pval = 0.0, 1.0

        if isinstance(pat, tuple) and len(pat) > 0 and isinstance(pat[0], str):
            pat_str = " -> ".join(pat)
            pat_len = len(pat)
            pat_steps = len(pat)
        else:
            pat_as_list = [frozenset(s) for s in pat]
            pat_str = _format_episode_pattern(pat_as_list)
            pat_len = sum(len(s) for s in pat)
            pat_steps = len(pat_as_list)

        prev_case = a / max(n_pos, 1)
        prev_ctrl = c / max(n_neg, 1)

        rows.append({
            "pattern": pat_str,
            "pattern_key": pat,
            "n_tokens": pat_len,
            "n_steps": pat_steps,
            "n_admissions_present": n_present,
            "n_present_readmit1": a,
            "n_present_readmit0": c,
            "n_total_readmit1": n_pos,
            "n_total_readmit0": n_neg,
            "odds_ratio": float(or_est),
            "log_or": log_or,
            "chi2": float(chi2),
            "pval": float(pval),
            "prev_case": float(prev_case),
            "prev_ctrl": float(prev_ctrl),
            "prev_diff": float(prev_case - prev_ctrl),
        })
        n_scored += 1

    print(f"  [scoring] {n_scored} patterns scored (from {n_pats} total)")
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Bonferroni correction
    n_tests = len(df)
    df["pval_bonferroni"] = (df["pval"] * n_tests).clip(upper=1.0)
    df["significant_bonf"] = df["pval_bonferroni"] < 0.05

    return df.sort_values(["odds_ratio", "n_admissions_present"], ascending=[False, False]).reset_index(drop=True)


def _cap_by_discriminative_per_length(df: pd.DataFrame, topn_per_length: int,
                                       cap_metric: str = "or") -> pd.DataFrame:
    """Keep the most discriminative patterns at each step count.

    At each n_steps (number of visit blocks), keeps topn_per_length/2 most
    risk-associated and topn_per_length/2 most protective patterns.

    Args:
        df: scored patterns with 'odds_ratio', 'prev_diff', and 'n_steps' columns
        topn_per_length: max patterns to keep per step count (split equally
                         between risk and protective)
        cap_metric: scoring metric for retention:
            - "or": |log OR| (default, amplifies rare extreme patterns)
            - "prev_diff": |prevalence_cases - prevalence_controls|
              (favors patterns affecting the most patients)

    Returns:
        Filtered DataFrame, re-sorted by metric descending.
    """
    if df.empty or topn_per_length <= 0:
        return df

    half = max(1, topn_per_length // 2)
    kept = []

    if cap_metric == "prev_diff":
        # Rank by prevalence difference: positive = risk, negative = protective
        for n_steps, grp in df.groupby("n_steps"):
            risk = grp[grp["prev_diff"] > 0].nlargest(half, "prev_diff")
            prot = grp[grp["prev_diff"] <= 0].nsmallest(half, "prev_diff")
            kept.append(pd.concat([risk, prot]).drop_duplicates())
        sort_cols = ["prev_diff", "n_admissions_present"]
        metric_label = "|prev_diff|"
    else:
        # Default: rank by OR
        for n_steps, grp in df.groupby("n_steps"):
            risk = grp.nlargest(half, "odds_ratio")
            prot = grp.nsmallest(half, "odds_ratio")
            kept.append(pd.concat([risk, prot]).drop_duplicates())
        sort_cols = ["odds_ratio", "n_admissions_present"]
        metric_label = "|log OR|"

    if not kept:
        return df.iloc[:0]
    result = pd.concat(kept, ignore_index=True)
    n_before = len(df)
    n_after = len(result)
    steps = sorted(df["n_steps"].unique())
    print(f"  [cap] {n_before} → {n_after} patterns by {metric_label} "
          f"(top {topn_per_length}/n_steps at steps {steps})")
    return result.sort_values(sort_cols, ascending=[False, False]).reset_index(drop=True)


def _jaccard_dedup_patterns(pat_patients, jaccard_threshold, rep_strategy="support"):
    """Deduplicate patterns by Jaccard similarity on carrier sets.

    Greedy clustering: iterate patterns by priority (based on rep_strategy),
    assign each to existing cluster if Jaccard >= threshold, otherwise start
    a new cluster.

    Uses numpy bit arrays for fast set operations on large carrier sets.

    Args:
        pat_patients: dict of pattern_key → set of patient indices
        jaccard_threshold: merge patterns with Jaccard >= this value
        rep_strategy: how to sort patterns for representative selection
            - "support": highest carrier count first (most common)
            - "n_tokens": most tokens first (most specific)
            - "n_steps": most visit blocks first (longest temporal chain)

    Returns:
        dict of representative_key → merged set of patient indices
        (non-representative patterns are dropped)
    """
    if jaccard_threshold <= 0 or not pat_patients:
        return pat_patients

    import numpy as np

    # Build sortable list
    items = []
    for pk, carriers in pat_patients.items():
        n_tokens = sum(len(s) for s in pk)
        n_steps = len(pk)
        if rep_strategy == "n_tokens":
            sort_val = (-n_tokens, -len(carriers))
        elif rep_strategy == "n_steps":
            sort_val = (-n_steps, -len(carriers))
        else:  # "support"
            sort_val = (-len(carriers), -n_tokens)
        items.append((pk, carriers, sort_val, len(carriers)))

    items.sort(key=lambda x: x[2])
    n_original = len(items)

    # Convert carrier sets to bit arrays for fast intersection
    # Map all patient indices to dense range
    all_patients = set()
    for _, carriers, _, _ in items:
        all_patients.update(carriers)
    patient_to_idx = {p: i for i, p in enumerate(sorted(all_patients))}
    n_patients = len(patient_to_idx)

    # Pack into uint8 arrays (each bit = one patient) using numpy packbits
    # For fast bitwise AND/OR operations
    def to_bitvec(carrier_set):
        bits = np.zeros(n_patients, dtype=np.bool_)
        for p in carrier_set:
            bits[patient_to_idx[p]] = True
        return np.packbits(bits)

    print(f"  [jaccard_dedup] Building bit vectors for {n_original} patterns...")
    bitvecs = []
    for pk, carriers, sv, sz in items:
        bitvecs.append(to_bitvec(carriers))

    clusters = []  # list of (rep_idx, rep_bitvec, rep_size, merged_carriers_set)
    cluster_for = {}  # item_idx → cluster_idx

    n_merged = 0
    report_interval = max(1, n_original // 10)

    for i, (pk, carriers, _, sz) in enumerate(items):
        if i > 0 and i % report_interval == 0:
            print(f"  [jaccard_dedup] {i}/{n_original} processed, {len(clusters)} clusters...")

        bv = bitvecs[i]
        best_cluster = -1
        best_jaccard = 0.0

        for ci, (rep_idx, rep_bv, rep_size, _) in enumerate(clusters):
            # Quick reject: Jaccard upper bound = min(|A|,|B|) / max(|A|,|B|)
            min_sz = min(sz, rep_size)
            max_sz = max(sz, rep_size)
            if max_sz == 0:
                continue
            upper_bound = min_sz / max_sz
            if upper_bound < jaccard_threshold:
                continue

            # Compute exact Jaccard via bit operations
            intersection = int(np.unpackbits(bv & rep_bv).sum())
            union = int(np.unpackbits(bv | rep_bv).sum())
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard >= jaccard_threshold and jaccard > best_jaccard:
                best_jaccard = jaccard
                best_cluster = ci

        if best_cluster >= 0:
            # Merge into existing cluster
            rep_idx, rep_bv, rep_size, merged_set = clusters[best_cluster]
            merged_set.update(carriers)
            clusters[best_cluster] = (rep_idx, rep_bv | bv, len(merged_set), merged_set)
            n_merged += 1
        else:
            # Start new cluster
            clusters.append((i, bv.copy(), sz, carriers.copy()))

    # Build result: rep pattern_key → merged carrier set
    result = {}
    for rep_idx, _, _, merged_set in clusters:
        pk = items[rep_idx][0]
        result[pk] = merged_set

    n_after = len(result)
    print(f"  [jaccard_dedup] {n_original} → {n_after} patterns "
          f"(threshold={jaccard_threshold}, strategy={rep_strategy})")
    return result


# --- Top-level mine_patterns dispatcher ---


def mine_patterns(seqs_flat, seqs_visit_blocks, y, method, min_support_frac,
                  ngram_min_len=1, ngram_max_len=4,
                  prefixspan_min_len=1, prefixspan_topn=500,
                  episode_max_len=4, episode_topn=500,
                  episode_min_steps=1,
                  cap_by_or_per_length=0,
                  cap_metric="or",
                  jaccard_dedup=0.0, jaccard_rep="support",
                  scoring_min_support_frac=-1) -> pd.DataFrame:
    n_total = len(y)
    case_idx = np.where(y == 1)[0]
    ctrl_idx = np.where(y == 0)[0]

    # Scoring-level support: separate from mining-level support
    if scoring_min_support_frac < 0:
        # Default: use same fraction as mining, applied to full cohort
        min_sup_scoring = max(1, int(np.ceil(min_support_frac * n_total)))
    elif scoring_min_support_frac == 0:
        # No scoring filter
        min_sup_scoring = 1
    else:
        min_sup_scoring = max(1, int(np.ceil(scoring_min_support_frac * n_total)))
    
    print(f"[mining] Scoring min support: {min_sup_scoring} ({scoring_min_support_frac})")

    # Mining-level support: per group
    min_sup_full = min_sup_scoring  # used at scoring stage

    if method == "episode":
        db_case = [seqs_visit_blocks[i] for i in case_idx]
        db_ctrl = [seqs_visit_blocks[i] for i in ctrl_idx]
        min_sup_case = max(1, int(np.ceil(min_support_frac * len(db_case))))
        min_sup_ctrl = max(1, int(np.ceil(min_support_frac * len(db_ctrl))))

        print(f"[mining] Episode mining: cases={len(db_case)}, controls={len(db_ctrl)}")
        freq_case = mine_episodes(db_case, min_sup_case, max_len=episode_max_len,
                                  topn=episode_topn, min_pattern_items=1)
        freq_ctrl = mine_episodes(db_ctrl, min_sup_ctrl, max_len=episode_max_len,
                                  topn=episode_topn, min_pattern_items=1)

        def _to_key(pat_list):
            return tuple(tuple(sorted(s)) for s in pat_list)

        # Build pattern → patient indices mapping
        # Mining returns indices relative to db_case / db_ctrl, so map back to full dataset
        pat_patients = {}  # pattern_key → set of full-dataset indices
        for _, p, case_idxs in freq_case:
            pk = _to_key(p)
            pat_patients[pk] = {case_idx[i] for i in case_idxs}
        for _, p, ctrl_idxs in freq_ctrl:
            pk = _to_key(p)
            if pk in pat_patients:
                pat_patients[pk] |= {ctrl_idx[i] for i in ctrl_idxs}
            else:
                pat_patients[pk] = {ctrl_idx[i] for i in ctrl_idxs}

        all_pattern_keys = set(pat_patients.keys())

        # ── Jaccard deduplication (before cross-check and scoring) ────────
        if jaccard_dedup > 0:
            pat_patients = _jaccard_dedup_patterns(
                pat_patients, jaccard_dedup, rep_strategy=jaccard_rep)
            all_pattern_keys = set(pat_patients.keys())

        # Now verify patterns in the OTHER group (case patterns in controls, vice versa)
        # Mining only found support within each group — need to check across
        pats_case_only = {_to_key(p) for _, p, _ in freq_case} & all_pattern_keys
        pats_ctrl_only = {_to_key(p) for _, p, _ in freq_ctrl} & all_pattern_keys

        # Pre-convert pattern keys to frozenset lists for subpattern checking
        pat_fsets = {}
        pat_tokens = {}
        for pk in all_pattern_keys:
            fsets = [frozenset(s) for s in pk]
            pat_fsets[pk] = fsets
            pat_tokens[pk] = set().union(*fsets)

        # For patterns found only in cases, check controls (and vice versa)
        pats_needing_cross_check = (pats_case_only - pats_ctrl_only) | (pats_ctrl_only - pats_case_only)
        if pats_needing_cross_check:
            print(f"  [scoring] Cross-checking {len(pats_needing_cross_check)} patterns across groups...")
            case_set = set(case_idx)
            ctrl_set = set(ctrl_idx)
            n_xcheck = len(pats_needing_cross_check)
            for xi, pk in enumerate(pats_needing_cross_check):
                if xi > 0 and xi % 500 == 0:
                    print(f"    cross-check {xi}/{n_xcheck}...")
                # If pattern was found in cases, check control patients
                # If pattern was found in controls, check case patients
                if pk in pats_case_only and pk not in pats_ctrl_only:
                    check_indices = ctrl_idx
                elif pk in pats_ctrl_only and pk not in pats_case_only:
                    check_indices = case_idx
                else:
                    continue
                fsets = pat_fsets[pk]
                required_tokens = pat_tokens[pk]
                for pi in check_indices:
                    patient_all_tokens = set()
                    for block in seqs_visit_blocks[pi]:
                        patient_all_tokens.update(block)
                    if not required_tokens.issubset(patient_all_tokens):
                        continue
                    if _is_subpattern(fsets, seqs_visit_blocks[pi]):
                        pat_patients[pk].add(pi)

        # Build admission_sets from pat_patients (inverted: patient → set of patterns)
        n_patients = len(seqs_visit_blocks)
        admission_sets = [set() for _ in range(n_patients)]
        for pk, patient_idxs in pat_patients.items():
            for pi in patient_idxs:
                admission_sets[pi].add(pk)

        print(f"[mining] method=episode | case={len(pats_case_only)} | ctrl={len(pats_ctrl_only)} | union={len(all_pattern_keys)}")
        df_scored = _score_patterns_admission_level(admission_sets, y, all_pattern_keys, min_sup_full)

        # Filter by minimum number of steps (visit blocks in pattern)
        if episode_min_steps > 1 and not df_scored.empty:
            n_before = len(df_scored)
            df_scored = df_scored[df_scored["n_steps"] >= episode_min_steps].reset_index(drop=True)
            print(f"  [min_steps] {n_before} → {len(df_scored)} patterns (≥{episode_min_steps} steps)")

        # Optional: cap by |log OR| per pattern length to keep the most
        # discriminative patterns at each length.
        if cap_by_or_per_length > 0 and not df_scored.empty:
            df_scored = _cap_by_discriminative_per_length(
                df_scored, topn_per_length=cap_by_or_per_length, cap_metric=cap_metric)

        return df_scored

    elif method == "ngram":
        seqs_case = [seqs_flat[i] for i in case_idx]
        seqs_ctrl = [seqs_flat[i] for i in ctrl_idx]
        min_sup_case = max(1, int(np.ceil(min_support_frac * len(seqs_case))))
        min_sup_ctrl = max(1, int(np.ceil(min_support_frac * len(seqs_ctrl))))
        sets_case = _admission_pattern_sets_ngram(seqs_case, ngram_min_len, ngram_max_len)
        sets_ctrl = _admission_pattern_sets_ngram(seqs_ctrl, ngram_min_len, ngram_max_len)
        counter_case = Counter()
        for s in sets_case:
            for pat in s:
                counter_case[pat] += 1
        counter_ctrl = Counter()
        for s in sets_ctrl:
            for pat in s:
                counter_ctrl[pat] += 1
        pats_case = {p for p, c in counter_case.items() if c >= min_sup_case}
        pats_ctrl = {p for p, c in counter_ctrl.items() if c >= min_sup_ctrl}
        all_patterns = pats_case | pats_ctrl
        admission_sets = _admission_pattern_sets_ngram(seqs_flat, ngram_min_len, ngram_max_len)
        print(f"[mining] method=ngram | case={len(pats_case)} | ctrl={len(pats_ctrl)} | union={len(all_patterns)}")
        return _score_patterns_admission_level(admission_sets, y, all_patterns, min_sup_full)

    elif method == "prefixspan":
        seqs_case = [seqs_flat[i] for i in case_idx]
        seqs_ctrl = [seqs_flat[i] for i in ctrl_idx]
        min_sup_case = max(1, int(np.ceil(min_support_frac * len(seqs_case))))
        min_sup_ctrl = max(1, int(np.ceil(min_support_frac * len(seqs_ctrl))))
        sets_case, raw_case = _admission_pattern_sets_prefixspan(
            seqs_case, min_sup_case, prefixspan_min_len, topn=prefixspan_topn)
        sets_ctrl, raw_ctrl = _admission_pattern_sets_prefixspan(
            seqs_ctrl, min_sup_ctrl, prefixspan_min_len, topn=prefixspan_topn)
        pats_case = {pat for _, pat in raw_case}
        pats_ctrl = {pat for _, pat in raw_ctrl}
        all_patterns = pats_case | pats_ctrl
        admission_sets = []
        for seq in seqs_flat:
            present = set()
            for pat in all_patterns:
                if _is_subsequence(pat, seq):
                    present.add(pat)
            admission_sets.append(present)
        print(f"[mining] method=prefixspan | case={len(pats_case)} | ctrl={len(pats_ctrl)} | union={len(all_patterns)}")
        return _score_patterns_admission_level(admission_sets, y, all_patterns, min_sup_full)

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# SUBSEQUENCE MATCHING FOR VALIDATION
# ============================================================================
