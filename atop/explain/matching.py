"""Pattern matching against patient visit blocks.

Provides both per-patient matching (_match_all_patterns) and bulk indexed
matching (PatternIndex) for population-level queries.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from atop.mining.patterns import (
    _is_subsequence, _is_subpattern, _match_subpattern, _parse_episode_pattern,
)


# ═══════════════════════════════════════════════════════════════════════
# Single-patient matching (unchanged API)
# ═══════════════════════════════════════════════════════════════════════

def _match_all_patterns(df_patterns, salient_visit_blocks, is_episode):
    """
    Match a patient's salient visit blocks against ALL mined patterns.
    Returns list of (pattern_str, odds_ratio, support, pattern_tokens) tuples
    for every matched pattern, not just top-100.
    """
    matched = []
    if df_patterns.empty or not salient_visit_blocks:
        return matched

    for _, r in df_patterns.iterrows():
        pat_str = r["pattern"]
        if is_episode:
            episode_pat = _parse_episode_pattern(pat_str)
            pat_fsets = [frozenset(itemset) for itemset in episode_pat]
            if _is_subpattern(pat_fsets, salient_visit_blocks):
                pat_tokens = set()
                for itemset in episode_pat:
                    pat_tokens.update(itemset)
                matched.append((pat_str, r["odds_ratio"],
                                r["n_admissions_present"], pat_tokens))
        else:
            flat_pat = [p.strip() for p in pat_str.split(" -> ")]
            flat_salient = []
            for block in salient_visit_blocks:
                flat_salient.extend(sorted(block))
            if _is_subsequence(tuple(flat_pat), flat_salient):
                matched.append((pat_str, r["odds_ratio"],
                                r["n_admissions_present"], set(flat_pat)))
    return matched


# ═══════════════════════════════════════════════════════════════════════
# Bulk indexed matching (new — O(patterns × avg_candidates) instead of
# O(patterns × all_patients))
# ═══════════════════════════════════════════════════════════════════════

class PatternIndex:
    """
    Token-inverted index over patient visit blocks for fast pattern matching.

    Build once from df_test, then query per-pattern to get matching patient
    indices without scanning all patients.

    Usage:
        idx = PatternIndex.from_df(df_test)
        matched_rows = idx.patients_matching_pattern(pat_fsets, pat_tokens)
    """

    def __init__(self,
                 visit_blocks: List[List[FrozenSet[str]]],
                 row_indices: List[int],
                 token_to_rows: Dict[str, Set[int]]):
        self._visit_blocks = visit_blocks
        self._row_indices = row_indices
        self._token_to_rows = token_to_rows

    @classmethod
    def from_df(cls, df_test: pd.DataFrame) -> "PatternIndex":
        """Build index from test dataframe with salient_visit_blocks column."""
        visit_blocks = []
        row_indices = []
        token_to_rows: Dict[str, Set[int]] = {}

        for i, (_, row) in enumerate(df_test.iterrows()):
            svb = row.get("salient_visit_blocks", [])
            if not svb:
                visit_blocks.append([])
                row_indices.append(i)
                continue

            visit_blocks.append(svb)
            row_indices.append(i)

            # Index all tokens in this patient's visit blocks
            all_tokens = set()
            for block in svb:
                all_tokens.update(block)
            for tok in all_tokens:
                if tok not in token_to_rows:
                    token_to_rows[tok] = set()
                token_to_rows[tok].add(i)

        return cls(visit_blocks, row_indices, token_to_rows)

    def candidates_for_tokens(self, tokens: Set[str]) -> Set[int]:
        """
        Get row indices of patients that contain ALL tokens in the set.
        This is the fast pre-filter: intersect posting lists.
        """
        if not tokens:
            return set()

        sets = []
        for tok in tokens:
            if tok not in self._token_to_rows:
                return set()  # token not in any patient → no matches
            sets.append(self._token_to_rows[tok])

        # Intersect smallest first for efficiency
        sets.sort(key=len)
        result = sets[0].copy()
        for s in sets[1:]:
            result &= s
            if not result:
                return set()
        return result

    def patients_matching_pattern(self,
                                   pat_fsets: List[FrozenSet[str]],
                                   pat_tokens: Set[str],
                                   is_episode: bool = True) -> List[int]:
        """
        Return row indices of patients matching the pattern.
        Uses token index for fast candidate filtering, then verifies ordering.
        """
        # Fast pre-filter: all pattern tokens must be present
        candidates = self.candidates_for_tokens(pat_tokens)
        if not candidates:
            return []

        # Verify ordering for candidates only
        matched = []
        for i in candidates:
            svb = self._visit_blocks[i]
            if not svb:
                continue
            if is_episode:
                if _is_subpattern(pat_fsets, svb):
                    matched.append(i)
            else:
                flat_salient = []
                for block in svb:
                    flat_salient.extend(sorted(block))
                flat_pat = []
                for fs in pat_fsets:
                    flat_pat.extend(sorted(fs))
                if _is_subsequence(tuple(flat_pat), flat_salient):
                    matched.append(i)
        return matched

    def patients_matching_with_blocks(self,
                                      pat_fsets: List[FrozenSet[str]],
                                      pat_tokens: Set[str],
                                      is_episode: bool = True
                                      ) -> List[Tuple[int, list]]:
        """
        Return (row_index, matched_blocks) for patients matching the pattern.
        matched_blocks is a list of (pattern_itemset_idx, sequence_block_idx) tuples.
        """
        candidates = self.candidates_for_tokens(pat_tokens)
        if not candidates:
            return []

        results = []
        for i in candidates:
            svb = self._visit_blocks[i]
            if not svb:
                continue
            if is_episode:
                mb = _match_subpattern(pat_fsets, svb)
                if mb is not None:
                    results.append((i, mb))
            else:
                # For flat patterns, each token is a single-item frozenset
                mb = _match_subpattern(pat_fsets, svb)
                if mb is not None:
                    results.append((i, mb))
        return results


def build_pattern_match_matrix(df_patterns: pd.DataFrame,
                                df_test: pd.DataFrame,
                                is_episode: bool) -> pd.DataFrame:
    """
    Build a (patterns × test_patients) boolean match matrix using indexed matching.
    Returns DataFrame with pattern index as rows, test row index as columns,
    and True/False values.

    Much faster than iterating all patients for each pattern.
    """
    index = PatternIndex.from_df(df_test)
    n_patients = len(df_test)

    results = []
    for pat_idx, pat_row in df_patterns.iterrows():
        pat_str = pat_row["pattern"]
        if is_episode:
            episode_pat = _parse_episode_pattern(pat_str)
            pat_fsets = [frozenset(itemset) for itemset in episode_pat]
            pat_tokens = set()
            for itemset in episode_pat:
                pat_tokens.update(itemset)
        else:
            flat = [p.strip() for p in pat_str.split(" -> ")]
            pat_fsets = [frozenset([t]) for t in flat]
            pat_tokens = set(flat)

        matched_rows = index.patients_matching_pattern(
            pat_fsets, pat_tokens, is_episode)

        results.append({
            "pattern_idx": pat_idx,
            "pattern": pat_str,
            "matched_rows": set(matched_rows),
            "n_matched": len(matched_rows),
            "pat_tokens": pat_tokens,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
# Utility functions (unchanged)
# ═══════════════════════════════════════════════════════════════════════

def _compute_pattern_mass(matched_patterns, tok_ig_map, total_ig):
    """Compute attribution mass for each matched pattern, return sorted list."""
    results = []
    for pat_str, orr, sup, pat_tokens in matched_patterns:
        mass_val = sum(tok_ig_map.get(t, 0.0) for t in pat_tokens)
        mass_pct = mass_val / total_ig
        results.append({
            "pattern": pat_str, "odds_ratio": orr,
            "support": int(sup), "tokens": pat_tokens,
            "mass": mass_val, "mass_pct": mass_pct,
        })
    results.sort(key=lambda d: d["mass"], reverse=True)
    return results


def _extract_pattern_tokens(pat_str: str) -> List[str]:
    """Extract individual clinical tokens from a pattern string (flat or episode)."""
    tokens = []
    parts = pat_str.split(" -> ")
    for p in parts:
        p = p.strip()
        if p.startswith("{") and p.endswith("}"):
            items = [x.strip() for x in p[1:-1].split(",")]
            tokens.extend(items)
        else:
            tokens.append(p)
    return tokens
