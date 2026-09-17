"""Jaccard cluster representatives over patient-carrier sets.

Used by ``scripts/make_fig3.py`` (Panel C dedup) and any downstream consumer
that wants the same set of "phenotype representative" patterns Fig 3 displays.

Apr 2026 rewrite: the greedy sweep was O(N\u00B2) with packed-bit intersection
per pair (~10-40 min at v2 scale depending on threshold). The new
implementation computes ALL pairwise intersections once via a single
``scipy.sparse`` matrix product (``M @ M.T`` over a patterns \u00D7 patients
boolean matrix) and then runs the greedy sweep on the precomputed sparse
intersection matrix. Walltime drops to seconds on the same dataset and the
cluster assignments are bit-identical to the packed-bit version (same
greedy order + same Jaccard definition).
"""
from __future__ import annotations

import os
import pickle
from typing import Optional, Set

import numpy as np
import pandas as pd


def _build_sparse_presence(df_pw_with_ck: pd.DataFrame):
    """Build a (N_patterns \u00D7 N_patients) sparse boolean carrier matrix.

    Returns (M, pattern_strs, sizes):
        M             : csr_matrix of shape (N_patterns, N_patients), int8 ones
        pattern_strs  : list of pattern strings in row order
        sizes         : np.ndarray of row sums (carrier counts per pattern)
    """
    from scipy.sparse import csr_matrix

    # Order patterns by |ig_signed_mean| descending \u2014 greedy sweep order.
    df = df_pw_with_ck.sort_values(
        "ig_signed_mean", key=abs, ascending=False
    ).reset_index(drop=True)

    # Build pattern \u2192 carrier-set iterator. Skip rows with non-set / empty.
    rows, cols = [], []
    patient_idx: dict = {}
    pattern_strs: list = []
    sizes: list = []
    for i, row in df.iterrows():
        ck = row["_ck"]
        if not isinstance(ck, set) or not ck:
            pattern_strs.append(row["pattern"])
            sizes.append(0)
            continue
        pattern_strs.append(row["pattern"])
        sizes.append(len(ck))
        for p in ck:
            j = patient_idx.get(p)
            if j is None:
                j = len(patient_idx)
                patient_idx[p] = j
            rows.append(i)
            cols.append(j)

    n_pat = len(pattern_strs)
    n_patients = len(patient_idx)
    data = np.ones(len(rows), dtype=np.int8)
    M = csr_matrix((data, (rows, cols)), shape=(n_pat, n_patients), dtype=np.int32)
    return M, pattern_strs, np.asarray(sizes, dtype=np.int64)


def _greedy_jaccard_clusters(df_pw_with_ck: pd.DataFrame,
                              jaccard_threshold: float) -> Set[str]:
    """Greedy sweep ordered by ``|ig_signed_mean|`` desc.

    For each unassigned pattern, grow a cluster of subsequent patterns whose
    carrier-set Jaccard \u2265 threshold; keep the seed as the representative.

    Implementation
    --------------
    1. Build sparse carrier matrix ``M`` (patterns \u00D7 patients, int).
    2. Compute ``I = M @ M.T`` \u2014 sparse (patterns \u00D7 patterns) of pairwise
       intersection counts. Only nonzero entries represent pairs that
       actually share a carrier; this is typically ~1-5% of all pairs.
    3. Greedy sweep: for each unassigned pattern i (in |IG| order), pull
       row i's nonzero columns, compute ``J = I / (s_i + s_j \u2212 I)`` in a
       single vectorized step, and mark every j >= threshold as assigned.

    Output is bit-identical to the legacy packed-bit implementation:
    same sort order, same Jaccard definition, same "merge first-seen
    surviving pattern" semantics.
    """
    from scipy.sparse import csr_matrix  # noqa: F401 \u2014 ensures availability

    # 1. Build sparse carrier matrix.
    M, pattern_strs, sizes = _build_sparse_presence(df_pw_with_ck)
    n_pat = M.shape[0]
    if n_pat == 0:
        return set()

    # 2. All pairwise intersections in one op. Output is CSR sparse.
    print(f"    [jaccard] sparse intersection: {n_pat:,} patterns, "
          f"{M.nnz:,} non-zero carrier entries")
    I = M @ M.T  # shape (n_pat, n_pat), int
    I = I.tocsr()
    print(f"    [jaccard] pairwise intersection matrix: "
          f"{I.nnz:,} non-zero pairs "
          f"({100 * I.nnz / max(n_pat * n_pat, 1):.3f}% dense)")

    # 3. Greedy sweep using the precomputed intersections.
    clusters: list = []
    assigned = np.zeros(n_pat, dtype=bool)
    indptr = I.indptr
    indices = I.indices
    data = I.data
    progress_step = max(1000, n_pat // 20)

    for i in range(n_pat):
        if assigned[i]:
            continue
        if sizes[i] == 0:
            assigned[i] = True
            clusters.append((i, [i]))
            continue

        members = [i]
        # Pull row i's nonzero columns + intersection values
        row_start, row_end = indptr[i], indptr[i + 1]
        js = indices[row_start:row_end]
        inters = data[row_start:row_end]

        # Keep only j > i AND unassigned AND size_j > 0
        mask = (js > i) & (~assigned[js]) & (sizes[js] > 0)
        js = js[mask]
        inters = inters[mask]
        if js.size:
            unions = sizes[i] + sizes[js] - inters
            # Guard against union == 0 (defensive; unions are strictly positive
            # when both sizes > 0 and we computed a real intersection).
            valid = unions > 0
            jaccards = np.zeros_like(unions, dtype=np.float64)
            jaccards[valid] = inters[valid] / unions[valid]
            hits = jaccards >= jaccard_threshold
            js_hit = js[hits]
            if js_hit.size:
                assigned[js_hit] = True
                members.extend(js_hit.tolist())

        assigned[i] = True
        clusters.append((i, members))

        if (i + 1) % progress_step == 0:
            print(f"    [jaccard] swept {i+1}/{n_pat} patterns, "
                  f"{len(clusters)} clusters so far", flush=True)

    print(f"    [jaccard] done: {len(clusters)} clusters from {n_pat} patterns")
    return set(pattern_strs[ri] for ri, _ in clusters)


def get_jaccard_representatives(
    csv_dir: str,
    split: str,
    jaccard_threshold: float,
    use_cache: bool = True,
) -> Optional[Set[str]]:
    """Return the set of representative patterns at the given Jaccard threshold.

    Disk cache: ``<csv_dir>/jaccard_reps_<split>_j<pct>.pkl``. Invalidated when
    ``pathway_importance_<split>.csv`` is newer than the cache.

    Returns ``None`` if pathway CSV or carrier sets pkl are missing.
    """
    # New layout drops the fig3_ prefix; fall back to the legacy name for
    # back-compat when reading old analysis caches.
    pw_csv = os.path.join(csv_dir, f"pathway_importance_{split}.csv")
    if not os.path.exists(pw_csv):
        pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    pkl_path = os.path.join(csv_dir, f"carrier_sets_{split}.pkl")

    if not os.path.exists(pw_csv) or not os.path.exists(pkl_path):
        print(f"  [jaccard] WARN: missing {pw_csv} or {pkl_path}")
        return None

    cache_path = os.path.join(
        csv_dir, f"jaccard_reps_{split}_j{int(jaccard_threshold * 100)}.pkl"
    )
    if use_cache and os.path.exists(cache_path) and \
            os.path.getmtime(cache_path) >= os.path.getmtime(pw_csv):
        with open(cache_path, "rb") as f:
            rep_patterns = pickle.load(f)
        print(f"  [jaccard] cached \u2192 {len(rep_patterns)} representatives "
              f"({os.path.basename(cache_path)})")
        return rep_patterns

    print(f"  [jaccard] computing representatives at threshold {jaccard_threshold} "
          f"(vectorized \u2014 seconds at ~40k patterns)...", flush=True)
    df_pw = pd.read_csv(pw_csv)
    with open(pkl_path, "rb") as f:
        carriers = pickle.load(f)

    df_pw["_ck"] = df_pw["pattern"].map(carriers)
    df_pw = df_pw[df_pw["_ck"].notna() & (df_pw["n_present"] >= 3)].copy()
    if df_pw.empty:
        return None

    rep_patterns = _greedy_jaccard_clusters(df_pw, jaccard_threshold)
    print(f"  [jaccard] {len(df_pw)} patterns \u2192 {len(rep_patterns)} representatives")

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(rep_patterns, f)
        print(f"  [jaccard] cached \u2192 {cache_path}")

    return rep_patterns
