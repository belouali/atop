"""
Pick four exemplar patients for Figure 2 and its supplements.

Applies the selection criteria from TASKS/_shared_context.md:
  main_tp       confident TP, cardiac/renal phenotype preferred
  alt_tp        confident TP, a *different* phenotype (psychiatric or oncologic)
  confident_tn  y_true=0, y_prob ≤ 0.1, obstetric preferred
  disagreement  FP or FN with high confidence (deferred content; picks
                the highest-confidence FP we can find so the stub exists)

Writes ../runs/full_CPD/explain/exemplars.json and prints a top-10-per-category
console report so the pick can be spot-checked.

Run from atop_package/:
    python scripts/select_exemplars.py
"""
from __future__ import annotations
import os, sys, json, glob, pickle
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from atop.explain.matching import _match_all_patterns
from atop.mining.patterns import _is_subpattern, _parse_episode_pattern
from atop.run_layout import RunLayout

import argparse
_p = argparse.ArgumentParser(description="Select exemplar patients (main_tp / alt_tp / confident_tn / disagreement)")
_p.add_argument("--run_dir", default="../runs/mimiciv_readmission_30d",
                help="Run directory (v2 layout: reads from 3_attribution/4_mining/2_predictions, writes to 5_analysis)")
_p.add_argument("--output", default=None,
                help="Override exemplars.json output path (default: <run_dir>/5_analysis/exemplars.json)")
_args = _p.parse_args()

BUNDLE = _args.run_dir
_layout = RunLayout(BUNDLE)
OUT = _args.output or _layout.exemplars_json

# ──────────────────────────────────────────────────────────────────────────
# Phenotype heuristics, token-prefix matching over a candidate's top matched
# patterns. Ordered for readability; each patient is assigned the label whose
# prefix set captures the largest fraction of their matched-pattern tokens.
# ──────────────────────────────────────────────────────────────────────────
PHENOTYPES = {
    "cardiac_renal": [
        "C:10_I25", "C:10_I50", "C:10_I13", "C:10_I48", "C:10_I42",
        "C:10_N17", "C:10_N18", "C:10_N19",
    ],
    "psychiatric": [
        "C:10_F32", "C:10_F33", "C:10_F41", "C:10_F43", "C:10_F29",
        "C:10_F20", "C:10_F31", "C:10_F40", "C:10_F10",
        "C:10_R4585",    # suicidal ideation
    ],
    "oncologic": [
        "C:10_C",        # malignant neoplasms
        "C:10_D63",      # neoplastic anemia
        "C:10_Z51",      # cancer therapy
        "C:10_G893",     # neoplasm pain
    ],
    "obstetric": [
        "C:10_Z37",      # outcome of delivery
        "C:10_O",        # O-chapter pregnancy codes
        "P:10_10D0",     # delivery procedures
        "P:10_10E0",     # delivery-associated procedures
    ],
}


def phenotype_of(matched_tokens: list[str]) -> str:
    """Label a bag of tokens by the phenotype whose prefixes best cover it."""
    if not matched_tokens:
        return "unknown"
    counts = Counter()
    for tok in matched_tokens:
        for label, prefixes in PHENOTYPES.items():
            if any(tok.startswith(p) for p in prefixes):
                counts[label] += 1
                break
    if not counts:
        return "other"
    top_label, top_n = counts.most_common(1)[0]
    # Require top phenotype to cover at least 15% of tokens to claim it;
    # otherwise "other" is more honest.
    if top_n / len(matched_tokens) < 0.15:
        return "other"
    return top_label


# ──────────────────────────────────────────────────────────────────────────
# Load inputs
# ──────────────────────────────────────────────────────────────────────────
print("[1] Loading inputs...")
pred = pd.read_csv(_layout.test_predictions_csv,
                   dtype={"subject_id": str, "hadm_id": str})
with open(_layout.sequences_pkl("test"), "rb") as f:
    seq = pickle.load(f)
seq["patient_id"]    = seq["patient_id"].astype(str)
seq["index_hadm_id"] = seq["index_hadm_id"].astype(str)
print(f"    test_predictions: {len(pred):,} rows")
print(f"    sequences_test:   {len(seq):,} rows")

pat_files = sorted(glob.glob(os.path.join(_layout.mining_dir, "all_scored_*.csv")),
                   key=os.path.getmtime)
df_patterns = pd.read_csv(pat_files[-1])
print(f"    mining dictionary: {os.path.basename(pat_files[-1])} "
      f"({len(df_patterns):,} patterns)")

# Merge on (subject_id, hadm_id) ↔ (patient_id, index_hadm_id)
df = pred.merge(
    seq[["patient_id", "index_hadm_id", "n_visits",
         "len_salient", "len_full", "salient_visit_blocks"]],
    left_on=["subject_id", "hadm_id"],
    right_on=["patient_id", "index_hadm_id"],
    how="inner",
)
print(f"    merged cohort: {len(df):,} patients")


# ──────────────────────────────────────────────────────────────────────────
# Filter candidate pools
# ──────────────────────────────────────────────────────────────────────────
pos = df[df["y_true"] == 1]
p80_prob = float(np.percentile(pos["y_prob"], 80))
print(f"\n[2] Confidence thresholds:")
print(f"    P80 of positive predictions: y_prob = {p80_prob:.3f}")

# ──────────────────────────────────────────────────────────────────────────
# Uniform n_visits range across all exemplar pools, keeps Fig 2 / Fig S4
# panels visually balanced (was: 4-9 for TP, 2+ for TN, mismatch made
# protective panels look threadbare next to a 6-visit risk patient).
# ──────────────────────────────────────────────────────────────────────────
N_VISITS_MIN = 3
N_VISITS_MAX = 5

cand_tp = df[
    (df["y_true"] == 1)
    & (df["y_prob"] >= max(0.5, p80_prob))
    & (df["n_visits"] >= N_VISITS_MIN) & (df["n_visits"] <= N_VISITS_MAX)
    & (df["len_salient"] >= 10)
].copy()

# TN candidates: percentile-based threshold within the in-window negative cohort.
# A static y_prob ≤ 0.15 doesn't work because multi-visit patients (3-5
# visits) systematically score higher \u2014 n_visits is a model feature, so the
# model raises baseline risk for them. Using a percentile of the in-window
# negatives keeps the meaning ("model confidently said this patient won't
# readmit") relative to the cohort the exemplar lives in.
neg_in_window = df[
    (df["y_true"] == 0)
    & (df["n_visits"] >= N_VISITS_MIN) & (df["n_visits"] <= N_VISITS_MAX)
    & (df["len_salient"] >= 6)     # protective patients have fewer salient tokens
]
p10_neg_in_window = float(np.percentile(neg_in_window["y_prob"], 10)) \
                    if len(neg_in_window) else 0.15
print(f"    P10 of in-window negatives: y_prob = {p10_neg_in_window:.3f}")
cand_tn = neg_in_window[neg_in_window["y_prob"] <= p10_neg_in_window].copy()

cand_fp = df[
    (df["y_true"] == 0)
    & (df["y_prob"] >= 0.6)
    & (df["n_visits"] >= N_VISITS_MIN) & (df["n_visits"] <= N_VISITS_MAX)
].copy()
print(f"    TP candidates (confident): {len(cand_tp):,}")
print(f"    TN candidates (confident): {len(cand_tn):,}")
print(f"    FP candidates (high-conf): {len(cand_fp):,}")


# ──────────────────────────────────────────────────────────────────────────
# Pre-parse all patterns ONCE.
#
# The library helper _match_all_patterns(df_patterns, svb, is_episode) is
# convenient but inefficient inside a loop \u2014 it re-runs
# _parse_episode_pattern + frozenset construction for all 70k patterns on
# every call. With ~1k candidates that's ~70M wasted parses.
#
# We pre-parse once and add a token pre-filter so per-candidate work is
# O(patterns_with_subset_tokens), not O(all_patterns). Empirically ~25-30\u00D7
# faster on a v2-scale mining dictionary.
# ──────────────────────────────────────────────────────────────────────────
print("\n[3a] Pre-parsing pattern dictionary...")
_PARSED = []  # list of (pat_str, pat_fsets, pat_tokens, odds_ratio, n_present)
for _idx, _r in df_patterns.iterrows():
    _ep = _parse_episode_pattern(_r["pattern"])
    _fsets = [frozenset(s) for s in _ep]
    _toks = set().union(*_ep) if _ep else set()
    _PARSED.append((_r["pattern"], _fsets, _toks,
                    _r["odds_ratio"], _r["n_admissions_present"]))
print(f"    parsed {len(_PARSED):,} patterns")


def _match_for_patient(svb):
    """Token-prefiltered episode matching for a single patient's visit blocks."""
    if not svb:
        return []
    # Build patient's token set once for fast pat_tokens \u2286 patient_tokens check
    patient_tokens = set()
    for block in svb:
        patient_tokens.update(block)
    matched = []
    for pat_str, pat_fsets, pat_tokens, orr, npres in _PARSED:
        # Cheap pre-filter: every pattern token must be present in the patient
        if not pat_tokens.issubset(patient_tokens):
            continue
        # Expensive ordering check only on survivors
        if _is_subpattern(pat_fsets, svb):
            matched.append((pat_str, orr, npres, pat_tokens))
    return matched


# ──────────────────────────────────────────────────────────────────────────
# Score candidates: run pattern matching + phenotype labelling
# ──────────────────────────────────────────────────────────────────────────
def score(cand: pd.DataFrame, tag: str) -> pd.DataFrame:
    print(f"\n[3] Matching patterns for {len(cand):,} {tag} candidates...")
    # Sort by y_prob to prioritize the strongest candidates \u2014 we can stop early
    # once we have enough per-phenotype variety.
    if tag == "tn":
        cand = cand.sort_values("y_prob", ascending=True).head(400).copy()
    else:
        cand = cand.sort_values("y_prob", ascending=False).head(400).copy()

    out_rows = []
    for i, (_, r) in enumerate(cand.iterrows()):
        if i and i % 50 == 0:
            print(f"    {tag}: {i}/{len(cand)} scored")
        svb = r["salient_visit_blocks"]
        matched = _match_for_patient(svb)
        if not matched:
            continue
        # Top-20 by |odds_ratio - 1| as a proxy for pattern importance
        top = sorted(matched, key=lambda m: -abs(m[1] - 1))[:20]
        toks = []
        for _, _, _, pat_tokens in top:
            toks.extend(pat_tokens)
        pheno = phenotype_of(toks)
        out_rows.append({
            "patient_id":    r["patient_id"],
            "index_hadm_id": r["index_hadm_id"],
            "y_true":        int(r["y_true"]),
            "y_prob":        float(r["y_prob"]),
            "n_visits":      int(r["n_visits"]),
            "len_salient":   int(r["len_salient"]),
            "len_full":      int(r["len_full"]),
            "n_matched":     len(matched),
            "phenotype":     pheno,
            "top_pattern":   top[0][0] if top else "",
        })
    return pd.DataFrame(out_rows)


scored_tp = score(cand_tp, "tp")
scored_tn = score(cand_tn, "tn")
scored_fp = score(cand_fp, "fp") if len(cand_fp) else pd.DataFrame()


def dump_top(df_scored: pd.DataFrame, tag: str, sort_by="y_prob",
             ascending=False, n=10):
    print(f"\n── Top {n} {tag} candidates (sorted by {sort_by}"
          f"{' ↑' if ascending else ' ↓'}) ──")
    if df_scored is None or df_scored.empty:
        print("    (none, filters returned zero candidates with matched patterns)")
        return
    shown = df_scored.sort_values(sort_by, ascending=ascending).head(n)
    cols = ["patient_id", "index_hadm_id", "y_prob", "n_visits",
            "len_salient", "n_matched", "phenotype"]
    print(shown[cols].to_string(index=False))


dump_top(scored_tp, "TP (main / alt)")
dump_top(scored_tn, "TN (confident_tn)", sort_by="y_prob", ascending=True)
if len(scored_fp):
    dump_top(scored_fp, "FP (disagreement)")


# ──────────────────────────────────────────────────────────────────────────
# Select final exemplars
# ──────────────────────────────────────────────────────────────────────────
def pick(df_scored: pd.DataFrame, phenotype: str,
         min_matched: int = 20, sort_by: str = "y_prob",
         ascending: bool = False, exclude_pids: set[str] | None = None):
    """Return first row matching phenotype + min_matched (or the strongest
    runner-up if phenotype is absent in the pool)."""
    if df_scored.empty:
        return None
    pool = df_scored[df_scored["n_matched"] >= min_matched]
    if exclude_pids:
        pool = pool[~pool["patient_id"].isin(exclude_pids)]
    preferred = pool[pool["phenotype"] == phenotype].sort_values(
        sort_by, ascending=ascending)
    if len(preferred):
        return preferred.iloc[0].to_dict()
    # fallback, strongest candidate regardless of phenotype
    fallback = pool.sort_values(sort_by, ascending=ascending)
    return fallback.iloc[0].to_dict() if len(fallback) else None


print("\n" + "=" * 60)
print("FINAL PICKS")
print("=" * 60)

# main_tp: richest confident-TP within the locked n_visits range.
# Uses the same {N_VISITS_MIN..N_VISITS_MAX} window as alt_tp and
# confident_tn so all three Fig 2 / Fig S4 panels are visually comparable.
def _pick_main(df_scored, min_matched=50):
    if df_scored.empty:
        return None
    pool = df_scored[
        df_scored["n_matched"] >= min_matched
    ].sort_values(["y_prob", "n_matched"], ascending=[False, False])
    return pool.iloc[0].to_dict() if len(pool) else None

main_tp = _pick_main(scored_tp)
if main_tp is None:
    print(f"\u26A0  No {N_VISITS_MIN}-{N_VISITS_MAX}-visit TP with \u226550 matched patterns; "
          f"falling back to cardiac_renal")
    main_tp = pick(scored_tp, "cardiac_renal")

# alt_tp: different phenotype from main_tp (try psychiatric first, then oncologic)
alt_pheno_try = [p for p in ("psychiatric", "oncologic")
                 if p != main_tp.get("phenotype")]
alt_tp = None
for ph in alt_pheno_try:
    alt_tp = pick(scored_tp, ph, exclude_pids={main_tp["patient_id"]})
    if alt_tp and alt_tp["phenotype"] == ph:
        break
if alt_tp is None:
    alt_tp = pick(scored_tp, "__none__", exclude_pids={main_tp["patient_id"]})

# confident_tn: obstetric preferred, lowest y_prob.
# min_matched=5 (vs 20 for TPs), protective patterns are rarer; obstetric
# especially has few cross-visit pattern matches.
confident_tn = pick(scored_tn, "obstetric",
                    min_matched=5, sort_by="y_prob", ascending=True)
if confident_tn is None:
    confident_tn = pick(scored_tn, "__none__",
                        min_matched=5, sort_by="y_prob", ascending=True)

# disagreement: highest-confidence FP
disagreement = None
if len(scored_fp):
    disagreement = scored_fp.sort_values("y_prob", ascending=False).iloc[0].to_dict()
    disagreement["type"] = "FP"

# complex_tp: confident TP with many visits (≥15), the long-trajectory
# counterpoint to main_tp. Used as a supplement exemplar demonstrating
# AToP on a chronic-disease history (e.g., cirrhosis + cancer).
def _pick_complex(df_scored, min_visits=15, min_matched=100,
                  exclude_pids=None):
    if df_scored.empty:
        return None
    pool = df_scored[
        (df_scored["n_visits"] >= min_visits)
        & (df_scored["n_matched"] >= min_matched)
    ]
    if exclude_pids:
        pool = pool[~pool["patient_id"].isin(exclude_pids)]
    pool = pool.sort_values(["n_matched", "y_prob"], ascending=[False, False])
    return pool.iloc[0].to_dict() if len(pool) else None

complex_tp = _pick_complex(scored_tp,
                           exclude_pids={main_tp["patient_id"],
                                         alt_tp["patient_id"] if alt_tp else None})

exemplars = {
    "main_tp":      main_tp,
    "alt_tp":       alt_tp,
    "complex_tp":   complex_tp,
    "confident_tn": confident_tn,
    "disagreement": disagreement,
}
for key, pick_row in exemplars.items():
    if pick_row is None:
        print(f"\n{key}: ⚠️  none found")
        continue
    print(f"\n{key}:")
    for k in ("patient_id", "index_hadm_id", "y_true", "y_prob",
             "n_visits", "len_salient", "n_matched", "phenotype"):
        print(f"  {k:<15} {pick_row.get(k)}")
    if pick_row.get("top_pattern"):
        print(f"  top_pattern     {pick_row['top_pattern'][:70]}")

# ──────────────────────────────────────────────────────────────────────────
# Write exemplars.json
# ──────────────────────────────────────────────────────────────────────────
def _clean(d):
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer, np.int64)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            out[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out

payload = {k: _clean(v) for k, v in exemplars.items()}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(payload, f, indent=2, default=str)
print(f"\n✅ Wrote {OUT}")
