"""run_validation_topup, GPU-only validation top-up for the Fig 4 whitelist.

Reads ``<run_dir>/5_analysis/validation_whitelist.txt`` (built by
``scripts/build_validation_whitelist.py``), determines which patterns are
**not** already in ``validation_<split>.csv``, and runs ``run_validation``
only for those, then merges the new rows into the cached CSV.

Why a thin wrapper instead of just bumping ``validate_top_k`` in
``run_atop.py``: the explainer's ``report()`` orchestration regenerates
several artifacts. We just want to top up the masking deltas for a fixed
pattern set, so the dedicated runner is faster and (more importantly)
doesn't risk shifting any upstream outputs.

Two non-obvious config requirements (both important):

  1. **Match Notebook 1B's mining knobs** so we hit the existing cached
     scored CSV (``all_scored_<hash>.csv``). Defaults below mirror the
     1B notebook (``min_support_frac=0.001``,
     ``scoring_min_support_frac=0``). Overriding either re-scores from
     scratch, ~10 min of wasted GPU.
  2. **Override ``episode_min_steps`` to 1** for the validation step.
     The default (``=2``) post-filters ``df_patterns`` to cross-visit
     patterns only, which silently drops every full-pool whitelist
     entry from ``run_validation``'s ``.isin(whitelist)`` filter.
     Setting it to 1 keeps single-block patterns visible without
     touching the cache (``min_steps`` is a post-load filter, not part
     of the cache hash).

Idempotent, re-running after success is a no-op.

Colab usage (from the repo root after ``%cd atop_package``):
    !python scripts/run_validation_topup.py \
        --run_dir ../runs/mimiciv_readmission_30d \
        --mimic_dir ../data/mimiciv/3.1/hosp \
        --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atop.run_layout import RunLayout


def main() -> None:
    p = argparse.ArgumentParser(
        description="Top up validation_<split>.csv with the union whitelist.")
    p.add_argument("--run_dir", required=True,
                   help="v2 run directory (with 0_inputs/, 1_model/, etc.)")
    p.add_argument("--mimic_dir", required=True,
                   help="MIMIC-IV hosp/ directory (needed by AToPExplainer.from_bundle).")
    p.add_argument("--device", default="auto",
                   help="cuda / cpu / auto (default auto).")
    p.add_argument("--split", default="test")
    p.add_argument("--whitelist_path", default=None,
                   help="Whitelist file (default: <run_dir>/5_analysis/"
                        "validation_whitelist.txt).")

    # ── Mining knobs, defaults mirror Notebook 1B so we hit the cache ──
    p.add_argument("--min_support_frac", type=float, default=0.001,
                   help="Mining min-support fraction per group (default 0.001 \u2014 "
                        "matches Notebook 1B).")
    p.add_argument("--scoring_min_support_frac", type=float, default=0.0,
                   help="Scoring min-support fraction (default 0 \u2014 matches "
                        "Notebook 1B; disables the extra full-cohort filter).")
    p.add_argument("--episode_max_len", type=int, default=4,
                   help="Max pattern length in visit blocks (default 4 \u2014 matches "
                        "Notebook 1B).")
    p.add_argument("--jaccard_dedup", type=float, default=0.0,
                   help="Mining-time Jaccard dedup (default 0 \u2014 matches Notebook 1B).")

    # ── Validation knobs ──────────────────────────────────────────────
    p.add_argument("--n_shuffle_draws", type=int, default=None,
                   help="Override config.n_shuffle_draws.")
    p.add_argument("--max_adm_per_pattern", type=int, default=None,
                   help="Override config.validate_max_admissions_per_pattern.")

    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be validated, then exit.")
    args = p.parse_args()

    layout = RunLayout(args.run_dir)
    whitelist_path = args.whitelist_path or os.path.join(
        layout.analysis_dir, "validation_whitelist.txt")
    val_csv = layout.validation_csv(args.split)

    # ── Load whitelist ────────────────────────────────────────────────
    if not os.path.exists(whitelist_path):
        sys.exit(f"  [ERROR] Missing {whitelist_path}\n"
                 f"  Run scripts/build_validation_whitelist.py first.")
    with open(whitelist_path) as f:
        whitelist = [l.strip() for l in f if l.strip()]
    print(f"[1] Whitelist: {len(whitelist)} patterns "
          f"({os.path.basename(whitelist_path)})")

    # ── Determine missing patterns ────────────────────────────────────
    if os.path.exists(val_csv):
        df_cached = pd.read_csv(val_csv)
        cached_set = set(df_cached["pattern"].tolist())
        print(f"[2] Cached: {len(cached_set)} patterns in "
              f"{os.path.basename(val_csv)}")
    else:
        df_cached = pd.DataFrame()
        cached_set = set()
        print(f"[2] No existing validation CSV \u2014 starting fresh.")

    missing = [p for p in whitelist if p not in cached_set]
    print(f"[3] Need GPU compute: {len(missing)} patterns "
          f"({len(whitelist) - len(missing)} already cached)")

    if not missing:
        print("    Nothing to do. Exiting.")
        return

    if args.dry_run:
        print("\nDry run \u2014 first 10 patterns that would be validated:")
        for p in missing[:10]:
            print(f"  {p}")
        return

    # ── Boot the explainer (loads model + tensors + IG cache) ─────────
    print("\n[4] Loading model bundle\u2026")
    from atop.explainer import AToPExplainer
    explainer = AToPExplainer.from_bundle(
        args.run_dir, mimic_dir=args.mimic_dir, device=args.device)

    # Apply mining-knob overrides BEFORE mine_patterns so we hit Notebook
    # 1B's scored cache. (These don't touch min_steps, which is a post-load
    # filter \u2014 we override that separately below.)
    explainer.config.min_support_frac = args.min_support_frac
    explainer.config.scoring_min_support_frac = args.scoring_min_support_frac
    explainer.config.episode_max_len = args.episode_max_len
    explainer.config.jaccard_dedup = args.jaccard_dedup
    # CRITICAL: keep single-block patterns in df_patterns so the
    # full-pool whitelist patterns survive run_validation's .isin() filter.
    explainer.config.episode_min_steps = 1

    print(f"  [config] min_support_frac         = {args.min_support_frac}")
    print(f"  [config] scoring_min_support_frac = {args.scoring_min_support_frac}")
    print(f"  [config] episode_max_len          = {args.episode_max_len}")
    print(f"  [config] jaccard_dedup            = {args.jaccard_dedup}")
    print(f"  [config] episode_min_steps        = 1  (override \u2014 keep single-block "
          f"patterns visible to whitelist filter)")

    explainer.compute_attributions()
    explainer.mine_patterns()

    # Sanity: how many whitelist patterns are visible in df_patterns?
    visible = sum(1 for p in missing
                  if p in set(explainer.df_patterns["pattern"].tolist()))
    print(f"\n  [sanity] {visible}/{len(missing)} missing whitelist patterns "
          f"present in df_patterns_train")
    if visible < len(missing):
        invisible = [p for p in missing
                     if p not in set(explainer.df_patterns["pattern"].tolist())]
        print(f"  [WARN] {len(invisible)} whitelist patterns NOT in df_patterns_train "
              f"\u2014 they will be silently dropped by run_validation. First 5:")
        for p in invisible[:5]:
            print(f"    {p}")

    # ── Run validation only for missing patterns ──────────────────────
    print(f"\n[5] Running masking validation for {visible} patterns\u2026")
    cfg = explainer.config
    n_shuffle = args.n_shuffle_draws or cfg.n_shuffle_draws
    max_adm = args.max_adm_per_pattern or cfg.validate_max_admissions_per_pattern
    # top_k caps risk + protective separately. We want every whitelist entry
    # to validate, so pass a number larger than any plausible whitelist size.
    top_k = max(1000, len(missing) * 2)

    from atop.explain.validation import run_validation
    # checkpoint_path = the canonical validation CSV. Each flush atomically
    # merges the new public-column rows into it (keyed on `pattern`), so a
    # Colab disconnect mid-run leaves the cache in a consistent partial
    # state, the next invocation's missing-pattern computation just sees a
    # smaller missing set and resumes from there.
    df_new = run_validation(
        explainer.model, explainer.device,
        explainer.df_test, explainer.df_patterns,
        explainer.tensors_by_key_test,
        explainer.icd_titles, explainer.vocab, explainer.vocab_inv,
        top_k, max_adm, n_shuffle,
        explainer._mining_method, layout.analysis_dir,
        df_ig=explainer.df_ig_test,
        pattern_whitelist=set(missing),
        checkpoint_path=val_csv,
        checkpoint_every=5,
    )

    if df_new is None or df_new.empty:
        sys.exit("  [ERROR] run_validation returned no rows.")

    # ── Merge + save (drop internal `_` columns to match existing schema) ─
    print(f"\n[6] Merging {len(df_new)} new rows into "
          f"{os.path.basename(val_csv)}\u2026")
    if not df_cached.empty:
        df_out = pd.concat([df_cached, df_new], ignore_index=True)
        df_out = df_out.drop_duplicates(subset=["pattern"], keep="last")
    else:
        df_out = df_new

    public_cols = [c for c in df_out.columns if not c.startswith("_")]
    df_out.to_csv(val_csv, index=False, columns=public_cols)
    print(f"    \u2192 {val_csv} ({len(df_out)} total patterns)")

    # ── Coverage report ──────────────────────────────────────────────
    final_set = set(df_out["pattern"].tolist())
    missing_after = [p for p in whitelist if p not in final_set]
    if missing_after:
        print(f"\n  [WARN] {len(missing_after)} whitelist patterns still missing "
              f"after run \u2014 likely zero matches in test split:")
        for p in missing_after[:5]:
            print(f"    {p}")
        if len(missing_after) > 5:
            print(f"    \u2026 and {len(missing_after) - 5} more")
    else:
        print(f"\n  \u2713 All {len(whitelist)} whitelist patterns now in cache.")


if __name__ == "__main__":
    main()
