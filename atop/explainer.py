"""
AToPExplainer — the main user-facing API for AToP.

Usage mirrors SHAP / Captum / LIME:

    explainer = AToPExplainer.from_bundle("runs/exp01", mimic_dir="/data/mimiciv/hosp")

    # Compute token-level attributions via Integrated Gradients
    explainer.compute_attributions()

    # Mine temporal patterns from attributed sequences
    explainer.mine_patterns(method="episode")

    # Explain a single patient
    result = explainer.explain_instance(patient_id="12345", hadm_id="67890")

    # Validate pattern reliance (masking + shuffling)
    explainer.validate(top_k=10)

    # Generate all figures and tables
    explainer.report(out_dir="explanations/")

    # Compare against LACE baseline
    explainer.compare_lace("lace_scores_all.csv")
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from atop.config import AToPConfig, set_seed, pick_device, PAD_IDX, SPECIAL_TOKENS
from atop.registry import load_bundle
from atop.data.mimic import load_mimic_tables, build_readmission_labels
from atop.data.tokenization import build_patient_sequences
from atop.data.datasets import (
    build_vocabulary, MIMICReadmissionDataset, collate_fn, split_samples_by_patient,
)
from atop.models.single_stream_transformer import SingleStreamTransformer
from atop.attribution.ig import IntegratedGradientsCustom
from atop.attribution.saliency import process_split_for_sequences
from atop.mining.patterns import mine_patterns
from atop.explain.matching import _match_all_patterns, _compute_pattern_mass
from atop.explain.validation import run_validation
from atop.explain.figures import (
    fig1_dataset_performance, fig2_patient_ig, fig3_top_patterns,
    fig5_validation, build_shap_comparison_figure,
    fig6_global_importance_comparison,
    fig_supp_ig_heatmap_multi_patient, fig_supp_ig_stream_heatmap,
    fig_supp_pattern_admission_heatmap, fig_supp_pattern_decomposition,
    save_table1, _filter_multi_token,
)
from atop.baselines.lace import load_lace_scores, run_lace_comparison
from atop.utils import load_icd_titles, load_drug_names, auroc_np, pr_auc_np


class AToPExplainer:
    """
    Attribution-guided Temporal Pattern explainer.

    Holds a trained model + data + computed state (attributions, patterns).
    Methods mirror a typical XAI workflow:

        compute_attributions()  →  mine_patterns()  →  validate()  →  report()
    """

    def __init__(
        self,
        model: SingleStreamTransformer,
        vocab: Dict[str, int],
        config: AToPConfig,
        device: torch.device,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]],
        icd_titles: Optional[Dict] = None,
        drug_names: Optional[Dict[str, str]] = None,
    ):
        self.model = model
        self.vocab = vocab
        self.vocab_inv = {idx: tok for tok, idx in vocab.items()}
        self.config = config
        self.device = device

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.icd_titles = icd_titles or {}
        self.drug_names = drug_names or {}

        # Datasets
        self._train_ds = MIMICReadmissionDataset(train_samples, vocab, config.max_seq_len)
        self._val_ds = MIMICReadmissionDataset(val_samples, vocab, config.max_seq_len)
        self._test_ds = MIMICReadmissionDataset(test_samples, vocab, config.max_seq_len)

        # Populated by compute_attributions()
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.df_ig_train: Optional[pd.DataFrame] = None
        self.df_ig_test: Optional[pd.DataFrame] = None
        self.tensors_by_key_test: Optional[Dict] = None
        self.tensors_by_key_train: Optional[Dict] = None  # None for train (too large)
        self._df_mine: Optional[pd.DataFrame] = None
        self._bundle_dir: str = ""

        # Populated by mine_patterns()
        self.df_patterns: Optional[pd.DataFrame] = None
        self._mining_method: Optional[str] = None

        # Populated by validate()
        self.df_validation: Optional[pd.DataFrame] = None

        # IG engine
        n_steps = config.ig_n_steps if config.ig_n_steps > 0 else 30
        self._ig = IntegratedGradientsCustom(model, n_steps=n_steps)

    # ── Constructors ─────────────────────────────────────────────────────

    @classmethod
    def from_bundle(
        cls,
        run_dir: str,
        mimic_dir: str,
        device: str = "auto",
        use_icd_titles: bool = True,
    ) -> "AToPExplainer":
        """
        Load a trained model bundle and prepare data for explanation.

        This is the primary constructor — mirrors SHAP's pattern of
        ``shap.Explainer(model, data)``.

        Args:
            run_dir: Path to model bundle (from train_model.py)
            mimic_dir: Path to MIMIC-IV hosp/ directory
            device: "auto", "cuda", or "cpu"
            use_icd_titles: Load human-readable ICD code titles
        """
        dev = pick_device(device)
        config, model, vocab, splits = load_bundle(run_dir, dev)
        set_seed(config.seed)

        # Rebuild data with identical config
        admissions, diagnoses, procedures, prescriptions = load_mimic_tables(mimic_dir)
        adm_labels = build_readmission_labels(
            admissions,
            exclude_elective_readmissions=config.exclude_elective_readmissions)

        samples = build_patient_sequences(
            adm_labels, diagnoses, procedures, prescriptions,
            max_visits=config.max_visits,
            one_per_patient=config.one_per_patient,
            max_drug_freq=config.max_drug_freq,
            drug_exclude_substrings=config.drug_exclude_substrings or None,
            token_types=config.token_types,
            chronic_filter=config.chronic_filter,
            first_occurrence_only=config.first_occurrence_only,
            first_occurrence_drugs_only=config.first_occurrence_drugs_only,
            harmonize_icd=config.harmonize_icd,
            icd_exclude_prefixes=config.icd_exclude_prefixes or None,
            drug_mapping=config.drug_mapping,
            cache_dir=os.path.join(mimic_dir, ".atop_cache"))

        # Reproduce splits from bundle
        train_pids = set(str(p) for p in splits.get("train", []))
        if train_pids:
            # Bundle has explicit splits — use them directly as the patient filter
            # (this supersedes max_patients since the bundle already applied it)
            val_pids = set(str(p) for p in splits.get("val", []))
            test_pids = set(str(p) for p in splits.get("test", []))
            all_split_pids = train_pids | val_pids | test_pids

            # Filter samples to only those patients in the bundle's splits
            samples = [s for s in samples if str(s["patient_id"]) in all_split_pids]

            train_s = [s for s in samples if str(s["patient_id"]) in train_pids]
            val_s = [s for s in samples if str(s["patient_id"]) in val_pids]
            test_s = [s for s in samples if str(s["patient_id"]) in test_pids]
        else:
            # No splits in bundle — apply max_patients then split fresh
            if config.max_patients > 0:
                all_pids = list(set(s["patient_id"] for s in samples))
                if len(all_pids) > config.max_patients:
                    rng = np.random.RandomState(config.seed)
                    keep = set(rng.choice(all_pids, size=config.max_patients, replace=False))
                    samples = [s for s in samples if s["patient_id"] in keep]
            train_s, val_s, test_s = split_samples_by_patient(samples, seed=config.seed)

        icd_titles = load_icd_titles(mimic_dir) if use_icd_titles else {}
        drug_names = load_drug_names(config.drug_mapping) if config.drug_mapping else {}

        # Set module-level drug name registry for label formatting
        from atop.explain.label_utils import set_drug_names
        set_drug_names(drug_names)

        print(f"[AToP] Ready: train={len(train_s):,} val={len(val_s):,} test={len(test_s):,}")
        obj = cls(model, vocab, config, dev, train_s, val_s, test_s, icd_titles, drug_names)
        obj._bundle_dir = run_dir
        return obj

    # ── Core pipeline steps ──────────────────────────────────────────────

    def compute_attributions(
        self,
        splits: str = "train+test",
        ig_max_train_samples: Optional[int] = None,
    ) -> "AToPExplainer":
        """
        Compute Integrated Gradients attributions and extract salient tokens.

        Args:
            splits: Which splits to process ("train", "test", "train+test")
            ig_max_train_samples: Subsample train set for IG (None = use config)

        Returns self for chaining.
        """
        cfg = self.config
        ig_bs = cfg.ig_batch_size
        max_train = ig_max_train_samples or cfg.ig_max_train_samples

        # ── Cache directory ─────────────────────────────────────────────
        cache_dir = os.path.join(self._bundle_dir, "ig_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Model identity marker: invalidate IG cache if model changed
        import hashlib
        model_pt = os.path.join(self._bundle_dir, "model.pt")
        if os.path.exists(model_pt):
            mtime = str(os.path.getmtime(model_pt))
            vsize = str(len(self.vocab))
            model_hash = hashlib.md5((mtime + vsize).encode()).hexdigest()[:10]
        else:
            model_hash = "unknown"
        cache_marker = os.path.join(cache_dir, f".model_{model_hash}")
        if not os.path.exists(cache_marker):
            # Model changed — invalidate all cached IG
            import glob
            for old_file in glob.glob(os.path.join(cache_dir, "*")):
                if os.path.isfile(old_file):
                    os.remove(old_file)
            with open(cache_marker, "w") as f:
                f.write(f"vocab_size={len(self.vocab)}\n")
            print(f"[IG cache] Model changed (hash={model_hash}) — cleared stale cache")

        if "train" in splits:
            cache_train = os.path.join(cache_dir, "sequences_train.pkl")
            cache_ig_train = os.path.join(cache_dir, "ig_train.csv")

            if os.path.exists(cache_train):
                print("[AToP] Loading cached IG for train split...")
                self.df_train = pd.read_pickle(cache_train)
                if os.path.exists(cache_ig_train):
                    self.df_ig_train = pd.read_csv(cache_ig_train)
            else:
                print("[AToP] Computing IG on train split...")
                if max_train > 0 and len(self._train_ds) > max_train:
                    indices = sorted(random.sample(range(len(self._train_ds)), max_train))
                    ig_train_ds = torch.utils.data.Subset(self._train_ds, indices)
                    print(f"  Subsampled {max_train:,} of {len(self._train_ds):,}")
                else:
                    ig_train_ds = self._train_ds

                self.df_train, self.df_ig_train, _ = process_split_for_sequences(
                    self.model, self._ig,
                    DataLoader(ig_train_ds, batch_size=ig_bs, shuffle=False, collate_fn=collate_fn),
                    self.device, self.vocab_inv, self.icd_titles,
                    cfg.ig_mass, cfg.ig_max_tokens, store_tensors=False)

                # Save cache
                self.df_train.to_pickle(cache_train)
                if self.df_ig_train is not None:
                    self.df_ig_train.to_csv(cache_ig_train, index=False)
                print(f"  [cache] Saved train IG to {cache_dir}/")

            if cfg.mine_on_trainval:
                df_val, _, _ = process_split_for_sequences(
                    self.model, self._ig,
                    DataLoader(self._val_ds, batch_size=ig_bs, shuffle=False, collate_fn=collate_fn),
                    self.device, self.vocab_inv, self.icd_titles,
                    cfg.ig_mass, cfg.ig_max_tokens, store_tensors=False)
                self._df_mine = pd.concat([self.df_train, df_val], ignore_index=True)
            else:
                self._df_mine = self.df_train

        if "test" in splits:
            cache_test = os.path.join(cache_dir, "sequences_test.pkl")
            cache_ig_test = os.path.join(cache_dir, "ig_test.csv")
            cache_tensors = os.path.join(cache_dir, "tensors_test.pt")

            if os.path.exists(cache_test) and os.path.exists(cache_tensors):
                print("[AToP] Loading cached IG for test split...")
                self.df_test = pd.read_pickle(cache_test)
                self.df_ig_test = pd.read_csv(cache_ig_test) if os.path.exists(cache_ig_test) else None
                self.tensors_by_key_test = torch.load(cache_tensors, weights_only=False)
            else:
                print("[AToP] Computing IG on test split...")
                self.df_test, self.df_ig_test, self.tensors_by_key_test = \
                    process_split_for_sequences(
                        self.model, self._ig,
                        DataLoader(self._test_ds, batch_size=ig_bs, shuffle=False, collate_fn=collate_fn),
                        self.device, self.vocab_inv, self.icd_titles,
                        cfg.ig_mass, cfg.ig_max_tokens, store_tensors=True)

                # Save cache
                self.df_test.to_pickle(cache_test)
                if self.df_ig_test is not None:
                    self.df_ig_test.to_csv(cache_ig_test, index=False)
                torch.save(self.tensors_by_key_test, cache_tensors)
                print(f"  [cache] Saved test IG to {cache_dir}/")

        return self

    def mine_patterns(
        self,
        method: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Mine temporal patterns from attributed training sequences.

        Two-level caching:
          Level 1 (pickle): Raw mined patterns + carrier sets from mine_episodes.
            Keyed by (method, min_support_frac, episode_max_len, n_train).
            If a cached result exists with LOWER min_support, reuses it
            (just filters by support threshold).
            Always mines with topn=0 to capture everything at the given support.
            Kill switch: skips cache if > 500K raw patterns.
          Post-cache filters (all applied after load, not cached):
            episode_topn, jaccard_dedup, cross-check, OR scoring,
            episode_min_steps, cap_by_or_per_length.

        Args:
            method: "episode", "prefixspan", or "ngram" (None = use config)
            **kwargs: Override any mining parameter from config

        Returns:
            DataFrame of mined patterns with odds ratios and support.
        """
        if self._df_mine is None:
            raise RuntimeError("Call compute_attributions() first.")

        import pickle, glob

        cfg = self.config
        self._mining_method = method or cfg.mining_method

        # ── Resolve params ───────────────────────────────────────────────
        min_support_frac = kwargs.get("min_support_frac", cfg.min_support_frac)
        episode_max_len = kwargs.get("episode_max_len", cfg.episode_max_len)
        episode_topn = kwargs.get("episode_topn", cfg.episode_topn)
        jaccard_dedup_thresh = kwargs.get("jaccard_dedup", cfg.jaccard_dedup)
        jaccard_rep_strategy = kwargs.get("jaccard_rep", cfg.jaccard_rep)
        episode_min_steps = kwargs.get("episode_min_steps", cfg.episode_min_steps)
        cap_by_or = kwargs.get("cap_by_or_per_length", cfg.cap_by_or_per_length)
        cap_metric = kwargs.get("cap_metric", cfg.cap_metric)

        MAX_CACHEABLE = 500_000

        n_train = len(self._df_mine)
        y = self._df_mine["readmission"].astype(int).to_numpy()
        case_idx = np.where(y == 1)[0]
        ctrl_idx = np.where(y == 0)[0]

        # ── Level 1: Raw mining cache ────────────────────────────────────
        cache_dir = os.path.join(self._bundle_dir, "mining_cache")
        os.makedirs(cache_dir, exist_ok=True)
        raw_prefix = f"raw_{self._mining_method}_{episode_max_len}_{n_train}"

        # Find existing caches with same method/max_len/n_train
        existing = {}
        for fpath in glob.glob(os.path.join(cache_dir, f"{raw_prefix}_*.pkl")):
            try:
                frac_str = os.path.basename(fpath).split("_")[-1].replace(".pkl", "")
                existing[float(frac_str)] = fpath
            except ValueError:
                continue

        # Find usable cache: need support <= requested (has everything we need)
        usable = None
        for cached_frac in sorted(existing.keys()):
            if cached_frac <= min_support_frac:
                usable = (cached_frac, existing[cached_frac])
                break

        freq_case = None
        freq_ctrl = None

        if usable:
            cached_frac, cached_path = usable
            print(f"[mining cache] Loading raw patterns (cached support={cached_frac}, "
                  f"requested={min_support_frac})")
            with open(cached_path, "rb") as f:
                cache_data = pickle.load(f)
            freq_case = cache_data["freq_case"]
            freq_ctrl = cache_data["freq_ctrl"]

            # Filter to requested support if cached was lower
            if cached_frac < min_support_frac:
                min_sup_case = max(1, int(np.ceil(min_support_frac * len(case_idx))))
                min_sup_ctrl = max(1, int(np.ceil(min_support_frac * len(ctrl_idx))))
                n_c_before, n_k_before = len(freq_case), len(freq_ctrl)
                freq_case = [(s, p, idx) for s, p, idx in freq_case if s >= min_sup_case]
                freq_ctrl = [(s, p, idx) for s, p, idx in freq_ctrl if s >= min_sup_ctrl]
                print(f"  [support filter] cases: {n_c_before} → {len(freq_case)}, "
                      f"controls: {n_k_before} → {len(freq_ctrl)}")
        else:
            # ── Mine from scratch ────────────────────────────────────────
            from atop.mining.patterns import mine_episodes
            db_case = [self._df_mine.iloc[i]["salient_visit_blocks"] for i in case_idx]
            db_ctrl = [self._df_mine.iloc[i]["salient_visit_blocks"] for i in ctrl_idx]
            min_sup_case = max(1, int(np.ceil(min_support_frac * len(db_case))))
            min_sup_ctrl = max(1, int(np.ceil(min_support_frac * len(db_ctrl))))

            print(f"[mining] Episode mining: cases={len(db_case)}, controls={len(db_ctrl)}")
            freq_case = mine_episodes(db_case, min_sup_case, max_len=episode_max_len,
                                      topn=0, min_pattern_items=1)
            freq_ctrl = mine_episodes(db_ctrl, min_sup_ctrl, max_len=episode_max_len,
                                      topn=0, min_pattern_items=1)

            # Cache raw output (with kill switch)
            n_raw = len(freq_case) + len(freq_ctrl)
            if n_raw <= MAX_CACHEABLE:
                cache_path = os.path.join(cache_dir, f"{raw_prefix}_{min_support_frac}.pkl")
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "freq_case": freq_case,
                        "freq_ctrl": freq_ctrl,
                        "min_support_frac": min_support_frac,
                    }, f)
                print(f"[mining cache] Saved {n_raw} raw patterns → {cache_path}")
            else:
                print(f"[mining cache] {n_raw} exceeds kill switch ({MAX_CACHEABLE}) — not caching")

        # ── Post-mining filter: episode_topn ─────────────────────────────
        if episode_topn > 0:
            freq_case.sort(key=lambda x: -x[0])
            freq_ctrl.sort(key=lambda x: -x[0])
            n_c, n_k = len(freq_case), len(freq_ctrl)
            freq_case = freq_case[:episode_topn]
            freq_ctrl = freq_ctrl[:episode_topn]
            print(f"  [topn={episode_topn}] cases: {n_c} → {len(freq_case)}, "
                  f"controls: {n_k} → {len(freq_ctrl)}")

        # Note: min_steps is NOT applied here — all patterns (single-token,
        # same-visit, cross-visit) are scored and cached. min_steps is applied
        # as a display filter when generating figures.

        # ── Build carrier sets ───────────────────────────────────────────
        def _to_key(pat_list):
            return tuple(tuple(sorted(s)) for s in pat_list)

        # Check for scored CSV cache (covers cross-check + OR, keyed by same raw mining params)
        import hashlib, json
        scored_hash_params = {
            "method": self._mining_method,
            "min_support_frac": min_support_frac,
            "episode_max_len": episode_max_len,
            "episode_topn": episode_topn,
            "n_train": n_train,
        }
        scored_hash = hashlib.md5(json.dumps(scored_hash_params, sort_keys=True).encode()).hexdigest()[:12]
        scored_csv = os.path.join(cache_dir, f"all_scored_{scored_hash}.csv")

        if os.path.exists(scored_csv):
            df_scored = pd.read_csv(scored_csv)
            print(f"[mining cache] Loaded scored patterns: {len(df_scored)} (hash={scored_hash})")
        else:
            pat_patients = {}
            for _, p, idxs in freq_case:
                pk = _to_key(p)
                pat_patients[pk] = {case_idx[i] for i in idxs}
            for _, p, idxs in freq_ctrl:
                pk = _to_key(p)
                if pk in pat_patients:
                    pat_patients[pk] |= {ctrl_idx[i] for i in idxs}
                else:
                    pat_patients[pk] = {ctrl_idx[i] for i in idxs}

            all_pattern_keys = set(pat_patients.keys())
            print(f"[mining] Union: {len(all_pattern_keys)} unique patterns")

            # ── Cross-check ──────────────────────────────────────────────
            from atop.mining.patterns import _is_subpattern
            pats_case_only = {_to_key(p) for _, p, _ in freq_case} & all_pattern_keys
            pats_ctrl_only = {_to_key(p) for _, p, _ in freq_ctrl} & all_pattern_keys

            pat_fsets = {pk: [frozenset(s) for s in pk] for pk in all_pattern_keys}
            pat_tokens = {pk: set().union(*pat_fsets[pk]) for pk in all_pattern_keys}

            seqs_vb = self._df_mine["salient_visit_blocks"].tolist()
            pats_xcheck = (pats_case_only - pats_ctrl_only) | (pats_ctrl_only - pats_case_only)
            if pats_xcheck:
                # Build per-patient token sets once (avoid rebuilding per pattern)
                patient_all_tokens = []
                for pi in range(len(seqs_vb)):
                    toks = set()
                    for block in seqs_vb[pi]:
                        toks.update(block)
                    patient_all_tokens.append(toks)

                n_xc = len(pats_xcheck)
                print(f"  [cross-check] {n_xc} patterns across groups...")

                from concurrent.futures import ProcessPoolExecutor, as_completed
                import multiprocessing as mp

                def _xcheck_batch(batch_pks, pat_fsets_b, pat_tokens_b,
                                  check_indices, seqs, pat_all_toks):
                    """Check a batch of patterns against a group of patients."""
                    results = {}
                    for pk in batch_pks:
                        fsets = pat_fsets_b[pk]
                        req_toks = pat_tokens_b[pk]
                        matched = set()
                        for pi in check_indices:
                            if not req_toks.issubset(pat_all_toks[pi]):
                                continue
                            if _is_subpattern(fsets, seqs[pi]):
                                matched.add(pi)
                        results[pk] = matched
                    return results

                # Split into case-only and ctrl-only batches
                case_only_pks = [pk for pk in pats_xcheck
                                 if pk in pats_case_only and pk not in pats_ctrl_only]
                ctrl_only_pks = [pk for pk in pats_xcheck
                                 if pk in pats_ctrl_only and pk not in pats_case_only]

                BATCH_SIZE = 200
                n_done = 0
                for group_pks, check_idx, label in [
                    (case_only_pks, ctrl_idx, "case→ctrl"),
                    (ctrl_only_pks, case_idx, "ctrl→case"),
                ]:
                    if not group_pks:
                        continue
                    for bi in range(0, len(group_pks), BATCH_SIZE):
                        batch = group_pks[bi:bi + BATCH_SIZE]
                        results = _xcheck_batch(
                            batch, pat_fsets, pat_tokens,
                            check_idx, seqs_vb, patient_all_tokens)
                        for pk, matched in results.items():
                            pat_patients[pk].update(matched)
                        n_done += len(batch)
                        if n_done % 500 <= BATCH_SIZE:
                            print(f"    {n_done}/{n_xc} ({label})...")

            # ── OR scoring ───────────────────────────────────────────────
            from atop.mining.patterns import _score_patterns_admission_level
            n_patients = len(seqs_vb)
            min_sup_full = max(1, int(np.ceil(min_support_frac * n_train)))
            admission_sets = [set() for _ in range(n_patients)]
            for pk, pidxs in pat_patients.items():
                for pi in pidxs:
                    admission_sets[pi].add(pk)

            print(f"[mining] Scoring {len(all_pattern_keys)} patterns...")
            df_scored = _score_patterns_admission_level(admission_sets, y, all_pattern_keys, min_sup_full)

            # Save scored CSV
            df_scored.to_csv(scored_csv, index=False)
            print(f"[mining cache] Saved {len(df_scored)} scored patterns → {scored_csv}")

        # ── Post-scoring filters (cheap, from cached CSV) ────────────────
        # Jaccard dedup (on scored df — uses n_present as proxy, not carrier sets)
        # Note: post-hoc Jaccard in figures uses test carrier sets for display
        if episode_min_steps > 1 and not df_scored.empty:
            n_before = len(df_scored)
            df_scored = df_scored[df_scored["n_steps"] >= episode_min_steps].reset_index(drop=True)
            print(f"  [min_steps] {n_before} → {len(df_scored)} (≥{episode_min_steps} steps)")

        if cap_by_or > 0 and not df_scored.empty:
            from atop.mining.patterns import _cap_by_discriminative_per_length
            df_scored = _cap_by_discriminative_per_length(
                df_scored, topn_per_length=cap_by_or, cap_metric=cap_metric)

        self.df_patterns = df_scored
        print(f"[AToP] Final: {len(self.df_patterns)} patterns ({self._mining_method})")
        return self.df_patterns

    def explain_instance(
        self,
        patient_id: str,
        hadm_id: str,
    ) -> Dict[str, Any]:
        """
        Explain a single patient/admission.

        Returns dict with:
          - ig_tokens: List of (token, ig_value) sorted by |IG|
          - salient_visit_blocks: Visit-grouped salient tokens
          - matched_patterns: Patterns present in this patient, ranked by mass
        """
        if self.df_ig_test is None or self.df_test is None:
            raise RuntimeError("Call compute_attributions() first.")
        if self.df_patterns is None:
            raise RuntimeError("Call mine_patterns() first.")

        pid, hid = str(patient_id), str(hadm_id)
        is_episode = (self._mining_method == "episode")

        # IG tokens
        ig_rows = self.df_ig_test[
            (self.df_ig_test["patient_id"] == pid) &
            (self.df_ig_test["index_hadm_id"] == hid)]
        if ig_rows.empty:
            return {"error": f"Patient {pid}/{hid} not found in test IG data"}

        ig_tokens = list(zip(ig_rows["token_str"], ig_rows["ig_abs"]))
        ig_tokens.sort(key=lambda x: x[1], reverse=True)

        # Visit blocks
        test_row = self.df_test[
            (self.df_test["patient_id"] == pid) &
            (self.df_test["index_hadm_id"] == hid)]
        svb = test_row.iloc[0].get("salient_visit_blocks", []) if len(test_row) > 0 else []

        # Pattern matching
        matched = _match_all_patterns(self.df_patterns, svb, is_episode)
        tok_ig = dict(zip(ig_rows["token_str"], ig_rows["ig_abs"]))
        total_ig = sum(tok_ig.values()) + 1e-12
        scored = _compute_pattern_mass(matched, tok_ig, total_ig)

        return {
            "patient_id": pid,
            "hadm_id": hid,
            "ig_tokens": ig_tokens,
            "salient_visit_blocks": svb,
            "n_matched_patterns": len(matched),
            "matched_patterns": scored,
        }

    def validate(
        self,
        top_k: Optional[int] = None,
        max_admissions_per_pattern: Optional[int] = None,
        n_shuffle_draws: Optional[int] = None,
        out_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Validate pattern reliance via masking and shuffling.

        Returns DataFrame of validation results.
        """
        if self.df_test is None or self.df_patterns is None:
            raise RuntimeError("Call compute_attributions() and mine_patterns() first.")

        cfg = self.config
        self.df_validation = run_validation(
            self.model, self.device, self.df_test, self.df_patterns,
            self.tensors_by_key_test, self.icd_titles, self.vocab, self.vocab_inv,
            top_k or cfg.validate_top_k,
            max_admissions_per_pattern or cfg.validate_max_admissions_per_pattern,
            n_shuffle_draws or cfg.n_shuffle_draws,
            self._mining_method,
            out_dir or ".",
            df_ig=self.df_ig_test,
        )
        return self.df_validation

    # ── Reporting ────────────────────────────────────────────────────────

    def report(self, out_dir: str, figures: Optional[List[str]] = None, **kwargs) -> None:
        """
        Generate figures, tables, and data CSVs.

        Args:
            out_dir: Base output directory.
            figures: List of figure names to generate. Default (None) = all.
                     Valid names: "fig2", "fig3", "fig4", "fig5",
                                  "supp_heatmap", "supp_decomposition",
                                  "supp_ood", "tables", "data"
                     Example: figures=["fig3", "fig5"] regenerates only those.

        Output structure:
            out_dir/
                figures/
                    main/       ← Paper figures (test split only)
                    supplement/ ← Supplementary figures
                    csv/        ← CSVs backing figures
                tables/         ← Summary tables
                data/           ← Raw data (pickles, IG CSVs)
        """
        ALL_FIGURES = {"fig2", "fig3", "fig4", "fig5", "attn_flow",
                       "supp_heatmap", "supp_decomposition", "supp_ood",
                       "tables", "data"}
        if figures is None:
            selected = ALL_FIGURES
        else:
            selected = set(figures)
            unknown = selected - ALL_FIGURES
            if unknown:
                print(f"  [WARN] Unknown figure names ignored: {unknown}")
                print(f"         Valid names: {sorted(ALL_FIGURES)}")
            selected = selected & ALL_FIGURES
        main_dir = os.path.join(out_dir, "figures", "main")
        supp_dir = os.path.join(out_dir, "figures", "supplement")
        csv_dir = os.path.join(out_dir, "figures", "csv")
        tbl_dir = os.path.join(out_dir, "tables")
        dat_dir = os.path.join(out_dir, "data")
        for d in [main_dir, supp_dir, csv_dir, tbl_dir, dat_dir]:
            os.makedirs(d, exist_ok=True)

        if self.df_patterns is None:
            raise RuntimeError("Call compute_attributions() and mine_patterns() first.")

        method = self._mining_method
        cfg = self.config

        # ── Data: intermediate pickles ───────────────────────────────────
        if "data" in selected:
            if self.df_train is not None:
                self.df_train.to_pickle(os.path.join(dat_dir, "sequences_train.pkl"))
            if self.df_test is not None:
                self.df_test.to_pickle(os.path.join(dat_dir, "sequences_test.pkl"))
            if self.df_ig_train is not None:
                self.df_ig_train.to_csv(os.path.join(dat_dir, "ig_train.csv"), index=False)
            if self.df_ig_test is not None:
                self.df_ig_test.to_csv(os.path.join(dat_dir, "ig_test.csv"), index=False)

        # ── Tables ───────────────────────────────────────────────────────
        if "tables" in selected:
            self.df_patterns.to_csv(os.path.join(tbl_dir, "mined_patterns_train.csv"), index=False)

            if self.df_train is not None and self.df_test is not None:
                all_labels = np.array([s["readmit_30d"] for s in
                                       self.train_samples + self.val_samples + self.test_samples])
                n_multi = sum(1 for s in self.train_samples + self.test_samples if s["n_visits"] > 1)
                summary = {
                    "n_samples": len(self.train_samples) + len(self.val_samples) + len(self.test_samples),
                    "n_unique_patients": len(set(
                        s["patient_id"] for s in self.train_samples + self.val_samples + self.test_samples)),
                    "n_readmit": int(all_labels.sum()),
                    "n_no_readmit": int((1 - all_labels).sum()),
                    "readmission_rate": float(all_labels.mean()),
                    "n_multi_visit": n_multi,
                    "pct_multi_visit": 100 * n_multi / max(len(all_labels), 1),
                    "vocab_size": len(self.vocab),
                }
                df_all = pd.concat([self.df_train, self.df_test], ignore_index=True)
                save_table1(df_all, summary, os.path.join(tbl_dir, "table1_population.csv"))

        # ── Mine patterns per split ──────────────────────────────────────
        df_patterns_train = self.df_patterns

        # ── Evaluate train-mined patterns on test (no test label leakage) ──
        # The pattern dictionary is built on train. On test we only compute:
        #   - prevalence (label-free)
        #   - mean IG mass (label-free)
        #   - OR on test (as a validation metric, not selection criterion)
        df_patterns_test = pd.DataFrame()

        if df_patterns_train is not None and not df_patterns_train.empty:
            # Re-use the train-mined pattern set for test evaluation
            # Panel C (fig3) will compute IG-based scoring on test patients
            # using only the train-discovered patterns
            df_patterns_test = df_patterns_train.copy()
            print(f"[AToP] Using {len(df_patterns_test)} train-mined patterns for test evaluation")

        # ══════════════════════════════════════════════════════════════════
        # MAIN FIGURES (test split only)
        # ══════════════════════════════════════════════════════════════════
        df_seq = self.df_test
        df_ig = self.df_ig_test
        df_pats = df_patterns_test
        tensors = self.tensors_by_key_test
        split = "test"

        if df_seq is None or df_ig is None:
            print("[WARN] No test data available — skipping main figures")
            return

        print(f"\n{'='*60}")
        print(f"[AToP] Generating MAIN figures (test split)...")
        print(f"{'='*60}")

        # ── Fig 1: already generated during training ─────────────────────

        # ── Fig 2: Patient exemplars — KernelSHAP vs IG vs visit tokens vs patterns
        if "fig2" in selected:
            n_exemplars = kwargs.get("n_exemplars", 3)
            picks = self._pick_exemplar_patients_from(
                df_seq, df_ig, df_pats, tensors, n=n_exemplars)
            for pi, pick in enumerate(picks):
                key = (pick["patient_id"], pick["index_hadm_id"])
                key_s = (str(pick["patient_id"]), str(pick["index_hadm_id"]))
                tensor_key = None
                if tensors:
                    if key in tensors:
                        tensor_key = key
                    elif key_s in tensors:
                        tensor_key = key_s
                if tensor_key is not None:
                    try:
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        # First exemplar → main dir, rest → supplement
                        fig2_dir = main_dir if pi == 0 else supp_dir
                        suffix = "" if pi == 0 else f"_ex{pi+1}"
                        build_shap_comparison_figure(
                            fig2_dir, self.model, self.device, self.icd_titles, pick,
                            tensors[tensor_key], self.vocab_inv, self._ig,
                            df_pats, df_seq,
                            cfg.shap_max_features, cfg.shap_nsamples, method,
                            split=split, csv_dir=csv_dir, filename_suffix=suffix)
                        print(f"  [fig2] Exemplar {pi+1}/{len(picks)}: patient {pick['patient_id']} "
                              f"(visits={pick['n_visits']}, matched={pick.get('n_matched','?')})")
                    except Exception as e:
                        import traceback
                        print(f"  [WARN] fig2 exemplar {pi+1} failed: {e}")
                        traceback.print_exc()
                else:
                    print(f"  [WARN] fig2 exemplar {pi+1} skipped — no stored tensor for {key}")
            if not picks:
                print(f"  [WARN] fig2 skipped — no exemplar patients found")

        # ── Fig 3: Global IG vs DeepSHAP vs AToP (two versions)
        df_fig3_patterns = None
        if "fig3" in selected and tensors and df_pats is not None and not df_pats.empty:
            try:
                df_fig3_patterns = fig6_global_importance_comparison(
                    main_dir, self.model, self.device, self.icd_titles,
                    self.vocab_inv, self._ig, df_ig, df_pats, df_seq,
                    tensors, method, cfg.shap_nsamples, n_show=cfg.n_show,
                    split=split, supp_dir=supp_dir, csv_dir=csv_dir)
            except Exception as e:
                import traceback
                print(f"  [WARN] fig3 failed: {e}")
                traceback.print_exc()

        # Build pattern whitelist from fig3 Panel C (cross-visit patterns)
        # If Jaccard clustering is enabled, use cluster representatives instead
        fig3_pattern_list = None
        if df_fig3_patterns is not None and not df_fig3_patterns.empty:
            fig3_pattern_list = set(df_fig3_patterns["pattern"].tolist())
            print(f"  [fig3→fig4/fig5] Passing {len(fig3_pattern_list)} cross-visit patterns")

            # Jaccard dedup: cluster patterns and use representatives as whitelist
            jaccard_thresh = kwargs.get("jaccard_dedup", cfg.jaccard_dedup)
            if jaccard_thresh > 0 and "_carrier_keys" in df_fig3_patterns.columns:
                try:
                    from atop.explain.figures import fig_supp_jaccard_clusters
                    import pickle as _pkl

                    # Load carrier sets if available (saved during fig3)
                    carrier_pkl = os.path.join(csv_dir, f"carrier_sets_{split}.pkl")
                    if os.path.exists(carrier_pkl):
                        with open(carrier_pkl, "rb") as _f:
                            _carriers = _pkl.load(_f)
                        _df_jac = df_fig3_patterns.copy()
                        if "_carrier_keys" not in _df_jac.columns:
                            _df_jac["_carrier_keys"] = _df_jac["pattern"].map(_carriers)
                            _df_jac = _df_jac[_df_jac["_carrier_keys"].notna()].copy()

                        # Run greedy Jaccard clustering
                        _df_jac = _df_jac[_df_jac["n_present"] >= 3].copy()
                        _sorted = _df_jac.sort_values("ig_signed_mean", key=abs, ascending=False)
                        _reps = set()
                        _assigned = set()
                        for idx, row in _sorted.iterrows():
                            if idx in _assigned:
                                continue
                            _reps.add(row["pattern"])
                            ck_i = row["_carrier_keys"]
                            if not isinstance(ck_i, set) or not ck_i:
                                _assigned.add(idx)
                                continue
                            for idx2, row2 in _sorted.iterrows():
                                if idx2 <= idx or idx2 in _assigned:
                                    continue
                                ck_j = row2["_carrier_keys"]
                                if not isinstance(ck_j, set) or not ck_j:
                                    continue
                                inter = len(ck_i & ck_j)
                                union = len(ck_i | ck_j)
                                if union > 0 and inter / union >= jaccard_thresh:
                                    _assigned.add(idx2)
                            _assigned.add(idx)

                        print(f"  [fig3→fig4] Jaccard dedup (>={jaccard_thresh}): "
                              f"{len(fig3_pattern_list)} → {len(_reps)} cluster representatives")
                        fig3_pattern_list = _reps
                except Exception as e:
                    print(f"  [fig3→fig4] Jaccard dedup skipped: {e}")

        # ── Fig 4: Perturbation validation (multi-token only for main)
        if "fig4" in selected and df_pats is not None and not df_pats.empty and tensors:
            try:
                from atop.explain.validation import run_validation

                # Incremental cache: load existing validation results, only validate NEW patterns
                val_cache_path = os.path.join(csv_dir, f"validation_{split}.csv")
                df_val_cached = None
                cached_patterns = set()
                if os.path.exists(val_cache_path):
                    df_val_cached = pd.read_csv(val_cache_path)
                    cached_patterns = set(df_val_cached["pattern"].tolist())
                    print(f"  [fig4] Loaded {len(cached_patterns)} cached validation results")

                # Determine which patterns need validation
                if fig3_pattern_list is not None:
                    needed_patterns = fig3_pattern_list - cached_patterns
                else:
                    needed_patterns = None  # validate all (no whitelist)

                if needed_patterns is not None and len(needed_patterns) == 0:
                    print(f"  [fig4] All patterns already validated — skipping model inference")
                    df_val = df_val_cached
                else:
                    if needed_patterns is not None:
                        print(f"  [fig4] {len(needed_patterns)} new patterns to validate "
                              f"({len(cached_patterns)} already cached)")
                    df_val_new = run_validation(
                        self.model, self.device, df_seq, df_pats,
                        tensors, self.icd_titles, self.vocab, self.vocab_inv,
                        cfg.validate_top_k, cfg.validate_max_admissions_per_pattern,
                        cfg.n_shuffle_draws, method, main_dir,
                        df_ig=df_ig,
                        pattern_whitelist=needed_patterns if needed_patterns else fig3_pattern_list,
                    )

                    # Merge with cached results
                    if df_val_cached is not None and not df_val_new.empty:
                        df_val = pd.concat([df_val_cached, df_val_new], ignore_index=True)
                        df_val = df_val.drop_duplicates(subset=["pattern"], keep="last")
                    else:
                        df_val = df_val_new

                # Save full accumulated cache
                df_val.to_csv(val_cache_path, index=False,
                              columns=[c for c in df_val.columns if not c.startswith("_")])

                # Main paper: use fig3 whitelist if available (same patterns as Panel C)
                if fig3_pattern_list is not None:
                    df_val_main = df_val[df_val["pattern"].isin(fig3_pattern_list)].copy()
                else:
                    # Fallback: cross-visit filter
                    from atop.explain.figures import _count_pattern_tokens, _count_pattern_blocks
                    df_val_main = df_val[
                        (df_val["pattern"].apply(_count_pattern_tokens) >= 2) &
                        (df_val["pattern"].apply(_count_pattern_blocks) >= 2)
                    ].copy()
                if not df_val_main.empty:
                    fig5_validation(main_dir, df_val_main, cfg.n_shuffle_draws,
                                   icd_titles=self.icd_titles, split=split)
                    print(f"  [fig4_validation] {len(df_val_main)} patterns validated (aligned with fig3)")

                # Supplement: all patterns (incl. single-token)
                fig5_validation(supp_dir, df_val, cfg.n_shuffle_draws,
                               icd_titles=self.icd_titles, split=f"{split}_all")

                # Supplement: OOD diagnostic
                if "supp_ood" in selected:
                    from atop.explain.figures import fig_supp_ood_diagnostic
                    fig_supp_ood_diagnostic(supp_dir, df_val, split=split)
            except Exception as e:
                import traceback
                print(f"  [WARN] fig4 failed: {e}")
                traceback.print_exc()

        # ── Fig 5: Reversed-order analysis (temporal specificity)
        if "fig5" in selected and df_pats is not None and not df_pats.empty:
            try:
                from atop.explain.validation import run_reversed_order_analysis
                from atop.explain.figures import fig5_reversed_order
                df_rev = run_reversed_order_analysis(
                    df_pats, df_seq, df_ig, method,
                    icd_titles=self.icd_titles, top_k=15,
                    pattern_whitelist=fig3_pattern_list,
                )
                if not df_rev.empty:
                    fig5_reversed_order(main_dir, df_rev, icd_titles=self.icd_titles,
                                       split=split, csv_dir=csv_dir)
            except Exception as e:
                import traceback
                print(f"  [WARN] fig5 failed: {e}")
                traceback.print_exc()

        # ══════════════════════════════════════════════════════════════════
        # SUPPLEMENTARY FIGURES
        # ══════════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print(f"[AToP] Generating SUPPLEMENTARY figures...")
        print(f"{'='*60}")

        # Attention flow analysis
        if "attn_flow" in selected and tensors:
            # Load df_fig3_patterns from cache if not already set (e.g., --figures attn_flow)
            if df_fig3_patterns is None:
                _pw_cache = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
                if os.path.exists(_pw_cache):
                    df_fig3_patterns = pd.read_csv(_pw_cache)
                    print(f"  [attn_flow] Loaded {len(df_fig3_patterns)} patterns from cached CSV")
                else:
                    print(f"  [attn_flow] No cached pathway CSV found — run fig3 first")

        if "attn_flow" in selected and tensors and df_fig3_patterns is not None:
            try:
                from atop.explain.attention_flow import compute_attention_flow, fig_attention_flow
                attn_csv = os.path.join(csv_dir, f"attention_flow_{split}.csv")
                if os.path.exists(attn_csv):
                    df_attn = pd.read_csv(attn_csv)
                    print(f"  [attn_flow] Loaded cached results: {len(df_attn)} patterns")
                else:
                    print(f"  [attn_flow] Computing attention flow for top patterns...")
                    # Load carrier sets for targeted patient sampling
                    _carrier_sets = None
                    _carrier_pkl = os.path.join(csv_dir, f"carrier_sets_{split}.pkl")
                    if os.path.exists(_carrier_pkl):
                        import pickle as _pkl
                        with open(_carrier_pkl, "rb") as _f:
                            _carrier_sets = _pkl.load(_f)
                        print(f"  [attn_flow] Loaded {len(_carrier_sets)} carrier sets for targeted sampling")
                    # Force CPU — flash attention on CUDA doesn't return attention weights
                    import copy
                    _attn_model = copy.deepcopy(self.model).cpu()
                    _attn_device = torch.device("cpu")
                    _attn_tensors = {k: v.cpu() for k, v in tensors.items()}
                    df_attn = compute_attention_flow(
                        _attn_model, _attn_device, _attn_tensors, self.vocab_inv,
                        df_pats, df_fig3_patterns,
                        n_patterns=cfg.attention_n_patterns, max_patients=cfg.attention_max_patients,
                        carrier_sets=_carrier_sets)
                    del _attn_model, _attn_tensors
                    if not df_attn.empty:
                        df_attn.to_csv(attn_csv, index=False)
                fig_attention_flow(supp_dir, df_attn, icd_titles=self.icd_titles,
                                    n_show=cfg.n_show, split=split)
            except Exception as e:
                import traceback
                print(f"  [WARN] attn_flow failed: {e}")
                traceback.print_exc()

        # Supp: Training curves (already generated during training as fig_train_curves)

        # Supp: Pattern admission heatmap (multi-token)
        if df_pats is not None and not df_pats.empty:
            df_pats_multi = _filter_multi_token(df_pats, min_tokens=2)
            if not df_pats_multi.empty:
                if "supp_heatmap" in selected:
                    try:
                        fig_supp_pattern_admission_heatmap(
                            supp_dir, df_pats_multi, df_seq, self.icd_titles,
                            method, df_ig=df_ig, split=split)
                    except Exception as e:
                        print(f"  [WARN] supp_presence failed: {e}")

                if "supp_decomposition" in selected:
                    try:
                        fig_supp_pattern_decomposition(
                            supp_dir, df_pats_multi, df_seq, df_ig, self.icd_titles,
                            method, split=split)
                    except Exception as e:
                        print(f"  [WARN] supp_decomposition failed: {e}")

        # ── README for figures ─────────────────────────────────────────────
        readme = (
            "# Figure–Manuscript Mapping\n\n"
            "| Manuscript | File | Description |\n"
            "|---|---|---|\n"
            "| Fig 1 | (run dir) `fig1_dataset_performance.png` | ROC, calibration, dataset summary |\n"
            f"| Fig 2 | `main/fig2_patient_explanation_{split}.png` | Single patient: KernelSHAP vs IG vs visit timeline vs patterns |\n"
            f"| Fig 3 | `main/fig3_global_comparison_{split}.png` | Global: IG vs DeepSHAP vs AToP (≥2-token patterns) |\n"
            f"| Fig 4 | `main/fig4_temporal_validation_{split}.png` | Perturbation validation (≥2-token, mask/shuffle visits/shuffle within) |\n"
            "| S1 | (run dir) `fig_train_curves.png` | Training curves |\n"
            f"| S2 | `supplement/fig3_global_comparison_{split}_all.png` | Fig 3 with all patterns incl. single-token |\n"
            f"| S3 | `supplement/fig3_global_comparison_train.png` | Fig 3 on TRAIN split (overfitting check) |\n"
            f"| S4 | `supplement/fig4_temporal_validation_{split}_all.png` | Fig 4 with all patterns incl. single-token |\n"
            f"| S5 | `supplement/supp_pattern_presence_{split}.png` | Pattern admission heatmap |\n"
            f"| S6 | `supplement/supp_pattern_decomposition_{split}.png` | Per-token IG breakdown within patterns |\n"
        )

        # ── Train-split fig3 (supplement for reviewer comparison) ─────────
        if ("fig3" in selected and self.df_ig_train is not None
                and self.df_train is not None and df_pats is not None):
            try:
                print(f"\n{'='*60}")
                print(f"[AToP] Generating TRAIN-split fig3 supplement...")
                print(f"{'='*60}")
                fig6_global_importance_comparison(
                    supp_dir, self.model, self.device, self.icd_titles,
                    self.vocab_inv, self._ig, self.df_ig_train, df_pats,
                    self.df_train, {},  # empty tensors → skips GradientSHAP
                    method, cfg.shap_nsamples, n_show=cfg.n_show,
                    split="train", supp_dir=supp_dir, csv_dir=csv_dir)
            except Exception as e:
                import traceback
                print(f"  [WARN] Train-split fig3 failed: {e}")
                traceback.print_exc()
        with open(os.path.join(out_dir, "figures", "README.md"), "w") as f:
            f.write(readme)

        print(f"\n{'='*60}")
        print(f"[AToP] All figures saved to {out_dir}/figures/")
        print(f"  Main:         {main_dir}")
        print(f"  Supplementary: {supp_dir}")
        print(f"  CSVs:          {csv_dir}")
        print(f"{'='*60}")

    # ── LACE comparison ──────────────────────────────────────────────────

    def compare_lace(
        self,
        lace_csv: str,
        out_dir: Optional[str] = None,
        transformer_auroc: Optional[float] = None,
        transformer_prauc: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compare model against LACE baseline using pre-computed scores.

        Args:
            lace_csv: Path to lace_scores_all.csv (from precompute_lace.py)
            out_dir: Base output dir — saves to {out_dir}/tables/lace_baseline.csv
        """
        lace_df = load_lace_scores(lace_csv)
        # Write to tables/ subdir if out_dir is provided
        save_dir = os.path.join(out_dir, "tables") if out_dir else "."
        os.makedirs(save_dir, exist_ok=True)
        return run_lace_comparison(
            lace_df, self.train_samples, self.test_samples,
            save_dir,
            transformer_test_auroc=transformer_auroc,
            transformer_test_prauc=transformer_prauc)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _pick_exemplar_patient(self) -> Optional[Dict]:
        """Pick test patient with most matched patterns for figure exemplar.
        
        Samples up to 2000 multi-visit test patients (for speed), then picks
        the one with most pattern matches and highest attribution mass.
        Prefers readmitted + multi-visit patients.
        """
        if self.df_ig_test is None or self.df_test is None or self.df_patterns is None:
            print("  [exemplar] Skipped — missing IG, test, or patterns data")
            return None

        is_episode = (self._mining_method == "episode")
        
        print(f"  [exemplar] df_test={len(self.df_test)}, "
              f"df_patterns={len(self.df_patterns) if self.df_patterns is not None else 'None'}, "
              f"tensors={len(self.tensors_by_key_test) if self.tensors_by_key_test else 0}")
        
        # Filter to patients with stored tensors and salient visit blocks
        n_no_tensor = 0
        n_no_svb = 0
        eligible = []
        for _, row in self.df_test.iterrows():
            pid, hid = row["patient_id"], row["index_hadm_id"]
            if (pid, hid) not in self.tensors_by_key_test:
                n_no_tensor += 1
                continue
            svb = row.get("salient_visit_blocks", [])
            if not svb:
                n_no_svb += 1
                continue
            eligible.append(row)
        
        print(f"  [exemplar] Eligible={len(eligible)}, "
              f"no_tensor={n_no_tensor}, no_svb={n_no_svb}")
        
        # Debug: check for type mismatch between df_test keys and tensor keys
        if n_no_tensor > 0 and self.tensors_by_key_test:
            sample_tensor_key = next(iter(self.tensors_by_key_test))
            sample_df_row = self.df_test.iloc[0]
            sample_df_key = (sample_df_row["patient_id"], sample_df_row["index_hadm_id"])
            print(f"  [exemplar] tensor key sample: {sample_tensor_key} "
                  f"(types: {type(sample_tensor_key[0]).__name__}, {type(sample_tensor_key[1]).__name__})")
            print(f"  [exemplar] df_test key sample: {sample_df_key} "
                  f"(types: {type(sample_df_key[0]).__name__}, {type(sample_df_key[1]).__name__})")
        
        if not eligible:
            print("  [exemplar] No eligible patients (no tensors or visit blocks)")
            return None
        
        # Prioritize multi-visit readmitted patients, sample for speed
        import random
        random.seed(42)
        # Sort: readmitted multi-visit first
        eligible.sort(key=lambda r: (
            -(r.get("n_visits", 1) > 1),
            -(r.get("readmission", 0)),
            -len(r.get("salient_visit_blocks", []))
        ))
        eligible = eligible[:2000]  # cap for speed
        
        print(f"  [exemplar] Scanning {len(eligible)} candidates...")
        
        # Precompute IG lookup — sum across positions for repeated tokens
        ig_cache = {}
        for (pid, hid), grp in self.df_ig_test.groupby(["patient_id", "index_hadm_id"]):
            ig_cache[(pid, hid)] = grp.groupby("token_str")["ig_abs"].sum().to_dict()
        
        candidates = []
        for i, row in enumerate(eligible):
            pid, hid = row["patient_id"], row["index_hadm_id"]
            svb = row.get("salient_visit_blocks", [])
            matched = _match_all_patterns(self.df_patterns, svb, is_episode)
            if not matched:
                continue

            tok_ig = ig_cache.get((pid, hid), {})
            total_ig = sum(tok_ig.values()) + 1e-12
            total_mass = sum(
                sum(tok_ig.get(t, 0.0) for t in pt) / total_ig
                for _, _, _, pt in matched)

            candidates.append({
                "patient_id": pid, "index_hadm_id": hid,
                "readmission": row["readmission"],
                "n_visits": row.get("n_visits", 1),
                "n_matched": len(matched), "total_mass": total_mass})
            
            # Early stop if we have a good candidate
            if len(candidates) >= 50:
                break

        if not candidates:
            print("  [exemplar] No patients matched any patterns")
            return None

        cdf = pd.DataFrame(candidates)
        cdf["tier"] = 2
        cdf.loc[cdf["n_visits"] > 1, "tier"] = 1
        cdf.loc[(cdf["n_visits"] > 1) & (cdf["readmission"] == 1.0), "tier"] = 0
        cdf = cdf.sort_values(["tier", "n_matched", "total_mass"],
                               ascending=[True, False, False])
        c = cdf.iloc[0]
        print(f"  [exemplar] Selected patient {c['patient_id']} "
              f"(visits={int(c['n_visits'])}, matched={c['n_matched']}, "
              f"readmit={c['readmission']})")
        return {"patient_id": c["patient_id"], "index_hadm_id": c["index_hadm_id"],
                "readmission": c["readmission"], "n_visits": int(c["n_visits"])}

    def _pick_exemplar_patients_from(self, df_seq, df_ig, df_pats, tensors,
                                      n=3) -> List[Dict]:
        """Pick N diverse exemplar patients for fig2.
        
        Selection strategy:
          1. Score all eligible patients by pattern coverage
          2. Pick top readmitted multi-visit patient
          3. Pick top readmitted patient with different dominant phenotype
          4. Pick top non-readmitted patient (for contrast)
        """
        if df_ig is None or df_seq is None or df_pats is None or df_pats.empty:
            print("  [exemplar] Skipped — missing data")
            return []

        is_episode = (self._mining_method == "episode")
        ig_col = "ig_signed" if "ig_signed" in df_ig.columns else "ig_abs"

        # Precompute IG cache
        ig_cache = {}
        for (pid, hid), grp in df_ig.groupby(["patient_id", "index_hadm_id"]):
            ig_cache[(str(pid), str(hid))] = grp.groupby("token_str")[ig_col].sum().to_dict()

        # Filter eligible patients
        eligible = []
        for _, row in df_seq.iterrows():
            pid, hid = str(row["patient_id"]), str(row["index_hadm_id"])
            svb = row.get("salient_visit_blocks", [])
            if not svb:
                continue
            if tensors is not None:
                key_s = (pid, hid)
                key_orig = (row["patient_id"], row["index_hadm_id"])
                if key_s not in tensors and key_orig not in tensors:
                    continue
            eligible.append(row)

        if not eligible:
            print("  [exemplar] No eligible patients")
            return []

        # Prioritize multi-visit patients
        eligible.sort(key=lambda r: (
            -(r.get("n_visits", 1) > 1),
            -(r.get("readmission", 0)),
            -len(r.get("salient_visit_blocks", []))
        ))
        eligible = eligible[:2000]

        print(f"  [exemplar] Scanning {len(eligible)} candidates for {n} exemplars...")
        candidates = []
        for row in eligible:
            pid, hid = str(row["patient_id"]), str(row["index_hadm_id"])
            svb = row.get("salient_visit_blocks", [])
            matched = _match_all_patterns(df_pats, svb, is_episode)
            if not matched:
                continue
            tok_ig = ig_cache.get((pid, hid), {})
            total_ig = sum(abs(v) for v in tok_ig.values()) + 1e-12
            total_mass = sum(
                sum(abs(tok_ig.get(t, 0.0)) for t in pt) / total_ig
                for _, _, _, pt in matched)
            # Collect matched pattern strings for diversity check
            matched_pats = set(m[0] for m in matched)
            candidates.append({
                "patient_id": pid, "index_hadm_id": hid,
                "readmission": row["readmission"],
                "n_visits": row.get("n_visits", 1),
                "n_matched": len(matched), "total_mass": total_mass,
                "matched_patterns": matched_pats})

        if not candidates:
            print("  [exemplar] No patients matched any patterns")
            return []

        cdf = pd.DataFrame(candidates)
        cdf["tier"] = 2  # single-visit non-readmitted
        cdf.loc[cdf["n_visits"] > 1, "tier"] = 1  # multi-visit
        cdf.loc[(cdf["n_visits"] > 1) & (cdf["readmission"] == 1.0), "tier"] = 0  # multi-visit readmitted

        # Greedy diverse selection
        picks = []
        used_patterns = set()

        for target_tier, label in [(0, "readmitted multi-visit"),
                                    (0, "readmitted diverse"),
                                    (1, "non-readmitted multi-visit"),
                                    (2, "any")]:
            if len(picks) >= n:
                break
            pool = cdf[cdf["tier"] <= target_tier].copy()
            # Exclude already-picked patients
            pool = pool[~pool["patient_id"].isin([p["patient_id"] for p in picks])]
            if pool.empty:
                continue

            if len(picks) > 0:
                # Score by diversity: prefer patients with patterns NOT already covered
                pool["novel_patterns"] = pool["matched_patterns"].apply(
                    lambda ps: len(ps - used_patterns))
                pool = pool.sort_values(["novel_patterns", "n_matched", "total_mass"],
                                         ascending=[False, False, False])
            else:
                pool = pool.sort_values(["n_matched", "total_mass"],
                                         ascending=[False, False])

            c = pool.iloc[0]
            used_patterns.update(c["matched_patterns"])
            picks.append({
                "patient_id": c["patient_id"],
                "index_hadm_id": c["index_hadm_id"],
                "readmission": c["readmission"],
                "n_visits": int(c["n_visits"]),
                "n_matched": c["n_matched"],
            })
            print(f"  [exemplar] Pick {len(picks)}: patient {c['patient_id']} "
                  f"({label}, visits={int(c['n_visits'])}, matched={c['n_matched']})")

        return picks
