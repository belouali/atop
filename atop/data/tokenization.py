"""Token construction, filtering, and patient sequence building."""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# ICD-9 → ICD-10 Harmonization via CMS GEMs
# ============================================================================

_GEM_CACHE: Dict[str, Dict[str, str]] = {}  # {"cm": {icd9→icd10}, "pcs": {icd9→icd10}}

def _load_gem_mapping(gem_dir: str) -> Dict[str, Dict[str, str]]:
    """Load CMS General Equivalence Mappings (ICD-9 → ICD-10).
    
    Looks for CSV files:
      - icd9toicd10cmgem.csv  (diagnoses)
      - icd9toicd10pcsgem.csv (procedures)
    
    Only keeps 1:1 forward mappings (no_map=0, combination=0).
    Returns {"cm": {icd9→icd10}, "pcs": {icd9→icd10}}.
    """
    import os
    global _GEM_CACHE
    if _GEM_CACHE:
        return _GEM_CACHE
    
    result = {"cm": {}, "pcs": {}}
    
    for label, filename in [("cm", "icd9toicd10cmgem.csv"), ("pcs", "icd9toicd10pcsgem.csv")]:
        path = os.path.join(gem_dir, filename)
        if not os.path.isfile(path):
            print(f"  [GEM] {filename} not found in {gem_dir}, skipping {label}")
            continue
        
        df = pd.read_csv(path, dtype=str)
        # Standardize column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        icd9_col = [c for c in df.columns if "icd9" in c][0]
        icd10_col = [c for c in df.columns if "icd10" in c][0]
        
        # Filter: keep only no_map=0, combination=0
        if "no_map" in df.columns:
            df = df[df["no_map"].astype(str) == "0"]
        if "combination" in df.columns:
            df = df[df["combination"].astype(str) == "0"]
        
        # Find 1:1 mappings (ICD-9 codes that map to exactly one ICD-10)
        counts = df.groupby(icd9_col)[icd10_col].nunique()
        one_to_one = set(counts[counts == 1].index)
        df_11 = df[df[icd9_col].isin(one_to_one)].drop_duplicates(subset=[icd9_col])
        
        mapping = dict(zip(df_11[icd9_col].str.strip(), df_11[icd10_col].str.strip()))
        result[label] = mapping
        print(f"  [GEM] {label}: {len(df)} total rows → {len(mapping)} 1:1 mappings")
    
    _GEM_CACHE = result
    return result


def _find_gem_dir() -> Optional[str]:
    """Try to find directory containing GEM CSV files."""
    import os
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data_files"),
        os.path.expanduser("~/.atop"),
        ".",
    ]
    for d in candidates:
        cm_path = os.path.join(d, "icd9toicd10cmgem.csv")
        if os.path.isfile(cm_path):
            return d
    return None


def _harmonize_icd9_to_icd10(token: str, gem_cm: Dict[str, str], gem_pcs: Dict[str, str]) -> str:
    """Convert an ICD-9 token to ICD-10 if a 1:1 mapping exists.
    
    Token format: C:9_XXXX → C:10_YYYY (uses CM mapping)
                  P:9_XXXX → P:10_YYYY (uses PCS mapping)
    If no mapping, returns original token unchanged.
    """
    m = re.match(r'^([CP]):9_(.+)$', token)
    if not m:
        return token
    
    stream = m.group(1)
    icd9_code = m.group(2).replace(".", "").strip()
    
    gem = gem_cm if stream == "C" else gem_pcs
    icd10_code = gem.get(icd9_code)
    if icd10_code:
        return f"{stream}:10_{icd10_code}"
    return token


def _should_exclude_icd(token: str, prefixes: List[str]) -> bool:
    """Check if a diagnosis/procedure token should be excluded by ICD prefix.
    
    Token format: C:10_Z2082 or C:9_V72 or C:Z2082
    Matches against ICD-10 codes only (after harmonization).
    """
    # Extract code from token
    m = re.match(r'^[CP]:(?:10_)?(.+)$', token)
    if not m:
        return False
    code = m.group(1)
    
    for prefix in prefixes:
        if code.startswith(prefix):
            return True
    return False


# ============================================================================
# Chronic condition ICD prefix lists (for chronic_filter option)
# ============================================================================

_CHRONIC_ICD10_PREFIXES = (
    # Endocrine: diabetes, thyroid, metabolic
    "E08", "E09", "E10", "E11", "E13",  # diabetes
    "E03", "E05",                         # thyroid
    "E66",                                # obesity
    "E78",                                # hyperlipidemia
    # Circulatory: hypertension, heart disease, cerebrovascular
    "I10", "I11", "I12", "I13",           # hypertension
    "I20", "I25",                         # chronic ischemic heart disease
    "I42", "I43", "I50",                  # cardiomyopathy, heart failure
    "I48",                                # atrial fibrillation
    "I65", "I66", "I67", "I69",           # cerebrovascular chronic
    "I70", "I71", "I73",                  # peripheral vascular
    "I83",                                # varicose veins
    # Respiratory: COPD, asthma
    "J41", "J42", "J43", "J44",           # COPD
    "J45",                                # asthma
    "J84",                                # interstitial lung disease
    # Kidney
    "N18", "N19",                         # CKD
    # Gastrointestinal (chronic)
    "K21",                                # GERD
    "K50", "K51",                         # Crohn's, ulcerative colitis
    # Sleep disorders (chronic)
    "G47",                                # sleep apnea, insomnia, narcolepsy
    # Mental/behavioral: chronic
    "F20", "F21", "F25",                  # schizophrenia
    "F30", "F31", "F32", "F33", "F34",    # mood disorders
    "F40", "F41",                         # anxiety
    "F10", "F11", "F12", "F13", "F14", "F15", "F17", "F19",  # substance use
    # Musculoskeletal
    "M05", "M06",                         # rheumatoid arthritis
    "M15", "M16", "M17", "M18", "M19",   # osteoarthritis
    "M80", "M81",                         # osteoporosis
    # Liver
    "K70", "K73", "K74", "K76",           # chronic liver disease
    "B18",                                # chronic hepatitis
    # Neurological
    "G20",                                # Parkinson's
    "G30",                                # Alzheimer's
    "G35",                                # MS
    "G40",                                # epilepsy
    "G43",                                # migraine
    # Cancer (history/ongoing)
    "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09",
    "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
    "C20", "C21", "C22", "C23", "C24", "C25", "C26",
    "C30", "C31", "C32", "C33", "C34",
    "C43", "C44", "C45", "C46", "C47", "C48", "C49",
    "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58",
    "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67", "C68",
    "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76",
    "C77", "C78", "C79", "C80",
    "C81", "C82", "C83", "C84", "C85", "C86", "C88",
    "C90", "C91", "C92", "C93", "C94", "C95", "C96",
    # HIV
    "B20",
    # Z-codes: history, status
    "Z85", "Z86", "Z87",                  # personal history
    "Z94", "Z95", "Z96",                  # transplant, implant status
    "Z79",                                # long-term (current) drug therapy
    "Z99",                                # dependence on machines
)

_CHRONIC_ICD9_PREFIXES = (
    # Diabetes
    "250",
    # Hypertension
    "401", "402", "403", "404", "405",
    # Hyperlipidemia
    "272",
    # Ischemic heart disease (chronic)
    "414",
    # Heart failure
    "428",
    # Atrial fibrillation
    "42731",
    # COPD/Asthma
    "491", "492", "493", "496",
    # CKD
    "585", "586",
    # Depression / mood
    "296", "300", "311",
    # Schizophrenia
    "295",
    # Substance use
    "303", "304", "305",
    # Rheumatic
    "714", "715",
    # Liver
    "571", "573",
    # Peripheral vascular
    "440", "443",
    # Obesity
    "278",
    # Cerebrovascular chronic
    "437", "438",
    # Parkinson, Alzheimer, epilepsy
    "332", "331", "345",
    # Cancer
    "140", "141", "142", "143", "144", "145", "146", "147", "148", "149",
    "150", "151", "152", "153", "154", "155", "156", "157", "158", "159",
    "160", "161", "162", "163", "164", "165",
    "170", "171", "172", "174", "175", "176",
    "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189",
    "190", "191", "192", "193", "194", "195",
    "200", "201", "202", "203", "204", "205", "206", "207", "208",
    # HIV
    "042",
    # V-codes history
    "V10", "V12", "V13",
)


def _is_chronic_token(token_str: str) -> bool:
    """Check if a diagnosis token represents a chronic condition."""
    # Token format: C:{version}_{icd_code}
    if not token_str.startswith("C:"):
        return False
    parts = token_str.split("_", 1)
    if len(parts) != 2:
        return False
    version_str = parts[0][2:]  # "9" or "10"
    icd_code = parts[1].upper().strip()

    if version_str == "10":
        for prefix in _CHRONIC_ICD10_PREFIXES:
            if icd_code.startswith(prefix):
                return True
    elif version_str == "9":
        for prefix in _CHRONIC_ICD9_PREFIXES:
            if icd_code.startswith(prefix):
                return True
    return False



def _precompute_visit_tokens(diagnoses: pd.DataFrame, procedures: pd.DataFrame,
                              prescriptions: pd.DataFrame,
                              max_drug_freq: float = 0.0,
                              drug_exclude_substrings: list = None,
                              token_types: str = "CPD",
                              harmonize_icd: bool = False,
                              icd_exclude_prefixes: list = None,
                              drug_mapping: str = ""):
    """
    Pre-compute token lists for EVERY hadm_id using vectorized pandas.
    Returns:
      hadm_tokens: dict mapping hadm_id (str) -> list of token strings.
      hadm_dx_seq: dict mapping hadm_id -> [(token_str, seq_num), ...]
                   Diagnoses with seq_num preserved for chronic filtering.
    Runs once over the entire dataset, no per-sample iteration needed.

    token_types: string of characters indicating which streams to include:
      C = conditions (diagnoses), P = procedures, D = drugs (prescriptions).
    If max_drug_freq > 0, drug tokens appearing in more than that fraction
    of all admissions are excluded before returning. This is a label-agnostic
    prevalence filter analogous to stopword removal in NLP.
    If harmonize_icd=True, ICD-9 codes are mapped to ICD-10 via CMS GEMs.
    If icd_exclude_prefixes is set, tokens matching those ICD prefixes are dropped.
    If drug_mapping is a path to an NDC-to-ingredient CSV (from build_ndc_rxnorm_map.py),
    drug tokens are mapped to RxNorm ingredient codes (D:RX_{rxcui}) instead of
    free-text drug names, grouping equivalent drugs under a single token.
    """
    import time
    t0 = time.time()

    token_types = token_types.upper()
    print(f"  [token_types] Including streams: {' '.join('C=conditions' if c=='C' else 'P=procedures' if c=='P' else 'D=drugs' for c in token_types)}")

    # Load GEM mapping if harmonization requested
    gem_cm, gem_pcs = {}, {}
    if harmonize_icd:
        gem_dir = _find_gem_dir()
        if not gem_dir:
            raise FileNotFoundError(
                "GEM CSV files not found. Place icd9toicd10cmgem.csv and "
                "icd9toicd10pcsgem.csv in atop/data_files/")
        gems = _load_gem_mapping(gem_dir)
        gem_cm = gems["cm"]
        gem_pcs = gems["pcs"]
        print(f"  [harmonize] CM: {len(gem_cm):,} mappings, PCS: {len(gem_pcs):,} mappings")

    hadm_tokens: Dict[str, List[str]] = {}
    hadm_dx_seq: Dict[str, List[Tuple[str, int]]] = {}

    # --- Diagnoses: C:{version}_{code} (version 9 or 10 only) ---
    if "C" in token_types and not diagnoses.empty:
        dx = diagnoses.copy()
        dx["hadm_id"] = dx["hadm_id"].astype(str)
        v = pd.to_numeric(dx["icd_version"], errors="coerce")
        dx["token"] = np.where(
            v.isin([9, 10]),
            "C:" + v.astype("Int64").astype(str) + "_" + dx["icd_code"].astype(str),
            "C:" + dx["icd_code"].astype(str),
        )

        # ICD-9 → ICD-10 harmonization
        if harmonize_icd and gem_cm:
            original_unique = dx["token"].nunique()
            dx["token"] = dx["token"].apply(lambda t: _harmonize_icd9_to_icd10(t, gem_cm, gem_pcs))
            harmonized_unique = dx["token"].nunique()
            n_merged = original_unique - harmonized_unique
            print(f"  [harmonize] Conditions: {original_unique} → {harmonized_unique} unique tokens "
                  f"({n_merged} ICD-9 codes merged into ICD-10)")

        # ICD prefix exclusion
        if icd_exclude_prefixes:
            mask = dx["token"].apply(lambda t: _should_exclude_icd(t, icd_exclude_prefixes))
            n_excl = mask.sum()
            excl_tokens = dx.loc[mask, "token"].nunique()
            print(f"  [icd_exclude] Conditions: dropped {n_excl:,} rows ({excl_tokens} unique tokens)")
            dx = dx[~mask]

        if "seq_num" in dx.columns:
            dx["seq_num_int"] = pd.to_numeric(dx["seq_num"], errors="coerce").fillna(99).astype(int)
            dx = dx.sort_values(["hadm_id", "seq_num_int"])
        else:
            dx["seq_num_int"] = 99
        dx = dx.drop_duplicates(subset=["hadm_id", "token"], keep="first")
        # Store tokens with seq_num for chronic filter
        for hid, grp in dx.groupby("hadm_id"):
            toks = grp["token"].tolist()
            seqs = grp["seq_num_int"].tolist()
            hadm_tokens.setdefault(hid, []).extend(toks)
            hadm_dx_seq[hid] = list(zip(toks, seqs))

    # --- Procedures: P:{version}_{code} (version 9 or 10 only) ---
    if "P" in token_types and not procedures.empty:
        px = procedures.copy()
        px["hadm_id"] = px["hadm_id"].astype(str)
        v = pd.to_numeric(px["icd_version"], errors="coerce")
        px["token"] = np.where(
            v.isin([9, 10]),
            "P:" + v.astype("Int64").astype(str) + "_" + px["icd_code"].astype(str),
            "P:" + px["icd_code"].astype(str),
        )

        # ICD-9 → ICD-10 harmonization for procedures
        if harmonize_icd and gem_pcs:
            original_unique = px["token"].nunique()
            px["token"] = px["token"].apply(lambda t: _harmonize_icd9_to_icd10(t, gem_cm, gem_pcs))
            harmonized_unique = px["token"].nunique()
            n_merged = original_unique - harmonized_unique
            print(f"  [harmonize] Procedures: {original_unique} → {harmonized_unique} unique tokens "
                  f"({n_merged} ICD-9 codes merged into ICD-10)")

        # ICD prefix exclusion for procedures
        if icd_exclude_prefixes:
            mask = px["token"].apply(lambda t: _should_exclude_icd(t, icd_exclude_prefixes))
            n_excl = mask.sum()
            if n_excl > 0:
                print(f"  [icd_exclude] Procedures: dropped {n_excl:,} rows")
            px = px[~mask]

        if "seq_num" in px.columns:
            px = px.sort_values(["hadm_id", "seq_num"])
        px = px.drop_duplicates(subset=["hadm_id", "token"], keep="first")
        px_map = px.groupby("hadm_id")["token"].agg(list).to_dict()
        for hid, toks in px_map.items():
            hadm_tokens.setdefault(hid, []).extend(toks)

    # --- Prescriptions: D:{ingredient}, sorted by starttime ---
    if "D" in token_types and not prescriptions.empty:
        rx = prescriptions.copy()
        rx["hadm_id"] = rx["hadm_id"].astype(str)

        # Load drug mapping if provided (supports both NDC→ingredient and drugname→ingredient formats)
        ndc_map = {}    # normalized_ndc → ingredient token
        name_map = {}   # lowercase drug name → ingredient token
        if drug_mapping and os.path.exists(drug_mapping):
            df_map = pd.read_csv(drug_mapping, dtype=str).fillna("")
            for _, mr in df_map.iterrows():
                # Determine token: prefer RxCUI if available, else use normalized name
                ing_rxcui = mr.get("ingredient_rxcui", "")
                ing_name = mr.get("ingredient_name", "") or mr.get("ingredient_normalized", "")
                if not ing_name:
                    continue
                if ing_rxcui:
                    tok = f"D:RX_{ing_rxcui}"
                else:
                    # Use normalized ingredient name as token
                    tok = f"D:{ing_name}"
                # Map by NDC if available
                if mr.get("ndc", ""):
                    ndc_map[mr["ndc"]] = tok
                # Map by drug name if available
                if mr.get("drug_name", ""):
                    name_map[mr["drug_name"].lower().strip()] = tok
            n_unique_toks = len(set(ndc_map.values()) | set(name_map.values()))
            print(f"  [drug mapping] Loaded {len(ndc_map)} NDC + {len(name_map)} "
                  f"name mappings → {n_unique_toks} unique ingredients")

        def _normalize_ndc(val):
            if pd.isna(val):
                return ""
            s = str(val).strip()
            # Strip float decimal suffix (e.g., '51079007320.0' → '51079007320')
            if "." in s:
                s = s.split(".")[0]
            s = s.replace("-", "").replace(" ", "")
            if not s or s == "0":
                return ""
            if len(s) < 11:
                s = s.zfill(11)
            if len(s) > 11:
                s = s[:11]
            return s

        if ndc_map or name_map:
            # Use RxNorm ingredient mapping
            has_ndc = "ndc" in rx.columns
            has_generic = "drug_name_generic" in rx.columns
            has_drug = "drug" in rx.columns

            def _map_row(row):
                # Try NDC first
                if has_ndc and pd.notna(row.get("ndc")):
                    ndc_norm = _normalize_ndc(row["ndc"])
                    if ndc_norm in ndc_map:
                        return ndc_map[ndc_norm]
                # Fall back to drug name
                for col in ["drug_name_generic", "drug"]:
                    if col in row.index and pd.notna(row.get(col)):
                        dname = str(row[col]).strip().lower()
                        if dname in name_map:
                            return name_map[dname]
                return None

            rx["token"] = rx.apply(_map_row, axis=1)
            n_before = len(rx)
            rx = rx.dropna(subset=["token"])
            n_unmapped = n_before - len(rx)
            if n_unmapped > 0:
                print(f"  [drug mapping] {n_unmapped} prescription rows unmapped "
                      f"({100*n_unmapped/n_before:.1f}%)")
        else:
            # Original behavior: use drug name as token
            drug_col = None
            for candidate in ["drug", "drug_name_generic", "drug_name_poe", "ndc"]:
                if candidate in rx.columns:
                    drug_col = candidate
                    break
            if drug_col:
                rx = rx.dropna(subset=[drug_col])
                rx["token"] = "D:" + rx[drug_col].astype(str)
            else:
                rx["token"] = None

        rx = rx.dropna(subset=["token"])
        if "starttime" in rx.columns:
            rx = rx.sort_values(["hadm_id", "starttime"], na_position="last")
        rx = rx.drop_duplicates(subset=["hadm_id", "token"], keep="first")

        # --- Prevalence-based drug filtering (label-agnostic) ---
        if max_drug_freq > 0:
            n_total_admissions = len(set(rx["hadm_id"]))
            drug_doc_freq = rx.groupby("token")["hadm_id"].nunique()
            drug_frac = drug_doc_freq / n_total_admissions
            excluded = set(drug_frac[drug_frac > max_drug_freq].index)
            if excluded:
                print(f"  [drug filter] Excluding {len(excluded)} drug tokens with "
                      f"admission freq > {max_drug_freq:.0%}:")
                for d in sorted(excluded):
                    freq = drug_frac[d]
                    print(f"    {d} ({freq:.1%} of admissions)")
                rx = rx[~rx["token"].isin(excluded)]
            else:
                print(f"  [drug filter] No drug tokens exceed {max_drug_freq:.0%} threshold")

        # --- Substring-based drug filtering ---
        if drug_exclude_substrings:
            drug_names = rx["token"].str[2:]  # strip "D:" prefix for matching
            mask = pd.Series(False, index=rx.index)
            for sub in drug_exclude_substrings:
                mask |= drug_names.str.contains(sub, case=False, na=False)
            excluded_sub = set(rx.loc[mask, "token"].unique())
            if excluded_sub:
                print(f"  [drug filter] Excluding {len(excluded_sub)} drug tokens by substring match:")
                for d in sorted(excluded_sub):
                    print(f"    {d}")
                rx = rx[~rx["token"].isin(excluded_sub)]

        # --- RxCUI-based exclusion (for RxNorm-mapped tokens) ---
        from atop.config import DEFAULT_DRUG_EXCLUDE_RXCUIS
        rxcui_exclude = set(DEFAULT_DRUG_EXCLUDE_RXCUIS)
        rxcui_found = rxcui_exclude & set(rx["token"].unique())
        if rxcui_found:
            print(f"  [drug filter] Excluding {len(rxcui_found)} non-informative RxCUI tokens:")
            for d in sorted(rxcui_found):
                print(f"    {d}")
            rx = rx[~rx["token"].isin(rxcui_found)]

        rx_map = rx.groupby("hadm_id")["token"].agg(list).to_dict()
        for hid, toks in rx_map.items():
            hadm_tokens.setdefault(hid, []).extend(toks)

    print(f"  Pre-computed tokens for {len(hadm_tokens):,} admissions ({time.time()-t0:.1f}s)")
    return hadm_tokens, hadm_dx_seq



def _preprocessing_cache_key(
    token_types, max_drug_freq, drug_exclude_substrings, chronic_filter,
    first_occurrence_only, first_occurrence_drugs_only, harmonize_icd,
    icd_exclude_prefixes, drug_mapping,
    max_visits, one_per_patient, n_admissions,
) -> tuple:
    """Hash preprocessing parameters into a cache key. Returns (hash, params_dict)."""
    import hashlib, json
    params = {
        "token_types": token_types,
        "max_drug_freq": max_drug_freq,
        "drug_exclude": sorted(drug_exclude_substrings) if drug_exclude_substrings else [],
        "chronic_filter": chronic_filter,
        "first_occurrence_only": first_occurrence_only,
        "first_occurrence_drugs_only": first_occurrence_drugs_only,
        "harmonize_icd": harmonize_icd,
        "icd_exclude": sorted(icd_exclude_prefixes) if icd_exclude_prefixes else [],
        "drug_mapping": os.path.basename(drug_mapping) if drug_mapping else "",
        "max_visits": max_visits,
        "one_per_patient": one_per_patient,
        "n_admissions": n_admissions,
    }
    h = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    return h, params


def _update_cache_manifest(cache_dir: str, cache_hash: str, params: dict,
                           n_samples: int, cache_path: str):
    """Append or update an entry in the cache manifest CSV."""
    import csv
    from datetime import datetime
    manifest_path = os.path.join(cache_dir, "cache_manifest.csv")

    # Read existing entries
    entries = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries[row["hash"]] = row

    # Add/update this entry
    entries[cache_hash] = {
        "hash": cache_hash,
        "filename": os.path.basename(cache_path),
        "n_samples": n_samples,
        "created": datetime.now().isoformat(timespec="seconds"),
        "token_types": params["token_types"],
        "max_drug_freq": params["max_drug_freq"],
        "chronic_filter": params["chronic_filter"],
        "first_occurrence_only": params["first_occurrence_only"],
        "first_occurrence_drugs_only": params["first_occurrence_drugs_only"],
        "harmonize_icd": params["harmonize_icd"],
        "drug_mapping": params["drug_mapping"],
        "max_visits": params["max_visits"],
        "n_admissions": params["n_admissions"],
    }

    # Write back
    fieldnames = ["hash", "filename", "n_samples", "created",
                  "token_types", "max_drug_freq", "chronic_filter",
                  "first_occurrence_only", "first_occurrence_drugs_only",
                  "harmonize_icd", "drug_mapping", "max_visits", "n_admissions"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in sorted(entries.values(), key=lambda x: x["created"]):
            writer.writerow(entry)


def build_patient_sequences(
    adm_labels: pd.DataFrame,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
    prescriptions: pd.DataFrame,
    max_visits: int = 20,
    one_per_patient: bool = False,
    max_drug_freq: float = 0.0,
    drug_exclude_substrings: list = None,
    token_types: str = "CPD",
    chronic_filter: bool = False,
    first_occurrence_only: bool = False,
    first_occurrence_drugs_only: bool = False,
    harmonize_icd: bool = False,
    icd_exclude_prefixes: list = None,
    drug_mapping: str = "",
    cache_dir: str = "",
) -> List[Dict[str, Any]]:
    """
    Build one sample per patient's index admission.
    Two-phase approach for speed:
      Phase 1: Pre-compute token list per hadm_id (vectorized, runs once).
      Phase 2: Assemble per-sample sequences by referencing pre-computed tokens.

    If cache_dir is set, caches results as a pickle keyed by parameter hash.
    Subsequent calls with identical parameters load from cache (~seconds vs ~minutes).

    If chronic_filter=True: For chronic diagnosis codes (per AHRQ CCIR),
    only include at a visit if (a) seq_num==1 (primary diagnosis), or
    (b) it's the first visit where this code appears for the patient.
    Acute codes are always included regardless of repetition.
    If harmonize_icd=True, ICD-9 codes are mapped to ICD-10 via CMS GEMs.
    If icd_exclude_prefixes is set, tokens matching those ICD prefixes are dropped.
    """
    import time, pickle

    # ── Check cache ──────────────────────────────────────────────────────
    cache_path = None
    cache_params = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_hash, cache_params = _preprocessing_cache_key(
            token_types, max_drug_freq, drug_exclude_substrings,
            chronic_filter, first_occurrence_only, first_occurrence_drugs_only,
            harmonize_icd, icd_exclude_prefixes, drug_mapping,
            max_visits, one_per_patient, len(adm_labels))
        cache_path = os.path.join(cache_dir, f"samples_{cache_hash}.pkl")
        if os.path.exists(cache_path):
            print(f"[cache] Loading preprocessed samples from {cache_path}")
            t0 = time.time()
            with open(cache_path, "rb") as f:
                samples = pickle.load(f)
            print(f"[cache] Loaded {len(samples):,} samples in {time.time()-t0:.1f}s "
                  f"(hash={cache_hash})")
            return samples

    print("[data] Building patient visit-block sequences...")

    # Phase 1: pre-compute tokens per hadm_id
    hadm_tokens, hadm_dx_seq = _precompute_visit_tokens(
        diagnoses, procedures, prescriptions,
        max_drug_freq=max_drug_freq,
        drug_exclude_substrings=drug_exclude_substrings,
        token_types=token_types,
        harmonize_icd=harmonize_icd,
        icd_exclude_prefixes=icd_exclude_prefixes,
        drug_mapping=drug_mapping)

    if chronic_filter:
        print("  [chronic_filter] Enabled: chronic dx kept only at first occurrence or seq_num=1")

    # Sort all admissions per patient chronologically
    adm_labels = adm_labels.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    # Phase 2: assemble samples
    t0 = time.time()
    samples = []
    n_chronic_filtered = 0
    n_chronic_total = 0

    for pid, grp in adm_labels.groupby("subject_id"):
        grp = grp.sort_values("admittime").reset_index(drop=True)
        n_visits = len(grp)
        hadm_ids = grp["hadm_id"].tolist()
        labels = grp["readmit_30d"].tolist()
        admittimes = grp["admittime"].tolist()

        # Pre-fetch token lists for all this patient's admissions
        visit_token_lists = [hadm_tokens.get(hid, []) for hid in hadm_ids]

        # If chronic filtering: compute filtered token lists per visit
        if chronic_filter and "C" in token_types.upper():
            seen_chronic_codes = set()
            filtered_visit_tokens = []
            for vi, hid in enumerate(hadm_ids):
                dx_seq = hadm_dx_seq.get(hid, [])
                # Determine which dx tokens to keep vs filter
                keep_dx = set()
                for tok, seq in dx_seq:
                    is_chronic = _is_chronic_token(tok)
                    if is_chronic:
                        n_chronic_total += 1
                        if seq == 1 or tok not in seen_chronic_codes:
                            # Keep: primary diagnosis or first occurrence
                            keep_dx.add(tok)
                        else:
                            # Filter: chronic carry-forward
                            n_chronic_filtered += 1
                    else:
                        keep_dx.add(tok)
                    seen_chronic_codes.add(tok) if is_chronic else None

                # Rebuild token list: keep non-dx tokens + filtered dx
                orig_tokens = visit_token_lists[vi]
                new_tokens = []
                for t in orig_tokens:
                    if t.startswith("C:"):
                        if t in keep_dx:
                            new_tokens.append(t)
                    else:
                        new_tokens.append(t)
                filtered_visit_tokens.append(new_tokens)
            visit_token_lists = filtered_visit_tokens

        # First-occurrence filter: keep each code only in the earliest
        # admission where it appears. Removes chronic carry-forward across
        # ALL token types (C, P, D), not just chronic diagnoses.
        if first_occurrence_only:
            seen_tokens = set()
            fo_filtered = []
            for vi, toks in enumerate(visit_token_lists):
                new_toks = []
                for t in toks:
                    if t not in seen_tokens:
                        new_toks.append(t)
                        seen_tokens.add(t)
                fo_filtered.append(new_toks)
            visit_token_lists = fo_filtered
        elif first_occurrence_drugs_only:
            # Apply first-occurrence filtering only to drug tokens (D: prefix).
            # Conditions and procedures are kept at every visit they appear.
            seen_drugs = set()
            fo_filtered = []
            for vi, toks in enumerate(visit_token_lists):
                new_toks = []
                for t in toks:
                    if t.startswith("D:"):
                        if t not in seen_drugs:
                            new_toks.append(t)
                            seen_drugs.add(t)
                    else:
                        new_toks.append(t)
                fo_filtered.append(new_toks)
            visit_token_lists = fo_filtered

        if one_per_patient:
            index_positions = [n_visits - 2] if n_visits >= 2 else [0]
        else:
            index_positions = range(n_visits)

        for idx_pos in index_positions:
            start = max(0, idx_pos - max_visits + 1)

            flat_tokens = ["[CLS]"]
            token_visit_idx = [None]
            n_blocks = 0
            visit_hadm_ids_used = []

            for vi_offset, vi_abs in enumerate(range(start, idx_pos + 1)):
                vtokens = visit_token_lists[vi_abs]
                flat_tokens.append("[VISIT]")
                token_visit_idx.append(vi_offset)
                for t in vtokens:
                    flat_tokens.append(t)
                    token_visit_idx.append(vi_offset)
                n_blocks += 1
                visit_hadm_ids_used.append(hadm_ids[vi_abs])

            samples.append({
                "patient_id": pid,
                "index_hadm_id": hadm_ids[idx_pos],
                "readmit_30d": int(labels[idx_pos]),
                "n_visits": n_blocks,
                "visit_hadm_ids": visit_hadm_ids_used,
                "flat_tokens": flat_tokens,
                "token_visit_idx": token_visit_idx,
                "admittime": admittimes[idx_pos],
            })

    elapsed = time.time() - t0
    print(f"  Total samples: {len(samples):,} (assembled in {elapsed:.1f}s)")
    n_multi = sum(1 for s in samples if s["n_visits"] > 1)
    print(f"  Multi-visit samples (>=2 visits): {n_multi:,} ({100*n_multi/max(len(samples),1):.1f}%)")
    if chronic_filter and n_chronic_total > 0:
        print(f"  [chronic_filter] Filtered {n_chronic_filtered:,}/{n_chronic_total:,} "
              f"chronic dx tokens ({100*n_chronic_filtered/n_chronic_total:.1f}% removed)")

    # ── Save cache ───────────────────────────────────────────────────────
    if cache_path:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"[cache] Saved {len(samples):,} samples to {cache_path} ({size_mb:.0f} MB)")
            _update_cache_manifest(cache_dir, cache_hash, cache_params,
                                   len(samples), cache_path)
        except Exception as e:
            print(f"[cache] Warning: could not save cache: {e}")

    return samples



# ============================================================================
# VOCABULARY
# ============================================================================
