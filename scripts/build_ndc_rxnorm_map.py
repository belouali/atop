#!/usr/bin/env python3
"""
Build NDC/drug-name → RxNorm ingredient mapping for MIMIC-IV prescriptions.

Two-stage strategy (both proven to work with RxNorm API):
  Stage 1: NDC (zero-padded 11-digit) → RxCUI → Ingredient via /related.json?tty=IN
  Stage 2: Drug name → Ingredient via /approximateTerm.json (for NDC misses + no-NDC rows)

Outputs:
  ndc_to_ingredient.csv — mapping table with columns:
      ndc, drug_name, rxcui, ingredient_rxcui, ingredient_name, source

Usage:
  python scripts/build_ndc_rxnorm_map.py \
      --mimic_dir /data/mimiciv/3.1/hosp \
      --out ndc_to_ingredient.csv \
      --rate_limit 15
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"


# ── Helpers ────────────────────────────────────────────────────────────

def normalize_ndc(ndc_str) -> str:
    """Normalize NDC to 11-digit HIPAA format.
    MIMIC stores NDCs as floats: '51079007320.0' → '51079007320'."""
    if pd.isna(ndc_str):
        return ""
    s = str(ndc_str).strip()
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


def ndc_to_rxcui(ndc: str, session: requests.Session) -> Optional[str]:
    """NDC (11-digit) → RxCUI. Proven to work with zero-padded NDCs."""
    try:
        r = session.get(f"{RXNORM_BASE}/rxcui.json",
                        params={"idtype": "NDC", "id": ndc}, timeout=10)
        r.raise_for_status()
        ids = r.json().get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None
    except Exception:
        return None


def rxcui_to_ingredient(rxcui: str, session: requests.Session
                        ) -> Optional[Tuple[str, str]]:
    """RxCUI → ingredient (rxcui, name).
    Key fix: tty parameter passed as 'IN' not 'IN+MIN+PIN' to avoid
    requests URL-encoding '+' as '%2B'."""
    try:
        # Try IN (single ingredient) first
        r = session.get(f"{RXNORM_BASE}/rxcui/{rxcui}/related.json",
                        params={"tty": "IN"}, timeout=10)
        r.raise_for_status()
        for grp in r.json().get("relatedGroup", {}).get("conceptGroup", []):
            for prop in grp.get("conceptProperties", []):
                return (prop["rxcui"], prop["name"])

        # Try MIN (multi-ingredient)
        r = session.get(f"{RXNORM_BASE}/rxcui/{rxcui}/related.json",
                        params={"tty": "MIN"}, timeout=10)
        r.raise_for_status()
        for grp in r.json().get("relatedGroup", {}).get("conceptGroup", []):
            for prop in grp.get("conceptProperties", []):
                return (prop["rxcui"], prop["name"])

        # Check if rxcui itself is an ingredient
        r = session.get(f"{RXNORM_BASE}/rxcui/{rxcui}/properties.json", timeout=10)
        r.raise_for_status()
        props = r.json().get("properties", {})
        if props.get("tty") in ("IN", "MIN", "PIN"):
            return (rxcui, props.get("name", ""))

        return None
    except Exception:
        return None


def drugname_to_ingredient(name: str, session: requests.Session
                           ) -> Optional[Tuple[str, str]]:
    """Drug name → ingredient via /approximateTerm. Proven 5/5 in testing."""
    try:
        r = session.get(f"{RXNORM_BASE}/approximateTerm.json",
                        params={"term": name, "maxEntries": 1}, timeout=10)
        r.raise_for_status()
        candidates = r.json().get("approximateGroup", {}).get("candidate", [])
        if not candidates:
            return None
        rxcui = candidates[0].get("rxcui")
        if not rxcui:
            return None

        # Get ingredient from this RxCUI
        result = rxcui_to_ingredient(rxcui, session)
        if result:
            return result

        return None
    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build NDC/drug-name → RxNorm ingredient mapping for MIMIC-IV")
    parser.add_argument("--mimic_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="ndc_to_ingredient.csv")
    parser.add_argument("--rate_limit", type=int, default=15,
                        help="Max API requests per second")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from partial CSV")
    args = parser.parse_args()

    delay = 1.0 / max(args.rate_limit, 1)

    # Load prescriptions
    for ext in [".csv.gz", ".csv"]:
        path = os.path.join(args.mimic_dir, f"prescriptions{ext}")
        if os.path.exists(path):
            print(f"Loading {path}...")
            rx = pd.read_csv(path, low_memory=False)
            break
    else:
        print("ERROR: prescriptions.csv(.gz) not found"); sys.exit(1)

    print(f"  {len(rx):,} prescription rows")

    has_ndc = "ndc" in rx.columns
    has_drug = "drug" in rx.columns
    print(f"  Columns: ndc={has_ndc}, drug={has_drug}")

    # Build unique NDC → drug name lookup
    ndc_drugname = {}
    unique_ndcs = set()
    if has_ndc:
        for _, r in rx[["ndc", "drug"]].drop_duplicates().iterrows():
            ndc = normalize_ndc(r.get("ndc"))
            if ndc:
                unique_ndcs.add(ndc)
                if has_drug and pd.notna(r.get("drug")):
                    ndc_drugname[ndc] = str(r["drug"]).strip()
        print(f"  Unique valid NDCs: {len(unique_ndcs):,}")

    # Drug names for rows without NDC
    unique_drugnames = set()
    if has_drug:
        all_names = rx["drug"].dropna().str.strip().str.lower().unique()
        unique_drugnames = set(all_names)
        print(f"  Unique drug names: {len(unique_drugnames):,}")

    # Load resume
    already = {}
    if args.resume and os.path.exists(args.resume):
        prev = pd.read_csv(args.resume, dtype=str).fillna("")
        for _, r in prev.iterrows():
            key = r.get("ndc", "") or r.get("drug_name", "")
            if key:
                already[key] = r.to_dict()
        print(f"  Resuming: {len(already)} already mapped")

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    results = list(already.values())

    # ── Stage 1: NDC → RxCUI → Ingredient ─────────────────────────────
    if unique_ndcs:
        todo = sorted(ndc for ndc in unique_ndcs if ndc not in already)
        print(f"\n{'='*60}")
        print(f"Stage 1: NDC → Ingredient ({len(todo)} to process)")
        print(f"{'='*60}")

        n_mapped = 0
        n_failed = 0

        for i, ndc in enumerate(todo):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"\r  [{i+1}/{len(todo)}] mapped={n_mapped} failed={n_failed}",
                      end="", flush=True)

            rxcui = ndc_to_rxcui(ndc, session)
            time.sleep(delay)

            if rxcui:
                ing = rxcui_to_ingredient(rxcui, session)
                time.sleep(delay)

                if ing:
                    results.append({
                        "ndc": ndc,
                        "drug_name": ndc_drugname.get(ndc, "").lower(),
                        "rxcui": rxcui,
                        "ingredient_rxcui": ing[0],
                        "ingredient_name": ing[1],
                        "source": "ndc_api",
                    })
                    n_mapped += 1
                else:
                    n_failed += 1
            else:
                n_failed += 1

            if (i + 1) % 500 == 0:
                _save_partial(results, args.out)

        print(f"\n  Stage 1: {n_mapped} mapped, {n_failed} failed")

    # ── Stage 2: Drug name fallback for NDC misses ─────────────────────
    # Collect drug names that need API lookup
    names_to_try = set()

    # NDCs that failed stage 1 → try their drug names
    mapped_ndcs = {r["ndc"] for r in results if r.get("ndc")}
    if has_drug:
        for ndc in unique_ndcs:
            if ndc not in mapped_ndcs and ndc in ndc_drugname:
                names_to_try.add(ndc_drugname[ndc].lower())

    # Drug names from rows with no valid NDC
    if has_drug:
        rx["_ndc_norm"] = rx["ndc"].apply(normalize_ndc)
        no_ndc = rx[rx["_ndc_norm"] == ""]
        if not no_ndc.empty:
            no_ndc_names = no_ndc["drug"].dropna().str.strip().str.lower().unique()
            names_to_try.update(no_ndc_names)

    # Remove names already resolved
    already_names = {r.get("drug_name", "").lower()
                     for r in results if r.get("ingredient_rxcui")}
    names_to_try -= already_names
    names_to_try -= {""}

    if names_to_try:
        todo_names = sorted(names_to_try)
        print(f"\n{'='*60}")
        print(f"Stage 2: Drug name → Ingredient ({len(todo_names)} names)")
        print(f"{'='*60}")

        n_mapped = 0
        # Cache: first-word → ingredient, so we don't re-query "metoprolol" 50 times
        word_cache: Dict[str, Optional[Tuple[str, str]]] = {}

        for i, dname in enumerate(todo_names):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"\r  [{i+1}/{len(todo_names)}] mapped={n_mapped}",
                      end="", flush=True)

            # Extract first word (usually the ingredient)
            first_word = dname.split()[0] if dname.split() else dname

            if first_word in word_cache:
                ing = word_cache[first_word]
            else:
                ing = drugname_to_ingredient(first_word, session)
                time.sleep(delay)
                word_cache[first_word] = ing

            if ing:
                results.append({
                    "ndc": "",
                    "drug_name": dname,
                    "rxcui": "",
                    "ingredient_rxcui": ing[0],
                    "ingredient_name": ing[1],
                    "source": "drugname_api",
                })
                n_mapped += 1

            if (i + 1) % 500 == 0:
                _save_partial(results, args.out)

        print(f"\n  Stage 2: {n_mapped}/{len(todo_names)} mapped")

    # ── Save final ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    if df.empty:
        print("No mappings produced!")
        return

    df = df.drop_duplicates(subset=["ndc", "drug_name"], keep="first")
    df.to_csv(args.out, index=False)

    n_with_rxcui = (df["ingredient_rxcui"].astype(str) != "").sum()
    n_ingredients = df["ingredient_rxcui"].nunique()
    print(f"\nSaved {len(df)} mappings → {args.out}")
    print(f"  With ingredient RxCUI: {n_with_rxcui}")
    print(f"  Unique ingredients: {n_ingredients}")

    # Coverage
    if has_drug:
        rx["_drug_lower"] = rx["drug"].str.strip().str.lower()
        mapped_drug_set = set(df[df["drug_name"] != ""]["drug_name"].str.lower())
        mapped_ndc_set = set(df[df["ndc"] != ""]["ndc"])
        if "_ndc_norm" not in rx.columns:
            rx["_ndc_norm"] = rx["ndc"].apply(normalize_ndc)
        covered = rx["_drug_lower"].isin(mapped_drug_set) | rx["_ndc_norm"].isin(mapped_ndc_set)
        print(f"  Row coverage: {covered.sum():,}/{len(rx):,} ({100*covered.mean():.1f}%)")


def _save_partial(results, out_path):
    partial = out_path.replace(".csv", "_partial.csv")
    pd.DataFrame(results).to_csv(partial, index=False)
    print(f" [saved {len(results)} to {partial}]", end="")


if __name__ == "__main__":
    main()
