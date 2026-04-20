#!/usr/bin/env python3
"""Replot figures from pre-computed CSVs — no data loading, no model, no mining.

Usage:
    python scripts/replot.py --run_dir runs/full_CPD_v3 --figures fig3 --n_show 25
    python scripts/replot.py --run_dir runs/full_CPD_v3 --figures fig3,fig4,fig5 --n_show 20

Reads from {run_dir}/explain/csv/ and writes to {run_dir}/explain/main/ and supp/.
"""

import argparse
import os
import sys
import json
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Module-level name maps (loaded once, used everywhere) ─────────────
_SHORT_NAME_MAP = {}
_DRUG_NAMES = {}
_ICD_TITLES = {}
_MAPPING_LOG = []  # list of (raw_token, resolved_name) pairs


def _load_short_names_csv():
    """Load short_names.csv directly — no atop import needed."""
    global _SHORT_NAME_MAP
    sn_path = os.path.join(os.path.dirname(__file__), "short_names.csv")
    if os.path.exists(sn_path):
        with open(sn_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    _SHORT_NAME_MAP[row[0].strip()] = row[1].strip()


def _load_drug_names(run_dir):
    """Load drug RxCUI→name mapping from run config."""
    global _DRUG_NAMES
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"  [drugs] No config.json at {config_path}")
        return
    with open(config_path) as f:
        cfg = json.load(f)
    drug_mapping = cfg.get("drug_mapping", "")
    
    # Try the path from config
    if drug_mapping and os.path.exists(drug_mapping):
        _read_drug_csv(drug_mapping)
        return
    
    # Fallback: search common locations
    candidates = [
        os.path.join(run_dir, "ndc_to_ingredient.csv"),
        os.path.join(os.path.dirname(run_dir), "data", "ndc_to_ingredient.csv"),
    ]
    if drug_mapping:
        # Try same filename in run_dir
        candidates.insert(0, os.path.join(run_dir, os.path.basename(drug_mapping)))
        # Try relative to mimic_dir from config
        mimic_dir = cfg.get("mimic_dir", "")
        if mimic_dir:
            candidates.append(os.path.join(mimic_dir, os.path.basename(drug_mapping)))
            candidates.append(os.path.join(mimic_dir, "ndc_to_ingredient.csv"))
    
    for c in candidates:
        if os.path.exists(c):
            _read_drug_csv(c)
            return
    
    print(f"  [drugs] Drug mapping not found. Config path: {drug_mapping}")
    print(f"  [drugs] Searched: {candidates[:3]}")


def _read_drug_csv(path):
    """Read drug mapping CSV and populate _DRUG_NAMES."""
    global _DRUG_NAMES
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    
    # The vocab uses ingredient_rxcui as the token ID: D:RX_{ingredient_rxcui}
    # Map ingredient_rxcui → ingredient_name
    if "ingredient_rxcui" in df.columns and "ingredient_name" in df.columns:
        for rxcui, name in df.dropna(subset=["ingredient_rxcui", "ingredient_name"]).groupby("ingredient_rxcui")["ingredient_name"].first().items():
            _DRUG_NAMES[f"D:RX_{int(rxcui)}"] = str(name)
    elif "rxcui" in df.columns and "ingredient" in df.columns:
        for _, r in df.dropna(subset=["rxcui", "ingredient"]).iterrows():
            _DRUG_NAMES[f"D:RX_{int(r['rxcui'])}"] = str(r["ingredient"])
    else:
        print(f"  [drugs] Unrecognized columns: {df.columns.tolist()}")
        return
    print(f"  [drugs] Loaded {len(_DRUG_NAMES)} drug names from {os.path.basename(path)}")


def _load_icd_titles(run_dir):
    """Load ICD code → title mapping as fallback for unresolved codes."""
    global _ICD_TITLES
    candidates = [
        os.path.join(run_dir, "icd_titles.json"),
        os.path.join(run_dir, "explain", "icd_titles.json"),
        os.path.join(run_dir, "explain", "figures", "icd_titles.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate) as f:
                _ICD_TITLES = json.load(f)
            print(f"  [icd] Loaded {len(_ICD_TITLES)} ICD titles from {os.path.basename(os.path.dirname(candidate))}/")
            return
    # Not found — try to build from token CSV
    tok_csv_candidates = [
        os.path.join(run_dir, "explain", "figures", "csv", "fig3_token_importance_test.csv"),
    ]
    for tc in tok_csv_candidates:
        if os.path.exists(tc):
            import re
            df = pd.read_csv(tc)
            if "token" in df.columns and "token_readable" in df.columns:
                for _, r in df.dropna(subset=["token", "token_readable"]).iterrows():
                    tok = str(r["token"])
                    readable = str(r["token_readable"])
                    m = re.match(r'^[CPD]:(?:\d+_)?(.+)$', tok)
                    if m:
                        code = m.group(1)
                        # Extract description from "C:F329 (MDD, single ep.)"
                        m2 = re.match(r'^[CPD]:[A-Za-z0-9_.]+ \((.+?)[\)]*$', readable)
                        if m2:
                            _ICD_TITLES[code] = m2.group(1).rstrip(')')
                print(f"  [icd] Built {len(_ICD_TITLES)} titles from token CSV")
                return
    print(f"  [icd] No ICD titles found")


def parse_args():
    p = argparse.ArgumentParser(description="Replot AToP figures from CSVs")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="",
                   help="Override output dir (default: {run_dir}/explain)")
    p.add_argument("--figures", type=str, default="fig3",
                   help="Comma-separated: fig3, fig4, fig5")
    p.add_argument("--n_show", type=int, default=15)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--jaccard_threshold", type=float, default=0.5,
                   help="Jaccard threshold for cluster supplement (default: 0.5)")
    p.add_argument("--short_names", action="store_true",
                   help="Use short clinical names without codes")
    p.add_argument("--panel_c_split", action="store_true",
                   help="Split Panel C into risk/protective sub-panels (for fig3_clean)")
    p.add_argument("--fontsize", type=int, default=13,
                   help="Font size for y-axis labels (default: 13)")
    p.add_argument("--show_n", action="store_true",
                   help="Show (n=XXX) carrier counts on Panel B and C labels")
    p.add_argument("--panel_c_mode", default="balanced",
                   choices=["balanced", "top"],
                   help="Panel C selection: 'balanced' = N/2 risk + N/2 protective, 'top' = top N by |IG|")
    p.add_argument("--fig_width", type=int, default=54,
                   help="Figure width in inches (default: 54)")
    p.add_argument("--bar_squeeze", type=float, default=0.55,
                   help="Bar plot width as fraction of panel (default: 0.55, lower=more text space)")
    p.add_argument("--population_avg", action="store_true",
                   help="Use population-averaged IG (0 for absent patients) instead of carrier-averaged")
    return p.parse_args()


def _build_icd_titles(df_tok):
    """Build icd_titles dict from token CSV: token -> readable name."""
    titles = {}
    for _, r in df_tok.iterrows():
        tok = r["token"]
        readable = r["token_readable"]
        titles[tok] = readable
    return titles


# Direct token → figure label mapping (AUTHORITATIVE)
TOKEN_LABELS = {
    "C:10_R079": "Chest pain",
    "C:10_R45851": "SI",
    "C:10_F329": "MDD (single)",
    "C:10_F332": "MDD (recurrent severe)",
    "C:10_F322": "MDD (severe)",
    "C:10_F29": "Psychosis",
    "C:10_F39": "Mood disorder",
    "C:10_F319": "Bipolar disorder",
    "C:10_F3110": "Bipolar (mania)",
    "C:10_F4310": "PTSD",
    "C:10_N179": "AKI",
    "C:10_D630": "Cancer anemia",
    "C:10_G893": "Cancer pain",
    "C:10_I10": "HTN",
    "C:10_K219": "GERD",
    "C:10_Z370": "Live birth",
    "C:10_R55": "Syncope",
    "C:10_R188": "Ascites",
    "C:10_O001": "Ectopic pregnancy",
    "C:10_J0390": "Tonsillitis",
    "C:10_J36": "Peritonsillar abscess",
    "C:10_F10129": "Alcohol intox",
    "C:10_F1010": "Alcohol abuse",
    "C:10_T43222A": "SSRI OD (self-harm)",
    "C:10_T391X2A": "Acetamin OD (self-harm)",
    "C:10_T781XXA": "Food reaction",
    "C:9_9654": "Analgesic OD",
    "C:9_30981": "PTSD",
    "C:9_78659": "Chest pain (other)",
    "P:9_5491": "Abd. drainage",
    "P:9_8938": "Resp measurements",
    "P:9_9925": "Chemo infusion",
    "P:10_10D00Z1": "C-section",
    "P:10_10E0XZZ": "Resp monitoring",
    "D:RX_2418": "Vitamin D",
    "D:RX_1657128": "Influenza A Ag",
    # Common chronic conditions
    "C:10_E119": "T2DM",
    "C:10_E785": "HLD",
    "C:10_I4891": "AFib",
    "C:10_I509": "CHF",
    "C:10_J449": "COPD",
    "C:10_N390": "UTI",
    "C:10_G4733": "Sleep apnea",
    "C:10_E039": "Hypothyroidism",
    "C:10_N400": "BPH",
    "C:10_F17210": "Nicotine dep.",
    "C:10_Z87891": "Hx nicotine dep.",
    "C:10_Z915": "Hx self-harm",
    "C:10_O701": "Perineal laceration",
}

WORD_SHORTENINGS = [
    (", unspecified", ""),
    (", unsp.", ""),
    ("unspecified ", ""),
    ("not elsewhere classified", "NEC"),
    ("not otherwise specified", ""),
    ("intentional self-harm", "self-harm"),
    ("initial encounter", ""),
    ("uncomplicated", ""),
    ("Poisoning by ", ""),
    ("Personal history of", "Hx"),
    ("complicating pregnancy, childbirth, or the puerperium", "in pregnancy"),
    (", delivered, with or without mention of antepartum condition", ""),
    (", unspecified as to episode of care or not applicable", ""),
]


def _resolve_token(tok):
    """Step 1: Raw token -> full readable name via ICD titles / drug names."""
    import re
    tok = tok.strip()
    m = re.match(r'^([CPD]):((?:9|10)_)?(.+)$', tok)
    if not m:
        return tok
    prefix = m.group(1)
    ver_part = m.group(2) or ""
    code = m.group(3)
    version = ver_part.rstrip("_") if ver_part else ""
    if prefix == "D":
        drug_key = f"D:{code}"
        if drug_key in _DRUG_NAMES:
            return _DRUG_NAMES[drug_key]
        if code.startswith("RX_"):
            full_key = f"D:RX_{code[3:]}"
            if full_key in _DRUG_NAMES:
                return _DRUG_NAMES[full_key]
        return code
    if version:
        full_key = f"{prefix}:{version}_{code}"
        if full_key in _ICD_TITLES:
            return _ICD_TITLES[full_key]
    bare_key = f"{prefix}:{code}"
    if bare_key in _ICD_TITLES:
        return _ICD_TITLES[bare_key]
    if code in _ICD_TITLES:
        return _ICD_TITLES[code]
    return f"{prefix}:{code}"


def _abbreviate(text):
    """Step 2: Full ICD title -> short figure label."""
    if not text:
        return text
    if text in _SHORT_NAME_MAP:
        return _SHORT_NAME_MAP[text]
    for long, short in WORD_SHORTENINGS:
        if long in text:
            text = text.replace(long, short)
    import re
    text = text.strip().rstrip(",").strip()
    text = re.sub(r"\s+", " ", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if len(text) > 45:
        text = text[:42] + "..."
    return text


def _format_token(tok):
    """Full pipeline: raw token -> figure label."""
    raw = tok.strip()
    # Step 0a: Direct lookup in TOKEN_LABELS (hardcoded)
    if raw in TOKEN_LABELS:
        label = TOKEN_LABELS[raw]
        _MAPPING_LOG.append((raw, label, label))
        return label
    # Step 0b: Direct lookup in SHORT_NAME_MAP (from CSV, includes raw tokens)
    if raw in _SHORT_NAME_MAP:
        label = _SHORT_NAME_MAP[raw]
        _MAPPING_LOG.append((raw, label, label))
        return label
    # Step 1: Resolve to full name via ICD titles / drug names
    resolved = _resolve_token(raw)
    # Step 2: Abbreviate
    abbreviated = _abbreviate(resolved)
    _MAPPING_LOG.append((raw, resolved, abbreviated))
    return abbreviated


def _format_pattern_short(raw_pattern):
    """Format raw pattern string with short names, preserving {co-occurrence} structure."""
    steps = raw_pattern.split(" -> ")
    formatted_steps = []
    for step in steps:
        step = step.strip()
        if step.startswith("{") and step.endswith("}"):
            inner = step[1:-1]
            tokens = [t.strip() for t in inner.split(",")]
            formatted_steps.append("{" + ", ".join(_format_token(t) for t in tokens) + "}")
        else:
            formatted_steps.append(_format_token(step))
    return " \u2192 ".join(formatted_steps)

    steps = raw_pattern.split(" -> ")
    formatted_steps = []
    for step in steps:
        step = step.strip()
        if step.startswith("{") and step.endswith("}"):
            inner = step[1:-1]
            tokens = [t.strip() for t in inner.split(",")]
            short_tokens = [_fmt_raw_token(t) for t in tokens]
            formatted_steps.append("{" + ", ".join(short_tokens) + "}")
        else:
            formatted_steps.append(_fmt_raw_token(step))

    return " \u2192 ".join(formatted_steps)


def _fmt_tok_from_csv(tok, df_tok):
    """Look up readable name from token CSV."""
    row = df_tok[df_tok["token"] == tok]
    if not row.empty:
        return row.iloc[0]["token_readable"]
    return tok


def _count_pattern_blocks(pat_str):
    """Count visit blocks in a pattern string."""
    return len(pat_str.split(" -> "))


def replot_fig3(csv_dir, main_dir, supp_dir, n_show, split, dpi, run_dir=None):
    """Replot fig3 from CSVs."""
    tok_csv = os.path.join(csv_dir, f"fig3_token_importance_{split}.csv")
    pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    pw_xvisit_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}_xvisit.csv")
    pw_all_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}_all.csv")

    if not os.path.exists(tok_csv):
        print(f"  [replot] Token CSV not found: {tok_csv}")
        return None
    if not os.path.exists(pw_csv):
        print(f"  [replot] Pathway CSV not found: {pw_csv}")
        return None

    df_tok = pd.read_csv(tok_csv)
    df_pw = pd.read_csv(pw_csv)
    n_test = int(round(df_pw["n_present"].iloc[0] / df_pw["prevalence"].iloc[0])) if not df_pw.empty and "prevalence" in df_pw.columns else 0

    # Build Series for Panels A and B
    ig_global = pd.Series(
        df_tok.set_index("token")["ig_mean"].dropna().to_dict())
    shap_global = pd.Series(
        df_tok.set_index("token")["shap_mean"].dropna().to_dict())

    # Build readable name lookup from token CSV
    tok_readable = dict(zip(df_tok["token"], df_tok["token_readable"]))

    # Try to load icd_titles for pattern formatting fallback
    icd_titles = {}
    if run_dir:
        for candidate in [
            os.path.join(run_dir, "icd_titles.json"),
            os.path.join(run_dir, "explain", "icd_titles.json"),
        ]:
            if os.path.exists(candidate):
                with open(candidate) as f:
                    icd_titles = json.load(f)
                print(f"  [replot] Loaded {len(icd_titles)} ICD titles from {candidate}")
                break

    def _format_pattern_label(pat_str):
        """Convert raw pattern string to readable format using token CSV lookup.
        
        Input:  {C:10_F329, C:10_R45851}
        Output: {C:F329 (MDD, single ep.), C:R45851 (SI)}
        """
        # Try pattern_readable column first (handled by caller)
        # Otherwise reconstruct from token lookup
        import re
        
        def _fmt_single_token(tok):
            """Format a single token like C:10_F329 → C:F329 (MDD, single ep.)"""
            tok = tok.strip()
            if tok in tok_readable:
                return tok_readable[tok]
            # Try to find in icd_titles by stripping numeric prefix
            # Token format: C:10_F329 → code is F329
            parts = tok.split("_", 1)
            if len(parts) == 2:
                prefix_with_num = parts[0]  # e.g. C:10
                code = parts[1]  # e.g. F329
                prefix = prefix_with_num.split(":")[0]  # e.g. C
                if code in icd_titles:
                    desc = icd_titles[code]
                    if len(desc) > 40:
                        desc = desc[:37] + "..."
                    return f"{prefix}:{code} ({desc})"
                return f"{prefix}:{code}"
            return tok
        
        # Parse the pattern: split by " -> " for sequential steps,
        # then handle {set} notation within each step
        steps = pat_str.split(" -> ")
        formatted_steps = []
        for step in steps:
            step = step.strip()
            if step.startswith("{") and step.endswith("}"):
                # Co-occurrence set: {tok1, tok2}
                inner = step[1:-1]
                tokens = [t.strip() for t in inner.split(",")]
                formatted = ", ".join(_fmt_single_token(t) for t in tokens)
                formatted_steps.append("{" + formatted + "}")
            else:
                formatted_steps.append(_fmt_single_token(step))
        return " \u2192 ".join(formatted_steps)

    def render(df_pathways, out_path, panel_c_label=""):
        N = n_show
        fig, axes = plt.subplots(1, 3, figsize=(32, max(10, N * 0.6)))

        suffix = f" — {panel_c_label}" if panel_c_label else ""
        fig.suptitle(
            f"Global importance comparison: IG vs GradientSHAP vs AToP "
            f"({split.upper()}){suffix}\n"
            "All panels computed on the same test cohort — "
            "single-token (A, B) vs occurrence-aggregated pattern attribution (C)",
            fontsize=12, fontweight="bold", y=1.02,
        )

        # Panel A: IG
        ax = axes[0]
        if not ig_global.empty:
            ig_top = ig_global.reindex(
                ig_global.abs().sort_values(ascending=False).head(N).index)
            ig_top = ig_top.reindex(ig_top.abs().sort_values(ascending=True).index)
            labels_a = [tok_readable.get(t, t) for t in ig_top.index]
            vals_a = ig_top.values.tolist()
            colors_a = ["#c44e52" if v > 0 else "#4c72b0" for v in vals_a]
            ax.barh(range(len(labels_a)), vals_a, color=colors_a,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(labels_a)))
            ax.set_yticklabels(labels_a, fontsize=8.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean IG\n\u2190 protective    risk \u2192", fontsize=9)
        ax.set_title(f"A: Integrated Gradients\n(single-token, all {split} patients)",
                     fontsize=10)

        # Panel B: SHAP
        ax = axes[1]
        if not shap_global.empty:
            shap_top = shap_global.reindex(
                shap_global.abs().sort_values(ascending=False).head(N).index)
            shap_top = shap_top.reindex(
                shap_top.abs().sort_values(ascending=True).index)
            labels_b = [tok_readable.get(t, t) for t in shap_top.index]
            vals_b = shap_top.values.tolist()
            colors_b = ["#c44e52" if v > 0 else "#4c72b0" for v in vals_b]
            ax.barh(range(len(labels_b)), vals_b, color=colors_b,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(labels_b)))
            ax.set_yticklabels(labels_b, fontsize=8.5)
        ax.axvline(0, color="black", linewidth=0.8)
        n_shap = n_test if n_test else "?"
        ax.set_xlabel(f"Mean GradientSHAP ({n_shap} patients)\n"
                      "\u2190 protective    risk \u2192", fontsize=9)
        ax.set_title(f"B: GradientSHAP\n(single-token, all {split} patients)",
                     fontsize=10)

        # Panel C: AToP patterns
        ax = axes[2]
        if not df_pathways.empty:
            df_p = df_pathways.copy()
            df_p["_abs_ig"] = df_p["ig_signed_mean"].abs()
            df_p["_abs_ig_bin"] = df_p["_abs_ig"].round(2)
            df_p["_n_blocks"] = df_p["pattern"].apply(_count_pattern_blocks)
            top_paths = (df_p
                         .sort_values(["_abs_ig_bin", "_n_blocks", "_abs_ig"],
                                      ascending=[False, False, False])
                         .head(N)
                         .sort_values("ig_signed_mean", key=abs, ascending=True))
            labels_c = []
            colors_c = []
            for _, r in top_paths.iterrows():
                if "pattern_readable" in r.index and pd.notna(r.get("pattern_readable")):
                    pat = str(r["pattern_readable"])
                else:
                    pat = _format_pattern_label(r["pattern"])
                direction = "\u25B2" if r["ig_signed_mean"] > 0 else "\u25BC"
                labels_c.append(f"{direction} {pat}  (n={r['n_present']})")
                colors_c.append("#c44e52" if r["ig_signed_mean"] > 0 else "#4c72b0")
            ax.barh(range(len(labels_c)), top_paths["ig_signed_mean"].values,
                    color=colors_c, edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(labels_c)))
            ax.set_yticklabels(labels_c, fontsize=8)
        else:
            ax.text(0.5, 0.5, "No patterns", ha="center", va="center")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean \u03A3IG at matched positions\n"
                      "\u2190 protective    risk \u2192", fontsize=9)
        c_title = "C: AToP temporal patterns"
        if panel_c_label:
            c_title += f"\n({panel_c_label})"
        ax.set_title(c_title, fontsize=10)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#c44e52", label="\u25B2 Risk (positive IG)"),
            Patch(facecolor="#4c72b0", label="\u25BC Protective (negative IG)"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2,
                   fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [replot] Saved \u2192 {out_path}")

    # Main: multi-token
    render(df_pw, os.path.join(main_dir, f"fig3_global_comparison_{split}.png"),
           panel_c_label="\u22652-token patterns (incl. same-visit)")

    # Supplement: cross-visit
    if os.path.exists(pw_xvisit_csv):
        df_xvisit = pd.read_csv(pw_xvisit_csv)
        render(df_xvisit,
               os.path.join(supp_dir, f"fig3_global_comparison_{split}_xvisit.png"),
               panel_c_label="cross-visit patterns only")

    # Supplement: all
    if os.path.exists(pw_all_csv):
        df_all = pd.read_csv(pw_all_csv)
        render(df_all,
               os.path.join(supp_dir, f"fig3_global_comparison_{split}_all.png"),
               panel_c_label="all patterns (incl. single-token)")

    return df_pw


def replot_fig3_clean(csv_dir, out_dir, n_show, split, dpi, short_names=False, 
                      jaccard_threshold=0.2, panel_c_split=False, fontsize=13,
                      population_avg=False):
    """Clean 3-panel fig3 with short names, better fonts, SHAP from cache, and Jaccard Panel C."""
    import pickle

    # Load short names mapping if needed
    if short_names:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from atop.explain.figures import load_short_names, _SHORT_NAME_MAP
            if not _SHORT_NAME_MAP:
                sn_path = os.path.join(os.path.dirname(__file__), "short_names.csv")
                load_short_names(sn_path)
                print(f"  [fig3_clean] Loaded {len(_SHORT_NAME_MAP)} short name mappings")
        except Exception as e:
            print(f"  [fig3_clean] Short names loading skipped: {e}")

    tok_csv = os.path.join(csv_dir, f"fig3_token_importance_{split}.csv")
    shap_csv = os.path.join(csv_dir, f"shap_global_{split}.csv")
    pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    pkl_path = os.path.join(csv_dir, f"carrier_sets_{split}.pkl")

    if not os.path.exists(tok_csv):
        print(f"  [fig3_clean] Missing {tok_csv}")
        return

    df_tok = pd.read_csv(tok_csv)
    name_map = dict(zip(df_tok["token"], df_tok["token_readable"]))

    # --- Panel A data: IG ---
    df_ig = df_tok.dropna(subset=["ig_mean"]).copy()
    ig_vals = pd.Series(df_ig.set_index("token")["ig_mean"])

    # For population averaging, load single-token prevalence from _all pathway CSV
    if population_avg and n_test > 0:
        pw_all_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}_all.csv")
        if os.path.exists(pw_all_csv):
            df_pw_all = pd.read_csv(pw_all_csv)
            # Single-token patterns have no " -> "
            df_single = df_pw_all[~df_pw_all["pattern"].str.contains(" -> ", na=False)].copy()
            if not df_single.empty and "prevalence" in df_single.columns:
                # Map pattern (token) to prevalence
                tok_prev = dict(zip(df_single["pattern"], df_single["prevalence"]))
                # Scale ig_vals: population_mean = carrier_mean × prevalence
                ig_pop = {}
                for tok, val in ig_vals.items():
                    prev = tok_prev.get(tok, 0)
                    ig_pop[tok] = val * prev if prev > 0 else 0
                ig_vals = pd.Series(ig_pop)
                print(f"  [fig3_clean] Panel A: population-averaged ({len(tok_prev)} token prevalences)")

    # --- Panel B data: SHAP (prefer cache) — already population-averaged ---
    if os.path.exists(shap_csv):
        df_shap = pd.read_csv(shap_csv).dropna(subset=["shap_mean"])
        shap_vals = pd.Series(df_shap.set_index("token")["shap_mean"])
    else:
        shap_vals = pd.Series(df_tok.set_index("token")["shap_mean"].dropna())

    # --- Panel C data: Jaccard-deduplicated patterns ---
    df_pw = pd.read_csv(pw_csv) if os.path.exists(pw_csv) else pd.DataFrame()

    # Estimate n_test from pathway CSV
    n_test = 0
    if not df_pw.empty and "prevalence" in df_pw.columns and "n_present" in df_pw.columns:
        row0 = df_pw.iloc[0]
        if row0["prevalence"] > 0:
            n_test = int(round(row0["n_present"] / row0["prevalence"]))

    # Population averaging: multiply carrier-averaged values by prevalence
    if population_avg and n_test > 0:
        print(f"  [fig3_clean] Population-averaging (n_test={n_test})")
        # Panel A: ig_mean is carrier-averaged. We don't have n_carriers per token
        # in the token CSV, but SHAP CSV has all tokens. Use SHAP token count as proxy:
        # tokens in ig_vals that also appear in shap have population presence.
        # For now, estimate: if token has ig_mean, it appeared in the saliency set.
        # The token CSV was built from df_ig which only has salient tokens.
        # We can't perfectly recover n_carriers, so we scale IG to match SHAP's
        # population-averaged approach by using prevalence from the pathway CSV
        # for Panel C, and noting Panel A remains carrier-averaged.
        # 
        # Actually: for Panel A consistency, we can use the pathway CSV to find
        # single-token "patterns" and their prevalence. But single tokens aren't
        # in the pathway CSV. So Panel A stays carrier-averaged.
        # 
        # Best approach: just scale Panel C to population-averaged
        pass  # Panel A stays carrier-averaged, Panel C gets scaled below
    panel_c_labels, panel_c_vals, panel_c_colors = [], [], []

    if not df_pw.empty and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            carriers = pickle.load(f)
        df_pw["_ck"] = df_pw["pattern"].map(carriers)
        df_pw = df_pw[df_pw["_ck"].notna() & (df_pw["n_present"] >= 3)].copy()

        if not df_pw.empty:
            # Jaccard clustering
            pats = df_pw.sort_values("ig_signed_mean", key=abs, ascending=False).reset_index(drop=True)
            all_p = set()
            cl, sl = [], []
            for _, row in pats.iterrows():
                ck = row["_ck"]
                if isinstance(ck, set):
                    all_p.update(ck); cl.append(ck); sl.append(len(ck))
                else:
                    cl.append(set()); sl.append(0)
            p2i = {p: i for i, p in enumerate(sorted(all_p))}
            np_ = len(p2i)
            bvs = []
            for c in cl:
                b = np.zeros(np_, dtype=np.bool_)
                for p in c: b[p2i[p]] = True
                bvs.append(np.packbits(b))
            clusters, assigned = [], set()
            for i in range(len(pats)):
                if i in assigned: continue
                if sl[i] == 0: assigned.add(i); clusters.append((i,[i])); continue
                members = [i]
                for j in range(i+1, len(pats)):
                    if j in assigned or sl[j] == 0: continue
                    if min(sl[i],sl[j])/max(sl[i],sl[j]) < jaccard_threshold: continue
                    inter = int(np.unpackbits(bvs[i] & bvs[j]).sum())
                    union = int(np.unpackbits(bvs[i] | bvs[j]).sum())
                    if union > 0 and inter/union >= jaccard_threshold:
                        members.append(j); assigned.add(j)
                assigned.add(i); clusters.append((i, members))

            df_reps = pd.DataFrame([pats.iloc[ri] for ri, _ in clusters])
            label_col = "pattern_readable" if "pattern_readable" in df_reps.columns else "pattern"
            half = max(1, n_show // 2)
            df_r = df_reps[df_reps["ig_signed_mean"] > 0].nlargest(half, "ig_signed_mean")
            df_p = df_reps[df_reps["ig_signed_mean"] < 0].nsmallest(half, "ig_signed_mean")
            top_c = pd.concat([df_r, df_p]).sort_values("ig_signed_mean", key=abs, ascending=True)

            for _, r in top_c.iterrows():
                if short_names and "pattern" in r.index:
                    lbl = _format_pattern_short(r["pattern"])
                else:
                    lbl = str(r[label_col])
                panel_c_labels.append(lbl)
                val = r["ig_signed_mean"]
                if population_avg and n_test > 0 and "prevalence" in r.index:
                    val = val * r.get("prevalence", r.get("n_present", 0) / n_test)
                panel_c_vals.append(val)
                panel_c_colors.append("#c44e52" if r["ig_signed_mean"] > 0 else "#4c72b0")

    # --- Render figure ---
    N = n_show
    n_c = len(panel_c_vals)
    jt_pct = int(jaccard_threshold * 100)
    bar_height = 0.5
    avg_label = "population-averaged" if population_avg else "carrier-averaged"
    ig_xlabel = f"Mean IG ({avg_label})\n\u2190 protective    risk \u2192"
    c_xlabel = f"Mean \u03A3IG ({avg_label})\n\u2190 protective    risk \u2192"

    # Separate risk/protective for Panel C
    c_risk_labels = [l for l, v in zip(panel_c_labels, panel_c_vals) if v > 0]
    c_risk_vals = [v for v in panel_c_vals if v > 0]
    c_prot_labels = [l for l, v in zip(panel_c_labels, panel_c_vals) if v <= 0]
    c_prot_vals = [abs(v) for v in panel_c_vals if v <= 0]

    if panel_c_split:
        # 4 panels: A, B, C-risk, C-protective
        # C panels get 1.5x width to accommodate longer pattern labels
        n_cr = len(c_risk_vals)
        n_cp = len(c_prot_vals)
        max_rows = max(N, n_cr, n_cp, 10)
        fig, axes = plt.subplots(1, 4, figsize=(64, max(10, max_rows * 0.45)),
                                  gridspec_kw={"width_ratios": [1, 1, 1.5, 1.5]})

        fig.suptitle(
            f"Global importance: IG vs GradientSHAP vs AToP ({split.upper()})",
            fontsize=16, fontweight="bold", y=1.01)

        # Panel A: IG
        ax = axes[0]
        if not ig_vals.empty:
            ig_top = ig_vals.reindex(ig_vals.abs().sort_values(ascending=False).head(N).index)
            ig_top = ig_top.reindex(ig_top.abs().sort_values(ascending=True).index)
            labels_a = [name_map.get(t, t) for t in ig_top.index]
            if short_names:
                labels_a = [_to_short_name(l) for l in labels_a]
            colors_a = ["#c44e52" if v > 0 else "#4c72b0" for v in ig_top.values]
            ax.barh(range(len(labels_a)), ig_top.values, height=bar_height, color=colors_a,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(labels_a)))
            ax.set_yticklabels(labels_a, fontsize=fontsize)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(ig_xlabel, fontsize=12)
        ax.set_title("A: Integrated Gradients\n(single-token)", fontsize=14)

        # Panel B: SHAP
        ax = axes[1]
        if not shap_vals.empty:
            shap_top = shap_vals.reindex(shap_vals.abs().sort_values(ascending=False).head(N).index)
            shap_top = shap_top.reindex(shap_top.abs().sort_values(ascending=True).index)
            labels_b = [name_map.get(t, t) for t in shap_top.index]
            if short_names:
                labels_b = [_to_short_name(l) for l in labels_b]
            colors_b = ["#c44e52" if v > 0 else "#4c72b0" for v in shap_top.values]
            ax.barh(range(len(labels_b)), shap_top.values, height=bar_height, color=colors_b,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(labels_b)))
            ax.set_yticklabels(labels_b, fontsize=fontsize)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean GradientSHAP\n\u2190 protective    risk \u2192", fontsize=12)
        ax.set_title("B: GradientSHAP\n(single-token)", fontsize=14)

        # Panel C-Risk
        ax = axes[2]
        if c_risk_vals:
            # Sort: smallest at top
            order = sorted(range(n_cr), key=lambda i: c_risk_vals[i])
            ax.barh(range(n_cr), [c_risk_vals[i] for i in order], height=bar_height,
                    color="#c44e52", edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(n_cr))
            ax.set_yticklabels([c_risk_labels[i] for i in order], fontsize=fontsize)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"Mean 03A3IG ({avg_label})\n2192 risk contribution", fontsize=12)
        ax.set_title(f"C: Risk patterns\n(Jaccard \u2265 {jaccard_threshold})", fontsize=14)

        # Panel C-Protective
        ax = axes[3]
        if c_prot_vals:
            order = sorted(range(n_cp), key=lambda i: c_prot_vals[i])
            ax.barh(range(n_cp), [c_prot_vals[i] for i in order], height=bar_height,
                    color="#4c72b0", edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(n_cp))
            ax.set_yticklabels([c_prot_labels[i] for i in order], fontsize=fontsize)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"|Mean 03A3IG| ({avg_label})\n2192 protective contribution", fontsize=12)
        ax.set_title(f"C: Protective patterns\n(Jaccard \u2265 {jaccard_threshold})", fontsize=14)

    else:
        # 3 panels: A, B, C-mixed
        max_rows = max(N, n_c, 10)
        fig, axes = plt.subplots(1, 3, figsize=(52, max(10, max_rows * 0.45)),
                                  gridspec_kw={"width_ratios": [1, 1, 1.8]})

        fig.suptitle(
            f"Global importance: IG vs GradientSHAP vs AToP ({split.upper()})",
            fontsize=16, fontweight="bold", y=1.01)

        # Panel A: IG
        ax = axes[0]
        if not ig_vals.empty:
            ig_top = ig_vals.reindex(ig_vals.abs().sort_values(ascending=False).head(N).index)
            ig_top = ig_top.reindex(ig_top.abs().sort_values(ascending=True).index)
            labels_a = [name_map.get(t, t) for t in ig_top.index]
            if short_names:
                labels_a = [_to_short_name(l) for l in labels_a]
            colors_a = ["#c44e52" if v > 0 else "#4c72b0" for v in ig_top.values]
            ax.barh(range(len(labels_a)), ig_top.values, height=bar_height, color=colors_a,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(labels_a)))
            ax.set_yticklabels(labels_a, fontsize=fontsize)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(ig_xlabel, fontsize=12)
        ax.set_title("A: Integrated Gradients\n(single-token)", fontsize=14)

        # Panel B: SHAP
        ax = axes[1]
        if not shap_vals.empty:
            shap_top = shap_vals.reindex(shap_vals.abs().sort_values(ascending=False).head(N).index)
            shap_top = shap_top.reindex(shap_top.abs().sort_values(ascending=True).index)
            labels_b = [name_map.get(t, t) for t in shap_top.index]
            if short_names:
                labels_b = [_to_short_name(l) for l in labels_b]
            colors_b = ["#c44e52" if v > 0 else "#4c72b0" for v in shap_top.values]
            ax.barh(range(len(labels_b)), shap_top.values, height=bar_height, color=colors_b,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(labels_b)))
            ax.set_yticklabels(labels_b, fontsize=fontsize)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean GradientSHAP\n\u2190 protective    risk \u2192", fontsize=12)
        ax.set_title("B: GradientSHAP\n(single-token)", fontsize=14)

        # Panel C: mixed
        ax = axes[2]
        if panel_c_vals:
            ax.barh(range(n_c), panel_c_vals, height=bar_height, color=panel_c_colors,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(n_c))
            ax.set_yticklabels(panel_c_labels, fontsize=fontsize)
        else:
            ax.text(0.5, 0.5, "No patterns", ha="center", va="center")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(c_xlabel, fontsize=12)
        ax.set_title(f"C: AToP temporal patterns\n(Jaccard \u2265 {jaccard_threshold})", fontsize=14)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#c44e52", label="\u25B2 Risk (positive)"),
        Patch(facecolor="#4c72b0", label="\u25BC Protective (negative)"),
    ], loc="lower center", ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Squeeze bar plots to give y-tick labels more room
    # Each axes: shift right edge inward so bars take ~40% of panel width
    for ax in (axes if not hasattr(axes, '__len__') else axes):
        pos = ax.get_position()
        # Keep left edge (where text is), shrink right edge
        new_width = pos.width * 0.55
        ax.set_position([pos.x0, pos.y0, new_width, pos.height])

    sfx = "_short" if short_names else ""
    sfx += "_split" if panel_c_split else ""
    sfx += "_pop" if population_avg else ""
    out_path = os.path.join(out_dir, f"fig3_clean_{split}_j{jt_pct}{sfx}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig3_clean] Saved \u2192 {out_path}")


def replot_fig3_main(csv_dir, out_dir, n_show, split, dpi, short_names=False,
                     jaccard_threshold=0.2, fontsize=13, show_n=False, panel_c_mode="balanced",
                     fig_width=54, bar_squeeze=0.55):
    """Publication figure 3: three complementary views of model attribution.
    
    Panel A: Population-averaged IG (single-token) — E[IG] across all test patients.
             Shows what matters on average in the whole cohort (prevalence-weighted).
    Panel B: Carrier-averaged IG (single-token) — E[IG | present] with n_present.
             Shows what matters when present (conditional importance).
    Panel C: Carrier-averaged ΣIG (temporal patterns) — E[ΣIG | present] with n_present.
             Shows what matters when a trajectory occurs (temporal narratives).
    
    B and C use the same unit (patient-level IG sums, averaged over carriers)
    and are directly comparable. A shows the population view.
    E[score_j] = P(present_j) × E[score_j | present_j]
    """
    import pickle

    # Load short names
    if short_names:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from atop.explain.figures import load_short_names, _SHORT_NAME_MAP
            if not _SHORT_NAME_MAP:
                sn_path = os.path.join(os.path.dirname(__file__), "short_names.csv")
                load_short_names(sn_path)
        except Exception as e:
            print(f"  [fig3_main] Short names loading skipped: {e}")

    tok_csv = os.path.join(csv_dir, f"fig3_token_importance_{split}.csv")
    pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    pw_all_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}_all.csv")
    pkl_path = os.path.join(csv_dir, f"carrier_sets_{split}.pkl")

    if not os.path.exists(tok_csv):
        print(f"  [fig3_main] Missing {tok_csv}")
        return

    df_tok = pd.read_csv(tok_csv)
    name_map = dict(zip(df_tok["token"], df_tok["token_readable"]))

    # --- Estimate n_test ---
    df_pw = pd.read_csv(pw_csv) if os.path.exists(pw_csv) else pd.DataFrame()
    n_test = 0
    if not df_pw.empty and "prevalence" in df_pw.columns and "n_present" in df_pw.columns:
        row0 = df_pw.iloc[0]
        if row0["prevalence"] > 0:
            n_test = int(round(row0["n_present"] / row0["prevalence"]))

    # --- Panel A: Population-averaged IG ---
    # E[IG] = E[IG | present] × P(present) = ig_mean × prevalence
    df_ig_tok = df_tok.dropna(subset=["ig_mean"]).copy()
    ig_carrier = pd.Series(df_ig_tok.set_index("token")["ig_mean"])
    tok_n_present = {}
    tok_prevalence = {}

    if "n_present" in df_tok.columns and "prevalence" in df_tok.columns:
        for _, r in df_tok.dropna(subset=["ig_mean"]).iterrows():
            tok_n_present[r["token"]] = int(r.get("n_present", 0))
            tok_prevalence[r["token"]] = r.get("prevalence", 0)
        ig_pop = pd.Series({tok: ig_carrier[tok] * tok_prevalence.get(tok, 0)
                            for tok in ig_carrier.index})
        print(f"  [fig3_main] Panel A: population-averaged ({len(ig_pop)} tokens)")
    else:
        print(f"  [fig3_main] Panel A: token CSV missing n_present/prevalence — re-run run_atop.py")
        ig_pop = ig_carrier
        # Try to get n_present from _all CSV as fallback
        pw_all_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}_all.csv")
        if os.path.exists(pw_all_csv):
            df_all = pd.read_csv(pw_all_csv)
            df_single = df_all[~df_all["pattern"].str.contains(" -> ", na=False)]
            for _, r in df_single.iterrows():
                tok_n_present[r["pattern"]] = int(r["n_present"])
                tok_prevalence[r["pattern"]] = r.get("prevalence", 0)

    # --- Panel B: Carrier-averaged IG (same data, but displayed with n_present) ---
    # ig_carrier already set above

    # --- Panel C: Carrier-averaged ΣIG, Jaccard-deduplicated ---
    panel_c_labels, panel_c_vals, panel_c_n = [], [], []
    panel_c_colors = []

    if not df_pw.empty and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            carriers = pickle.load(f)
        df_pw["_ck"] = df_pw["pattern"].map(carriers)
        df_pw = df_pw[df_pw["_ck"].notna() & (df_pw["n_present"] >= 3)].copy()

        if not df_pw.empty:
            # Jaccard clustering
            pats = df_pw.sort_values("ig_signed_mean", key=abs, ascending=False).reset_index(drop=True)
            all_p = set()
            cl, sl = [], []
            for _, row in pats.iterrows():
                ck = row["_ck"]
                if isinstance(ck, set):
                    all_p.update(ck); cl.append(ck); sl.append(len(ck))
                else:
                    cl.append(set()); sl.append(0)
            p2i = {p: i for i, p in enumerate(sorted(all_p))}
            np_ = len(p2i)
            bvs = []
            for c in cl:
                b = np.zeros(np_, dtype=np.bool_)
                for p in c: b[p2i[p]] = True
                bvs.append(np.packbits(b))
            clusters, assigned = [], set()
            for i in range(len(pats)):
                if i in assigned: continue
                if sl[i] == 0: assigned.add(i); clusters.append((i,[i])); continue
                members = [i]
                for j in range(i+1, len(pats)):
                    if j in assigned or sl[j] == 0: continue
                    if min(sl[i],sl[j])/max(sl[i],sl[j]) < jaccard_threshold: continue
                    inter = int(np.unpackbits(bvs[i] & bvs[j]).sum())
                    union = int(np.unpackbits(bvs[i] | bvs[j]).sum())
                    if union > 0 and inter/union >= jaccard_threshold:
                        members.append(j); assigned.add(j)
                assigned.add(i); clusters.append((i, members))

            df_reps = pd.DataFrame([pats.iloc[ri] for ri, _ in clusters])
            label_col = "pattern_readable" if "pattern_readable" in df_reps.columns else "pattern"
            if panel_c_mode == "top":
                df_reps["_abs_ig"] = df_reps["ig_signed_mean"].abs()
                top_c = df_reps.nlargest(n_show, "_abs_ig")
                top_c = top_c.sort_values("_abs_ig", ascending=True)
                top_c = top_c.drop(columns=["_abs_ig"])
            else:
                half = max(1, n_show // 2)
                df_r = df_reps[df_reps["ig_signed_mean"] > 0].nlargest(half, "ig_signed_mean")
                df_p = df_reps[df_reps["ig_signed_mean"] < 0].nsmallest(half, "ig_signed_mean")
                top_c = pd.concat([df_r, df_p]).sort_values("ig_signed_mean", key=abs, ascending=True)

            for _, r in top_c.iterrows():
                if short_names and "pattern" in r.index:
                    lbl = _format_pattern_short(r["pattern"])
                else:
                    lbl = str(r[label_col])
                panel_c_labels.append(lbl)
                panel_c_vals.append(r["ig_signed_mean"])
                panel_c_n.append(int(r["n_present"]))
                panel_c_colors.append("#c44e52" if r["ig_signed_mean"] > 0 else "#4c72b0")

    # --- Render 3-panel figure ---
    N = n_show
    n_c = len(panel_c_vals)
    max_rows = max(N, n_c, 10)
    bar_height = 0.5

    # --- Export panel CSVs with full mapping ---
    jt_pct = int(jaccard_threshold * 100)
    sfx = "_short" if short_names else ""

    # Panel A CSV
    if not ig_pop.empty:
        ig_top_a = ig_pop.reindex(ig_pop.abs().sort_values(ascending=False).head(N).index)
        rows_a = []
        for t in ig_top_a.index:
            resolved = _resolve_token(t)
            abbreviated = _abbreviate(resolved)
            rows_a.append({
                "raw_token": t,
                "full_name": resolved,
                "label": abbreviated,
                "ig_value": ig_top_a[t],
                "n_present": tok_n_present.get(t, 0),
                "prevalence": tok_prevalence.get(t, 0),
            })
        df_a = pd.DataFrame(rows_a)
        a_csv = os.path.join(out_dir, f"panel_a_{split}_j{jt_pct}{sfx}.csv")
        df_a.to_csv(a_csv, index=False)
        print(f"  [fig3_main] Panel A CSV ({len(df_a)} tokens) \u2192 {a_csv}")

    # Panel B CSV
    if not ig_carrier.empty:
        ig_top_b_data = ig_carrier.reindex(ig_carrier.abs().sort_values(ascending=False).head(N).index)
        rows_b = []
        for t in ig_top_b_data.index:
            resolved = _resolve_token(t)
            abbreviated = _abbreviate(resolved)
            rows_b.append({
                "raw_token": t,
                "full_name": resolved,
                "label": abbreviated,
                "ig_value": ig_top_b_data[t],
                "n_present": tok_n_present.get(t, 0),
            })
        df_b = pd.DataFrame(rows_b)
        b_csv = os.path.join(out_dir, f"panel_b_{split}_j{jt_pct}{sfx}.csv")
        df_b.to_csv(b_csv, index=False)
        print(f"  [fig3_main] Panel B CSV ({len(df_b)} tokens) \u2192 {b_csv}")

    # Panel C CSV
    if panel_c_vals:
        rows_c = []
        for lbl, val, n in zip(panel_c_labels, panel_c_vals, panel_c_n):
            rows_c.append({
                "label": lbl,
                "ig_value": val,
                "n_present": n,
            })
        df_c = pd.DataFrame(rows_c)
        c_csv = os.path.join(out_dir, f"panel_c_{split}_j{jt_pct}{sfx}.csv")
        df_c.to_csv(c_csv, index=False)
        print(f"  [fig3_main] Panel C CSV ({len(df_c)} patterns) \u2192 {c_csv}")

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, max(10, max_rows * 0.45)),
                              gridspec_kw={"width_ratios": [1, 1, 1]})

    fs_title = fontsize + 2
    fs_xlabel = fontsize - 1
    fs_suptitle = fontsize + 4
    fs_legend = fontsize - 1

    fig.suptitle(
        f"Global attribution comparison ({split.upper()}, n={n_test})",
        fontsize=fs_suptitle, fontweight="bold", y=1.01)

    # Panel A: Population-averaged IG
    ax = axes[0]
    if not ig_pop.empty:
        ig_top = ig_pop.reindex(ig_pop.abs().sort_values(ascending=False).head(N).index)
        ig_top = ig_top.reindex(ig_top.abs().sort_values(ascending=True).index)
        if short_names:
            labels_a = [_format_pattern_short(t) for t in ig_top.index]
        else:
            labels_a = [name_map.get(t, t) for t in ig_top.index]
        colors_a = ["#c44e52" if v > 0 else "#4c72b0" for v in ig_top.values]
        ax.barh(range(len(labels_a)), ig_top.values, height=bar_height, color=colors_a,
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(labels_a)))
        ax.set_yticklabels(labels_a, fontsize=fontsize)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("E[IG] across all patients\n\u2190 protective    risk \u2192", fontsize=fs_xlabel)
    ax.set_title("A: Population IG (single-token)\nprevalence-weighted", fontsize=fs_title, pad=15)
    ax.tick_params(axis='x', labelsize=fs_xlabel - 1)

    # Panel B: Carrier-averaged IG with n_present
    ax = axes[1]
    if not ig_carrier.empty:
        ig_top_b = ig_carrier.reindex(ig_carrier.abs().sort_values(ascending=False).head(N).index)
        ig_top_b = ig_top_b.reindex(ig_top_b.abs().sort_values(ascending=True).index)
        labels_b = []
        for t in ig_top_b.index:
            if short_names:
                lbl = _format_pattern_short(t)
            else:
                lbl = name_map.get(t, t)
                lbl = _to_short_name(lbl) if short_names else lbl
            n_pr = tok_n_present.get(t, 0)
            if show_n and n_pr > 0:
                lbl = f"{lbl}  (n={n_pr})"
            labels_b.append(lbl)
        colors_b = ["#c44e52" if v > 0 else "#4c72b0" for v in ig_top_b.values]
        ax.barh(range(len(labels_b)), ig_top_b.values, height=bar_height, color=colors_b,
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(labels_b)))
        ax.set_yticklabels(labels_b, fontsize=fontsize)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("E[IG | present]\n\u2190 protective    risk \u2192", fontsize=fs_xlabel)
    ax.set_title("B: Carrier IG (single-token)\nconditional on presence", fontsize=fs_title, pad=15)
    ax.tick_params(axis='x', labelsize=fs_xlabel - 1)

    # Panel C: Carrier-averaged ΣIG with n_present
    ax = axes[2]
    if panel_c_vals:
        if show_n:
            labels_c = [f"{l}  (n={n})" for l, n in zip(panel_c_labels, panel_c_n)]
        else:
            labels_c = panel_c_labels
        ax.barh(range(n_c), panel_c_vals, height=bar_height, color=panel_c_colors,
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_c))
        ax.set_yticklabels(labels_c, fontsize=fontsize)
    else:
        ax.text(0.5, 0.5, "No patterns", ha="center", va="center")
    ax.axvline(0, color="black", linewidth=0.8)
    jt_pct = int(jaccard_threshold * 100)
    ax.set_xlabel("E[\u03A3IG | present]\n\u2190 protective    risk \u2192", fontsize=fs_xlabel)
    ax.set_title(f"C: Carrier \u03A3IG (temporal patterns)\nJaccard \u2265 {jaccard_threshold}, one per phenotype", fontsize=fs_title, pad=15)
    ax.tick_params(axis='x', labelsize=fs_xlabel - 1)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#c44e52", label="\u25B2 Risk (positive)"),
        Patch(facecolor="#4c72b0", label="\u25BC Protective (negative)"),
    ], loc="lower center", ncol=2, fontsize=fs_legend, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Squeeze bar plots to give y-tick labels more room
    for ax in axes:
        pos = ax.get_position()
        new_width = pos.width * bar_squeeze
        ax.set_position([pos.x0, pos.y0, new_width, pos.height])

    sfx = "_short" if short_names else ""
    out_path = os.path.join(out_dir, f"fig3_main_{split}_j{jt_pct}{sfx}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig3_main] Saved \u2192 {out_path}")

    # Save mapping log
    print(f"  [fig3_main] Mapping log entries: {len(_MAPPING_LOG)}")
    seen = set()
    unique_mappings = []
    for entry in _MAPPING_LOG:
        raw = entry[0]
        if raw not in seen:
            seen.add(raw)
            unique_mappings.append(entry)
    map_df = pd.DataFrame(unique_mappings, columns=["raw_token", "full_name", "abbreviated"])
    map_csv = os.path.join(out_dir, f"name_mappings_{split}.csv")
    map_df.to_csv(map_csv, index=False)
    print(f"  [fig3_main] Mapping log ({len(map_df)} tokens) \u2192 {map_csv}")


def replot_fig4(csv_dir, main_dir, supp_dir, n_show, split, dpi, short_names=False):
    """Replot fig4 from validation CSV."""
    val_csv = os.path.join(csv_dir, f"validation_{split}.csv")
    if not os.path.exists(val_csv):
        print(f"  [replot] Validation CSV not found: {val_csv}")
        return

    from atop.explain.figures import fig5_validation
    df_val = pd.read_csv(val_csv)

    # Load icd titles
    tok_csv = os.path.join(csv_dir, f"fig3_token_importance_{split}.csv")
    icd_titles = {}
    if os.path.exists(tok_csv):
        icd_titles = _build_icd_titles(pd.read_csv(tok_csv))

    # Main: multi-token
    from atop.explain.figures import _count_pattern_tokens, _count_pattern_blocks
    df_val_multi = df_val[df_val["pattern"].apply(_count_pattern_tokens) >= 2].copy()
    if not df_val_multi.empty:
        fig5_validation(main_dir, df_val_multi, 50,
                        icd_titles=icd_titles, split=split, n_show=n_show)
        print(f"  [replot] fig4 main: {len(df_val_multi)} patterns")

    # Supplement: all
    fig5_validation(supp_dir, df_val, 50,
                    icd_titles=icd_titles, split=f"{split}_all", n_show=n_show)


def replot_fig5(csv_dir, main_dir, supp_dir, n_show, split, dpi):
    """Replot fig5 from reversed-order CSV."""
    rev_csv = os.path.join(csv_dir, f"reversed_order_{split}.csv")
    if not os.path.exists(rev_csv):
        print(f"  [replot] Reversed-order CSV not found: {rev_csv}")
        return

    from atop.explain.figures import fig5_reversed_order
    df_rev = pd.read_csv(rev_csv)
    if not df_rev.empty:
        fig5_reversed_order(main_dir, df_rev, icd_titles=None,
                            split=split, csv_dir=csv_dir)
        print(f"  [replot] fig5: {len(df_rev)} patterns")


def main():
    args = parse_args()

    # Load short names and drug names at startup
    _load_short_names_csv()
    if args.run_dir:
        _load_drug_names(args.run_dir)
        # Load ICD titles as fallback for codes not in short_names.csv
        _load_icd_titles(args.run_dir)
    if _SHORT_NAME_MAP:
        print(f"  Loaded {len(_SHORT_NAME_MAP)} short name mappings")
    if _DRUG_NAMES:
        print(f"  Loaded {len(_DRUG_NAMES)} drug name mappings")
    if _ICD_TITLES:
        print(f"  Loaded {len(_ICD_TITLES)} ICD titles as fallback")

    explain_dir = args.out_dir or os.path.join(args.run_dir, "explain")
    
    # Find CSV directory — layout may vary between runs
    csv_dir = None
    for candidate in [
        os.path.join(explain_dir, "figures", "csv"),
        os.path.join(explain_dir, "csv"),
        os.path.join(explain_dir, "figures", "main"),
        explain_dir,
        os.path.join(args.run_dir, "explain"),
    ]:
        tok_file = os.path.join(candidate, f"fig3_token_importance_{args.split}.csv")
        if os.path.exists(tok_file):
            csv_dir = candidate
            break
    
    if csv_dir is None:
        # Last resort: search recursively
        import glob
        matches = glob.glob(os.path.join(args.run_dir, "**", f"fig3_token_importance_{args.split}.csv"), recursive=True)
        if matches:
            csv_dir = os.path.dirname(matches[0])
    
    if csv_dir is None:
        print(f"ERROR: Could not find fig3_token_importance_{args.split}.csv anywhere under {args.run_dir}")
        print(f"Searched: {explain_dir}/figures/csv/, {explain_dir}/csv/, {explain_dir}/")
        sys.exit(1)
    
    main_dir = os.path.join(explain_dir, "figures", "main")
    supp_dir = os.path.join(explain_dir, "figures", "supplement")
    # Fall back to explain_dir if figures/ doesn't exist
    if not os.path.isdir(os.path.join(explain_dir, "figures")):
        main_dir = explain_dir
        supp_dir = explain_dir

    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(supp_dir, exist_ok=True)

    if not os.path.isdir(csv_dir):
        print(f"ERROR: CSV directory not found: {csv_dir}")
        sys.exit(1)

    figures = [f.strip() for f in args.figures.split(",")]
    print(f"[replot] Figures: {figures}, n_show={args.n_show}, split={args.split}")
    print(f"[replot] CSV dir: {csv_dir}")
    print(f"[replot] Output: main={main_dir}, supp={supp_dir}")
    print()

    if "fig3" in figures:
        replot_fig3(csv_dir, main_dir, supp_dir, args.n_show, args.split, args.dpi,
                    run_dir=args.run_dir)

    if "fig3_clean" in figures:
        replot_fig3_clean(csv_dir, main_dir, args.n_show, args.split, args.dpi,
                          args.short_names, args.jaccard_threshold, args.panel_c_split,
                          args.fontsize, args.population_avg)

    if "fig3_main" in figures:
        replot_fig3_main(csv_dir, main_dir, args.n_show, args.split, args.dpi,
                         args.short_names, args.jaccard_threshold, args.fontsize,
                         args.show_n, args.panel_c_mode, args.fig_width, args.bar_squeeze)

    if "fig4" in figures:
        replot_fig4(csv_dir, main_dir, supp_dir, args.n_show, args.split, args.dpi, args.short_names)

    if "fig5" in figures:
        replot_fig5(csv_dir, main_dir, supp_dir, args.n_show, args.split, args.dpi)

    if "jaccard" in figures:
        replot_jaccard(csv_dir, supp_dir, args.n_show, args.split, args.dpi,
                       args.jaccard_threshold, args.short_names)

    if "panel_a" in figures:
        replot_panel_a(csv_dir, main_dir, args.n_show, args.split, args.dpi, args.short_names)

    if "panel_b" in figures:
        replot_panel_b(csv_dir, main_dir, args.n_show, args.split, args.dpi, args.short_names)

    if "panel_c" in figures:
        replot_panel_c(csv_dir, main_dir, args.n_show, args.split, args.dpi, args.short_names)

    if "panel_c_split" in figures:
        replot_panel_c_split(csv_dir, main_dir, args.n_show, args.split, args.dpi, args.short_names)

    if "panel_c_stacked" in figures:
        replot_panel_c_stacked(csv_dir, main_dir, args.n_show, args.split, args.dpi,
                                args.jaccard_threshold, args.short_names)

    if "panel_ac_stacked" in figures:
        replot_panel_ac_stacked(csv_dir, main_dir, args.n_show, args.split, args.dpi,
                                 args.jaccard_threshold, args.short_names)

    if "panel_c_jaccard" in figures:
        replot_panel_c_jaccard(csv_dir, main_dir, args.n_show, args.split, args.dpi,
                                args.jaccard_threshold, args.short_names)

    if "attn_flow" in figures:
        replot_attn_flow(csv_dir, supp_dir, args.n_show, args.split, args.dpi, args.short_names)

    print("\n[replot] Done.")


def _plot_single_panel(values, labels, title, xlabel, out_path, n_show, dpi, short_names=False):
    """Render a single horizontal bar chart (standalone panel)."""
    N = min(n_show, len(values))
    if N == 0:
        print(f"  No data for {title}")
        return

    if short_names:
        labels = [_to_short_name(l) for l in labels]

    # Sort by absolute value, take top N
    idx_sorted = np.argsort(np.abs(values))[::-1][:N]
    # Re-sort for display (smallest abs at top)
    idx_display = idx_sorted[np.argsort(np.abs(values[idx_sorted]))]

    vals = values[idx_display]
    lbls = [labels[i] for i in idx_display]
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in vals]

    fig_h = max(6, N * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(range(N), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(N))
    ax.set_yticklabels(lbls, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#c44e52", label="▲ Risk (positive)"),
        Patch(facecolor="#4c72b0", label="▼ Protective (negative)"),
    ], loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def replot_panel_a(csv_dir, out_dir, n_show, split, dpi, short_names=False):
    """Standalone Panel A: single-token IG importance."""
    tok_csv = os.path.join(csv_dir, f"fig3_token_importance_{split}.csv")
    if not os.path.exists(tok_csv):
        print(f"  [panel_a] Missing {tok_csv}")
        return
    df = pd.read_csv(tok_csv).dropna(subset=["ig_mean"])
    if df.empty:
        print("  [panel_a] No IG data")
        return

    values = df["ig_mean"].values
    labels = df["token_readable"].fillna(df["token"]).values.tolist()
    sfx = "_short" if short_names else ""
    out_path = os.path.join(out_dir, f"panel_a_ig_{split}{sfx}.png")
    _plot_single_panel(values, labels,
                       f"Panel A: Integrated Gradients \u2014 single-token importance ({split.upper()})",
                       "Mean IG\n\u2190 protective    risk \u2192",
                       out_path, n_show, dpi, short_names=short_names)


def replot_panel_b(csv_dir, out_dir, n_show, split, dpi, short_names=False):
    """Standalone Panel B: single-token GradientSHAP importance."""
    # Primary source: shap_global CSV (direct cache, full values)
    shap_csv = os.path.join(csv_dir, f"shap_global_{split}.csv")
    tok_csv = os.path.join(csv_dir, f"fig3_token_importance_{split}.csv")
    
    if os.path.exists(shap_csv):
        df_shap = pd.read_csv(shap_csv).dropna(subset=["shap_mean"])
        if df_shap.empty:
            print("  [panel_b] No SHAP data in cache")
            return
        # Load readable names from token CSV if available
        name_map = {}
        if os.path.exists(tok_csv):
            df_tok = pd.read_csv(tok_csv)
            name_map = dict(zip(df_tok["token"], df_tok["token_readable"]))
        
        values = df_shap["shap_mean"].values
        labels = [name_map.get(t, t) for t in df_shap["token"].values]
    elif os.path.exists(tok_csv):
        # Fallback: token importance CSV
        df = pd.read_csv(tok_csv).dropna(subset=["shap_mean"])
        if df.empty:
            print("  [panel_b] No SHAP data")
            return
        values = df["shap_mean"].values
        labels = df["token_readable"].fillna(df["token"]).values.tolist()
    else:
        print(f"  [panel_b] Missing both {shap_csv} and {tok_csv}")
        return

    sfx = "_short" if short_names else ""
    out_path = os.path.join(out_dir, f"panel_b_shap_{split}{sfx}.png")
    _plot_single_panel(values, labels,
                       f"Panel B: GradientSHAP \u2014 single-token importance ({split.upper()})",
                       "Mean GradientSHAP\n\u2190 protective    risk \u2192",
                       out_path, n_show, dpi, short_names=short_names)


def replot_panel_c(csv_dir, out_dir, n_show, split, dpi, short_names=False):
    """Standalone Panel C: AToP temporal pattern importance."""
    pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    if not os.path.exists(pw_csv):
        print(f"  [panel_c] Missing {pw_csv}")
        return
    df = pd.read_csv(pw_csv)
    if df.empty:
        print("  [panel_c] No pattern data")
        return

    # Use readable labels if available
    label_col = "pattern_readable" if "pattern_readable" in df.columns else "pattern"
    values = df["ig_signed_mean"].values
    labels = []
    for _, r in df.iterrows():
        direction = "\u25B2" if r["ig_signed_mean"] > 0 else "\u25BC"
        labels.append(f"{direction} {r[label_col]}  (n={r['n_present']})")

    values = np.array(values)
    sfx = "_short" if short_names else ""
    out_path = os.path.join(out_dir, f"panel_c_patterns_{split}{sfx}.png")
    _plot_single_panel(values, labels,
                       f"Panel C: AToP temporal patterns ({split.upper()})",
                       "Mean \u03A3IG over pattern carriers\n\u2190 protective    risk \u2192",
                       out_path, n_show, dpi, short_names=short_names)


def replot_panel_c_split(csv_dir, out_dir, n_show, split, dpi, short_names=False):
    """Standalone Panel C split: top N/2 risk + top N/2 protective."""
    pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    if not os.path.exists(pw_csv):
        print(f"  [panel_c_split] Missing {pw_csv}")
        return
    df = pd.read_csv(pw_csv)
    if df.empty:
        print("  [panel_c_split] No pattern data")
        return

    label_col = "pattern_readable" if "pattern_readable" in df.columns else "pattern"
    half = max(1, n_show // 2)

    df_risk = df[df["ig_signed_mean"] > 0].nlargest(half, "ig_signed_mean")
    df_prot = df[df["ig_signed_mean"] < 0].nsmallest(half, "ig_signed_mean")
    n_risk = len(df_risk)
    n_prot = len(df_prot)

    if n_risk + n_prot == 0:
        print("  [panel_c_split] No risk or protective patterns")
        return

    fig, axes = plt.subplots(1, 2, figsize=(24, max(6, max(n_risk, n_prot) * 0.45)),
                              gridspec_kw={"width_ratios": [1, 1]})

    fig.suptitle(
        f"Panel C: AToP temporal patterns ({split.upper()})\n"
        f"Top {half} risk + top {half} protective by |mean \u03A3IG|",
        fontsize=12, fontweight="bold", y=1.02)

    # Left: Risk
    ax = axes[0]
    if n_risk > 0:
        df_r = df_risk.sort_values("ig_signed_mean", ascending=True)
        labels = []
        for _, r in df_r.iterrows():
            lbl = f"\u25B2 {r[label_col]}  (n={r['n_present']})"
            if short_names:
                lbl = _to_short_name(lbl)
            labels.append(lbl)
        ax.barh(range(n_risk), df_r["ig_signed_mean"].values, color="#c44e52",
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_risk))
        ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean \u03A3IG\n\u2192 risk contribution")
    ax.set_title(f"Risk patterns (IG > 0, top {half})", fontsize=10)

    # Right: Protective
    ax = axes[1]
    if n_prot > 0:
        df_pr = df_prot.sort_values("ig_signed_mean", ascending=False)
        labels = []
        for _, r in df_pr.iterrows():
            lbl = f"\u25BC {r[label_col]}  (n={r['n_present']})"
            if short_names:
                lbl = _to_short_name(lbl)
            labels.append(lbl)
        ax.barh(range(n_prot), df_pr["ig_signed_mean"].abs().values, color="#4c72b0",
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_prot))
        ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("|Mean \u03A3IG|\n\u2192 protective contribution")
    ax.set_title(f"Protective patterns (IG < 0, top {half})", fontsize=10)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#c44e52", label="\u25B2 Risk"),
        Patch(facecolor="#4c72b0", label="\u25BC Protective"),
    ], loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    sfx = "_short" if short_names else ""
    out_path = os.path.join(out_dir, f"panel_c_split_{split}{sfx}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved \u2192 {out_path}")


def replot_panel_c_stacked(csv_dir, out_dir, n_show, split, dpi, jaccard_threshold, short_names=False):
    """Panel C stacked: Jaccard-deduplicated patterns in same style as Panel A/B.
    
    Single bar chart, risk (red) and protective (blue) mixed, sorted by |IG|.
    No cluster info, no n values — clean publication figure matching Panel A/B style.
    """
    import pickle

    pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    pkl_path = os.path.join(csv_dir, f"carrier_sets_{split}.pkl")

    if not os.path.exists(pw_csv):
        print(f"  [panel_c_stacked] Missing {pw_csv}")
        return
    if not os.path.exists(pkl_path):
        print(f"  [panel_c_stacked] Missing {pkl_path}")
        return

    df = pd.read_csv(pw_csv)
    with open(pkl_path, "rb") as f:
        carriers = pickle.load(f)

    df["_carrier_keys"] = df["pattern"].map(carriers)
    df = df[df["_carrier_keys"].notna()].copy()
    df = df[df["n_present"] >= 3].copy()

    if df.empty:
        print("  [panel_c_stacked] No patterns")
        return

    print(f"  [panel_c_stacked] Clustering {len(df)} patterns at Jaccard >= {jaccard_threshold}...")

    # Jaccard clustering (same as panel_c_jaccard)
    patterns = df.sort_values("ig_signed_mean", key=abs, ascending=False).reset_index(drop=True)

    all_patients = set()
    carrier_list = []
    size_list = []
    for _, row in patterns.iterrows():
        ck = row["_carrier_keys"]
        if isinstance(ck, set):
            all_patients.update(ck)
            carrier_list.append(ck)
            size_list.append(len(ck))
        else:
            carrier_list.append(set())
            size_list.append(0)

    patient_to_idx = {p: i for i, p in enumerate(sorted(all_patients))}
    n_patients = len(patient_to_idx)

    bitvecs = []
    for carr in carrier_list:
        bits = np.zeros(n_patients, dtype=np.bool_)
        for p in carr:
            bits[patient_to_idx[p]] = True
        bitvecs.append(np.packbits(bits))

    clusters = []
    assigned = set()
    for i in range(len(patterns)):
        if i in assigned:
            continue
        if size_list[i] == 0:
            assigned.add(i)
            clusters.append((i, [i]))
            continue
        bv_i = bitvecs[i]
        sz_i = size_list[i]
        members = [i]
        for j in range(i + 1, len(patterns)):
            if j in assigned or size_list[j] == 0:
                continue
            upper_bound = min(sz_i, size_list[j]) / max(sz_i, size_list[j])
            if upper_bound < jaccard_threshold:
                continue
            intersection = int(np.unpackbits(bv_i & bitvecs[j]).sum())
            union = int(np.unpackbits(bv_i | bitvecs[j]).sum())
            if union > 0 and intersection / union >= jaccard_threshold:
                members.append(j)
                assigned.add(j)
        assigned.add(i)
        clusters.append((i, members))

    print(f"  [panel_c_stacked] {len(patterns)} \u2192 {len(clusters)} clusters")

    # Get representatives
    reps = []
    for rep_idx, _ in clusters:
        reps.append(patterns.iloc[rep_idx])
    df_reps = pd.DataFrame(reps)

    # Select top N/2 risk + N/2 protective, then merge and sort by |IG|
    label_col = "pattern_readable" if "pattern_readable" in df_reps.columns else "pattern"
    half = max(1, n_show // 2)

    df_risk = df_reps[df_reps["ig_signed_mean"] > 0].nlargest(half, "ig_signed_mean")
    df_prot = df_reps[df_reps["ig_signed_mean"] < 0].nsmallest(half, "ig_signed_mean")
    top = pd.concat([df_risk, df_prot])

    N = len(top)
    if N == 0:
        print("  [panel_c_stacked] No patterns to plot")
        return

    # Sort by absolute IG (smallest at top for display)
    top = top.reindex(top["ig_signed_mean"].abs().sort_values(ascending=True).index)

    values = top["ig_signed_mean"].values
    labels = []
    for _, r in top.iterrows():
        if short_names and "pattern" in r.index:
            lbl = _format_pattern_short(r["pattern"])
        else:
            lbl = str(r[label_col])
        labels.append(lbl)

    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in values]

    fig_h = max(8, N * 0.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.barh(range(N), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean \u03A3IG over pattern carriers\n\u2190 protective    risk \u2192", fontsize=11)

    jt_pct = int(jaccard_threshold * 100)
    ax.set_title(
        f"Panel C: AToP temporal patterns ({split.upper()})\n"
        f"Jaccard-deduplicated (\u2265 {jaccard_threshold}), one representative per phenotype",
        fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#c44e52", label="\u25B2 Risk (positive IG)"),
        Patch(facecolor="#4c72b0", label="\u25BC Protective (negative IG)"),
    ], loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    sfx = "_short" if short_names else ""
    out_path = os.path.join(out_dir, f"panel_c_stacked_{split}_j{jt_pct}{sfx}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved \u2192 {out_path}")


def replot_panel_ac_stacked(csv_dir, out_dir, n_show, split, dpi, jaccard_threshold, short_names=False):
    """Merged Panel A + C: single tokens and temporal patterns in one chart.
    
    Takes top single tokens from Panel A (IG) and top Jaccard-deduplicated
    patterns from Panel C, merges them, sorts by |IG|. Tokens marked with
    a dot prefix, patterns with an arrow indicator.
    """
    import pickle

    tok_csv = os.path.join(csv_dir, f"fig3_token_importance_{split}.csv")
    pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    pkl_path = os.path.join(csv_dir, f"carrier_sets_{split}.pkl")

    if not os.path.exists(tok_csv) or not os.path.exists(pw_csv) or not os.path.exists(pkl_path):
        print(f"  [panel_ac_stacked] Missing required files")
        return

    # --- Panel A: ALL single tokens with IG ---
    df_tok = pd.read_csv(tok_csv).dropna(subset=["ig_mean"])

    tok_entries = []
    for _, r in df_tok.iterrows():
        lbl = str(r["token_readable"]) if pd.notna(r.get("token_readable")) else r["token"]
        tok_entries.append({"label": lbl, "value": r["ig_mean"], "type": "token"})

    # --- Panel C: Jaccard-deduplicated patterns ---
    df = pd.read_csv(pw_csv)
    with open(pkl_path, "rb") as f:
        carriers = pickle.load(f)

    df["_carrier_keys"] = df["pattern"].map(carriers)
    df = df[df["_carrier_keys"].notna()].copy()
    df = df[df["n_present"] >= 3].copy()

    if not df.empty:
        # Jaccard clustering
        patterns = df.sort_values("ig_signed_mean", key=abs, ascending=False).reset_index(drop=True)
        all_patients = set()
        carrier_list, size_list = [], []
        for _, row in patterns.iterrows():
            ck = row["_carrier_keys"]
            if isinstance(ck, set):
                all_patients.update(ck)
                carrier_list.append(ck)
                size_list.append(len(ck))
            else:
                carrier_list.append(set())
                size_list.append(0)

        patient_to_idx = {p: i for i, p in enumerate(sorted(all_patients))}
        n_patients = len(patient_to_idx)
        bitvecs = []
        for carr in carrier_list:
            bits = np.zeros(n_patients, dtype=np.bool_)
            for p in carr:
                bits[patient_to_idx[p]] = True
            bitvecs.append(np.packbits(bits))

        clusters, assigned = [], set()
        for i in range(len(patterns)):
            if i in assigned:
                continue
            if size_list[i] == 0:
                assigned.add(i)
                clusters.append((i, [i]))
                continue
            bv_i, sz_i = bitvecs[i], size_list[i]
            members = [i]
            for j in range(i + 1, len(patterns)):
                if j in assigned or size_list[j] == 0:
                    continue
                if min(sz_i, size_list[j]) / max(sz_i, size_list[j]) < jaccard_threshold:
                    continue
                inter = int(np.unpackbits(bv_i & bitvecs[j]).sum())
                union = int(np.unpackbits(bv_i | bitvecs[j]).sum())
                if union > 0 and inter / union >= jaccard_threshold:
                    members.append(j)
                    assigned.add(j)
            assigned.add(i)
            clusters.append((i, members))

        df_reps = pd.DataFrame([patterns.iloc[rep_idx] for rep_idx, _ in clusters])
        label_col = "pattern_readable" if "pattern_readable" in df_reps.columns else "pattern"

        pat_entries = []
        for _, r in df_reps.iterrows():
            if short_names and "pattern" in r.index:
                lbl = _format_pattern_short(r["pattern"])
            else:
                lbl = str(r[label_col])
            pat_entries.append({"label": lbl, "value": r["ig_signed_mean"], "type": "pattern"})
    else:
        pat_entries = []

    # --- Merge and rank by |value|, take top N ---
    all_entries = tok_entries + pat_entries
    all_entries.sort(key=lambda e: abs(e["value"]), reverse=True)
    all_entries = all_entries[:n_show]
    # Re-sort for display: smallest at top
    all_entries.sort(key=lambda e: abs(e["value"]))

    N = len(all_entries)
    if N == 0:
        print("  [panel_ac_stacked] No data")
        return

    values = np.array([e["value"] for e in all_entries])
    labels = []
    for e in all_entries:
        lbl = e["label"]
        if short_names:
            lbl = _to_short_name(lbl)
        # Add type indicator
        if e["type"] == "pattern":
            lbl = "\u25B8 " + lbl  # ▸ for patterns
        else:
            lbl = "\u2022 " + lbl  # • for single tokens
        labels.append(lbl)

    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in values]
    # Lighter shade for single tokens, full shade for patterns
    colors_final = []
    for i, e in enumerate(all_entries):
        if e["type"] == "token":
            colors_final.append("#e8a0a2" if values[i] > 0 else "#a0b8d0")
        else:
            colors_final.append("#c44e52" if values[i] > 0 else "#4c72b0")

    fig_h = max(8, N * 0.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.barh(range(N), values, color=colors_final, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean IG attribution\n\u2190 protective    risk \u2192", fontsize=11)

    jt_pct = int(jaccard_threshold * 100)
    ax.set_title(
        f"Combined: single-token IG + AToP temporal patterns ({split.upper()})\n"
        f"\u2022 Single tokens (light)    \u25B8 Temporal patterns (bold, Jaccard \u2265 {jaccard_threshold})",
        fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#c44e52", label="\u25B8 Risk pattern"),
        Patch(facecolor="#e8a0a2", label="\u2022 Risk token"),
        Patch(facecolor="#4c72b0", label="\u25B8 Protective pattern"),
        Patch(facecolor="#a0b8d0", label="\u2022 Protective token"),
    ], loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    sfx = "_short" if short_names else ""
    out_path = os.path.join(out_dir, f"panel_ac_stacked_{split}_j{jt_pct}{sfx}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved \u2192 {out_path}")


def replot_panel_c_jaccard(csv_dir, out_dir, n_show, split, dpi, jaccard_threshold, short_names=False):
    """Panel C using Jaccard cluster representatives — one pattern per phenotype.
    
    Runs Jaccard clustering on carrier sets, picks one representative per cluster
    (highest |IG|), then shows top N/2 risk + N/2 protective representatives.
    Each bar represents a distinct patient phenotype.
    """
    import pickle

    pw_csv = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    pkl_path = os.path.join(csv_dir, f"carrier_sets_{split}.pkl")

    if not os.path.exists(pw_csv):
        print(f"  [panel_c_jaccard] Missing {pw_csv}")
        return
    if not os.path.exists(pkl_path):
        print(f"  [panel_c_jaccard] Missing {pkl_path}")
        return

    df = pd.read_csv(pw_csv)
    with open(pkl_path, "rb") as f:
        carriers = pickle.load(f)

    df["_carrier_keys"] = df["pattern"].map(carriers)
    df = df[df["_carrier_keys"].notna()].copy()
    df = df[df["n_present"] >= 3].copy()

    if df.empty:
        print("  [panel_c_jaccard] No patterns with carrier sets")
        return

    print(f"  [panel_c_jaccard] Clustering {len(df)} patterns at Jaccard >= {jaccard_threshold}...")

    # Run Jaccard clustering (same logic as fig_supp_jaccard_clusters)
    patterns = df.sort_values("ig_signed_mean", key=abs, ascending=False).reset_index(drop=True)

    # Build bit vectors for fast Jaccard
    all_patients = set()
    carrier_list = []
    size_list = []
    for _, row in patterns.iterrows():
        ck = row["_carrier_keys"]
        if isinstance(ck, set):
            all_patients.update(ck)
            carrier_list.append(ck)
            size_list.append(len(ck))
        else:
            carrier_list.append(set())
            size_list.append(0)

    patient_to_idx = {p: i for i, p in enumerate(sorted(all_patients))}
    n_patients = len(patient_to_idx)

    bitvecs = []
    for carr in carrier_list:
        bits = np.zeros(n_patients, dtype=np.bool_)
        for p in carr:
            bits[patient_to_idx[p]] = True
        bitvecs.append(np.packbits(bits))

    clusters = []
    assigned = set()
    for i in range(len(patterns)):
        if i in assigned:
            continue
        if size_list[i] == 0:
            assigned.add(i)
            clusters.append((i, [i]))
            continue
        bv_i = bitvecs[i]
        sz_i = size_list[i]
        members = [i]
        for j in range(i + 1, len(patterns)):
            if j in assigned or size_list[j] == 0:
                continue
            upper_bound = min(sz_i, size_list[j]) / max(sz_i, size_list[j])
            if upper_bound < jaccard_threshold:
                continue
            intersection = int(np.unpackbits(bv_i & bitvecs[j]).sum())
            union = int(np.unpackbits(bv_i | bitvecs[j]).sum())
            if union > 0 and intersection / union >= jaccard_threshold:
                members.append(j)
                assigned.add(j)
        assigned.add(i)
        clusters.append((i, members))

    print(f"  [panel_c_jaccard] {len(patterns)} → {len(clusters)} clusters")

    # Build representative list
    reps = []
    for rep_idx, member_idxs in clusters:
        row = patterns.iloc[rep_idx].copy()
        row["cluster_size"] = len(member_idxs)
        cluster_patients = set().union(*(carrier_list[m] for m in member_idxs))
        row["cluster_patients"] = len(cluster_patients)
        reps.append(row)
    df_reps = pd.DataFrame(reps)

    label_col = "pattern_readable" if "pattern_readable" in df_reps.columns else "pattern"
    half = max(1, n_show // 2)

    df_risk = df_reps[df_reps["ig_signed_mean"] > 0].nlargest(half, "ig_signed_mean")
    df_prot = df_reps[df_reps["ig_signed_mean"] < 0].nsmallest(half, "ig_signed_mean")
    n_risk = len(df_risk)
    n_prot = len(df_prot)

    if n_risk + n_prot == 0:
        print("  [panel_c_jaccard] No risk or protective clusters")
        return

    fig, axes = plt.subplots(1, 2, figsize=(26, max(6, max(n_risk, n_prot) * 0.5)),
                              gridspec_kw={"width_ratios": [1, 1]})

    jt_pct = int(jaccard_threshold * 100)
    fig.suptitle(
        f"Panel C: AToP temporal patterns — Jaccard-deduplicated ({split.upper()})\n"
        f"One representative per phenotype cluster (Jaccard \u2265 {jaccard_threshold}), "
        f"{len(clusters)} clusters from {len(patterns)} patterns",
        fontsize=11, fontweight="bold", y=1.02)

    # Left: Risk
    ax = axes[0]
    if n_risk > 0:
        df_r = df_risk.sort_values("ig_signed_mean", ascending=True)
        labels = []
        for _, r in df_r.iterrows():
            lbl = f"\u25B2 {r[label_col]}"
            if short_names:
                lbl = _to_short_name(lbl)
            lbl += f"\n    cluster={int(r['cluster_size'])}, patients={int(r['cluster_patients'])}, n={int(r['n_present'])}"
            labels.append(lbl)
        ax.barh(range(n_risk), df_r["ig_signed_mean"].values, color="#c44e52",
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_risk))
        ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean \u03A3IG\n\u2192 risk contribution")
    ax.set_title(f"Risk phenotypes (top {half})", fontsize=10)

    # Right: Protective
    ax = axes[1]
    if n_prot > 0:
        df_pr = df_prot.sort_values("ig_signed_mean", ascending=False)
        labels = []
        for _, r in df_pr.iterrows():
            lbl = f"\u25BC {r[label_col]}"
            if short_names:
                lbl = _to_short_name(lbl)
            lbl += f"\n    cluster={int(r['cluster_size'])}, patients={int(r['cluster_patients'])}, n={int(r['n_present'])}"
            labels.append(lbl)
        ax.barh(range(n_prot), df_pr["ig_signed_mean"].abs().values, color="#4c72b0",
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(n_prot))
        ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("|Mean \u03A3IG|\n\u2192 protective contribution")
    ax.set_title(f"Protective phenotypes (top {half})", fontsize=10)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#c44e52", label="\u25B2 Risk"),
        Patch(facecolor="#4c72b0", label="\u25BC Protective"),
    ], loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    sfx = "_short" if short_names else ""
    out_path = os.path.join(out_dir, f"panel_c_jaccard_{split}_j{jt_pct}{sfx}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved \u2192 {out_path}")


def replot_attn_flow(csv_dir, out_dir, n_show, split, dpi, short_names=False):
    """Replot attention flow from cached CSV."""
    attn_csv = os.path.join(csv_dir, f"attention_flow_{split}.csv")
    if not os.path.exists(attn_csv):
        print(f"  [attn_flow] Missing {attn_csv}")
        return

    df_attn = pd.read_csv(attn_csv)
    if df_attn.empty:
        print("  [attn_flow] No data")
        return

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from atop.explain.attention_flow import fig_attention_flow

    # Load short names if needed
    if short_names:
        try:
            from atop.explain.figures import load_short_names, _SHORT_NAME_MAP
            if not _SHORT_NAME_MAP:
                sn_path = os.path.join(os.path.dirname(__file__), "short_names.csv")
                load_short_names(sn_path)
        except Exception:
            pass

    tok_csv = os.path.join(csv_dir, f"fig3_token_importance_{split}.csv")
    icd_titles = {}
    if os.path.exists(tok_csv):
        icd_titles = _build_icd_titles(pd.read_csv(tok_csv))

    fig_attention_flow(out_dir, df_attn, icd_titles=icd_titles,
                        n_show=n_show, split=split, short_names=short_names)
    print(f"  [attn_flow] Done")


def replot_jaccard(csv_dir, supp_dir, n_show, split, dpi, jaccard_threshold, short_names=False):
    """Replot Jaccard cluster supplement from saved carrier sets."""
    import pickle

    # Load pathway CSV and carrier sets pickle
    csv_path = os.path.join(csv_dir, f"fig3_pathway_importance_{split}.csv")
    pkl_path = os.path.join(csv_dir, f"carrier_sets_{split}.pkl")

    if not os.path.exists(csv_path):
        print(f"  [jaccard] Missing {csv_path}")
        return
    if not os.path.exists(pkl_path):
        print(f"  [jaccard] Missing {pkl_path} — run pipeline first to generate carrier sets")
        return

    df = pd.read_csv(csv_path)
    with open(pkl_path, "rb") as f:
        carriers = pickle.load(f)

    print(f"  [jaccard] Loaded {len(df)} patterns, {len(carriers)} carrier sets")
    print(f"  [jaccard] Threshold: {jaccard_threshold}")

    # Attach carrier sets to df
    df["_carrier_keys"] = df["pattern"].map(carriers)
    df = df[df["_carrier_keys"].notna()].copy()
    df = df[df["n_present"] >= 3].copy()

    if df.empty:
        print("  [jaccard] No patterns with carrier sets")
        return

    # Load icd titles if available
    tok_csv = os.path.join(csv_dir, f"fig3_token_importance_{split}.csv")
    icd_titles = {}
    if os.path.exists(tok_csv):
        icd_titles = _build_icd_titles(pd.read_csv(tok_csv))

    # Load drug name mappings for RxCUI resolution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from atop.explain.label_utils import set_drug_names
        from atop.utils import load_drug_names
        # csv_dir is {run_dir}/explain/figures/csv → go up 3 levels to run_dir
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(csv_dir)))
        config_path = os.path.join(run_dir, "config.json")
        print(f"  [jaccard] Looking for config at {config_path}")
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            drug_mapping = cfg.get("drug_mapping", "")
            if drug_mapping and os.path.exists(drug_mapping):
                drug_names = load_drug_names(drug_mapping)
                set_drug_names(drug_names)
                print(f"  [jaccard] Loaded {len(drug_names)} drug name mappings")
            else:
                print(f"  [jaccard] Drug mapping not found: {drug_mapping}")
        else:
            print(f"  [jaccard] Config not found at {config_path}")
    except Exception as e:
        print(f"  [jaccard] Drug name loading skipped: {e}")

    from atop.explain.figures import fig_supp_jaccard_clusters

    fig_supp_jaccard_clusters(supp_dir, df, icd_titles, n_show=n_show,
                               split=split, jaccard_threshold=jaccard_threshold,
                               short_names=short_names, csv_dir=csv_dir)
    print(f"  [jaccard] Done (threshold={jaccard_threshold})")


if __name__ == "__main__":
    main()
