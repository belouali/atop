"""Clinical label shortening and formatting utilities.

Provides consistent, readable labels for ICD codes and drug names across
all figures. Two-level approach:
  1. Clinical abbreviation map — well-known conditions get standard abbreviations
  2. Generic word shortenings — common filler words abbreviated

Drug names are resolved via a module-level registry set by set_drug_names().
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

# Module-level drug name registry: D:RX_XXXX → ingredient name
_DRUG_NAMES: Dict[str, str] = {}


def set_drug_names(drug_names: Dict[str, str]):
    """Set the module-level drug name lookup (called once by explainer)."""
    global _DRUG_NAMES
    _DRUG_NAMES = drug_names or {}


def get_drug_name(tok: str) -> str:
    """Look up a drug token's readable name, or return the raw token."""
    if tok in _DRUG_NAMES:
        return _DRUG_NAMES[tok]
    return tok[2:] if tok.startswith("D:") else tok

# ═══════════════════════════════════════════════════════════════════════
# Level 1: Clinical abbreviations (condition → standard abbreviation)
# ═══════════════════════════════════════════════════════════════════════

CLINICAL_ABBREVIATIONS = {
    # Cardiovascular
    "Congestive heart failure": "CHF",
    "congestive heart failure": "CHF",
    "Essential (primary) hypertension": "HTN",
    "essential (primary) hypertension": "HTN",
    "Unspecified essential hypertension": "HTN, unsp.",
    "Coronary atherosclerosis of native coronary artery": "CAD (native)",
    "Coronary atherosclerosis": "CAD",
    "Atrial fibrillation": "AFib",
    "atrial fibrillation": "AFib",
    "Unspecified atrial fibrillation": "AFib, unsp.",
    "Aortic valve disorders": "Aortic valve dis.",
    # Metabolic
    "Type 2 diabetes mellitus": "T2DM",
    "Diabetes mellitus": "DM",
    "Hyperlipidemia": "HLD",
    "Other and unspecified hyperlipidemia": "HLD, unsp.",
    "Hypo-osmolality and hyponatremia": "Hyponatremia",
    # Renal
    "Chronic kidney disease": "CKD",
    "End stage renal disease": "ESRD",
    "Acute kidney failure": "AKI",
    # Psych
    "Major depressive disorder, single episode, unspecified": "MDD, single ep.",
    "Major depressive disorder": "MDD",
    "Major depressive affective disorder": "MDD",
    "Suicidal ideations": "SI",
    "Suicidal ideation": "SI",
    "Unspecified psychosis": "Psychosis, unsp.",
    "Unspecified psychosis not due to a substance or known physiological condition": "Psychosis, unsp.",
    "Bipolar I disorder, most recent episode (or current) unspecified": "Bipolar I, unsp.",
    "Bipolar disorder, current episode manic": "Bipolar, manic",
    "Schizoaffective disorder, unspecified": "Schizoaffective, unsp.",
    "Unspecified episodic mood disorder": "Mood disorder, unsp.",
    "Depressive disorder, not elsewhere classified": "Depressive dis. NEC",
    # Respiratory
    "Chronic obstructive pulmonary disease": "COPD",
    "Unspecified asthma": "Asthma, unsp.",
    "Pneumonia": "PNA",
    # Substance
    "Alcohol abuse, unspecified": "EtOH abuse, unsp.",
    "Alcohol abuse with intoxication, unspecified": "EtOH intox., unsp.",
    "Acute alcoholic intoxication in alcoholism": "Acute EtOH intox.",
    "Nicotine dependence, unspecified, uncomplicated": "Nicotine dep.",
    "Nicotine dependence": "Nicotine dep.",
    "Personal history of nicotine dependence": "Hx nicotine dep.",
    "Opioid abuse, unspecified": "Opioid abuse, unsp.",
    # Pain
    "Chest pain, unspecified": "Chest pain, unsp.",
    "Other chest pain": "Chest pain, other",
    "Neoplasm related pain (acute) (chronic)": "Neoplasm pain",
    # Neuro
    "Syncope and collapse": "Syncope",
    "Cerebral edema": "Cerebral edema",
    "Headache": "Headache",
    # OB/trauma
    "Single live birth": "Single live birth",
    "Motorcycle driver injured in collision with fixed or stationary object": "Motorcycle crash (fixed obj.)",
    "Motorcycle driver injured in collision": "Motorcycle crash",
    # COVID
    "Contact with and (suspected) exposure to COVID-19": "COVID-19 exposure",
    # Admin
    "Personal history of": "Hx",
    "Physical restraints status": "Restraints",
    # Other
    "Anemia in neoplastic disease": "Anemia (neoplastic)",
    "Dehydration": "Dehydration",
    "Obstruction of bile duct": "Bile duct obstruction",
    "Other specified metabolic disorders": "Metabolic dis., other",
    "Homelessness": "Homelessness",
}

# ═══════════════════════════════════════════════════════════════════════
# Level 2: Generic word shortenings
# ═══════════════════════════════════════════════════════════════════════

WORD_SHORTENINGS = [
    ("unspecified", "unsp."),
    ("without mention of", "w/o"),
    ("not elsewhere classified", "NEC"),
    ("not otherwise specified", "NOS"),
    ("Contact with and (suspected) exposure to", "Exposure:"),
    ("Personal history of", "Hx:"),
    ("Other and unspecified", "Other/unsp."),
    ("Antineoplastic and immunosuppressive drugs causing adverse effects in therapeutic use",
     "Chemo/immunosup. adverse effects"),
    ("Antineoplastic and immunosuppressive", "Chemo/immunosup."),
    ("Other current conditions classifiable elsewhere of mother", "Maternal conditions NEC"),
    ("Other and unspecified cord entanglement", "Cord entanglement"),
    ("Gestational [pregnancy-induced] hypertension without significant proteinuria",
     "Gestational HTN"),
    ("Elevated blood-pressure reading, without diagnosis of hypertension",
     "Elevated BP reading"),
    # Generic
    (", delivered, with or without mention of antepartum condition", ", delivered"),
    (", unspecified as to episode of care or not applicable", ""),
    ("in diseases classified elsewhere", "in other dis."),
    ("complicating pregnancy, childbirth, or the puerperium", "in pregnancy"),
]


# ═══════════════════════════════════════════════════════════════════════
# Main formatting functions
# ═══════════════════════════════════════════════════════════════════════

def shorten_title(title: str) -> str:
    """Apply clinical abbreviations then generic shortenings to a title string."""
    if not title:
        return title
    
    # Level 1: exact match on known clinical phrases
    for phrase, abbrev in CLINICAL_ABBREVIATIONS.items():
        if title == phrase:
            return abbrev
        # Also try as substring for longer titles
        if phrase in title:
            title = title.replace(phrase, abbrev)
            break  # only apply first match to avoid double-replacement
    
    # Level 2: generic word shortenings
    for long, short in WORD_SHORTENINGS:
        if long in title:
            title = title.replace(long, short)
    
    # Clean up artifacts
    title = title.strip().rstrip(",").strip()
    title = re.sub(r"\s+", " ", title)  # collapse whitespace
    
    return title


def format_token_short(tok: str, icd_titles: Dict[Tuple[str, str], str]) -> str:
    """Format a token with shortened clinical labels.
    
    Like format_token_readable but applies clinical abbreviations.
    Drug tokens are resolved via the module-level _DRUG_NAMES registry.
    """
    if tok.startswith("D:"):
        name = get_drug_name(tok)
        return f"D:{name}"
    if ":" in tok:
        prefix = tok[0]
        # Parse version and code
        rest = tok.split(":", 1)[1]
        if "_" in rest:
            ver, code = rest.split("_", 1)
        else:
            ver, code = "", rest
        lookup_key = ("D" if prefix == "C" else "P", code)
        title = icd_titles.get(lookup_key, "")
        if title:
            short = shorten_title(title)
            return f"{prefix}:{code} ({short})"
    return tok


def format_pattern_short(pat_str: str, icd_titles: Dict,
                          max_tok_len: int = 50) -> str:
    """Format a pattern string with shortened labels."""
    parts = pat_str.split(" -> ")
    readable_parts = []
    for p in parts:
        p = p.strip()
        if p.startswith("{") and p.endswith("}"):
            items = [x.strip() for x in p[1:-1].split(",")]
            readable = [format_token_short(it, icd_titles) for it in items]
            readable = [r if len(r) <= max_tok_len else r[:max_tok_len-1] + "…"
                        for r in readable]
            readable_parts.append("{" + ", ".join(readable) + "}")
        else:
            r = format_token_short(p, icd_titles)
            if len(r) > max_tok_len:
                r = r[:max_tok_len-1] + "…"
            readable_parts.append(r)
    return " → ".join(readable_parts)


def active_stream_legend(tokens_or_df, legend_type="patch"):
    """Build legend elements only for streams present in the data.
    
    Args:
        tokens_or_df: list of token strings, or DataFrame with 'token_str' column
        legend_type: 'patch' for Patch elements, 'line' for Line2D markers
    
    Returns:
        list of legend handles
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    if hasattr(tokens_or_df, "columns"):
        # DataFrame
        col = "token_str" if "token_str" in tokens_or_df.columns else "token"
        all_tokens = tokens_or_df[col].tolist() if col in tokens_or_df.columns else []
    elif isinstance(tokens_or_df, (list, set)):
        all_tokens = list(tokens_or_df)
    else:
        all_tokens = []
    
    streams_present = set()
    for t in all_tokens:
        t = str(t)
        if t.startswith("C:"):
            streams_present.add("C")
        elif t.startswith("P:"):
            streams_present.add("P")
        elif t.startswith("D:"):
            streams_present.add("D")
    
    stream_info = [
        ("C", "#4C72B0", "C: Conditions"),
        ("P", "#DD8452", "P: Procedures"),
        ("D", "#55A868", "D: Drugs"),
    ]
    
    elements = []
    for key, color, label in stream_info:
        if key in streams_present:
            if legend_type == "line":
                elements.append(
                    Line2D([0], [0], marker="s", color="w",
                           markerfacecolor=color, markersize=7, label=label))
            else:
                elements.append(Patch(facecolor=color, label=label))
    
    return elements
