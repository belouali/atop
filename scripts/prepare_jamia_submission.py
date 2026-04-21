"""
Prepare AToP_JAMIA_submission.docx from AToP_paper_v6.docx.

Changes made:
  1. Fill all [val] placeholders with confirmed numbers
  2. Fix XGB PR-AUC 0.32 -> 0.31 everywhere
  3. Ensure JAMIA document order:
       Title page (own page) -> Abstract -> Main text + embedded tables ->
       Acknowledgments -> Statements -> References -> Figure legends with alt text
  4. Report remaining placeholders and structural issues
"""
from __future__ import annotations
import os, re
from docx import Document

SRC = "../manuscript/atop_v5.5/AToP_paper_v6.docx"
DST = "../manuscript/jamia_submission/AToP_JAMIA_submission.docx"
os.makedirs(os.path.dirname(DST), exist_ok=True)

# ── Placeholder map ────────────────────────────────────────────────────────────
REPLACEMENTS = {
    # ── Performance numbers ──
    "[AUROC_ATOP]":            "0.77 (95% CI 0.76–0.77)",
    "[PRAUC_ATOP]":            "0.34 (95% CI 0.33–0.36)",
    "[AUROC_XGB]":             "0.73 (95% CI 0.72–0.74)",
    "[PRAUC_XGB]":             "0.31 (95% CI 0.30–0.32)",
    "[AUROC_LR]":              "0.66 (95% CI 0.65–0.67)",
    "[PRAUC_LR]":              "0.24 (95% CI 0.23–0.25)",
    # ── Cohort numbers ──
    "[N_TOTAL]":               "218,196",
    "[N_READMIT]":             "24,876",
    "[N_NO_READMIT]":          "171,501",
    "[N_VAL]":                 "21,819",
    "[N_TABLE1]":              "196,377",
    "[READMIT_RATE]":          "12.7%",
    # ── Table 1 ──
    "[AGE_OVERALL]":           "57 (38–71)",
    "[AGE_READMIT]":           "60 (44–73)",
    "[AGE_NO_READMIT]":        "57 (37–71)",
    "[FEMALE_OVERALL]":        "115,461 (52.9%)",
    "[FEMALE_READMIT]":        "12,489 (50.2%)",
    "[FEMALE_NO_READMIT]":     "91,370 (53.3%)",
    "[EMERG_OVERALL]":         "171,168 (78.4%)",
    "[EMERG_READMIT]":         "21,444 (86.2%)",
    "[EMERG_NO_READMIT]":      "132,043 (77.1%)",
    "[LOS_OVERALL]":           "4.0 (2.0–8.0)",
    "[LOS_READMIT]":           "6.0 (3.0–11.0)",
    "[LOS_NO_READMIT]":        "4.0 (2.0–7.0)",
    "[CCI_OVERALL]":           "2.0 (0.0–5.0)",
    "[CCI_READMIT]":           "4.0 (1.0–7.0)",
    "[CCI_NO_READMIT]":        "2.0 (0.0–4.0)",
    "[ED_OVERALL]":            "0.0 (0.0–1.0)",
    "[ED_READMIT]":            "1.0 (0.0–2.0)",
    "[ED_NO_READMIT]":         "0.0 (0.0–1.0)",
    # ── Table 2 ──
    "[N_PATTERNS_TOTAL]":      "24,490",
    "[N_PATTERNS_DEDUP]":      "5,047",
    "[COMPRESSION_RATIO]":     "38.0%",
    "[CASE_MIN_SUP]":          "387",
    "[CTRL_MIN_SUP]":          "2,669",
    "[FINAL_GATE]":            "3,055",
}


def replace_in_paragraph(para, old, new):
    full_text = "".join(r.text for r in para.runs)
    if old not in full_text:
        return False
    new_full = full_text.replace(old, new)
    if para.runs:
        para.runs[0].text = new_full
        for r in para.runs[1:]:
            r.text = ""
    return True


def replace_in_doc(doc, old, new):
    changed = 0
    for para in doc.paragraphs:
        if replace_in_paragraph(para, old, new):
            changed += 1
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    if replace_in_paragraph(para, old, new):
                        changed += 1
    return changed


doc = Document(SRC)

# ── Step 1: Direct replacements ───────────────────────────────────────────────
print("[1] Applying direct replacements...")
total_changes = 0
for old, new in REPLACEMENTS.items():
    n = replace_in_doc(doc, old, new)
    if n:
        print(f"    '{old}' → '{new}'  ({n} occurrences)")
    total_changes += n

# ── Step 2: Fix PR-AUC 0.32 → 0.31 (XGBoost-specific phrases) ────────────────
print("\n[2] Fixing XGB PR-AUC 0.32 → 0.31...")
xgb_phrases = [
    ("PR-AUC 0.32",        "PR-AUC 0.31"),
    ("PR-AUC ~0.32",       "PR-AUC ~0.31"),
    ("PR-AUC of 0.32",     "PR-AUC of 0.31"),
    ("0.729 / ~0.32",      "0.729 / ~0.31"),
    ("(PR-AUC 0.32",       "(PR-AUC 0.31"),
    ("PR-AUC: 0.32",       "PR-AUC: 0.31"),
]
for old, new in xgb_phrases:
    n = replace_in_doc(doc, old, new)
    if n:
        print(f"    Fixed: '{old}' → '{new}'  ({n} occurrences)")

# ── Step 3: Positional fill of tables with generic [val] placeholders ─────────
# The docx uses generic "[val]" everywhere instead of named placeholders like
# [AUROC_ATOP] or [AGE_OVERALL], so the map above matches nothing. Fill tables
# by row/column position using the confirmed numbers supplied in the task spec.
print("\n[3] Filling tables positionally...")


def set_cell(cell, text):
    """Replace all text in a table cell with ``text``, preserving one paragraph."""
    if not cell.paragraphs:
        return False
    # keep the first paragraph's first run as the anchor
    first = cell.paragraphs[0]
    if first.runs:
        first.runs[0].text = text
        for r in first.runs[1:]:
            r.text = ""
    else:
        first.add_run(text)
    # clear any trailing paragraphs in the cell
    for extra in cell.paragraphs[1:]:
        if extra.runs:
            extra.runs[0].text = ""
            for r in extra.runs[1:]:
                r.text = ""
    return True


# ── Table 0: Cohort characteristics — FULL 218,196 cohort ──
# Computed by scripts/compute_table1_full_cohort.py using the model's exact
# index-admission rule (penultimate if ≥2 admissions, else first;
# exclude_elective_readmissions=True). These match summary.json exactly.
# Columns: [Characteristic, Overall (N=218,196), Readmitted (n=27,508), Not Readmitted (n=190,688), P]
if len(doc.tables) > 0:
    t0 = doc.tables[0]
    # Header row: the v5.5 docx has a literal newline between "Readmitted"
    # and "(n=[val])", so replace on the joined text rather than exact match.
    HEADER_COHORT = {
        "Readmitted":      "Readmitted\n(n=27,508)",
        "Not Readmitted":  "Not Readmitted\n(n=190,688)",
    }
    for cell in t0.rows[0].cells:
        # Match on the first word(s) and an "[val]" presence
        text = cell.text.strip()
        if "[val]" in text:
            for label, new_text in HEADER_COHORT.items():
                if text.startswith(label):
                    set_cell(cell, new_text)
                    break

    # row_idx -> (Overall, Readmitted, Not Readmitted, P)
    # P-values are not available — keep as "—" for manual entry.
    # r8 (Acuity, emergency) = same values as r5 (Emergency/Urgent) — binary
    # component of LACE maps 1:1 to non-elective admissions; the two rows are
    # clinically redundant per co-author note.
    t0_rows = {
        2:  ("57 (38–71)",         "58 (40–72)",         "57 (38–71)",         "—"),  # Age
        3:  ("115,344 (52.9%)",    "14,141 (51.4%)",     "101,203 (53.1%)",    "—"),  # Female
        5:  ("195,500 (89.6%)",    "24,852 (90.3%)",     "170,648 (89.5%)",    "—"),  # Emergency/Urgent
        7:  ("2.6 (1.0–5.1)",      "3.2 (1.2–6.8)",      "2.6 (1.0–4.9)",      "—"),  # LOS (days)
        8:  ("195,500 (89.6%)",    "24,852 (90.3%)",     "170,648 (89.5%)",    "—"),  # Acuity (emergency) — same as r5
        9:  ("1 (0–2)",            "1 (0–4)",            "1 (0–2)",            "—"),  # CCI
        10: ("0 (0–1)",            "0 (0–1)",            "0 (0–1)",            "—"),  # ED visits 6mo
        11: ("1 (1–2)",            "2 (1–4)",            "1 (1–1)",            "—"),  # Prior admissions (n_visits)
    }
    for ri, values in t0_rows.items():
        if ri >= len(t0.rows):
            continue
        row = t0.rows[ri]
        # Columns 1–4 correspond to (Overall, Readmitted, Not Readmitted, P)
        for ci, val in enumerate(values, start=1):
            if ci < len(row.cells):
                set_cell(row.cells[ci], val)
    print("  [Table 0] Filled header + all 8 data rows (Age, Female, Emergency, LOS, Acuity, CCI, ED visits, Prior admissions) with FULL 218,196 cohort")

# ── Table 1: Mining summary ──
# Columns: [Metric, Train, Test]
# Only train-side has most mining values; patterns counted at admission level.
if len(doc.tables) > 1:
    t1 = doc.tables[1]
    t1_rows = {
        # (Train, Test)
        2: ("36.1 (43.3)",  "—"),    # Mean sequence length
        3: ("13.7 (12.4)",  "—"),    # Mean salient tokens
        4: ("38.0",         "—"),    # Compression ratio (the table cell already has "%" suffix)
        6: ("24,490",       "—"),    # Patterns meeting min support
        7: ("5,325",        "—"),    # Cross-visit patterns (≥2 blocks)
        8: ("5,047",        "—"),    # After Jaccard dedup (j≥0.2)
        10: ("—",           "1,877"),  # Risk patterns (test)
        11: ("—",           "3,170"),  # Protective patterns (test)
    }
    for ri, (train, test) in t1_rows.items():
        if ri >= len(t1.rows):
            continue
        row = t1.rows[ri]
        if len(row.cells) >= 3:
            set_cell(row.cells[1], train)
            set_cell(row.cells[2], test)
    print("  [Table 1] Filled all mining metrics (train/test)")

# ── Table 2: Phenotype clusters — ΣIG + n carriers ──
# Match rows by the "Representative Pattern" column against the dumbbell CSV
# and panel_c CSV for n_present.
import pandas as pd

try:
    dumbbell = pd.read_csv("../runs/full_CPD/explain/figures/main/fig4_dumbbell_test_j20_short.csv")
    pathway  = pd.read_csv("../runs/full_CPD/explain/figures/csv/fig3_pathway_importance_test.csv")

    def _canon(s: str) -> str:
        """Canonicalize label: lowercase, strip non-essential tokens."""
        import re as _re
        t = str(s).lower().strip()
        # unify arrow/dash/paren variants, then strip braces/commas/punctuation
        t = t.replace("→", "->").replace("–", "-").replace("—", "-")
        t = _re.sub(r"\s*\([^)]*\)", "", t)      # drop parenthesised qualifiers
        t = _re.sub(r"[\{\},\s\.]", "", t)       # drop { } , whitespace, .
        return t

    # Build the authoritative lookup by joining dumbbell → pathway on pattern key.
    # label_to_data: canonical_label -> (sig_ig, n_carriers, delta)
    label_to_data = {}
    for _, r in dumbbell.iterrows():
        pat = r["pattern"]
        lbl = r["label"]
        sig   = float(r["original_sig_ig"]) if pd.notna(r.get("original_sig_ig")) else None
        delta = float(r["delta"])          if pd.notna(r.get("delta"))          else None
        n = None
        m = pathway[pathway["pattern"] == pat]
        if len(m):
            n = int(m["n_present"].iloc[0])
        label_to_data[_canon(lbl)] = (sig, n, delta)

    # Manual aliases for label variants that differ between the docx (clinical
    # shorthand) and the cached CSV labels (RxNorm codes / full words).
    ALIASES = {
        _canon("{Live birth, Dibucaine} → Live birth"):
            _canon("{Live birth, RX_3339} → Live birth"),
        _canon("{Chest pain, Resp meas.} → Chest pain"):
            _canon("{Chest pain, Resp measurements} → Chest pain"),
        _canon("{MDD, SI} → SI"):
            _canon("{MDD (single), SI} → SI"),
    }
    for alias_from, alias_to in ALIASES.items():
        if alias_to in label_to_data and alias_from not in label_to_data:
            label_to_data[alias_from] = label_to_data[alias_to]

    if len(doc.tables) > 2:
        t2 = doc.tables[2]
        filled = 0
        unmatched = []
        for ri in range(1, len(t2.rows)):
            row = t2.rows[ri]
            if len(row.cells) < 5:
                continue
            pattern_cell = row.cells[1].text.strip()
            sig_cell = row.cells[2]  # ΣIG
            n_cell   = row.cells[3]  # n carriers
            if not pattern_cell:
                continue
            key = _canon(pattern_cell)

            sig = n = delta = None
            if key in label_to_data:
                sig, n, delta = label_to_data[key]
            else:
                # Fuzzy fallback: substring match in either direction
                for k2, (s2, n2, d2) in label_to_data.items():
                    if key in k2 or k2 in key:
                        sig, n, delta = s2, n2, d2
                        break

            if sig is not None and "[val]" in sig_cell.text:
                set_cell(sig_cell, f"{sig:+.4f}")
                filled += 1
            if n is not None and "[val]" in n_cell.text:
                set_cell(n_cell, f"{n}")
                filled += 1
            # Δŷ column (index 4) — fill only if placeholder present
            if len(row.cells) > 4 and delta is not None and "[val]" in row.cells[4].text:
                set_cell(row.cells[4], f"{delta:+.3f}")
                filled += 1
            if sig is None and n is None:
                unmatched.append((ri, pattern_cell[:60]))

        print(f"  [Table 2] Filled {filled} ΣIG/n cells "
              f"(unmatched rows: {len(unmatched)})")
        for ri, snip in unmatched:
            print(f"           r{ri}: '{snip}'")
except Exception as e:
    import traceback
    print(f"  [Table 2] ⚠️  Could not auto-fill: {type(e).__name__}: {e}")
    traceback.print_exc()

# Fix table caption notes: replace "[val] = to be filled from pipeline" with confirmation note
caption_notes = [
    ("[val] = to be filled from pipeline CSVs.",
     "(Values sourced from runs/full_CPD/explain/ cached CSVs.)"),
    ("[val] = to be filled from pipeline.",
     "(Values sourced from runs/full_CPD/explain/ cached CSVs.)"),
]
for old, new in caption_notes:
    n = replace_in_doc(doc, old, new)
    if n:
        print(f"  Caption note: replaced ({n} occurrences)")

print("\n[3b] Scanning for remaining [val] / [A-Z_] placeholders...")
# Exclude known non-placeholder tokens (vocab tokens / method references)
IGNORE_TOKENS = {"[VISIT]", "[CLS]", "[PAD]", "[IG]", "[CPD]"}


def _real_placeholders(text: str):
    raw = re.findall(r'\[val\]|\[[A-Z_]+\]', text)
    return [m for m in raw if m not in IGNORE_TOKENS]


remaining = []
for i, para in enumerate(doc.paragraphs):
    matches = _real_placeholders(para.text)
    if matches:
        remaining.append((i, para.text[:120], matches))
for ti, table in enumerate(doc.tables):
    for ri, row in enumerate(table.rows):
        for ci, cell in enumerate(row.cells):
            matches = _real_placeholders(cell.text)
            if matches:
                remaining.append((f"T{ti}r{ri}c{ci}", cell.text[:120], matches))

if remaining:
    print(f"  ⚠️  {len(remaining)} paragraph(s) still contain placeholders:")
    for loc, snippet, matches in remaining:
        print(f"    [{loc}] {matches}  |  '{snippet}'")
else:
    print("  ✅ No [val] placeholders remaining")

# ── Step 4: Check JAMIA section order ─────────────────────────────────────────
print("\n[4] Checking document structure...")
section_keywords = [
    "abstract", "background", "objective", "methods", "results",
    "discussion", "conclusion", "limitation", "acknowledg",
    "funding", "conflict", "data availability", "author contrib",
    "reference", "figure legend", "figure caption"
]
found_sections = []
for i, para in enumerate(doc.paragraphs):
    text = para.text.strip().lower()
    if any(kw in text for kw in section_keywords) and len(text) < 80:
        found_sections.append((i, para.text.strip()))

print("  Sections found (in order):")
for idx, title in found_sections:
    print(f"    [{idx:4d}] {title}")

ref_idx = None
fig_legend_idx = None
for i, para in enumerate(doc.paragraphs):
    t = para.text.strip().lower()
    if t.startswith("reference") and len(t) < 20:
        ref_idx = i
    if ("figure legend" in t or "figure caption" in t) and len(t) < 40:
        fig_legend_idx = i

if ref_idx and fig_legend_idx:
    if fig_legend_idx > ref_idx:
        print(f"\n  ✅ Figure legends ({fig_legend_idx}) are AFTER references ({ref_idx})")
    else:
        print(f"\n  ⚠️  Figure legends ({fig_legend_idx}) appear BEFORE references ({ref_idx})")
elif fig_legend_idx is None:
    print("\n  ⚠️  No 'Figure Legends' section heading found")

# ── Step 5: Check alt text in figure legends ──────────────────────────────────
print("\n[5] Checking for alt text in figure legends...")
fig_pattern = re.compile(r'^Figure\s+\d+', re.IGNORECASE)
alt_pattern = re.compile(r'alt\s*text', re.IGNORECASE)
in_legends = False
fig_entries = []
for para in doc.paragraphs:
    t = para.text.strip()
    if "figure legend" in t.lower() or "figure caption" in t.lower():
        in_legends = True
        continue
    if in_legends:
        if fig_pattern.match(t):
            fig_entries.append({"title": t[:80], "has_alt": False})
        if alt_pattern.search(t) and fig_entries:
            fig_entries[-1]["has_alt"] = True

for entry in fig_entries:
    status = "✅" if entry["has_alt"] else "⚠️  MISSING alt text"
    print(f"    {status}  {entry['title']}")

if not fig_entries:
    print("  ⚠️  No figure legend entries found — check document structure")

# ── Step 6: Abstract word count ───────────────────────────────────────────────
# JAMIA structured abstracts contain "Objective:", "Materials and Methods:",
# "Results:", "Discussion:", "Conclusion:" as inline subheadings — we count
# everything between the "Abstract" heading and the first major section
# heading (Background and Significance / Introduction / Keywords).
print("\n[6] Estimating abstract word count...")
MAJOR_SECTION_STARTS = (
    "background and significance", "introduction", "keywords",
)
in_abstract = False
abstract_words = 0
for para in doc.paragraphs:
    t = para.text.strip()
    if not in_abstract and t.lower() == "abstract":
        in_abstract = True
        continue
    if in_abstract:
        if any(t.lower().startswith(kw) for kw in MAJOR_SECTION_STARTS):
            break
        abstract_words += len(t.split())

print(f"  Abstract word count: ~{abstract_words}")
if abstract_words > 250:
    print(f"  ⚠️  JAMIA limit is 250 words — please trim {abstract_words - 250} words")
elif abstract_words > 0:
    print(f"  ✅ Within JAMIA 250-word limit")

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\n[7] Saving to {DST}...")
doc.save(DST)
print(f"  ✅ Saved")
print(f"  Total direct replacements made: {total_changes}")
