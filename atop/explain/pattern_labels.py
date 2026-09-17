"""Pattern + token label resolution for publication figures.

Used by scripts/make_fig*.py and scripts/make_table*.py to render
human-readable token + pattern labels (e.g. ``C:10_I509`` →
``CHF`` or ``Heart failure, unspecified``).

Three sources are merged, in priority order:

1. Hand-curated short labels (``TOKEN_LABELS``), clinical abbreviations
   used in the JAMIA paper figures.
2. ``short_names.csv`` overrides, caller-provided per-token rewrites
   (e.g. for the AMIA / JAMIA submission style guide).
3. ICD title lookup (``icd_titles.json`` from the run dir) +
   word-level abbreviations (``WORD_SHORTENINGS``).

Drug tokens (``D:RX_*``) resolve via ``ndc_to_ingredient.csv`` from the
MIMIC-IV hosp dir; falls back to the bare code if unavailable.
"""
from __future__ import annotations

import csv
import json
import os
import re
from typing import Dict, Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Hand-curated short labels, clinical abbreviations used in the paper
# ──────────────────────────────────────────────────────────────────────
TOKEN_LABELS: Dict[str, str] = {
    # House style for figures (Apr 2026):
    #   * No parentheses inside token labels \u2014 (single), (severe), (mania),
    #     (self-harm), (other) read as if they were second tokens inside a
    #     co-occurrence set. Drop them and reorder words to natural clinical
    #     phrasing instead.
    #   * "Cancer anemia" was an ad-hoc shorthand \u2014 use the clinical name
    #     "Cancer anemia" (ICD-10 D63.0).
    #   * ICD-9 198.5 ("Secondary malignant neoplasm of bone and bone marrow")
    #     is what oncologists call "Bone metastases" \u2014 use that.
    #   * NOS suffix on every label whose underlying ICD title is the
    #     "unspecified" / "unsp." catch-all variant (e.g. F29 "Unspecified
    #     psychosis", F39 "Unspecified mood disorder"). Standard medical
    #     shorthand for "not otherwise specified".
    "C:10_R079": "Chest pain NOS", "C:10_R45851": "SI", "C:10_F329": "MDD single ep. NOS",
    "C:10_F332": "Recurrent severe MDD", "C:10_F322": "Severe MDD",
    "C:10_F29": "Psychosis NOS", "C:10_F39": "Mood disorder NOS",
    "C:10_F319": "Bipolar disorder NOS",
    "C:10_F3110": "Bipolar manic", "C:10_F4310": "PTSD NOS", "C:10_N179": "AKI NOS",
    "C:10_D630": "Cancer anemia", "C:10_G893": "Cancer pain", "C:10_I10": "Hypertension",
    "C:10_K219": "GERD", "C:10_Z370": "Live birth", "C:10_R55": "Syncope",
    "C:10_R188": "Ascites", "C:10_O001": "Ectopic pregnancy",
    "C:10_J0390": "Tonsillitis NOS", "C:10_J36": "Peritonsillar abscess",
    "C:10_F10129": "Alcohol intox NOS", "C:10_F1010": "Alcohol abuse uncomp.",
    "C:10_T43222A": "SSRI OD self-harm", "C:10_T391X2A": "Acetamin OD self-harm",
    "C:10_T781XXA": "Food reaction", "C:9_9654": "Analgesic OD",
    "C:9_30981": "PTSD", "C:9_78659": "Atypical chest pain",
    "C:9_1985": "Bone metastases",
    # ICD-9/-10 long-name shortenings encountered in v2 panels (Apr 2026):
    "C:9_2724":     "Hyperlipidemia NOS",      # "Other and unspecified hyperlipidemia"
    "C:9_64233":    "Gestational hypertension", # "Transient hypertension of pregnancy, antepartum..."
    "C:9_64403":    "Threatened preterm labor", # "Threatened premature labor, antepartum..."
    "C:9_64893":    "Maternal comorbidity",     # "Other current conditions classifiable elsewhere of mother, antepartum..."
    "C:10_S72011A": "R hip fracture",           # "Unspecified intracapsular fracture of right femur, initial encounter for closed fracture"
    "C:10_S81012A": "L knee laceration",        # "Laceration without foreign body, left knee, initial encounter"
    "C:10_K8064":   "Gallstone cholecystitis",  # "Calculus of gallbladder and bile duct with chronic cholecystitis without obstruction"
    "C:10_C250":    "Pancreatic head cancer",   # "Malignant neoplasm of head of pancreas"
    "C:9_1570":     "Pancreatic head cancer",   # ICD-9 equivalent
    "P:9_5491": "Abd. drainage", "P:9_8938": "Resp measurements",
    "P:9_9925": "Chemo infusion", "P:10_10D00Z1": "C-section",
    "P:10_10E0XZZ": "Resp monitoring",
    # Long ICD-10-PCS / ICD-9 long-name shortenings encountered in the
    # full-pool (Panel C / Table S2) co-occurrence patterns. Without these
    # the resolver's 45-char fallback truncation kicks in mid-word ("...").
    "P:10_4A02X4Z": "ECG",                     # "Measurement of Cardiac Electrical Activity, External, Sampling and Pressure"
    "P:10_BW03ZZZ": "Chest X-ray",             # "Plain Radiography of Chest"
    "P:9_7569":     "OB lac. repair",          # "Repair of other current obstetric laceration"
    "C:9_E9500":    "Analgesic self-poisoning",  # "Suicide and self-inflicted poisoning by analgesics, antipyretics, and antirheumatics"
    "C:10_J910":    "Malignant pleural effusion",
    "C:10_Z781":    "Physical restraint",      # "Physical restraint status"
    "C:10_F259":    "Schizoaffective NOS",     # "Schizoaffective disorder, unspecified"
    "D:RX_2418": "Vitamin D",
    "D:RX_1657128": "Influenza A Ag", "C:10_E119": "T2DM", "C:10_E785": "Hyperlipidemia NOS",
    "C:10_I4891": "AFib NOS", "C:10_I509": "CHF NOS", "C:10_J449": "COPD NOS",
    "C:10_N390": "UTI", "C:10_G4733": "Sleep apnea", "C:10_E039": "Hypothyroidism NOS",
    "C:10_N400": "BPH", "C:10_F17210": "Nicotine dep.",
    "C:10_Z87891": "Hx nicotine dep.", "C:10_Z915": "Hx self-harm",
    "C:10_O701": "Perineal laceration",
    "C:10_E871": "Hyponatremia",   # ICD-10 "Hypo-osmolality and hyponatremia"

    # ── Apr 27 2026 batch: long codes surfaced by Fig S11 (cohort SHAP) +
    # remaining long IG-Panel-B codes. Added so no rendered label reaches
    # the resolver's 45-char fallback truncation. Groupings:
    # Procedures (ICD-9-PCS / ICD-10-PCS, often >40 chars natively)
    "P:9_5187":     "ERCP biliary stent",               # Endoscopic insertion of stent (tube) into bile duct
    "P:9_5114":     "Endoscopic biliary biopsy",        # Other closed [endoscopic] biopsy of biliary tract
    "P:9_5011":     "Liver biopsy",                     # Closed (percutaneous) [needle] biopsy of liver
    "P:9_527":      "Pancreaticoduodenectomy",          # Radical pancreaticoduodenectomy (Whipple)
    "P:10_0F9930Z": "Percutaneous biliary drainage",    # Drainage of Common Bile Duct with Drainage Device, Percutaneous
    "P:10_3E04305": "IV antineoplastic",                # Introduction of Other Antineoplastic into Central Vein
    "P:10_3E0G36Z": "Enteral nutrition",                # Introduction of Nutritional Substance into Upper GI, Percutaneous
    "P:10_0W9G3ZX": "Paracentesis",                     # Drainage of Peritoneal Cavity, Percutaneous (diagnostic)
    "P:10_10907ZC": "Amniotomy",                        # Drainage of Amniotic Fluid, Therapeutic from Products of Conception
    "P:9_9925":     "CVC placement",                    # Other infusion through central venous catheter
    "P:9_3897":     "Central venous catheter",          # Venous catheterization, NEC
    "P:10_8E0W4CZ": "Robotic abd. surgery",             # Robotic Assisted Procedure of Trunk Region
    "P:10_0DTJ4ZZ": "Lap. appendectomy",                # Resection of Appendix, Percutaneous Endoscopic
    "P:10_3E033PZ": "IV antiplatelet",                  # Introduction of Platelet Inhibitor into Peripheral Vein
    "P:10_10D17Z9": "Manual placenta removal",          # Manual Extraction of Products of Conception
    "P:10_0YU50JZ": "R inguinal mesh repair",           # Supplement Right Inguinal Region with Synthetic
    "P:10_0W9J30Z": "Pelvic drainage",                  # Drainage of Pelvic Cavity
    "P:10_0DV64CZ": "Gastric banding",                  # Restriction of Stomach with Extraluminal Device
    "P:10_0D9W30Z": "Peritoneal drainage",              # Drainage of Peritoneum
    "P:10_00UT0JZ": "Dural mesh repair",                # Supplement Spinal Meninges with Synthetic Substitute
    "P:10_0HQ9XZZ": "Perineal skin repair",             # Repair Perineum Skin, External Approach
    "P:10_0FT40ZZ": "Open cholecystectomy",             # Resection of Gallbladder, Open Approach
    "P:10_5A1221Z": "Continuous CO monitoring",         # Performance of Cardiac Output, Continuous

    # ICD-9 diagnoses (mostly fracture/injury codes left by IG Panel B)
    "C:9_81354": "Open radius/ulna fracture",            # Open fx, lower end of radius with ulna
    "C:9_6824":  "Hand cellulitis/abscess",              # Cellulitis and abscess of hand, except fingers/thumb
    "C:9_64891": "Maternal complications NOS",           # Other current conditions classifiable elsewhere of mother
    "C:9_5409":  "Acute appendicitis",                   # Acute appendicitis without mention of peritonitis
    "C:9_99812": "Procedural hematoma",                  # Hematoma complicating a procedure
    "C:9_8028":  "Closed facial fracture",               # Closed fracture of other facial bones
    "C:9_9694":  "Benzodiazepine poisoning",             # Poisoning by benzodiazepine-based tranquilizers

    # ICD-10 diagnoses by body system
    #, Trauma / orthopedics
    "C:10_V4959XA": "MVA passenger injury",              # Passenger injured in collision with other motor vehicles
    "C:10_S43006A": "Shoulder dislocation NOS",          # Unspecified dislocation of shoulder joint
    "C:10_S32009A": "Lumbar vertebra fracture NOS",      # Fracture of unspecified lumbar vertebra
    "C:10_M810":    "Osteoporosis",                      # Age-related osteoporosis without current pathological fracture
    #, Obstetrics
    "C:10_O700":    "1st-degree perineal lac.",          # First degree perineal laceration during delivery
    "C:10_O364XX0": "Intrauterine fetal death",          # Maternal care for intrauterine death
    "C:10_O0281":   "Abnormal early-pregnancy hCG",      # Inappropriate change in quantitative hCG
    "C:10_Z3A32":   "32 weeks gestation",                # 32 weeks gestation of pregnancy
    "C:10_Z3A38":   "38 weeks gestation",                # 38 weeks gestation of pregnancy
    #, GI / hepatobiliary
    "C:10_K5190":   "Ulcerative colitis NOS",            # Ulcerative colitis, unspecified, without complications
    "C:10_K4031":   "Obstructed recurrent inguinal hernia",  # Unilateral inguinal hernia, with obstruction, recurrent
    "C:10_K668":    "Peritoneal disorder NOS",           # Other specified disorders of peritoneum
    "C:10_K861":    "Chronic pancreatitis",              # Other chronic pancreatitis
    #, Pulmonary / cardiac
    "C:10_J918":    "Pleural effusion NOS",              # Pleural effusion in other conditions classified elsewhere
    "C:10_J9601":   "Acute hypoxic resp. failure",       # Acute respiratory failure with hypoxia
    "C:10_J0430":   "Supraglottitis NOS",                # Supraglottitis, unspecified, without obstruction
    "C:10_I824Z1":  "DVT, R distal lower ext.",          # Acute DVT of unspecified deep veins, R distal LE
    "C:10_I82412":  "L femoral DVT",                     # Acute embolism and thrombosis of left femoral vein
    "C:10_I5023":   "Acute on chronic systolic CHF",     # Acute on chronic systolic CHF
    "C:10_I5020":   "Systolic CHF NOS",                  # Unspecified systolic CHF
    "C:10_I2699":   "Pulmonary embolism",                # Other pulmonary embolism without acute cor pulmonale
    "C:10_I2692":   "Saddle PE",                         # Saddle embolus of pulmonary artery
    "C:10_I671":    "Unruptured cerebral aneurysm",      # Cerebral aneurysm, nonruptured
    #, Neuro / psych
    "C:10_G40301":  "Generalized epilepsy w/ status",    # Generalized idiopathic epilepsy w/ status epilepticus
    "C:10_F3010":   "Manic episode NOS",                 # Manic episode without psychotic symptoms, unspecified
    "C:10_F339":    "Recurrent MDD NOS",                 # Major depressive disorder, recurrent, unspecified
    "C:10_F312":    "Bipolar manic severe",              # Bipolar disorder, current episode manic severe
    "C:10_H81399":  "Peripheral vertigo NOS",            # Other peripheral vertigo, unspecified ear
    "C:9_E9503":    "Self-poisoning (tranquilizers)",    # Suicide and self-inflicted poisoning by tranquilizers
    "C:10_Z9119":   "Care noncompliance",                # Patient's noncompliance with other medical
    "C:10_Z21":     "Asymptomatic HIV",                  # Asymptomatic human immunodeficiency virus infection
    "C:10_Z818":    "Hx mental illness (family)",        # Family history of other mental and behavioral
    "C:10_O4202":   "Term PROM",                         # Full-term premature rupture of membranes, onset of labor
    "C:9_66331":    "Cord entanglement",                 # Other and unspecified cord entanglement w/ obstructed labor
    "C:9_65961":    "Elderly multigravida",              # Elderly multigravida, delivered with/without antepartum condition
    #, Oncology (secondary mets)
    "C:10_C787":    "Liver metastases",                  # Secondary malignant neoplasm of liver
    "C:10_C7802":   "L lung metastases",                 # Secondary malignant neoplasm of left lung
    "C:10_C772":    "Intra-abdominal LN metastases",     # Secondary malignant neoplasm of intra-abdominal lymph nodes
    "C:10_C7800":   "Lung metastases",                   # Secondary malignant neoplasm of unspecified lung
    "C:10_C771":    "Mediastinal LN metastases",         # Secondary malignant neoplasm of intrathoracic LN
    "C:10_C9000":   "Multiple myeloma",                   # Multiple myeloma not having achieved remission
    "C:10_Z5111":   "Chemo encounter",                   # Encounter for antineoplastic chemotherapy
    "C:10_T451X5A": "Adverse effect of chemo",           # Adverse effect of antineoplastic and immunosuppressive drugs
    "C:9_E9331":    "Chemo adverse effect",              # Antineoplastic and immunosuppressive drugs causing adverse effects
    "C:10_Z800":    "Hx GI cancer (family)",             # Family history of malignant neoplasm of digestive organs
    "D:RX_1731071": "Fosaprepitant",                     # already short — antiemetic for chemo
    #, Heme / renal / skin
    "C:10_D62":     "Posthemorrhagic anemia",            # Acute posthemorrhagic anemia
    "C:10_D684":    "Acquired coagulopathy",             # Acquired coagulation factor deficiency
    "C:10_L03113":  "R upper limb cellulitis",           # Cellulitis of right upper limb
    "C:10_N183":    "CKD stage 3",                       # Chronic kidney disease, stage 3
    #, History / status codes
    "C:10_Z7901":   "Chronic anticoagulation",           # Long term use of anticoagulants
    "C:10_Z87440":  "Hx UTI",                            # Personal history of urinary (tract) infections
}

# Word-level abbreviations applied AFTER lookup, before truncation.
# House style: "unspecified" / "unsp." in the underlying ICD title becomes
# the standard medical " NOS" suffix (not otherwise specified). Leading
# "Unspecified " is dropped and re-attached as " NOS" later via _abbreviate.
WORD_SHORTENINGS = [
    (", unspecified", " NOS"), (", unsp.", " NOS"),
    ("not elsewhere classified", "NEC"), ("not otherwise specified", "NOS"),
    ("intentional self-harm", "self-harm"), ("initial encounter", ""),
    # Preserve the "uncomplicated" qualifier as ``uncomp.`` \u2014 it's
    # clinically meaningful (distinguishes the chronic/baseline form from
    # acute complications like with-intoxication or with-withdrawal).
    (", uncomplicated", " uncomp."), ("uncomplicated", "uncomp."),
    ("Poisoning by ", ""), ("Personal history of", "Hx"),
    ("complicating pregnancy, childbirth, or the puerperium", "in pregnancy"),
    (", delivered, with or without mention of antepartum condition", ""),
    (", unspecified as to episode of care or not applicable", ""),
]


class PatternLabelResolver:
    """Stateful token + pattern label formatter.

    Build once per run, then call ``format_token`` / ``format_pattern``.
    """

    def __init__(
        self,
        run_dir: str,
        short_names_csv: Optional[str] = None,
        drug_mapping_csv: Optional[str] = None,
        verbose: bool = True,
    ):
        self._short_names: Dict[str, str] = {}
        self._drug_names: Dict[str, str] = {}
        self._icd_titles: Dict[str, str] = {}

        self._load_short_names(short_names_csv, verbose=verbose)
        self._load_drug_names(run_dir, drug_mapping_csv, verbose=verbose)
        self._load_icd_titles(run_dir, verbose=verbose)

    # ── Loaders ───────────────────────────────────────────────────────
    def _load_short_names(self, csv_path: Optional[str], verbose: bool) -> None:
        if csv_path and os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        self._short_names[row[0].strip()] = row[1].strip()
        if verbose:
            print(f"  [labels] {len(self._short_names)} short-name overrides")

    def _load_drug_names(self, run_dir: str, explicit: Optional[str], verbose: bool) -> None:
        candidates = [explicit] if explicit else []
        # Try config.json's drug_mapping path first
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                cfg_path = cfg.get("drug_mapping", "")
                if cfg_path:
                    candidates.append(cfg_path)
            except Exception:
                pass
        # Fallbacks
        candidates += [
            os.path.join(run_dir, "ndc_to_ingredient.csv"),
            os.path.join(run_dir, "..", "..", "data", "mimiciv", "3.1", "hosp", "ndc_to_ingredient.csv"),
            os.path.join(os.path.dirname(run_dir), "data", "mimiciv", "3.1", "hosp", "ndc_to_ingredient.csv"),
            "../data/mimiciv/3.1/hosp/ndc_to_ingredient.csv",
        ]
        resolved = next((p for p in candidates if p and os.path.exists(p)), None)
        if resolved:
            df = pd.read_csv(resolved)
            if "ingredient_rxcui" in df.columns and "ingredient_name" in df.columns:
                for rxcui, name in (
                    df.dropna(subset=["ingredient_rxcui", "ingredient_name"])
                      .groupby("ingredient_rxcui")["ingredient_name"].first().items()
                ):
                    self._drug_names[f"D:RX_{int(rxcui)}"] = str(name).strip().title()
            if verbose:
                print(f"  [labels] {len(self._drug_names)} drug names from {resolved}")
        elif verbose:
            print("  [labels] no ndc_to_ingredient.csv found")

    def _load_icd_titles(self, run_dir: str, verbose: bool) -> None:
        # Search v2 layout first (0_inputs/), then legacy locations.
        candidates = [
            os.path.join(run_dir, "0_inputs", "icd_titles.json"),
            os.path.join(run_dir, "icd_titles.json"),
            os.path.join(run_dir, "explain", "icd_titles.json"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                with open(cand) as f:
                    self._icd_titles = json.load(f)
                if verbose:
                    print(f"  [labels] {len(self._icd_titles)} ICD titles from {cand}")
                return
        if verbose:
            print("  [labels] no icd_titles.json found")

    # ── Token resolution ──────────────────────────────────────────────
    def _resolve_token(self, tok: str) -> str:
        """Resolve a token like ``C:10_I509`` to its raw title (e.g. ``Heart failure, unspecified``)."""
        tok = tok.strip()
        m = re.match(r"^([CPD]):((?:9|10)_)?(.+)$", tok)
        if not m:
            return tok
        prefix, ver_part, code = m.group(1), m.group(2) or "", m.group(3)
        version = ver_part.rstrip("_") if ver_part else ""
        if prefix == "D":
            for key in (f"D:{code}", f"D:RX_{code[3:]}" if code.startswith("RX_") else ""):
                if key and key in self._drug_names:
                    return self._drug_names[key]
            return code
        if version:
            full_key = f"{prefix}:{version}_{code}"
            if full_key in self._icd_titles:
                return self._icd_titles[full_key]
        for key in (f"{prefix}:{code}", code):
            if key in self._icd_titles:
                return self._icd_titles[key]
        return f"{prefix}:{code}"

    def _abbreviate(self, text: str) -> str:
        """Apply WORD_SHORTENINGS + truncation to a raw title."""
        if not text:
            return text
        if text in self._short_names:
            return self._short_names[text]
        # Leading "Unspecified <thing>" \u2192 "<thing> NOS" so the standard
        # medical "not otherwise specified" suffix lands at the end of the
        # label rather than as a confusing prefix.
        if text.lower().startswith("unspecified "):
            text = text[len("unspecified "):].rstrip() + " NOS"
        for long, short in WORD_SHORTENINGS:
            if long in text:
                text = text.replace(long, short)
        # Collapse double spaces / trailing commas left by the shortening.
        text = re.sub(r"\s+", " ", text).strip().rstrip(",").strip()
        # Avoid double-NOS if a title was both ", unspecified" and started
        # with "Unspecified " (rare but happens with reflexive titles).
        text = re.sub(r"\bNOS\s+NOS\b", "NOS", text)
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        if len(text) > 45:
            text = text[:42] + "..."
        return text

    # ── Public API ────────────────────────────────────────────────────
    def format_token(self, tok: str) -> str:
        """Render a single token in the paper's short style.

        Order: TOKEN_LABELS → short_names CSV override → ICD/drug
        title (with WORD_SHORTENINGS + truncation).
        """
        raw = tok.strip()
        if raw in TOKEN_LABELS:
            return TOKEN_LABELS[raw]
        if raw in self._short_names:
            return self._short_names[raw]
        return self._abbreviate(self._resolve_token(raw))

    def format_pattern(self, raw_pattern: str, short_names: bool = True) -> str:
        """Render a pattern string like ``{C:D630, C:R188} -> C:N179`` to readable form.

        When ``short_names=False``, returns the raw pattern unchanged (useful
        for CSV exports that need machine-parseable patterns).
        """
        if not short_names:
            return raw_pattern
        steps = raw_pattern.split(" -> ")
        formatted = []
        for step in steps:
            step = step.strip()
            if step.startswith("{") and step.endswith("}"):
                # Raw patterns delimit set members with ", " (token codes
                # contain no commas, so this is unambiguous on input).
                tokens = [t.strip() for t in step[1:-1].split(",")]
                # On OUTPUT we join with "; " so set membership stays
                # unambiguous when readable labels themselves contain commas
                # (e.g. "Chest pain, atypical").
                formatted.append("{" + "; ".join(self.format_token(t) for t in tokens) + "}")
            else:
                formatted.append(self.format_token(step))
        return " \u2192 ".join(formatted)
