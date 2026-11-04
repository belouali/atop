"""clinical_categories, rule-based clinical grouping of readable pattern labels.

Each pattern is assigned to one clinical category by keyword matching on
its readable label (as produced by ``PatternLabelResolver.format_pattern``
with ``short_names=True``). Categories are tested in the order listed in
``CLINICAL_CATEGORY_RULES`` and the first match wins; a label that matches
no category is "Other". Matching is a case-sensitive substring test, so
both capitalised and lowercase forms are listed where labels use either.

The categories are for presentation only. They play no part in pattern
mining, attribution, deduplication or ranking.

Usage:
    from atop.explain.clinical_categories import classify_clinical_category
    classify_clinical_category("{MDD, SI} → SI")   # "Psychiatric"
"""
from __future__ import annotations

OTHER = "Other"

# Order matters: first match wins. The order lets the more clinically
# specific category win when a label could fit several: "MDD + alcohol"
# is Psychiatric, and cancer takes precedence over hepatobiliary disease
# (pancreatic cancer with biliary obstruction is Oncologic).
CLINICAL_CATEGORY_RULES: list[tuple[str, set[str]]] = [
    ("Psychiatric",
     {"MDD", "Severe MDD", "Recurrent severe MDD", "Depression NOS",
      "Mood disorder NOS", "Bipolar", "Manic", "PTSD", "Anxiety",
      "Schizo", "schizo",
      "Psychosis", "Delusional", "SI", "Suicide",
      "Hx self-harm", "self-harm", "Risperidone", "Olanzapine",
      "Fluoxetine", "Mood disorder", "Brief psychotic"}),
    ("Oncologic",
     {"Cancer", "metastases", "Bone met", "Pancreatic", "Liver met",
      "Lung met", "Mal. pleural", "Ascites", "Chemo", "Cancer pain",
      "Cancer anemia",
      "Malignant neoplasm", "malignant neoplasm",
      "neoplasm", "Neoplasm",
      "Antineoplastic", "antineoplastic"}),
    # Bile duct, liver, gallbladder and pancreatic disease without a
    # cancer term (those are Oncologic above).
    ("Hepatobiliary",
     {"bile duct", "Bile duct", "biliary", "Biliary", "ERCP",
      "cholangitis", "Cholangitis", "cholecystitis", "Cholecystitis",
      "Cholestasis", "cholestasis", "cirrhosis", "Cirrhosis",
      "Hepatitis", "hepatitis", "Liver failure", "liver failure",
      "Gallbladder", "gallbladder", "Gallstone", "gallstone",
      "Pancreatitis", "pancreatitis"}),
    ("Obstetric",
     {"Live birth", "C-section", "Dibucaine", "Perineal", "gestation",
      "pregnancy", "Pregnancy", "Oxytocin", "OB lac", "Bupivacaine",
      "Maternal", "maternal",
      "Gestational", "gestational",
      "preterm labor", "Preterm labor",
      "Amniotic", "amniotic",
      "multigravida", "Multigravida"}),
    ("Substance use",
     {"Alcohol intox", "Alcohol abuse", "Cannabis", "Benzodiazepine",
      "Acetamin OD", "SSRI OD", "Analgesic OD", "self-poisoning"}),
    # Tested after Substance use so that a pattern such as
    # {Alcohol abuse; Resp measurements} is not labelled a chest pain
    # workup on the strength of a diagnostic-test token alone.
    ("Chest pain workup",
     {"Chest pain NOS", "ECG", "Chest X-ray", "Resp measurements",
      "Atypical chest pain"}),
]

CATEGORY_ORDER: list[str] = [name for name, _ in CLINICAL_CATEGORY_RULES] + [OTHER]


def classify_clinical_category(label: str) -> str:
    """Return the first category whose keywords occur in ``label``."""
    for name, keywords in CLINICAL_CATEGORY_RULES:
        for kw in keywords:
            if kw in label:
                return name
    return OTHER
