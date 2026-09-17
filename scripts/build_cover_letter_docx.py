"""build_cover_letter_docx, the JAMIA submission cover letter.

JAMIA asks the cover letter to flag "special considerations", explicitly
including related papers by the same authors already published or under
consideration, and any previous reviews. The AMIA 2026 Symposium abstract on
this work is disclosed here for that reason.

The corresponding author's telephone lives here rather than on the title
page: JAMIA's title-page checklist lists it, but published articles print
only name, address and email, so it is an editorial-office detail.

Usage:
    python scripts/build_cover_letter_docx.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _docx_helpers import new_document, para, title  # noqa: E402

OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "manuscript", "jamia_submission", "AToP_JAMIA_cover_letter.docx")

PAPER_TITLE = ("AToP: An Explainability Framework for Transformer-Based EHR "
               "Predictions Using Attribution-Guided Temporal Pattern Mining")


def build() -> None:
    doc = new_document()

    para(doc, "[AUTHOR: date]")
    para(doc, "")
    para(doc, "The Editors")
    para(doc, "Journal of the American Medical Informatics Association")
    para(doc, "")
    para(doc, f"**Re: Submission of “{PAPER_TITLE}” "
              f"(Research and Applications)**")
    para(doc, "")
    para(doc, "Dear Editors,")
    para(doc, "")

    para(doc, "We are pleased to submit the above manuscript for "
              "consideration as a Research and Applications article. It "
              "presents AToP (Attribution to Temporal Patterns), a post-hoc "
              "explainability framework that converts per-token attributions "
              "from a Transformer trained on longitudinal EHR data into "
              "structured temporal-pattern explanations, demonstrated on "
              "unplanned 30-day readmission in MIMIC-IV v3.1.")

    para(doc, "The contribution is not a new attribution or pattern-mining "
              "algorithm but their composition, and specifically a "
              "refinement of the aggregation convention the existing methods "
              "share. Standard global importance averages attribution across "
              "all patients, which dilutes rare features with large "
              "per-patient impact and cannot represent multi-token "
              "phenotypes at all. AToP instead reports carrier-conditional "
              "attribution and extends the unit of explanation from isolated "
              "tokens to data-discovered multi-visit patterns. The two "
              "conventions produce substantially different pictures of the "
              "same model: the carrier-conditional and population-weighted "
              "token rankings share only 7 of 30 features, and the "
              "all-pattern and cross-visit pattern rankings overlap in 1 of "
              "50 representatives.")

    para(doc, "We have tried to be explicit about what the framework does "
              "and does not establish. Pattern-level values are allocations "
              "under a stated rule rather than identified effects; an "
              "ERASER-style progressive-masking test shows the ranking is "
              "faithful to the model (mean |Δŷ| 1.9× that of "
              "matched random tokens, p < 0.001); reversed-order and "
              "block-shuffle controls support the temporal reading; and a "
              "KernelSHAP replication shows the token-level findings reflect "
              "the aggregation convention rather than the attribution "
              "method, while pattern-level rankings remain "
              "method-dependent. Clinical plausibility review was internal "
              "and non-blinded, and we make no claim of clinical utility.")

    para(doc, "**Special considerations.** An abstract describing this work "
              "was submitted to the AMIA 2026 Annual Symposium "
              "[AUTHOR: confirm status — submitted / accepted / "
              "presented]. The present manuscript is a substantially "
              "expanded treatment: it adds the carrier-conditional "
              "estimand analysis, the pattern-level aggregation and "
              "deduplication, the faithfulness and temporal-order "
              "validation, the KernelSHAP cross-method replication, and the "
              "visit-matched outcome analysis, none of which appear in the "
              "abstract. There are no previous reviews of this manuscript "
              "to report.")

    para(doc, "The manuscript is original, has not been published "
              "previously, and is not under consideration elsewhere. Both "
              "authors have approved the submission and agree to be "
              "accountable for the work. We declare no competing interests. "
              "The framework code is publicly available at "
              "https://github.com/belouali/atop, and all analytic outputs "
              "reported in the manuscript can be regenerated from MIMIC-IV "
              "v3.1 using those scripts.")

    para(doc, "Thank you for considering our submission. We look forward to "
              "the reviewers' comments.")
    para(doc, "")
    para(doc, "Sincerely,")
    para(doc, "")
    para(doc, "Anas Belouali, PhD")
    para(doc, "Biomedical Informatics and Data Science")
    para(doc, "Johns Hopkins University School of Medicine")
    para(doc, "2024 East Monument Street, Suite 1-200, Baltimore, MD 21205, "
              "USA")
    para(doc, "Email: anas.belouali@georgetown.edu")
    para(doc, "Telephone: [AUTHOR: phone]")
    para(doc, "ORCID: 0000-0002-2780-2500")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    doc.save(OUT_PATH)
    print(f"✓ Wrote {OUT_PATH}")
    print(f"  Paragraphs: {len(doc.paragraphs)}")


if __name__ == "__main__":
    build()
