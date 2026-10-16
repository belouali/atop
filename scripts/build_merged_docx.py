"""build_merged_docx, Single Word file combining main paper + figures
+ supplement, mirroring the self-contained PDF.

Reads the existing main DOCX and supplement DOCX (built by
build_revised_main_docx.py and build_revised_supplement_docx.py), then
emits one merged DOCX with this layout:

  1. Title + structured abstract + body (Background → Conclusion +
     References), from main DOCX, with the existing "Figure Legends"
     section replaced by an embedded-image "Figures" section.
  2. Page break.
  3. "Supplementary Material" heading + supplement contents from
     supplement DOCX, with S-figure PNGs embedded inline below their
     captions.

Equations carry trailing tab + (1) / (2) markers from the main DOCX
builder; this script preserves them as-is so Word's ruler tab stops
position the equation number to the right.

Output: manuscript/jamia_submission/AToP_merged.docx
"""
from __future__ import annotations

import os
import re
import sys
from copy import deepcopy
from io import BytesIO

from docx import Document
from docx.shared import Inches, Pt
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.section import WD_SECTION, WD_ORIENT
from docx.oxml.ns import qn
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(HERE, "..", "..", "manuscript", "jamia_submission")
MAIN_DOCX = os.path.join(SUBMISSION_DIR, "AToP_JAMIA_submission.docx")
SUPP_DOCX = os.path.join(SUBMISSION_DIR, "AToP_JAMIA_supplement.docx")
FIG_DIR   = os.path.join(SUBMISSION_DIR, "figures")
SUP_FIG_DIR = os.path.join(FIG_DIR, "supplements")
OUT_PATH  = os.path.join(SUBMISSION_DIR, "AToP_merged.docx")

# Page geometry: portrait Word page minus margins. The default python-docx
# page is 8.5 × 11 inches with 1" margins → ~6.5" usable width.
USABLE_W_INCHES = 6.5
# Landscape pages for wide figures: 0.6" margins on 11 x 8.5 in leave
# 9.8 x 7.3 in; the figure is capped so its caption fits underneath.
WIDE_ASPECT = 1.2
LANDSCAPE_MARGIN_INCHES = 0.6
LANDSCAPE_W_INCHES = 9.4
LANDSCAPE_FIG_H_INCHES = 5.6


def _start_section(doc, landscape: bool):
    """Start a new page in landscape or portrait orientation."""
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    w, h = sec.page_width, sec.page_height
    if landscape:
        if w < h:
            sec.page_width, sec.page_height = h, w
        sec.orientation = WD_ORIENT.LANDSCAPE
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Inches(LANDSCAPE_MARGIN_INCHES))
    else:
        if w > h:
            sec.page_width, sec.page_height = h, w
        sec.orientation = WD_ORIENT.PORTRAIT
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Inches(1))
    return sec

# Main figures: ordered list, each with a path + caption + alt text.
# Pulled from the main builder so the merged DOCX renders the same five
# figures the self-contained PDF embeds.
MAIN_FIGURES = [
    {
        "n": 1, "label": "Figure 1", "file": "Figure1.png",
        "caption": (
            "**Figure 1.** AToP framework overview. Five-step pipeline: "
            "(1) per-token Integrated Gradients on the trained "
            "Transformer; (2) salient-token selection (the tokens "
            "contributing 80% of per-patient attribution mass); "
            "(3) frequent-episode mining from training-cohort salient "
            "subsequences, with cross-visit locks (≥2 visits, ≥2 tokens) "
            "applied at reporting; (4) carrier-conditional token-level "
            "and pattern-level Σ IG aggregation; and (5) Jaccard ≥ 0.2 "
            "deduplication on test-set carrier sets, retaining the "
            "highest-magnitude representative per phenotype cluster."),
        "alt": (
            "Schematic flow diagram with five sequential steps "
            "linking the trained Transformer to per-token attributions, "
            "salient-token selection, frequent-episode mining, "
            "carrier-conditional aggregation, and Jaccard-based phenotype "
            "deduplication."),
    },
    {
        "n": 2, "label": "Figure 2", "file": "Figure2.png",
        "caption": (
            "**Figure 2.** Single-patient AToP explanation. (A) Visit "
            "timeline: each visit shows its top-3 salient tokens by "
            "|IG| color-coded by sign; the index visit (prediction "
            "point for next-30-day readmission) is outlined; lower "
            "axis shows real elapsed days and the observed R30 "
            "outcome. (B) Allocated signed Σ IG decomposition (logit "
            "space; not unique). Each bar = marginal Σ IG over "
            "(position, token) pairs newly claimed by one matched "
            "pattern under a deterministic non-overlapping position-"
            "level allocation; bars are ordered chronologically; "
            "Residual-salient and Non-salient bars close the gap to "
            "total Σ IG. By completeness, total Σ IG ≈ z(x) − "
            "z(x_baseline) on the logit scale; ŷ and ŷ_baseline are "
            "annotated for clinical context but are not additively "
            "related to Σ IG. Label prefixes show visit reach "
            "(V1→V5 = cross-visit; V5: = single-visit)."),
        "alt": (
            "Two-panel figure. Panel A shows a horizontal visit "
            "timeline with one box per admission, each containing the "
            "top salient tokens for that visit with signed IG values, "
            "and a real-days axis below. Panel B shows a horizontal "
            "SHAP-style chevron waterfall of pattern-level Δlogit "
            "contributions ordered chronologically, with a residual "
            "salient bar and a non-salient bar at the bottom."),
    },
    {
        "n": 3, "label": "Figure 3", "file": "Figure3.png",
        "caption": (
            "**Figure 3.** Global token-level attribution on the test "
            "set: population-weighted vs carrier-conditional. (A) "
            "Population mean (all patients; non-carriers scored 0). "
            "(B) Carrier-conditional mean (among carriers). Let "
            "score_j = |IG|(j); the two panels are linked by E[score_j] "
            "= P(present(j)) × E[score_j | present(j)]. Top-30 tokens "
            "by absolute magnitude per panel. Panel B uses the "
            "canonical x-axis (±2.28); Panel A is on its own axis "
            "(±0.04) because population weighting reduces magnitudes "
            "by ~50×."),
        "alt": (
            "Two-panel horizontal bar chart. Left panel: population-"
            "weighted token attribution. Right panel: carrier-"
            "conditional token attribution. Bars are red for risk-"
            "increasing and blue for protective."),
    },
    {
        "n": 4, "label": "Figure 4", "file": "Figure4.png",
        "caption": (
            "**Figure 4.** Global pattern-level attribution on the "
            "test set: all patterns vs cross-visit only. (A) "
            "Carrier-conditional Σ IG, all patterns (single-"
            "token, single-visit co-occurrence, and cross-visit "
            "patterns combined). (B) Carrier-conditional Σ IG, "
            "cross-visit only (≥2 visit blocks, ≥2 tokens; the "
            "canonical AToP output). Bars report attribution mass "
            "concentrated on constituent tokens under the non-"
            "overlapping position-level allocation rule of Figure 2; "
            "they are allocated attribution mass, not identified "
            "pattern-level effects. Top-30 representatives by "
            "|Σ IG|, after Jaccard ≥ 0.2 deduplication. Both panels "
            "share x-axis (±2.28)."),
        "alt": (
            "Two-panel horizontal bar chart of pattern-level signed "
            "Σ IG. Left panel: all patterns. Right panel: cross-"
            "visit patterns only. Bars are red for risk-increasing "
            "and blue for protective."),
    },
    {
        "n": 5, "label": "Figure 5", "file": "Figure5.png",
        "caption": (
            "**Figure 5.** Perturbation-validation 2-panel dumbbell. "
            "Note on scale: the masking effect Δŷ is in predicted-"
            "probability space, while IG is in pre-sigmoid logit "
            "space; the two are not directly additive across this "
            "figure and Figures 2/3/4. (A) Top-30 representatives from "
            "all patterns; (B) Top-30 cross-visit pattern "
            "representatives (≥2 visits, ≥2 tokens). Each row shows "
            "mean Δŷ = ŷ_original − ŷ_masked across the pattern's "
            "test-set carriers; n annotated. Δŷ > 0 (right, red): "
            "masking lowered predicted risk, so the pattern is "
            "risk-supporting; Δŷ < 0 (left, blue): masking raised "
            "predicted risk, so the pattern is protective. Color and "
            "orientation match Figures 3 and 4. Δŷ is per-instance "
            "under the chosen non-overlapping allocation; alternative "
            "allocation rules would distribute Δŷ differently."),
        "alt": (
            "Two-panel dumbbell figure. Each row shows a colored "
            "arrow from zero to the masking effect, red for risk-"
            "supporting patterns (right), blue for protective (left). "
            "Left panel = all patterns. Right panel = cross-visit "
            "patterns."),
    },
]


def _add_runs_with_md_bold(paragraph, text: str):
    """Add runs to a paragraph, parsing **bold** markdown into bold
    runs."""
    parts = text.split("**")
    for i, p in enumerate(parts):
        if not p:
            continue
        run = paragraph.add_run(p)
        if i % 2 == 1:
            run.bold = True


def _embed_image_scaled(doc: Document, path: str, max_w_in: float,
                        max_h_in: float = 8.8):
    """Add a centered picture, scaled to max_w_in inches wide."""
    if not os.path.exists(path):
        return
    # Determine actual aspect ratio so we can leave a comfortable
    # vertical margin for caption + alt text.
    with Image.open(path) as im:
        iw, ih = im.size
    aspect = iw / ih
    # Cap the height so tall figures do not overflow the page.
    width = max_w_in
    height = width / aspect
    if height > max_h_in:
        height = max_h_in
        width = height * aspect

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(path, width=Inches(width))


def _add_figure_caption(doc: Document, caption_md: str, alt_text: str):
    """Add caption (with bold-markdown parsed) + alt text in italic."""
    p = doc.add_paragraph()
    _add_runs_with_md_bold(p, caption_md)
    # Caption styling: 9pt-ish via style if available
    for run in p.runs:
        run.font.size = Pt(9)

    p2 = doc.add_paragraph()
    run = p2.add_run("Alt text: " + alt_text)
    run.italic = True
    run.font.size = Pt(8.5)


def _parse_main_legends(path):
    """Read the main DOCX's 'Figure Legends' section.

    Returns {"Figure N": {"caption": str, "alt": str}} so figure captions have
    exactly one source of truth (the main DOCX) across every deliverable.
    """
    out, in_legends, cur = {}, False, None
    for p in Document(path).paragraphs:
        t = (p.text or "").strip()
        if not t:
            continue
        if t == "Figure Legends":
            in_legends = True
            continue
        if not in_legends:
            continue
        if t in ("References", "Author Contributions", "Funding",
                 "Conflicts of Interest", "Acknowledgments"):
            break
        m = re.match(r"^(Figure\s+\d+)\.\s*(.+)$", t, flags=re.S)
        if m:
            cur = m.group(1)
            out[cur] = {"caption": m.group(2).strip(), "alt": ""}
        elif cur and t.lower().startswith("alt text:"):
            out[cur]["alt"] = t.split(":", 1)[1].strip()
        elif cur:
            out[cur]["caption"] += " " + t
    return out


def _page_break(doc: Document):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def _trim_main_after_references(main_doc: Document):
    """Strip the main DOCX of everything from the 'Figure Legends'
    heading onward, since we'll replace it with an embedded-image
    'Figures' section."""
    body = main_doc.element.body
    # Find the paragraph whose text == "Figure Legends".
    children = list(body.iterchildren())
    cut_index = None
    for i, child in enumerate(children):
        if child.tag != qn("w:p"):
            continue
        # Read the paragraph text by walking its runs
        text = "".join(t.text or "" for t in child.iter(qn("w:t"))).strip()
        if text == "Figure Legends":
            cut_index = i
            break
    if cut_index is not None:
        for child in children[cut_index:]:
            # The body's trailing w:sectPr (page size, margins,
            # orientation) is the LAST child, so a blind delete-to-end
            # removed it. A body with no sectPr is invalid OOXML and Word
            # reports the file as unreadable.
            if child.tag == qn("w:sectPr"):
                continue
            body.remove(child)


def _append_figures_section(doc: Document):
    """Append the 'Figures' section: heading + each figure on its own
    page (page break + image + caption + alt).

    Legend text is read from the main DOCX rather than duplicated here, so
    the merged document always matches the submission document."""
    _page_break(doc)
    # Same look as the manuscript's level-1 headings (Times New Roman,
    # bold, 14 pt, black), not Word's built-in Heading 1 theme style.
    h = doc.add_paragraph()
    _hr = h.add_run("Figures")
    _hr.bold = True
    _hr.font.size = Pt(H1_PT)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Single source of truth: use the legend text from the main DOCX's
    # "Figure Legends" section so the merged file can never drift from the
    # submission DOCX. MAIN_FIGURES captions are a fallback only.
    legends = _parse_main_legends(MAIN_DOCX)
    for f in MAIN_FIGURES:
        fpath = os.path.join(FIG_DIR, f["file"])
        with Image.open(fpath) as im:
            iw, ih = im.size
            dpi = (im.info.get("dpi") or (300, 300))[0] or 300
        # Landscape page only for figures drawn at landscape print size
        # (Figures 4 and 5: ~9 in wide); Figure 1 already fits portrait and
        # Figure 2 is a large raster that is scaled to portrait width.
        wide = (iw / ih > WIDE_ASPECT) and (7.0 <= iw / dpi <= 10.5)
        if wide:
            # Wide figures (Figures 4 and 5) get their own landscape page so
            # they keep print size instead of shrinking to portrait width.
            _start_section(doc, landscape=True)
            _embed_image_scaled(doc, fpath, LANDSCAPE_W_INCHES, max_h_in=LANDSCAPE_FIG_H_INCHES)
        else:
            _page_break(doc)
            _embed_image_scaled(doc, fpath, USABLE_W_INCHES)
        lg = legends.get(f["label"], {})
        cap = lg.get("caption")
        cap = f"**{f['label']}.** {cap}" if cap else f["caption"]
        _add_figure_caption(doc, cap, lg.get("alt") or f["alt"])
        if wide:
            _start_section(doc, landscape=False)


SUPP_SECTIONS = ("Supplementary Methods", "Supplementary Results",
                 "Supplementary Figures", "Supplementary Tables")
H1_PT = 14.0   # _docx_helpers.HEADING1_SIZE


def _is_h1(par) -> bool:
    """True for a real level-1 heading (14 pt bold), false for a body
    paragraph that merely repeats the same words."""
    r = par.runs[0] if par.runs else None
    return bool(r and r.bold and r.font.size and r.font.size.pt == H1_PT)


def _set_caption_size(par, pt):
    """Force every run in a copied caption to `pt`, so supplement captions
    match the main figure legends in the merged document."""
    for r in par.runs:
        r.font.size = Pt(pt)


def _append_supplement(doc: Document):
    """Append the supplement VERBATIM - front page, contents table,
    Supplementary Methods/Results, figures (already embedded in the
    supplement itself) and tables, with its own pagination intact.

    The previous implementation re-walked the supplement and rebuilt it
    piecemeal: it re-embedded figure images that the supplement already
    contains, suppressed what it guessed were duplicate headings, and
    started copying only at "Supplementary Figures". That guessing lost
    every Table S* title, the whole Supplementary Methods and
    Supplementary Results sections, and all page breaks. Copying the
    document as-is removes the guesswork; the only real work is
    re-registering image relationships, since a deepcopy carries rIds
    that mean nothing in the destination package.
    """
    _page_break(doc)
    start_idx = len(doc.paragraphs)

    supp = Document(SUPP_DOCX)
    src_part, dst_part = supp.part, doc.part
    rid_map = {}
    n_img = 0

    # wp:docPr ids must be unique across the WHOLE document. The main
    # paper's figures already occupy 1..N and the supplement's restart at
    # 1, so a straight copy produces duplicate ids and Word refuses to
    # open the file ("unreadable content").
    next_id = 1
    for dp in doc.element.body.findall(".//" + qn("wp:docPr")):
        try:
            next_id = max(next_id, int(dp.get("id")) + 1)
        except (TypeError, ValueError):
            pass

    for child in list(supp.element.body.iterchildren()):
        if child.tag == qn("w:sectPr"):
            continue                      # keep the merged doc's own setup
        new = deepcopy(child)
        for dp in new.findall(".//" + qn("wp:docPr")):
            dp.set("id", str(next_id))
            next_id += 1
        for blip in new.findall(".//" + qn("a:blip")):
            rid = blip.get(qn("r:embed"))
            if not rid:
                continue
            if rid not in rid_map:
                # Re-add the image BYTES rather than relating to the
                # supplement's existing part object. That object's
                # partname is already /word/media/imageN.png, which
                # collides with the main document's own imageN.png: both
                # rIds then resolve to the same file and the supplement's
                # first five figures rendered as copies of Figures 1-5.
                # get_or_add_image_part assigns a fresh partname in THIS
                # package (and dedupes by hash).
                rid_map[rid], _ = dst_part.get_or_add_image(
                    BytesIO(src_part.related_parts[rid].blob))
            blip.set(qn("r:embed"), rid_map[rid])
            n_img += 1
        doc.element.body.append(new)

    print(f"    supplement appended verbatim "
          f"({len(supp.paragraphs)} paragraphs, {len(supp.tables)} tables, "
          f"{n_img} images re-linked)")
    _normalise_supplement_type(doc, start_idx)
    _assert_unique_docpr_ids(doc)
    _ensure_sectpr_last(doc)


def _ensure_sectpr_last(doc) -> None:
    """w:sectPr must be the final child of w:body. Raw body.append()
    calls (used to copy the supplement verbatim) add content after it,
    so move it back to the end before saving."""
    body = doc.element.body
    sect = body.find(qn("w:sectPr"))
    if sect is None:
        raise SystemExit("body has no w:sectPr - Word will reject the file")
    if body[-1] is not sect:
        body.remove(sect)
        body.append(sect)
        print("    moved w:sectPr back to end of body")


def _assert_unique_docpr_ids(doc) -> None:
    """Duplicate wp:docPr ids make Word declare the file unreadable.
    Fail the build rather than ship a document that will not open."""
    from collections import Counter
    ids = [dp.get("id")
           for dp in doc.element.body.findall(".//" + qn("wp:docPr"))]
    dupes = {k: v for k, v in Counter(ids).items() if v > 1}
    if dupes:
        raise SystemExit(f"duplicate wp:docPr ids would corrupt the "
                          f"document: {dupes}")
    print(f"    {len(ids)} drawing ids, all unique")


def _normalise_supplement_type(doc, start_idx: int) -> None:
    """Match the supplement's caption type to the main paper's.

    The supplement sets its figure captions at body size; the main paper
    sets figure legends at 9 pt. Verbatim copying therefore lands two
    caption sizes in one document. Only captions inside the figure/table
    sections are touched - the front-page contents list repeats the same
    "Figure S1." strings as ordinary body text and must stay at body size.
    """
    paras = doc.paragraphs
    body_start = None
    for i in range(start_idx, len(paras)):
        t = paras[i].text.strip()
        if t in ("Supplementary Figures", "Supplementary Tables") \
                and _is_h1(paras[i]):
            body_start = i
            break
    if body_start is None:
        return
    n = 0
    for i in range(body_start, len(paras)):
        par = paras[i]
        t = par.text.strip()
        if not t or not par.runs:
            continue
        r0 = par.runs[0]
        sz = r0.font.size.pt if r0.font.size else None
        # leave section headings (14 pt) and item titles (12 pt) alone
        if sz in (H1_PT, 12.0):
            continue
        # Figure captions and legend/abbreviation blocks take the main
        # paper's 9 pt. Table TITLES do not: main-paper table titles sit
        # at body size, so shrinking "Table S1." to 9 pt would swap one
        # mismatch for another.
        if re.match(r"^Figure\s+S\d", t) or \
                t.startswith(("Abbreviations:", "Note:")):
            for r in par.runs:
                r.font.size = Pt(9)
            n += 1
    print(f"    normalised {n} supplement captions to 9 pt")


def build():
    print(f"[1] Loading main DOCX:  {os.path.basename(MAIN_DOCX)}")
    doc = Document(MAIN_DOCX)
    _trim_main_after_references(doc)

    print(f"[2] Appending Figures section ({len(MAIN_FIGURES)} figures)")
    _append_figures_section(doc)

    print(f"[3] Appending Supplementary Material")
    _append_supplement(doc)

    print(f"[4] Saving merged DOCX: {OUT_PATH}")
    doc.save(OUT_PATH)
    sz = os.path.getsize(OUT_PATH) / 1024
    print(f"    Wrote {OUT_PATH} ({sz:.0f} KB)")


if __name__ == "__main__":
    build()
