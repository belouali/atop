"""build_self_contained_paper_pdf, One-PDF deliverable for AToP.

Reads the existing main DOCX and supplement DOCX, restructures the section
layout for a "paper-natural" reading (single Introduction instead of
Background and Significance + Objective; numbered sections), and emits a
single self-contained PDF with the layout style of a Google research
paper:

  • Title (large bold), authors (small with affiliation superscripts)
  • Abstract block (no "Objective:/Methods:/..." sub-labels)
  • Numbered sections (1. Introduction, 2. Methods, 3. Results,
    4. Discussion, 5. Conclusion, References)
  • Body in serif (Times-Roman), 10pt, 1.18 line-height, justified
  • Run-in subsection headings (bold + period)
  • Page header (italic running title) + page number bottom-right
  • Figures section after References, each figure on its own area with
    "Figure N |" caption directly underneath
  • Supplementary Material section after Figures, with all S-figures and
    S-tables embedded inline

Output: manuscript/jamia_submission/AToP_self_contained.pdf

Usage:
    python scripts/build_self_contained_paper_pdf.py
"""
from __future__ import annotations

import os
import sys
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from docx import Document
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch

LETTER_LS = landscape(letter)  # (792, 612), landscape orientation
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Image, Table, TableStyle, PageBreak, KeepTogether, NextPageTemplate,
    CondPageBreak,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


_RLParagraph = Paragraph


def Paragraph(text, *args, **kwargs):  # noqa: N802 (drop-in for reportlab)
    return _RLParagraph(_fix_glyphs(text), *args, **kwargs)

# ── Register TTF fonts with full Unicode coverage ─────────────────────
# The default Times-Roman PostScript Type 1 font in reportlab covers
# Latin-1 only, Greek (Σ, σ), modifier letters (ŷ), arrows (→) all
# render as ■. Times New Roman TTF on macOS has full coverage.
def _register_ttfs():
    candidates = {
        "TNR":      "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "TNR-Bold": "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "TNR-Italic": "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
        "TNR-BoldItalic": "/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf",
    }
    registered = True
    for name, path in candidates.items():
        if not os.path.exists(path):
            registered = False
            break
        pdfmetrics.registerFont(TTFont(name, path))
    if registered:
        # Tell reportlab how to substitute bold/italic variants
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily("TNR", normal="TNR", bold="TNR-Bold",
                            italic="TNR-Italic", boldItalic="TNR-BoldItalic")
        return "TNR"
    # Fallback: use built-in Times-Roman (may render boxes for some glyphs)
    return None


SERIF_FAMILY = _register_ttfs()


# ── Glyph coverage ─────────────────────────────────────────────────────
# Times New Roman lacks most Unicode subscript letters (e.g. U+2C7C "ⱼ",
# "ₘ", "₀") and some math symbols ("∈", "∝"), which reportlab renders as
# empty boxes. Every Paragraph passes through _fix_glyphs: Unicode
# sub/superscript letters become <sub>/<super> markup with ordinary
# glyphs, and any other character missing from the serif font is drawn
# from Apple Symbols when that font has it.
_SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₕₖₗₘₙₚₛₜᵢⱼᵣᵤᵥ", "0123456789aeoxhklmnpstijruv"))
_SUP = dict(zip("ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ", "abcdefghijklmnoprstuvwxyz"))
_FALLBACK_FONT = None
if SERIF_FAMILY and os.path.exists("/System/Library/Fonts/Apple Symbols.ttf"):
    pdfmetrics.registerFont(TTFont("AppleSymbols",
                                   "/System/Library/Fonts/Apple Symbols.ttf"))
    _FALLBACK_FONT = "AppleSymbols"


def _fix_glyphs(xml: str) -> str:
    if not xml or not SERIF_FAMILY:
        return xml
    serif = pdfmetrics.getFont("TNR").face.charToGlyph
    fallback = (pdfmetrics.getFont(_FALLBACK_FONT).face.charToGlyph
                if _FALLBACK_FONT else {})
    out, mode, buf = [], None, []

    def flush():
        if buf:
            tag = {"sub": "sub", "sup": "super"}.get(mode)
            body = "".join(buf)
            out.append(f"<{tag}>{body}</{tag}>" if tag else
                       f'<font name="{_FALLBACK_FONT}">{body}</font>')
            buf.clear()

    for ch in xml:
        if ch in _SUB:
            kind, rep = "sub", _SUB[ch]
        elif ch in _SUP:
            kind, rep = "sup", _SUP[ch]
        elif ord(ch) > 127 and ord(ch) not in serif and ord(ch) in fallback:
            kind, rep = "fb", ch
        else:
            kind, rep = None, ch
        if kind != mode:
            flush()
            mode = kind
        (buf if kind else out).append(rep)
    flush()
    return "".join(out)


def _font(name: str, *, bold: bool = False, italic: bool = False) -> str:
    """Resolve a font name. Uses TTF Times New Roman if available, else
    falls back to the bundled Type 1 Times-Roman."""
    if SERIF_FAMILY:
        if bold and italic:
            return "TNR-BoldItalic"
        if bold:
            return "TNR-Bold"
        if italic:
            return "TNR-Italic"
        return "TNR"
    # Fallback (PostScript Type 1, limited unicode)
    if bold and italic:
        return "Times-BoldItalic"
    if bold:
        return "Times-Bold"
    if italic:
        return "Times-Italic"
    return "Times-Roman"


# ── Paths ─────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_DOCX = os.path.join(HERE, "..", "..", "manuscript", "jamia_submission",
                          "AToP_JAMIA_submission.docx")
SUPP_DOCX = os.path.join(HERE, "..", "..", "manuscript", "jamia_submission",
                          "AToP_JAMIA_supplement.docx")
FIG_DIR   = os.path.join(HERE, "..", "..", "manuscript", "jamia_submission",
                          "figures")
SUP_FIG_DIR = os.path.join(FIG_DIR, "supplements")
OUT_PDF   = os.path.join(HERE, "..", "..", "manuscript", "jamia_submission",
                          "AToP_self_contained.pdf")

RUNNING_TITLE = ("AToP: An Explainability Framework for Transformer-Based "
                  "EHR Predictions")


# ── Styles ────────────────────────────────────────────────────────────
def build_styles() -> Dict[str, ParagraphStyle]:
    """Google-paper-like style set, single column, Times-Roman-TTF 10pt."""
    base = getSampleStyleSheet()["Normal"]
    s: Dict[str, ParagraphStyle] = {}
    fn_normal = _font("normal")
    fn_bold   = _font("bold", bold=True)
    fn_italic = _font("italic", italic=True)

    s["title"] = ParagraphStyle(
        "title", parent=base,
        fontName=fn_bold, fontSize=20, leading=24,
        alignment=TA_LEFT, spaceAfter=10, spaceBefore=0,
        textColor=colors.black,
        allowWidows=0, allowOrphans=0,
    )
    s["author"] = ParagraphStyle(
        "author", parent=base,
        fontName=fn_normal, fontSize=10.5, leading=14,
        alignment=TA_LEFT, spaceAfter=2,
        allowWidows=0, allowOrphans=0,
    )
    s["affiliation"] = ParagraphStyle(
        "affiliation", parent=base,
        fontName=fn_italic, fontSize=9, leading=12,
        alignment=TA_LEFT, spaceAfter=6,
        textColor=colors.HexColor("#444444"),
        allowWidows=0, allowOrphans=0,
    )
    # Abstract uses normal-weight body font; the JAMIA structured
    # sub-labels ("Objective:", "Materials and Methods:", ...) come
    # through as bold runs from the source DOCX so they stand out.
    s["abstract"] = ParagraphStyle(
        "abstract", parent=base,
        fontName=fn_normal, fontSize=10, leading=12.5,
        alignment=TA_JUSTIFY, spaceBefore=4, spaceAfter=4,
        leftIndent=0, rightIndent=0,
        allowWidows=0, allowOrphans=0,
    )
    # Section heading: keepWithNext so a heading is never the last line on
    # a page. spaceBefore is biased larger than spaceAfter to visually
    # separate from the previous section.
    s["section"] = ParagraphStyle(
        "section", parent=base,
        fontName=fn_bold, fontSize=13, leading=16,
        alignment=TA_LEFT, spaceBefore=14, spaceAfter=6,
        keepWithNext=1,
        allowWidows=0, allowOrphans=0,
    )
    s["subsection"] = ParagraphStyle(
        "subsection", parent=base,
        fontName=fn_bold, fontSize=11, leading=14,
        alignment=TA_LEFT, spaceBefore=8, spaceAfter=2,
        keepWithNext=1,
        allowWidows=0, allowOrphans=0,
    )
    s["body"] = ParagraphStyle(
        "body", parent=base,
        fontName=fn_normal, fontSize=10, leading=12.5,
        alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=6,
        firstLineIndent=0,
        allowWidows=0, allowOrphans=0,
    )
    # Display equation: own line, centered, slightly larger, italic-leaning
    # serif. Mirrors the JAMIA / academic-paper convention of breaking
    # mathematical identities out of body prose.
    s["equation"] = ParagraphStyle(
        "equation", parent=base,
        fontName=fn_italic, fontSize=10.5, leading=14,
        alignment=TA_CENTER, spaceBefore=6, spaceAfter=8,
        leftIndent=0, rightIndent=0,
    )
    s["body_indent"] = ParagraphStyle(
        "body_indent", parent=base,
        fontName=fn_normal, fontSize=10, leading=12.5,
        alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=6,
        firstLineIndent=14,
    )
    s["caption"] = ParagraphStyle(
        "caption", parent=base,
        fontName=fn_normal, fontSize=9, leading=11.5,
        alignment=TA_JUSTIFY, spaceBefore=4, spaceAfter=12,
        textColor=colors.HexColor("#222222"),
        allowWidows=0, allowOrphans=0,
    )
    s["caption_label"] = ParagraphStyle(
        "caption_label", parent=base,
        fontName=fn_bold, fontSize=9, leading=11.5,
        alignment=TA_LEFT, spaceBefore=0, spaceAfter=0,
    )
    s["alt_text"] = ParagraphStyle(
        "alt_text", parent=base,
        fontName=fn_italic, fontSize=8.5, leading=11,
        alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=10,
        textColor=colors.HexColor("#555555"),
    )
    s["table_caption"] = ParagraphStyle(
        "table_caption", parent=base,
        fontName=fn_normal, fontSize=9, leading=11.5,
        alignment=TA_JUSTIFY, spaceBefore=4, spaceAfter=4,
    )
    s["ref"] = ParagraphStyle(
        "ref", parent=base,
        fontName=fn_normal, fontSize=9, leading=11.5,
        alignment=TA_LEFT, leftIndent=18, firstLineIndent=-18,
        spaceAfter=2,
    )
    return s


# ── DOCX → reportlab inline-markup helpers ────────────────────────────
def runs_to_markup(p: DocxParagraph) -> str:
    """Convert a python-docx paragraph's runs to reportlab inline markup
    (preserves bold + italic). HTML-escape special chars."""
    parts = []
    for run in p.runs:
        text = run.text
        if not text:
            continue
        # XML-escape the text
        text = (text.replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;"))
        if run.bold:
            text = f"<b>{text}</b>"
        if run.italic:
            text = f"<i>{text}</i>"
        parts.append(text)
    out = "".join(parts).strip()
    # Collapse internal multiple spaces
    out = re.sub(r"  +", " ", out)
    return out


def text_with_inline_bold(text: str) -> str:
    """Convert simple **bold** markdown markers to <b>..</b> for reportlab."""
    text = (text.replace("&", "&amp;")
                 .replace("<", "&lt;").replace(">", "&gt;"))
    parts = text.split("**")
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(f"<b>{p}</b>")
        else:
            out.append(p)
    return "".join(out)


_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _convert_md_bold(markup: str) -> str:
    """Convert literal **bold** markdown that survived from the source DOCX
    body text into reportlab <b>...</b> markup. Safe on already-marked-up
    text because <b>...</b> contains no asterisks."""
    return _MD_BOLD_RE.sub(r"<b>\1</b>", markup)


# ── Section-rename + structure model ──────────────────────────────────
# JAMIA section label  →  self-contained-paper section label
# (None means: drop the heading entirely; merge subsequent paragraphs
# into the previous numbered section.)
SECTION_REWRITES = {
    "Background and Significance": "1. Introduction",
    "Objective": None,                     # merged into Introduction
    "Materials and Methods":         "2. Methods",
    "Results":                       "3. Results",
    "Discussion":                    "4. Discussion",
    "Conclusion":                    "5. Conclusion",
    "References":                    "References",
    "Figure Legends":                None,  # we render figures inline at end
    # Back-matter prose sections. Identity-mapped so their TITLES render as
    # section headings; their bodies must render as normal paragraphs (see
    # PROSE_BODY_SECTIONS) rather than being misread as run-in subsections.
    "Author Contributions":          "Author Contributions",
    "Funding":                       "Funding",
    "Conflict of Interest":          "Conflict of Interest",
    "Data Availability":             "Data Availability",
}

# Headings to recognize as section heads in the source DOCX (level 1).
KNOWN_SECTIONS = set(SECTION_REWRITES.keys())

# Back-matter sections whose body is plain prose. A single-sentence body
# here (e.g. the Funding or Conflict-of-Interest statement) would otherwise
# trip the run-in-subsection heuristic and lose its trailing period; force
# such bodies to render as normal paragraphs.
PROSE_BODY_SECTIONS = {
    "Author Contributions", "Funding", "Conflict of Interest",
    "Data Availability",
}


@dataclass
class Block:
    """Generic content block for the rendered PDF."""
    kind: str       # "section" | "subsection" | "para" | "equation" |
                    # "table" | "spacer"
    text: str = ""
    rows: List[List[str]] = field(default_factory=list)
    header_row: List[str] = field(default_factory=list)


# Detect-and-strip a trailing parenthesised reference to a figure, table,
# or panel ("(Figure 3, Panels A and B)", "(Table 3, Figure 4)",
# "(Figure S11)"). Headings should read as clean noun phrases, papers
# don't repeat the figure number in the heading because the section text
# below already names the figure.
_PAREN_FIG_REF_RE = re.compile(
    r"\s*[(\[][^)\]]*"
    r"(?:Figure|Table|Fig\.|Tab\.|Panel)[^)\]]*[)\]]\s*\.?$",
    re.IGNORECASE,
)
# Also handle a non-parenthesised trailing reference like "... Figure 3."
_NAKED_FIG_REF_RE = re.compile(
    r"[\s,;–-]+(?:see\s+)?(?:Figure|Table)\s+S?\d+[a-z]?\.?$",
    re.IGNORECASE,
)


def _strip_fig_ref(text: str) -> str:
    """Strip any trailing figure / table / panel reference from a heading
    so the heading is a clean noun phrase. Iterates because some headings
    have multiple compound refs."""
    while True:
        new = _PAREN_FIG_REF_RE.sub("", text)
        if new == text:
            break
        text = new
    text = _NAKED_FIG_REF_RE.sub("", text)
    return text.rstrip(" .,;:–-")


def _is_display_equation(raw_text: str) -> bool:
    """Detect a paragraph that should be rendered as a centered display
    equation (own line, italic, slightly larger). Conventions:
      • Source emits these with 4-space leading indentation (legacy
        "para(doc, '    E[...] = ...')" pattern carried over from the
        DOCX where it appears as an indented standalone line).
      • The text is short, contains an equality / approximation token,
        and is *only* math, no sentence-ending period.
    """
    # Source DOCX preserves leading whitespace via the Run text; we lose
    # it after .strip(). Detect via raw_text BEFORE strip.
    if raw_text.startswith("    ") or raw_text.startswith("\t"):
        return True
    return False


# ── Parse the main DOCX into a structured stream of blocks ────────────
def parse_main_docx(path: str) -> Tuple[List[Block], Dict[str, Any]]:
    """Returns (list_of_blocks, meta) where meta carries title + authors +
    abstract paragraphs separately."""
    doc = Document(path)
    blocks: List[Block] = []
    meta = {"title": "", "authors": "", "affiliations": [], "abstract": []}

    # Walk children of body in document order, including tables.
    body = doc.element.body
    paragraphs = list(doc.paragraphs)
    tables = list(doc.tables)
    p_idx = 0
    t_idx = 0

    # We need the iteration in document order, python-docx tables do not
    # come interleaved with paragraphs by default. We instead walk
    # body.iter() and dispatch per-element.
    from docx.oxml.ns import qn

    # Build maps
    p_by_xml = {p._p: p for p in paragraphs}
    t_by_xml = {t._tbl: t for t in tables}

    # Track context
    in_abstract = False
    in_figure_legends = False
    in_skipped_section = False   # True when iterating body of "Objective"
    seen_sections = []
    pending_section_label = None  # e.g. "1. Introduction"
    section_count = 0

    # Title metadata pulled from the first ~5 paragraphs
    title_done = False

    def looks_like_heading(p: DocxParagraph) -> bool:
        """Heuristic: matches a known section name OR is a non-bold short
        line that ends a major block. We rely on the explicit set above."""
        return p.text.strip() in KNOWN_SECTIONS

    for child in body.iterchildren():
        tag = child.tag
        if tag == qn("w:p"):
            p = p_by_xml.get(child)
            if p is None:
                continue
            t = p.text.strip()
            if not t:
                continue

            # ── Title block (first 5-ish paragraphs) ──────────────
            if not title_done:
                if t.startswith("AToP:") and not meta["title"]:
                    meta["title"] = t
                    continue
                # Authors line: contains both 'PhD' or 'MD' superscripts
                # heuristic: 2nd non-empty paragraph
                if not meta["authors"] and (
                        "Belouali" in t or "Kharrazi" in t):
                    meta["authors"] = t
                    continue
                if t.startswith("¹") or t.startswith("²"):
                    meta["affiliations"].append(t)
                    continue
                if t == "Abstract":
                    title_done = True
                    in_abstract = True
                    continue
                # Random other early lines (date, etc.), skip
                continue

            # ── Abstract paragraphs (until we hit B&S) ────────────
            if in_abstract:
                if t in KNOWN_SECTIONS or t in SECTION_REWRITES:
                    in_abstract = False
                else:
                    # Preserve the JAMIA structured-abstract sub-labels
                    # (Objective:/Materials and Methods:/Results:/...)
                    # using inline rich text so they render as bold leads.
                    md = runs_to_markup(p)
                    meta["abstract"].append(md)
                    continue

            # ── Section heading? ─────────────────────────────────
            if t in KNOWN_SECTIONS:
                rewrite = SECTION_REWRITES[t]
                if rewrite is None:
                    # Heading dropped (e.g. "Objective", "Figure Legends").
                    # For "Objective" we ALSO skip the body paragraphs since
                    # they duplicate the abstract's Objective sentence and
                    # the Methods opener; introduces redundant prose if
                    # included in Introduction.
                    if t == "Objective":
                        in_skipped_section = True
                    if t == "Figure Legends":
                        in_figure_legends = True
                    pending_section_label = None
                    continue
                # Real numbered section heading
                in_figure_legends = False
                in_skipped_section = False
                blocks.append(Block(kind="section", text=rewrite))
                pending_section_label = rewrite
                seen_sections.append(rewrite)
                continue

            # Skip body paragraphs in dropped sections.
            if in_figure_legends or in_skipped_section:
                continue

            # ── Subsection / body paragraph dispatch ─────────────
            # Subsection heuristic: short heading-like line set entirely
            # in bold (clean noun phrase, optional trailing period or
            # "(Figure X)" suffix). We accept slightly longer
            # headings up to 200 chars so multi-clause Results subsection
            # titles are recognized; the heading test then strips any
            # trailing "(Figure X, Panels Y)" reference for a paper-clean
            # rendering.
            stripped = _strip_fig_ref(t)
            # Subsection titles are set as one bold run by the DOCX
            # builder's heading(); body paragraphs mix bold lead-ins with
            # regular text. The trailing period is no longer required
            # (JAMIA style: no full stop after a subheading).
            _runs = [r for r in p.runs if r.text.strip()]
            all_bold = bool(_runs) and all(r.bold for r in _runs)
            heading_like = (
                len(stripped) <= 140
                and all_bold
                and t[0].isupper()
                and not t.startswith(("Figure ", "Table ", "Alt text"))
                and "  " not in t
                # No mid-text full-stop (other than abbreviations); a quick
                # proxy is: only one period besides the final one.
                and t.count(". ") == 0
                # Bodies of back-matter prose sections are full sentences,
                # not run-in subsection titles; keep them as paragraphs so
                # their trailing period survives.
                and pending_section_label not in PROSE_BODY_SECTIONS
            )
            if heading_like:
                blocks.append(Block(kind="subsection",
                                      text=stripped.rstrip(".")))
                continue

            # Display equation? (source pattern: 4-space leading indent)
            if _is_display_equation(p.text):
                blocks.append(Block(kind="equation", text=t))
                continue
            # Normal body paragraph, preserve <b> from DOCX runs and
            # also convert any literal **bold** markdown that survived
            # from the source builder script.
            md = _convert_md_bold(runs_to_markup(p))
            if not md:
                continue
            blocks.append(Block(kind="para", text=md))

        elif tag == qn("w:tbl"):
            t = t_by_xml.get(child)
            if t is None:
                continue
            header = [c.text.strip() for c in t.rows[0].cells]
            data = [[c.text.strip() for c in row.cells] for row in t.rows[1:]]
            blocks.append(Block(kind="table", header_row=header, rows=data))

    return blocks, meta


# ── Parse the supplement DOCX ─────────────────────────────────────────
# Match a heading-style figure/table reference like "Figure S1." or
# "Figure S4a." or "Table S2." Used to detect the duplicate heading lines
# that the supplement DOCX emits before each body caption.
_S_FIG_HEADING_RE = re.compile(
    r"^(Figure|Table)\s+S(\d+)([A-Za-z]?)\.\s*(.*)$"
)
# Heading keywords from the supplement DOCX front-matter that we want to
# skip when rendering the self-contained PDF (the supplement is shown
# inline as a section of the same document, so its own title page +
# table of contents are redundant).
_SUPP_FRONT_MATTER_HEADERS = {
    "ONLINE SUPPLEMENT",
    "Submitted to Journal of the American Medical Informatics Association",
    "Contents",
}


def parse_supplement_docx(path: str) -> List[Block]:
    """Returns a list of blocks for the supplement (sections, figures,
    captions, tables).

    The supplement DOCX has three regions in document order:
      1. Front matter (title, "Submitted to JAMIA", "Contents")
      2. Table of contents (one line per S-figure / S-table)
      3. Actual figures + captions (after a "Supplementary Figures" heading)
         and actual tables (after a "Supplementary Tables" heading)

    For the self-contained PDF we want only region 3; regions 1-2 are
    redundant given that a top-level "Supplementary Material" section
    heading is emitted in the rendering script. Inside region 3, each
    figure appears twice: a bold heading line "Figure SN. <title>" and
    then a body caption "Figure SN. <full caption>". We suppress the
    heading line and emit only the caption as a single bold-prefixed
    block, matching the main-figure convention.
    """
    doc = Document(path)
    blocks: List[Block] = []
    body = doc.element.body
    from docx.oxml.ns import qn
    paragraphs = list(doc.paragraphs)
    tables = list(doc.tables)
    p_by_xml = {p._p: p for p in paragraphs}
    t_by_xml = {t._tbl: t for t in tables}

    in_main_section = False  # True after the first region heading
    last_seen_fig_id: Optional[str] = None  # e.g. "S1" or "S4a"
    # The front page repeats every section and exhibit title as a contents
    # list; it ends with the "*This supplementary material..." footnote.
    front_done = False

    for child in body.iterchildren():
        tag = child.tag
        if tag == qn("w:p"):
            p = p_by_xml.get(child)
            if p is None:
                continue
            t = p.text.strip()
            if not t:
                continue

            if not front_done:
                if t.lstrip("*").startswith("This supplementary material"):
                    front_done = True
                continue

            # Region entry points. The text sections (Supplementary
            # Methods, Supplementary Results) are rendered too; only the
            # front page and its contents list are skipped.
            if t in ("Supplementary Methods", "Supplementary Results",
                     "Supplementary Figures", "Supplementary Tables"):
                in_main_section = True
                last_seen_fig_id = None
                blocks.append(Block(kind="supp_section", text=t))
                continue

            # Skip everything before the first region heading
            if not in_main_section:
                continue

            # Level-2 headings inside the text sections
            sty = (p.style.name or "") if p.style is not None else ""
            runs = [r for r in p.runs if r.text.strip()]
            all_bold = bool(runs) and all(r.bold for r in runs)
            is_heading = (sty.lower().startswith("heading")
                          or (all_bold and len(t) < 90 and not t.endswith(".")))
            if (is_heading and not _S_FIG_HEADING_RE.match(t)
                    and not t.startswith(("Alt text:", "Abbreviations", "Note"))):
                blocks.append(Block(kind="supp_subsection", text=t))
                continue

            # Skip residual front-matter headers if they re-appear
            if t in _SUPP_FRONT_MATTER_HEADERS:
                continue

            m = _S_FIG_HEADING_RE.match(t)
            if m:
                kind, num, suffix, rest = m.groups()
                fig_id = f"S{num}{suffix}"
                rest = rest.strip()
                # Tables in the supplement DOCX have only ONE title
                # paragraph (Table SN. <title>) followed by the table
                # object and abbreviations. Figures have TWO matching
                # paragraphs: a heading and a body caption. So for
                # Tables we always emit immediately; for Figures we
                # suppress the first occurrence and emit the second.
                if kind == "Table":
                    md = _convert_md_bold(runs_to_markup(p))
                    # Strip the "Table SN. " prefix AND any wrapping
                    # bold tags so we can re-wrap cleanly. The original
                    # paragraph may be wrapped in <b>...</b> end-to-end
                    # (table_title helper bolds the whole line); leaving
                    # the trailing </b> would create unbalanced markup.
                    md_clean = re.sub(
                        rf"^(?:<b>)?{kind}\s+{fig_id}\.\s*(?:</b>)?",
                        "", md,
                    ).strip()
                    # Drop any leftover unmatched </b> at end + any
                    # stray <b> at start
                    md_clean = re.sub(r"^</?b>\s*", "", md_clean)
                    md_clean = re.sub(r"\s*</?b>$", "", md_clean)
                    bold_prefix = f"<b>{kind} {fig_id} |</b> "
                    blocks.append(Block(kind="caption",
                                          text=bold_prefix + md_clean))
                    continue
                # ── Figure case ──
                if fig_id != last_seen_fig_id:
                    # First time we see this id → heading; suppress
                    last_seen_fig_id = fig_id
                    continue
                # Second time → emit as a bold-prefixed caption
                md = _convert_md_bold(runs_to_markup(p))
                md_clean = re.sub(
                    rf"^(?:<b>)?{kind}\s+{fig_id}\.\s*(?:</b>)?",
                    "",
                    md,
                ).strip()
                bold_prefix = f"<b>{kind} {fig_id} |</b> "
                blocks.append(Block(kind="caption",
                                      text=bold_prefix + md_clean))
                continue

            # Alt-text line, render in italic muted color
            if t.startswith("Alt text:"):
                md = _convert_md_bold(runs_to_markup(p))
                blocks.append(Block(kind="alt", text=md))
                continue

            # Body paragraph (preserves bold runs)
            md = _convert_md_bold(runs_to_markup(p))
            if not md:
                continue
            blocks.append(Block(kind="para", text=md))

        elif tag == qn("w:tbl"):
            if not in_main_section:
                continue
            t = t_by_xml.get(child)
            if t is None:
                continue
            header = [c.text.strip() for c in t.rows[0].cells]
            data = [[c.text.strip() for c in row.cells] for row in t.rows[1:]]
            blocks.append(Block(kind="table", header_row=header, rows=data))

    return blocks


# ── Page templates: header on page > 1 + page numbers ─────────────────
class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, pagesize=letter,
                          leftMargin=1.05 * inch,
                          rightMargin=1.05 * inch,
                          topMargin=0.95 * inch,
                          bottomMargin=0.95 * inch,
                          **kwargs)
        # Body frame (portrait, full margins).
        frame = Frame(self.leftMargin, self.bottomMargin,
                       self.width, self.height,
                       id="main", showBoundary=0,
                       leftPadding=0, rightPadding=0,
                       topPadding=0, bottomPadding=0)
        # Landscape figure frame (rotated page, narrower margins so the
        # figure gets the full 11" width). Margins kept symmetric.
        ls_lr = 0.55 * inch
        ls_tb = 0.65 * inch
        ls_w  = LETTER_LS[0] - 2 * ls_lr
        ls_h  = LETTER_LS[1] - 2 * ls_tb
        ls_frame = Frame(ls_lr, ls_tb, ls_w, ls_h,
                          id="ls_main", showBoundary=0,
                          leftPadding=0, rightPadding=0,
                          topPadding=0, bottomPadding=0)
        self._ls_pagesize = LETTER_LS
        self._ls_margins = (ls_lr, ls_tb)

        # Three templates: first page (no header), inner pages (header),
        # landscape pages for wide figures.
        self.addPageTemplates([
            PageTemplate(id="first", frames=[frame],
                          onPage=self._first_page),
            PageTemplate(id="inner", frames=[frame],
                          onPage=self._inner_page),
            PageTemplate(id="landscape", frames=[ls_frame],
                          pagesize=LETTER_LS,
                          onPage=self._landscape_page),
        ])

    def _first_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFont(_font("italic", italic=True), 9)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawRightString(letter[0] - self.rightMargin,
                                self.bottomMargin / 2,
                                f"{doc.page}")
        canvas.restoreState()

    def _inner_page(self, canvas, doc):
        canvas.saveState()
        # Top header: italic running title + thin rule below
        canvas.setFont(_font("italic", italic=True), 9)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(self.leftMargin,
                           letter[1] - self.topMargin / 2 + 6,
                           RUNNING_TITLE)
        canvas.setStrokeColor(colors.HexColor("#888888"))
        canvas.setLineWidth(0.4)
        canvas.line(self.leftMargin,
                     letter[1] - self.topMargin / 2,
                     letter[0] - self.rightMargin,
                     letter[1] - self.topMargin / 2)
        # Bottom: page number + rule
        canvas.line(self.leftMargin, self.bottomMargin / 2 + 12,
                     letter[0] - self.rightMargin,
                     self.bottomMargin / 2 + 12)
        canvas.drawRightString(letter[0] - self.rightMargin,
                                self.bottomMargin / 2,
                                f"{doc.page}")
        canvas.restoreState()

    def _landscape_page(self, canvas, doc):
        """Header + page number on landscape figure pages, scaled to the
        rotated page size."""
        canvas.saveState()
        ls_lr, ls_tb = self._ls_margins
        page_w, page_h = self._ls_pagesize
        canvas.setFont(_font("italic", italic=True), 9)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(ls_lr, page_h - ls_tb / 2 + 6, RUNNING_TITLE)
        canvas.setStrokeColor(colors.HexColor("#888888"))
        canvas.setLineWidth(0.4)
        canvas.line(ls_lr, page_h - ls_tb / 2,
                     page_w - ls_lr, page_h - ls_tb / 2)
        canvas.line(ls_lr, ls_tb / 2 + 12,
                     page_w - ls_lr, ls_tb / 2 + 12)
        canvas.drawRightString(page_w - ls_lr, ls_tb / 2,
                                  f"{doc.page}")
        canvas.restoreState()


# ── Render: blocks → reportlab Story ───────────────────────────────────
def block_to_story(blk: Block, S: Dict[str, ParagraphStyle]):
    """Translate a content Block to a list of reportlab Flowables."""
    if blk.kind == "section":
        return [Spacer(1, 4), Paragraph(blk.text, S["section"])]
    if blk.kind == "subsection":
        return [Paragraph(blk.text, S["subsection"])]
    if blk.kind == "para":
        return [Paragraph(blk.text, S["body"])]
    if blk.kind == "caption":
        return [Paragraph(blk.text, S["caption"])]
    if blk.kind == "alt":
        return [Paragraph(f"<i>{blk.text}</i>", S["alt_text"])]
    if blk.kind == "equation":
        # Centered, italic display equation on its own line. If the
        # source text contains a trailing "(N)" or "\t\t(N)" tag (the
        # equation number convention), strip it and render it right-
        # aligned in the right margin via a 3-column borderless Table:
        # [empty | centered equation | right-aligned (N)].
        raw = blk.text
        # Detect equation number "(1)", "(2)", possibly preceded by tabs
        # or extra whitespace.
        m = re.search(r"[\t ]*\((\d+[a-z]?)\)\s*$", raw)
        if m:
            eq_text = raw[:m.start()].rstrip()
            num_text = f"({m.group(1)})"
        else:
            eq_text = raw
            num_text = None

        eq_xml = (eq_text.replace("&", "&amp;")
                          .replace("<", "&lt;")
                          .replace(">", "&gt;"))
        if num_text is None:
            return [Paragraph(eq_xml, S["equation"])]

        # Build a 3-column Table: equal side columns + a center column
        # for the equation, with the equation number on the far right.
        eq_para = Paragraph(eq_xml, S["equation"])
        num_para = Paragraph(num_text,
                              ParagraphStyle(
                                  "eqnum", parent=S["body"],
                                  fontName=_font("normal"), fontSize=10.5,
                                  alignment=TA_LEFT, spaceBefore=0,
                                  spaceAfter=0))
        usable_w = letter[0] - 2 * 1.05 * inch
        # Side margin width = 0.6", center = remaining
        side = 0.6 * inch
        center = usable_w - 2 * side
        t = Table([[ "", eq_para, num_para ]],
                   colWidths=[side, center, side])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return [t]
    if blk.kind == "spacer":
        return [Spacer(1, 8)]
    if blk.kind == "table":
        out = _render_table(blk, S)
        return out if isinstance(out, list) else [out]
    return []


def _column_widths(header: List[str], n_cols: int, usable: float) -> List[float]:
    """Custom column-width policy keyed off the header text. Long-content
    columns ("Pattern", "label", "Variable") get extra width; short
    numeric columns (n, ΣIG, Δŷ, OR, p, AUROC) get squeezed; everything
    else falls back to equal widths.
    """
    # Per-column requested weight. Larger weight → wider column.
    WEIGHTS = {
        "Pattern": 4.5, "Pattern (representative)": 4.5,
        "Comparison": 3.0,
        "Label": 3.0,
        "Variable": 2.0,
        "Overall": 1.4, "Readmitted": 1.4, "Not readmitted": 1.5,
        "Phenotype": 1.6,
        "Direction": 1.0,
        "Model": 2.0,
        "AUROC": 0.9, "ΔAUROC": 0.9, "95% CI": 1.4,
        "p (two-sided)": 1.1,
        "ΣIG": 0.7, "Δŷ": 0.7, "OR": 0.7, "n": 0.5,
        "Family": 1.5, "ΣIG obs.": 0.85, "ΣIG rev.": 0.85, "Atten. %": 0.75,
        "Fell %": 0.7, "Sign %": 0.7, "p": 0.9,
    }
    if not header:
        return [usable / n_cols] * n_cols
    # Pad header to n_cols if mismatch (e.g., empty cells)
    h = list(header) + [""] * (n_cols - len(header))
    weights = [WEIGHTS.get(c.strip(), 1.0) for c in h]
    total = sum(weights) or float(n_cols)
    return [usable * w / total for w in weights]


def _render_table(blk: Block, S: Dict[str, ParagraphStyle]):
    if not blk.rows:
        return Spacer(1, 0)
    cell_style = ParagraphStyle("cell", parent=S["body"],
                                  fontSize=8.5, leading=10.5,
                                  alignment=TA_LEFT, spaceAfter=0)
    cell_center = ParagraphStyle("cellc", parent=cell_style,
                                   alignment=TA_CENTER)
    cell_indent = ParagraphStyle("celli", parent=cell_style, leftIndent=10)
    head_style = ParagraphStyle("head", parent=cell_style,
                                  fontName=_font("bold", bold=True))
    head_center = ParagraphStyle("headc", parent=head_style,
                                   alignment=TA_CENTER)

    import re as _re
    _wordy = _re.compile(r"[A-Za-z]{5,}")
    n_cols_all = max(len(r) for r in ([blk.header_row] if blk.header_row else []) + blk.rows)
    centered = []
    for ci in range(n_cols_all):
        cells = [str(r[ci]) for r in blk.rows if ci < len(r) and str(r[ci]).strip()]
        centered.append(bool(ci > 0 and cells and all(not _wordy.search(x) for x in cells)
                             and any(_re.search(r"\d", x) for x in cells)))

    def _cell(c, i, head=False):
        c = c or ""
        if head:
            return Paragraph(c, head_center if centered[i] else head_style)
        if c.startswith("  "):
            n_sp = len(c) - len(c.lstrip(" "))
            lvl = max(1, n_sp // 4)
            st = ParagraphStyle(f"celli{lvl}", parent=cell_style, leftIndent=10 * lvl)
            return Paragraph(c.lstrip(" "), st)
        return Paragraph(c, cell_center if centered[i] else cell_style)

    data = []
    if blk.header_row:
        data.append([_cell(c, i, True) for i, c in enumerate(blk.header_row)])
    for row in blk.rows:
        data.append([_cell(c, i) for i, c in enumerate(row)])
    n_cols = max(len(r) for r in data)
    usable = letter[0] - 2 * 1.05 * inch  # ≈ 6.4 inches
    col_w = _column_widths(blk.header_row, n_cols, usable)
    t = Table(data, colWidths=col_w, repeatRows=1 if blk.header_row else 0)
    style = TableStyle([
        ("FONT", (0, 0), (-1, -1), _font("normal"), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F2F2")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ])
    t.setStyle(style)
    # Tables may split across pages (header row repeats); wrapping a tall
    # table in KeepTogether pushed it whole to the next page and stranded
    # its caption.
    return [t, Spacer(1, 8)]


# ── Figure embedding helpers ──────────────────────────────────────────
def _figure_image(path: str, max_w: float, max_h: float) -> Optional[Image]:
    """Returns a scaled Image flowable for the given PNG path, or None if
    the file is missing."""
    if not os.path.exists(path):
        return None
    img = Image(path)
    iw, ih = img.imageWidth, img.imageHeight
    scale = min(max_w / iw, max_h / ih)  # allow upscaling to fill width
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    img.hAlign = "CENTER"
    return img


def _image_aspect(path: str) -> float:
    """Returns width / height aspect ratio of an image, or 1.0 if not
    readable."""
    if not os.path.exists(path):
        return 1.0
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w, h = im.size
            return w / h
    except Exception:
        return 1.0


# Figures wider than this aspect ratio get rendered on a landscape page
# so they fill more horizontal space and the embedded text stays legible.
LANDSCAPE_ASPECT_THRESHOLD = 1.20


# ── Figure caption metadata for the main paper ────────────────────────
MAIN_FIGURES = [
    {
        "n": 1,
        "label": "Figure 1",
        "file": "Figure1.png",
        "fallback_file": "Figure1.pptx",
        "caption": (
            "AToP framework overview. Five-step pipeline: "
            "(1) per-token Integrated Gradients on the trained "
            "Transformer; (2) salient-token selection (the tokens "
            "contributing 80% of per-patient attribution mass); "
            "(3) frequent-episode mining from training-cohort salient "
            "subsequences, with cross-visit locks (≥2 visits, ≥2 tokens) "
            "applied at reporting; (4) carrier-conditional token-level "
            "and pattern-level ΣIG aggregation; and (5) Jaccard ≥ 0.2 "
            "deduplication on test-set carrier sets, retaining the "
            "highest-magnitude representative per phenotype cluster."),
        "alt": (
            "Schematic flow diagram with five sequential steps linking "
            "the trained Transformer to per-token attributions, "
            "salient-token selection, frequent-episode mining, "
            "carrier-conditional aggregation, and Jaccard-based phenotype "
            "deduplication."),
    },
    {
        "n": 2,
        "label": "Figure 2",
        "file": "Figure2.png",
        "caption": (
            "Single-patient AToP explanation. (A) Visit timeline: each visit "
            "shows its top-3 salient tokens by |IG| color-coded by sign; the "
            "index visit (prediction point for next-30-day readmission) is "
            "outlined; lower axis shows real elapsed days and the observed "
            "R30 outcome. (B) Allocated signed Σ IG decomposition "
            "(logit space; not unique). Each bar = marginal Σ IG over "
            "(position, token) pairs newly claimed by one matched pattern "
            "under a deterministic non-overlapping position-level "
            "allocation; bars are ordered chronologically; Residual-salient "
            "and Non-salient bars close the gap to total Σ IG. By "
            "completeness, total Σ IG ≈ z(x) − z(x_baseline) "
            "on the logit scale; ŷ and ŷ_baseline are annotated for "
            "clinical context but are not additively related to Σ IG. "
            "Label prefixes show visit reach (V1→V5 = cross-visit; "
            "V5: = single-visit)."),
        "alt": (
            "Two-panel figure. Panel A shows a horizontal visit timeline "
            "with one box per admission, each containing the top salient "
            "tokens for that visit with signed IG values, and a real-days "
            "axis below. Panel B shows a horizontal SHAP-style chevron "
            "waterfall of pattern-level Δlogit contributions ordered "
            "chronologically, with a residual salient bar and a non-salient "
            "bar at the bottom."),
    },
    {
        "n": 3,
        "label": "Figure 3",
        "file": "Figure3.png",  # token-level (former panels A+B)
        "caption": (
            "Global token-level attribution on the test set: "
            "population-weighted vs carrier-conditional. (A) "
            "Population mean (all patients; non-carriers scored 0). "
            "(B) Carrier-conditional mean (among carriers). Let "
            "score_j = |IG|(j); the two panels are linked by "
            "E[score_j] = P(present(j)) × E[score_j | present(j)]. "
            "Top-30 tokens by absolute magnitude per panel. Panel B "
            "uses the canonical x-axis (±2.28); Panel A is on its own "
            "axis (±0.04) because population weighting reduces "
            "magnitudes by ~50×."),
        "alt": (
            "Two-panel horizontal bar chart. Left panel: population-"
            "weighted token attribution. Right panel: carrier-"
            "conditional token attribution. Bars are red for risk-"
            "increasing and blue for protective."),
    },
    {
        "n": 4,
        "label": "Figure 4",
        "file": "Figure4.png",  # pattern-level (former panels C+D)
        "caption": (
            "Global pattern-level attribution on the test set: all-"
            "pattern shapes vs cross-visit-only. (A) Carrier-"
            "conditional Σ IG, all pattern shapes (single-token, "
            "single-visit co-occurrence, and cross-visit patterns "
            "combined). (B) Carrier-conditional Σ IG, cross-visit "
            "only (≥2 visit blocks, ≥2 tokens; the canonical AToP "
            "output). Bars report attribution mass concentrated on "
            "constituent tokens under the non-overlapping position-"
            "level allocation rule of Figure 2; they are allocated "
            "attribution mass, not identified pattern-level effects. "
            "Top-30 representatives by |Σ IG|, after Jaccard ≥ 0.2 "
            "deduplication. Both panels share x-axis (±2.28)."),
        "alt": (
            "Two-panel horizontal bar chart of pattern-level signed "
            "Σ IG. Left panel: all pattern shapes combined. Right "
            "panel: cross-visit patterns only. Bars are red for risk-"
            "increasing and blue for protective."),
    },
    {
        "n": 5,
        "label": "Figure 5",
        "file": "Figure5.png",  # validation dumbbell (was Figure 4)
        "caption": (
            "Perturbation-validation 2-panel dumbbell. Note on scale: "
            "the masking effect Δŷ is in predicted-probability space, "
            "while IG is in pre-sigmoid logit space; the two are not "
            "directly additive across this figure and Figures 2/3/4. "
            "(A) Top-30 representatives from all patterns; (B) Top-30 "
            "cross-visit pattern representatives (≥2 visits, ≥2 tokens). "
            "Each row shows mean Δŷ = ŷ_original − ŷ_masked across the "
            "pattern's test-set carriers; n annotated. Δŷ > 0 (right, "
            "red): masking lowered predicted risk, so the pattern is "
            "risk-supporting; Δŷ < 0 (left, blue): masking raised "
            "predicted risk, so the pattern is protective. Color and "
            "orientation match Figures 3 and 4. Δŷ is per-instance under "
            "the chosen non-overlapping allocation; alternative "
            "allocation rules would distribute Δŷ differently."),
        "alt": (
            "Two-panel dumbbell figure. Each row shows a colored "
            "arrow from zero to the masking effect, red for risk-"
            "supporting patterns (right), blue for protective (left). "
            "Left panel = all patterns. Right panel = cross-visit "
            "patterns."),
    },
]


def parse_figure_legends(path):
    """Read the main DOCX's 'Figure Legends' section so the PDF renders the
    SAME legend text as the DOCX.

    The legend section is the single source of truth: the FIGURES table below
    keeps hardcoded captions only as a fallback for figures the DOCX does not
    describe. Returns {label: {"caption": str, "alt": str}} keyed by the
    'Figure N' label used in FIGURES.
    """
    from docx import Document as _Doc
    out, in_legends, cur = {}, False, None
    for p in _Doc(path).paragraphs:
        t = (p.text or "").strip()
        if not t:
            continue
        if t == "Figure Legends":
            in_legends = True
            continue
        if not in_legends:
            continue
        # A new top-level section heading ends the legend block.
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


# ── Build! ────────────────────────────────────────────────────────────
ARXIV_PDF = os.path.join(HERE, "..", "..", "manuscript", "jamia_submission",
                          "AToP_arxiv_preprint.pdf")

# Shown under the affiliations in --preprint mode. JAMIA permits posting the
# Author's Original Version to a preprint server before acceptance; once the
# paper is accepted this line must say so instead, and the preprint may not be
# updated further.
PREPRINT_NOTE = ("Preprint: author's original version, not peer reviewed. "
                 "Under review. Code: https://github.com/belouali/atop.")


def build(preprint: bool = False):
    print(f"[1] Parsing main DOCX: {os.path.basename(MAIN_DOCX)}")
    main_blocks, meta = parse_main_docx(MAIN_DOCX)
    docx_legends = parse_figure_legends(MAIN_DOCX)
    print(f"    figure legends parsed from DOCX: {len(docx_legends)} "
          f"({', '.join(sorted(docx_legends))})")
    print(f"    {len(main_blocks)} content blocks; "
          f"abstract paragraphs: {len(meta['abstract'])}")

    print(f"[2] Parsing supplement DOCX: {os.path.basename(SUPP_DOCX)}")
    supp_blocks = parse_supplement_docx(SUPP_DOCX)
    print(f"    {len(supp_blocks)} supplement blocks")

    S = build_styles()
    doc = PaperDocTemplate(ARXIV_PDF if preprint else OUT_PDF)
    story = []

    # ── Title block ────────────────────────────────────────────────
    story.append(Paragraph(meta["title"] or RUNNING_TITLE, S["title"]))
    if meta["authors"]:
        story.append(Paragraph(meta["authors"], S["author"]))
    for aff in meta["affiliations"]:
        story.append(Paragraph(aff, S["affiliation"]))
    if preprint:
        story.append(Spacer(1, 4))
        story.append(Paragraph(PREPRINT_NOTE, S["affiliation"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%",
                              thickness=0.4,
                              color=colors.HexColor("#888888"),
                              spaceBefore=2, spaceAfter=10))

    # ── Abstract block, JAMIA structured format (Objective: / ...) ──
    # Render as one paragraph per sub-section so the bold leading labels
    # ("Objective:", "Materials and Methods:", etc.) are visually distinct.
    if meta["abstract"]:
        for ab in meta["abstract"]:
            story.append(Paragraph(ab, S["abstract"]))
        story.append(HRFlowable(width="100%",
                                  thickness=0.4,
                                  color=colors.HexColor("#888888"),
                                  spaceBefore=2, spaceAfter=10))

    # Switch to inner-page template (with running title) for everything
    # past page 1.
    story.append(NextPageTemplate("inner"))

    # ── Main body sections ─────────────────────────────────────────
    # Caption-table pairing strategy:
    #   • Caption uses keepWithNext=1, so reportlab treats the caption
    #     and the immediately following Table as a kept pair.
    #   • A CondPageBreak before the caption ensures we have at least
    #     ~6-7 table rows of room remaining; otherwise the pair starts
    #     on the next page. This prevents both (a) the orphan-caption
    #     case (caption at bottom, table at top of next page) and
    #     (b) the giant-whitespace case (entire pair pushed off a
    #     mostly-empty page).
    #   • We do NOT wrap in KeepTogether because for splittable Tables
    #     reportlab over-estimates the bundle height and pushes pairs
    #     even when they would fit, leaving large blank regions.
    table_cap_re = re.compile(r"^\s*(?:<b>)?Table\s+\d+\b", re.IGNORECASE)
    cap_keep_style = ParagraphStyle(
        "table_cap_keep", parent=S["body"],
        keepWithNext=1, spaceAfter=2,
    )
    # ~14 short rows × 18 pt + caption ~ 90 pt ≈ 340 pt
    MIN_CAPTION_TABLE_SPACE = 320

    n = len(main_blocks)
    i = 0
    while i < n:
        blk = main_blocks[i]
        if (blk.kind == "para"
                and i + 1 < n
                and main_blocks[i + 1].kind == "table"
                and table_cap_re.match(blk.text)):
            story.append(CondPageBreak(MIN_CAPTION_TABLE_SPACE))
            story.append(Paragraph(blk.text, cap_keep_style))
            story.extend(block_to_story(main_blocks[i + 1], S))
            i += 2
            continue
        story.extend(block_to_story(blk, S))
        i += 1

    # ── Figures section ────────────────────────────────────────────
    # Each main figure starts on its own page. Wide figures (aspect
    # ratio > LANDSCAPE_ASPECT_THRESHOLD) are routed to landscape
    # (rotated) pages so the image fills 11" of horizontal space
    # instead of 6.4", roughly doubling the readable size of dense
    # 4-panel figures like Figure 3.

    # Portrait page content area (used for narrow figures):
    page_w = letter[0] - 2 * 1.05 * inch
    page_h = letter[1] - 2 * 0.95 * inch
    fig_max_h_p = page_h * 0.92   # tall portrait figures keep print size; caption flows on
    # Landscape page content area:
    ls_w = LETTER_LS[0] - 2 * 0.55 * inch
    ls_h = LETTER_LS[1] - 2 * 0.65 * inch
    # Reserve ~25% of the landscape page for the caption + alt text.
    # Higher than 0.78 risks the caption + alt overflowing onto a
    # second page (a half-empty wasted page per figure); lower than
    # 0.72 leaves visible empty whitespace at the bottom.
    fig_max_h_ls = ls_h * 0.75

    for f in MAIN_FIGURES:
        fpath = os.path.join(FIG_DIR, f["file"])
        aspect = _image_aspect(fpath)
        use_landscape = aspect > LANDSCAPE_ASPECT_THRESHOLD

        # Switch templates and start a fresh page in the chosen mode.
        if use_landscape:
            story.append(NextPageTemplate("landscape"))
            story.append(PageBreak())
            img_w, img_h = ls_w, fig_max_h_ls
        else:
            story.append(NextPageTemplate("inner"))
            story.append(PageBreak())
            img_w, img_h = page_w, fig_max_h_p

        img = _figure_image(fpath, img_w, img_h)
        if img is not None:
            story.append(img)
        else:
            ph = Paragraph(
                f"<i>[Figure {f['n']} placeholder, source available "
                f"as <b>{f.get('fallback_file', f['file'])}</b>; "
                f"hand-export to PNG at 300 DPI before submission.]</i>",
                S["caption"])
            story.append(ph)
        # Single source of truth: prefer the DOCX legend text so the PDF and
        # the DOCX can never drift apart. Fall back to the hardcoded caption
        # only if this figure has no legend in the DOCX.
        _lg = docx_legends.get(f["label"], {})
        _cap = _lg.get("caption") or f["caption"]
        _alt = _lg.get("alt") or f["alt"]
        cap_text = (f"<b>{f['label']} |</b> " + _cap)
        story.append(Paragraph(cap_text, S["caption"]))
        story.append(Paragraph(f"<i>Alt text: {_alt}</i>",
                                  S["alt_text"]))

        # Restore portrait template for the following content (so the
        # next figure or section starts on a portrait page).
        if use_landscape:
            story.append(NextPageTemplate("inner"))

    # ── Supplementary Material section ────────────────────────────
    # The supplement parser already skipped the supplement DOCX's own
    # front matter (title + "Submitted to JAMIA" + Contents TOC) and
    # de-duplicated each figure's bold heading line so only one caption
    # is emitted per figure.
    #
    # We do NOT emit a stand-alone "Supplementary Material" heading on
    # its own page (that would create an orphan-heading whitespace
    # page). Instead the heading is placed at the top of the first
    # S-figure page, directly above the image, then each subsequent
    # S-figure starts on its own page.
    story.append(PageBreak())
    section_emitted = False

    # Match the figure id ("S1", "S4a", etc.) at the start of a caption
    # block: caption text begins "<b>Figure SN |</b> ...".
    s_fig_re = re.compile(r"^<b>(Figure|Table)\s+(S\d+[A-Za-z]?)\s*\|</b>")

    # Explicit figure-id -> file map from the supplement builder, so the
    # PDF cannot pick up a stale or renumbered file by name prefix.
    try:
        from build_revised_supplement_docx import IMAGE_PATHS as SUPP_IMAGE_PATHS
    except Exception:
        SUPP_IMAGE_PATHS = {}

    supp_heading_style = ParagraphStyle(
        "supp_heading", parent=S["subsection"], spaceBefore=10, spaceAfter=4)
    supp_subheading_style = ParagraphStyle(
        "supp_subheading", parent=S["body"], fontName=_font("bold", bold=True),
        spaceBefore=8, spaceAfter=2, keepWithNext=1)

    for blk in supp_blocks:
        if blk.kind in ("supp_section", "supp_subsection"):
            if not section_emitted:
                story.append(NextPageTemplate("inner"))
                story.append(PageBreak())
                story.append(Paragraph("Supplementary Material", S["section"]))
                section_emitted = True
            if blk.kind == "supp_section" and blk.text in ("Supplementary Figures",
                                                           "Supplementary Tables"):
                continue   # each figure/table starts its own page below
            story.append(Paragraph(blk.text, supp_heading_style if blk.kind == "supp_section"
                                   else supp_subheading_style))
            continue
        if blk.kind == "caption":
            m = s_fig_re.match(blk.text)
            if m:
                kind, fid = m.groups()
                # Locate the matching PNG(s) so we can sniff the aspect
                # ratio and decide portrait vs landscape. A figure may map
                # to several files (a figure spanning pages): each file
                # gets its own page and the caption follows the last one.
                cands = []
                if kind == "Figure":
                    mapped = SUPP_IMAGE_PATHS.get(f"Figure {fid}")
                    if isinstance(mapped, (list, tuple)):
                        cands = [m for m in mapped if m and os.path.exists(m)]
                    elif mapped and os.path.exists(mapped):
                        cands = [mapped]
                    if not cands and os.path.isdir(SUP_FIG_DIR):
                        for fn in sorted(os.listdir(SUP_FIG_DIR)):
                            if (fn.startswith(f"Figure{fid}_")
                                    and fn.endswith(".png")):
                                cands = [os.path.join(SUP_FIG_DIR, fn)]
                                break
                    if not cands:
                        print(f"  [WARN] no image for Figure {fid}")
                use_landscape = False
                for cand in (cands or [None]):
                    aspect = _image_aspect(cand) if cand else 1.0
                    use_landscape = (kind == "Figure"
                                      and aspect > LANDSCAPE_ASPECT_THRESHOLD)
                    if section_emitted:
                        story.append(NextPageTemplate(
                            "landscape" if use_landscape else "inner"))
                        story.append(PageBreak())
                    else:
                        # First S-figure inherits the page already created
                        # earlier so the "Supplementary Material" heading
                        # can sit at its top. The first figure stays on a
                        # portrait page so the heading style matches the
                        # rest of the document.
                        story.append(Paragraph("Supplementary Material",
                                                  S["section"]))
                        section_emitted = True
                        use_landscape = False  # keep heading on portrait
                    if cand:
                        if use_landscape:
                            img = _figure_image(cand, ls_w, fig_max_h_ls)
                        else:
                            img = _figure_image(cand, page_w, fig_max_h_p)
                        if img is not None:
                            story.append(img)
                # Emit the caption + (next iteration will emit the alt
                # text if present).
                story.extend(block_to_story(blk, S))
                if use_landscape:
                    # Restore portrait for the following content.
                    story.append(NextPageTemplate("inner"))
                continue
        story.extend(block_to_story(blk, S))

    # ── Build! ─────────────────────────────────────────────────────
    print(f"[3] Building PDF: {ARXIV_PDF if preprint else OUT_PDF}")
    doc.build(story)
    _out = ARXIV_PDF if preprint else OUT_PDF
    sz = os.path.getsize(_out) / 1024
    print(f"    Wrote {_out} ({sz:.0f} KB)")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--preprint", action="store_true",
                     help="emit the arXiv preprint variant (adds a preprint "
                          "status line; writes AToP_arxiv_preprint.pdf)")
    build(**vars(_ap.parse_args()))
