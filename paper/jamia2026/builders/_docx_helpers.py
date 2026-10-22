"""Shared formatting helpers for the JAMIA submission DOCX builders.

Used by ``build_revised_main_docx.py`` and ``build_revised_supplement_docx.py``.
Centralizes heading styles, table formatting, citation markers, and figure /
table placeholder paragraphs so both scripts produce visually consistent output.
"""
from __future__ import annotations

import os
import re

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt, Inches


# ── Style constants (JAMIA submission style guide) ────────────────────
BODY_FONT       = "Times New Roman"
BODY_SIZE       = 11
HEADING1_SIZE   = 14
HEADING2_SIZE   = 12
HEADING3_SIZE   = 11
TITLE_SIZE      = 16
LINE_SPACING    = 2.0   # JAMIA requires double-spaced manuscripts
TEXT_WIDTH_IN   = 6.5   # letter paper, 1-inch margins
HEADER_FILL     = "F2F2F2"


LEGEND_SPACING = 1.5   # figure legends and alt text, main and supplement (user rule, 2026-10-17)


def set_body_line_spacing(value: float) -> None:
    """Set the body line spacing used by new_document() and para()
    (2.0 for the main manuscript, 1.5 for the supplement)."""
    global LINE_SPACING
    LINE_SPACING = value


# ── Document setup ────────────────────────────────────────────────────
def new_document() -> Document:
    """Create a Document with sensible defaults: Times New Roman 11,
    double line spacing (JAMIA requirement), 1-inch margins on letter-size
    paper. Tables and captions override to tighter spacing locally."""
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(BODY_SIZE)
    pf = style.paragraph_format
    pf.line_spacing = LINE_SPACING
    pf.space_after = Pt(6)

    for section in doc.sections:
        section.top_margin    = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin   = Inches(1)
        section.right_margin  = Inches(1)
        _add_page_number_footer(section)
    return doc


def _add_page_number_footer(section) -> None:
    """Insert a centered "Page X of Y"-style page-number field into the
    section's footer. Uses Word's PAGE field code so it auto-updates as
    the document length changes, required by JAMIA Author Instructions."""
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Build: <run><fldChar fldCharType="begin"/></run><run><instrText> PAGE </instrText></run>
    #        <run><fldChar fldCharType="end"/></run>
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


# ── Section helpers ───────────────────────────────────────────────────
def title(doc, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(TITLE_SIZE)
    run.bold = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)


def heading(doc, text: str, level: int = 1) -> None:
    sizes = {1: HEADING1_SIZE, 2: HEADING2_SIZE, 3: HEADING3_SIZE}
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(sizes.get(level, BODY_SIZE))
    run.bold = True
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after  = Pt(6)


# House style: every in-text reference to a figure or table ("Figure 2B",
# "Figures S2A\u2013C", "Tables 3\u20134", "Tables S4 and S5") is set in bold.
# Applied automatically to body paragraphs, figure captions and table
# legends, so builders never need to mark these by hand.
_EXHIBIT_RE = re.compile(
    r"\b(?:Figures?|Tables?)[ \u00a0-]S?\d+[A-Za-z]?"
    r"(?:\s*[\u2013-]\s*(?:S?\d+)?[A-Za-z]?)?"
    r"(?:(?:,\s*|,?\s+and\s+|\s+or\s+)S?\d+[A-Za-z]?"
    r"(?:\s*[\u2013-]\s*(?:S?\d+)?[A-Za-z]?)?)*"
)


def bold_exhibit_refs(text: str) -> str:
    """Wrap figure/table references in ``**..**`` outside existing bold
    spans; adjacent spans are merged so the ``**`` parity stays valid."""
    if not text or not _EXHIBIT_RE.search(text):
        return text
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append(part)
        else:
            out.append(_EXHIBIT_RE.sub(lambda m: "**" + m.group(0) + "**", part))
    return "".join(out).replace("****", "")


# Numeric citations written as ``[n]``, ``[n,m]`` or ``[n–m]`` in the
# builders are rendered JAMIA-style: superscript numerals placed after any
# directly following punctuation ("models [3,4]." -> "models.<sup>3,4</sup>").
_CITE_RE = re.compile(r"\s*\[(\d+(?:[,–-]\d+)*)\]([.,])?")  # AMA: after periods and commas, inside colons and semicolons


def _emit_runs(paragraph, text: str, *, bold: bool = False,
               italic: bool = False, font_size_pt: int | None = None,
               cite_superscript: bool = True) -> None:
    """Add ``text`` to ``paragraph`` as runs, converting numeric citations
    to superscripts (see ``_CITE_RE``) unless ``cite_superscript`` is False."""
    def _run(t: str, superscript: bool = False):
        if not t:
            return
        r = paragraph.add_run(t)
        r.bold = bold
        r.italic = italic
        if font_size_pt is not None:
            r.font.size = Pt(font_size_pt)
        if superscript:
            r.font.superscript = True
    if not cite_superscript or not _CITE_RE.search(text):
        _run(text)
        return
    pos = 0
    for m in _CITE_RE.finditer(text):
        _run(text[pos:m.start()])
        if m.group(2):
            _run(m.group(2))
        _run(m.group(1), superscript=True)
        pos = m.end()
    _run(text[pos:])


def para(doc, text: str, *, italic: bool = False, bold: bool = False,
         cite_superscript: bool = True) -> None:
    """A normal-style paragraph; supports inline ** ** for bold runs.
    Figure and table references are bolded automatically; numeric
    citations become superscripts unless ``cite_superscript`` is False
    (used for the reference list itself)."""
    text = bold_exhibit_refs(text)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = LINE_SPACING
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        _emit_runs(p, part, bold=bold or (i % 2 == 1), italic=italic,
                   cite_superscript=cite_superscript)


def equation(doc, text: str, *, number: int | str | None = None) -> None:
    """Render a displayed mathematical equation on its own line.

    The output is a centered, italic paragraph with single line spacing.
    If ``number`` is given (e.g. ``1``, ``2``, ``"2a"``), an equation
    label like ``(1)`` is right-aligned at the page's right margin via
    a tab stop, matching the JAMIA / general-academic convention.

    The PDF builder (``build_self_contained_paper_pdf.py``) detects
    equation paragraphs by their leading 4-space indent and trailing
    ``(N)`` token; this helper preserves that wire format so the PDF
    rendering is unchanged. Authors of the build scripts no longer
    need to know about the wire format, they just call
    ``equation(doc, "Σᵢ IGᵢ ≈ z(x) − z(x_baseline)", number=1)``.

    Args:
        doc: the python-docx Document.
        text: the equation as plain text (use Unicode for math symbols
           , Σ, ŷ, ≈, etc.).
        number: optional equation number. Numbered equations render as
            "<text>      (number)" with the number in the right margin.
    """
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(6)
    pf.space_after  = Pt(8)
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Wire format expected by the PDF builder: leading 4 spaces +
    # trailing tabs + "(N)" if numbered. Tabs render correctly in Word
    # because the centered alignment + page width pushes the number to
    # the right edge naturally on the rendered line.
    body = "    " + text
    if number is not None:
        body = body + f"\t\t\t({number})"
    run = p.add_run(body)
    run.italic = True


def figure_placeholder(doc, label: str, legend: str, *,
                         alt: str = "",
                         image_path: str | None = None,
                         max_width_in: float = 6.5) -> None:
    """Insert a figure block: embed image if ``image_path`` exists, else fall
    back to a centered ``[FIGURE PLACEHOLDER: paste X here]`` line for manual
    insertion. Always renders the caption + alt-text below.

    The image is embedded full-width (``max_width_in`` inches, defaults to
    6.5” which is 8.5” page − 1” margins on each side).
    Word auto-scales the height to preserve aspect ratio.
    """
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)

    if image_path and os.path.exists(image_path):
        p.add_run().add_picture(image_path, width=Inches(max_width_in))
    else:
        r = p.add_run(f"[FIGURE PLACEHOLDER: paste {label} image here]")
        r.italic = True

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = LEGEND_SPACING
    r = p.add_run(f"{label}. ")
    r.bold = True
    _add_runs_with_inline_bold(p, legend, italic=False)

    if alt:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = LEGEND_SPACING
        r = p.add_run("Alt text: ")
        r.italic = True
        _add_runs_with_inline_bold(p, alt, italic=False)


def figure_landscape(doc, label: str, heading_text: str, legend: str, *,
                     image_path: str | None, width_in: float = 9.4,
                     margin_in: float = 0.6) -> None:
    """Insert a figure drawn at landscape print size (about 9.4 in wide) in
    its own landscape section: heading, image and caption on one page. The
    caption is set compact (9 pt, single spaced, the size the merged
    document gives every supplementary caption) because at body size and
    double spacing it would spill onto a second landscape page. The page
    setup in force before the call is restored in a new section, so the
    next exhibit starts on a fresh page without a page break. A missing
    image falls back to a placeholder line, as in ``figure_placeholder``.
    """
    from docx.enum.section import WD_ORIENT, WD_SECTION

    prev = doc.sections[-1]
    saved = {k: getattr(prev, k) for k in (
        "page_width", "page_height", "orientation", "left_margin",
        "right_margin", "top_margin", "bottom_margin")}

    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    w, h = sec.page_width, sec.page_height
    if w < h:
        sec.page_width, sec.page_height = h, w
    sec.orientation = WD_ORIENT.LANDSCAPE
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(margin_in))

    heading(doc, heading_text, level=2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    if image_path and os.path.exists(image_path):
        p.add_run().add_picture(image_path, width=Inches(width_in))
    else:
        r = p.add_run(f"[FIGURE PLACEHOLDER: paste {label} image here]")
        r.italic = True

    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = LEGEND_SPACING
    pf.space_before = Pt(6)
    pf.space_after = Pt(0)
    r = p.add_run(f"{label}. ")
    r.bold = True
    r.font.size = Pt(9)
    _add_runs_with_inline_bold(p, legend, italic=False, font_size_pt=9)

    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    for k, v in saved.items():
        setattr(sec, k, v)


def table_caption(doc, label: str, caption: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(f"{label}. ")
    r.bold = True
    p.add_run(caption)
    p.paragraph_format.space_before = Pt(12)


def _add_runs_with_inline_bold(paragraph, text: str, *,
                                 italic: bool = True,
                                 font_size_pt: int | None = None) -> None:
    """Add `text` to `paragraph` as runs, parsing ``**bold**`` markers
    into separate bold runs. All runs inherit `italic` and `font_size_pt`.
    """
    text = bold_exhibit_refs(text)
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        is_bold = part.startswith("**") and part.endswith("**")
        body = part[2:-2] if is_bold else part
        _emit_runs(paragraph, body, bold=is_bold, italic=italic,
                   font_size_pt=font_size_pt)


def table_title(doc, label: str, short_title: str) -> None:
    """Render the table title paragraph ABOVE the table.

    Format: ``**{label}.** {short_title}`` (label bold, title regular; house table style)
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{label}. ")
    r.bold = True
    _add_runs_with_inline_bold(p, short_title, italic=False)


def table_legend(doc, abbreviations: str | None = None,
                   note: str | None = None,
                   legend_size_pt: int = 9) -> None:
    """Render the legend paragraph BELOW the table.

    Format:
      ``***Abbreviations:*** *<abbrevs>*``  (italic, legend size, with
                                              inline **bold** parsed)
      <line break>
      ``***Note:*** *<note>*``               (same)

    Skips paragraph if both inputs are None/empty. Inline ``**bold**``
    markers in the text are properly rendered as bold runs.
    """
    if not (abbreviations or note):
        return
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15

    def _add_segment(label_txt: str, body_txt: str) -> None:
        rl = p.add_run(label_txt)
        rl.bold = True
        rl.font.size = Pt(legend_size_pt)
        # body: parse inline **bold** markers
        _add_runs_with_inline_bold(p, " " + body_txt,
                                     italic=False,
                                     font_size_pt=legend_size_pt)

    if abbreviations:
        _add_segment("Abbreviations:", abbreviations)
    if note:
        if abbreviations:
            br_run = p.add_run()
            br_run.add_break()
        _add_segment("Note:", note)


# Backwards-compat shim: keep table_caption_with_legend as title+legend
# above the table for any callers that haven't migrated. Prefer the
# split table_title/table_legend pair (legend BELOW table) for new code.
def table_caption_with_legend(doc, label: str, short_title: str,
                                abbreviations: str | None = None,
                                note: str | None = None,
                                legend_size_pt: int = 9) -> None:
    table_title(doc, label, short_title)
    table_legend(doc, abbreviations=abbreviations, note=note,
                  legend_size_pt=legend_size_pt)


SUBROW_INDENT_IN = 0.17   # indent for sub-rows written with leading spaces


def _tighten_cell(cell, *, bold: bool, text: str,
                    font_size_pt: int = BODY_SIZE - 1,
                    align: str = "left", line_spacing: float = 1.15) -> None:
    """Render `text` inside `cell` with single line spacing and zero
    paragraph margins so the table renders compact (Word's default
    inherits the document's line_spacing = 1.5, which makes tables
    balloon vertically). Also sets a small inner cell margin so text
    doesn't crowd the borders."""
    p = cell.paragraphs[0]
    pf = p.paragraph_format
    pf.line_spacing = line_spacing
    pf.space_before = Pt(0)
    pf.space_after  = Pt(0)
    p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if align == "center"
                   else WD_ALIGN_PARAGRAPH.LEFT)
    if text.startswith("  "):
        n_sp = len(text) - len(text.lstrip(" "))
        level = max(1, n_sp // 4)
        text = text.lstrip(" ")
        pf.left_indent = Inches(SUBROW_INDENT_IN * level)
    run = p.add_run(text)
    run.font.size = Pt(font_size_pt)
    if bold:
        run.bold = True


def _set_cell_margins(table, top_pt: float = 1.5, bottom_pt: float = 1.5,
                       left_pt: float = 3, right_pt: float = 3) -> None:
    """Apply tight inner cell margins on the whole table (in twentieths
    of a point: 1 pt = 20 units)."""
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    margins = OxmlElement("w:tblCellMar")
    for side, val in (("top", top_pt), ("bottom", bottom_pt),
                       ("left", left_pt), ("right", right_pt)):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), str(int(round(val * 20))))
        m.set(qn("w:type"), "dxa")
        margins.append(m)
    tbl_pr.append(margins)


def _auto_col_widths(header_row, data_rows, centered, font_size_pt,
                     total_in: float = TEXT_WIDTH_IN) -> list[float]:
    """Column widths (inches) that fill the text width: numeric columns
    take just what their longest cell needs; text columns share the
    rest, each capped at its natural width when there is room, with any
    surplus going to the widest text column (the pattern labels)."""
    cw = 0.0068 * font_size_pt          # average character width, inches
    pad = 0.13                          # cell margins + slack
    n = len(header_row)
    nat, kind = [], []
    for c in range(n):
        cells = [str(r[c]) for r in data_rows if c < len(r) and str(r[c]).strip()]
        hdr_word = max((len(w) for w in str(header_row[c]).split()), default=0)
        if centered[c]:
            L = max([len(x) for x in cells] + [hdr_word])
            nat.append(L * cw + pad); kind.append("fixed")
        else:
            L = max([len(x.strip()) for x in cells] + [len(str(header_row[c]))])
            nat.append(L * cw + pad); kind.append("flex")
    # Short label columns (anything but the widest text column) wrap at
    # word boundaries, so they only need their longest word; the room
    # saved goes to the pattern labels.
    flex_all = [i for i, k in enumerate(kind) if k == "flex"]
    if len(flex_all) > 1:
        widest0 = max(flex_all, key=lambda i: nat[i])
        for c in flex_all:
            if c == widest0:
                continue
            cells = [str(r[c]) for r in data_rows if c < len(r) and str(r[c]).strip()]
            words = [w for x in cells for w in x.split()] + str(header_row[c]).split()
            Lw = max((len(w) for w in words), default=1)
            nat[c] = min(nat[c], max(Lw * cw + pad, 0.6))
    widths = list(nat)
    fixed_total = sum(w for w, k in zip(nat, kind) if k == "fixed")
    flex_idx = [i for i, k in enumerate(kind) if k == "flex"]
    remaining = total_in - fixed_total
    if flex_idx and remaining > 0:
        flex_nat = sum(nat[i] for i in flex_idx)
        widest = max(flex_idx, key=lambda i: nat[i])
        others = sum(nat[i] for i in flex_idx if i != widest)
        if flex_nat <= remaining:
            widths[widest] += remaining - flex_nat
        else:
            # Short text columns (group labels, "Direction") keep their
            # natural width; the wide ones (pattern labels) share the
            # shortfall. Fall back to scaling everything if even that
            # leaves the wide columns too narrow.
            short = [i for i in flex_idx if nat[i] <= 1.1]
            wide = [i for i in flex_idx if i not in short]
            left = remaining - sum(nat[i] for i in short)
            if wide and left >= 0.3 * total_in:
                for i in short:
                    widths[i] = nat[i]
                wide_nat = sum(nat[i] for i in wide)
                for i in wide:
                    widths[i] = nat[i] * left / wide_nat
            else:
                scale = remaining / flex_nat
                for i in flex_idx:
                    widths[i] = nat[i] * scale
    total = sum(widths)
    return [w * total_in / total for w in widths]


def make_table(doc, header_row: list[str], data_rows: list[list[str]],
                 col_widths_in: list[float] | None = None,
                 font_size_pt: int = BODY_SIZE - 1,
                 bold_first_col: bool | None = None) -> None:
    """Build a table in the house style (as in the DepAnx and BioData
    Mining papers): rules above and below the table and under the shaded
    header row, no vertical lines; first column left-aligned, numeric
    columns centered, text columns left; sub-rows written with leading
    spaces rendered indented; group labels in bold. The table fills the
    text width, with column widths derived from the content so numeric
    columns stay narrow and label columns get the room (``col_widths_in``
    is kept for API compatibility and ignored)."""
    centered = _numeric_columns(header_row, data_rows)
    if bold_first_col is None:
        bold_first_col = str(header_row[0]).strip().lower() == "clinical group"
    widths_in = _auto_col_widths(header_row, data_rows, centered, font_size_pt)

    table = doc.add_table(rows=1 + len(data_rows), cols=len(header_row))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _add_table_borders(table)
    _set_cell_margins(table)

    hdr = table.rows[0]
    _mark_header_row(hdr)
    for i, cell_text in enumerate(header_row):
        cell = hdr.cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _tighten_cell(cell, bold=True, text=cell_text,
                       font_size_pt=font_size_pt,
                       align="center" if centered[i] else "left",
                       line_spacing=1.0)
        _shade_cell(cell, HEADER_FILL)
        _cell_bottom_rule(cell)

    for r_idx, row_data in enumerate(data_rows, start=1):
        row = table.rows[r_idx]
        others_empty = all(not str(v).strip() for v in row_data[1:])
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            text = str(val)
            bold = (c_idx == 0 and bool(text.strip())
                    and (bold_first_col or others_empty)
                    and not text.startswith("  "))
            _tighten_cell(cell, bold=bold, text=text,
                           font_size_pt=font_size_pt,
                           align="center" if centered[c_idx] else "left")

    # Fill the text width: table width 100%, per-column widths as a
    # percentage of it (tcW in fiftieths of a percent) plus a matching
    # grid, so Word's autofit keeps the proportions.
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW"); tblPr.append(tblW)
    tblW.set(qn("w:type"), "pct"); tblW.set(qn("w:w"), "5000")
    total = sum(widths_in)
    tblGrid = tbl.find(qn("w:tblGrid"))
    if tblGrid is not None:
        for gi, gridCol in enumerate(tblGrid.findall(qn("w:gridCol"))):
            if gi < len(widths_in):
                gridCol.set(qn("w:w"), str(int(round(widths_in[gi] * 1440))))
    for col_idx, w in enumerate(widths_in):
        pct = int(round(5000 * w / total))
        for row in table.rows:
            tcPr = row.cells[col_idx]._tc.get_or_add_tcPr()
            tcW = tcPr.find(qn("w:tcW"))
            if tcW is None:
                tcW = OxmlElement("w:tcW"); tcPr.insert(0, tcW)
            tcW.set(qn("w:type"), "pct"); tcW.set(qn("w:w"), str(pct))


def _add_table_borders(table) -> None:
    """Rules above and below the table only (no vertical or inner lines)."""
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_borders = OxmlElement("w:tblBorders")
    for border in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{border}")
        if border in ("top", "bottom"):
            b.set(qn("w:val"), "single")
            b.set(qn("w:sz"), "4")
            b.set(qn("w:color"), "000000")
        else:
            b.set(qn("w:val"), "none")
            b.set(qn("w:sz"), "0")
            b.set(qn("w:color"), "auto")
        tbl_borders.append(b)
    tbl_pr.append(tbl_borders)


_WORDY = re.compile(r"[A-Za-z]{5,}")


def numeric_like(text: str) -> bool:
    """True for cells that read as numbers or short symbols ('7 (5–9)',
    '<0.001', 'ref.', 'n/a', '+0.83'), i.e. no run of 5+ letters."""
    return not _WORDY.search(text or "")


def _numeric_columns(header_row, data_rows) -> list[bool]:
    """Center a column when every non-empty data cell is numeric-like and
    at least one cell contains a digit; the first column is never
    centered (it carries the row labels)."""
    out = []
    for c in range(len(header_row)):
        cells = [str(r[c]) for r in data_rows if c < len(r) and str(r[c]).strip()]
        ok = (c > 0 and cells and all(numeric_like(x) for x in cells)
              and any(re.search(r"\d", x) for x in cells))
        out.append(bool(ok))
    return out


def _mark_header_row(row) -> None:
    """Repeat the header row when a table breaks across pages."""
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    trPr.append(el)


def _shade_cell(cell, fill_hex: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def _cell_bottom_rule(cell) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), "4")
    b.set(qn("w:color"), "000000")
    borders.append(b)
    tcPr.append(borders)


def page_break(doc) -> None:
    """Start a new page.

    Implemented as an empty paragraph flagged page-break-before, with a
    1 pt paragraph mark, rather than a paragraph holding a w:br page
    break: a break paragraph that itself lands at the top of a page
    renders as a blank page in Word and LibreOffice, whereas a
    page-break-before paragraph already at a page top adds nothing."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn as _qn
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.page_break_before = True
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    rpr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz"); sz.set(_qn("w:val"), "2"); rpr.append(sz)
    p._p.get_or_add_pPr().append(rpr)


def hrule(doc) -> None:
    """Visual divider \u2014 used to separate top-of-doc metadata from body."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:color"), "777777")
    pBdr.append(bottom)
    pPr.append(pBdr)
