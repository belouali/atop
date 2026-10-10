"""Shared formatting helpers for the JAMIA submission DOCX builders.

Used by ``build_revised_main_docx.py`` and ``build_revised_supplement_docx.py``.
Centralizes heading styles, table formatting, citation markers, and figure /
table placeholder paragraphs so both scripts produce visually consistent output.
"""
from __future__ import annotations

import os

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


def para(doc, text: str, *, italic: bool = False, bold: bool = False) -> None:
    """A normal-style paragraph; supports inline ** ** for bold runs."""
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = LINE_SPACING
    if "**" not in text:
        run = p.add_run(text)
        run.italic = italic
        run.bold = bold
        return
    # Split on ** for inline bold
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        run = p.add_run(part)
        run.italic = italic
        run.bold = bold or (i % 2 == 1)


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
    r = p.add_run(f"{label}. ")
    r.bold = True
    p.add_run(legend)

    if alt:
        p = doc.add_paragraph()
        r = p.add_run("Alt text: ")
        r.italic = True
        p.add_run(alt)


def figure_two_part(doc, label: str, legend: str, *,
                    portrait_path: str | None, landscape_path: str | None,
                    portrait_width_in: float = 6.5,
                    landscape_width_in: float = 9.4,
                    landscape_margin_in: float = 0.6) -> None:
    """Insert a figure whose panels span two pages: a portrait part on the
    current page, then a landscape part in its own landscape section with
    the caption under it. The page setup in force before the call is
    restored afterwards (a new portrait section, so the next exhibit starts
    on a fresh page without an extra page break). Missing images fall back
    to placeholder lines, as in ``figure_placeholder``.
    """
    from docx.enum.section import WD_ORIENT, WD_SECTION

    def _picture(path, width_in, what):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        if path and os.path.exists(path):
            p.add_run().add_picture(path, width=Inches(width_in))
        else:
            r = p.add_run(f"[FIGURE PLACEHOLDER: paste {label} ({what}) here]")
            r.italic = True

    prev = doc.sections[-1]
    saved = {k: getattr(prev, k) for k in (
        "page_width", "page_height", "orientation", "left_margin",
        "right_margin", "top_margin", "bottom_margin")}

    _picture(portrait_path, portrait_width_in, "first page")

    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    w, h = sec.page_width, sec.page_height
    if w < h:
        sec.page_width, sec.page_height = h, w
    sec.orientation = WD_ORIENT.LANDSCAPE
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(landscape_margin_in))

    _picture(landscape_path, landscape_width_in, "second page")
    # The caption shares the landscape page with the second part. At body
    # size and double spacing it would spill onto a third, landscape page,
    # so it is set compact (9 pt, single spaced), the size the merged
    # document gives every supplementary caption.
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(6)
    pf.space_after = Pt(0)
    r = p.add_run(f"{label}. ")
    r.bold = True
    r.font.size = Pt(9)
    r = p.add_run(legend)
    r.font.size = Pt(9)

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
    import re
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        is_bold = part.startswith("**") and part.endswith("**")
        body = part[2:-2] if is_bold else part
        r = paragraph.add_run(body)
        if is_bold:
            r.bold = True
        if italic:
            r.italic = True
        if font_size_pt is not None:
            r.font.size = Pt(font_size_pt)


def table_title(doc, label: str, short_title: str) -> None:
    """Render the table title paragraph ABOVE the table.

    Format: ``**{label}. {short_title}**`` (whole line bold, body size)
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(f"{label}. {short_title}")
    r.bold = True


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
        rl.italic = True
        rl.font.size = Pt(legend_size_pt)
        # body: parse inline **bold** markers
        _add_runs_with_inline_bold(p, " " + body_txt,
                                     italic=True,
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


def _tighten_cell(cell, *, bold: bool, text: str,
                    font_size_pt: int = BODY_SIZE - 1) -> None:
    """Render `text` inside `cell` with single line spacing and zero
    paragraph margins so the table renders compact (Word's default
    inherits the document's line_spacing = 1.5, which makes tables
    balloon vertically). Also sets a small inner cell margin so text
    doesn't crowd the borders."""
    p = cell.paragraphs[0]
    pf = p.paragraph_format
    pf.line_spacing = 1.0          # single-spaced inside cells
    pf.space_before = Pt(0)
    pf.space_after  = Pt(0)
    run = p.add_run(text)
    run.font.size = Pt(font_size_pt)
    if bold:
        run.bold = True


def _set_cell_margins(table, top_pt: int = 2, bottom_pt: int = 2,
                       left_pt: int = 4, right_pt: int = 4) -> None:
    """Apply tight inner cell margins on the whole table (in twentieths
    of a point: 1 pt = 20 units)."""
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    margins = OxmlElement("w:tblCellMar")
    for side, val in (("top", top_pt), ("bottom", bottom_pt),
                       ("left", left_pt), ("right", right_pt)):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), str(val * 20))
        m.set(qn("w:type"), "dxa")
        margins.append(m)
    tbl_pr.append(margins)


def make_table(doc, header_row: list[str], data_rows: list[list[str]],
                 col_widths_in: list[float] | None = None,
                 font_size_pt: int = BODY_SIZE - 1) -> None:
    """Build a table with bold header row, thin borders, fixed column
    widths, and a tight row design (single-spaced cells + small
    cell margins). `font_size_pt` sets the in-cell font size; the
    default (BODY_SIZE - 1 = 10pt) matches Tables 2-4; Table 1 uses
    9pt to fit the wider 5-column layout."""
    table = doc.add_table(rows=1 + len(data_rows), cols=len(header_row))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _add_table_borders(table)
    _set_cell_margins(table)

    hdr = table.rows[0]
    for i, cell_text in enumerate(header_row):
        cell = hdr.cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _tighten_cell(cell, bold=True, text=cell_text,
                       font_size_pt=font_size_pt)

    for r_idx, row_data in enumerate(data_rows, start=1):
        row = table.rows[r_idx]
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _tighten_cell(cell, bold=False, text=str(val),
                           font_size_pt=font_size_pt)

    if col_widths_in:
        # Disable autofit, switch to fixed layout, and update BOTH the
        # per-cell tcW values AND the tblGrid column-width definitions.
        # In Word's fixed-layout mode, tblGrid wins over autofit, but
        # tblGrid is created with default widths when the table is
        # added — must overwrite to match the requested col widths.
        table.autofit = False
        table.allow_autofit = False
        tbl = table._tbl
        tblPr = tbl.tblPr
        tblLayout = OxmlElement("w:tblLayout")
        tblLayout.set(qn("w:type"), "fixed")
        tblPr.append(tblLayout)
        # Update tblGrid columns (in dxa = twentieths of a point;
        # 1 inch = 1440 dxa)
        tblGrid = tbl.find(qn("w:tblGrid"))
        if tblGrid is not None:
            gridCols = tblGrid.findall(qn("w:gridCol"))
            for gi, gridCol in enumerate(gridCols):
                if gi < len(col_widths_in):
                    dxa = int(round(col_widths_in[gi] * 1440))
                    gridCol.set(qn("w:w"), str(dxa))
        # Also update per-cell widths (belt-and-suspenders).
        for col_idx, width_in in enumerate(col_widths_in):
            for row in table.rows:
                row.cells[col_idx].width = Inches(width_in)


def _add_table_borders(table) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_borders = OxmlElement("w:tblBorders")
    for border in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{border}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), "888888")
        tbl_borders.append(b)
    tbl_pr.append(tbl_borders)


def page_break(doc) -> None:
    doc.add_page_break()


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
