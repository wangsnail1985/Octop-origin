"""Text extraction for the document types accepted by knowledge bases.

Two parallel entry points share the format detection and decoding helpers:

- :func:`parse_document` returns the raw extracted text (previews, legacy paths);
- :func:`parse_document_pages` returns anchor-bearing :class:`Page` units that
  indexing turns into anchored chunks (``p.12`` / ``§2.1`` / ``¶3`` citations).
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from octop.infra.knowledge.ocr import OCR_IMAGE_SUFFIXES

if TYPE_CHECKING:
    from octop.infra.knowledge.ocr import OcrExtractor

_PLAIN_TEXT_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".yaml",
    ".yml",
    ".jsonl",
}
_DOCX_FALLBACK_TAG = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
_DOCX_HTML_TYPES = {"application/xhtml+xml", "text/html"}
_DOCX_MAIN_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Spreadsheet row blocks: split a sheet so each block lands in one chunk.
_SHEET_BLOCK_CHARS = 350


@dataclass(frozen=True)
class Page:
    """One addressable unit of an extracted document.

    ``no`` is a physical page/slide number (PDF, PPTX) and stays ``None`` for
    formats without pages; ``anchor`` locates the unit inside the document
    (``§2.1 报销流程`` for heading units, ``¶N`` for plain paragraphs). A unit
    with both ``None`` (whole-file OCR fallback) is addressed by chunk order.
    """

    no: int | None
    anchor: str | None
    text: str


def pages_to_text(pages: Sequence[Page]) -> str:
    """Join page units back into one plain-text blob."""
    return "\n".join(unit.text for unit in pages)


# -- anchor helpers -----------------------------------------------------------

# A leading numbering such as ``2.1`` / ``3.`` / ``4、`` on a heading line.
_NUMBERED_TITLE_RE = re.compile(r"^(\d+(?:\.\d+)*)[.、]?\s+(.*)$")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HEADING_STYLE_RE = re.compile(r"^(?:heading|标题)\s*(\d+)$", re.IGNORECASE)
_TOC_STYLE_RE = re.compile(r"^(?:toc|目录)", re.IGNORECASE)


def _section_anchor(title: str) -> str:
    """Anchor for a heading unit: numbered when possible, title-only otherwise."""
    text = " ".join(title.split())
    if not text:
        return "§"
    match = _NUMBERED_TITLE_RE.match(text)
    if match and match.group(2).strip():
        return f"§{match.group(1)} {match.group(2).strip()}"
    if match:
        return f"§{match.group(1)}"
    return f"§ {text}"


def _paragraph_pages(text: str, *, first_no: int = 1) -> list[Page]:
    """Group a plain-text document into blank-line-separated paragraph units."""
    groups = (group.strip() for group in re.split(r"\n[ \t]*\n", text))
    return [
        Page(no=None, anchor=f"¶{index}", text=group)
        for index, group in enumerate((group for group in groups if group), start=first_no)
    ]


# -- pages-per-format ---------------------------------------------------------


def _pdf_pages(path: Path, *, ocr: OcrExtractor | None) -> list[Page]:
    from pypdf import PdfReader

    pages = [
        Page(no=number, anchor=None, text=page.extract_text() or "")
        for number, page in enumerate(PdfReader(path).pages, start=1)
    ]
    if not any(unit.text.strip() for unit in pages) and ocr is not None:
        return [Page(no=None, anchor=None, text=ocr(path))]
    return [unit for unit in pages if unit.text.strip()]


def _pptx_pages(path: Path) -> list[Page]:
    from pptx import Presentation

    pages: list[Page] = []
    for number, slide in enumerate(Presentation(str(path)).slides, start=1):
        parts = [shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text]
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text
            if notes:
                parts.append(notes)
        text = "\n".join(parts).strip()
        if text:
            pages.append(Page(no=number, anchor=None, text=text))
    return pages


def _markdown_pages(text: str) -> list[Page]:
    """Split on ATX headings; content before the first heading becomes ``¶0``."""
    pre: list[str] = []
    units: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        match = _MD_HEADING_RE.match(line)
        if match:
            units.append((_section_anchor(match.group(2).strip()), [line]))
        elif units:
            units[-1][1].append(line)
        else:
            pre.append(line)
    if not units:
        return [Page(no=None, anchor="¶1", text=text)] if text.strip() else []
    pages: list[Page] = []
    pre_text = "\n".join(pre).strip()
    if pre_text:
        # ¶0 keeps the preamble from colliding with the first heading's number.
        pages.append(Page(no=None, anchor="¶0", text=pre_text))
    pages.extend(
        Page(no=None, anchor=anchor, text="\n".join(lines).strip())
        for anchor, lines in units
        if "\n".join(lines).strip()
    )
    return pages


def _html_pages(raw: str) -> list[Page]:
    units = _html_units(raw)
    pre: list[str] = []
    pages: list[Page] = []
    for title, text in units:
        body = text.strip()
        if not body:
            continue
        if title is None:
            if pages:
                last = pages[-1]
                pages[-1] = Page(last.no, last.anchor, f"{last.text}\n{body}")
            else:
                pre.append(body)
            continue
        pages.append(Page(no=None, anchor=_section_anchor(title), text=f"{title}\n{body}"))
    if pre:
        pages.insert(0, Page(no=None, anchor="¶0", text="\n".join(pre).strip()))
    if not pages and raw.strip():
        return [Page(no=None, anchor="¶1", text=raw)]
    return pages


def _json_pages(raw: str) -> list[Page]:
    """One unit per top-level key, so array/child blocks stay addressable."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _paragraph_pages(raw)
    if not isinstance(parsed, dict):
        return [Page(no=None, anchor="¶1", text=json.dumps(parsed, ensure_ascii=False, indent=2))]
    pages: list[Page] = []
    for key, value in parsed.items():
        block = json.dumps({key: value}, ensure_ascii=False, indent=2)
        inner = "\n".join(block.splitlines()[1:-1]).strip()
        if inner:
            pages.append(Page(no=None, anchor=f"¶{len(pages) + 1}", text=inner))
    return pages or [Page(no=None, anchor="¶1", text="{}")]


def _docx_style_map(document: Any) -> dict[str, str]:
    """Map paragraph style ids to style names for heading detection."""
    styles: dict[str, str] = {}
    for style in document.styles:
        style_id = getattr(style, "style_id", None)
        name = getattr(style, "name", None)
        if style_id and name:
            styles[str(style_id)] = str(name)
    return styles


def _docx_paragraph_style(element: Any) -> str:
    from docx.oxml.ns import qn

    p_pr = element.find(qn("w:pPr"))
    if p_pr is None:
        return ""
    p_style = p_pr.find(qn("w:pStyle"))
    if p_style is None:
        return ""
    return str(p_style.get(qn("w:val")) or "")


def _docx_pages(body: Any, part: Any, styles: dict[str, str]) -> list[Page]:
    """Collect ``w:p``/``w:altChunk`` in document order, split on heading styles."""
    from docx.oxml.ns import qn

    alt_chunk_tag = qn("w:altChunk")
    pre: list[str] = []
    units: list[tuple[str, list[str]]] = []
    for element in body.iter(qn("w:p"), alt_chunk_tag):
        if next(element.iterancestors(_DOCX_FALLBACK_TAG), None) is not None:
            continue
        if element.tag == alt_chunk_tag:
            alt_pages = _docx_alt_chunk_pages(element, part)
            if not alt_pages:
                continue
            if units:
                units[-1][1].extend(unit.text for unit in alt_pages)
            else:
                pre.extend(unit.text for unit in alt_pages)
            continue
        text = str(element.text or "")
        style_name = styles.get(_docx_paragraph_style(element), "")
        if _TOC_STYLE_RE.match(style_name):
            # TOC entries duplicate the headings they point at; skip them.
            continue
        if _HEADING_STYLE_RE.match(style_name):
            if text.strip():
                units.append((_section_anchor(text.strip()), [text.strip()]))
            continue
        if units:
            units[-1][1].append(text)
        else:
            pre.append(text)
    pages: list[Page] = []
    pre_text = "\n".join(pre).strip()
    if units:
        if pre_text:
            pages.append(Page(no=None, anchor="¶0", text=pre_text))
        pages.extend(
            Page(no=None, anchor=anchor, text="\n".join(lines).strip())
            for anchor, lines in units
            if "\n".join(lines).strip()
        )
    elif pre_text:
        pages = _paragraph_pages(pre_text)
    return pages


def _docx_alt_chunk_pages(element: Any, part: Any) -> list[Page]:
    from docx.oxml.ns import qn

    chunk = part.related_parts.get(element.get(qn("r:id")) or "")
    if chunk is None:
        return []
    content_type = str(chunk.content_type)
    blob: bytes = chunk.blob
    if content_type in _DOCX_HTML_TYPES:
        return _html_pages(blob.decode("utf-8-sig", errors="replace"))
    if content_type == "text/plain":
        return _paragraph_pages(blob.decode("utf-8-sig", errors="replace"))
    if content_type == _DOCX_MAIN_TYPE:
        from docx import Document

        nested = Document(io.BytesIO(blob))
        return _docx_pages(nested.element.body, nested.part, _docx_style_map(nested))
    return []


def _sheet_row_pages(title: str, rows: Iterable[tuple[int, list[object]]]) -> list[Page]:
    """Split one sheet into ~350-character row blocks, anchored by row range."""
    blocks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    used = 0
    for row_no, row in rows:
        cells = _trim_row([_stringify_cell(cell) for cell in row])
        if not cells:
            continue
        line = "\t".join(cells)
        if current and used + len(line) > _SHEET_BLOCK_CHARS:
            blocks.append(current)
            current = []
            used = 0
        current.append((row_no, line))
        used += len(line) + 1
    if current:
        blocks.append(current)
    pages: list[Page] = []
    for number, block in enumerate(blocks, start=1):
        first, last = block[0][0], block[-1][0]
        anchor = f"§{title} 第{first}-{last}行" if first != last else f"§{title} 第{first}行"
        lines = [line for _, line in block]
        text = "\n".join((f"# {title}", *lines) if number == 1 else lines)
        pages.append(Page(no=None, anchor=anchor, text=text))
    return pages


def _xlsx_row_rows(sheet: Any) -> list[tuple[int, list[object]]]:
    return [
        (row_no, list(row)) for row_no, row in enumerate(sheet.iter_rows(values_only=True), start=1)
    ]


def parse_document_pages(path: Path, *, ocr: OcrExtractor | None = None) -> list[Page]:
    """Extract anchor-bearing page units from a supported local document."""
    suffix = path.suffix.lower()
    if suffix in OCR_IMAGE_SUFFIXES:
        if ocr is None:
            raise RuntimeError("knowledge OCR is not enabled")
        return [Page(no=None, anchor="¶1", text=ocr(path))]
    if suffix in _PLAIN_TEXT_SUFFIXES:
        text = _read_text(path)
        if suffix in {".md", ".markdown"}:
            return _markdown_pages(text)
        return _paragraph_pages(text)
    if suffix == ".json":
        return _json_pages(_read_text(path))
    if suffix in {".html", ".htm"}:
        return _html_pages(_read_text(path))
    if suffix == ".csv":
        return _sheet_row_pages(
            path.stem,
            [
                (row_no, list(row))
                for row_no, row in enumerate(
                    csv.reader(_read_text(path).splitlines(), delimiter=","), start=1
                )
            ],
        )
    if suffix == ".tsv":
        return _sheet_row_pages(
            path.stem,
            [
                (row_no, list(row))
                for row_no, row in enumerate(
                    csv.reader(_read_text(path).splitlines(), delimiter="\t"), start=1
                )
            ],
        )
    if suffix == ".pdf":
        return _pdf_pages(path, ocr=ocr)
    if suffix == ".docx":
        from docx import Document

        document = Document(str(path))
        return _docx_pages(document.element.body, document.part, _docx_style_map(document))
    if suffix in {".wps", ".doc"}:
        from octop.infra.utils.anydoc_convert import extract_markdown_file

        text = extract_markdown_file(path)
        if not text.strip():
            raise ValueError(f"failed to extract text from {suffix} document")
        return _markdown_pages(text)
    if suffix == ".pptx":
        return _pptx_pages(path)
    if suffix in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            pages: list[Page] = []
            for sheet in workbook.worksheets:
                pages.extend(_sheet_row_pages(str(sheet.title), _xlsx_row_rows(sheet)))
            return pages
        finally:
            workbook.close()
    if suffix == ".xls":
        return _xls_pages(path)
    raise ValueError(f"unsupported knowledge document extension: {suffix or '(none)'}")


def _xls_pages(path: Path) -> list[Page]:
    import xlrd

    book = xlrd.open_workbook(str(path), formatting_info=False)
    pages: list[Page] = []
    for sheet in book.sheets():
        rows: list[tuple[int, list[object]]] = []
        for row_idx in range(sheet.nrows):
            rows.append(
                (
                    row_idx + 1,
                    [
                        _xlrd_cell_value(book, sheet.cell(row_idx, col_idx))
                        for col_idx in range(sheet.ncols)
                    ],
                )
            )
        pages.extend(_sheet_row_pages(sheet.name, rows))
    return pages


# -- raw-text extraction (legacy preview path) --------------------------------


def parse_document(path: Path, *, ocr: OcrExtractor | None = None) -> str:
    """Extract searchable text from a supported local document."""
    suffix = path.suffix.lower()
    if suffix in OCR_IMAGE_SUFFIXES:
        if ocr is None:
            raise RuntimeError("knowledge OCR is not enabled")
        return ocr(path)
    if suffix in _PLAIN_TEXT_SUFFIXES:
        return _read_text(path)
    if suffix == ".json":
        return _parse_json(path)
    if suffix in {".html", ".htm"}:
        return _parse_html(path)
    if suffix == ".csv":
        return _parse_delimited(path, delimiter=",")
    if suffix == ".tsv":
        return _parse_delimited(path, delimiter="\t")
    if suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        if text.strip() or ocr is None:
            return text
        return ocr(path)
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix in {".wps", ".doc"}:
        from octop.infra.utils.anydoc_convert import extract_markdown_file

        text = extract_markdown_file(path)
        if not text.strip():
            raise ValueError(f"failed to extract text from {suffix} document")
        return text
    if suffix == ".pptx":
        from pptx import Presentation

        return "\n".join(
            shape.text
            for slide in Presentation(str(path)).slides
            for shape in slide.shapes
            if hasattr(shape, "text") and shape.text
        )
    if suffix in {".xlsx", ".xlsm"}:
        return _parse_xlsx(path)
    if suffix == ".xls":
        return _parse_xls(path)
    raise ValueError(f"unsupported knowledge document extension: {suffix or '(none)'}")


_TEXT_ENCODINGS = ("utf-8-sig", "gb18030")


def _read_text(path: Path) -> str:
    """Decode a knowledge text file, preferring UTF-8 then GB18030 (GBK).

    ``errors='replace'`` on UTF-8 used to swallow GBK Windows documents as
    mojibake, so retrieval never saw the original Chinese text.
    """
    data = path.read_bytes()
    for encoding in _TEXT_ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        else:
            return text.replace("\r\n", "\n").replace("\r", "\n")
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def _parse_json(path: Path) -> str:
    raw = _read_text(path)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _parse_docx(path: Path) -> str:
    from docx import Document

    document = Document(str(path))
    return _docx_body_text(document.element.body, document.part)


def _docx_body_text(body: Any, part: Any) -> str:
    """Collect every ``w:p`` and ``w:altChunk`` in a body, in document order.

    ``Document.paragraphs`` only lists ``w:p`` children of ``w:body``, so text inside
    tables, content controls (``w:sdt``), revision wrappers (``w:ins``), and text boxes
    is silently dropped. ``mc:Fallback`` duplicates its ``mc:Choice`` sibling, so its
    paragraphs are skipped.
    """
    from docx.oxml.ns import qn

    alt_chunk_tag = qn("w:altChunk")
    lines: list[str] = []
    for element in body.iter(qn("w:p"), alt_chunk_tag):
        if next(element.iterancestors(_DOCX_FALLBACK_TAG), None) is not None:
            continue
        if element.tag == alt_chunk_tag:
            lines.append(_docx_alt_chunk_text(element, part))
        else:
            lines.append(element.text)
    return "\n".join(lines)


def _docx_alt_chunk_text(element: Any, part: Any) -> str:
    """Expand a ``w:altChunk`` reference the way Word does when opening the file.

    Converters keep the bulk of the text in an embedded HTML or Word part and leave
    only a stub in ``document.xml``; unexpanded, that body is lost.
    """
    from docx.oxml.ns import qn

    chunk = part.related_parts.get(element.get(qn("r:id")) or "")
    if chunk is None:
        return ""
    content_type = str(chunk.content_type)
    blob: bytes = chunk.blob
    if content_type in _DOCX_HTML_TYPES:
        return _html_text(blob.decode("utf-8-sig", errors="replace"))
    if content_type == "text/plain":
        return blob.decode("utf-8-sig", errors="replace")
    if content_type == _DOCX_MAIN_TYPE:
        from docx import Document

        nested = Document(io.BytesIO(blob))
        return _docx_body_text(nested.element.body, nested.part)
    return ""


def _parse_html(path: Path) -> str:
    return _html_text(_read_text(path))


def _html_text(raw: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(raw)
    parser.close()
    text = parser.text().replace(_HTMLTextParser._SENTINEL, "\n")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


class _HTMLTextParser(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript", "template"})
    _BLOCK = frozenset(
        {
            "p",
            "div",
            "br",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "article",
            "section",
            "header",
            "footer",
            "blockquote",
            "pre",
            "table",
            "ul",
            "ol",
        }
    )
    _HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
    _SENTINEL = "\x00heading\x00"

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip += 1
            return
        if not self._skip and tag in self._HEADING_TAGS:
            self._parts.append(self._SENTINEL)
            return
        if not self._skip and tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            if self._skip:
                self._skip -= 1
            return
        if not self._skip and tag in self._HEADING_TAGS:
            self._parts.append(self._SENTINEL)
            return
        if not self._skip and tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip and data:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _html_units(raw: str) -> list[tuple[str | None, str]]:
    """Split HTML text into ``(heading_title | None, body_text)`` units.

    Each ``h1``-``h6`` element contributes a sentinel pair, so segments at odd
    positions in the split are heading titles and even positions are body text.
    """
    parser = _HTMLTextParser()
    parser.feed(raw)
    parser.close()
    flow = parser.text().split(_HTMLTextParser._SENTINEL)
    units: list[tuple[str | None, str]] = []
    for index, segment in enumerate(flow):
        lines = [line for line in (" ".join(part.split()) for part in segment.splitlines()) if line]
        if not lines:
            continue
        if index % 2 == 1:
            title = " ".join("".join(lines).split())
            units.append((title or None, title))
        else:
            units.append((None, "\n".join(lines)))
    return units


def _stringify_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def _trim_row(cells: list[str]) -> list[str]:
    while cells and cells[-1] == "":
        cells.pop()
    return cells


def _sheet_text(title: str, rows: Iterable[Iterable[object]]) -> str:
    lines = [f"# {title}"]
    has_cells = False
    for raw in rows:
        cells = _trim_row([_stringify_cell(cell) for cell in raw])
        if not cells:
            continue
        has_cells = True
        lines.append("\t".join(cells))
    if not has_cells:
        return ""
    return "\n".join(lines)


def _parse_delimited(path: Path, *, delimiter: str) -> str:
    reader = csv.reader(_read_text(path).splitlines(), delimiter=delimiter)
    return _sheet_text(path.stem, reader)


def _parse_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        parts: list[str] = []
        for sheet in workbook.worksheets:
            text = _sheet_text(str(sheet.title), sheet.iter_rows(values_only=True))
            if text:
                parts.append(text)
        return "\n\n".join(parts)
    finally:
        workbook.close()


def _parse_xls(path: Path) -> str:
    import xlrd

    book = xlrd.open_workbook(str(path), formatting_info=False)
    parts: list[str] = []
    for sheet in book.sheets():
        rows: list[list[object]] = []
        for row_idx in range(sheet.nrows):
            rows.append(
                [
                    _xlrd_cell_value(book, sheet.cell(row_idx, col_idx))
                    for col_idx in range(sheet.ncols)
                ]
            )
        text = _sheet_text(sheet.name, rows)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _xlrd_cell_value(book: object, cell: object) -> object:
    import xlrd

    ctype = getattr(cell, "ctype", xlrd.XL_CELL_EMPTY)
    value = getattr(cell, "value", None)
    if ctype in {xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK, xlrd.XL_CELL_ERROR}:
        return None
    if ctype == xlrd.XL_CELL_DATE:
        datemode = int(getattr(book, "datemode", 0))
        try:
            return xlrd.xldate_as_datetime(value, datemode).isoformat(sep=" ", timespec="seconds")
        except (OSError, OverflowError, TypeError, ValueError):
            return value
    return value
