"""Machine-readable knowledge citation payload embedded in tool results."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, TypedDict

CITATIONS_MARKER_PREFIX = "<!--octop-kb-citations:"
CITATIONS_MARKER_SUFFIX = "-->"


class _OptionalCitationFields(TypedDict, total=False):
    page: int
    anchor: str


class KnowledgeCitation(_OptionalCitationFields):
    kb_id: str
    kb_name: str
    doc_id: str
    filename: str
    path: str


def citations_from_ranked(ranked: Sequence[tuple[Any, Any, Any]]) -> list[KnowledgeCitation]:
    """Deduplicate ranked hits to one citation per document (retrieval order)."""
    seen: set[str] = set()
    out: list[KnowledgeCitation] = []
    for base, _hit, document in ranked:
        doc_id = str(getattr(document, "id", "") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        citation: KnowledgeCitation = {
            "kb_id": str(getattr(base, "id", "") or ""),
            "kb_name": str(getattr(base, "name", "") or ""),
            "doc_id": doc_id,
            "filename": str(getattr(document, "filename", "") or doc_id),
            "path": str(getattr(document, "path", "") or ""),
        }
        page = getattr(_hit, "page", None)
        anchor = getattr(_hit, "anchor", None)
        if isinstance(page, int):
            citation["page"] = page
        if anchor:
            citation["anchor"] = str(anchor)
        out.append(citation)
    return out


_CITATIONS_MARKER_RE = re.compile(
    r"\n*\s*"
    + re.escape(CITATIONS_MARKER_PREFIX)
    + r"\[.*?\]"
    + re.escape(CITATIONS_MARKER_SUFFIX)
    + r"\s*$",
    re.DOTALL,
)


def append_citations_marker(text: str, citations: Sequence[KnowledgeCitation]) -> str:
    """Append a HTML-comment marker the dashboard can parse; LLM usually ignores it."""
    body = (text or "").rstrip()
    if not body or not citations:
        return body
    payload = json.dumps(list(citations), ensure_ascii=False, separators=(",", ":"))
    return f"{body}\n\n{CITATIONS_MARKER_PREFIX}{payload}{CITATIONS_MARKER_SUFFIX}"


def strip_citations_marker(text: str) -> str:
    """Remove the dashboard citation marker from tool output (for model context)."""
    if not text or CITATIONS_MARKER_PREFIX not in text:
        return text
    return _CITATIONS_MARKER_RE.sub("", text).rstrip()
