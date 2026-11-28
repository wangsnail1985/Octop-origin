"""Lexical (exact-term) scoring for the second retrieval path.

Pure functions only: storage drivers fetch candidate rows with a ``LIKE``
pre-filter and hand them to :func:`rank_rows`, which scores coverage of the
query terms and applies the out-of-domain rejection floor.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from octop.infra.knowledge.index import Hit

# kb-test measured values: 2-gram cap, coverage floor, anchor signal weight.
_MAX_TERMS = 8
_COVERAGE_FLOOR = 0.30
_ANCHOR_WEIGHT = 2.0

_ASCII_TERM_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_TERM_RE = re.compile(r"[\u3400-\u9fff]+")


def terms(query: str) -> list[str]:
    """Extract search terms: ASCII words plus CJK bigrams, capped at 8.

    Single ASCII letters are noise; digits and longer words stay. CJK runs
    become overlapping bigrams so exact-word recall does not depend on
    segmentation.
    """
    query = " ".join((query or "").split())
    if not query:
        return []
    found: list[str] = []
    for run in _CJK_TERM_RE.findall(query):
        found.extend(run[i : i + 2] for i in range(len(run) - 1))
    for word in _ASCII_TERM_RE.findall(query):
        lowered = word.lower()
        if len(lowered) >= 2 or lowered.isdigit():
            found.append(lowered)
    return list(dict.fromkeys(found))[:_MAX_TERMS]


def coverage(text: str, term_list: Sequence[str]) -> float:
    """Share of *term_list* occurring anywhere in *text* (case-insensitive)."""
    if not term_list:
        return 0.0
    folded = (text or "").casefold()
    hits = sum(1 for term in term_list if term in folded)
    return hits / len(term_list)


def escape_like(term: str) -> str:
    """Escape LIKE wildcards so terms match literally."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def like_patterns(term_list: Sequence[str]) -> list[str]:
    return [f"%{escape_like(term)}%" for term in term_list]


def rank_rows(
    rows: Sequence[tuple[str, str, int, str, int | None, str | None]],
    term_list: Sequence[str],
    *,
    k: int,
) -> list[Hit]:
    """Score ``(chunk_id, doc_id, ordinal, text, page, anchor)`` rows by coverage.

    Coverage runs over the text plus the anchor (a heading that names the query
    counts even when the window itself is mid-sentence). An anchor hit doubles
    the score and exempts the row from the coverage floor; rows whose text
    merely sprinkles a generic term stay out.
    """
    from octop.infra.knowledge.index import Hit

    if k <= 0 or not term_list:
        return []
    scored: list[Hit] = []
    for chunk_id, doc_id, ordinal, text, page, anchor in rows:
        anchor_folded = (anchor or "").casefold()
        anchor_hit = bool(anchor_folded) and any(term in anchor_folded for term in term_list)
        ratio = coverage(f"{text}\n{anchor_folded}", term_list)
        if ratio <= 0.0:
            continue
        if not anchor_hit and ratio < _COVERAGE_FLOOR and len(term_list) > 1:
            continue
        score = ratio * (_ANCHOR_WEIGHT if anchor_hit else 1.0)
        scored.append(
            Hit(
                chunk_id=str(chunk_id),
                doc_id=str(doc_id),
                ordinal=int(ordinal),
                text=str(text),
                score=float(score),
                metadata={},
                page=int(page) if page is not None else None,
                anchor=str(anchor) if anchor else None,
            )
        )
    scored.sort(key=lambda hit: (-hit.score, hit.doc_id, hit.ordinal))
    return scored[:k]
