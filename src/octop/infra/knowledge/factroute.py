"""Detection of fact-style queries that should favor the page-index path.

Bilingual pattern table kept here (retrieval-internal rule, not user copy, so
it stays out of the JSON i18n bundles per the kb-1127 plan §5.4).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

BOOST_SETTING_KEY = "knowledge_fact_boost"

# zh: explicit page/section/slide references and counting questions.
_FACT_PATTERNS_ZH = (
    r"第\s*[一二三四五六七八九十百千零〇\d]+\s*[页章节条款步问答幕]",
    r"第\s*几\s*页",
    r"多少\s*次",
    r"几\s*次",
    r"出现.{0,6}次",
    r"编号",
    r"幻灯片",
    r"第\s*[一二三四五六七八九十百千零〇\d]+\s*条",
)
# en: page/slide/section/item references and counting questions.
_FACT_PATTERNS_EN = (
    r"\bpage\s+\d+\b",
    r"\bslide\s+\d+\b",
    r"\bsection\s+\d+(\.\d+)*\b",
    r"\bitem\s+\d+\b",
    r"\bchapter\s+\d+\b",
    r"\bhow\s+many\s+times\b",
    r"\bnumber\s+of\b",
    r"\boccurrences?\b",
)

_COMPILED = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in (*_FACT_PATTERNS_ZH, *_FACT_PATTERNS_EN)
)


def is_fact_query(query: str) -> bool:
    """True when the query names a position or asks for a count."""
    text = " ".join((query or "").split())
    if not text:
        return False
    return any(pattern.search(text) for pattern in _COMPILED)


def fact_boost_enabled(settings_get: Callable[[str], str | None]) -> bool:
    """Whether fact-style boosting is on (default: enabled)."""
    raw = settings_get(BOOST_SETTING_KEY)
    if raw is None or not str(raw).strip():
        return True
    try:
        decoded: object = json.loads(str(raw))
    except json.JSONDecodeError:
        decoded = str(raw)
    if isinstance(decoded, bool):
        return decoded
    return str(decoded).strip().lower() == "true"
