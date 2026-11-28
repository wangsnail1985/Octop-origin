"""Fixed-size, overlapping character chunking for knowledge documents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from octop.infra.knowledge.parse import Page


@dataclass(frozen=True)
class ChunkSlice:
    """One chunk window together with the anchor of the page unit it came from."""

    text: str
    page: int | None
    anchor: str | None


def _validate_window(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")


def _windows(content: str, *, size: int, overlap: int) -> list[str]:
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(content), step):
        chunks.append(content[start : start + size])
        if start + size >= len(content):
            break
    return chunks


def chunk_text(text: str, *, size: int = 800, overlap: int = 120) -> list[str]:
    """Split non-empty text into stable overlapping character windows."""
    _validate_window(size, overlap)
    content = text.strip()
    if not content:
        return []
    return _windows(content, size=size, overlap=overlap)


def chunk_pages(pages: Sequence[Page], *, size: int = 800, overlap: int = 120) -> list[ChunkSlice]:
    """Chunk each page unit separately, so no chunk spans two anchors.

    Oversized units are still split into overlapping windows, but every window
    keeps the unit's page number and anchor; fused retrieval then de-duplicates
    windows that share one anchor by their best rank.
    """
    _validate_window(size, overlap)
    chunks: list[ChunkSlice] = []
    for unit in pages:
        content = unit.text.strip()
        if not content:
            continue
        for window in _windows(content, size=size, overlap=overlap):
            chunks.append(ChunkSlice(text=window, page=unit.no, anchor=unit.anchor))
    return chunks
