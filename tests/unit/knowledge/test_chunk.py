"""Unit tests for document chunking."""

from __future__ import annotations

import pytest

from octop.infra.knowledge.chunk import chunk_pages, chunk_text
from octop.infra.knowledge.parse import Page


def test_chunk_text_preserves_content_with_configured_overlap() -> None:
    chunks = chunk_text("abcdefghij", size=4, overlap=1)

    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_text_strips_empty_content_and_rejects_invalid_window() -> None:
    assert chunk_text(" \n\t ") == []
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", size=4, overlap=4)


def test_chunk_pages_inherit_anchor_and_never_span_units() -> None:
    pages = [
        Page(no=None, anchor="¶0", text="first unit text"),
        Page(no=3, anchor=None, text="second unit text"),
    ]

    slices = chunk_pages(pages, size=15, overlap=3)

    assert slices
    assert not any(
        "first unit" in slice_.text and "second unit" in slice_.text for slice_ in slices
    )
    unit_one = [s for s in slices if "first unit" in s.text]
    unit_two = [s for s in slices if "second unit" in s.text]
    assert all((s.anchor, s.page) == ("¶0", None) for s in unit_one)
    assert all((s.anchor, s.page) == (None, 3) for s in unit_two)


def test_chunk_pages_long_unit_windows_share_one_anchor() -> None:
    slices = chunk_pages([Page(no=2, anchor=None, text="x" * 40)], size=15, overlap=3)

    assert len(slices) > 1
    assert all(slice_.page == 2 for slice_ in slices)
    assert "".join(window.text for window in slices).count("x") >= 40


def test_chunk_pages_skips_empty_units_and_validates_window() -> None:
    assert chunk_pages([Page(no=1, anchor=None, text="  ")]) == []
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages([Page(no=1, anchor=None, text="text")], size=4, overlap=4)
