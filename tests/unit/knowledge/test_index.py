"""Unit tests for per-knowledge-base sidecar indexes."""

from __future__ import annotations

import sqlite3
import struct

from octop.infra.knowledge.index import KnowledgeIndex


def test_index_replaces_document_chunks_and_returns_cosine_top_k(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path))
    index = KnowledgeIndex("kb-1")

    index.replace_doc_chunks(
        "doc-1",
        ["first", "second"],
        [[1.0, 0.0], [0.0, 1.0]],
        metadata=[{"page": 1}, {"page": 2}],
    )
    index.replace_doc_chunks("doc-2", ["third"], [[0.8, 0.2]])

    hits = index.search([1.0, 0.0], k=2)

    assert [hit.text for hit in hits] == ["first", "third"]
    assert hits[0].doc_id == "doc-1"
    assert hits[0].ordinal == 0
    assert hits[0].metadata == {"page": 1}

    index.replace_doc_chunks("doc-1", ["replacement"], [[0.0, 1.0]])
    assert [hit.text for hit in index.search([1.0, 0.0], k=5)] == ["third", "replacement"]

    index.delete_doc("doc-2")
    assert [hit.doc_id for hit in index.search([1.0, 0.0], k=5)] == ["doc-1"]
    assert index.list_doc_chunks("doc-1") == [(0, "replacement")]


def test_index_stores_and_returns_page_and_anchor(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path))
    index = KnowledgeIndex("kb-anchor")

    index.replace_doc_chunks(
        "doc-1",
        ["first", "second"],
        [[1.0, 0.0], [0.0, 1.0]],
        pages=[3, 4],
        anchors=["§1 概述", None],
    )

    hits = index.search([1.0, 0.0], k=1)
    assert hits[0].page == 3
    assert hits[0].anchor == "§1 概述"
    stored = index.read_doc_chunks("doc-1")
    assert [row[3:] for row in stored] == [(3, "§1 概述"), (4, None)]


def test_index_upgrades_legacy_sidecar_without_anchor_columns(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("OCTOP_HOME", str(home))
    legacy = home / "knowledge" / "kb-legacy" / "index.sqlite"
    legacy.parent.mkdir(parents=True)
    with sqlite3.connect(legacy) as conn:
        conn.executescript(
            """
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            "INSERT INTO chunks(chunk_id, doc_id, ordinal, text, embedding, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("doc-1:0", "doc-1", 0, "legacy text", struct.pack("<2f", 1.0, 0.0), "{}"),
        )

    index = KnowledgeIndex("kb-legacy")

    hits = index.search([1.0, 0.0], k=1)
    assert hits[0].text == "legacy text"
    assert hits[0].page is None
    assert hits[0].anchor is None
