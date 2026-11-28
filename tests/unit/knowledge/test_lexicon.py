"""Unit tests for the lexical (exact-term) retrieval path."""

from __future__ import annotations

from octop.infra.knowledge.index import KnowledgeIndex
from octop.infra.knowledge.lexicon import coverage, like_patterns, rank_rows, terms


def test_terms_mixes_cjk_bigrams_and_ascii_words() -> None:
    result = terms("报销 流程 GB18030 编码")

    assert "报销" in result
    assert "编码" in result
    assert "gb18030" in result
    assert all(len(term) >= 2 for term in result)


def test_terms_drops_single_ascii_letters_and_caps_at_eight() -> None:
    assert "a" not in terms("a b c")
    assert terms("Q3 报销") == ["报销", "q3"]
    long_query = " ".join(f"词{i}" for i in range(20))
    assert len(terms(long_query)) == 8
    assert terms("") == []


def test_coverage_counts_each_term_once() -> None:
    assert coverage("Apollo overview of Apollo", ["apollo", "budget"]) == 0.5
    assert coverage("", ["apollo"]) == 0.0
    assert coverage("text", []) == 0.0


def test_rank_rows_rejects_low_coverage_without_anchor_hit() -> None:
    rows = [("d:0", "d", 0, "Apollo overview", None, None)]

    # One of three terms hit: 0.33 stays; two of nine (0.22) is out of domain.
    kept = rank_rows(rows, ["apollo", "funding", "board"], k=5)
    dropped = rank_rows(rows, [*(["apollo"] * 1), *(["x"] * 7)], k=5)

    assert [hit.text for hit in kept] == ["Apollo overview"]
    assert dropped == []


def test_rank_rows_anchor_hit_is_a_strong_signal() -> None:
    rows = [
        ("d:0", "d", 0, "正文内容", None, "§6.1 报销流程"),
        ("d:1", "d", 1, "报销流程的一个完整说明", None, None),
    ]

    hits = rank_rows(rows, ["报销", "流程"], k=5)

    assert hits[0].anchor == "§6.1 报销流程"
    assert hits[0].score > hits[1].score
    assert hits[0].page is None


def test_rank_rows_single_term_matches_partial_text() -> None:
    rows = [("d:0", "d", 0, "GB18030 decode notes", None, None)]

    hits = rank_rows(rows, ["gb18030"], k=5)

    assert [hit.text for hit in hits] == ["GB18030 decode notes"]


def test_like_patterns_escape_wildcards() -> None:
    assert like_patterns(["100%"]) == ["%100\\%%"]
    assert like_patterns(["foo_bar"]) == ["%foo\\_bar%"]


def test_index_lexicon_search_matches_anchor_and_text(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path))
    index = KnowledgeIndex("kb-lex")
    index.replace_doc_chunks(
        "doc-1",
        ["报销需要在三个工作日内提交单据", "食堂菜单与报销无关", " unrelated english text"],
        [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9]],
        anchors=["§6.1 报销流程", None, None],
    )

    hits = index.lexicon_search("报销 流程 提交", k=5)

    assert hits
    assert hits[0].anchor == "§6.1 报销流程"
    assert "报销" in hits[0].text
    assert all("unrelated" not in hit.text for hit in hits)
    assert index.lexicon_search("报销", k=0) == []
