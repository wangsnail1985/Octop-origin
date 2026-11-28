"""Unit tests for reciprocal-rank fusion of retrieval paths."""

from __future__ import annotations

from octop.infra.knowledge.fuse import dedupe_key, rrf_fuse
from octop.infra.knowledge.index import Hit


def _hit(name: str, *, doc: str = "doc", page: int | None = None, anchor: str | None = None) -> Hit:
    return Hit(
        chunk_id=f"{name}",
        doc_id=doc,
        ordinal=0,
        text=name,
        score=1.0,
        metadata={},
        page=page,
        anchor=anchor,
    )


def test_best_rank_only_no_vote_stacking() -> None:
    # The section appears as three mid-ranked vector windows while the
    # exact-term path ranks it first. Summed votes (1/63+1/64+1/65+1/61) would
    # outrank everything; with best-rank-only the section keeps its lexicon
    # representative and scores below the vector top-1.
    windows = [
        _hit("f1"),
        _hit("f2"),
        *(_hit(f"w{n}", anchor="§6.1 报销流程") for n in range(3)),
    ]
    strong = _hit("exact", anchor="§6.1 报销流程")

    fused = rrf_fuse({"vector": windows, "lexicon": [strong]})

    assert [hit.chunk_id for hit in fused] == ["f1", "exact", "f2"]


def test_keys_merge_across_engines() -> None:
    vector_hit = _hit("v1", doc="doc-a", page=12)
    lexicon_hit = _hit("l1", doc="doc-a", page=12)

    fused = rrf_fuse({"vector": [vector_hit], "lexicon": [lexicon_hit]})

    assert len(fused) == 1
    assert fused[0].chunk_id == "v1"  # same score: engine order wins


def test_ties_resolve_by_engine_then_rank() -> None:
    a, b, c = _hit("a"), _hit("b"), _hit("c")

    fused = rrf_fuse({"vector": [a], "lexicon": [b, c]})

    assert [hit.chunk_id for hit in fused] == ["a", "b", "c"]


def test_fallback_key_uses_chunk_id_without_anchor() -> None:
    legacy = _hit("doc-1:3", doc="doc-1")

    assert dedupe_key(legacy) == ("doc-1", "doc-1:3")
    assert dedupe_key(_hit("x", doc="d", page=7)) == ("d", "p7")
    assert dedupe_key(_hit("x", doc="d", anchor="§2")) == ("d", "§2")
    # A page number outranks a heading anchor for the same doc.
    assert dedupe_key(_hit("x", doc="d", page=3, anchor="§2")) == ("d", "p3")


def test_empty_inputs_return_empty() -> None:
    assert rrf_fuse({}) == []
    assert rrf_fuse({"vector": [], "lexicon": []}) == []
