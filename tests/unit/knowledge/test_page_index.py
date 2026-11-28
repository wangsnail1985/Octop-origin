"""Unit tests for page-index storage, keyword/LLM search, and settings."""

from __future__ import annotations

from types import SimpleNamespace

from octop.infra.knowledge.page_index import (
    SqlitePageIndex,
    get_pageindex_llm_capability,
    score_nodes,
    set_pageindex_llm_settings,
)
from octop.infra.knowledge.parse import Page
from octop.infra.knowledge.tree import build_nodes


def _store(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path))
    return SqlitePageIndex("kb-tree")


def _nodes() -> list:
    return build_nodes(
        [
            Page(no=None, anchor="¶0", text="公司差旅制度总览"),
            Page(no=None, anchor="§6 报销", text=""),
            Page(no=None, anchor="§6.1 报销流程", text="票据需要在三个工作日内提交给财务部门"),
            Page(no=None, anchor="§7 考勤", text="考勤相关内容"),
        ],
        "doc-1",
    )


def test_replace_search_and_delete_roundtrip(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, monkeypatch)

    store.replace_doc_nodes("doc-1", _nodes())
    assert store.doc_node_rows("doc-1") == _nodes()
    assert [node.anchor for node in store.all_nodes()] == [
        "¶0",
        "§6 报销",
        "§6.1 报销流程",
        "§7 考勤",
    ]

    hits = store.search_keyword("报销 流程 提交", k=2)
    assert hits
    assert hits[0].anchor == "§6.1 报销流程"
    assert "票据" in hits[0].content

    store.delete_doc("doc-1")
    assert store.all_nodes() == []


def test_keyword_search_prefers_title_hits(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, monkeypatch)
    store.replace_doc_nodes("doc-1", _nodes())

    hits = store.search_keyword("报销 考勤 制度", k=3)

    # The preamble mentions all three; the §7 heading names its topic directly.
    titles = [node.title for node in hits]
    assert "考勤" in titles


def test_search_llm_parses_node_ids_and_degrades_on_failure(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, monkeypatch)
    store.replace_doc_nodes("doc-1", _nodes())

    class _Reply:
        def __init__(self, text: str) -> None:
            self._text = text

        def invoke(self, _messages: object) -> str:
            return self._text

    target = store.search_llm("报销流程", k=2, chat_model=_Reply('["doc-1:n2"]'))
    assert target is not None
    assert [node.anchor for node in target] == ["§6.1 报销流程"]

    assert store.search_llm("报销流程", k=2, chat_model=_Reply("not json at all")) is None

    class _Boom:
        def invoke(self, _messages: object) -> str:
            raise RuntimeError("model down")

    assert store.search_llm("报销流程", k=2, chat_model=_Boom()) is None


def test_score_nodes_rejects_generic_partial_matches() -> None:
    nodes = build_nodes([Page(no=None, anchor="¶1", text="食堂菜单")], "doc-1")

    assert score_nodes(nodes, "报销 流程 提交 财务") == []
    assert score_nodes([], "报销") == []


def _fake_provider(models: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        api_key="key",
        base_url="https://example",
        name="p",
        get_models=lambda: [{"id": m, "name": m} for m in models],
    )


def test_pageindex_llm_capability_and_settings(tmp_path) -> None:
    values: dict[str, str] = {}
    provider_repo = SimpleNamespace(
        get=lambda _id: _fake_provider(["small-chat", "big-chat"]) if _id == 7 else None
    )

    capability = get_pageindex_llm_capability(values.get, provider_repo)
    assert capability["usable"] is False

    set_pageindex_llm_settings(
        values.__setitem__, provider_id="7", model="small-chat", provider_repo=provider_repo
    )
    capability = get_pageindex_llm_capability(values.get, provider_repo)
    assert capability == {
        "provider_id": "7",
        "model": "small-chat",
        "ready": True,
        "usable": True,
    }

    import pytest

    with pytest.raises(ValueError, match="not available"):
        set_pageindex_llm_settings(
            values.__setitem__,
            provider_id="7",
            model="missing-model",
            provider_repo=provider_repo,
        )
