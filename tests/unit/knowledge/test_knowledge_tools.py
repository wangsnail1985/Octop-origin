"""Unit tests for built-in knowledge-base LangChain tools."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.config import var_child_runnable_config

from octop.config import DatabaseConfig, OctopConfig
from octop.infra.knowledge.plane import postgres_database
from octop.infra.knowledge.tools import KnowledgeToolServices, build_knowledge_tools


@contextmanager
def _configurable(**kwargs: object):
    token = var_child_runnable_config.set({"configurable": kwargs})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _tool_by_name(tools: list, name: str):
    for tool in tools:
        if tool.name == name:
            return tool
    raise KeyError(name)


def test_tool_services_follow_control_plane_driver() -> None:
    """Search must see a driver swap; indexing already reads services.config."""
    manager = SimpleNamespace(
        _config=OctopConfig(database=DatabaseConfig(driver="sqlite")),
        _repos=SimpleNamespace(
            knowledge_repo=object(),
            settings_repo=object(),
            provider_repo=object(),
        ),
    )
    services = KnowledgeToolServices(manager)
    assert postgres_database(services) is None

    manager._config = OctopConfig(
        database=DatabaseConfig(
            driver="postgresql",
            host="db.example",
            database="octop",
            user="octop",
        )
    )
    assert postgres_database(services) is manager._config.database


def test_search_knowledge_default_description_is_empty_catalog_fallback() -> None:
    tool = _tool_by_name(build_knowledge_tools(SimpleNamespace()), "search_knowledge")
    assert tool.description == "No knowledge bases are attached this turn."


@pytest.mark.asyncio
async def test_search_knowledge_requires_selected_bases() -> None:
    services = SimpleNamespace(
        knowledge_repo=MagicMock(),
        settings_repo=MagicMock(),
        provider_repo=MagicMock(),
    )
    tool = _tool_by_name(build_knowledge_tools(services), "search_knowledge")
    with _configurable(user="1", knowledge_base_ids=[]):
        out = await tool.ainvoke({"query": "billing policy"})
    assert "No knowledge bases selected" in out


@pytest.mark.asyncio
async def test_search_knowledge_returns_retrieved_context() -> None:
    services = SimpleNamespace(
        knowledge_repo=MagicMock(),
        settings_repo=MagicMock(),
        provider_repo=MagicMock(),
    )
    tool = _tool_by_name(build_knowledge_tools(services), "search_knowledge")
    retrieved = AsyncMock(return_value="[KB: Docs]\nuseful passage")
    with (
        _configurable(
            user="7",
            user_is_admin=False,
            knowledge_base_ids=["kb-1"],
            locale="zh",
        ),
        patch("octop.infra.knowledge.tools.retrieve_context", retrieved),
    ):
        out = await tool.ainvoke({"query": "refund rules", "k": 3})

    assert out == "[KB: Docs]\nuseful passage"
    retrieved.assert_awaited_once()
    kwargs = retrieved.await_args.kwargs
    assert kwargs["user_id"] == 7
    assert kwargs["is_admin"] is False
    assert kwargs["query"] == "refund rules"
    assert kwargs["knowledge_base_ids"] == ["kb-1"]
    assert kwargs["k"] == 3
    assert kwargs["locale"] == "zh"


@pytest.mark.asyncio
async def test_search_knowledge_empty_hits_message() -> None:
    services = SimpleNamespace(
        knowledge_repo=MagicMock(),
        settings_repo=MagicMock(),
        provider_repo=MagicMock(),
    )
    tool = _tool_by_name(build_knowledge_tools(services), "search_knowledge")
    with (
        _configurable(user="1", knowledge_base_ids=["kb-1"]),
        patch(
            "octop.infra.knowledge.tools.retrieve_context",
            AsyncMock(return_value=""),
        ),
    ):
        out = await tool.ainvoke({"query": "nothing"})
    assert "No relevant passages found" in out


@pytest.mark.asyncio
async def test_count_occurrences_counts_exactly_with_anchors(tmp_path, monkeypatch) -> None:
    import json as jsonlib

    from octop.infra.db.migrate import run_migrations
    from octop.infra.db.pool import SqlitePool
    from octop.infra.db.repos.knowledge import KnowledgeRepo
    from octop.infra.db.repos.settings import SettingsRepo
    from octop.infra.db.repos.users import UserRepo
    from octop.infra.knowledge.page_index import SqlitePageIndex
    from octop.infra.knowledge.parse import Page
    from octop.infra.knowledge.tree import build_nodes

    home = tmp_path / "home"
    monkeypatch.setenv("OCTOP_HOME", str(home))
    pool = SqlitePool(tmp_path / "octop.db")
    run_migrations(pool)
    repo = KnowledgeRepo(pool)
    owner = UserRepo(pool).create(username="owner", password_hash="h", role="user")
    base = repo.create_base(owner_user_id=owner, name="Docs")
    document = repo.create_document(
        kb_id=base.id,
        filename="policy.md",
        content_type="text/markdown",
        byte_size=1,
        status="ready",
    )
    SqlitePageIndex(base.id).replace_doc_nodes(
        document.id,
        build_nodes(
            [
                Page(no=None, anchor="§6 报销", text="报销 一次 报销"),
                Page(no=None, anchor="§7 其他", text="无关内容 报销"),
            ],
            document.id,
        ),
    )
    services = SimpleNamespace(
        knowledge_repo=repo, settings_repo=SettingsRepo(pool), provider_repo=None
    )
    tool = _tool_by_name(build_knowledge_tools(services), "count_occurrences")

    with _configurable(user=str(owner), user_is_admin=False, knowledge_base_ids=[base.id]):
        out = await tool.ainvoke({"query": "报销"})

    payload = jsonlib.loads(out)
    assert payload["total"] == 3
    assert payload["documents"][0]["filename"] == "policy.md"
    assert payload["documents"][0]["total"] == 3
    sections = payload["documents"][0]["sections"]
    assert sections[0]["anchor"] == "§6 报销"
    assert sections[0]["count"] == 2
    assert sections[1]["count"] == 1
    assert "approximate" not in payload


@pytest.mark.asyncio
async def test_count_occurrences_returns_zero_for_missing_terms(tmp_path, monkeypatch) -> None:
    import json as jsonlib

    from octop.infra.db.migrate import run_migrations
    from octop.infra.db.pool import SqlitePool
    from octop.infra.db.repos.knowledge import KnowledgeRepo
    from octop.infra.db.repos.settings import SettingsRepo
    from octop.infra.db.repos.users import UserRepo
    from octop.infra.knowledge.index import KnowledgeIndex

    home = tmp_path / "home"
    monkeypatch.setenv("OCTOP_HOME", str(home))
    pool = SqlitePool(tmp_path / "octop.db")
    run_migrations(pool)
    repo = KnowledgeRepo(pool)
    owner = UserRepo(pool).create(username="owner", password_hash="h", role="user")
    base = repo.create_base(owner_user_id=owner, name="Docs")
    document = repo.create_document(
        kb_id=base.id,
        filename="plan.md",
        content_type="text/markdown",
        byte_size=1,
        status="ready",
    )
    # No tree: chunk fallback counting, flagged approximate.
    KnowledgeIndex(base.id).replace_doc_chunks(document.id, ["plain body"], [[1.0, 0.0]])
    services = SimpleNamespace(
        knowledge_repo=repo, settings_repo=SettingsRepo(pool), provider_repo=None
    )
    tool = _tool_by_name(build_knowledge_tools(services), "count_occurrences")

    with _configurable(user=str(owner), user_is_admin=False, knowledge_base_ids=[base.id]):
        out = await tool.ainvoke({"query": "不存在术语"})

    payload = jsonlib.loads(out)
    assert payload["total"] == 0
    assert payload["documents"] == []
