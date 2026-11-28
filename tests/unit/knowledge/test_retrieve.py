"""Unit tests for knowledge retrieval during a chat turn."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from octop.infra.db.migrate import run_migrations
from octop.infra.db.pool import SqlitePool
from octop.infra.db.repos.knowledge import KnowledgeRepo
from octop.infra.db.repos.settings import SettingsRepo
from octop.infra.db.repos.users import UserRepo
from octop.infra.knowledge import retrieve as retrieve_module
from octop.infra.knowledge.citations import CITATIONS_MARKER_PREFIX
from octop.infra.knowledge.graph_store import (
    ChunkEntity,
    ChunkGraph,
    ChunkRelation,
    SqliteKnowledgeGraphStore,
)
from octop.infra.knowledge.index import Hit, KnowledgeIndex


def test_citation_text_prefers_page_then_anchor_then_chunk_order() -> None:
    base = SimpleNamespace(name="KB")
    document = SimpleNamespace(filename="a.md")
    with_page = Hit("c1", "a", 0, "t", 1.0, {}, page=12)
    with_anchor = Hit("c2", "a", 1, "t", 1.0, {}, anchor="§2.1 报销流程")
    legacy = Hit("c3", "a", 2, "t", 1.0, {})

    assert "第12页" in retrieve_module._citation_text(base, with_page, document, locale="zh")
    assert "§2.1 报销流程" in retrieve_module._citation_text(
        base, with_anchor, document, locale="zh"
    )
    assert "第3段" in retrieve_module._citation_text(base, legacy, document, locale="zh")
    assert "p.12" in retrieve_module._citation_text(base, with_page, document, locale="en")


def test_retrieve_context_filters_unreadable_knowledge_base(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path / "home"))
    pool = SqlitePool(tmp_path / "octop.db")
    run_migrations(pool)
    services = SimpleNamespace(
        knowledge_repo=KnowledgeRepo(pool),
        settings_repo=SettingsRepo(pool),
        user_repo=UserRepo(pool),
    )
    reader = services.user_repo.create(username="reader", password_hash="h", role="user")
    owner = services.user_repo.create(username="owner", password_hash="h", role="user")
    allowed = services.knowledge_repo.create_base(owner_user_id=reader, name="Allowed")
    hidden = services.knowledge_repo.create_base(owner_user_id=owner, name="Hidden")
    allowed_doc = services.knowledge_repo.create_document(
        kb_id=allowed.id,
        filename="allowed.md",
        content_type="text/markdown",
        byte_size=1,
        status="ready",
    )
    hidden_doc = services.knowledge_repo.create_document(
        kb_id=hidden.id,
        filename="hidden.md",
        content_type="text/markdown",
        byte_size=1,
        status="ready",
    )
    KnowledgeIndex(allowed.id).replace_doc_chunks(allowed_doc.id, ["allowed fact"], [[1.0, 0.0]])
    KnowledgeIndex(hidden.id).replace_doc_chunks(hidden_doc.id, ["secret fact"], [[1.0, 0.0]])
    services.settings_repo.set("knowledge_embedding_model", "test-model")
    monkeypatch.setattr(retrieve_module, "assert_knowledge_usable", lambda *_args: None)
    monkeypatch.setattr(
        retrieve_module, "embed_knowledge_texts", lambda _services, _texts: [[1.0, 0.0]]
    )

    context = asyncio.run(
        retrieve_module.retrieve_context(
            services,
            user_id=reader,
            is_admin=False,
            query="What facts are available?",
            knowledge_base_ids=[allowed.id, hidden.id],
        )
    )

    assert "allowed fact" in context
    assert "allowed.md" in context
    assert CITATIONS_MARKER_PREFIX in context
    assert "secret fact" not in context
    assert "hidden.md" not in context


def test_retrieve_context_skips_empty_non_text_turn() -> None:
    context = asyncio.run(
        retrieve_module.retrieve_context(
            SimpleNamespace(),
            user_id=1,
            is_admin=False,
            query=None,  # type: ignore[arg-type]
            knowledge_base_ids=["kb-1"],
        )
    )

    assert context == ""


def test_format_context_cites_only_documents_inside_the_budget() -> None:
    base = SimpleNamespace(id="kb", name="KB")
    doc_a = SimpleNamespace(id="a", filename="a.md", path="a.md")
    doc_b = SimpleNamespace(id="b", filename="b.md", path="b.md")
    hit_a = SimpleNamespace(text="alpha " * 40, ordinal=0, page=None, anchor=None)
    hit_b = SimpleNamespace(text="beta " * 40, ordinal=0, page=None, anchor=None)

    text = retrieve_module._format_context(
        [(base, hit_a, doc_a), (base, hit_b, doc_b)],
        char_budget=80,
        locale="en",
    )

    marker = text.split(CITATIONS_MARKER_PREFIX, 1)[1]
    assert "a.md" in marker
    assert "b.md" not in marker


def _graph_services(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path / "home"))
    pool = SqlitePool(tmp_path / "octop.db")
    run_migrations(pool)
    services = SimpleNamespace(
        knowledge_repo=KnowledgeRepo(pool),
        settings_repo=SettingsRepo(pool),
        user_repo=UserRepo(pool),
    )
    owner = services.user_repo.create(username="owner", password_hash="h", role="user")
    base = services.knowledge_repo.create_base(owner_user_id=owner, name="Docs")
    document = services.knowledge_repo.create_document(
        kb_id=base.id,
        filename="plan.md",
        content_type="text/markdown",
        byte_size=1,
        status="ready",
    )
    # Four passages rank by similarity; the fifth is unrelated by wording.
    texts = ["Apollo overview", "Apollo timeline", "Apollo risks", "Apollo team", "Budget rules"]
    vectors = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.0, 1.0]]
    KnowledgeIndex(base.id).replace_doc_chunks(document.id, texts, vectors)
    SqliteKnowledgeGraphStore(base.id).replace_document(
        document.id,
        [
            ChunkGraph(
                ordinal=4,
                entities=(ChunkEntity("Apollo"), ChunkEntity("Finance Board")),
                relations=(
                    ChunkRelation(
                        "Apollo", "Finance Board", "is funded by", "The board signs off.", 9
                    ),
                ),
            )
        ],
    )
    monkeypatch.setattr(retrieve_module, "assert_knowledge_usable", lambda *_args: None)
    monkeypatch.setattr(
        retrieve_module, "embed_knowledge_texts", lambda _services, _texts: [[1.0, 0.0]]
    )
    return services, owner, base


def _retrieve(services, owner, base, **kwargs):  # type: ignore[no-untyped-def]
    return asyncio.run(
        retrieve_module.retrieve_context(
            services,
            user_id=owner,
            is_admin=False,
            query="Who funds Apollo?",
            knowledge_base_ids=[base.id],
            **kwargs,
        )
    )


def test_retrieve_context_adds_graph_passage_and_relations(tmp_path, monkeypatch) -> None:
    services, owner, base = _graph_services(tmp_path, monkeypatch)
    services.settings_repo.set("knowledge_graph_enabled", "true")

    context = _retrieve(services, owner, base, k=4)

    # The linked passage takes the last of four slots; the weakest vector hit gives way.
    assert "Budget rules" in context
    assert "Apollo team" not in context
    assert context.index("Apollo overview") < context.index("Budget rules")
    assert "- Apollo -> is funded by -> Finance Board: The board signs off." in context
    assert context.index("Budget rules") < context.index("- Apollo -> is funded by")
    assert context.rstrip().endswith("-->")


def test_retrieve_context_ignores_the_graph_when_the_feature_is_off(tmp_path, monkeypatch) -> None:
    services, owner, base = _graph_services(tmp_path, monkeypatch)

    context = _retrieve(services, owner, base, k=4)

    assert "Apollo team" in context
    assert "Budget rules" not in context
    assert "is funded by" not in context


def test_retrieve_context_small_k_keeps_vector_passages(tmp_path, monkeypatch) -> None:
    services, owner, base = _graph_services(tmp_path, monkeypatch)
    services.settings_repo.set("knowledge_graph_enabled", "true")

    context = _retrieve(services, owner, base, k=2)

    # The graph path holds a fused slot (k/2 quota): the linked passage rides
    # alongside the best vector hit instead of being crowded out entirely.
    assert "Apollo overview" in context
    assert "Budget rules" in context
    assert "is funded by" in context


def test_retrieve_context_survives_a_broken_graph(tmp_path, monkeypatch) -> None:
    services, owner, base = _graph_services(tmp_path, monkeypatch)
    services.settings_repo.set("knowledge_graph_enabled", "true")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("graph storage is unavailable")

    monkeypatch.setattr(retrieve_module, "expand_query", boom)

    context = _retrieve(services, owner, base, k=4)

    assert "Apollo team" in context
    assert "is funded by" not in context


def _lexicon_services(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path / "home"))
    pool = SqlitePool(tmp_path / "octop.db")
    run_migrations(pool)
    services = SimpleNamespace(
        knowledge_repo=KnowledgeRepo(pool),
        settings_repo=SettingsRepo(pool),
        user_repo=UserRepo(pool),
    )
    owner = services.user_repo.create(username="owner", password_hash="h", role="user")
    base = services.knowledge_repo.create_base(owner_user_id=owner, name="Docs")
    document = services.knowledge_repo.create_document(
        kb_id=base.id,
        filename="policy.md",
        content_type="text/markdown",
        byte_size=1,
        status="ready",
    )
    # The anchored section is orthogonal to the query vector: only the exact-term
    # path can reach it.
    KnowledgeIndex(base.id).replace_doc_chunks(
        document.id,
        ["票据需要在三个工作日内提交给财务部门"],
        [[0.0, 1.0]],
        anchors=["§6.1 报销流程"],
    )
    monkeypatch.setattr(retrieve_module, "assert_knowledge_usable", lambda *_args: None)
    monkeypatch.setattr(
        retrieve_module, "embed_knowledge_texts", lambda _services, _texts: [[1.0, 0.0]]
    )
    return services, owner, base


def test_retrieve_context_lexicon_path_reaches_orthogonal_passage(tmp_path, monkeypatch) -> None:
    services, owner, base = _lexicon_services(tmp_path, monkeypatch)

    context = asyncio.run(
        retrieve_module.retrieve_context(
            services,
            user_id=owner,
            is_admin=False,
            query="报销 流程 怎么 提交",
            knowledge_base_ids=[base.id],
        )
    )

    assert "票据需要在三个工作日内提交" in context
    assert "§6.1 报销流程" in context


def test_retrieve_context_survives_a_broken_lexicon_path(tmp_path, monkeypatch) -> None:
    services, owner, base = _lexicon_services(tmp_path, monkeypatch)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("lexicon storage is unavailable")

    monkeypatch.setattr(KnowledgeIndex, "lexicon_search", boom)

    context = asyncio.run(
        retrieve_module.retrieve_context(
            services,
            user_id=owner,
            is_admin=False,
            query="票据 提交 期限",
            knowledge_base_ids=[base.id],
        )
    )

    assert "票据需要在三个工作日内提交" in context


def _advanced_services(tmp_path, monkeypatch, *, mode: str):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OCTOP_HOME", str(tmp_path / "home"))
    pool = SqlitePool(tmp_path / "octop.db")
    run_migrations(pool)
    services = SimpleNamespace(
        knowledge_repo=KnowledgeRepo(pool),
        settings_repo=SettingsRepo(pool),
        user_repo=UserRepo(pool),
    )
    owner = services.user_repo.create(username="owner", password_hash="h", role="user")
    base = services.knowledge_repo.create_base(owner_user_id=owner, name="Docs")
    if mode != "basic":
        services.knowledge_repo.update_base(base.id, retrieval_mode=mode)
    document = services.knowledge_repo.create_document(
        kb_id=base.id,
        filename="policy.md",
        content_type="text/markdown",
        byte_size=1,
        status="ready",
    )
    KnowledgeIndex(base.id).replace_doc_chunks(
        document.id, ["票据需要在三个工作日内提交"], [[1.0, 0.0]], anchors=["§6.1 报销流程"]
    )
    monkeypatch.setattr(retrieve_module, "assert_knowledge_usable", lambda *_args: None)
    monkeypatch.setattr(
        retrieve_module, "embed_knowledge_texts", lambda _services, _texts: [[1.0, 0.0]]
    )
    return services, owner, base


def test_advanced_mode_passes_the_llm_model_to_tree_and_graph(tmp_path, monkeypatch) -> None:
    from octop.infra.knowledge.page_index import SqlitePageIndex
    from octop.infra.knowledge.parse import Page
    from octop.infra.knowledge.tree import build_nodes

    services, owner, base = _advanced_services(tmp_path, monkeypatch, mode="advanced")
    document = services.knowledge_repo.list_documents(base.id)[0]
    SqlitePageIndex(base.id).replace_doc_nodes(
        document.id,
        build_nodes(
            [Page(no=None, anchor="§6.1 报销流程", text="票据需要在三个工作日内提交")], document.id
        ),
    )
    sentinel = object()
    monkeypatch.setattr(retrieve_module, "build_pageindex_chat_model", lambda _services: sentinel)
    seen: list[object] = []

    def fake_search_llm(self: object, query: str, k: int, chat_model: object) -> None:
        seen.append(chat_model)
        return None

    monkeypatch.setattr(SqlitePageIndex, "search_llm", fake_search_llm)

    context = asyncio.run(
        retrieve_module.retrieve_context(
            services,
            user_id=owner,
            is_admin=False,
            query="报销 流程 提交",
            knowledge_base_ids=[base.id],
        )
    )

    assert "票据需要在三个工作日内提交" in context
    assert seen == [sentinel]


def test_basic_mode_never_calls_the_llm_paths(tmp_path, monkeypatch) -> None:
    from octop.infra.knowledge.page_index import SqlitePageIndex

    services, owner, base = _advanced_services(tmp_path, monkeypatch, mode="basic")

    def fail(_services: object) -> object:
        raise AssertionError("basic mode must not build an LLM model")

    monkeypatch.setattr(retrieve_module, "build_pageindex_chat_model", fail)

    def fake_search_llm(self: object, query: str, k: int, chat_model: object) -> None:
        raise AssertionError("basic mode must not call LLM tree search")

    monkeypatch.setattr(SqlitePageIndex, "search_llm", fake_search_llm)

    context = asyncio.run(
        retrieve_module.retrieve_context(
            services,
            user_id=owner,
            is_admin=False,
            query="报销 流程 提交",
            knowledge_base_ids=[base.id],
        )
    )

    assert "票据需要在三个工作日内提交" in context
