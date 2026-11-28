"""Page-index tree storage plus keyword and optional LLM node retrieval.

SQLite stores the tree in the same ``index.sqlite`` sidecar as the chunks, so
one transaction replaces both; PostgreSQL gets ``public.knowledge_page_nodes``
shaped like ``knowledge_chunks``. Keyword search is the zero-LLM default; the
LLM mode asks a small chat model to pick nodes from a catalog and falls back
to keyword whenever its reply cannot be parsed.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from octop.config import DatabaseConfig
from octop.i18n import tr
from octop.infra.knowledge.lexicon import coverage, terms
from octop.infra.knowledge.tree import TreeNode
from octop.infra.utils.paths import PathLayout

# kb-test measured values: coverage floor for node-level matching.
_COVERAGE_FLOOR = 0.30
_TITLE_WEIGHT = 3.0

PAGEINDEX_LLM_PROVIDER_KEY = "knowledge_pageindex_llm_provider_id"
PAGEINDEX_LLM_MODEL_KEY = "knowledge_pageindex_llm_model"

_COLUMNS = (
    "node_id, doc_id, parent_id, path, title, level, page, anchor, position, text, content, summary"
)


def _node_from_row(row: tuple[Any, ...]) -> TreeNode:
    (
        node_id,
        doc_id,
        parent_id,
        path,
        title,
        level,
        page,
        anchor,
        position,
        text,
        content,
        summary,
    ) = row
    return TreeNode(
        node_id=str(node_id),
        doc_id=str(doc_id),
        parent_id=str(parent_id) if parent_id is not None else None,
        path=str(path or ""),
        title=str(title or ""),
        level=int(level),
        page=int(page) if page is not None else None,
        anchor=str(anchor) if anchor is not None else None,
        position=int(position),
        text=str(text or ""),
        content=str(content or ""),
        summary=str(summary or ""),
    )


def _node_row(node: TreeNode) -> tuple[object, ...]:
    return (
        node.node_id,
        node.doc_id,
        node.parent_id,
        node.path,
        node.title,
        node.level,
        node.page,
        node.anchor,
        node.position,
        node.text,
        node.content,
        node.summary,
    )


def score_nodes(nodes: Sequence[TreeNode], query: str) -> list[TreeNode]:
    """Rank nodes by query-term coverage; titles weigh three times the body."""
    term_list = terms(query)
    if not term_list:
        return []
    scored: list[tuple[float, TreeNode]] = []
    for node in nodes:
        head = coverage(f"{node.title}\n{node.anchor or ''}", term_list)
        body = coverage(node.content, term_list)
        combined = coverage(f"{node.title}\n{node.anchor or ''}\n{node.content}", term_list)
        if combined <= 0.0:
            continue
        if head <= 0.0 and combined < _COVERAGE_FLOOR and len(term_list) > 1:
            continue
        scored.append((_TITLE_WEIGHT * head + body, node))
    scored.sort(key=lambda item: (-item[0], item[1].node_id))
    return [node for _, node in scored]


class PageIndexStore:
    """Tree rows of one knowledge base. Subclasses supply the connection."""

    _mark = "?"
    _table = "page_nodes"

    def __init__(self, kb_id: str) -> None:
        self._kb_id = kb_id

    # -- connection plumbing -------------------------------------------------

    def _connect(self) -> Any:
        raise NotImplementedError

    def _present(self) -> bool:
        return True

    def _prepare(self, conn: Any) -> None:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                node_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                parent_id TEXT,
                path TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                level INTEGER NOT NULL DEFAULT 1,
                page INTEGER,
                anchor TEXT,
                position INTEGER NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_page_nodes_doc ON {self._table}(doc_id)")

    @contextmanager
    def _session(self) -> Iterator[Any]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _sql(self, statement: str) -> str:
        if self._mark == "?":
            return statement
        return statement.replace("%", "%%").replace("?", self._mark)

    def _fetch(self, statement: str, params: Sequence[object] = ()) -> list[tuple[Any, ...]]:
        if not self._present():
            return []
        with self._session() as conn:
            rows = conn.execute(self._sql(statement), tuple(params)).fetchall()
        return [tuple(row) for row in rows]

    # -- writes --------------------------------------------------------------

    def replace_doc_nodes(self, doc_id: str, nodes: Sequence[TreeNode]) -> None:
        """Atomically replace one document's tree; failure keeps the old tree."""
        rows = [_node_row(node) for node in nodes]
        with self._session() as conn:
            conn.execute(self._sql(f"DELETE FROM {self._table} WHERE doc_id = ?"), (doc_id,))
            if rows:
                conn.executemany(
                    self._sql(
                        f"INSERT INTO {self._table}({_COLUMNS}) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    rows,
                )

    def delete_doc(self, doc_id: str) -> None:
        if not self._present():
            return
        with self._session() as conn:
            conn.execute(self._sql(f"DELETE FROM {self._table} WHERE doc_id = ?"), (doc_id,))

    # -- reads ---------------------------------------------------------------

    def doc_node_rows(self, doc_id: str) -> list[TreeNode]:
        rows = self._fetch(
            f"SELECT {_COLUMNS} FROM {self._table} WHERE doc_id = ? ORDER BY position",
            (doc_id,),
        )
        return [_node_from_row(row) for row in rows]

    def node_at(self, doc_id: str, position: int) -> TreeNode | None:
        """The tree node at one document position, for graph references."""
        rows = self._fetch(
            f"SELECT {_COLUMNS} FROM {self._table} WHERE doc_id = ? AND position = ?",
            (doc_id, position),
        )
        return _node_from_row(rows[0]) if rows else None

    def all_nodes(self) -> list[TreeNode]:
        rows = self._fetch(
            f"SELECT {_COLUMNS} FROM {self._table} ORDER BY doc_id, position",
        )
        return [_node_from_row(row) for row in rows]

    def search_keyword(self, query: str, k: int) -> list[TreeNode]:
        """Node-level exact-term search over the whole tree."""
        if k <= 0:
            return []
        return score_nodes(self.all_nodes(), query)[:k]

    def search_llm(self, query: str, k: int, chat_model: Any) -> list[TreeNode] | None:
        """Ask a small model to pick catalog nodes; None means "degrade"."""
        nodes = self.all_nodes()
        if not nodes or k <= 0:
            return []
        catalog = "\n".join(
            f"{node.node_id}\t{node.path or '-'}\t{node.summary or node.title}" for node in nodes
        )
        prompt = tr(
            "knowledge.pageindex.llm_prompt",
            "en",
            catalog=catalog,
            query=query,
            k=k,
        )
        try:
            from langchain_core.messages import HumanMessage

            from octop.infra.utils.llm_text import llm_text_content

            reply = llm_text_content(chat_model.invoke([HumanMessage(content=prompt)]))
        except Exception:
            return None
        picked = _parse_node_ids(reply)
        if not picked:
            return None
        by_id = {node.node_id: node for node in nodes}
        selected = [by_id[node_id] for node_id in picked if node_id in by_id]
        return selected[:k] or None


def _parse_node_ids(reply: str) -> list[str]:
    text = (reply or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        decoded = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [
        str(item) for item in decoded if isinstance(item, (str, int)) and not isinstance(item, bool)
    ]


# -- LLM capability -----------------------------------------------------------


def _provider(provider_repo: Any, provider_id: str) -> Any | None:
    if provider_repo is None or not provider_id.isdigit():
        return None
    return provider_repo.get(int(provider_id))


def get_pageindex_llm_capability(settings_get: Any, provider_repo: Any = None) -> dict[str, Any]:
    """Stored small-model choice for LLM tree search and whether it is ready.

    There is no ``enabled`` flag: advanced retrieval mode opts in per knowledge
    base, this only says whether the instance has a usable model configured.
    """
    from octop.infra.knowledge.graph_extract import graph_models

    provider_id = (settings_get(PAGEINDEX_LLM_PROVIDER_KEY) or "").strip()
    model = (settings_get(PAGEINDEX_LLM_MODEL_KEY) or "").strip()
    ready = bool(model) and any(
        item["id"] == model for item in graph_models(_provider(provider_repo, provider_id))
    )
    return {"provider_id": provider_id, "model": model, "ready": ready, "usable": ready}


def set_pageindex_llm_settings(
    settings_set: Any,
    *,
    provider_id: str | None,
    model: str | None,
    provider_repo: Any = None,
) -> None:
    """Persist the tree-search model; a non-empty choice must be ready."""
    from octop.infra.knowledge.graph_extract import graph_models

    selected_provider = (provider_id or "").strip()
    selected_model = (model or "").strip()
    if selected_model and not any(
        item["id"] == selected_model
        for item in graph_models(_provider(provider_repo, selected_provider))
    ):
        raise ValueError("knowledge page-index model is not available on the selected provider")
    settings_set(PAGEINDEX_LLM_PROVIDER_KEY, selected_provider)
    settings_set(PAGEINDEX_LLM_MODEL_KEY, selected_model)


def build_pageindex_chat_model(services: Any) -> Any:
    """LangChain chat model for tree search, or ``None`` when not configured."""
    from octop.infra.agents.providers.probe import build_probe_chat_model

    capability = get_pageindex_llm_capability(
        services.settings_repo.get, getattr(services, "provider_repo", None)
    )
    if not capability["usable"]:
        return None
    provider = _provider(services.provider_repo, capability["provider_id"])
    if provider is None:
        return None
    return build_probe_chat_model(provider, model_id=capability["model"])


class SqlitePageIndex(PageIndexStore):
    """``page_nodes`` inside the knowledge base's ``index.sqlite``."""

    def __init__(self, kb_id: str) -> None:
        super().__init__(kb_id)
        self._path = PathLayout.from_env().knowledge_dir / kb_id / "index.sqlite"

    def _present(self) -> bool:
        return self._path.is_file()

    def _connect(self) -> Any:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path)
        self._prepare(conn)
        return conn


class PgPageIndex(PageIndexStore):
    """``public.knowledge_page_nodes`` shared by every knowledge base."""

    _mark = "%s"
    _table = "public.knowledge_page_nodes"

    def __init__(self, kb_id: str, database: DatabaseConfig) -> None:
        super().__init__(kb_id)
        self._conninfo = database.postgresql_conninfo()
        with self._session() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS public.knowledge_page_nodes (
                    kb_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    parent_id TEXT,
                    path TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    level INTEGER NOT NULL DEFAULT 1,
                    page INTEGER,
                    anchor TEXT,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (kb_id, node_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_page_nodes_doc
                ON public.knowledge_page_nodes (kb_id, doc_id)
                """
            )

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self._conninfo)

    def _prepare(self, conn: Any) -> None:  # schema handled in __init__
        return None

    def replace_doc_nodes(self, doc_id: str, nodes: Sequence[TreeNode]) -> None:
        rows = [(self._kb_id, *_node_row(node)) for node in nodes]
        with self._session() as conn:
            conn.execute(
                self._sql(f"DELETE FROM {self._table} WHERE kb_id = ? AND doc_id = ?"),
                (self._kb_id, doc_id),
            )
            if rows:
                conn.executemany(
                    self._sql(
                        f"INSERT INTO {self._table}(kb_id, {_COLUMNS}) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    rows,
                )

    def delete_doc(self, doc_id: str) -> None:
        with self._session() as conn:
            conn.execute(
                self._sql(f"DELETE FROM {self._table} WHERE kb_id = ? AND doc_id = ?"),
                (self._kb_id, doc_id),
            )

    def delete_kb(self) -> None:
        with self._session() as conn:
            conn.execute(self._sql(f"DELETE FROM {self._table} WHERE kb_id = ?"), (self._kb_id,))

    def doc_node_rows(self, doc_id: str) -> list[TreeNode]:
        with self._session() as conn:
            rows = conn.execute(
                self._sql(
                    f"SELECT {_COLUMNS} FROM {self._table} "
                    "WHERE kb_id = ? AND doc_id = ? ORDER BY position"
                ),
                (self._kb_id, doc_id),
            ).fetchall()
        return [_node_from_row(tuple(row)) for row in rows]

    def all_nodes(self) -> list[TreeNode]:
        with self._session() as conn:
            rows = conn.execute(
                self._sql(
                    f"SELECT {_COLUMNS} FROM {self._table} "
                    "WHERE kb_id = ? ORDER BY doc_id, position"
                ),
                (self._kb_id,),
            ).fetchall()
        return [_node_from_row(tuple(row)) for row in rows]
