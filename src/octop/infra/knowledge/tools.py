"""Built-in LangChain tools for on-demand knowledge-base retrieval."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from pydantic import Field

from octop.infra.knowledge.plane import open_knowledge_index, open_page_index
from octop.infra.knowledge.retrieve import DEFAULT_RETRIEVAL_K, retrieve_context

SEARCH_KNOWLEDGE_TOOL = "search_knowledge"
COUNT_OCCURRENCES_TOOL = "count_occurrences"

# API allows 2000-char KB descriptions; keep the tool schema compact.
_MAX_CATALOG_DESC_CHARS = 240

# Counting reads whole documents; cap how much of each we scan.
_MAX_SECTIONS_LISTED = 8
_EXCERPT_CHARS = 60

_SEARCH_DESC_BASE = (
    "Search the knowledge bases listed below for relevant passages. "
    "Use this tool when the user's question overlaps any of those topics. "
    "Write `query` as a focused search string (key terms, names, synonyms), "
    "not a chat reply. Do not invent citations; only use returned passages."
)

_SEARCH_DESC_NONE = "No knowledge bases are attached this turn."

_COUNT_DESC = (
    "Count exactly how many times a term or phrase occurs across the full text "
    "of knowledge-base documents, and where (page/section anchors). Use this "
    "for 'how many times / 出现多少次' questions instead of counting retrieved "
    "passages yourself. `query` is the exact term to count, not a question; "
    "optionally pass `kb_ids` to restrict the bases, otherwise the ones "
    "attached this turn are used."
)


def _clip_catalog_text(text: str, limit: int = _MAX_CATALOG_DESC_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."


def format_search_knowledge_description(
    catalog: Sequence[Mapping[str, str]] | None,
) -> str:
    """Build the LLM-facing tool description, including attached KB titles/descriptions."""
    entries: list[str] = []
    for item in catalog or ():
        name = _clip_catalog_text(str(item.get("name") or item.get("id") or ""), 80)
        if not name:
            continue
        description = _clip_catalog_text(str(item.get("description") or ""))
        if description:
            entries.append(f"- {name}: {description}")
        else:
            entries.append(f"- {name}")
    if not entries:
        return _SEARCH_DESC_NONE
    listed = "\n".join(entries)
    return f"{_SEARCH_DESC_BASE}\nAttached this turn:\n{listed}"


def _tool_ctx() -> tuple[int, bool, list[str], str]:
    cfg = get_config().get("configurable") or {}
    user_raw = cfg.get("user")
    if user_raw is None:
        raise ValueError("missing configurable.user")
    user_id = int(user_raw)
    is_admin = bool(cfg.get("user_is_admin"))
    raw_ids = cfg.get("knowledge_base_ids")
    ids: list[str] = []
    if isinstance(raw_ids, list):
        ids = [str(item).strip() for item in raw_ids if str(item).strip()]
    locale = str(cfg.get("locale") or "en")
    return user_id, is_admin, ids, locale


class KnowledgeToolServices:
    """Live repos and control-plane config for ``search_knowledge``.

    Indexing selects pgvector from ``services.config``. The chat tool has to
    read the same config, including after setup replaces it.
    """

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    @property
    def config(self) -> Any:
        return self._manager._config

    @property
    def knowledge_repo(self) -> Any:
        return self._manager._repos.knowledge_repo

    @property
    def settings_repo(self) -> Any:
        return self._manager._repos.settings_repo

    @property
    def provider_repo(self) -> Any:
        return self._manager._repos.provider_repo


def build_knowledge_tools(services: Any) -> list[StructuredTool]:
    """Return built-in knowledge retrieval tools (wired via HarnessAgentConfig.tools)."""

    async def search_knowledge(
        query: Annotated[
            str,
            Field(
                description=(
                    "Focused search string for the selected knowledge bases: "
                    "key terms, names, and synonyms from the user's question. "
                    "Not a chat reply."
                ),
            ),
        ],
        k: Annotated[
            int,
            Field(
                description="Max passages to return (default 8).",
                ge=1,
                le=20,
            ),
        ] = DEFAULT_RETRIEVAL_K,
    ) -> str:
        try:
            user_id, is_admin, kb_ids, locale = _tool_ctx()
            if not kb_ids:
                return "No knowledge bases selected for this turn."
            context = await retrieve_context(
                services,
                user_id=user_id,
                is_admin=is_admin,
                query=query,
                knowledge_base_ids=kb_ids,
                k=k,
                locale=locale,
            )
            if not context:
                return "No relevant passages found."
            return context
        except Exception as exc:
            return f"Knowledge search failed: {exc}"

    async def count_occurrences(
        query: Annotated[
            str,
            Field(
                description=("The exact term or phrase to count occurrences of (not a question)."),
            ),
        ],
        kb_ids: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Optional knowledge-base ids to restrict the count to; "
                    "defaults to the bases attached this turn."
                ),
            ),
        ] = None,
    ) -> str:
        try:
            user_id, is_admin, turn_ids, _locale = _tool_ctx()
            ids = [str(item).strip() for item in (kb_ids or turn_ids) if str(item).strip()]
            if not ids:
                return "No knowledge bases selected for this turn."
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: _count_occurrences_sync(
                    services,
                    user_id=user_id,
                    is_admin=is_admin,
                    term=query,
                    kb_ids=ids,
                ),
            )
        except Exception as exc:
            return f"Knowledge count failed: {exc}"

    return [
        StructuredTool.from_function(
            coroutine=search_knowledge,
            name=SEARCH_KNOWLEDGE_TOOL,
            # Static fallback; middleware hides or rewrites this per turn.
            description=format_search_knowledge_description([]),
        ),
        StructuredTool.from_function(
            coroutine=count_occurrences,
            name=COUNT_OCCURRENCES_TOOL,
            description=_COUNT_DESC,
        ),
    ]


def _excerpt(text: str, needle: str) -> str:
    position = text.casefold().find(needle)
    if position < 0:
        return ""
    start = max(position - _EXCERPT_CHARS // 2, 0)
    end = min(position + len(needle) + _EXCERPT_CHARS // 2, len(text))
    return " ".join(text[start:end].split())


def _count_occurrences_sync(
    services: Any,
    *,
    user_id: int,
    is_admin: bool,
    term: str,
    kb_ids: Sequence[str],
) -> str:
    """Exact string counting over whole documents; the model only narrates.

    Tree nodes cover their document without overlap, so counting is exact.
    Documents still stored as overlapping chunk windows are counted
    approximately and flagged as such.
    """
    needle = " ".join(term.split()).casefold()
    if not needle:
        return json.dumps({"term": term, "documents": [], "total": 0}, ensure_ascii=False)
    visible = (
        services.knowledge_repo.list_all()
        if is_admin
        else services.knowledge_repo.list_visible(user_id)
    )
    visible_by_id = {base.id: base for base in visible}
    documents: list[dict[str, Any]] = []
    total = 0
    approximate = False
    for kb_id in dict.fromkeys(kb_ids):
        base = visible_by_id.get(kb_id)
        if base is None:
            continue
        page_index = open_page_index(services, kb_id)
        index = open_knowledge_index(services, kb_id)
        for document in services.knowledge_repo.list_documents(kb_id):
            if document.is_dir or document.status != "ready":
                continue
            nodes = page_index.doc_node_rows(document.id)
            sections: list[dict[str, Any]] = []
            doc_total = 0
            if nodes:
                for node in nodes:
                    count = node.text.casefold().count(needle)
                    if count:
                        doc_total += count
                        sections.append(
                            {
                                "anchor": node.anchor,
                                "title": node.title,
                                "count": count,
                                "excerpt": _excerpt(node.text, needle),
                            }
                        )
            else:
                approximate = True
                for ordinal, text in index.list_doc_chunks(document.id):
                    count = text.casefold().count(needle)
                    if count:
                        doc_total += count
                        sections.append(
                            {
                                "anchor": None,
                                "chunk": ordinal + 1,
                                "count": count,
                                "excerpt": _excerpt(text, needle),
                            }
                        )
            if not doc_total:
                continue
            total += doc_total
            listed, rest = sections[:_MAX_SECTIONS_LISTED], sections[_MAX_SECTIONS_LISTED:]
            entry: dict[str, Any] = {
                "kb_name": base.name,
                "filename": document.filename,
                "total": doc_total,
                "sections": listed,
            }
            if rest:
                entry["other_sections"] = len(rest)
            documents.append(entry)
    payload: dict[str, Any] = {
        "term": term,
        "total": total,
        "documents": documents,
    }
    if approximate:
        payload["approximate"] = True
    return json.dumps(payload, ensure_ascii=False)
