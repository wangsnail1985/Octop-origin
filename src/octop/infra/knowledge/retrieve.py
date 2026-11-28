"""Retrieve readable knowledge-base chunks for one chat turn."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, cast

from octop.i18n import tr
from octop.infra.knowledge.citations import append_citations_marker, citations_from_ranked
from octop.infra.knowledge.embed import embed_knowledge_texts
from octop.infra.knowledge.factroute import fact_boost_enabled, is_fact_query
from octop.infra.knowledge.fuse import dedupe_key, rrf_fuse
from octop.infra.knowledge.gate import assert_knowledge_usable
from octop.infra.knowledge.graph import (
    MAX_FACTS,
    GraphExpansion,
    expand_query,
    graph_enabled,
    open_knowledge_graph,
)
from octop.infra.knowledge.index import Hit
from octop.infra.knowledge.page_index import build_pageindex_chat_model
from octop.infra.knowledge.plane import open_knowledge_index, open_page_index
from octop.infra.knowledge.rerank import candidate_count, get_rerank_capability, rerank_scores
from octop.infra.knowledge.tree import TreeNode

DEFAULT_RETRIEVAL_K = 8
DEFAULT_CONTEXT_CHAR_BUDGET = 6000
# Share of the returned passages, and of the character budget, the graph may claim.
_GRAPH_PASSAGE_DIVISOR = 4
_GRAPH_FACT_BUDGET_DIVISOR = 5

logger = logging.getLogger(__name__)


async def retrieve_context(
    services: Any,
    *,
    user_id: int,
    is_admin: bool,
    query: str | None,
    knowledge_base_ids: Sequence[str],
    k: int = DEFAULT_RETRIEVAL_K,
    char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET,
    locale: str = "en",
    visible_bases: Sequence[Any] | None = None,
) -> str:
    """Return formatted readable context, skipping unavailable or failed retrieval."""
    if not (query or "").strip() or not knowledge_base_ids or k <= 0 or char_budget <= 0:
        return ""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None,
            lambda: _retrieve_context_sync(
                services,
                user_id=user_id,
                is_admin=is_admin,
                query=query,
                knowledge_base_ids=knowledge_base_ids,
                k=k,
                char_budget=char_budget,
                locale=locale,
                visible_bases=visible_bases,
            ),
        )
    except Exception:
        logger.warning("knowledge retrieval skipped for user=%s", user_id, exc_info=True)
        return ""


def _retrieve_context_sync(
    services: Any,
    *,
    user_id: int,
    is_admin: bool,
    query: str | None,
    knowledge_base_ids: Sequence[str],
    k: int,
    char_budget: int,
    locale: str,
    visible_bases: Sequence[Any] | None,
) -> str:
    assert_knowledge_usable(services.settings_repo.get, getattr(services, "provider_repo", None))
    query_vectors = embed_knowledge_texts(services, [(query or "").strip()])
    if not query_vectors:
        return ""

    visible = visible_bases
    if visible is None:
        visible = (
            services.knowledge_repo.list_all()
            if is_admin
            else services.knowledge_repo.list_visible(user_id)
        )
    visible_by_id = {base.id: base for base in visible}
    selected_ids = _unique_ids(knowledge_base_ids)
    rerank = get_rerank_capability(
        services.settings_repo.get, getattr(services, "provider_repo", None)
    )["usable"]
    # A reranker needs more than ``k`` candidates to be worth asking.
    fetch_k = candidate_count(k) if rerank else k

    use_graph = graph_enabled(services)
    vector_entries: list[tuple[Any, Hit, Any]] = []
    lexicon_entries: list[tuple[Any, Hit, Any]] = []
    tree_entries: list[tuple[Any, Hit, Any]] = []
    facts: list[str] = []
    related: list[tuple[Any, Any, int]] = []
    tree_model: Any = None
    for kb_id in selected_ids:
        base = visible_by_id.get(kb_id)
        if base is None:
            continue
        ready_documents = {
            document.id: document
            for document in services.knowledge_repo.list_documents(kb_id)
            if document.status == "ready" and not document.is_dir
        }
        index = open_knowledge_index(services, kb_id)
        for hit in index.search(query_vectors[0], k=fetch_k):
            document = ready_documents.get(hit.doc_id)
            if document is not None:
                vector_entries.append((base, hit, document))
        try:
            for hit in index.lexicon_search(query or "", k=fetch_k):
                document = ready_documents.get(hit.doc_id)
                if document is not None:
                    lexicon_entries.append((base, hit, document))
        except Exception:
            logger.warning("knowledge lexicon search skipped for %s", kb_id, exc_info=True)
        try:
            if tree_model is None:
                tree_model = build_pageindex_chat_model(services)
            page_index = open_page_index(services, kb_id)
            for node in _search_tree(
                page_index, (query or "").strip(), k=fetch_k, chat_model=tree_model
            ):
                document = ready_documents.get(node.doc_id)
                if document is not None:
                    tree_entries.append((base, _node_hit(node), document))
        except Exception:
            logger.warning("knowledge page-index search skipped for %s", kb_id, exc_info=True)
        if use_graph:
            expansion = _expand(services, kb_id, query or "")
            facts.extend(expansion.facts)
            related.extend(
                (base, ready_documents[doc_id], ordinal)
                for doc_id, ordinal in expansion.chunks
                if doc_id in ready_documents
            )

    ranked = _fuse_entries(
        {"vector": vector_entries, "lexicon": lexicon_entries, "tree": tree_entries}
    )
    ranked = _boost_fact_hits(services, ranked, tree_entries, query or "")
    if rerank:
        ranked = _rerank(services, (query or "").strip(), ranked[:fetch_k])
    top = _with_graph_passages(services, ranked[:k], related, k=k)
    return _format_context(top, char_budget=char_budget, locale=locale, facts=facts[:MAX_FACTS])


def _search_tree(page_index: Any, query: str, *, k: int, chat_model: Any) -> list[TreeNode]:
    """LLM node picking when configured; keyword scoring as the floor."""
    if chat_model is not None:
        picked = page_index.search_llm(query, k, chat_model)
        if picked is not None:
            return cast("list[TreeNode]", picked)
    return cast("list[TreeNode]", page_index.search_keyword(query, k))


def _node_hit(node: TreeNode) -> Hit:
    return Hit(
        chunk_id=node.node_id,
        doc_id=node.doc_id,
        ordinal=node.position,
        text=node.content,
        score=0.0,
        metadata={},
        page=node.page,
        anchor=node.anchor,
    )


def _boost_fact_hits(
    services: Any,
    ranked: Sequence[tuple[Any, Hit, Any]],
    tree_entries: Sequence[tuple[Any, Hit, Any]],
    query: str,
) -> list[tuple[Any, Hit, Any]]:
    """Fact-style queries pin up to two page-index hits at the top."""
    if not ranked or not tree_entries:
        return list(ranked)
    if not fact_boost_enabled(services.settings_repo.get) or not is_fact_query(query):
        return list(ranked)
    tree_keys = {dedupe_key(hit) for _, hit, _ in tree_entries}
    boost_positions = [
        position for position, entry in enumerate(ranked) if dedupe_key(entry[1]) in tree_keys
    ][:2]
    if not boost_positions:
        return list(ranked)
    pinned = set(boost_positions)
    return [ranked[i] for i in boost_positions] + [
        entry for i, entry in enumerate(ranked) if i not in pinned
    ]


def _fuse_entries(
    engines: dict[str, Sequence[tuple[Any, Hit, Any]]],
) -> list[tuple[Any, Hit, Any]]:
    """Fuse the retrieval engines with reciprocal-rank fusion.

    Entries sharing one ``(doc, anchor)`` key keep a single vote at their best
    rank, so a section fragmented into many windows cannot outvote a strong hit.
    """
    registry: dict[tuple[str, str], tuple[Any, Hit, Any]] = {}

    def hits_of(entries: Sequence[tuple[Any, Hit, Any]]) -> list[Hit]:
        hits: list[Hit] = []
        for base, hit, document in entries:
            registry.setdefault(dedupe_key(hit), (base, hit, document))
            hits.append(hit)
        return hits

    fused = rrf_fuse({name: hits_of(entries) for name, entries in engines.items()})
    return [(registry[dedupe_key(hit)][0], hit, registry[dedupe_key(hit)][2]) for hit in fused]


def _expand(services: Any, kb_id: str, query: str) -> GraphExpansion:
    """Graph lookups are an addition; a failure leaves plain vector retrieval."""
    try:
        return expand_query(open_knowledge_graph(services, kb_id), query)
    except Exception:
        logger.warning("knowledge graph lookup skipped for %s", kb_id, exc_info=True)
        return GraphExpansion()


def _with_graph_passages(
    services: Any,
    top: list[tuple[Any, Hit, Any]],
    related: Sequence[tuple[Any, Any, int]],
    *,
    k: int,
) -> list[tuple[Any, Hit, Any]]:
    """Give the last few slots to graph-linked passages the vector search missed.

    These passages are connected to the question through an entity rather
    than by wording, so they would never win on similarity; they take at
    most a quarter of the result instead of competing on score.
    """
    slots = k // _GRAPH_PASSAGE_DIVISOR
    if slots <= 0 or not related:
        return top
    present = {hit.chunk_id for _, hit, _ in top}
    texts: dict[str, dict[int, str]] = {}
    extra: list[tuple[Any, Hit, Any]] = []
    for base, document, ordinal in related:
        chunk_id = f"{document.id}:{ordinal}"
        if chunk_id in present:
            continue
        if document.id not in texts:
            texts[document.id] = dict(
                open_knowledge_index(services, base.id).list_doc_chunks(document.id)
            )
        text = texts[document.id].get(ordinal)
        if text is None:
            continue
        present.add(chunk_id)
        extra.append((base, Hit(chunk_id, document.id, ordinal, text, 0.0, {}), document))
        if len(extra) == slots:
            break
    return top[: max(k - len(extra), 0)] + extra


def _rerank(
    services: Any, query: str, ranked: list[tuple[Any, Hit, Any]]
) -> list[tuple[Any, Hit, Any]]:
    """Reorder candidates by reranker relevance; keep the vector order if it fails."""
    try:
        scores = rerank_scores(services, query, [hit.text for _, hit, _ in ranked])
    except Exception:
        logger.warning("knowledge rerank failed; using vector order", exc_info=True)
        return ranked
    if scores is None:
        return ranked
    order = sorted(range(len(ranked)), key=lambda index: scores[index], reverse=True)
    return [ranked[index] for index in order]


def _unique_ids(ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(kb_id).strip() for kb_id in ids if str(kb_id).strip()))


def _format_context(
    ranked: Sequence[tuple[Any, Hit, Any]],
    *,
    char_budget: int,
    locale: str,
    facts: Sequence[str] = (),
) -> str:
    sections: list[str] = []
    included: list[tuple[Any, Hit, Any]] = []
    fact_block = _format_facts(
        facts, limit=char_budget // _GRAPH_FACT_BUDGET_DIVISOR, locale=locale
    )
    remaining = char_budget - len(fact_block)
    for base, hit, document in ranked:
        citation = _citation_text(base, hit, document, locale=locale)
        if remaining <= len(citation):
            break
        text = hit.text.strip()[: remaining - len(citation)]
        if not text:
            continue
        section = f"{citation}{text}"
        sections.append(section)
        included.append((base, hit, document))
        remaining -= len(section)
    if not sections:
        return ""
    body = tr("knowledge.retrieval.preamble", locale) + "\n\n" + "\n\n".join(sections)
    return append_citations_marker(body + fact_block, citations_from_ranked(included))


def _citation_text(base: Any, hit: Hit, document: Any, *, locale: str) -> str:
    """Cite a passage by its strongest anchor: page, heading, or chunk order."""
    if hit.page is not None:
        return tr(
            "knowledge.retrieval.citation_page",
            locale,
            name=base.name,
            filename=document.filename,
            page=hit.page,
        )
    if hit.anchor:
        return tr(
            "knowledge.retrieval.citation_anchor",
            locale,
            name=base.name,
            filename=document.filename,
            anchor=hit.anchor,
        )
    return tr(
        "knowledge.retrieval.citation_chunk",
        locale,
        name=base.name,
        filename=document.filename,
        ordinal=hit.ordinal + 1,
    )


def _format_facts(facts: Sequence[str], *, limit: int, locale: str) -> str:
    """Relations from the knowledge graph as a short list after the passages."""
    lines: list[str] = []
    used = 0
    for fact in dict.fromkeys(facts):
        line = f"- {fact}"
        if used + len(line) > limit:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return "\n\n" + tr("knowledge.retrieval.graph_preamble", locale) + "\n" + "\n".join(lines)
