"""Build a heading-level page-index tree from parsed page units.

The tree derives entirely from the anchors produced at parse time: numbered
heading paths (``§2.1``) imply hierarchy, everything else (``¶N`` paragraphs,
numbered pages/slides) becomes a flat leaf. No LLM is involved in building.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from octop.infra.knowledge.parse import Page

CONTENT_LIMIT = 1500
SUMMARY_LIMIT = 120
_TITLE_PREVIEW = 50

_PATH_RE = re.compile(r"^(\d+(?:\.\d+)*)$")


@dataclass(slots=True)
class TreeNode:
    """One node of a document's page-index tree."""

    node_id: str
    doc_id: str
    parent_id: str | None
    path: str
    title: str
    level: int
    page: int | None
    anchor: str | None
    position: int
    text: str
    content: str = ""
    summary: str = field(default="")


def _split_anchor(anchor: str) -> tuple[str, str]:
    """Split ``§2.1 报销流程`` into ``("2.1", "报销流程")`` (path may be empty)."""
    body = anchor[1:].strip()
    parts = body.split(None, 1)
    if parts and _PATH_RE.match(parts[0]):
        return parts[0], (parts[1].strip() if len(parts) > 1 else parts[0])
    return "", body


def _plain_title(unit: Page) -> str:
    if unit.no is not None:
        return f"第 {unit.no} 页"
    first_line = next(
        (line.strip() for line in unit.text.splitlines() if line.strip()),
        "",
    )
    return first_line[:_TITLE_PREVIEW] or "文首"


def build_nodes(pages: Sequence[Page], doc_id: str) -> list[TreeNode]:
    """Turn anchored page units into a tree (flat where anchors carry no path)."""
    nodes: list[TreeNode] = []
    last_by_path: dict[str, str] = {}
    for position, unit in enumerate(pages):
        text = unit.text.strip()
        anchor = unit.anchor
        # A heading with an empty body is still a container for its children;
        # only body-less paragraph/page units carry no information at all.
        if not text and not (anchor and anchor.startswith("§")):
            continue
        if anchor and anchor.startswith("§"):
            path, title = _split_anchor(anchor)
            level = len(path.split(".")) if path else 1
        else:
            path, title, level = "", _plain_title(unit), 1
        parent_id: str | None = None
        if path:
            prefix = ".".join(path.split(".")[:-1])
            parent_id = last_by_path.get(prefix) if prefix else None
        node = TreeNode(
            node_id=f"{doc_id}:n{position}",
            doc_id=doc_id,
            parent_id=parent_id,
            path=path,
            title=title,
            level=level,
            page=unit.no,
            anchor=anchor,
            position=position,
            text=text,
        )
        if path:
            last_by_path[path] = node.node_id
        nodes.append(node)
    _fill_content(nodes)
    return nodes


def _fill_content(nodes: Sequence[TreeNode]) -> None:
    """Parent nodes with no body aggregate their descendants' text."""
    children: dict[str | None, list[TreeNode]] = {}
    for node in nodes:
        children.setdefault(node.parent_id, []).append(node)
    for node in nodes:
        if node.text:
            node.content = node.text[:CONTENT_LIMIT]
        else:
            node.content = _aggregate(children, node.node_id)[:CONTENT_LIMIT]
        summary = " ".join((node.text or node.content).split())
        node.summary = summary[:SUMMARY_LIMIT]


def _aggregate(children: dict[str | None, list[TreeNode]], root_id: str) -> str:
    """Depth-first concatenation of descendant texts."""
    parts: list[str] = []
    stack = list(reversed(children.get(root_id, [])))
    while stack:
        node = stack.pop()
        if node.text:
            parts.append(node.text)
        stack.extend(reversed(children.get(node.node_id, [])))
        if sum(len(part) for part in parts) >= CONTENT_LIMIT:
            break
    return "\n".join(parts)
