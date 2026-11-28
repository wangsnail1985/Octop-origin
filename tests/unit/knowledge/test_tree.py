"""Unit tests for building page-index trees from anchored page units."""

from __future__ import annotations

from octop.infra.knowledge.parse import Page
from octop.infra.knowledge.tree import CONTENT_LIMIT, build_nodes


def test_build_nodes_derives_hierarchy_from_numbered_paths() -> None:
    pages = [
        Page(no=None, anchor="¶0", text="前言内容"),
        Page(no=None, anchor="§2 总则", text=""),
        Page(no=None, anchor="§2.1 适用范围", text="本办法适用于报销流程"),
        Page(no=None, anchor="§2.2 定义", text="术语定义"),
        Page(no=None, anchor="§3 附则", text="附则内容"),
    ]

    nodes = build_nodes(pages, "doc-1")

    by_path = {node.path: node for node in nodes}
    assert sorted(by_path) == ["", "2", "2.1", "2.2", "3"]
    assert by_path["2.1"].parent_id == by_path["2"].node_id
    assert by_path["2.2"].parent_id == by_path["2"].node_id
    assert by_path["2"].parent_id is None
    assert by_path["2.1"].level == 2
    # An empty parent aggregates descendant text so hitting it stays readable.
    assert "本办法适用于报销流程" in by_path["2"].content
    assert "术语定义" in by_path["2"].content


def test_build_nodes_keeps_preamble_and_flat_paragraphs() -> None:
    pages = [
        Page(no=None, anchor="¶0", text="文首内容第一行"),
        Page(no=None, anchor="¶1", text="无标题段落"),
        Page(no=None, anchor="§ 报销流程", text="报销流程说明"),
    ]

    nodes = build_nodes(pages, "doc-1")

    assert nodes[0].anchor == "¶0"
    assert nodes[0].title == "文首内容第一行"
    assert nodes[0].parent_id is None
    assert nodes[1].level == 1
    unnumbered = nodes[2]
    assert unnumbered.path == ""
    assert unnumbered.title == "报销流程"
    assert unnumbered.level == 1


def test_build_nodes_page_anchors_flatten_with_page_title() -> None:
    pages = [
        Page(no=1, anchor=None, text="第一页内容"),
        Page(no=2, anchor=None, text="第二页内容"),
    ]

    nodes = build_nodes(pages, "doc-1")

    assert [node.title for node in nodes] == ["第 1 页", "第 2 页"]
    assert all(node.page == int(node.title.split()[1]) for node in nodes)
    assert all(node.parent_id is None for node in nodes)


def test_build_nodes_positions_are_stable_and_ids_unique() -> None:
    pages = [Page(no=None, anchor=f"¶{index}", text=f"段{index}") for index in range(4)]

    nodes = build_nodes(pages, "doc-1")

    assert [node.position for node in nodes] == list(range(4))
    assert len({node.node_id for node in nodes}) == 4


def test_build_nodes_content_is_capped() -> None:
    pages = [Page(no=None, anchor="¶1", text="x" * (CONTENT_LIMIT + 500))]

    nodes = build_nodes(pages, "doc-1")

    assert len(nodes[0].content) == CONTENT_LIMIT
    assert nodes[0].summary == "x" * 120


def test_build_nodes_skips_empty_units() -> None:
    pages = [Page(no=None, anchor="¶0", text="  "), Page(no=None, anchor="§1 概述", text="正文")]

    nodes = build_nodes(pages, "doc-1")

    assert [node.anchor for node in nodes] == ["§1 概述"]
