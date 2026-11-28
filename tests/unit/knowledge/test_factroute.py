"""Unit tests for fact-style query detection and boosting."""

from __future__ import annotations

from octop.infra.knowledge.factroute import fact_boost_enabled, is_fact_query


def test_detects_zh_fact_queries() -> None:
    assert is_fact_query("报销流程在第12页怎么写")
    assert is_fact_query("第三章讲的是什么")
    assert is_fact_query("这个词出现了多少次")
    assert is_fact_query("第二条说了什么")


def test_detects_en_fact_queries() -> None:
    assert is_fact_query("what does page 12 say")
    assert is_fact_query("how many times is the term used")
    assert is_fact_query("see section 3.2 for details")


def test_non_fact_queries_stay_ordinary() -> None:
    assert not is_fact_query("报销的流程是什么")
    assert not is_fact_query("how do I submit an expense claim")
    assert not is_fact_query("")
    assert not is_fact_query("第几个方案比较好")  # "第几页" requires 页


def test_fact_boost_defaults_on_and_reads_setting() -> None:
    values: dict[str, str] = {}
    assert fact_boost_enabled(values.get) is True
    values["knowledge_fact_boost"] = "false"
    assert fact_boost_enabled(values.get) is False
    values["knowledge_fact_boost"] = "true"
    assert fact_boost_enabled(values.get) is True
