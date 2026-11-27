"""The frozen local expert market snapshot (octop-qingai product decision)."""

from __future__ import annotations

import pytest

import octop.infra.market.http as market_http
from octop.infra.market import (
    clear_market_cache,
    get_market_expert,
    list_market_experts,
)


@pytest.fixture(autouse=True)
def _isolated_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTOP_PORTAL_ORIGIN", "")
    clear_market_cache()


def test_expert_market_never_touches_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, *, limit: int) -> bytes:
        del url, limit
        raise AssertionError("expert market must not fetch the portal")

    monkeypatch.setattr(market_http, "fetch_bytes", boom)
    items, scenes = list_market_experts()
    assert items
    assert scenes


def test_snapshot_icons_are_all_local() -> None:
    items, _ = list_market_experts()
    for item in items:
        icon_url = item.get("icon_url")
        assert isinstance(icon_url, str) and icon_url.startswith("/experts/avatars/")
        assert icon_url.endswith(".svg")


def test_same_slug_uses_same_name_avatar() -> None:
    expert = get_market_expert("wechat-ops")
    assert expert is not None
    assert expert["icon_url"] == "/experts/avatars/wechat-ops.svg"


def test_detail_served_from_snapshot() -> None:
    expert = get_market_expert("wechat-ops")
    assert expert is not None
    assert expert["slug"] == "wechat-ops"
    assert get_market_expert("does-not-exist-anywhere") is None
