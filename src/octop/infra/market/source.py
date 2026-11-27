"""Load a plugin catalog from the Octop portal; the expert market is a frozen local snapshot."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import octop.infra.market.http as market_http
from octop.infra.market.errors import PortalMarketError
from octop.infra.market.normalize import sort_recent

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 30.0

_cache_lock = threading.Lock()
_plugin_cache: tuple[float, list[dict[str, Any]]] | None = None

# Frozen expert market (product decision, octop-qingai): the portal feed is no
# longer contacted for experts; `local_experts.json` is a captured snapshot
# whose icon_url entries all point at bundled `/experts/avatars/*.svg`.
_LOCAL_EXPERTS_PATH = Path(__file__).with_name("local_experts.json")
_local_experts_cache: list[dict[str, Any]] | None = None


def clear_market_cache() -> None:
    """Drop the in-process catalog cache. Tests call this between cases."""
    global _plugin_cache
    with _cache_lock:
        _plugin_cache = None


def cached(scope: str) -> list[dict[str, Any]]:
    """Return one scope: experts read the frozen local snapshot, plugins the portal."""
    if scope == "experts":
        return _load_local_experts()
    global _plugin_cache
    now = time.monotonic()
    with _cache_lock:
        if _plugin_cache is not None and now - _plugin_cache[0] < _CACHE_TTL_SECONDS:
            return _plugin_cache[1]
        items = _load_portal(scope)
        stamped = (now, items)
        _plugin_cache = stamped
        return items


def _load_local_experts() -> list[dict[str, Any]]:
    """Return the bundled expert market snapshot, memoized in-process."""
    global _local_experts_cache
    if _local_experts_cache is None:
        try:
            payload = json.loads(_LOCAL_EXPERTS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PortalMarketError("local expert catalog is unreadable") from exc
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise PortalMarketError("local expert catalog is invalid")
        logger.info(
            "serving frozen local expert market (%d items)", len(items)
        )
        _local_experts_cache = sort_recent(
            [item for item in items if isinstance(item, dict)]
        )
    return _local_experts_cache


def _load_portal(scope: str) -> list[dict[str, Any]]:
    base = market_http.portal_base()
    if not base:
        raise PortalMarketError("portal url is empty")
    url = f"{base}/api/market/{scope}"
    try:
        payload = market_http.fetch_json(url, limit=market_http.MAX_INDEX_BYTES)
    except PortalMarketError as exc:
        logger.info("portal %s catalog unavailable: %s", scope, exc)
        raise
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise PortalMarketError("portal catalog is invalid")
    data = payload.get("data")
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise PortalMarketError("portal catalog is invalid")
    return sort_recent([item for item in items if isinstance(item, dict)])
