"""企业知识库检索 — 调用 kb-service 双引擎（语义向量 + PageIndex）检索工具。

身份模型（v4.1 账户体系）：插件按「当前 octop 用户 → kb-service 账户」映射取
凭据，登录换取 Bearer 令牌后调用 ``GET /search``；kb-service 侧按账户身份
（user/dept）做标签权限过滤，octop 侧不参与权限判断。

配置文件 ``kb-search.json``（与本文件同目录，改动即时生效，无需重启）::

    {
      "base_url": "http://127.0.0.1:8125",
      "request_timeout_seconds": 30,
      "default_top_k": 8,
      "max_content_chars": 800,
      "default_account": {"username": "svc-octop", "password": "..."},
      "users": {"test2": {"username": "u1", "password": "..."}},
      "roles": {"学校教务处": {"username": "octop-jwc", "password": "..."}}
    }

身份解析顺序：octop **角色名** → ``roles`` 映射（角色对应 kb 部门账户）→
``users`` 映射（按用户名）→ ``default_account`` 共享服务账户兜底。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx
from langgraph.config import get_config
from octop_harness.plugins import PluginContext

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "kb-search.json"
_TOKEN_REFRESH_MARGIN_S = 60.0
_MAX_TOP_K = 20
_MIN_TOP_K = 1
_MAX_RESULTS_SHOWN = 10

# key = f"{base_url}|{kb_username}" -> (token, expires_at_epoch_seconds)
_tokens: dict[str, tuple[str, float]] = {}
_tokens_lock = asyncio.Lock()

# octop username/id -> (role_name, expires_at_epoch_seconds)
_role_cache: dict[str, tuple[str, float]] = {}
_ROLE_CACHE_TTL_S = 30.0


def _load_config() -> dict[str, Any]:
    """Read plugin config at call time so edits apply without restart."""
    try:
        cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("kb-search: cannot read %s: %s", _CONFIG_PATH, exc)
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _current_octop_user() -> str:
    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return ""
    value = configurable.get("user")
    return str(value).strip() if value else ""


def _octop_db_path() -> Path | None:
    """Locate octop.db (OCTOP_HOME env, else two levels above plugins/<id>/)."""
    home = os.environ.get("OCTOP_HOME")
    candidates = []
    if home:
        candidates.append(Path(home) / "octop.db")
    candidates.append(Path(__file__).resolve().parent.parent.parent / "octop.db")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _octop_user_role(user_ref: str) -> str:
    """Current octop user's role name (e.g. 学校教务处), cached briefly.

    Read-only lookup in octop.db; falls back to the raw role id when
    ``role_name`` is empty (e.g. the built-in admin role).
    """
    if not user_ref:
        return ""
    now = time.time()
    hit = _role_cache.get(user_ref)
    if hit and hit[1] > now:
        return hit[0]
    role = ""
    db_path = _octop_db_path()
    if db_path is not None:
        try:
            con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=2)
            try:
                row = con.execute(
                    "SELECT role_name, role FROM users WHERE username=? OR CAST(id AS TEXT)=?",
                    (user_ref, user_ref),
                ).fetchone()
            finally:
                con.close()
            if row:
                role = str(row[0] or row[1] or "").strip()
        except (sqlite3.Error, OSError) as exc:
            logger.warning("kb-search: cannot read octop role for %s: %s", user_ref, exc)
    _role_cache[user_ref] = (role, now + _ROLE_CACHE_TTL_S)
    return role


def _lookup_entry(mapping: Any, key: str) -> Any:
    """Case-insensitive dict lookup returning a valid account entry or None."""
    if not isinstance(mapping, dict) or not key:
        return None
    entry = mapping.get(key)
    if entry is None:
        for map_key, value in mapping.items():
            if str(map_key).strip().lower() == key.strip().lower():
                entry = value
                break
    if isinstance(entry, dict) and entry.get("username") and entry.get("password"):
        return entry
    return None


def _resolve_account(cfg: dict[str, Any]) -> tuple[str, str, str]:
    """Return ``(octop_user, kb_username, kb_password)`` for this turn.

    Resolution order: octop role → per-user map → default (shared) account.
    """
    octop_user = _current_octop_user()

    role_entry = _lookup_entry(cfg.get("roles"), _octop_user_role(octop_user))
    if role_entry is not None:
        return octop_user, str(role_entry["username"]), str(role_entry["password"])

    users = cfg.get("users")
    user_entry = _lookup_entry(users, octop_user)
    if user_entry is not None:
        return octop_user, str(user_entry["username"]), str(user_entry["password"])

    default_account = cfg.get("default_account")
    if isinstance(default_account, dict) and default_account.get("username"):
        return (
            octop_user,
            str(default_account.get("username", "")),
            str(default_account.get("password", "")),
        )
    return octop_user, "", ""


def _base_url(cfg: dict[str, Any]) -> str:
    return str(cfg.get("base_url") or "").strip().rstrip("/")


def _timeout(cfg: dict[str, Any]) -> float:
    try:
        return float(cfg.get("request_timeout_seconds") or 30)
    except (TypeError, ValueError):
        return 30.0


async def _login(cfg: dict[str, Any], client: httpx.AsyncClient, username: str, password: str) -> str:
    resp = await client.post(
        f"{_base_url(cfg)}/auth/login",
        json={"username": username, "password": password},
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != 0:
        raise RuntimeError(str(payload.get("message") or "login failed"))
    data = payload.get("data") or {}
    token = str(data.get("token") or "").strip()
    if not token:
        raise RuntimeError("kb-service returned an empty token")
    try:
        expires_in = float(data.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600.0
    key = f"{_base_url(cfg)}|{username}"
    _tokens[key] = (token, time.time() + max(expires_in, _TOKEN_REFRESH_MARGIN_S * 2))
    return token


async def _get_token(cfg: dict[str, Any], client: httpx.AsyncClient, *, force: bool = False) -> str:
    _octop_user, username, password = _resolve_account(cfg)
    if not username or not password:
        raise RuntimeError(
            "kb-search is not configured: set default_account (or per-user users map) in kb-search.json"
        )
    key = f"{_base_url(cfg)}|{username}"
    async with _tokens_lock:
        if not force:
            hit = _tokens.get(key)
            if hit and hit[1] - _TOKEN_REFRESH_MARGIN_S > time.time():
                return hit[0]
        _tokens.pop(key, None)
        return await _login(cfg, client, username, password)


async def _search_once(
    cfg: dict[str, Any],
    client: httpx.AsyncClient,
    token: str,
    query: str,
    top_k: int,
) -> dict[str, Any]:
    resp = await client.get(
        f"{_base_url(cfg)}/search",
        params={"q": query, "top_k": top_k},
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code == 401:
        raise PermissionError("token expired")
    resp.raise_for_status()
    return resp.json()


async def search_knowledge_base(query: str, top_k: int = 8) -> str:
    """检索企业知识库，返回按账户权限过滤、带页码/锚点溯源的知识片段。

    Args:
        query: 聚焦的检索关键词串（关键术语/名称/同义词），不要写成口语化问题。
        top_k: 返回片段数，1-20，默认 8。
    """
    q = (query or "").strip()
    if not q:
        return "检索失败：query 不能为空。"
    try:
        n = max(_MIN_TOP_K, min(int(top_k), _MAX_TOP_K))
    except (TypeError, ValueError):
        n = 8

    cfg = _load_config()
    if not _base_url(cfg):
        return "检索失败：kb-search 插件未配置 base_url（kb-search.json）。"
    try:
        async with httpx.AsyncClient(timeout=_timeout(cfg)) as client:
            token = await _get_token(cfg, client)
            try:
                payload = await _search_once(cfg, client, token, q, n)
            except PermissionError:
                payload = await _search_once(
                    cfg, client, await _get_token(cfg, client, force=True), q, n
                )
    except httpx.HTTPError as exc:
        logger.warning("kb-search: request failed: %s", exc)
        return f"检索失败：知识库服务暂不可用（{exc.__class__.__name__}）。"
    except RuntimeError as exc:
        logger.warning("kb-search: auth/config error: %s", exc)
        return f"检索失败：{exc}"

    if payload.get("code") != 0:
        return f"检索失败：知识库返回错误（{payload.get('message') or payload.get('code')}）。"
    data = payload.get("data") or {}
    results = data.get("results") or []
    if not isinstance(results, list):
        results = []
    if not results:
        return f"知识库中未找到与「{q}」相关的内容。"

    max_chars = cfg.get("max_content_chars")
    try:
        max_chars = max(200, int(max_chars)) if max_chars else 800
    except (TypeError, ValueError):
        max_chars = 800

    lines: list[str] = []
    degraded = bool(data.get("degraded"))
    if degraded:
        lines.append("注意：知识库部分检索引擎降级，结果可能不全。")
    lines.append(f"知识库检索「{q}」命中 {len(results[:_MAX_RESULTS_SHOWN])} 条：")
    for index, item in enumerate(results[:_MAX_RESULTS_SHOWN], start=1):
        if not isinstance(item, dict):
            continue
        doc = str(item.get("doc_name") or item.get("doc_id") or "未知文档").strip()
        page = item.get("page")
        anchor = str(item.get("anchor") or "").strip()
        location = f" 第{page}页" if isinstance(page, int) and page > 0 else ""
        if not location and anchor:
            location = f" 锚点「{anchor}」"
        score = item.get("score")
        source = str(item.get("source_type") or "").strip()
        meta_bits = [
            bit for bit in (f"相关度 {score}" if score is not None else "", source) if bit
        ]
        meta = f"（{'，'.join(meta_bits)}）" if meta_bits else ""
        content = str(item.get("content") or "").strip()
        if len(content) > max_chars:
            content = content[: max_chars - 1].rstrip() + "…"
        lines.append(f"\n[{index}] 《{doc}》{location}{meta}\n{content}")
    lines.append(
        "\n回答时请标注来源（文档名与页码/锚点）；以上片段之外的内容不要伪造成知识库引用。"
    )
    return "\n".join(lines)


def setup(ctx: PluginContext) -> None:
    ctx.tool(
        "search_knowledge_base",
        search_knowledge_base,
        description=(
            "Search the enterprise knowledge base (kb-service dual engine: semantic "
            "vector + PageIndex page-level retrieval, permission-filtered by account). "
            "Use this whenever the question may be answered by internal documents "
            "(规范/方案/制度/资料). Write `query` as focused keywords, not a chat "
            "reply. Results carry document name, page/anchor citations — cite them "
            "in the answer and do not invent sources."
        ),
    )
