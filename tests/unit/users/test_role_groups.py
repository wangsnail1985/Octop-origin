"""Role → user-group sync and department-scoped knowledge base sharing."""

from __future__ import annotations

from pathlib import Path

import pytest

from octop.infra.db.migrate import run_migrations
from octop.infra.db.pool import SqlitePool
from octop.infra.db.repos.user_roles import UserRoleRepo
from octop.infra.db.services import RepoBundle
from octop.infra.errors import ErrorCode, OctopError
from octop.infra.users.groups import RESOURCE_KNOWLEDGE_BASE, GroupAccess
from octop.infra.users.role_groups import (
    _synced_group_for_role,
    disable_role_group,
    ensure_role_group,
    resolve_department_group,
    sync_user_membership,
)


@pytest.fixture
def repos(tmp_path: Path) -> RepoBundle:
    pool = SqlitePool(tmp_path / "octop.db")
    run_migrations(pool)
    return RepoBundle.from_pool(pool)


def _role(repos: RepoBundle, role_id: str, name: str) -> str:
    UserRoleRepo(repos.db).create(
        user_role_name=name,
        permissions=["knowledge_bases"],
        policies=[],
        user_role_id=role_id,
    )
    return role_id


def _user(repos: RepoBundle, username: str, role: str) -> int:
    return repos.user_repo.create(username=username, role=role)


def _member_group_ids(repos: RepoBundle, user_id: int) -> set[str]:
    return set(repos.user_group_member_repo.group_ids_for_user(user_id))


def test_ensure_role_group_creates_named_sync_group(repos: RepoBundle) -> None:
    _role(repos, "jwc", "教务处")
    group = ensure_role_group(repos, role_id="jwc", role_name="教务处")
    assert group.sync_role_id == "jwc"
    assert group.name == "教务处"
    assert group.enabled is True


def test_ensure_role_group_renames_in_place(repos: RepoBundle) -> None:
    _role(repos, "jwc", "教务处")
    ensure_role_group(repos, role_id="jwc", role_name="教务处")
    renamed = ensure_role_group(repos, role_id="jwc", role_name="教务处（新）")
    assert renamed.name == "教务处（新）"
    assert repos.user_group_repo.get_by_sync_role_id("jwc") is not None
    assert repos.user_group_repo.get_by_name("教务处") is None


def test_ensure_role_group_adopts_manual_group_with_same_name(repos: RepoBundle) -> None:
    _role(repos, "jwc", "教务处")
    manual = repos.user_group_repo.create(name="教务处", description=None)
    member_id = _user(repos, "maomao", "user")
    repos.user_group_member_repo.add(manual.user_group_id, member_id)

    adopted = ensure_role_group(repos, role_id="jwc", role_name="教务处")

    # The hand-made group is attached to the role, keeping its id and members.
    assert adopted.user_group_id == manual.user_group_id
    assert adopted.sync_role_id == "jwc"
    assert member_id in repos.user_group_member_repo.user_ids_for_group(
        manual.user_group_id
    )
    assert repos.user_group_repo.get_by_sync_role_id("jwc") is not None


def test_sync_user_membership_follows_role(repos: RepoBundle) -> None:
    _role(repos, "jwc", "教务处")
    _role(repos, "fin", "财务处")
    user_id = _user(repos, "test2", "user")

    sync_user_membership(repos, user_id=user_id, role_id="jwc")
    jwc = repos.user_group_repo.get_by_sync_role_id("jwc")
    assert _member_group_ids(repos, user_id) == {jwc.user_group_id}

    # Switching roles moves the user between synced groups.
    sync_user_membership(repos, user_id=user_id, role_id="fin")
    fin = repos.user_group_repo.get_by_sync_role_id("fin")
    assert _member_group_ids(repos, user_id) == {fin.user_group_id}

    # Built-in roles clear synced membership and never join a group.
    sync_user_membership(repos, user_id=user_id, role_id="admin")
    assert _member_group_ids(repos, user_id) == set()


def test_sync_user_membership_keeps_manual_groups(repos: RepoBundle) -> None:
    _role(repos, "jwc", "教务处")
    manual = repos.user_group_repo.create(name="项目组", description=None)
    user_id = _user(repos, "test2", "user")
    repos.user_group_member_repo.add(manual.user_group_id, user_id)

    sync_user_membership(repos, user_id=user_id, role_id="jwc")
    jwc = repos.user_group_repo.get_by_sync_role_id("jwc")
    assert _member_group_ids(repos, user_id) == {jwc.user_group_id, manual.user_group_id}

    sync_user_membership(repos, user_id=user_id, role_id="admin")
    assert _member_group_ids(repos, user_id) == {manual.user_group_id}


def test_resolve_department_group_lazy_projects_existing_roles(repos: RepoBundle) -> None:
    # Role written without ever going through the sync (pre-existing DB row).
    _role(repos, "jwc", "教务处")
    group = resolve_department_group(repos, role_id="jwc")
    assert group.sync_role_id == "jwc"


def test_resolve_department_group_rejects_builtin_and_unknown(repos: RepoBundle) -> None:
    with pytest.raises(OctopError) as admin_exc:
        resolve_department_group(repos, role_id="admin")
    assert admin_exc.value.status == 400
    with pytest.raises(OctopError) as none_exc:
        resolve_department_group(repos, role_id=None)
    assert none_exc.value.status == 400
    with pytest.raises(OctopError) as unknown_exc:
        resolve_department_group(repos, role_id="ghost")
    assert unknown_exc.value.status == 400


def test_lazy_lookup_never_projects_builtin_roles(repos: RepoBundle) -> None:
    # The audience endpoint looks up the caller's group lazily; a built-in
    # role must resolve to "no group" without creating one.
    assert _synced_group_for_role(repos, role_id="admin") is None
    assert _synced_group_for_role(repos, role_id="user") is None
    assert repos.user_group_repo.list_all() == []


def test_department_share_visible_only_to_role_members(repos: RepoBundle) -> None:
    _role(repos, "jwc", "教务处")
    owner_id = _user(repos, "test2", "jwc")
    peer_id = _user(repos, "colleague", "jwc")
    outsider_id = _user(repos, "other", "user")

    sync_user_membership(repos, user_id=owner_id, role_id="jwc")
    sync_user_membership(repos, user_id=peer_id, role_id="jwc")
    group = resolve_department_group(repos, role_id="jwc")

    access = GroupAccess(repos)
    access.set_visibility(RESOURCE_KNOWLEDGE_BASE, "kb1", "groups", [group.user_group_id])

    assert access.allows(
        user_id=owner_id,
        is_admin=False,
        resource_type=RESOURCE_KNOWLEDGE_BASE,
        resource_key="kb1",
        is_owner=True,
    )
    assert access.allows(
        user_id=peer_id,
        is_admin=False,
        resource_type=RESOURCE_KNOWLEDGE_BASE,
        resource_key="kb1",
        is_owner=False,
    )
    assert not access.allows(
        user_id=outsider_id,
        is_admin=False,
        resource_type=RESOURCE_KNOWLEDGE_BASE,
        resource_key="kb1",
        is_owner=False,
    )

    # Public visibility derives to "groups", never back to "all".
    state = access.visibility(RESOURCE_KNOWLEDGE_BASE, "kb1", shared=True)
    assert state == {"visibility": "groups", "group_ids": [group.user_group_id]}


def test_disable_role_group_hides_share_without_widening(repos: RepoBundle) -> None:
    _role(repos, "jwc", "教务处")
    member_id = _user(repos, "test2", "jwc")
    sync_user_membership(repos, user_id=member_id, role_id="jwc")
    group = resolve_department_group(repos, role_id="jwc")

    access = GroupAccess(repos)
    access.set_visibility(RESOURCE_KNOWLEDGE_BASE, "kb1", "groups", [group.user_group_id])

    disable_role_group(repos, role_id="jwc")
    disabled = repos.user_group_repo.get(group.user_group_id)
    assert disabled is not None and disabled.enabled is False
    # The FK reference is released so the role row itself can be deleted.
    assert disabled.sync_role_id is None
    UserRoleRepo(repos.db).delete("jwc")
    assert UserRoleRepo(repos.db).get("jwc") is None

    assert not access.allows(
        user_id=member_id,
        is_admin=False,
        resource_type=RESOURCE_KNOWLEDGE_BASE,
        resource_key="kb1",
        is_owner=False,
    )
    state = access.visibility(RESOURCE_KNOWLEDGE_BASE, "kb1", shared=True)
    assert state["visibility"] == "groups"

    with pytest.raises(OctopError) as excinfo:
        resolve_department_group(repos, role_id="jwc")
    assert excinfo.value.code is ErrorCode.FORBIDDEN
