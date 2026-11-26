"""Role-template driven user groups (「本部门可见」 support).

Each custom role template projects onto one user group with the same name
(``user_group.sync_role_id``). Group membership follows the role: users gain
and lose membership automatically as their role changes. Knowledge bases can
then be shared to the owner's department by resolving the owner's role group.

Built-in roles (``admin`` / ``user``) never project onto a group.
"""

from __future__ import annotations

from octop.infra.db.repos.user_groups import UserGroupRow
from octop.infra.db.repos.user_roles import BUILTIN_USER_ROLE_IDS, UserRoleRepo
from octop.infra.db.services import RepoBundle
from octop.infra.errors import ErrorCode, OctopError


def _is_builtin_role(role_id: str | None) -> bool:
    return role_id is None or str(role_id) in BUILTIN_USER_ROLE_IDS


def ensure_role_group(repos: RepoBundle, *, role_id: str, role_name: str) -> UserGroupRow:
    """Create or rename the user group projected from a role template.

    Idempotent: an existing synced group is renamed in place. A pre-existing
    manual group with the same name is adopted (attached to the role) instead
    of duplicated, so departments created by hand keep their members.
    """
    repo = repos.user_group_repo
    existing = repo.get_by_sync_role_id(role_id)
    if existing is not None:
        if existing.name != role_name:
            repo.update(existing.user_group_id, name=role_name)
        row = repo.get(existing.user_group_id)
        assert row is not None
        return row
    conflict = repo.get_by_name(role_name)
    if conflict is not None:
        repo.adopt_as_sync_role(conflict.user_group_id, role_id)
        row = repo.get(conflict.user_group_id)
        assert row is not None
        return row
    return repo.create(
        name=role_name,
        description=f"Synced from role {role_id}",
        sync_role_id=role_id,
    )


def disable_role_group(repos: RepoBundle, *, role_id: str) -> None:
    """Detach and disable the synced group when its role template is deleted.

    The group and its resource grants are kept (disabled groups no longer grant
    access), so shared resources stay scoped instead of falling back to
    "everyone" once their grants disappear. Clearing ``sync_role_id`` releases
    the FK reference so the role row itself can be deleted.
    """
    if _is_builtin_role(role_id):
        return
    repo = repos.user_group_repo
    existing = repo.get_by_sync_role_id(role_id)
    if existing is None:
        return
    repo.clear_sync_role(existing.user_group_id)
    if existing.enabled:
        repo.update(existing.user_group_id, enabled=False)


def _synced_group_for_role(repos: RepoBundle, *, role_id: str) -> UserGroupRow | None:
    repo = repos.user_group_repo
    group = repo.get_by_sync_role_id(role_id)
    if group is None:
        # Lazily project groups for roles that predate the sync (existing DBs).
        role = UserRoleRepo(repos.db).get(role_id)
        if role is None:
            return None
        group = ensure_role_group(repos, role_id=role_id, role_name=role.user_role_name)
    return group


def sync_user_membership(repos: RepoBundle, *, user_id: int, role_id: str | None) -> None:
    """Align one user's synced-group memberships with their role."""
    if _is_builtin_role(role_id):
        role_id = None
    repo = repos.user_group_repo
    members = repos.user_group_member_repo
    for group_id in sorted(members.group_ids_for_user(user_id)):
        row = repo.get(group_id)
        if row is not None and row.sync_role_id is not None and row.sync_role_id != role_id:
            members.remove(group_id, user_id)
    if role_id is None:
        return
    target = _synced_group_for_role(repos, role_id=role_id)
    if target is None:
        return
    members.add(target.user_group_id, user_id)  # idempotent upsert


def resolve_department_group(repos: RepoBundle, *, role_id: str | None) -> UserGroupRow:
    """Return the group a knowledge base owner's role projects onto.

    Raises when the role is built-in or has no synced group, so callers can
    surface a clear 400 instead of silently widening the audience.
    """
    if _is_builtin_role(role_id):
        raise OctopError(
            ErrorCode.FORBIDDEN,
            "your role has no department group; ask an admin to assign a custom role",
            status=400,
        )
    group = _synced_group_for_role(repos, role_id=str(role_id))
    if group is None or not group.enabled:
        raise OctopError(
            ErrorCode.FORBIDDEN,
            "your role has no department group; ask an admin to assign a custom role",
            status=400,
        )
    return group
