"""User group catalog."""

from __future__ import annotations

from dataclasses import dataclass

from octop.infra.db.pool import DatabasePool
from octop.infra.db.repos._base import DbRow, insert_returning_id, map_rows, now_ts
from octop.infra.utils.ulid import new_short_id


@dataclass(frozen=True)
class UserGroupRow:
    id: int
    user_group_id: str
    name: str
    description: str | None
    enabled: bool
    avatar_icon: str | None
    created_at: int
    updated_at: int
    sync_role_id: str | None = None

    @classmethod
    def from_row(cls, row: DbRow) -> UserGroupRow:
        keys = row.keys()
        raw_description = row["description"]
        description = str(raw_description).strip() if isinstance(raw_description, str) else ""
        raw_icon = row["avatar_icon"] if "avatar_icon" in keys else None
        icon = str(raw_icon).strip() if isinstance(raw_icon, str) else ""
        enabled = True
        if "enabled" in keys and row["enabled"] is not None:
            enabled = bool(int(row["enabled"]))
        sync_role_id = row["sync_role_id"] if "sync_role_id" in keys else None
        return cls(
            id=int(row["id"]),
            user_group_id=str(row["user_group_id"]),
            name=str(row["name"]),
            description=description or None,
            enabled=enabled,
            avatar_icon=icon or None,
            created_at=int(row["created_at"]),
            updated_at=int(row["updated_at"]),
            sync_role_id=str(sync_role_id) if sync_role_id else None,
        )


class UserGroupRepo:
    def __init__(self, db: DatabasePool) -> None:
        self._db = db

    def list_all(self) -> list[UserGroupRow]:
        with self._db.connect() as conn:
            rows = conn.execute("SELECT * FROM user_group ORDER BY name").fetchall()
        return map_rows(rows, UserGroupRow)

    def get(self, user_group_id: str) -> UserGroupRow | None:
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_group WHERE user_group_id = ?",
                (user_group_id,),
            ).fetchone()
        return UserGroupRow.from_row(row) if row else None

    def get_by_name(self, name: str) -> UserGroupRow | None:
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_group WHERE name = ?",
                (name,),
            ).fetchone()
        return UserGroupRow.from_row(row) if row else None

    def create(
        self,
        *,
        name: str,
        description: str | None,
        enabled: bool = True,
        avatar_icon: str | None = None,
        user_group_id: str | None = None,
        sync_role_id: str | None = None,
    ) -> UserGroupRow:
        ts = now_ts()
        if user_group_id:
            group_id = user_group_id
        else:
            group_id = ""
            for _ in range(8):
                candidate = new_short_id()
                if self.get(candidate) is None:
                    group_id = candidate
                    break
            if not group_id:
                raise RuntimeError("failed to allocate unique user_group_id")
        with self._db.transaction() as conn:
            insert_returning_id(
                conn,
                "INSERT INTO user_group("
                "user_group_id, name, description, enabled, avatar_icon, sync_role_id,"
                " created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    group_id,
                    name,
                    description,
                    int(enabled),
                    avatar_icon,
                    sync_role_id,
                    ts,
                    ts,
                ),
            )
        row = self.get(group_id)
        if row is None:
            raise RuntimeError(f"user group insert failed: {group_id}")
        return row

    def get_by_sync_role_id(self, role_id: str) -> UserGroupRow | None:
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_group WHERE sync_role_id = ?",
                (role_id,),
            ).fetchone()
        return UserGroupRow.from_row(row) if row else None

    def clear_sync_role(self, user_group_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE user_group SET sync_role_id = NULL, updated_at = ? WHERE user_group_id = ?",
                (now_ts(), user_group_id),
            )

    def adopt_as_sync_role(self, user_group_id: str, role_id: str) -> None:
        """Attach an existing (manually created) group to a role template."""
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE user_group SET sync_role_id = ?, updated_at = ? WHERE user_group_id = ?",
                (role_id, now_ts(), user_group_id),
            )

    def update(
        self,
        user_group_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        update_description: bool = False,
        enabled: bool | None = None,
        avatar_icon: str | None = None,
        update_avatar_icon: bool = False,
    ) -> None:
        fields: list[str] = []
        params: list[object] = []
        if name is not None:
            fields.append("name = ?")
            params.append(name)
        if update_description:
            fields.append("description = ?")
            params.append(description)
        if enabled is not None:
            fields.append("enabled = ?")
            params.append(int(enabled))
        if update_avatar_icon:
            fields.append("avatar_icon = ?")
            params.append(avatar_icon)
        if not fields:
            return
        fields.append("updated_at = ?")
        params.append(now_ts())
        params.append(user_group_id)
        with self._db.transaction() as conn:
            conn.execute(
                f"UPDATE user_group SET {', '.join(fields)} WHERE user_group_id = ?",
                tuple(params),
            )

    def delete(self, user_group_id: str) -> bool:
        with self._db.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM user_group WHERE user_group_id = ?",
                (user_group_id,),
            )
            return int(cur.rowcount or 0) > 0
