-- Schema v26: user groups can be a projection of a role template
-- (department visibility). sync_role_id is NULL for manually managed groups.
-- SQLite applies this through _ensure_user_group_profile_schema so a replay
-- onto a database that already has the column does not fail.

ALTER TABLE user_group ADD COLUMN sync_role_id TEXT REFERENCES user_role(user_role_id);

UPDATE _schema_version SET version = 26;
