-- Schema v26: user groups can be a projection of a role template
-- (department visibility). sync_role_id is NULL for manually managed groups.

ALTER TABLE user_group ADD COLUMN IF NOT EXISTS sync_role_id TEXT REFERENCES user_role(user_role_id);

UPDATE _schema_version SET version = 26;
