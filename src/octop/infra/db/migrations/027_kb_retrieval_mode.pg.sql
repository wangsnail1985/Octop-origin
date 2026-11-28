-- Schema v27: knowledge bases pick their retrieval cost level.
-- 'basic' (default): vector + exact-term + keyword tree search, zero query-time LLM.
-- 'advanced': additionally uses the small model for tree node picking and
-- graph entity linking when one is configured instance-wide.

ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS retrieval_mode TEXT NOT NULL DEFAULT 'basic';

UPDATE _schema_version SET version = 27;
