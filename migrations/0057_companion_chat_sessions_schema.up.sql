-- =============================================================================
-- chora-ai-kernel-orchestrator : 0057_companion_chat_sessions_schema.up.sql
-- Database  : chora_ai_kernel
-- ADR       : ADR-254 D6 (companion_chat sessions live in Postgres, never
--             memstore; owner ruling "Store" 2026-08-22), relayed as a binding
--             coordinator instruction 2026-08-22 16:33Z (WP-A chat conversion).
--
-- What this does
--   Creates the dedicated schema the companion_chat agent's ADK
--   session/database service uses, and grants the agent's role the right to
--   create its own tables inside it. The library auto-migrates its three
--   session tables on first boot over a connection whose
--   search_path=companion_chat_sessions; the kennel creates NO table here and
--   owns nothing inside the schema (library-owned tables, no RLS, no tenant
--   column; provisional ruling pending the owner MCQ at G1a).
--
-- Tenant isolation (provisional, coordinator ruling 2026-08-22)
--   The ADK app_name is keyed per tenant, "companion_chat:{tenant_id}";
--   user_id = gcid; session_id = conversation_id. The agent process is the
--   sole accessor of the schema (it runs on the kennel's app_rw DSN for the
--   build window; a schema-scoped role is a post-G4 follow-up). If the owner
--   rules for a tenant column + policy, a later migration shapes it.
--
-- PII
--   Chat content is learner PII. The chora_ai_kernel PII_Closure_Map
--   (services/chora-ai-kernel-orchestrator/config/PII_Closure_Map.yaml) carries
--   companion_chat_sessions.* keyed by gcid (pseudonymise / crypto-shred path,
--   ADR-186); it was added in the same commit as this file.
--
-- Grants
--   9999_grant_app_roles.sql covers schema public only (ALL ... IN SCHEMA
--   public + default privileges for the migrate role). This schema's tables
--   are created by app_rw itself, so app_rw owns them and needs no further
--   table grant; the schema-level USAGE, CREATE is the one grant that must
--   exist before the agent boots, and it is idempotent, so it lives here
--   (a targeted run of this file alone is complete).
--
-- Idempotency: CREATE SCHEMA IF NOT EXISTS + GRANT (a no-op on re-apply).
-- HARD RULE: cross-database queries forbidden; chora_ai_kernel-local only.
-- =============================================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS companion_chat_sessions;

GRANT USAGE, CREATE ON SCHEMA companion_chat_sessions TO chora_ai_kernel_app_rw;

COMMENT ON SCHEMA companion_chat_sessions IS
  'ADR-254 D6: companion_chat ADK session tables (library-owned, auto-migrated by the agent on boot; app_name companion_chat:{tenant_id}, user_id gcid, session_id conversation_id). Learner PII: see config/PII_Closure_Map.yaml.';

COMMIT;
