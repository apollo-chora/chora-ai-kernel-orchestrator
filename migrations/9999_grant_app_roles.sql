-- =============================================================================
-- chora-ai-kernel-orchestrator : 9999_grant_app_roles.sql
--
-- Domain        : AI Kernel (supporting/platform)
-- Database      : chora_ai_kernel
-- Author        : agent A-Migration-Grants (Phyllis Pre-S9 hardening burst)
-- Date          : 2026-05-11
-- Architecture  : Architecture Review locked 2026-05-07 (Tier 2 D7)
--
-- Purpose:
--   Production gap fix — migrations across all 11 services create tables but
--   never GRANT privileges to the `app_rw` / `app_ro` roles. The `migrate`
--   role bypasses RLS and owns the tables (so its own grant works), so the
--   gap was masked during bring-up. This migration restores baseline
--   privileges that application traffic needs.
--
-- Idempotency:
--   - GRANT is idempotent by design — re-applying is a no-op.
--   - ALTER DEFAULT PRIVILEGES is idempotent on the same (role, schema)
--     pair (Postgres deduplicates the default ACL).
--   - This file is the LAST in lex order (9999 prefix) — every prior
--     migration's tables/sequences/functions are picked up by the
--     `ALL ... IN SCHEMA public` clauses, AND every future migration's
--     newly-created objects automatically inherit the default privileges.
--
-- Resilience (per feedback_resilience_priority memory):
--   - Dead-pod resume: this file is a single transactional unit; if the
--     migrate pod dies mid-apply, replacement re-runs idempotently.
--   - Multi-user concurrent: GRANT is atomic per-object; concurrent GRANT
--     statements from multiple migrate sessions are safe (Postgres locks
--     the catalog row briefly).
--   - No bypass of RLS: app_rw / app_ro do NOT have BYPASSRLS attribute;
--     their grants are subject to every per-table RLS policy. Only the
--     migrate role is exempt (it owns the tables and never receives real
--     application traffic).
--
-- Role privileges:
--   app_rw    SELECT, INSERT, UPDATE, DELETE on all tables;
--             USAGE+SELECT+UPDATE on all sequences;
--             EXECUTE on all functions
--   app_ro    SELECT on all tables;
--             USAGE+SELECT on all sequences;
--             EXECUTE on all functions (for SECURITY DEFINER paths)
--   migrate   already has all privileges via OWNER role — no GRANT needed
--
-- HARD RULE: cross-database queries forbidden. This migration only touches
-- chora_ai_kernel-local roles + objects.
-- =============================================================================

-- ----------------------------------------------------------------------------
-- 1) Schema USAGE — both app roles need to "see" the public schema before
--    they can touch any object inside it.
-- ----------------------------------------------------------------------------

GRANT USAGE ON SCHEMA public TO chora_ai_kernel_app_rw, chora_ai_kernel_app_ro;

-- 1b) companion_chat_sessions (migrations 0057 + 0058, ADR-254 D6): the chat
--     agent's ADK session tables. USAGE for the app role, and the table grant
--     re-asserted here so a FULL run keeps parity with 0058 (which carries the
--     same grant so a targeted run is complete on its own). app_ro gets nothing
--     here: chat content is learner PII and no read-only consumer exists.
GRANT USAGE ON SCHEMA companion_chat_sessions TO chora_ai_kernel_app_rw;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA companion_chat_sessions
  TO chora_ai_kernel_app_rw;

-- ----------------------------------------------------------------------------
-- 2) Existing-object grants — covers every table/sequence/function created
--    by migrations 0001..N-1 already applied to live DBs.
-- ----------------------------------------------------------------------------

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public
  TO chora_ai_kernel_app_rw;

GRANT SELECT ON ALL TABLES IN SCHEMA public
  TO chora_ai_kernel_app_ro;

GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA public
  TO chora_ai_kernel_app_rw;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public
  TO chora_ai_kernel_app_ro;

GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public
  TO chora_ai_kernel_app_rw, chora_ai_kernel_app_ro;

-- ----------------------------------------------------------------------------
-- 3) Default privileges for FUTURE objects — so subsequent migrations
--    creating new tables/sequences/functions Just Work without app_rw /
--    app_ro losing access. Tied to the migrate role (the role that runs
--    DDL and therefore creates the objects).
-- ----------------------------------------------------------------------------

ALTER DEFAULT PRIVILEGES FOR ROLE chora_ai_kernel_migrate IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES
  TO chora_ai_kernel_app_rw;

ALTER DEFAULT PRIVILEGES FOR ROLE chora_ai_kernel_migrate IN SCHEMA public
  GRANT SELECT ON TABLES
  TO chora_ai_kernel_app_ro;

ALTER DEFAULT PRIVILEGES FOR ROLE chora_ai_kernel_migrate IN SCHEMA public
  GRANT USAGE, SELECT, UPDATE ON SEQUENCES
  TO chora_ai_kernel_app_rw;

ALTER DEFAULT PRIVILEGES FOR ROLE chora_ai_kernel_migrate IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES
  TO chora_ai_kernel_app_ro;

ALTER DEFAULT PRIVILEGES FOR ROLE chora_ai_kernel_migrate IN SCHEMA public
  GRANT EXECUTE ON FUNCTIONS
  TO chora_ai_kernel_app_rw, chora_ai_kernel_app_ro;

-- =============================================================================
-- VERIFICATION (run manually after apply):
--
--   -- App RW should be able to SELECT/INSERT/UPDATE/DELETE on a tenant-scoped
--   -- table; RLS should still apply.
--   SET ROLE chora_ai_kernel_app_rw;
--   BEGIN;
--     SET LOCAL chora.tenant_id = '00000000-0000-7000-8000-000000000001';
--     SELECT count(*) FROM tenants;   -- should return RLS-filtered count
--   ROLLBACK;
--
--   -- App RO should NOT be able to INSERT.
--   SET ROLE chora_ai_kernel_app_ro;
--   BEGIN;
--     INSERT INTO tenants (id, name) VALUES (gen_random_uuid(), 'should-fail');
--     -- Expected: ERROR: permission denied for table tenants
--   ROLLBACK;
--
--   RESET ROLE;
-- =============================================================================
