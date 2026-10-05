-- =============================================================================
-- chora-ai-kernel-orchestrator : 0005_grant_outbox_explicit.sql
--
-- Domain        : AI Kernel (supporting/platform)
-- Database      : chora_ai_kernel
-- Author        : BE-AI-Assist Claude (post-Gates-#7+#8 rollout)
-- Date          : 2026-05-17
-- Architecture  : Architecture Review locked 2026-05-07 (Tier 2 D7)
--
-- Purpose:
--   Explicit grant on the outbox + inbox + idempotency tables in
--   chora_ai_kernel. The 9999_grant_app_roles.sql migration is marked
--   applied in the runner tracker, but the GRANT ON ALL TABLES IN
--   SCHEMA public was issued at a time when ai_kernel_outbox_events did
--   not yet exist (the table was created by a later 0003_outbox.sql
--   apply), and the ALTER DEFAULT PRIVILEGES clause only fires for
--   tables created by chora_ai_kernel_migrate — but 0003 appears to
--   have been applied by a different role (postgres root in some bring-
--   up paths). Net effect: ai_kernel_outbox_events + ai_kernel_outbox_dead_letters
--   + ai_kernel_outbox_dispatch_checkpoints + idempotency_keys are
--   owned by a role other than migrate, and the app_rw role can't
--   touch them. Surfaced 2026-05-17 evening when the outbox dispatcher
--   tight loop spammed psycopg.errors.InsufficientPrivilege after the
--   AsyncConnection coroutine fix (91e7cfc3) unblocked actual DB calls.
--
-- Idempotency:
--   - GRANT is idempotent by design.
--
-- Resilience (per feedback_resilience_priority memory):
--   - Single transactional unit; mid-apply pod death = idempotent retry.
-- =============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON ai_kernel_outbox_events
  TO chora_ai_kernel_app_rw;

GRANT SELECT ON ai_kernel_outbox_events
  TO chora_ai_kernel_app_ro;

GRANT SELECT, INSERT, UPDATE, DELETE ON ai_kernel_outbox_dead_letters
  TO chora_ai_kernel_app_rw;

GRANT SELECT ON ai_kernel_outbox_dead_letters
  TO chora_ai_kernel_app_ro;

GRANT SELECT, INSERT, UPDATE, DELETE ON ai_kernel_outbox_dispatch_checkpoints
  TO chora_ai_kernel_app_rw;

GRANT SELECT ON ai_kernel_outbox_dispatch_checkpoints
  TO chora_ai_kernel_app_ro;

-- 0004 idempotency_keys (inbox dedupe — same backfill gap)
GRANT SELECT, INSERT, UPDATE, DELETE ON idempotency_keys
  TO chora_ai_kernel_app_rw;

GRANT SELECT ON idempotency_keys
  TO chora_ai_kernel_app_ro;

-- =============================================================================
-- VERIFICATION (manual; after apply):
--
--   SET ROLE chora_ai_kernel_app_rw;
--   SELECT count(*) FROM ai_kernel_outbox_events;   -- should return 0+ rows
--   SELECT count(*) FROM idempotency_keys;          -- should return 0+ rows
--   RESET ROLE;
-- =============================================================================
