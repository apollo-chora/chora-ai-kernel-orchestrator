-- =============================================================================
-- chora-ai-kernel-orchestrator : 0004_inbox_idempotency.sql
--
-- Inbox-side dedupe table for the orchestrator's Pub/Sub subscriber. Mirrors
-- the libs/chora-go-common/idempotent template at
-- libs/chora-go-common/idempotent/sql_fixtures/idempotency_keys.up.sql.
--
-- Why we need this:
--   * The qgen_crew_subscriber (and future per-crew subscribers in this
--     "kennel" orchestrator) handle at-least-once Pub/Sub delivery.
--   * Per [[feedback-d6-resilience-first-class]] + .claude/skills/data-consistency,
--     every subscriber MUST de-duplicate by envelope.event_id BEFORE applying
--     the event side effect.
--   * The LangGraph PostgresSaver naturally idempotent-collapses on
--     thread_id, but a graph re-run from checkpoint is wasted CPU when the
--     same event is redelivered — the inbox short-circuits.
--
-- TTL purge: handled by the Python `InboxIdempotencyStore.cleanup_expired()`
-- method, invoked on a schedule via a separate Cloud Run Job (or pg_cron
-- once provisioned). The cleanup task issues a parameterised DELETE bound
-- to (now()) — operational table, exempt from ddd-enforcement.md soft-delete.
--
-- Database: chora_ai_kernel.
-- Date    : 2026-05-17.
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS idempotency_keys (
    key          TEXT        PRIMARY KEY,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ttl_at       TIMESTAMPTZ NOT NULL,
    -- Optional opaque hash of the upstream operation result. Currently
    -- unused — left for future "prove cached outcome" parity with
    -- libs/chora-go-common/idempotent.PostgresStore.
    result_hash  TEXT
);

CREATE INDEX IF NOT EXISTS idempotency_keys_ttl_at_idx
    ON idempotency_keys (ttl_at);

COMMIT;
