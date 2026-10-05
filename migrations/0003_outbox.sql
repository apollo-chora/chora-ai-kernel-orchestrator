-- =============================================================================
-- chora-ai-kernel-orchestrator : 0003_outbox.sql
--
-- Adds the transactional outbox + DLQ tables for the AI Kernel orchestrator's
-- emitted events, mirroring the closure-saga reference implementation in
-- services/chora-closure-orchestrator/migrations/0002_outbox.sql.
--
-- Per `feedback_d6_resilience_first_class` B.6.2 sub-deliverable (a) and the
-- expanded D6.3 multi-tenant + multi-workflow chaos scope (user directive
-- 2026-05-12), this migration installs a producer-side durable-emission
-- table for the orchestrator's chora.ai_kernel.* event stream.
--
-- Domain  : AI Kernel / LangGraph Orchestrator (this service owns these tables)
-- Database: chora_ai_kernel (SAME database as closure_outbox_events + the
--           LangGraph PostgresSaver checkpoints — the orchestrator + closure
--           saga both live here).
-- Date    : 2026-05-12
--
-- NAMING — tables are prefixed ``ai_kernel_outbox_*`` (NOT ``closure_outbox_*``)
-- to avoid collision with the existing closure-saga outbox in the same DB.
--
-- HARD INVARIANTS
--   * Outbox rows live in the SAME database as LangGraph PostgresSaver
--     checkpoints (chora_ai_kernel). When an orchestrator node emits an
--     event, the outbox row write + the LangGraph checkpoint write are both
--     persisted to this DB; the dispatcher publishes to Cloud Pub/Sub.
--   * tenant_id is captured as a top-level column for D6.3 multi-tenant
--     isolation indexing + RLS — production orchestrator runs may carry
--     events from many tenants through the same Pub/Sub pipe.
--   * idempotency_key + envelope mandatory per CLAUDE.md cross-cutting rule.
--   * Cross-DB queries remain forbidden — domain subscribers read events
--     from Pub/Sub, never from this table.
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS ai_kernel_outbox_events (
    id              TEXT        PRIMARY KEY,                 -- UUIDv7
    workflow_id     UUID        NOT NULL,                    -- aggregate root (workflow / crew run)
    tenant_id       UUID        NOT NULL,                    -- D6.3 isolation
    gcid            UUID        NOT NULL,                    -- subject
    event_type      TEXT        NOT NULL,                    -- e.g., 'ai_kernel.invocation.invoked'
    topic           TEXT        NOT NULL,                    -- 'chora.ai_kernel.invocation.invoked.v1'
    payload         BYTEA       NOT NULL,                    -- Protobuf bytes (POC: JSON)
    envelope        JSONB       NOT NULL,                    -- full envelope: event_id,
                                                             -- idempotency_key, traceparent,
                                                             -- tracestate, source_project,
                                                             -- source_service, schema_version
    idempotency_key TEXT        NOT NULL,                    -- dedupe key (extracted from envelope)
    occurred_at     TIMESTAMPTZ NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','published','failed','deadlettered')),
    retry_count     INT         NOT NULL DEFAULT 0,
    last_error      TEXT        NOT NULL DEFAULT '',
    last_attempt_at TIMESTAMPTZ,
    published_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Dispatcher poll query — find next pending event by occurred_at.
CREATE INDEX IF NOT EXISTS ai_kernel_outbox_events_pending_idx
    ON ai_kernel_outbox_events (occurred_at ASC) WHERE status = 'pending';

-- D6.3 multi-tenant isolation lookup: dispatcher MAY filter per-tenant.
CREATE INDEX IF NOT EXISTS ai_kernel_outbox_events_tenant_idx
    ON ai_kernel_outbox_events (tenant_id, status, occurred_at);

-- Per-workflow event lookup (debugging / replay).
CREATE INDEX IF NOT EXISTS ai_kernel_outbox_events_workflow_idx
    ON ai_kernel_outbox_events (workflow_id, occurred_at);

-- Per-topic dispatcher worker mode.
CREATE INDEX IF NOT EXISTS ai_kernel_outbox_events_topic_idx
    ON ai_kernel_outbox_events (topic, status);

-- Idempotency dedupe — events emitted twice (e.g., workflow resume re-emits)
-- collapse on this key. Unique index because dedupe MUST be exact.
CREATE UNIQUE INDEX IF NOT EXISTS ai_kernel_outbox_events_idempotency_idx
    ON ai_kernel_outbox_events (idempotency_key);

-- Dispatcher checkpoint — at most one row per (worker_id, topic).
-- Tracks the last successfully published event so the worker can resume
-- after a pod-death without re-publishing already-acked events.
CREATE TABLE IF NOT EXISTS ai_kernel_outbox_dispatch_checkpoints (
    worker_id                  TEXT        NOT NULL,
    topic                      TEXT        NOT NULL,
    last_processed_outbox_id   TEXT        NOT NULL,
    last_processed_occurred_at TIMESTAMPTZ NOT NULL,
    updated_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (worker_id, topic)
);

CREATE INDEX IF NOT EXISTS ai_kernel_outbox_dispatch_checkpoints_topic_idx
    ON ai_kernel_outbox_dispatch_checkpoints (topic, updated_at DESC);

-- DLQ pointer — events that exceed max_retries land here. resolved_at
-- is set when an operator manually replays via the orchestrator runbook
-- (per B.6.2 sub-deliverable d "Orchestrator-side DLQ awareness").
CREATE TABLE IF NOT EXISTS ai_kernel_outbox_dead_letters (
    outbox_event_id   TEXT        PRIMARY KEY REFERENCES ai_kernel_outbox_events(id),
    failure_reason    TEXT        NOT NULL,
    attempt_count     INT         NOT NULL,
    worker_id         TEXT        NOT NULL,
    deadlettered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at       TIMESTAMPTZ,
    resolution_note   TEXT        NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS ai_kernel_outbox_dead_letters_unresolved_idx
    ON ai_kernel_outbox_dead_letters (deadlettered_at DESC) WHERE resolved_at IS NULL;

-- Multi-tenant RLS — same pattern as the closure outbox (and 0001_initial.sql).
-- POC connections do NOT set app.current_tenant (the AI Kernel orchestrator
-- writes events across many tenants per session); production wires this
-- via per-connection GUC. RLS policy remains permissive when the GUC is
-- unset to keep the POC dispatcher functional; per ADR-141 tenant
-- isolation is also enforced at the application + envelope layer.
ALTER TABLE ai_kernel_outbox_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY ai_kernel_outbox_events_tenant_isolation ON ai_kernel_outbox_events
    USING (
        current_setting('app.current_tenant', TRUE) IS NULL
        OR current_setting('app.current_tenant', TRUE) = ''
        OR tenant_id::TEXT = current_setting('app.current_tenant', TRUE)
    );

COMMIT;
