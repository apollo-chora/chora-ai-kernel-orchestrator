-- =============================================================================
-- chora-ai-kernel-orchestrator : 0056_agent_dispatch_parks.up.sql
--
-- ADR           : ADR-254 D5 (park ledger + park reaper), ADR-253 D3a
-- Domain        : AI Kernel (supporting/platform)
-- Database      : chora_ai_kernel
-- Date          : 2026-08-22
--
-- Purpose:
--   The park ledger: one row per agent dispatch park, written by
--   TransactionalDispatchSaver in the SAME transaction as the LangGraph park
--   (checkpoint_writes on __interrupt__) and the dispatch row in
--   ai_kernel_outbox_events. It is the only place a park is queryable by age:
--   LangGraph's checkpoint tables carry no application timestamp (they are
--   created lazily by the saver, never by a migration) and the outbox row's
--   occurred_at vanishes under any pruning. Every parked run carries a
--   deadline (deadline_at = parked_at + min(604800s, role_deadline)); the
--   reaper settles FAILED any row whose agent never answered (four arms:
--   request dead-lettered, deadline passed, completion dead-lettered, outbox
--   dead-lettered), and the completion router settles 'completed' when the
--   agent did. A completion that arrives after a reap is recorded
--   (late_completion_at) and never resumed.
--
--   RLS GUC convention (LOCKED, per 0007 / 0055): chora.tenant_id, NULLIF-safe
--   casts (empty GUC -> NULL, never ''::uuid; the 22P02 lesson). The policy
--   carries the 0003_outbox SWEEPER MODE precedent: an UNSET/empty GUC matches
--   all rows, because the reaper scan and the completion router are
--   intra-service dispatcher machinery spanning every tenant, exactly like the
--   outbox dispatcher. A tenant-scoped session (GUC set) stays confined to its
--   tenant. app_rw holds NOBYPASSRLS; this is the established in-database
--   operational-table posture, not a new ADR-165/184/192 bypass surface; the
--   ADR-184 closure-saga PERMISSIVE policy on chora_ai_kernel is inherited.
--
--   Operational table: no soft-delete column (a true DELETE is never issued either;
--   rows settle in place and are retained as the park history).
--
--   Also: the outbox failed-row retry index. ADR-254 D5 makes the dispatcher
--   re-read status='failed' rows with backoff before dead-lettering them to
--   ai_kernel_outbox_dead_letters; the existing partial index covers only
--   'pending'.
--
--   Idempotent: IF NOT EXISTS for table/index + DO $$ duplicate_object for the
--   policy, so the runner can re-apply safely. Grants ride
--   9999_grant_app_roles.sql (run the FULL migrations job, never targeted).
--
--   HARD RULE: cross-database queries forbidden. Nothing outside this service
--   reads this table; the caller learns of a reaped run from its own result
--   topic (status FAILED) and from chora.ai_kernel.crew.run_failed.v1.
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS ai_kernel_agent_dispatch_parks (
    -- The dispatch idempotency key: agent_dispatch.{role}.{execution_id}
    -- (agent_dispatch.dispatch_idempotency_key). Same key as the outbox row.
    idempotency_key     TEXT         PRIMARY KEY,
    -- The aggregate root the outbox row keys on (UUID NOT NULL there too).
    workflow_id         UUID         NOT NULL,
    -- The LangGraph thread the run is parked under (may be non-UUID).
    thread_id           TEXT         NOT NULL,
    -- Which crew's runner resumes this thread (oe_grading, companion_diagnosis,
    -- qgen, single_agent:<role>, kg_exploration, companion_reflection).
    crew                TEXT         NOT NULL,
    tenant_id           UUID         NOT NULL,
    gcid                UUID         NOT NULL,
    agent_role          TEXT         NOT NULL,
    request_topic       TEXT         NOT NULL,
    completion_topic    TEXT         NOT NULL,
    -- W3C trace context of the dispatch so a reap links its spans and the
    -- run_failed event to the originating run.
    traceparent         TEXT         NOT NULL DEFAULT '',
    tracestate          TEXT         NOT NULL DEFAULT '',
    parked_at           TIMESTAMPTZ  NOT NULL,
    deadline_at         TIMESTAMPTZ  NOT NULL,
    state               TEXT         NOT NULL DEFAULT 'parked'
                        CHECK (state IN ('parked','completed','reaped')),
    settled_at          TIMESTAMPTZ,
    -- 'completion' | 'reaper:request_dead_lettered' | 'reaper:request_expired'
    -- | 'reaper:completion_dead_lettered' | 'reaper:outbox_dead_lettered'
    settled_by          TEXT         NOT NULL DEFAULT '',
    -- A completion that arrived after the row had been reaped: recorded for
    -- the record, never resumed.
    late_completion_at  TIMESTAMPTZ,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT ai_kernel_agent_dispatch_parks_deadline_after_park
        CHECK (deadline_at > parked_at)
);

-- Reaper arm (b): the park-age scan (state = 'parked' AND deadline_at < now()).
CREATE INDEX IF NOT EXISTS ai_kernel_agent_dispatch_parks_expiry_idx
    ON ai_kernel_agent_dispatch_parks (state, deadline_at);

-- Generic-workflow backpressure: parked count per tenant and role.
CREATE INDEX IF NOT EXISTS ai_kernel_agent_dispatch_parks_tenant_role_idx
    ON ai_kernel_agent_dispatch_parks (tenant_id, agent_role, state);

-- QGenResumeSweeper park-aware predicate and operator lookups by thread.
CREATE INDEX IF NOT EXISTS ai_kernel_agent_dispatch_parks_thread_idx
    ON ai_kernel_agent_dispatch_parks (thread_id);

ALTER TABLE ai_kernel_agent_dispatch_parks ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    CREATE POLICY ai_kernel_agent_dispatch_parks_tenant_isolation
        ON ai_kernel_agent_dispatch_parks
        USING (
            NULLIF(current_setting('chora.tenant_id', TRUE), '') IS NULL
            OR tenant_id = NULLIF(current_setting('chora.tenant_id', TRUE), '')::uuid
        )
        WITH CHECK (
            NULLIF(current_setting('chora.tenant_id', TRUE), '') IS NULL
            OR tenant_id = NULLIF(current_setting('chora.tenant_id', TRUE), '')::uuid
        );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Outbox failed-row retry (ADR-254 D5): the dispatcher re-reads 'failed' rows
-- ordered by their last attempt; 'deadlettered' is terminal and not indexed.
CREATE INDEX IF NOT EXISTS ai_kernel_outbox_events_failed_idx
    ON ai_kernel_outbox_events (last_attempt_at ASC) WHERE status = 'failed';

COMMIT;
