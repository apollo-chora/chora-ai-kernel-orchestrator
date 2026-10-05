-- =============================================================================
-- chora-ai-kernel-orchestrator : 0055_ai_assist_inflight_registry.up.sql
--
-- ADR           : ADR-251 D4 (CHO-2398) - durable acceptance + resume-on-boot
--                 for the AI Assist set lane.
-- Domain        : AI Kernel (supporting/platform)
-- Database      : chora_ai_kernel
-- Date          : 2026-08-16
--
-- Purpose:
--   The in-flight job registry. handle_started splits at a durable boundary:
--   ACCEPTANCE (decode, validate, INSERT here, seed the first checkpoint, ACK)
--   then DRIVE (background task). Pod death after ACK can no longer rely on
--   Pub/Sub redelivery, so on boot (and after a cost-pause resume) the
--   orchestrator SWEEPS this table and resumes every unfinished job from its
--   LangGraph checkpoint. A row is REMOVED in the same transaction boundary
--   that makes the job's terminal event durable in ai_kernel_outbox_events
--   (ADR-251 D4 locked semantics: registry presence == unfinished; this is
--   operational machinery, not domain data, so removal is a true DELETE and
--   there is deliberately no deleted_at).
--
--   RLS GUC convention (LOCKED, per 0007_prompt_override_registry.up.sql):
--     SET LOCAL chora.tenant_id = '<uuid>';
--   NULLIF-safe casts (empty GUC -> NULL, never ''::uuid; the 22P02 lesson).
--   The tenant policy carries the 0003_outbox SWEEPER MODE precedent: an
--   UNSET/empty GUC matches all rows, because the boot sweep and the terminal
--   delete are intra-service dispatcher machinery spanning every tenant,
--   exactly like the outbox dispatcher on ai_kernel_outbox_events. A
--   tenant-scoped session (GUC set) stays confined to its tenant. app_rw
--   holds NOBYPASSRLS; this is the established in-database operational-table
--   posture, not a new ADR-165/184/192 bypass surface.
--
--   Idempotent: IF NOT EXISTS for table/index + DO $$ duplicate_object for
--   the policy, so the runner can re-apply safely. Grants ride
--   9999_grant_app_roles.sql (run the FULL migrations job, never targeted:
--   a targeted run that skips 9999 leaves app_rw at 42501).
--
--   HARD RULE: cross-database queries forbidden. chora-creation learns about
--   chunk results via chora.creation.ai_assist.chunk_completed.v1, never by
--   reading this table.
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS ai_assist_inflight_jobs (
    -- The assist/job id (UUIDv7, minted by chora-creation and carried on
    -- started.v2; also the LangGraph thread_id the drive checkpoints under).
    assist_id             UUID         PRIMARY KEY,
    tenant_id             UUID         NOT NULL,
    author_gcid           UUID         NOT NULL,

    -- The ACCEPTED started payload (decoded, validated), everything the
    -- background drive and a boot resume need to rebuild the runner input.
    started_payload_json  JSONB        NOT NULL,

    -- W3C trace context captured at acceptance so a resumed drive links its
    -- spans to the originating workflow trace.
    traceparent           TEXT         NOT NULL DEFAULT '',
    tracestate            TEXT         NOT NULL DEFAULT '',

    accepted_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),

    -- Resume bookkeeping (observability of the boot sweep; a double resume is
    -- idempotent at the LangGraph checkpoint layer, these count the attempts).
    resume_count          INTEGER      NOT NULL DEFAULT 0,
    last_resumed_at       TIMESTAMPTZ,

    updated_at            TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Sweep order: oldest accepted first, so long-waiting jobs resume before
-- fresh ones after a cost-pause gap.
CREATE INDEX IF NOT EXISTS idx_ai_assist_inflight_accepted_at
    ON ai_assist_inflight_jobs (accepted_at);

ALTER TABLE ai_assist_inflight_jobs ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    CREATE POLICY ai_assist_inflight_tenant_isolation ON ai_assist_inflight_jobs
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

COMMIT;
