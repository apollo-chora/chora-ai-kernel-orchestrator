-- =============================================================================
-- chora-ai-kernel-orchestrator : 0059_kernel_rls_fail_closed.down.sql
--
-- Restores the fail-OPEN policies exactly as 0055 and 0056 shipped them.
--
-- ⚠ THIS DOWN REOPENS A CROSS-TENANT READ SURFACE. That is its purpose and it
--   should be run for exactly one reason: you are rolling the kernel binary
--   BACK to a digest that does not set chora.kernel_sweeper, and you must run
--   this BEFORE that rollback lands or the old binary meets a policy it cannot
--   satisfy (2 hard errors on the INSERT paths, 13 silent zero-row reads —
--   a reaper that reaps nothing and tests green).
--
--   After such a rollback the fail-open hole is live again, with the measured
--   exposure the up-migration documents (41 park rows across 2 tenants visible
--   to any GUC-unset session as of 2026-08-23). Treat that as an open incident,
--   not a resting state, and re-apply the up as soon as the binary is forward
--   again.
-- =============================================================================

BEGIN;

DROP POLICY IF EXISTS ai_assist_inflight_tenant_isolation ON ai_assist_inflight_jobs;

CREATE POLICY ai_assist_inflight_tenant_isolation ON ai_assist_inflight_jobs
    USING (
        NULLIF(current_setting('chora.tenant_id', TRUE), '') IS NULL
        OR tenant_id = NULLIF(current_setting('chora.tenant_id', TRUE), '')::uuid
    )
    WITH CHECK (
        NULLIF(current_setting('chora.tenant_id', TRUE), '') IS NULL
        OR tenant_id = NULLIF(current_setting('chora.tenant_id', TRUE), '')::uuid
    );

DROP POLICY IF EXISTS ai_kernel_agent_dispatch_parks_tenant_isolation
    ON ai_kernel_agent_dispatch_parks;

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

COMMIT;
