-- =============================================================================
-- chora-ai-kernel-orchestrator : 0056_agent_dispatch_parks.down.sql
--
-- Reverses 0056 only: the park ledger and the outbox failed-row retry index.
-- ai_kernel_outbox_events itself is 0003's and is left untouched.
-- =============================================================================

BEGIN;

DROP INDEX IF EXISTS ai_kernel_outbox_events_failed_idx;

DROP TABLE IF EXISTS ai_kernel_agent_dispatch_parks;

COMMIT;
