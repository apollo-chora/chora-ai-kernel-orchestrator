-- =============================================================================
-- chora-ai-kernel-orchestrator : 0055_ai_assist_inflight_registry.down.sql
--
-- Reverts 0011 (ADR-251 D4 in-flight registry). Operational machinery only:
-- dropping the table forfeits resume-on-boot for any job in flight at the
-- moment of the drop (those jobs' checkpoints remain but nothing sweeps
-- them), so drain in-flight jobs before applying this down migration.
-- =============================================================================

BEGIN;

DROP TABLE IF EXISTS ai_assist_inflight_jobs;

COMMIT;
