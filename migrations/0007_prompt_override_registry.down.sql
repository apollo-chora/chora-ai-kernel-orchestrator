-- =============================================================================
-- chora-ai-kernel-orchestrator : 0007_prompt_override_registry.down.sql
--
-- ADR           : ADR-197 (M-B.1) — rollback for 0007_prompt_override_registry.
-- Database      : chora_ai_kernel
--
-- NOTE: the migration runner (chora-infra/scripts/migrations-runner/runner.sh)
-- applies *.up.sql / *.sql in lexicographic order and SKIPS *.down.sql. This
-- file exists for manual/local rollback only and is never auto-applied.
--
-- Drops the prompt-override registry. Idempotent (IF EXISTS). Safe because the
-- registry is overrides-only with no back-fill — dropping it returns the system
-- to embedded-default behaviour (its pre-0007 state).
-- =============================================================================

BEGIN;

DROP TABLE IF EXISTS prompt_override_segment;
DROP TABLE IF EXISTS prompt_override_plan;

COMMIT;
