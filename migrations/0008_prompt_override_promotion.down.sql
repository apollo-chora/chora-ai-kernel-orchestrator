-- =============================================================================
-- chora-ai-kernel-orchestrator : 0008_prompt_override_promotion.down.sql
--
-- ADR           : ADR-197 (M-C.1) — rollback for 0008_prompt_override_promotion.
-- Database      : chora_ai_kernel
--
-- NOTE: the migration runner (chora-infra/scripts/migrations-runner/runner.sh)
-- applies *.up.sql / *.sql in lexicographic order and SKIPS *.down.sql. This
-- file exists for manual/local rollback only and is never auto-applied.
--
-- Reverts the promotion lifecycle back to the 0007 flat status
-- (draft|active|archived) + drops the activation-provenance columns. Idempotent
-- (IF EXISTS). Any rows in a state removed by the narrower CHECK (pending_eval /
-- pending_hitl / rejected) MUST be resolved before rollback — the registry is
-- overrides-only with no back-fill, so in the normal (empty) case this is safe.
-- =============================================================================

BEGIN;

ALTER TABLE prompt_override_plan
    DROP COLUMN IF EXISTS eval_run_id;
ALTER TABLE prompt_override_plan
    DROP COLUMN IF EXISTS approved_at;
ALTER TABLE prompt_override_plan
    DROP COLUMN IF EXISTS approved_by;

ALTER TABLE prompt_override_plan
    DROP CONSTRAINT IF EXISTS prompt_override_plan_status_chk;

ALTER TABLE prompt_override_plan
    ADD CONSTRAINT prompt_override_plan_status_chk
        CHECK (status IN ('draft', 'active', 'archived'));

COMMIT;
