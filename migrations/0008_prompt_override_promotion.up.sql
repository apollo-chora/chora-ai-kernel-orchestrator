-- =============================================================================
-- chora-ai-kernel-orchestrator : 0008_prompt_override_promotion.up.sql
--
-- ADR           : ADR-197 — Versioned agent prompt registry + O+ prompt
--                 explainability (M-C.1: promotion state machine + write
--                 paths + activation audit, behaviour-neutral).
-- Domain        : AI Kernel (supporting/platform)
-- Database      : chora_ai_kernel
-- Date          : 2026-06-28
-- Architecture  : ADR-197 §Decision (promotion lifecycle); modelled on the
--                 closure-saga state machine
--                 (services/chora-closure-orchestrator/.../domain/closure/state.py)
--                 and migration 0007_prompt_override_registry.
--
-- Purpose:
--   Promote the prompt-override registry from a flat draft|active|archived
--   status to the full promotion lifecycle:
--
--     draft -> pending_eval -> pending_hitl -> active -> archived
--                  |                  |
--                  +--> rejected <----+ --> archived
--
--   A plan is authored (draft), submitted for automated eval (pending_eval),
--   passes eval and waits for human sign-off (pending_hitl), is activated
--   (active) — at which point the prior active plan for that scope/tenant is
--   archived (the 0007 partial unique index guarantees one active per scope/
--   tenant). A plan that fails eval OR HITL is rejected, and rejected/active
--   plans are eventually archived (terminal).
--
--   Three columns capture the activation provenance (IMDA D1 accountability):
--     - approved_by  UUID        — the admin GCID who activated the plan.
--     - approved_at  TIMESTAMPTZ — when activation happened.
--     - eval_run_id  TEXT        — the eval run that gated pending_eval ->
--                                  pending_hitl (links to the O+ eval record).
--
--   BEHAVIOUR-NEUTRAL: no caller writes these states yet (M-C.2 wires the H+
--   write endpoints + O+ promotion console). With zero override rows the
--   PromptResolver (M-B.1) still returns source='embedded' — byte-identical.
--   NO back-fill / NO seed.
--
--   Idempotent: DROP CONSTRAINT IF EXISTS before ADD (drop-then-add is
--   re-appliable) + ADD COLUMN IF NOT EXISTS, so the runner can re-apply safely.
--
--   HARD RULE: cross-database queries forbidden. These columns are local to
--   chora_ai_kernel; H+/O+ reach them via gRPC/events only.
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- 1. Widen the status CHECK to the full promotion lifecycle.
--
-- draft|active|archived (0007) -> draft|pending_eval|pending_hitl|active|
-- rejected|archived. DROP IF EXISTS then ADD so a re-apply is a no-op (the
-- drop removes any prior definition, the add re-creates the current one).
-- -----------------------------------------------------------------------------
ALTER TABLE prompt_override_plan
    DROP CONSTRAINT IF EXISTS prompt_override_plan_status_chk;

ALTER TABLE prompt_override_plan
    ADD CONSTRAINT prompt_override_plan_status_chk
        CHECK (status IN (
            'draft',
            'pending_eval',
            'pending_hitl',
            'active',
            'rejected',
            'archived'
        ));

-- -----------------------------------------------------------------------------
-- 2. Activation-provenance columns (IMDA D1). All NULLable — a draft plan has
--    no approver / approval time / eval run until it progresses.
-- -----------------------------------------------------------------------------
ALTER TABLE prompt_override_plan
    ADD COLUMN IF NOT EXISTS approved_by UUID;          -- admin GCID who activated
ALTER TABLE prompt_override_plan
    ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ;   -- when activation happened
ALTER TABLE prompt_override_plan
    ADD COLUMN IF NOT EXISTS eval_run_id TEXT;          -- gating eval run (O+ link)

COMMIT;

-- =============================================================================
-- VERIFICATION (run manually after apply):
--
--   -- status CHECK now admits the 6 lifecycle states
--   SELECT pg_get_constraintdef(oid) FROM pg_constraint
--    WHERE conname = 'prompt_override_plan_status_chk';
--   -- => CHECK (status IN ('draft','pending_eval','pending_hitl','active',
--   --                      'rejected','archived'))
--
--   -- the three provenance columns exist + are NULLable
--   SELECT column_name, is_nullable, data_type
--     FROM information_schema.columns
--    WHERE table_name = 'prompt_override_plan'
--      AND column_name IN ('approved_by','approved_at','eval_run_id');
--   -- => 3 rows, all is_nullable = YES
--
--   -- behaviour-neutral: still no rows seeded
--   SELECT count(*) FROM prompt_override_plan;   -- => 0
-- =============================================================================
