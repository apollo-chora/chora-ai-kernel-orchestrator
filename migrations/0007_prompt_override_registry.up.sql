-- =============================================================================
-- chora-ai-kernel-orchestrator : 0007_prompt_override_registry.up.sql
--
-- ADR           : ADR-197 — Versioned agent prompt registry + O+ prompt
--                 explainability (M-B.1: registry foundation, behaviour-neutral)
-- Domain        : AI Kernel (supporting/platform)
-- Database      : chora_ai_kernel
-- Date          : 2026-06-28
-- Architecture  : ADR-197 §Decision.2 + §Addendum "M-B design grounded in
--                 deployed reality"; modelled on ADR-178 PricePlanResolver
--                 (chora-identity migration 0017_mana_price_plan_rules_layer).
--
-- Purpose:
--   Add the OPTIONAL prompt-override layer. Two tables:
--     1. prompt_override_plan    — named, versioned, platform-or-tenant-scoped
--                                  override plans (one active per scope/tenant).
--     2. prompt_override_segment — the overridden behavioural blocks
--                                  ([ROLE]/[TASK]/[EXAMPLES]/tone) keyed by
--                                  (plan, agent_id, segment_id).
--
--   OVERRIDES-ONLY model (owner constraint, ADR-197 §Decision.1): the embedded
--   default prompt stays the engineer-deployed source of truth IN THE GO BINARY.
--   This registry stores ONLY overrides. With zero rows, PromptResolver returns
--   source='embedded' / empty segments and composers use the embedded default →
--   byte-identical behaviour. THEREFORE: NO back-fill / NO seed here.
--
--   Resolution precedence (PromptResolver, mirrors ADR-178 COALESCE ladder):
--     active tenant override plan → active platform override plan → embedded.
--
--   RLS GUC convention (LOCKED, per 0001_initial.sql + identity
--   0003_user_economy_rls.sql + 0017_mana_price_plan_rules_layer.up.sql):
--     SET LOCAL chora.tenant_id = '<uuid>';
--     SET LOCAL chora.role      = '<learner|instructor|admin|auditor>';
--   The policies are NULLIF-safe (empty-GUC -> NULL, never ''::uuid) to avoid
--   the 22P02 "invalid input syntax for type uuid" RLS cast bug seen when the
--   placeholder GUC reverts to '' (chora-identity mig 0019 / FU-4b learning).
--
--   Idempotent: IF NOT EXISTS for tables/indexes + DO $$ ... duplicate_object
--   for policies/triggers, so the runner can re-apply safely.
--
--   HARD RULE: cross-database queries forbidden. These tables are local to
--   chora_ai_kernel; H+/O+ reach them via gRPC/events only. New tables use
--   gen_random_uuid in SQL (one-off platform rows); app code mints UUIDv7.
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- 1. prompt_override_plan — the plan axis (platform + tenant scope).
--
-- A plan is a named, versioned set of override segments. The platform owns the
-- canonical override plans; a tenant may own a plan that overrides selected
-- segments for its own agents. Precedence is encoded by scope + tenant_id and
-- resolved at request time by PromptResolver.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_override_plan (
    plan_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_code      VARCHAR(64)  NOT NULL,
    scope          VARCHAR(16)  NOT NULL,                 -- platform|tenant
    tenant_id      UUID,                                  -- NULL when scope='platform'
    status         VARCHAR(16)  NOT NULL DEFAULT 'draft', -- draft|active|archived
    effective_from TIMESTAMPTZ  NOT NULL DEFAULT now(),
    created_by     UUID,                                  -- admin GCID (audit)
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT prompt_override_plan_scope_chk  CHECK (scope  IN ('platform','tenant')),
    CONSTRAINT prompt_override_plan_status_chk CHECK (status IN ('draft','active','archived')),
    CONSTRAINT prompt_override_plan_scope_tenant_chk CHECK (
        (scope = 'platform' AND tenant_id IS NULL)
     OR (scope = 'tenant'   AND tenant_id IS NOT NULL)
    )
);

-- Exactly one ACTIVE platform plan, and at most one ACTIVE plan per tenant.
CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_plan_active_platform
    ON prompt_override_plan (scope) WHERE status = 'active' AND scope = 'platform';
CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_plan_active_tenant
    ON prompt_override_plan (tenant_id) WHERE status = 'active' AND scope = 'tenant';

CREATE INDEX IF NOT EXISTS idx_prompt_plan_lookup
    ON prompt_override_plan (scope, status);

DO $$ BEGIN
    CREATE TRIGGER trg_prompt_override_plan_updated_at
        BEFORE UPDATE ON prompt_override_plan
        FOR EACH ROW EXECUTE FUNCTION ai_kernel_set_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- -----------------------------------------------------------------------------
-- 2. prompt_override_segment — the override text axis (one editable block).
--
-- One row per (plan, agent_id, segment_id). Only BEHAVIOURAL blocks are stored
-- here ([ROLE]/[TASK]/[EXAMPLES]/tone); the structural [EXPECTED OUTPUT] JSON
-- contract + immutable safety preamble are NEVER overridable (ADR-197
-- §Decision.3) — they always come from the embedded default. The composer
-- enforces the allow-list; this table just holds the override bodies.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_override_segment (
    segment_row_id UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id        UUID         NOT NULL REFERENCES prompt_override_plan(plan_id) ON DELETE CASCADE,
    agent_id       VARCHAR(64)  NOT NULL,
    segment_id     VARCHAR(64)  NOT NULL,
    body           TEXT         NOT NULL,
    content_hash   VARCHAR(64),                           -- sha256 of body (audit / cache key)
    version        INT          NOT NULL DEFAULT 1,
    note           TEXT         NOT NULL DEFAULT '',
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- A plan overrides each (agent_id, segment_id) at most once. NULLS NOT DISTINCT
-- (PG15+) so a duplicate collides even if a key part is ever NULL.
CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_segment_plan_agent_segment
    ON prompt_override_segment (plan_id, agent_id, segment_id) NULLS NOT DISTINCT;
CREATE INDEX IF NOT EXISTS idx_prompt_segment_agent
    ON prompt_override_segment (agent_id, segment_id);

DO $$ BEGIN
    CREATE TRIGGER trg_prompt_override_segment_updated_at
        BEFORE UPDATE ON prompt_override_segment
        FOR EACH ROW EXECUTE FUNCTION ai_kernel_set_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- -----------------------------------------------------------------------------
-- RLS — prompt_override_plan + prompt_override_segment (tenant axis via plan).
--
-- A tenant admin can read/write ONLY its own tenant's plans; platform-scope
-- plans are world-readable (every tenant resolves against them as the
-- platform-override rung). admin role bypass for O+ auditor / platform ops.
--
-- prompt_override_segment has no tenant_id column — it inherits scope through
-- its plan, so its policy is an EXISTS subquery against prompt_override_plan
-- (RLS subqueries re-apply the parent policy).
-- -----------------------------------------------------------------------------

ALTER TABLE prompt_override_plan ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    CREATE POLICY prompt_plan_scope_isolation ON prompt_override_plan
        FOR ALL USING (
            scope = 'platform'
            OR tenant_id = NULLIF(current_setting('chora.tenant_id', true), '')::uuid
            OR current_setting('chora.role', true) = 'admin'
        );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

ALTER TABLE prompt_override_segment ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    CREATE POLICY prompt_segment_scope_isolation ON prompt_override_segment
        FOR ALL USING (
            current_setting('chora.role', true) = 'admin'
            OR EXISTS (
                SELECT 1 FROM prompt_override_plan pp
                 WHERE pp.plan_id = prompt_override_segment.plan_id
                   AND ( pp.scope = 'platform'
                      OR pp.tenant_id = NULLIF(current_setting('chora.tenant_id', true), '')::uuid )
            )
        );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMIT;

-- =============================================================================
-- VERIFICATION (run manually after apply):
--
--   -- both tables present + RLS enabled
--   SELECT relname, relrowsecurity FROM pg_class
--    WHERE relname IN ('prompt_override_plan','prompt_override_segment');
--   -- => relrowsecurity = t for both
--
--   -- overrides-only: no rows seeded (resolution falls back to embedded)
--   SELECT count(*) FROM prompt_override_plan;     -- => 0
--   SELECT count(*) FROM prompt_override_segment;  -- => 0
--
--   -- at most one active platform plan (partial unique index enforces it)
--   SELECT count(*) FROM prompt_override_plan WHERE scope='platform' AND status='active';
--   -- => 0 or 1
-- =============================================================================
