-- =============================================================================
-- chora-ai-kernel-orchestrator : 0002_agent_registry_and_langgraph.sql
--
-- Domain        : AI Kernel (supporting/platform)
-- Database      : chora_ai_kernel
-- Author        : agent-A-AIK-Orchestrator (S3.4 P0)
-- Date          : 2026-05-09
-- Architecture  : Architecture Review locked 2026-05-07 (Tier 2 D5 + Tier 5 D20)
--
-- Adds:
--   - chora_agent_registry : per-agent metadata (risk_tier + autonomy_level
--     per ADR-141; the 6 AI Assist agents live here alongside the 24-agent
--     legacy cast). Populated at orchestrator startup from the in-process
--     `AI_ASSIST_AGENT_REGISTRY` (Sequel comic cast plus the 6-agent gate).
--
-- Note on LangGraph checkpoint tables:
--   The PostgresSaver creates its own `checkpoints` + `checkpoint_writes`
--   schema lazily on first connection (see langgraph-checkpoint-postgres
--   ≥3.0). We do NOT pre-create those tables here — letting LangGraph own
--   the schema avoids drift on minor version bumps. The legacy
--   `langgraph_state_snapshots` table from 0001_initial.sql remains for
--   admin trace ad-hoc reporting and is NOT used as the saver backing
--   store.
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- chora_agent_registry — per-agent risk + autonomy + guardrail YAML
--
-- Per `ai-runtime-guardrails` skill + ADR-141 the registry is the
-- authoritative source for risk_tier (1-4) + autonomy_level (HOOTL/HOTL/
-- HITL Level 0-2; Level 3 PROHIBITED). The Guardrail Service reads this
-- table to load the per-agent YAML guardrail composition; the O+ admin
-- console reads it for the IMDA dimension dashboard.
-- -----------------------------------------------------------------------------
CREATE TYPE agent_autonomy_level AS ENUM (
    'HOOTL',
    'HOTL',
    'HITL_LEVEL_0',
    'HITL_LEVEL_1',
    'HITL_LEVEL_2'
    -- LEVEL_3 deliberately omitted — prohibited per ADR-141
);

CREATE TABLE chora_agent_registry (
    agent_id              VARCHAR(64)            PRIMARY KEY,
    role                  VARCHAR(64)            NOT NULL,
    display_name          VARCHAR(128)           NOT NULL,
    risk_tier             INTEGER                NOT NULL CHECK (risk_tier BETWEEN 1 AND 4),
    autonomy_level        agent_autonomy_level   NOT NULL,
    guardrail_yaml_ref    VARCHAR(256)           NOT NULL,
    imda_dimension        VARCHAR(64)            NOT NULL,
    hitl_imda_dimension   VARCHAR(64)            NOT NULL DEFAULT 'fairness_and_human_oversight',
    owner_team            VARCHAR(32)            NOT NULL DEFAULT 'Team 3',
    version               INTEGER                NOT NULL DEFAULT 1 CHECK (version >= 1),
    active                BOOLEAN                NOT NULL DEFAULT TRUE,
    created_at            TIMESTAMPTZ            NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ            NOT NULL DEFAULT now(),
    -- Canonical IMDA dimensions per ADR-141 — soft constraint at the app
    -- layer (the registry seeder rejects unknown values). At schema level
    -- we accept any string so v1 deprecated aliases coming from older
    -- services get accepted on input + canonicalised by `imda.Canonicalise()`.
    CONSTRAINT chora_agent_registry_role_known CHECK (
        role IN (
            -- 6 AI Assist gate roles
            'validator', 'classifier', 'web-researcher',
            'qa-generator', 'evaluator', 'reporter',
            -- Legacy 24-agent cast (Sequel comic) — kept for compatibility.
            'orchestrator', 'tutor', 'weakness-analyzer', 'learning-path',
            'adaptive-difficulty', 'rag-memory', 'learning-analytics',
            'content-analyst', 'explainer', 'ai-governance',
            'company-policy', 'content-gen', 'content-review',
            'training-compliance', 'media-transcription', 'translation',
            'trigger-optimizer', 'exam-marking', 'familiar', 'model-broker',
            'pvp-screener', 'assessment', 'mcp-tool-router',
            'story-point-estimator', 'exam-prep-coach'
        )
    )
);

CREATE INDEX idx_agent_registry_role     ON chora_agent_registry (role);
CREATE INDEX idx_agent_registry_active   ON chora_agent_registry (active) WHERE active = TRUE;
CREATE INDEX idx_agent_registry_risktier ON chora_agent_registry (risk_tier);

CREATE TRIGGER trg_agent_registry_updated_at
    BEFORE UPDATE ON chora_agent_registry
    FOR EACH ROW EXECUTE FUNCTION ai_kernel_set_updated_at();

-- The registry is platform-wide (no tenant_id column) — it is read-mostly
-- and shared across tenants. RLS is therefore NOT enabled here.

COMMIT;
