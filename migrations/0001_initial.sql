-- =============================================================================
-- chora-ai-kernel-orchestrator : 0001_initial.sql
--
-- Domain        : AI Kernel (supporting/platform)
-- Database      : chora_ai_kernel
-- Author        : agent-a5e52e89b73ede1d2 (db-migrations-11-services)
-- Date          : 2026-05-08
-- Architecture  : Architecture Review locked 2026-05-07 (Tier 2 D5 + D6 + D8)
--
-- Aggregates owned by this database:
--   - crew_compositions (fixed core crews per Content {verb} domain + dynamic specialists)
--   - orchestration_runs (per-request LangGraph orchestration session)
--   - langgraph_state_snapshots (PostgresSaver checkpoints — replay support)
--   - agent_decision_embeddings (pgvector — agent retrieval / similarity)
--
-- Hybrid kernel: Python LangGraph orchestrator + Go executor (Tier 2 D5).
-- PostgresSaver state lives here so HITL interrupts and replay both work.
-- =============================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector 0.8+ (CLAUDE.md §1)

CREATE OR REPLACE FUNCTION ai_kernel_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- ENUMs
-- -----------------------------------------------------------------------------
CREATE TYPE crew_domain         AS ENUM ('creation', 'consumption', 'delivery', 'sharing', 'a2a', 'kernel');
CREATE TYPE orchestration_status AS ENUM ('pending', 'running', 'completed', 'failed', 'interrupted');

-- -----------------------------------------------------------------------------
-- crew_compositions — fixed core crews per Content {verb} domain + dynamic
--
-- Per Tier 5 D20 + crew-composition skill: each domain owns a fixed core crew
-- modeled on AssessorFlow's 6-agent gate. Dynamic specialists are added per
-- run via orchestration_runs.dynamic_agent_ids.
-- -----------------------------------------------------------------------------
CREATE TABLE crew_compositions (
    composition_id       UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id            UUID         NULL,                        -- NULL = platform-wide default crew
    domain               crew_domain  NOT NULL,
    name                 VARCHAR(128) NOT NULL,
    description          TEXT         NOT NULL DEFAULT '',
    agent_ids            JSONB        NOT NULL DEFAULT '[]'::jsonb,   -- AGID[] (NOT GCID — agents)
    version              INTEGER      NOT NULL DEFAULT 1 CHECK (version >= 1),
    active               BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (domain, name, version)
);

CREATE INDEX idx_crew_compositions_tenant ON crew_compositions (tenant_id) WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_crew_compositions_domain ON crew_compositions (domain);
CREATE INDEX idx_crew_compositions_active ON crew_compositions (active) WHERE active = TRUE;

CREATE TRIGGER trg_crew_compositions_updated_at
    BEFORE UPDATE ON crew_compositions
    FOR EACH ROW EXECUTE FUNCTION ai_kernel_set_updated_at();

ALTER TABLE crew_compositions ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON crew_compositions
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = current_setting('chora.tenant_id', true)::uuid
    );

-- -----------------------------------------------------------------------------
-- orchestration_runs — per-request LangGraph session
-- -----------------------------------------------------------------------------
CREATE TABLE orchestration_runs (
    run_id                  UUID                 PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID                 NOT NULL,
    request_id              VARCHAR(128)         NOT NULL,
    gcid                    UUID                 NULL,                  -- requesting user (nullable for system runs)
    composition_id          UUID                 NULL REFERENCES crew_compositions(composition_id) ON DELETE RESTRICT,
    intent                  VARCHAR(128)         NOT NULL,
    status                  orchestration_status NOT NULL DEFAULT 'pending',
    started_at              TIMESTAMPTZ          NOT NULL DEFAULT now(),
    completed_at            TIMESTAMPTZ          NULL,
    total_cost_sgd_micros   BIGINT               NOT NULL DEFAULT 0 CHECK (total_cost_sgd_micros >= 0),
    total_input_tokens      INTEGER              NOT NULL DEFAULT 0 CHECK (total_input_tokens >= 0),
    total_output_tokens     INTEGER              NOT NULL DEFAULT 0 CHECK (total_output_tokens >= 0),
    error_summary           TEXT                 NULL,
    traceparent             VARCHAR(64)          NULL,
    dynamic_agent_ids       JSONB                NOT NULL DEFAULT '[]'::jsonb,
    created_at              TIMESTAMPTZ          NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ          NOT NULL DEFAULT now()
);

CREATE INDEX idx_runs_tenant     ON orchestration_runs (tenant_id, started_at DESC);
CREATE INDEX idx_runs_request    ON orchestration_runs (request_id);
CREATE INDEX idx_runs_gcid       ON orchestration_runs (gcid) WHERE gcid IS NOT NULL;
CREATE INDEX idx_runs_status     ON orchestration_runs (status);
CREATE INDEX idx_runs_intent     ON orchestration_runs (intent);

CREATE TRIGGER trg_orch_runs_updated_at
    BEFORE UPDATE ON orchestration_runs
    FOR EACH ROW EXECUTE FUNCTION ai_kernel_set_updated_at();

ALTER TABLE orchestration_runs ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON orchestration_runs
    FOR ALL USING (tenant_id = current_setting('chora.tenant_id', true)::uuid);

-- -----------------------------------------------------------------------------
-- langgraph_state_snapshots — PostgresSaver checkpoints
--
-- LangGraph's PostgresSaver writes one row per node-state transition. Schema
-- compatible with langgraph-checkpoint-postgres (key node_id + run_id + ts).
-- -----------------------------------------------------------------------------
CREATE TABLE langgraph_state_snapshots (
    snapshot_id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id               UUID         NOT NULL REFERENCES orchestration_runs(run_id) ON DELETE RESTRICT,
    tenant_id            UUID         NOT NULL,
    node_id              VARCHAR(128) NOT NULL,
    parent_snapshot_id   UUID         NULL REFERENCES langgraph_state_snapshots(snapshot_id) ON DELETE RESTRICT,
    state_json           JSONB        NOT NULL,
    metadata             JSONB        NOT NULL DEFAULT '{}'::jsonb,
    captured_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_snapshots_run      ON langgraph_state_snapshots (run_id, captured_at);
CREATE INDEX idx_snapshots_tenant   ON langgraph_state_snapshots (tenant_id);
CREATE INDEX idx_snapshots_node     ON langgraph_state_snapshots (node_id);

ALTER TABLE langgraph_state_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON langgraph_state_snapshots
    FOR ALL USING (tenant_id = current_setting('chora.tenant_id', true)::uuid);

-- -----------------------------------------------------------------------------
-- agent_decision_embeddings — pgvector embeddings of past agent decisions
--
-- Used by Model Broker Classifier Service for fuzzy routing (Tier 2 D6) and
-- by Familiar/Mentor agents for retrieval-augmented persona consistency.
-- -----------------------------------------------------------------------------
CREATE TABLE agent_decision_embeddings (
    embedding_id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id            UUID         NOT NULL,
    agid                 UUID         NOT NULL,
    decision_log_id      UUID         NOT NULL,                  -- cross-DB ref to chora_observability.agent_decision_log
    embedding            vector(768)  NOT NULL,
    model_id             VARCHAR(64)  NOT NULL,
    embedding_version    INTEGER      NOT NULL DEFAULT 1,
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_agent_dec_emb_cosine
    ON agent_decision_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX idx_agent_dec_emb_tenant ON agent_decision_embeddings (tenant_id);
CREATE INDEX idx_agent_dec_emb_agid   ON agent_decision_embeddings (agid);

ALTER TABLE agent_decision_embeddings ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON agent_decision_embeddings
    FOR ALL USING (tenant_id = current_setting('chora.tenant_id', true)::uuid);

COMMIT;
