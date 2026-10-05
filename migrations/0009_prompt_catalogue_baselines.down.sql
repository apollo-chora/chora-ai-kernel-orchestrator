-- 0009_prompt_catalogue_baselines.down.sql (CHO-2368 rollback)
--
-- Removes the catalogue carve-out: deletes the seeded baseline plans (their
-- segments cascade via the plan FK), drops the per-agent override uniques +
-- the baseline unique, restores the 0007 global one-active-per-scope uniques,
-- and drops the 0009 columns.

BEGIN;

DELETE FROM prompt_override_plan WHERE kind = 'baseline';

DROP INDEX IF EXISTS uq_prompt_plan_baseline;
DROP INDEX IF EXISTS uq_prompt_plan_active_platform;
DROP INDEX IF EXISTS uq_prompt_plan_active_tenant;
DROP INDEX IF EXISTS idx_prompt_plan_agent_kind;

CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_plan_active_platform
    ON prompt_override_plan (scope) WHERE status = 'active' AND scope = 'platform';
CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_plan_active_tenant
    ON prompt_override_plan (tenant_id) WHERE status = 'active' AND scope = 'tenant';

ALTER TABLE prompt_override_plan DROP CONSTRAINT IF EXISTS prompt_override_plan_kind_chk;
ALTER TABLE prompt_override_plan DROP COLUMN IF EXISTS kind;
ALTER TABLE prompt_override_plan DROP COLUMN IF EXISTS agent_id;
ALTER TABLE prompt_override_plan DROP COLUMN IF EXISTS version_label;

ALTER TABLE prompt_override_segment DROP COLUMN IF EXISTS locked;
ALTER TABLE prompt_override_segment DROP COLUMN IF EXISTS position;

COMMIT;
