-- =============================================================================
-- chora-ai-kernel-orchestrator : 0010_prompt_version_label_dedup.up.sql
--
-- ADR           : ADR-197 (versioned agent prompt registry + O+ explainability)
-- Story         : CHO-2379 (owner feedback 2026-07-27)
-- Domain        : AI Kernel (supporting/platform)
-- Database      : chora_ai_kernel
-- Date          : 2026-07-28
--
-- Purpose:
--   Make a platform override plan's (agent_id, version_label) unique, and
--   relabel the live rows that already collide.
--
--   Why this is not cosmetic. ARCHIVED is terminal (ADR-197 six-state gate), so
--   a rollback plus a re-promotion always mints a FRESH plan row. Until
--   CHO-2379 that fresh row reused the embedded revision's constant label, so
--   the live catalogue carried three qgen_question plans all reading 1.1.0
--   (archived / rejected / active). Two real consequences:
--
--     1. the catalogue's single-version read
--        (prompt_override_repository._GET_VERSION_PLAN_SQL) disambiguates a
--        duplicate label with ORDER BY created_at DESC LIMIT 1, so the older
--        plans' segments were UNREACHABLE in the O+ prompt modal;
--     2. the Vertex eval-run name is derived from the label, and reuse hit a
--        409 AlreadyExists (worked around with an r2 suffix in CHO-2368).
--
--   The BE half of the fix auto-bumps at create-draft
--   (domain/prompt_registry/version_labels.py); this migration cleans the rows
--   already written and installs the guard that keeps them clean.
--
-- What it does:
--   1. Relabel. Within each (agent_id, major.minor family), duplicates are
--      ranked by created_at (plan_id breaks a tie). The OLDEST attempt keeps
--      its label; every later attempt climbs to family-max-patch + n, so the
--      newest attempt ends HIGHEST and a re-promotion always sorts above the
--      one it supersedes. plan_code follows the new label - leaving it stale
--      would re-introduce the duplicate on the other axis.
--
--      NOTHING is deleted or archived: an override plan is an audited
--      promotion attempt. Relabelling the ACTIVE plan is safe because the
--      label is display plus stamp identity only - the resolver picks the
--      active plan by (agent, status), and historical StepStamps keep the
--      label they were written with as a point-in-time fact.
--
--   2. Guard. A partial unique index on (agent_id, version_label) for
--      kind='override' AND scope='platform'. Baselines keep their own 0009
--      index (uq_prompt_plan_baseline, kind='baseline'); tenant-scope
--      overrides are deliberately out of scope, so their label space stays
--      free for a per-tenant naming convention.
--
--   Idempotent: re-applying is a no-op once no duplicates remain (the relabel
--   CTE selects nothing) and the index uses IF NOT EXISTS.
--
--   No new tables, so 9999_grant_app_roles.sql needs no addition.
-- =============================================================================

BEGIN;

-- 1. One-shot relabel of the colliding rows -----------------------------------
WITH numbered AS (
    SELECT plan_id,
           agent_id,
           version_label,
           created_at,
           split_part(version_label, '.', 1) || '.'
             || split_part(version_label, '.', 2) AS family,
           ROW_NUMBER() OVER (
               PARTITION BY agent_id, version_label
               ORDER BY created_at, plan_id
           ) AS rn
      FROM prompt_override_plan
     WHERE scope = 'platform'
       AND kind = 'override'
       AND version_label ~ '^[0-9]+\.[0-9]+\.[0-9]+$'
),
fam_max AS (
    SELECT agent_id,
           family,
           MAX(split_part(version_label, '.', 3)::int) AS max_patch
      FROM numbered
     GROUP BY agent_id, family
),
relabel AS (
    -- rn > 1 is applied BEFORE this ROW_NUMBER (window functions run after
    -- WHERE), so the ranking counts only the rows that actually move.
    SELECT n.plan_id,
           n.family || '.' || (
               f.max_patch + ROW_NUMBER() OVER (
                   PARTITION BY n.agent_id, n.family
                   ORDER BY n.created_at, n.plan_id
               )
           )::text AS new_label
      FROM numbered n
      JOIN fam_max f USING (agent_id, family)
     WHERE n.rn > 1
)
UPDATE prompt_override_plan p
   SET version_label = r.new_label,
       plan_code     = p.agent_id || '-' || r.new_label
  FROM relabel r
 WHERE p.plan_id = r.plan_id;

-- 2. The guard ----------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_plan_override_version
    ON prompt_override_plan (agent_id, version_label)
 WHERE kind = 'override' AND scope = 'platform';

COMMIT;

-- =============================================================================
-- VERIFICATION (run manually after apply):
--
--   -- zero duplicate platform override labels
--   SELECT agent_id, version_label, count(*)
--     FROM prompt_override_plan
--    WHERE scope='platform' AND kind='override'
--    GROUP BY 1,2 HAVING count(*) > 1;
--   -- => 0 rows
--
--   -- the qgen_question ladder, active on the HIGHEST label
--   SELECT version_label, status, plan_code, created_at
--     FROM prompt_override_plan
--    WHERE scope='platform' AND kind='override' AND agent_id='qgen_question'
--    ORDER BY created_at;
--   -- => 1.1.0 archived | 1.1.1 rejected | 1.1.2 active
--
--   -- the guard bites (run inside a transaction and ROLL BACK)
--   BEGIN;
--     INSERT INTO prompt_override_plan
--       (plan_code, scope, tenant_id, status, kind, agent_id, version_label)
--     VALUES ('probe-dup','platform',NULL,'draft','override',
--             'qgen_question','1.1.2');
--   -- => ERROR: duplicate key value violates unique constraint
--   --           "uq_prompt_plan_override_version"
--   ROLLBACK;
-- =============================================================================
