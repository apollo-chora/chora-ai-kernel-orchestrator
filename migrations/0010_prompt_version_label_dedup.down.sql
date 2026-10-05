-- 0010_prompt_version_label_dedup.down.sql (CHO-2379 rollback)
--
-- Drops ONLY the uniqueness guard. The one-shot relabel is deliberately NOT
-- reversed: re-introducing duplicate (agent_id, version_label) rows would put
-- the superseded plans back behind the catalogue's newest-wins single-version
-- read (ORDER BY created_at DESC LIMIT 1) and re-arm the eval-run 409. The
-- relabelled rows stay valid without the index; only the guard is optional.
--
-- 0009's uq_prompt_plan_baseline and the per-agent active-override uniques are
-- untouched - they predate 0010 and belong to the catalogue carve-out.

BEGIN;

DROP INDEX IF EXISTS uq_prompt_plan_override_version;

COMMIT;
