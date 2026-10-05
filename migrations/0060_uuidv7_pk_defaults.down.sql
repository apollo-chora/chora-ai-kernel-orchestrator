-- =============================================================================
-- chora-ai-kernel-orchestrator : 0060_uuidv7_pk_defaults.down.sql
--
-- Reverts every primary-key uuid default from uuidv7() back to
-- gen_random_uuid(). Metadata-only and lossless: rows already written with a v7
-- key keep it, because a default only governs future inserts.
--
-- ⚠ This reverts columns that are AT uuidv7() now, which after the UP is the
-- same set the UP changed. If a later migration sets a v7 default on a new
-- table, this DOWN will revert that one too. That is deliberate: the DOWN
-- restores the pre-ruling convention wholesale rather than replaying history.
-- =============================================================================

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $$
DECLARE
  r record;
  n integer := 0;
BEGIN
  FOR r IN
    SELECT c.table_schema AS s, c.table_name AS t, c.column_name AS col
    FROM information_schema.columns c
    JOIN information_schema.key_column_usage k
      ON  k.table_schema = c.table_schema
      AND k.table_name   = c.table_name
      AND k.column_name  = c.column_name
    JOIN information_schema.table_constraints tc
      ON  tc.constraint_name = k.constraint_name
      AND tc.table_schema    = k.table_schema
      AND tc.constraint_type = 'PRIMARY KEY'
    WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
      AND c.data_type = 'uuid'
      AND c.column_default = 'uuidv7()'
  LOOP
    EXECUTE format('ALTER TABLE %I.%I ALTER COLUMN %I SET DEFAULT gen_random_uuid()', r.s, r.t, r.col);
    n := n + 1;
  END LOOP;
  RAISE NOTICE '0060: reverted % primary-key uuid default(s) to gen_random_uuid()', n;
END $$;

COMMIT;
