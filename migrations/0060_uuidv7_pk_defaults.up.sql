-- =============================================================================
-- chora-ai-kernel-orchestrator : 0060_uuidv7_pk_defaults.up.sql
--
-- Database : chora_ai_kernel
-- Context  : Owner ruling 2026-09-01. The UUIDv7 invariant in CLAUDE.md and
--            .claude/rules/ddd-enforcement.md held at the APPLICATION layer
--            only: a live census on 2026-09-01 counted 3,882 uuid.NewV7() call
--            sites in Go against 6 primary-key columns in this database still
--            defaulting to gen_random_uuid(), which is a v4. A row inserted
--            without an explicit id, by a seed, a backfill or hand SQL, got a
--            v4 and nothing said so. PostgreSQL 18 ships a native uuidv7(), and
--            this instance is 18.4, so the invariant can now hold at both
--            layers instead of one.
--
-- SAFETY
-- ------
-- Metadata-only. Changing a column DEFAULT rewrites no rows and touches no
-- index; existing v4 keys keep their values and stay valid. Only rows inserted
-- AFTER this migration get a v7. The two id shapes coexist, which is expected:
-- uuid is uuid, and nothing in the schema or the Go code parses a version out
-- of a key.
--
-- ⚠ IDEMPOTENT BY CONSTRUCTION, and it has to be. This selects the columns to
-- change at run time rather than naming them, so a re-run finds nothing left at
-- gen_random_uuid() and is a no-op. That matters because the apply job replays
-- from a GCS snapshot rather than from git, so a file can legitimately run
-- twice, and because any table added between authoring and applying is picked
-- up rather than missed.
--
-- ⚠ Column count is NOT asserted. It was measured live at authoring time and is
-- recorded above for provenance only. Asserting it would make this migration
-- fail on any table added or dropped in the interim, which is a worse failure
-- than a drifting comment.
-- =============================================================================

BEGIN;

SET LOCAL lock_timeout = '5s';

-- Fail loud rather than silently leaving every default at v4 on an older server.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'uuidv7') THEN
    RAISE EXCEPTION '0060: uuidv7() is not available on this server; PostgreSQL 18+ is required';
  END IF;
END $$;

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
      AND c.column_default = 'gen_random_uuid()'
  LOOP
    EXECUTE format('ALTER TABLE %I.%I ALTER COLUMN %I SET DEFAULT uuidv7()', r.s, r.t, r.col);
    n := n + 1;
  END LOOP;
  RAISE NOTICE '0060: switched % primary-key uuid default(s) from gen_random_uuid() to uuidv7()', n;
END $$;

-- Fail loud rather than reporting success on a server that accepted nothing.
DO $$
DECLARE
  leftover integer;
BEGIN
  SELECT count(*) INTO leftover
    FROM information_schema.columns
   WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
     AND data_type = 'uuid'
     AND column_default = 'gen_random_uuid()';
  IF leftover > 0 THEN
    RAISE EXCEPTION '0060: % uuid column(s) still default to gen_random_uuid() after the sweep', leftover;
  END IF;
END $$;

COMMIT;
