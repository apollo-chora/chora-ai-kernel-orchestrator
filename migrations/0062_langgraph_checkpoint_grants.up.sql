-- 0062_langgraph_checkpoint_grants — grant the app roles on the LangGraph
-- checkpoint tables created by 0061.
--
-- Why this is a separate migration: 9999_grant_app_roles is applied-once, so
-- its `GRANT ... ON ALL TABLES IN SCHEMA public` ran BEFORE 0061 created these
-- tables and therefore did not cover them. (On a fresh database the ordering
-- still holds — 9999 sorts last and would cover them — so this file is
-- belt-and-braces there and load-bearing on an existing database.)
--
-- Idempotent: GRANT is idempotent by design.

GRANT SELECT, INSERT, UPDATE, DELETE ON
    checkpoint_migrations,
    checkpoints,
    checkpoint_blobs,
    checkpoint_writes
TO chora_ai_kernel_app_rw;

GRANT SELECT ON
    checkpoint_migrations,
    checkpoints,
    checkpoint_blobs,
    checkpoint_writes
TO chora_ai_kernel_app_ro;
