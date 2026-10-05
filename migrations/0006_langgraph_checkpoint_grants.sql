-- 0006_langgraph_checkpoint_grants — reassign LangGraph checkpoint tables
-- to chora_ai_kernel_migrate + grant app-role access.
--
-- BUG (pre-fix):
--   The four LangGraph PostgresSaver runtime tables (checkpoints,
--   checkpoint_blobs, checkpoint_writes, checkpoint_migrations) are
--   created on first call to PostgresSaver.setup(). In the M14.2 wave
--   the first caller was chora-closure-orchestrator (Phase-D POC) which
--   shared the chora_ai_kernel DB temporarily — so the tables were
--   owned by `chora_closure_poc`, NOT `chora_ai_kernel_migrate`.
--
--   chora_ai_kernel_app_rw inherits its grants from 9999 +
--   ALTER DEFAULT PRIVILEGES tied to `chora_ai_kernel_migrate` — so
--   tables owned by `chora_closure_poc` got NO grants. Result: the
--   orchestrator's chora_ai_kernel_app_rw connection raised
--   `psycopg.errors.InsufficientPrivilege: permission denied for table
--   checkpoints` on the first aget_tuple call after a qgen crew message
--   started running the LangGraph graph.
--
--   Surfaced 2026-05-17 evening as the third layer of
--   E2E-BE-MCQ-AI-ASSIST-ORCH-CONSUME (after the proto-encoder + Lazy
--   async-sync delegation fixes). Symmetric to the resume-prompt §2
--   ownership-reassign work for ai_kernel_outbox_* tables.
--
-- Fix:
--   1) ALTER TABLE OWNER → chora_ai_kernel_migrate so the canonical
--      9999_grant_app_roles default-privileges chain covers future
--      langgraph schema migrations too.
--   2) Explicit GRANT to app_rw + app_ro on the four runtime tables for
--      this revision (idempotent — re-applying is safe).
--
-- Idempotency: ALTER TABLE OWNER + GRANT are both idempotent under
-- IF EXISTS guards (Postgres does not natively support
-- `ALTER TABLE IF EXISTS ... OWNER TO`, so the DO-block form is used
-- to skip absent tables — safe when migration runs against a fresh DB
-- where langgraph hasn't yet created the tables).
--
-- Applied 2026-05-17 session-time via the postgres root account through
-- the orchestrator pod's cloudsql-proxy port-forward (the migrate role
-- could not reassign tables it didn't own). Tracker row inserted at
-- session-time so future Cloud Run Job apply is a no-op.

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'checkpoints',
        'checkpoint_blobs',
        'checkpoint_writes',
        'checkpoint_migrations'
    ]
    LOOP
        IF EXISTS (SELECT 1 FROM pg_class WHERE relname = t AND relnamespace = 'public'::regnamespace) THEN
            EXECUTE format('ALTER TABLE public.%I OWNER TO chora_ai_kernel_migrate', t);
        END IF;
    END LOOP;
END
$$;

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'checkpoints',
        'checkpoint_blobs',
        'checkpoint_writes',
        'checkpoint_migrations'
    ]
    LOOP
        IF EXISTS (SELECT 1 FROM pg_class WHERE relname = t AND relnamespace = 'public'::regnamespace) THEN
            EXECUTE format(
                'GRANT SELECT, INSERT, UPDATE, DELETE ON public.%I TO chora_ai_kernel_app_rw',
                t
            );
            EXECUTE format(
                'GRANT SELECT ON public.%I TO chora_ai_kernel_app_ro',
                t
            );
        END IF;
    END LOOP;
END
$$;
