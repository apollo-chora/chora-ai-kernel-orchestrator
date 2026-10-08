-- 0061_langgraph_checkpoint_tables — pre-create the LangGraph PostgresSaver
-- runtime tables so the app role never needs CREATE on schema public.
--
-- BUG (pre-fix):
--   AsyncPostgresSaver.setup() issues `CREATE TABLE IF NOT EXISTS` at first
--   use. In the compose stack the orchestrator connects as
--   chora_ai_kernel_app_rw, which has USAGE but NOT CREATE on schema public
--   (the PostgreSQL 15+ default). The qgen lane builds the transactional
--   saver during lifespan startup, so an enabled qgen crew aborts the whole
--   orchestrator:
--     psycopg.errors.InsufficientPrivilege: permission denied for schema public
--     LINE 1: CREATE TABLE IF NOT EXISTS checkpoint_migrations (
--
--   The GKE deployment never hit this because the tables were created once,
--   out-of-band, by a role that could (see 0006_langgraph_checkpoint_grants).
--
-- Fix:
--   1) Create the four runtime tables + their indexes here, as the migrate
--      role (the schema owner), so they exist before the app connects.
--   2) Seed checkpoint_migrations with every version the installed langgraph
--      ships (0-9) so setup() sees them all applied and issues no DDL of its
--      own. The column set below is the post-migration shape (blob nullable
--      per v4, task_path per v9), so a fresh create lands on the final schema.
--   3) 9999_grant_app_roles runs afterwards and grants ON ALL TABLES IN SCHEMA
--      public, covering these tables.
--
-- Idempotent: IF NOT EXISTS + ON CONFLICT DO NOTHING.

CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);

INSERT INTO checkpoint_migrations (v)
VALUES (0), (1), (2), (3), (4), (5), (6), (7), (8), (9)
ON CONFLICT (v) DO NOTHING;
