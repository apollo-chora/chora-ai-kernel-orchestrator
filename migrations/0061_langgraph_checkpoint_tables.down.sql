-- 0061_langgraph_checkpoint_tables (down) — drop the LangGraph checkpoint
-- runtime tables created by the up migration.
--
-- DESTRUCTIVE: this discards every persisted LangGraph checkpoint (parked
-- dispatch state, crew run history). Only run it when the orchestrator is
-- stopped and the parked state is known to be disposable.

DROP TABLE IF EXISTS checkpoint_writes;
DROP TABLE IF EXISTS checkpoint_blobs;
DROP TABLE IF EXISTS checkpoints;
DROP TABLE IF EXISTS checkpoint_migrations;
