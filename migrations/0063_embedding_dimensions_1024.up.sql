-- Widen agent_decision_embeddings.embedding to the embedding route's native
-- width.
--
-- The deployment's embedding route (registry id `text-embedding-004`) resolves
-- to LiquidAI's LFM2.5 embedding model on OpenRouter, which returns 1024-dim
-- vectors and REJECTS a `dimensions` override ("produces 1024-dimensional
-- embeddings"). The pgvector column must match.
--
-- pgvector cannot cast between widths, so an ALTER on a POPULATED column
-- discards the stored vectors. This deployment has never persisted an agent
-- decision embedding (the table is empty), which is what makes this a pure
-- schema change; a populated deployment would need a re-embed pass instead.
--
-- The ivfflat index is dropped and rebuilt because its operator class is bound
-- to the column width. Recreated with the same shape as 0001_initial.sql.
DROP INDEX IF EXISTS idx_agent_dec_emb_cosine;
ALTER TABLE agent_decision_embeddings ALTER COLUMN embedding TYPE vector(1024);
CREATE INDEX idx_agent_dec_emb_cosine
    ON agent_decision_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
