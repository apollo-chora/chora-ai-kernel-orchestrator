-- Destructive rollback: 1024 -> 768 discards every stored decision embedding.
DROP INDEX IF EXISTS idx_agent_dec_emb_cosine;
ALTER TABLE agent_decision_embeddings ALTER COLUMN embedding TYPE vector(768);
CREATE INDEX idx_agent_dec_emb_cosine
    ON agent_decision_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
