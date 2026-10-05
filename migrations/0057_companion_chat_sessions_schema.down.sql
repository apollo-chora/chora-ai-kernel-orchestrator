-- =============================================================================
-- chora-ai-kernel-orchestrator : 0057_companion_chat_sessions_schema.down.sql
--
-- Rollback of 0057. RESTRICT on purpose (the default, no cascading drop): the
-- DROP refuses loudly (2BP01 dependent objects) while the agent's session tables exist, so
-- a rollback can never destroy learner chat history by accident. Retire the
-- chat agent, migrate or crypto-shred the sessions first, then re-run.
-- =============================================================================

BEGIN;

REVOKE USAGE, CREATE ON SCHEMA companion_chat_sessions FROM chora_ai_kernel_app_rw;

DROP SCHEMA IF EXISTS companion_chat_sessions;

COMMIT;
