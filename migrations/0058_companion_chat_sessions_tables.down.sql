-- =============================================================================
-- chora-ai-kernel-orchestrator : 0058_companion_chat_sessions_tables.down.sql
--
-- Rollback of 0058. REFUSES while any of the four tables holds a row: a
-- rollback must never destroy learner chat history by accident (retire the chat
-- agent, migrate or crypto-shred the sessions first, then re-run). The check
-- runs as the migrate role (table owner, not subject to the policy) so it sees
-- every tenant's rows; FORCE RLS does not apply to the owner.
-- =============================================================================

BEGIN;

DO $$
DECLARE
  n_sessions bigint;
  n_events bigint;
  n_app bigint;
  n_user bigint;
BEGIN
  SELECT count(*) INTO n_sessions FROM companion_chat_sessions.sessions;
  SELECT count(*) INTO n_events FROM companion_chat_sessions.events;
  SELECT count(*) INTO n_app FROM companion_chat_sessions.app_states;
  SELECT count(*) INTO n_user FROM companion_chat_sessions.user_states;
  IF n_sessions + n_events + n_app + n_user > 0 THEN
    RAISE EXCEPTION
      'refusing to roll back 0058: companion_chat_sessions holds chat history (sessions=% events=% app_states=% user_states=%); retire the chat agent and shred the sessions first',
      n_sessions, n_events, n_app, n_user;
  END IF;
END $$;

REVOKE SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA companion_chat_sessions
  FROM chora_ai_kernel_app_rw;

DROP TABLE IF EXISTS companion_chat_sessions.events;
DROP TABLE IF EXISTS companion_chat_sessions.sessions;
DROP TABLE IF EXISTS companion_chat_sessions.app_states;
DROP TABLE IF EXISTS companion_chat_sessions.user_states;

COMMIT;
