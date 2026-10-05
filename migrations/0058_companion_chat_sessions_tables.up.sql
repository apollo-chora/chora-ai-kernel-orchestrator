-- =============================================================================
-- chora-ai-kernel-orchestrator : 0058_companion_chat_sessions_tables.up.sql
-- Database  : chora_ai_kernel, schema companion_chat_sessions (created by 0057)
-- ADR       : ADR-254 D6 + owner MCQ ruling at G1a (2026-08-22): the
--             companion_chat ADK session tables carry a TENANT COLUMN + RLS
--             (tenant-in-key declined). Relayed by the coordinator 16:50Z;
--             DDL handed over by WP-A (the chat agent's owner) 17:08Z.
--
-- What this does
--   Pre-creates the four tables the google.golang.org/adk session/database
--   library (adk v1.2.1-0.20260508) expects, with EXACTLY the column shapes
--   gorm's postgres dialector derives from the library's models (string ->
--   text, time.Time precision:6 -> timestamptz(6), []byte -> bytea, *bool ->
--   boolean, the JSON value types -> jsonb), PLUS on every table
--     tenant_id uuid NOT NULL DEFAULT (current_setting('chora.tenant_id'))::uuid
--   ENABLE + FORCE ROW LEVEL SECURITY and a STRICT tenant policy for the app
--   role. The agent runs NO DDL, ever: its boot asserts this shape from
--   pg_catalog (four tables, the tenant_id column, relrowsecurity AND
--   relforcerowsecurity, at least one policy) and refuses to serve otherwise.
--   Owner of the tables = the migrate role, which is what makes FORCE bind the
--   app role (a table owner bypasses RLS; app_rw is not the owner).
--
-- Tenant GUC discipline (WP-A, binding)
--   Every session-service call checks out a DEDICATED pooled connection, runs
--   set_config('chora.tenant_id', <tenant>, false), reads it back (mismatch =
--   error), binds the library to that one connection, runs the call, RESETs the
--   GUC and returns the connection. The tenant is parsed from the ADK user_id
--   "{tenant_id}:{gcid}" and must be a UUID or the call is refused. So the
--   policy here is STRICT on purpose: NO sweeper mode, NO NULLIF, NO
--   platform-wide always-true clause (the ADR-184 closure_saga_platform_rw shape on
--   the closure tables would let any app_rw connection without a GUC read
--   every tenant's chat, the opposite of the ruling). An unset GUC FAILS
--   (unrecognized configuration parameter), never reads zero rows silently,
--   and the DEFAULT refuses an INSERT without a tenant. The closure
--   pseudonymise (config/PII_Closure_Map.yaml, keyed (tenant_id, user_id))
--   must set the GUC per tenant to see its rows.
--
-- Keys (as built by WP-A): app_name = the constant "companion_chat",
--   user_id = "{tenant_id}:{gcid}", session_id = conversation_id.
--   app_states: the library keys it on app_name alone; under RLS two tenants'
--   rows would share one key space and the second tenant's first upsert would
--   hit the PK, so the PK here is (app_name, tenant_id), which the library
--   tolerates (it never names the PK; WP-A's recommendation).
--
-- Grants
--   9999_grant_app_roles.sql covers schema public; the schema-level grant for
--   companion_chat_sessions is here (idempotent) AND re-asserted in 9999 so a
--   full run keeps parity. The agent's role creates nothing in the schema.
--
-- Idempotent: CREATE TABLE IF NOT EXISTS, DO $$ duplicate_object for the
-- policies, GRANT is a no-op on re-apply. HARD RULE: chora_ai_kernel-local
-- only; cross-database queries forbidden.
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS companion_chat_sessions.sessions (
  app_name    text NOT NULL,
  user_id     text NOT NULL,
  id          text NOT NULL,
  state       jsonb,
  create_time timestamptz(6),
  update_time timestamptz(6),
  tenant_id   uuid NOT NULL DEFAULT (current_setting('chora.tenant_id'))::uuid,
  PRIMARY KEY (app_name, user_id, id)
);

CREATE TABLE IF NOT EXISTS companion_chat_sessions.events (
  id                          text NOT NULL,
  app_name                    text NOT NULL,
  user_id                     text NOT NULL,
  session_id                  text NOT NULL,
  invocation_id               text,
  author                      text,
  actions                     bytea,
  long_running_tool_ids_json  jsonb,
  branch                      text,
  timestamp                   timestamptz(6),
  content                     jsonb,
  grounding_metadata          jsonb,
  custom_metadata             jsonb,
  usage_metadata              jsonb,
  citation_metadata           jsonb,
  partial                     boolean,
  turn_complete               boolean,
  error_code                  text,
  error_message               text,
  interrupted                 boolean,
  tenant_id                   uuid NOT NULL DEFAULT (current_setting('chora.tenant_id'))::uuid,
  PRIMARY KEY (id, app_name, user_id, session_id),
  CONSTRAINT fk_sessions_events FOREIGN KEY (app_name, user_id, session_id)
    REFERENCES companion_chat_sessions.sessions (app_name, user_id, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS companion_chat_sessions.app_states (
  app_name    text NOT NULL,
  state       jsonb,
  update_time timestamptz(6),
  tenant_id   uuid NOT NULL DEFAULT (current_setting('chora.tenant_id'))::uuid,
  PRIMARY KEY (app_name, tenant_id)
);

CREATE TABLE IF NOT EXISTS companion_chat_sessions.user_states (
  app_name    text NOT NULL,
  user_id     text NOT NULL,
  state       jsonb,
  update_time timestamptz(6),
  tenant_id   uuid NOT NULL DEFAULT (current_setting('chora.tenant_id'))::uuid,
  PRIMARY KEY (app_name, user_id)
);

ALTER TABLE companion_chat_sessions.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE companion_chat_sessions.sessions FORCE ROW LEVEL SECURITY;
ALTER TABLE companion_chat_sessions.events ENABLE ROW LEVEL SECURITY;
ALTER TABLE companion_chat_sessions.events FORCE ROW LEVEL SECURITY;
ALTER TABLE companion_chat_sessions.app_states ENABLE ROW LEVEL SECURITY;
ALTER TABLE companion_chat_sessions.app_states FORCE ROW LEVEL SECURITY;
ALTER TABLE companion_chat_sessions.user_states ENABLE ROW LEVEL SECURITY;
ALTER TABLE companion_chat_sessions.user_states FORCE ROW LEVEL SECURITY;

DO $$
BEGIN
    CREATE POLICY tenant_isolation ON companion_chat_sessions.sessions
        USING (tenant_id = (current_setting('chora.tenant_id'))::uuid)
        WITH CHECK (tenant_id = (current_setting('chora.tenant_id'))::uuid);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE POLICY tenant_isolation ON companion_chat_sessions.events
        USING (tenant_id = (current_setting('chora.tenant_id'))::uuid)
        WITH CHECK (tenant_id = (current_setting('chora.tenant_id'))::uuid);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE POLICY tenant_isolation ON companion_chat_sessions.app_states
        USING (tenant_id = (current_setting('chora.tenant_id'))::uuid)
        WITH CHECK (tenant_id = (current_setting('chora.tenant_id'))::uuid);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE POLICY tenant_isolation ON companion_chat_sessions.user_states
        USING (tenant_id = (current_setting('chora.tenant_id'))::uuid)
        WITH CHECK (tenant_id = (current_setting('chora.tenant_id'))::uuid);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA companion_chat_sessions
  TO chora_ai_kernel_app_rw;

COMMENT ON SCHEMA companion_chat_sessions IS
  'ADR-254 D6 + owner ruling 2026-08-22: companion_chat ADK session tables, pre-created by migration 0058 with tenant_id + ENABLE/FORCE RLS (strict chora.tenant_id policy); the agent runs no DDL and asserts this shape at boot. Keys: app_name companion_chat, user_id {tenant_id}:{gcid}, session_id conversation_id. Learner PII: config/PII_Closure_Map.yaml, keyed (tenant_id, user_id); set the GUC per tenant to see rows.';

COMMIT;
