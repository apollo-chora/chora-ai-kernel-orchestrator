"""RED: migration 0058: the companion_chat session TABLES with the owner-ruled
tenant column + RLS (ADR-254 D6, owner MCQ at G1a 2026-08-22).

WP-A's chat binary runs NO DDL: its boot asserts from pg_catalog that the four
tables exist in companion_chat_sessions with a tenant_id column, rowsecurity AND
forcerowsecurity, and a policy; so this migration pre-creates the four
gorm-derived tables (DDL handed over by WP-A, adk v1.2.1-0.20260508) and shapes
them per the ruling. The agent sets chora.tenant_id on a dedicated connection
around every session-service call, so the policy is STRICT (no sweeper mode:
an unset GUC on chat content must fail, never read every tenant).
"""

from __future__ import annotations

import re
from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
UP = MIGRATIONS / "0058_companion_chat_sessions_tables.up.sql"
DOWN = MIGRATIONS / "0058_companion_chat_sessions_tables.down.sql"
GRANTS = MIGRATIONS / "9999_grant_app_roles.sql"

TABLES = ("sessions", "events", "app_states", "user_states")
STRICT_GUC = "(current_setting('chora.tenant_id'))::uuid"


def test_up_and_down_exist_and_the_number_is_unique() -> None:
    assert UP.is_file(), UP
    assert DOWN.is_file(), DOWN
    assert sorted(p.name for p in MIGRATIONS.glob("0058_*")) == [DOWN.name, UP.name]


def test_up_creates_the_four_gorm_tables_schema_qualified() -> None:
    text = UP.read_text()
    assert "BEGIN;" in text and "COMMIT;" in text
    for table in TABLES:
        assert f"CREATE TABLE IF NOT EXISTS companion_chat_sessions.{table} (" in text, table
    assert "SET search_path" not in text, "schema-qualify; never lean on search_path in a migration"
    # the library's column shapes (gorm postgres dialector), spot-checked
    assert re.search(r"app_name\s+text\s+NOT NULL", text)
    assert re.search(r"create_time\s+timestamptz\(6\)", text)
    assert re.search(r"actions\s+bytea", text)
    assert re.search(r"content\s+jsonb", text)
    assert re.search(r"turn_complete\s+boolean", text)
    fk = "REFERENCES companion_chat_sessions.sessions (app_name, user_id, id) ON DELETE CASCADE"
    assert fk in text


def test_every_table_gets_the_tenant_column_with_the_guc_default() -> None:
    code = "\n".join(line for line in UP.read_text().splitlines() if not line.lstrip().startswith("--"))
    pattern = r"tenant_id\s+uuid\s+NOT NULL\s+DEFAULT \(current_setting\('chora\.tenant_id'\)\)::uuid"
    occurrences = re.findall(pattern, code)
    assert len(occurrences) == len(TABLES), occurrences


def test_app_states_is_keyed_per_tenant() -> None:
    """The library keys app_states on app_name alone; under RLS two tenants'
    rows would share one key space and the second tenant's first upsert would
    hit the PK. The library never names the PK, so (app_name, tenant_id) is
    tolerated and is the shape that cannot collide."""
    text = UP.read_text()
    block = text[text.index("companion_chat_sessions.app_states (") :]
    block = block[: block.index(");")]
    assert "PRIMARY KEY (app_name, tenant_id)" in block


def test_rls_is_enabled_and_forced_with_a_strict_policy_on_every_table() -> None:
    text = UP.read_text()
    for table in TABLES:
        assert f"ALTER TABLE companion_chat_sessions.{table} ENABLE ROW LEVEL SECURITY" in text, table
        assert f"ALTER TABLE companion_chat_sessions.{table} FORCE ROW LEVEL SECURITY" in text, table
        assert re.search(
            rf"CREATE POLICY tenant_isolation ON companion_chat_sessions\.{table}\s+"
            rf"USING \(tenant_id = {re.escape(STRICT_GUC)}\)\s+"
            rf"WITH CHECK \(tenant_id = {re.escape(STRICT_GUC)}\)",
            text,
        ), table
    # STRICT on purpose: no sweeper mode, no NULLIF, no platform-wide always-true clause
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("--"))
    assert "NULLIF(current_setting('chora.tenant_id'" not in code
    assert "USING (true)" not in code
    assert "app.current_tenant" not in code
    assert "duplicate_object" in text, "idempotent policy creation (re-apply safe)"


def test_up_grants_app_rw_and_restates_the_schema_comment() -> None:
    text = UP.read_text()
    grant = (
        r"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA companion_chat_sessions"
        r"\s+TO chora_ai_kernel_app_rw"
    )
    assert re.search(grant, text)
    assert "COMMENT ON SCHEMA companion_chat_sessions IS" in text
    # 0057's comment said user_id = gcid; the binary keys "{tenant_id}:{gcid}"
    assert "{tenant_id}:{gcid}" in text and "user_id gcid" not in text


def test_9999_reasserts_the_schema_grant() -> None:
    text = GRANTS.read_text()
    assert "ON ALL TABLES IN SCHEMA companion_chat_sessions" in text
    assert "GRANT USAGE ON SCHEMA companion_chat_sessions TO chora_ai_kernel_app_rw" in text


def test_down_refuses_to_drop_populated_chat_history() -> None:
    text = DOWN.read_text()
    assert "RAISE EXCEPTION" in text, "a rollback must refuse while chat rows exist"
    for table in TABLES:
        assert f"DROP TABLE IF EXISTS companion_chat_sessions.{table}" in text, table
    assert "CASCADE" not in text.replace("ON DELETE CASCADE", "")


def test_no_em_dash() -> None:
    for p in (UP, DOWN):
        assert "\u2014" not in p.read_text(), p
