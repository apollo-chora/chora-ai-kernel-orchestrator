"""RED: migration 0057: the ``companion_chat_sessions`` schema (ADR-254 D6).

The companion_chat agent keeps its ADK sessions in Postgres, in a dedicated
schema of ``chora_ai_kernel``; the library auto-migrates its own tables inside
that schema on first boot over a connection whose ``search_path`` is the
schema. The kennel owns ``chora_ai_kernel``'s migrations, so the schema and the
grant the agent's role needs to create tables in it live here. Binding relay
from the coordinator, 2026-08-22 16:33Z: CREATE SCHEMA + GRANT USAGE, CREATE
to ``chora_ai_kernel_app_rw``; no tenant column and no RLS on the library-owned
tables (provisional ruling pending the owner MCQ).

The DOWN is deliberately RESTRICT (no CASCADE): a rollback that would destroy
learner chat history refuses loudly instead of dropping it.
"""

from __future__ import annotations

import re
from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
UP = MIGRATIONS / "0057_companion_chat_sessions_schema.up.sql"
DOWN = MIGRATIONS / "0057_companion_chat_sessions_schema.down.sql"


def test_up_and_down_exist_and_the_number_is_unique() -> None:
    assert UP.is_file(), UP
    assert DOWN.is_file(), DOWN
    assert sorted(p.name for p in MIGRATIONS.glob("0057_*")) == [DOWN.name, UP.name]


def test_up_creates_the_schema_and_grants_usage_create_to_app_rw() -> None:
    text = UP.read_text()
    assert "BEGIN;" in text and "COMMIT;" in text
    assert re.search(r"CREATE SCHEMA IF NOT EXISTS companion_chat_sessions\s*;", text)
    assert re.search(
        r"GRANT USAGE, CREATE ON SCHEMA companion_chat_sessions TO chora_ai_kernel_app_rw\s*;",
        text,
    )
    # The agent's session tables are library-owned: the kennel creates NO table
    # in the schema, so nothing here may look like a kennel-owned table.
    assert "CREATE TABLE" not in text
    assert "ROW LEVEL SECURITY" not in text
    assert "app.current_tenant" not in text


def test_up_records_the_pii_and_isolation_posture_in_its_header() -> None:
    text = UP.read_text()
    # The schema holds learner chat content: the closure map entry exists and
    # the migration says so; the per-tenant isolation is the ADK app_name key.
    assert "PII_Closure_Map" in text
    assert "companion_chat:{tenant_id}" in text
    assert "ADR-254" in text


def test_down_is_restrict_not_cascade() -> None:
    text = DOWN.read_text()
    assert "REVOKE USAGE, CREATE ON SCHEMA companion_chat_sessions FROM chora_ai_kernel_app_rw" in text
    assert re.search(r"DROP SCHEMA IF EXISTS companion_chat_sessions\s*;", text)
    assert not re.search(r"DROP SCHEMA[^;]*CASCADE", text), "a rollback must never destroy learner chat history"


def test_no_em_dash_anywhere() -> None:
    for p in (UP, DOWN):
        assert "\u2014" not in p.read_text(), p
