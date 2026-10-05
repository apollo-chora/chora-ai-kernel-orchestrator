"""RED: migration 0056: the park ledger + the outbox failed-row retry index.

Guards the file against the posture drift that already exists in this service
(two tenant GUC names, a non-NULLIF-safe policy in 0001): the ledger follows
0055 exactly (``chora.tenant_id``, NULLIF-safe, sweeper mode on an unset GUC,
``WITH CHECK``, idempotent DDL, no ``deleted_at`` because this is operational
machinery, grants via 9999).
"""

from __future__ import annotations

import re
from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
UP = MIGRATIONS / "0056_agent_dispatch_parks.up.sql"
DOWN = MIGRATIONS / "0056_agent_dispatch_parks.down.sql"


def test_up_and_down_exist_and_the_number_is_unique() -> None:
    assert UP.is_file(), UP
    assert DOWN.is_file(), DOWN
    assert sorted(p.name for p in MIGRATIONS.glob("0056_*")) == [DOWN.name, UP.name]


def test_up_creates_the_ledger_with_the_0055_posture() -> None:
    text = UP.read_text()
    assert "BEGIN;" in text and "COMMIT;" in text
    assert "CREATE TABLE IF NOT EXISTS ai_kernel_agent_dispatch_parks" in text
    for column in (
        "idempotency_key",
        "workflow_id",
        "thread_id",
        "crew",
        "tenant_id",
        "gcid",
        "agent_role",
        "request_topic",
        "completion_topic",
        "traceparent",
        "tracestate",
        "parked_at",
        "deadline_at",
        "state",
        "settled_at",
        "settled_by",
        "late_completion_at",
    ):
        assert re.search(rf"^\s*{column}\s", text, re.MULTILINE), column
    assert "CHECK (state IN ('parked','completed','reaped'))" in text.replace(
        "CHECK (state IN ('parked', 'completed', 'reaped'))",
        "CHECK (state IN ('parked','completed','reaped'))",
    )
    assert "ALTER TABLE ai_kernel_agent_dispatch_parks ENABLE ROW LEVEL SECURITY" in text
    assert "NULLIF(current_setting('chora.tenant_id', TRUE), '')" in text
    assert "WITH CHECK" in text
    assert "duplicate_object" in text
    assert "app.current_tenant" not in text, "the outbox's other GUC name must not spread"
    assert "deleted_at" not in text, "operational table: no soft-delete column"


def test_up_indexes_the_two_scans_the_reaper_and_the_router_run() -> None:
    text = UP.read_text()
    assert re.search(r"CREATE INDEX IF NOT EXISTS \w+\s+ON ai_kernel_agent_dispatch_parks \(state, deadline_at\)", text)
    assert re.search(
        r"CREATE INDEX IF NOT EXISTS \w+\s+ON ai_kernel_agent_dispatch_parks \(tenant_id, agent_role, state\)",
        text,
    )
    assert re.search(r"CREATE INDEX IF NOT EXISTS \w+\s+ON ai_kernel_agent_dispatch_parks \(thread_id\)", text)


def test_up_adds_the_failed_row_retry_index_on_the_outbox() -> None:
    """The dispatcher starts re-reading status='failed' rows (ADR-254 D5); the
    existing partial index only covers 'pending'."""
    text = UP.read_text()
    assert re.search(
        r"CREATE INDEX IF NOT EXISTS \w+\s+ON ai_kernel_outbox_events \(last_attempt_at ASC\) WHERE status = 'failed'",
        text,
    )


def test_down_drops_only_what_up_created() -> None:
    text = DOWN.read_text()
    assert "DROP TABLE IF EXISTS ai_kernel_agent_dispatch_parks" in text
    assert re.search(r"DROP INDEX IF EXISTS \w*failed\w*", text)
    assert "DROP TABLE IF EXISTS ai_kernel_outbox_events" not in text
