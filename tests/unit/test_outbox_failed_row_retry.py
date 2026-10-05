"""RED: ADR-254 D5, the outbox ``failed`` status stops being a black hole.

Measured on 2026-08-22: ``PostgresOutboxStore.mark_failed`` writes
``status='failed'`` and ``fetch_pending`` reads ONLY ``status='pending'``, and
nothing anywhere converts one back. One publish failure therefore strands the
row forever: never retried, ``retry_count`` never exceeds 1, and the
dispatcher's ``deadletter()`` branch is unreachable, so
``ai_kernel_outbox_dead_letters`` is dead code. For a dispatch-request row
that means the run parks forever with no DLQ message anywhere.

The contract pinned here: ``fetch_pending`` also returns ``failed`` rows once
their backoff has elapsed (10 s doubling to a 600 s cap, mirroring the lane
subscriptions' ``retry_policy``), oldest first, so the existing dispatcher loop
retries them and dead-letters them at ``max_attempts`` exactly as the in-memory
store already behaves. ``deadlettered`` and ``published`` rows are never
re-read.
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.store import (
    FAILED_RETRY_BASE_SECONDS,
    FAILED_RETRY_MAX_SECONDS,
    PostgresOutboxStore,
)


class _FakeCursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        self._conn.executed.append((" ".join(sql.split()), params))

    async def fetchall(self) -> list[Any]:
        return []


class _FakeConn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.commits = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    async def commit(self) -> None:
        self.commits += 1


def test_backoff_mirrors_the_lane_retry_policy() -> None:
    """agent_dispatch_lanes.tf: retry_policy minimum_backoff 10s, maximum 600s."""
    assert FAILED_RETRY_BASE_SECONDS == 10
    assert FAILED_RETRY_MAX_SECONDS == 600


@pytest.mark.asyncio
async def test_fetch_pending_also_reads_failed_rows_once_their_backoff_elapsed() -> None:
    conn = _FakeConn()
    store = PostgresOutboxStore(conn=conn, worker_id="w1")

    rows = await store.fetch_pending(limit=100)

    assert rows == []
    sql, params = conn.executed[0]
    assert "status = 'pending'" in sql
    assert "status = 'failed'" in sql
    assert "last_attempt_at IS NULL" in sql
    assert "LEAST(%(retry_max_s)s, %(retry_base_s)s * power(2, GREATEST(retry_count, 1) - 1))" in sql
    assert "< now()" in sql
    assert "ORDER BY occurred_at ASC" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "'deadlettered'" not in sql
    assert "'published'" not in sql
    assert params == {
        "limit": 100,
        "retry_base_s": FAILED_RETRY_BASE_SECONDS,
        "retry_max_s": FAILED_RETRY_MAX_SECONDS,
    }
    # The empty path still releases the read transaction with COMMIT.
    assert conn.commits == 1
