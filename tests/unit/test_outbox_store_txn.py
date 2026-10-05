"""Tests for ``PostgresOutboxStore.fetch_pending`` transaction lifecycle.

Regression for the outbox-dispatcher wedge surfaced 2026-06-02: the
dispatcher shares ONE psycopg AsyncConnection (autocommit OFF) across the
outbox store + qgen subscriber + checkpointer. ``fetch_pending`` runs
``SELECT ... FOR UPDATE SKIP LOCKED``, which OPENS a transaction. When there
are ZERO pending rows the SELECT returns empty and — pre-fix — nothing
commits/rolls back, so the connection is left "idle in transaction"
indefinitely, wedging every other user of the shared connection.

The contract these tests pin:

* On an EMPTY result, ``fetch_pending`` MUST release the read transaction
  (commit or rollback) so the shared connection returns to idle. No locks
  were acquired (zero rows), so a rollback is the cheapest safe release.
* The non-empty path is unchanged here: the dispatcher still holds the
  ``FOR UPDATE SKIP LOCKED`` locks until ``mark_published``/``mark_failed``
  commits per row (covered by test_outbox_dispatcher.py). These tests only
  pin the empty-result release so the wedge cannot recur.
"""

from __future__ import annotations

from typing import Any

from chora_ai_kernel_orchestrator.adapter.pubsub.store import PostgresOutboxStore


class _FakeCursor:
    """Async-context-manager cursor stand-in. Records executed SQL +
    returns the configured rows from fetchall()."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.executed: list[str] = []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append(sql)

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeConn:
    """Async psycopg-connection stand-in that records commit/rollback so a
    test can assert the read transaction is released."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.commits = 0
        self.rollbacks = 0
        self.last_cursor: _FakeCursor | None = None

    def cursor(self) -> _FakeCursor:
        self.last_cursor = _FakeCursor(self._rows)
        return self.last_cursor

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


async def test_fetch_pending_releases_txn_when_empty() -> None:
    """Empty outbox: fetch_pending MUST release the dangling read txn."""
    conn = _FakeConn(rows=[])
    store = PostgresOutboxStore(conn=conn, worker_id="w1")

    rows = await store.fetch_pending(limit=100)

    assert rows == []
    # The wedge fix: on an empty result the read transaction is released
    # (commit OR rollback) so the shared connection returns to idle.
    assert (conn.commits + conn.rollbacks) >= 1, (
        "fetch_pending left the connection idle-in-transaction on an empty "
        "result — this is the dispatcher wedge regression"
    )


async def test_fetch_pending_does_not_hold_txn_open_across_calls() -> None:
    """Repeated empty drains MUST NOT accumulate open transactions — each
    empty fetch releases its own read txn."""
    conn = _FakeConn(rows=[])
    store = PostgresOutboxStore(conn=conn, worker_id="w1")

    for _ in range(3):
        assert await store.fetch_pending(limit=100) == []

    # One release per empty drain cycle.
    assert (conn.commits + conn.rollbacks) >= 3
