"""RED→GREEN: InflightRegistry.sweep() must RELEASE its read transaction.

``get()`` commits and documents exactly why ("Releases its read transaction
with a commit (the shared-connection rule: never an empty-path rollback)").
``sweep()`` — the boot resume scan, on the SAME shared connection — did not.
The sibling pattern: one method carried the lesson, its neighbour missed it.

⚠ THIS IS NOT A STYLE NIT, IT IS LIVE AND IT BLOCKS DDL. Observed on the
production database 2026-08-23T13:36Z, against kennel pod
chora-ai-kernel-orchestrator-8f57fdbfc-htbx9 (started 13:20:16Z, digest
de30b4d6ffcc):

    pid 63943  chora_ai_kernel_app_rw  "idle in transaction"  xact_age 15m18s
    query: SELECT assist_id, tenant_id, author_gcid, started_payload_json,
           resume_count FROM ai_assist_inflight_jobs ORDER BY accepted_at
    locks: AccessShareLock on ai_assist_inflight_jobs (+ pkey, + accepted_at idx)

``ORDER BY accepted_at`` with no WHERE is _SWEEP_SQL and nothing else, which is
what discriminates it from ``get()``. The boot sweep opens a transaction, never
commits, and the backend sits idle-in-transaction holding AccessShareLock for
the WHOLE LIFE OF THE POD.

Consequence that makes this a prerequisite rather than a cleanup: AccessShareLock
is incompatible with the ACCESS EXCLUSIVE that DROP POLICY / CREATE POLICY take,
so ANY policy DDL on this table blocks behind it indefinitely — and a queued
ACCESS EXCLUSIVE request also blocks every later lock request on the table,
stalling the kennel's own writes. A rolled-back dry run of the G2 fail-closed
migration hung on exactly this and had to be killed. So the G2 policy migration
CANNOT apply until this commit is in the running image.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.postgres.inflight_registry import (
    InflightRegistry,
)


class _Cursor:
    def __init__(self, rows: list[Any], log: list[tuple[str, Any]]) -> None:
        self._rows = rows
        self._log = log

    async def __aenter__(self) -> _Cursor:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        self._log.append((sql, params))

    async def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> list[Any]:
        return list(self._rows)


class _Conn:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows
        self.log: list[tuple[str, Any]] = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self._rows, self.log)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


def _row(assist_id: str, tenant: str) -> tuple[Any, ...]:
    payload = {"assist_id": assist_id, "tenant_id": tenant}
    return (assist_id, tenant, "g-1", json.dumps(payload), 0)


@pytest.mark.asyncio
async def test_sweep_releases_its_read_transaction_with_a_commit() -> None:
    """The boot sweep must not leave the backend idle-in-transaction."""
    conn = _Conn(rows=[_row("a-1", "t-1"), _row("a-2", "t-2")])
    rows = await InflightRegistry(conn=conn).sweep()
    assert len(rows) == 2
    assert conn.commits == 1, (
        "sweep() left its transaction OPEN: the backend sits idle-in-transaction "
        "holding AccessShareLock for the life of the pod and blocks all policy DDL"
    )


@pytest.mark.asyncio
async def test_sweep_releases_with_commit_even_when_empty() -> None:
    """The empty path must COMMIT, never ROLLBACK.

    A rollback on this shared connection destroys co-tenant writes made by the
    other adapters riding it (the 2026-08-14 incident that store.py carries).
    An empty sweep is the common case on a healthy boot, so it is the path most
    likely to be taken.
    """
    conn = _Conn(rows=[])
    rows = await InflightRegistry(conn=conn).sweep()
    assert rows == []
    assert conn.commits == 1
    assert conn.rollbacks == 0, "an empty-path ROLLBACK destroys co-tenant writes"
