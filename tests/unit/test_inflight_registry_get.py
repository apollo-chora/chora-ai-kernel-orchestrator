"""RED: InflightRegistry.get (ADR-254 D5, the qgen flip).

A completion resumes a PARKED qgen job; the runner that settles it needs the
job's started payload back (traceparent, requested_count, type_plan, regen
spec, source_files), and the in-flight registry row is the durable copy that
outlives the drive. ``get`` reads ONE row by assist_id; None when the job is
no longer in flight.
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

    def cursor(self) -> _Cursor:
        return _Cursor(self._rows, self.log)

    async def commit(self) -> None:
        self.commits += 1


@pytest.mark.asyncio
async def test_get_returns_the_started_payload_row() -> None:
    started = {"assist_id": "a-1", "tenant_id": "t", "author_gcid": "g", "prompt": "P", "type_plan": []}
    conn = _Conn(rows=[("a-1", "t", "g", json.dumps(started), 2)])
    row = await InflightRegistry(conn=conn).get("a-1")
    assert row is not None
    assert row["assist_id"] == "a-1" and row["tenant_id"] == "t" and row["author_gcid"] == "g"
    assert row["started_payload"] == started and row["resume_count"] == 2
    sql, params = conn.log[0]
    assert "ai_assist_inflight_jobs" in sql and "assist_id = %(assist_id)s" in sql
    assert params == {"assist_id": "a-1"}
    assert conn.commits == 1, "a read on the shared connection releases its transaction"


@pytest.mark.asyncio
async def test_get_returns_none_when_not_in_flight() -> None:
    conn = _Conn(rows=[])
    assert await InflightRegistry(conn=conn).get("missing") is None


@pytest.mark.asyncio
async def test_get_accepts_a_jsonb_dict_payload() -> None:
    conn = _Conn(rows=[("a-2", "t", "g", {"assist_id": "a-2"}, 0)])
    row = await InflightRegistry(conn=conn).get("a-2")
    assert row is not None and row["started_payload"] == {"assist_id": "a-2"}
