"""RED: the ledger store must hand back STRINGS for the UUID columns.

Measured live on 2026-08-22 (first reaper proof): psycopg returns the
``workflow_id`` / ``tenant_id`` / ``gcid`` columns of
``ai_kernel_agent_dispatch_parks`` as ``uuid.UUID`` objects; ``_to_record``
passed them through, ``run_failed_event`` copied them into the event body, and
the outbox writer's ``json.dumps`` raised ``TypeError: Object of type UUID is
not JSON serializable`` AFTER the row had already been marked reaped, so the
reap lost its ``run_failed.v1`` terminal. The unit fakes used strings and never
saw it. Pin the coercion here, against a DB-shaped row.
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.park_ledger import (
    PARK_COLUMNS,
    ParkLedgerStore,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.reaper import (
    ReaperArm,
    reaped_completion,
    run_failed_event,
)

_NOW = _dt.datetime(2026, 8, 22, 16, 0, tzinfo=_dt.UTC)


class _FakeCursor:
    def __init__(self, row: Any) -> None:
        self._row = row

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        return None

    async def fetchone(self) -> Any:
        return self._row

    async def fetchall(self) -> list[Any]:
        return [self._row]


class _FakeConn:
    def __init__(self, row: Any) -> None:
        self._row = row

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._row)

    async def commit(self) -> None:
        return None


def _db_shaped_row() -> tuple[Any, ...]:
    values: dict[str, Any] = {
        "idempotency_key": "agent_dispatch.oe_evaluate.x:tsq-1:1",
        "workflow_id": uuid.UUID("01a02062-e5b4-7870-8fca-53ce363cd542"),
        "thread_id": "01a02062-e5b4-7870-8fca-53ce363cd542",
        "crew": "oe_grading",
        "tenant_id": uuid.UUID("11111111-1111-7111-8111-111111111111"),
        "gcid": uuid.UUID("00000000-0000-7000-8000-000000001999"),
        "agent_role": "oe_evaluate",
        "request_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
        "completion_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1",
        "traceparent": None,
        "tracestate": None,
        "parked_at": _NOW,
        "deadline_at": _NOW + _dt.timedelta(days=7),
        "state": "parked",
        "settled_at": None,
        "settled_by": None,
        "late_completion_at": None,
    }
    return tuple(values[c] for c in PARK_COLUMNS)


@pytest.mark.asyncio
async def test_uuid_columns_come_back_as_strings_and_the_event_json_encodes() -> None:
    store = ParkLedgerStore(conn=_FakeConn(_db_shaped_row()))
    park = await store.get("agent_dispatch.oe_evaluate.x:tsq-1:1")
    assert park is not None
    assert isinstance(park.workflow_id, str) and park.workflow_id == "01a02062-e5b4-7870-8fca-53ce363cd542"
    assert isinstance(park.tenant_id, str) and park.tenant_id == "11111111-1111-7111-8111-111111111111"
    assert isinstance(park.gcid, str) and park.gcid == "00000000-0000-7000-8000-000000001999"
    assert park.traceparent == "" and park.tracestate == "" and park.settled_by == ""

    event = run_failed_event(
        park,
        arm=ReaperArm.REQUEST_DEAD_LETTERED,
        reason="r",
        original_topic="t",
        delivery_attempt=5,
        reaped_at=_NOW,
        source_project="chora-489812",
    )
    body = json.dumps(event["body"], separators=(",", ":"))
    envelope = json.dumps(event["envelope"])
    assert "01a02062-e5b4-7870-8fca-53ce363cd542" in body
    assert "11111111-1111-7111-8111-111111111111" in envelope

    completion = reaped_completion(park, arm=ReaperArm.REQUEST_DEAD_LETTERED, reason="r", reaped_at=_NOW)
    json.dumps(completion)


@pytest.mark.asyncio
async def test_fetch_paths_coerce_too() -> None:
    store = ParkLedgerStore(conn=_FakeConn(_db_shaped_row()))
    rows = await store.fetch_expired(limit=5)
    assert rows and isinstance(rows[0].tenant_id, str)
    rows = await store.fetch_outbox_dead_lettered(limit=5)
    assert rows and isinstance(rows[0].workflow_id, str)
