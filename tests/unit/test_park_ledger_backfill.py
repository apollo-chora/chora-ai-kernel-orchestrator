"""RED: parks made BEFORE migration 0056 landed get a ledger row at startup.

The OE requests parked during the 2026-08-21 outage (and any run parked under
the ledger-less saver) have no ``ai_kernel_agent_dispatch_parks`` row, so the
deadline scan could never reap them. The backfill reads the outbox: every
published dispatch request whose completion was never consumed (no
``<key>.completed`` inbox key) and that has no ledger row yet is a park the
kennel is still holding. It is inserted with ``parked_at = occurred_at`` and a
deadline from the policy, ``ON CONFLICT DO NOTHING``, so running it on every
boot is safe.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.park_ledger import ParkLedgerStore
from chora_ai_kernel_orchestrator.domain.agent_dispatch.deadline_policy import (
    ParkDeadlinePolicy,
)

_OCCURRED = _dt.datetime(2026, 8, 21, 9, 0, tzinfo=_dt.UTC)


def _outbox_row(key: str, role: str = "oe_evaluate", body_extra: dict[str, Any] | None = None) -> tuple[Any, ...]:
    body = {
        "agent_role": role,
        "thread_id": "01a02062-e5b4-7870-8fca-53ce363cd542",
        "idempotency_key": key,
        "reply_topic": f"chora.ai_kernel.agent_dispatch.{role}_completed.v1",
        "traceparent": "00-aa-bb-01",
        "tracestate": "",
    }
    body.update(body_extra or {})
    return (
        key,
        "01a02062-e5b4-7870-8fca-53ce363cd542",
        "11111111-1111-7111-8111-111111111111",
        "00000000-0000-7000-8000-000000001999",
        f"chora.ai_kernel.agent_dispatch.{role}_requested.v1",
        json.dumps(body),
        _OCCURRED,
    )


class _FakeCursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self.rowcount = 1

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        self._conn.executed.append((" ".join(sql.split()), params))

    async def fetchall(self) -> list[Any]:
        return list(self._conn.candidates)


class _FakeConn:
    def __init__(self, candidates: list[tuple[Any, ...]]) -> None:
        self.candidates = candidates
        self.executed: list[tuple[str, Any]] = []
        self.commits = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    async def commit(self) -> None:
        self.commits += 1


@pytest.mark.asyncio
async def test_backfill_selects_published_requests_with_no_completion_and_no_ledger_row() -> None:
    conn = _FakeConn([])
    store = ParkLedgerStore(conn=conn)
    inserted = await store.backfill_from_outbox(
        crew_for_role={"oe_evaluate": "oe_grading"},
        deadline_policy=ParkDeadlinePolicy(),
    )
    assert inserted == 0
    sql, _ = conn.executed[0]
    assert "FROM ai_kernel_outbox_events o" in sql
    assert "o.event_type LIKE 'ai_kernel.agent_dispatch.%_requested'" in sql
    assert "o.status = 'published'" in sql
    assert "NOT EXISTS (SELECT 1 FROM idempotency_keys k WHERE k.key = o.idempotency_key || '.completed')" in sql
    assert (
        "NOT EXISTS (SELECT 1 FROM ai_kernel_agent_dispatch_parks p WHERE p.idempotency_key = o.idempotency_key)"
    ) in sql
    assert "convert_from(o.payload, 'UTF8')" in sql
    assert conn.commits >= 1


@pytest.mark.asyncio
async def test_backfill_inserts_one_ledger_row_per_candidate_with_the_policy_deadline() -> None:
    conn = _FakeConn(
        [
            _outbox_row("agent_dispatch.oe_evaluate.s1:q1:1"),
            _outbox_row("agent_dispatch.oe_moderate.s1:q1:1", role="oe_moderate"),
        ]
    )
    store = ParkLedgerStore(conn=conn)
    policy = ParkDeadlinePolicy(per_role_seconds={"oe_moderate": 3600})

    inserted = await store.backfill_from_outbox(
        crew_for_role={"oe_evaluate": "oe_grading", "oe_moderate": "oe_grading"},
        deadline_policy=policy,
    )

    assert inserted == 2
    inserts = [(s, p) for s, p in conn.executed if s.startswith("INSERT INTO ai_kernel_agent_dispatch_parks")]
    assert len(inserts) == 2
    first, second = inserts[0][1], inserts[1][1]
    assert first["idempotency_key"] == "agent_dispatch.oe_evaluate.s1:q1:1"
    assert first["crew"] == "oe_grading"
    assert first["agent_role"] == "oe_evaluate"
    assert first["thread_id"] == "01a02062-e5b4-7870-8fca-53ce363cd542"
    assert first["request_topic"] == "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1"
    assert first["completion_topic"] == "chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1"
    assert first["parked_at"] == _OCCURRED
    assert first["deadline_at"] == _OCCURRED + _dt.timedelta(days=7)
    assert second["deadline_at"] == _OCCURRED + _dt.timedelta(seconds=3600)
    assert "ON CONFLICT (idempotency_key) DO NOTHING" in inserts[0][0]


@pytest.mark.asyncio
async def test_backfill_skips_a_role_with_no_registered_crew_loudly_and_continues() -> None:
    conn = _FakeConn(
        [
            _outbox_row("agent_dispatch.ghost_role.s1:1", role="ghost_role"),
            _outbox_row("agent_dispatch.oe_evaluate.s2:q1:1"),
        ]
    )
    store = ParkLedgerStore(conn=conn)
    inserted = await store.backfill_from_outbox(
        crew_for_role={"oe_evaluate": "oe_grading"},
        deadline_policy=ParkDeadlinePolicy(),
    )
    assert inserted == 1
    inserts = [p for s, p in conn.executed if s.startswith("INSERT INTO ai_kernel_agent_dispatch_parks")]
    assert [p["agent_role"] for p in inserts] == ["oe_evaluate"]


@pytest.mark.asyncio
async def test_backfill_skips_a_request_whose_body_carries_no_thread() -> None:
    conn = _FakeConn([_outbox_row("agent_dispatch.oe_evaluate.s3:q1:1", body_extra={"thread_id": ""})])
    store = ParkLedgerStore(conn=conn)
    inserted = await store.backfill_from_outbox(
        crew_for_role={"oe_evaluate": "oe_grading"},
        deadline_policy=ParkDeadlinePolicy(),
    )
    assert inserted == 0
