"""RED: ADR-254 D5: the park ledger (migration 0056).

Two adapters over ``ai_kernel_agent_dispatch_parks``:

* ``ParkLedgerWriter.queue_park``: INSERTs the park row on the CALLER's open
  transaction (the transactional dispatch saver's), never commits, and is a
  no-op on re-execution (``ON CONFLICT (idempotency_key) DO NOTHING``) because
  LangGraph re-runs a node from the top on resume.
* ``ParkLedgerStore``: the reaper's and the router's view: settle rows, scan
  for expired parks, join the outbox dead-letters, count in-flight per tenant.

The fakes record SQL + params so each query's shape is asserted, not assumed.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    build_dispatch_request,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.park_ledger import (
    PARK_COLUMNS,
    PARK_LEDGER_TABLE,
    ParkLedgerStore,
    ParkLedgerWriter,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.park import ParkState
from chora_ai_kernel_orchestrator.domain.agent_dispatch.reaper import ReaperArm


class _FakeCursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self.rowcount = conn.rowcount

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        self._conn.executed.append((" ".join(sql.split()), params))

    async def fetchone(self) -> Any:
        return self._conn.fetchone_row

    async def fetchall(self) -> list[Any]:
        return list(self._conn.fetchall_rows)


class _FakeConn:
    def __init__(
        self,
        *,
        fetchone_row: Any = None,
        fetchall_rows: list[Any] | None = None,
        rowcount: int = 0,
    ) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.commits = 0
        self.fetchone_row = fetchone_row
        self.fetchall_rows = fetchall_rows or []
        self.rowcount = rowcount

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    async def commit(self) -> None:
        self.commits += 1


_TENANT = "11111111-1111-7111-8111-111111111111"
_THREAD = "01a02062-e5b4-7870-8fca-53ce363cd542"


def _request() -> dict[str, Any]:
    return build_dispatch_request(
        agent_role="oe_evaluate",
        execution_id=f"{_THREAD}:tsq-1:1",
        tenant_id=_TENANT,
        gcid="00000000-0000-7000-8000-000000001999",
        thread_id=_THREAD,
        input_payload='{"mode":"evaluate"}',
        traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        source_project="chora-489812",
    )


def _parked_at(request: dict[str, Any]) -> _dt.datetime:
    return _dt.datetime.fromisoformat(request["envelope"]["occurred_at"])


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_table_name_and_columns_are_the_0056_shape() -> None:
    assert PARK_LEDGER_TABLE == "ai_kernel_agent_dispatch_parks"
    assert PARK_COLUMNS == (
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
    )


def test_writer_requires_a_crew() -> None:
    with pytest.raises(ValueError):
        ParkLedgerWriter(conn=_FakeConn(), crew="  ")


@pytest.mark.asyncio
async def test_queue_park_inserts_on_the_callers_transaction_and_never_commits() -> None:
    conn = _FakeConn()
    request = _request()
    deadline = _parked_at(request) + _dt.timedelta(seconds=120)
    writer = ParkLedgerWriter(conn=conn, crew="oe_grading")

    key = await writer.queue_park(request, deadline_at=deadline)

    assert key == request["idempotency_key"]
    assert conn.commits == 0, "the saver owns the transaction boundary"
    assert len(conn.executed) == 1
    sql, params = conn.executed[0]
    assert sql.startswith(f"INSERT INTO {PARK_LEDGER_TABLE}")
    assert "ON CONFLICT (idempotency_key) DO NOTHING" in sql
    assert params == {
        "idempotency_key": request["idempotency_key"],
        "workflow_id": request["workflow_id"],
        "thread_id": _THREAD,
        "crew": "oe_grading",
        "tenant_id": _TENANT,
        "gcid": "00000000-0000-7000-8000-000000001999",
        "agent_role": "oe_evaluate",
        "request_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
        "completion_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1",
        "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        "tracestate": "",
        "parked_at": _parked_at(request),
        "deadline_at": deadline,
    }


@pytest.mark.asyncio
async def test_queue_park_refuses_a_deadline_that_is_naive_or_not_after_parked_at() -> None:
    request = _request()
    writer = ParkLedgerWriter(conn=_FakeConn(), crew="oe_grading")
    with pytest.raises(ValueError):
        await writer.queue_park(request, deadline_at=_dt.datetime(2026, 8, 22, 12, 0, 0))
    with pytest.raises(ValueError):
        await writer.queue_park(request, deadline_at=_parked_at(request))


@pytest.mark.asyncio
async def test_queue_park_refuses_a_request_without_a_thread() -> None:
    request = _request()
    request["body"]["thread_id"] = ""
    writer = ParkLedgerWriter(conn=_FakeConn(), crew="oe_grading")
    with pytest.raises(ValueError):
        await writer.queue_park(request, deadline_at=_parked_at(request) + _dt.timedelta(seconds=60))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _row(**over: Any) -> tuple[Any, ...]:
    parked = _dt.datetime(2026, 8, 22, 12, 0, 0, tzinfo=_dt.UTC)
    values: dict[str, Any] = {
        "idempotency_key": "agent_dispatch.oe_evaluate.x:1",
        "workflow_id": _THREAD,
        "thread_id": _THREAD,
        "crew": "oe_grading",
        "tenant_id": _TENANT,
        "gcid": "00000000-0000-7000-8000-000000001999",
        "agent_role": "oe_evaluate",
        "request_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
        "completion_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1",
        "traceparent": "00-aa-bb-01",
        "tracestate": "",
        "parked_at": parked,
        "deadline_at": parked + _dt.timedelta(days=7),
        "state": "parked",
        "settled_at": None,
        "settled_by": "",
        "late_completion_at": None,
    }
    values.update(over)
    return tuple(values[c] for c in PARK_COLUMNS)


@pytest.mark.asyncio
async def test_get_maps_the_row_in_column_order() -> None:
    conn = _FakeConn(fetchone_row=_row())
    store = ParkLedgerStore(conn=conn)
    park = await store.get("agent_dispatch.oe_evaluate.x:1")
    assert park is not None
    assert park.idempotency_key == "agent_dispatch.oe_evaluate.x:1"
    assert park.thread_id == _THREAD
    assert park.crew == "oe_grading"
    assert park.agent_role == "oe_evaluate"
    assert park.state is ParkState.PARKED
    assert park.is_parked
    sql, params = conn.executed[0]
    assert f"FROM {PARK_LEDGER_TABLE}" in sql
    assert "WHERE idempotency_key = %(key)s" in sql
    assert params == {"key": "agent_dispatch.oe_evaluate.x:1"}
    for column in PARK_COLUMNS:
        assert column in sql


@pytest.mark.asyncio
async def test_get_returns_none_when_there_is_no_row() -> None:
    store = ParkLedgerStore(conn=_FakeConn(fetchone_row=None))
    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_mark_completed_flips_only_a_parked_row_and_commits() -> None:
    conn = _FakeConn(rowcount=1)
    store = ParkLedgerStore(conn=conn)
    assert await store.mark_completed("k") is True
    sql, params = conn.executed[0]
    assert sql.startswith(f"UPDATE {PARK_LEDGER_TABLE}")
    assert "state = 'completed'" in sql
    assert "settled_by = 'completion'" in sql
    assert "settled_at = now()" in sql
    assert "WHERE idempotency_key = %(key)s AND state = 'parked'" in sql
    assert params == {"key": "k"}
    assert conn.commits == 1


@pytest.mark.asyncio
async def test_mark_completed_reports_false_when_the_row_was_not_parked() -> None:
    store = ParkLedgerStore(conn=_FakeConn(rowcount=0))
    assert await store.mark_completed("k") is False


@pytest.mark.asyncio
async def test_mark_reaped_records_the_arm_and_guards_on_parked() -> None:
    conn = _FakeConn(rowcount=1)
    store = ParkLedgerStore(conn=conn)
    assert await store.mark_reaped("k", arm=ReaperArm.REQUEST_EXPIRED) is True
    sql, params = conn.executed[0]
    assert "state = 'reaped'" in sql
    assert "settled_by = %(settled_by)s" in sql
    assert "WHERE idempotency_key = %(key)s AND state = 'parked'" in sql
    assert params == {"key": "k", "settled_by": "reaper:request_expired"}
    assert conn.commits == 1


@pytest.mark.asyncio
async def test_record_late_completion_stamps_a_settled_row_only() -> None:
    conn = _FakeConn(rowcount=1)
    store = ParkLedgerStore(conn=conn)
    await store.record_late_completion("k")
    sql, params = conn.executed[0]
    assert "late_completion_at = now()" in sql
    assert "WHERE idempotency_key = %(key)s AND state <> 'parked'" in sql
    assert params == {"key": "k"}
    assert conn.commits == 1


@pytest.mark.asyncio
async def test_fetch_expired_scans_parked_rows_past_their_deadline_oldest_first() -> None:
    conn = _FakeConn(fetchall_rows=[_row(), _row(idempotency_key="agent_dispatch.oe_evaluate.y:1")])
    store = ParkLedgerStore(conn=conn)
    rows = await store.fetch_expired(limit=50)
    assert [r.idempotency_key for r in rows] == [
        "agent_dispatch.oe_evaluate.x:1",
        "agent_dispatch.oe_evaluate.y:1",
    ]
    sql, params = conn.executed[0]
    assert "WHERE state = 'parked' AND deadline_at < now()" in sql
    assert "ORDER BY deadline_at ASC" in sql
    assert "LIMIT %(limit)s" in sql
    assert params == {"limit": 50}
    # A read on a shared autocommit connection releases with COMMIT, never ROLLBACK.
    assert conn.commits == 1


@pytest.mark.asyncio
async def test_fetch_outbox_dead_lettered_joins_the_outbox_on_the_request_key() -> None:
    conn = _FakeConn(fetchall_rows=[_row()])
    store = ParkLedgerStore(conn=conn)
    rows = await store.fetch_outbox_dead_lettered(limit=20)
    assert len(rows) == 1
    sql, params = conn.executed[0]
    assert "JOIN ai_kernel_outbox_events o ON o.idempotency_key = p.idempotency_key" in sql
    assert "p.state = 'parked'" in sql
    assert "o.status = 'deadlettered'" in sql
    assert params == {"limit": 20}


@pytest.mark.asyncio
async def test_count_parked_is_per_tenant_and_role() -> None:
    conn = _FakeConn(fetchone_row=(3,))
    store = ParkLedgerStore(conn=conn)
    assert await store.count_parked(tenant_id=_TENANT, agent_role="companion_chat") == 3
    sql, params = conn.executed[0]
    assert "SELECT count(*)" in sql
    assert "WHERE tenant_id = %(tenant_id)s AND agent_role = %(agent_role)s AND state = 'parked'" in sql
    assert params == {"tenant_id": _TENANT, "agent_role": "companion_chat"}


@pytest.mark.asyncio
async def test_thread_is_parked_answers_from_the_ledger() -> None:
    assert await ParkLedgerStore(conn=_FakeConn(fetchone_row=(1,))).thread_is_parked(_THREAD) is True
    conn = _FakeConn(fetchone_row=None)
    assert await ParkLedgerStore(conn=conn).thread_is_parked(_THREAD) is False
    sql, params = conn.executed[0]
    assert "WHERE thread_id = %(thread_id)s AND state = 'parked'" in sql
    assert params == {"thread_id": _THREAD}


@pytest.mark.asyncio
async def test_store_refuses_blank_keys() -> None:
    store = ParkLedgerStore(conn=_FakeConn())
    with pytest.raises(ValueError):
        await store.get("")
    with pytest.raises(ValueError):
        await store.mark_completed(" ")
    with pytest.raises(ValueError):
        await store.count_parked(tenant_id="", agent_role="r")
