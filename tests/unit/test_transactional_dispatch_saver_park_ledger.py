"""RED: ADR-254 D5: the park ledger row commits WITH the park and the outbox row.

ADR-253 D3a made the LangGraph park and the dispatch outbox row one transaction.
The reaper needs a third fact in that same transaction: "this thread is parked
on role R since T with deadline D". If the ledger row committed separately, a
crash between the two would leave a park the reaper can never see (silent
forever, the exact class this refactor exists to remove) or a ledger row for a
park that never happened (a phantom FAILED).

The fake connection journals BEGIN / execute / COMMIT / ROLLBACK in order.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.domain.agent_dispatch.deadline_policy import (
    ParkDeadlinePolicy,
)


class _FakeCursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        if "ai_kernel_agent_dispatch_parks" in sql:
            self._conn.journal.append("execute:park_insert")
            self._conn.park_params.append(params)
            if self._conn.fail_on == "park":
                raise RuntimeError("park insert exploded")
        else:
            self._conn.journal.append("execute:outbox_insert")
            if self._conn.fail_on == "outbox":
                raise RuntimeError("outbox insert exploded")


class _FakeTx:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal

    async def __aenter__(self) -> _FakeTx:
        self._journal.append("BEGIN")
        return self

    async def __aexit__(self, exc_type: Any, *_: Any) -> bool:
        self._journal.append("ROLLBACK" if exc_type else "COMMIT")
        return False


class _FakeConn:
    def __init__(self, fail_on: str | None = None) -> None:
        self.journal: list[str] = []
        self.park_params: list[dict[str, Any]] = []
        self.fail_on = fail_on

    def transaction(self) -> _FakeTx:
        return _FakeTx(self.journal)

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)


class _FakeInner:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal

    async def aput_writes(self, config, writes, task_id, task_path="") -> None:
        self._journal.append("execute:checkpoint_writes")

    def get_next_version(self, current, channel=None):
        return 1


class _Interrupt:
    def __init__(self, value: Any) -> None:
        self.value = value


_ROLE = "oe_evaluate"


def _dispatch() -> dict[str, Any]:
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
        build_dispatch_request,
    )

    return build_dispatch_request(
        agent_role=_ROLE,
        execution_id="01a02062-e5b4-7870-8fca-53ce363cd542:tsq-1:1",
        tenant_id="11111111-1111-7111-8111-111111111111",
        gcid="00000000-0000-7000-8000-000000001999",
        thread_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        input_payload='{"mode":"evaluate"}',
        traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        source_project="chora-489812",
    )


def _build(conn: _FakeConn, *, policy: ParkDeadlinePolicy | None, with_ledger: bool = True):
    from chora_ai_kernel_orchestrator.adapter.checkpointer.transactional_dispatch_saver import (  # noqa: E501
        TransactionalDispatchSaver,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_outbox_writer import (  # noqa: E501
        AgentDispatchOutboxWriter,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.park_ledger import ParkLedgerWriter

    return TransactionalDispatchSaver(
        inner=_FakeInner(conn.journal),
        conn=conn,
        outbox_writer=AgentDispatchOutboxWriter(conn=conn, source_project="chora-489812"),
        park_ledger_writer=(ParkLedgerWriter(conn=conn, crew="oe_grading") if with_ledger else None),
        deadline_policy=policy,
    )


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": "01a02062-e5b4-7870-8fca-53ce363cd542"}}


@pytest.mark.asyncio
async def test_park_writes_checkpoint_outbox_and_ledger_in_one_transaction() -> None:
    conn = _FakeConn()
    saver = _build(conn, policy=ParkDeadlinePolicy())
    writes = [("__interrupt__", (_Interrupt({"__chora_agent_dispatch__": _dispatch()}),))]

    await saver.aput_writes(_config(), writes, task_id="t1")

    assert conn.journal == [
        "BEGIN",
        "execute:checkpoint_writes",
        "execute:outbox_insert",
        "execute:park_insert",
        "COMMIT",
    ]


@pytest.mark.asyncio
async def test_a_ledger_failure_rolls_back_the_checkpoint_and_the_outbox_row() -> None:
    conn = _FakeConn(fail_on="park")
    saver = _build(conn, policy=ParkDeadlinePolicy())
    writes = [("__interrupt__", (_Interrupt({"__chora_agent_dispatch__": _dispatch()}),))]

    with pytest.raises(RuntimeError, match="park insert exploded"):
        await saver.aput_writes(_config(), writes, task_id="t1")

    assert conn.journal[-1] == "ROLLBACK"
    assert "COMMIT" not in conn.journal


@pytest.mark.asyncio
async def test_a_non_park_write_touches_neither_outbox_nor_ledger() -> None:
    conn = _FakeConn()
    saver = _build(conn, policy=ParkDeadlinePolicy())
    await saver.aput_writes(_config(), [("some_channel", {"k": "v"})], task_id="t1")
    assert conn.journal == ["BEGIN", "execute:checkpoint_writes", "COMMIT"]


@pytest.mark.asyncio
async def test_the_deadline_comes_from_the_policy_for_that_role() -> None:
    conn = _FakeConn()
    saver = _build(conn, policy=ParkDeadlinePolicy(per_role_seconds={_ROLE: 120}))
    request = _dispatch()
    writes = [("__interrupt__", (_Interrupt({"__chora_agent_dispatch__": request}),))]

    await saver.aput_writes(_config(), writes, task_id="t1")

    parked_at = _dt.datetime.fromisoformat(request["envelope"]["occurred_at"])
    assert conn.park_params[0]["parked_at"] == parked_at
    assert conn.park_params[0]["deadline_at"] == parked_at + _dt.timedelta(seconds=120)


def test_ledger_and_policy_come_together_or_not_at_all() -> None:
    conn = _FakeConn()
    with pytest.raises(ValueError):
        _build(conn, policy=None, with_ledger=True)
    with pytest.raises(ValueError):
        _build(conn, policy=ParkDeadlinePolicy(), with_ledger=False)


@pytest.mark.asyncio
async def test_without_a_ledger_the_d3a_behaviour_is_unchanged() -> None:
    conn = _FakeConn()
    saver = _build(conn, policy=None, with_ledger=False)
    writes = [("__interrupt__", (_Interrupt({"__chora_agent_dispatch__": _dispatch()}),))]
    await saver.aput_writes(_config(), writes, task_id="t1")
    assert conn.journal == ["BEGIN", "execute:checkpoint_writes", "execute:outbox_insert", "COMMIT"]
