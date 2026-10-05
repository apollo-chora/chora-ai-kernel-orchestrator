"""RED — ADR-253 D3a: the checkpoint write and the dispatch outbox row commit
in ONE transaction, or neither does.

The seam was measured, not assumed (spike 2026-08-20): LangGraph records a
park as a ``put_writes`` on channel ``__interrupt__``, NOT as a ``put``. So the
transactional unit is ``aput_writes`` — that write IS the durable fact "this
thread is parked awaiting agent X", and the dispatch intent must commit with it.

Every test here drives the saver through a fake async psycopg connection that
records the ORDER of transaction/commit/execute so the atomicity claim is
asserted against observable behaviour rather than against the implementation.
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fakes — a psycopg-shaped async connection that records what happened.
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, journal: list[str], fail_on: str | None) -> None:
        self._journal = journal
        self._fail_on = fail_on

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> None:
        self._journal.append("execute:outbox_insert")
        if self._fail_on == "outbox":
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
    """Records BEGIN / COMMIT / ROLLBACK / execute in call order."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.journal: list[str] = []
        self._fail_on = fail_on

    def transaction(self) -> _FakeTx:
        return _FakeTx(self.journal)

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.journal, self._fail_on)


class _FakeInner:
    """Async-surface checkpointer stand-in (AsyncPostgresSaver shape)."""

    def __init__(self, journal: list[str], fail_on: str | None = None) -> None:
        self._journal = journal
        self._fail_on = fail_on
        self.sync_calls: list[str] = []

    async def aput_writes(self, config, writes, task_id, task_path="") -> None:
        self._journal.append("execute:checkpoint_writes")
        if self._fail_on == "checkpoint":
            raise RuntimeError("checkpoint write exploded")

    async def aput(self, config, checkpoint, metadata, new_versions):
        self._journal.append("execute:checkpoint_put")
        return config

    async def aget_tuple(self, config):
        self._journal.append("execute:select")
        return None

    # If anything ever calls the SYNC surface we want the test to see it —
    # that is the event-loop-stall defect ADR-253 1.4 says must be closed.
    def put_writes(self, *a, **k):
        self.sync_calls.append("put_writes")

    def put(self, *a, **k):
        self.sync_calls.append("put")


class _Interrupt:
    def __init__(self, value: Any) -> None:
        self.value = value


def _dispatch() -> dict[str, Any]:
    """The REAL request shape, from the real builder — a hand-rolled fixture
    here would let the writer and the builder drift apart unnoticed."""
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
        build_dispatch_request,
    )

    return build_dispatch_request(
        agent_role="oe_evaluate",
        execution_id="01a02062-e5b4-7870-8fca-53ce363cd542:tsq-1:1",
        tenant_id="11111111-1111-7111-8111-111111111111",
        gcid="gcid-1",
        thread_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        input_payload='{"mode":"evaluate"}',
        traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        source_project="chora-489812",
    )


def _build(conn: _FakeConn, inner: _FakeInner):
    from chora_ai_kernel_orchestrator.adapter.checkpointer.transactional_dispatch_saver import (  # noqa: E501
        TransactionalDispatchSaver,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_outbox_writer import (  # noqa: E501
        AgentDispatchOutboxWriter,
    )

    return TransactionalDispatchSaver(
        inner=inner,
        conn=conn,
        outbox_writer=AgentDispatchOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        ),
    )


_CFG = {
    "configurable": {"thread_id": "01a02062-e5b4-7870-8fca-53ce363cd542", "checkpoint_ns": "", "checkpoint_id": "c1"}
}


# ---------------------------------------------------------------------------
# The load-bearing property
# ---------------------------------------------------------------------------


async def test_dispatch_and_checkpoint_write_share_one_transaction() -> None:
    """D3a: BEGIN, both writes, COMMIT — one transaction, both inside it."""
    conn = _FakeConn()
    inner = _FakeInner(conn.journal)
    saver = _build(conn, inner)

    await saver.aput_writes(
        _CFG,
        [("__interrupt__", (_Interrupt({"__chora_agent_dispatch__": _dispatch()}),))],
        "task-1",
    )

    assert conn.journal == [
        "BEGIN",
        "execute:checkpoint_writes",
        "execute:outbox_insert",
        "COMMIT",
    ], conn.journal


async def test_a_failed_outbox_insert_rolls_the_checkpoint_write_back() -> None:
    """Neither-or-both. If the dispatch intent cannot be recorded, the park
    must not be recorded either — otherwise the run parks forever with nothing
    dispatched, which is the exact failure D3a exists to remove."""
    conn = _FakeConn(fail_on="outbox")
    inner = _FakeInner(conn.journal)
    saver = _build(conn, inner)

    with pytest.raises(RuntimeError, match="outbox insert exploded"):
        await saver.aput_writes(
            _CFG,
            [("__interrupt__", (_Interrupt({"__chora_agent_dispatch__": _dispatch()}),))],
            "t",
        )

    assert "COMMIT" not in conn.journal
    assert conn.journal[-1] == "ROLLBACK", conn.journal


async def test_a_failed_checkpoint_write_queues_no_dispatch() -> None:
    """The other direction: no park recorded means no agent is asked to work."""
    conn = _FakeConn()
    inner = _FakeInner(conn.journal, fail_on="checkpoint")
    saver = _build(conn, inner)

    with pytest.raises(RuntimeError, match="checkpoint write exploded"):
        await saver.aput_writes(
            _CFG,
            [("__interrupt__", (_Interrupt({"__chora_agent_dispatch__": _dispatch()}),))],
            "t",
        )

    assert "execute:outbox_insert" not in conn.journal
    assert conn.journal[-1] == "ROLLBACK", conn.journal


# ---------------------------------------------------------------------------
# Discrimination — a HITL interrupt is NOT a dispatch
# ---------------------------------------------------------------------------


async def test_a_plain_hitl_interrupt_queues_no_dispatch_row() -> None:
    """The growth-edge crew parks on a HUMAN interrupt (ADR-205 WS-2). It must
    never be mistaken for an agent dispatch — that would publish a phantom
    request nobody can answer."""
    conn = _FakeConn()
    inner = _FakeInner(conn.journal)
    saver = _build(conn, inner)

    await saver.aput_writes(
        _CFG,
        [("__interrupt__", (_Interrupt({"proposed_edges": [], "familiar": {}}),))],
        "t",
    )

    assert "execute:outbox_insert" not in conn.journal
    assert conn.journal == ["BEGIN", "execute:checkpoint_writes", "COMMIT"], conn.journal


async def test_an_ordinary_state_write_queues_no_dispatch_row() -> None:
    conn = _FakeConn()
    inner = _FakeInner(conn.journal)
    saver = _build(conn, inner)

    await saver.aput_writes(_CFG, [("answers", [1, 2]), ("i", 3)], "t")

    assert "execute:outbox_insert" not in conn.journal


# ---------------------------------------------------------------------------
# autocommit=False hygiene — every op must be released, reads with COMMIT
# ---------------------------------------------------------------------------


async def test_aput_is_committed() -> None:
    """A checkpoint that is never committed is a checkpoint that does not exist
    — the 'an ACK is not a COMMIT' class."""
    conn = _FakeConn()
    inner = _FakeInner(conn.journal)
    saver = _build(conn, inner)

    await saver.aput(_CFG, {"id": "c2", "channel_values": {}}, {}, {})

    assert conn.journal == ["BEGIN", "execute:checkpoint_put", "COMMIT"], conn.journal


async def test_a_read_releases_with_commit_not_rollback() -> None:
    """A shared-connection read that releases with ROLLBACK destroys writes made
    by anything else on that connection."""
    conn = _FakeConn()
    inner = _FakeInner(conn.journal)
    saver = _build(conn, inner)

    await saver.aget_tuple(_CFG)

    assert conn.journal == ["BEGIN", "execute:select", "COMMIT"], conn.journal


# ---------------------------------------------------------------------------
# The event-loop defect ADR-253 1.4 says must be CLOSED, not inherited
# ---------------------------------------------------------------------------


async def test_the_async_surface_never_calls_the_sync_saver() -> None:
    """LazyPostgresSaver.aput calls the SYNCHRONOUS PostgresSaver.put with no
    offload, stalling the loop on every checkpoint. This design raises
    checkpoint frequency, so the replacement must not inherit that shape."""
    conn = _FakeConn()
    inner = _FakeInner(conn.journal)
    saver = _build(conn, inner)

    await saver.aput(_CFG, {"id": "c3", "channel_values": {}}, {}, {})
    await saver.aput_writes(_CFG, [("answers", [])], "t")
    await saver.aget_tuple(_CFG)

    assert inner.sync_calls == [], f"sync saver called from the async path: {inner.sync_calls}"


async def test_it_is_a_valid_langgraph_checkpointer() -> None:
    """graph.compile(checkpointer=...) gates on isinstance(BaseCheckpointSaver)."""
    from langgraph.checkpoint.base import BaseCheckpointSaver

    saver = _build(_FakeConn(), _FakeInner([]))
    assert isinstance(saver, BaseCheckpointSaver)


async def test_a_missing_collaborator_is_refused_at_construction() -> None:
    """A saver missing its outbox writer would checkpoint happily and dispatch
    nothing — every run would park forever, with no error anywhere."""
    from chora_ai_kernel_orchestrator.adapter.checkpointer.transactional_dispatch_saver import (  # noqa: E501
        TransactionalDispatchSaver,
    )

    conn, inner = _FakeConn(), _FakeInner([])
    for kwargs in (
        {"inner": None, "conn": conn, "outbox_writer": object()},
        {"inner": inner, "conn": None, "outbox_writer": object()},
        {"inner": inner, "conn": conn, "outbox_writer": None},
    ):
        with pytest.raises(ValueError, match="required"):
            TransactionalDispatchSaver(**kwargs)


async def test_the_sync_surface_refuses_loudly() -> None:
    """Every BaseCheckpointSaver method raises NotImplementedError by default, so
    an un-overridden sync call would surface as that rather than as the real
    problem. This saver is only correct on the async path; silently doing
    nothing on a sync call would lose checkpoints AND dispatches."""
    saver = _build(_FakeConn(), _FakeInner([]))
    for name in ("put", "put_writes", "get_tuple", "list", "delete_thread"):
        with pytest.raises(NotImplementedError, match="async-only"):
            getattr(saver, name)()


async def test_alist_releases_the_connection_before_the_caller_iterates() -> None:
    """An async generator that yielded across the transaction boundary would
    hold this shared connection open for the whole of the caller's loop."""
    conn = _FakeConn()

    class _Listing(_FakeInner):
        async def alist(self, config, *, filter=None, before=None, limit=None):  # noqa: A002
            self._journal.append("execute:select")
            for row in ("a", "b"):
                yield row

    saver = _build(conn, _Listing(conn.journal))
    rows = [r async for r in saver.alist(_CFG)]

    assert rows == ["a", "b"]
    assert conn.journal == ["BEGIN", "execute:select", "COMMIT"]
    # COMMIT landed before the caller saw a row, not after the loop ended.
    assert conn.journal.index("COMMIT") == len(conn.journal) - 1


async def test_adelete_thread_is_committed() -> None:
    conn = _FakeConn()

    class _Deleting(_FakeInner):
        async def adelete_thread(self, thread_id: str) -> None:
            self._journal.append("execute:delete")

    await _build(conn, _Deleting(conn.journal)).adelete_thread("01a02062-e5b4-7870-8fca-53ce363cd542")
    assert conn.journal == ["BEGIN", "execute:delete", "COMMIT"]


async def test_it_adopts_the_inner_serde() -> None:
    """The checkpoint bytes must encode the same way as the library's, or a
    checkpoint written by this saver cannot be read back by anything else."""
    sentinel = object()

    class _WithSerde(_FakeInner):
        serde = sentinel

    saver = _build(_FakeConn(), _WithSerde([]))
    assert saver.serde is sentinel


# ---------------------------------------------------------------------------
# The class of defect that reached production on 2026-08-20.
#
# BaseCheckpointSaver implements EVERY method, and the ones it does not really
# implement raise a bare NotImplementedError. Overriding "the methods a saver
# obviously needs" is therefore not enough: the first un-overridden one is a
# runtime crash, and because the platform's application logs are dropped at the
# sink it surfaced as a silently NACKing subscriber, not as an error anyone
# could see.
#
# get_next_version is the specific one that bit. It is called on every put to
# compute channel versions, and the base implementation raises as soon as the
# current version is a STRING — which is exactly what AsyncPostgresSaver uses.
# The fakes in this file all returned no version at all, so nothing here
# exercised it, and the real-graph tests elsewhere ran on InMemorySaver, whose
# integer versions take the base class's working branch. Green on both sides,
# broken on the wire.
# ---------------------------------------------------------------------------


async def test_it_delegates_get_next_version_to_the_inner_saver() -> None:
    """AsyncPostgresSaver uses STRING versions. BaseCheckpointSaver's default
    raises NotImplementedError on a string, so a saver that does not delegate
    dies on the second superstep of every run."""

    class _StrVersions(_FakeInner):
        def get_next_version(self, current, channel=None):
            return "00000000000000000000000000000002.0.999"

    saver = _build(_FakeConn(), _StrVersions([]))

    assert saver.get_next_version(None, None) == "00000000000000000000000000000002.0.999"
    # The failing case: a string current. The base class raises here.
    assert (
        saver.get_next_version("00000000000000000000000000000001.0.111", None)
        == "00000000000000000000000000000002.0.999"
    )


async def test_no_base_class_method_is_left_raising_not_implemented() -> None:
    """Guards the whole class, not just the one that bit. Any BaseCheckpointSaver
    method whose default raises must be overridden here — either delegated, or
    refused with a message that names the cause."""
    import inspect

    from langgraph.checkpoint.base import BaseCheckpointSaver

    from chora_ai_kernel_orchestrator.adapter.checkpointer.transactional_dispatch_saver import (  # noqa: E501
        TransactionalDispatchSaver,
    )

    overridden = set(TransactionalDispatchSaver.__dict__)
    unhandled = []
    for name, fn in vars(BaseCheckpointSaver).items():
        if name.startswith("_") or not callable(fn) or name in overridden:
            continue
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        if "NotImplementedError" in src:
            unhandled.append(name)

    assert unhandled == [], (
        f"these BaseCheckpointSaver methods still fall through to a bare NotImplementedError: {sorted(unhandled)}"
    )


async def test_concurrent_operations_do_not_interleave_transactions() -> None:
    """The defect that broke the live lane on 2026-08-20.

    LangGraph runs graph tasks as concurrent asyncio tasks, and a psycopg
    AsyncConnection is NOT safe for concurrent use: two coroutines each opening
    `conn.transaction()` on the same connection close them out of order and
    psycopg raises OutOfOrderTransactionNesting. That aborted the exit-time
    checkpoint persistence, which meant LangGraph's _suppress_interrupt raised
    instead of returning True, which meant GraphInterrupt escaped ainvoke, which
    meant the runner NACKed every message and the run never parked.

    Every transaction must therefore be serialised. The journal here must read
    BEGIN..COMMIT BEGIN..COMMIT — never two BEGINs in a row.
    """
    import asyncio

    conn = _FakeConn()

    class _Slow(_FakeInner):
        async def aput_writes(self, config, writes, task_id, task_path=""):
            self._journal.append("execute:checkpoint_writes")
            await asyncio.sleep(0)  # force a scheduling point mid-transaction

    saver = _build(conn, _Slow(conn.journal))

    await asyncio.gather(*[saver.aput_writes(_CFG, [("answers", [i])], f"task-{i}") for i in range(4)])

    depth = 0
    for entry in conn.journal:
        if entry == "BEGIN":
            depth += 1
            assert depth == 1, f"transactions interleaved: {conn.journal}"
        elif entry in ("COMMIT", "ROLLBACK"):
            depth -= 1
    assert depth == 0
    assert conn.journal.count("BEGIN") == 4
