"""RED→GREEN tests for ReconnectingAsyncConnection (§4.1 resilience).

Reproduces + fixes the OE-grading inbox/outbox stale-connection failure
([[project_oe_fullmatrix_e2e_2026_06_03]] infra finding 1): after a
cost-pause/resume (or any DB blip) the single long-lived ``psycopg``
``AsyncConnection`` shared by the InboxIdempotencyStore + outbox writer + outbox
store is ``closed``; the next ``conn.cursor()`` raised
``OperationalError: the connection is closed``, so the first grading message
failed 5× and dead-lettered to a topic with no consumer → silently lost.

ReconnectingAsyncConnection wraps the shared connection and transparently
(re)connects on first use and whenever the underlying connection is ``closed``,
so the same blip becomes a self-healing reconnect instead of message loss.
Per [[feedback-d6-resilience-first-class]] / [[feedback-resilience-priority]].
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.reconnecting_connection import (
    ReconnectingAsyncConnection,
)


class _OperationalError(Exception):
    """Stand-in for psycopg.OperationalError ('the connection is closed')."""


@dataclass
class _FakeCursor:
    executed: list[tuple[str, Any]] = field(default_factory=list)

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    async def fetchone(self) -> Any:
        return None

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


@dataclass
class _FakeTransaction:
    """Mimics psycopg.AsyncTransaction as an async context manager."""

    async def __aenter__(self) -> _FakeTransaction:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


@dataclass
class _FakeUnderlyingConn:
    """Mimics the slice of psycopg.AsyncConnection the adapters use."""

    closed: int = 0
    broken: int = 0
    autocommit_calls: list[bool] = field(default_factory=list)
    commits: int = 0
    rollbacks: int = 0
    tx_entered: int = 0
    cur: _FakeCursor = field(default_factory=_FakeCursor)

    def cursor(self, *args: Any, **kwargs: Any) -> _FakeCursor:
        if self.closed:
            raise _OperationalError("the connection is closed")
        return self.cur

    def transaction(self, *args: Any, **kwargs: Any) -> _FakeTransaction:
        if self.closed:
            raise _OperationalError("the connection is closed")
        self.tx_entered += 1
        return _FakeTransaction()

    async def set_autocommit(self, value: bool) -> None:
        self.autocommit_calls.append(value)

    async def commit(self) -> None:
        if self.closed:
            raise _OperationalError("the connection is closed")
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def close(self) -> None:
        self.closed = 1


def _connect_factory() -> tuple[list[_FakeUnderlyingConn], Any]:
    conns: list[_FakeUnderlyingConn] = []

    async def connect(_dsn: str) -> _FakeUnderlyingConn:
        c = _FakeUnderlyingConn()
        conns.append(c)
        return c

    return conns, connect


class TestLazyConnect:
    @pytest.mark.asyncio
    async def test_connects_lazily_on_first_cursor(self) -> None:
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        assert conns == []  # nothing connected at construction
        async with w.cursor() as cur:
            await cur.execute("SELECT 1")
        assert len(conns) == 1

    @pytest.mark.asyncio
    async def test_reuses_open_connection(self) -> None:
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        async with w.cursor():
            pass
        async with w.cursor():
            pass
        assert len(conns) == 1  # no needless reconnect while open


class TestTransaction:
    """ADR-197 M-B.2 — the PromptResolver's PostgresPromptOverrideRepository
    wraps its SELECT in ``conn.transaction()`` so the SET LOCAL tenant GUC holds
    on the OE wiring's autocommit conn. The wrapper must expose transaction()
    with the same lazy-connect + self-heal semantics as cursor()."""

    @pytest.mark.asyncio
    async def test_transaction_lazily_connects(self) -> None:
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        assert conns == []
        async with w.transaction():
            pass
        assert len(conns) == 1
        assert conns[0].tx_entered == 1

    @pytest.mark.asyncio
    async def test_transaction_and_cursor_compose_on_one_conn(self) -> None:
        # Mirrors the repository: `async with conn.transaction(), conn.cursor()`.
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        async with w.transaction(), w.cursor() as cur:
            await cur.execute("SELECT set_config('chora.tenant_id', %s, TRUE)", ("t",))
        assert len(conns) == 1
        assert conns[0].tx_entered == 1

    @pytest.mark.asyncio
    async def test_transaction_reconnects_when_closed(self) -> None:
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        async with w.transaction():
            pass
        conns[0].closed = 1  # blip
        async with w.transaction():
            pass
        assert len(conns) == 2  # self-healed
        assert conns[1].tx_entered == 1


class TestReconnect:
    @pytest.mark.asyncio
    async def test_reconnects_when_underlying_closed(self) -> None:
        """THE §4.1 fix: a closed underlying conn → next use reconnects, no raise."""
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        async with w.cursor():
            pass
        # Simulate the cost-pause/resume blip: the DB closed the connection.
        conns[0].closed = 1
        # Before the fix this raised OperationalError; now it self-heals.
        async with w.cursor() as cur:
            await cur.execute("SELECT 1")
        assert len(conns) == 2  # a fresh connection was established
        assert conns[1].closed == 0

    @pytest.mark.asyncio
    async def test_reconnects_when_underlying_broken(self) -> None:
        """Silent death: psycopg leaves .closed==0 but flags .broken after a
        lost socket. The wrapper must treat .broken as needing a reconnect."""
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        async with w.cursor():
            pass
        conns[0].closed = 0
        conns[0].broken = 1  # silently dead
        assert w.closed is True  # detected via .broken
        async with w.cursor():
            pass
        assert len(conns) == 2

    @pytest.mark.asyncio
    async def test_concurrent_reconnect_opens_exactly_one_connection(self) -> None:
        """A backlog of messages flushing after a DB blip must trigger ONE
        reconnect, not one-per-message (the asyncio.Lock + double-check)."""
        conns: list[_FakeUnderlyingConn] = []

        async def slow_connect(_dsn: str) -> _FakeUnderlyingConn:
            # Yield control so a second concurrent _ensure can interleave at the
            # await point — this is the exact TOCTOU window the lock closes.
            await asyncio.sleep(0)
            c = _FakeUnderlyingConn()
            conns.append(c)
            return c

        w = ReconnectingAsyncConnection("dsn", autocommit=True, connect=slow_connect)
        await w.connect()  # conns[0]
        conns[0].closed = 1  # blip closes the live conn

        async def use() -> None:
            async with w.cursor() as cur:
                await cur.execute("SELECT 1")

        # Fire several concurrent users against the closed connection.
        await asyncio.gather(*(use() for _ in range(5)))
        # Exactly one reconnect (conns[0] + conns[1]) — not five.
        assert len(conns) == 2, f"expected 1 reconnect, got {len(conns) - 1}"

    @pytest.mark.asyncio
    async def test_commit_reconnects_when_closed(self) -> None:
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        async with w.cursor():
            pass
        conns[0].closed = 1
        await w.commit()
        assert len(conns) == 2
        assert conns[1].commits == 1


class TestAutocommit:
    @pytest.mark.asyncio
    async def test_applies_autocommit_on_connect_and_reconnect(self) -> None:
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", autocommit=True, connect=connect)
        async with w.cursor():
            pass
        assert conns[0].autocommit_calls == [True]
        conns[0].closed = 1
        async with w.cursor():
            pass
        # The reconnect must re-apply autocommit (else the OE INSERT never flushes).
        assert conns[1].autocommit_calls == [True]

    @pytest.mark.asyncio
    async def test_no_autocommit_when_disabled(self) -> None:
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        async with w.cursor():
            pass
        assert conns[0].autocommit_calls == []


class TestProxiesAndState:
    @pytest.mark.asyncio
    async def test_closed_property(self) -> None:
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        assert w.closed is True  # not yet connected
        async with w.cursor():
            pass
        assert w.closed is False
        conns[0].closed = 1
        assert w.closed is True

    @pytest.mark.asyncio
    async def test_close_closes_underlying(self) -> None:
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        async with w.cursor():
            pass
        await w.close()
        assert conns[0].closed == 1

    @pytest.mark.asyncio
    async def test_rollback_safe_when_never_connected(self) -> None:
        _conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        # No live conn — rollback must be a no-op, not a crash.
        await w.rollback()

    @pytest.mark.asyncio
    async def test_eager_connect(self) -> None:
        conns, connect = _connect_factory()
        w = ReconnectingAsyncConnection("dsn", autocommit=True, connect=connect)
        await w.connect()
        assert len(conns) == 1
        assert conns[0].autocommit_calls == [True]


# -----------------------------------------------------------------------------
# Integration: InboxIdempotencyStore over the wrapper survives a stale conn.
# This reproduces the EXACT §4.1 failure end-to-end — the OE-grading inbox is
# the first thing the grading-message handler touches; a closed conn there is
# what dead-lettered the submission.
# -----------------------------------------------------------------------------


@dataclass
class _DBTableCursor:
    """Cursor backed by a SHARED rows dict (models the persistent DB table that
    survives connection churn — a fresh connection still sees prior rows)."""

    rows: dict[str, Any]
    last_result: Any = None

    async def execute(self, sql: str, params: Any = None) -> None:
        import datetime as _dt

        s = sql.lower()
        if "select" in s and "idempotency_keys" in s:
            key = params["key"]
            ttl_at = self.rows.get(key)
            now = _dt.datetime.now(tz=_dt.UTC)
            self.last_result = (key,) if ttl_at is not None and ttl_at > now else None
        elif "insert" in s and "idempotency_keys" in s:
            self.rows[params["key"]] = params["ttl_at"]

    async def fetchone(self) -> Any:
        return self.last_result

    async def __aenter__(self) -> _DBTableCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


@dataclass
class _DBTableConn:
    rows: dict[str, Any]
    closed: int = 0
    autocommit_calls: list[bool] = field(default_factory=list)

    def cursor(self, *a: Any, **k: Any) -> _DBTableCursor:
        if self.closed:
            raise _OperationalError("the connection is closed")
        return _DBTableCursor(rows=self.rows)

    async def set_autocommit(self, value: bool) -> None:
        self.autocommit_calls.append(value)

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = 1


class TestInboxOverReconnectingConn:
    @pytest.mark.asyncio
    async def test_grading_message_processes_after_stale_conn(self) -> None:
        import datetime as _dt

        from chora_ai_kernel_orchestrator.adapter.pubsub.inbox_idempotency import (
            InboxIdempotencyStore,
        )

        shared_rows: dict[str, Any] = {}
        conns: list[_DBTableConn] = []

        async def connect(_dsn: str) -> _DBTableConn:
            c = _DBTableConn(rows=shared_rows)
            conns.append(c)
            return c

        wrapper = ReconnectingAsyncConnection("dsn", autocommit=True, connect=connect)
        store = InboxIdempotencyStore(conn=wrapper)

        ran1 = await store.process(
            key="submission-A",
            ttl=_dt.timedelta(days=7),
            fn=_noop,
        )
        assert ran1 is True
        assert "submission-A" in shared_rows

        # Cost-pause/resume blip closes the live connection.
        conns[0].closed = 1

        # The NEXT grading message must NOT dead-letter — the wrapper reconnects.
        ran2 = await store.process(
            key="submission-B",
            ttl=_dt.timedelta(days=7),
            fn=_noop,
        )
        assert ran2 is True
        assert len(conns) == 2  # reconnected
        assert "submission-B" in shared_rows
        # Idempotency still holds across the reconnect (shared DB table).
        again = await store.process(
            key="submission-A",
            ttl=_dt.timedelta(days=7),
            fn=_noop,
        )
        assert again is False  # already seen — dedupe survives reconnect


async def _noop() -> None:
    return None


# -----------------------------------------------------------------------------
# Stale server-side close: the case the flag-based _is_closed MISSES.
#
# A Cloud SQL / proxy idle-close leaves psycopg's ``.closed``==0 and ``.broken``
# ==0 client-side until the first real op hits the dead socket and raises
# ``OperationalError: server closed the connection unexpectedly``. The flag-only
# check hands that dead conn back, the caller's query fails → NACK, and the path
# only self-heals on the NEXT Pub/Sub redelivery — but the weakness-diagnose FE
# polls a bounded window and shows "Couldn't finish" before then. An idle-gated
# liveness probe on borrow (test-on-borrow) catches it inline so the FIRST
# attempt succeeds. This reproduces the 2026-07-05 KG-walk finding.
# -----------------------------------------------------------------------------


class _FakeClock:
    """Deterministic monotonic clock for the idle-gate."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


@dataclass
class _StaleCursor:
    conn: _StaleConn

    async def execute(self, sql: str, params: Any = None) -> None:
        if self.conn.server_dead:
            raise _OperationalError("consuming input failed: server closed the connection unexpectedly")
        self.conn.executed.append((sql, params))

    async def fetchone(self) -> Any:
        return None

    async def __aenter__(self) -> _StaleCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


@dataclass
class _StaleConn:
    """Flags stay clean (closed==0/broken==0) but a server-idle-close makes any
    real op raise — the silent-death the flag check can't see until first use."""

    closed: int = 0
    broken: int = 0
    server_dead: bool = False
    executed: list[tuple[str, Any]] = field(default_factory=list)
    autocommit_calls: list[bool] = field(default_factory=list)

    def cursor(self, *a: Any, **k: Any) -> _StaleCursor:
        if self.closed:
            raise _OperationalError("the connection is closed")
        return _StaleCursor(self)

    async def set_autocommit(self, value: bool) -> None:
        self.autocommit_calls.append(value)

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = 1


def _stale_connect_factory() -> tuple[list[_StaleConn], Any]:
    conns: list[_StaleConn] = []

    async def connect(_dsn: str) -> _StaleConn:
        c = _StaleConn()
        conns.append(c)
        return c

    return conns, connect


class TestStaleServerCloseProbe:
    @pytest.mark.asyncio
    async def test_probes_and_reconnects_after_idle_when_server_closed(self) -> None:
        """After idle beyond the threshold, borrow must probe, detect the silent
        death, reconnect, and the caller's execute must SUCCEED on the fresh conn
        (inline — not deferred to a Pub/Sub redelivery)."""
        clock = _FakeClock(0.0)
        conns, connect = _stale_connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect, liveness_idle_seconds=10.0, monotonic=clock.now)
        async with w.cursor() as cur:
            await cur.execute("real-work-1")
        assert len(conns) == 1

        # Server silently closes the idle conn: flags stay clean, ops now raise.
        conns[0].server_dead = True
        clock.advance(30.0)  # past the 10s idle threshold

        # Before the fix this raised OperationalError; the probe now self-heals.
        async with w.cursor() as cur:
            await cur.execute("real-work-2")
        assert len(conns) == 2, "expected an inline reconnect on the stale borrow"
        assert conns[1].server_dead is False
        assert ("real-work-2", None) in conns[1].executed

    @pytest.mark.asyncio
    async def test_skips_probe_during_burst(self) -> None:
        """Back-to-back borrows within the idle window must NOT probe — hot paths
        (qgen/oe share this wrapper) aren't taxed with a SELECT 1 per message."""
        clock = _FakeClock(0.0)
        conns, connect = _stale_connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect, liveness_idle_seconds=10.0, monotonic=clock.now)
        for _ in range(5):
            clock.advance(1.0)  # each borrow within the 10s window
            async with w.cursor() as cur:
                await cur.execute("real-work")
        assert len(conns) == 1
        # Only the 5 real executes — no extra SELECT 1 liveness probes.
        assert conns[0].executed == [("real-work", None)] * 5

    @pytest.mark.asyncio
    async def test_healthy_conn_after_idle_probes_but_keeps_same_conn(self) -> None:
        """When the idle conn is actually still alive, the probe passes and the
        SAME connection is reused (probe cost, but no needless reconnect)."""
        clock = _FakeClock(0.0)
        conns, connect = _stale_connect_factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect, liveness_idle_seconds=10.0, monotonic=clock.now)
        async with w.cursor() as cur:
            await cur.execute("real-work-1")
        clock.advance(30.0)  # idle, but conn stays healthy
        async with w.cursor() as cur:
            await cur.execute("real-work-2")
        assert len(conns) == 1  # probed, alive → reused, no reconnect
        # A SELECT 1 probe fired once, then the real execute.
        assert ("SELECT 1", None) in conns[0].executed
        assert ("real-work-2", None) in conns[0].executed
