"""RED->GREEN tests: the qgen shared connection must self-heal.

Live incident 2026-07-23 (cost-pause resume, prod ``chora-489812``): Cloud SQL
restarted, sent ``AdminShutdown`` to the orchestrator, and from that moment the
outbox drain loop logged

    outbox_dispatcher.drain_once_failed
    psycopg.OperationalError: the connection is closed

every 2 seconds for the pod's whole lifetime -- 0 successful drains in 12
minutes -- while ``/readyz`` kept returning 200 and the pod sat 3/3 Ready. Only
``kubectl rollout restart`` recovered it.

Root cause: ``build_qgen_crew_from_env`` bound the RAW
``psycopg.AsyncConnection`` returned by ``_connect_with_retry`` straight to
``db_conn``. Its two sibling composition roots already wrap the identical
shared-connection pattern in ``ReconnectingAsyncConnection``
(``oe_grading_crew_wiring``, ``weakness_analyser_crew_wiring``); qgen was the one
left raw.

The blast radius is service-wide, not qgen-local: main.py starts exactly ONE
outbox drain task per process and -- because the qgen crew is enabled -- that
dispatcher runs on THIS connection (see the comment in oe_grading_crew_wiring:
the OE dispatcher is built but deliberately never started). So after any DB
restart every orchestrator-produced event (weakness.analyzed,
weakness.review_pending, agent_decision.logged, ai_assist.completed,
grading.submission_completed) is written durably by the healthy reconnecting
writers and then NEVER published, until a human restarts the pod.

Per [[feedback-resilience-priority]] / [[agentic-resilience-d6]] pillar 2.
"""

from __future__ import annotations

import asyncio
import inspect
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

    async def fetchall(self) -> list[Any]:
        return []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


@dataclass
class _FakeUnderlyingConn:
    """Mimics the slice of psycopg.AsyncConnection the qgen adapters use."""

    closed: int = 0
    broken: int = 0
    autocommit_calls: list[bool] = field(default_factory=list)
    rollbacks: int = 0
    cur: _FakeCursor = field(default_factory=_FakeCursor)

    def cursor(self, *args: Any, **kwargs: Any) -> _FakeCursor:
        if self.closed or self.broken:
            raise _OperationalError("the connection is closed")
        return self.cur

    async def set_autocommit(self, value: bool) -> None:
        self.autocommit_calls.append(value)

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def close(self) -> None:
        self.closed = 1


class _FakePsycopg:
    """Stand-in for the ``psycopg`` module.

    ``fail_first`` reproduces the cold-start race the qgen wiring's
    ``_connect_with_retry`` exists to absorb, so a reconnect inherits the same
    backoff instead of giving up on the first refused connection.
    """

    def __init__(self, fail_first: int = 0) -> None:
        self.conns: list[_FakeUnderlyingConn] = []
        self.attempts = 0
        self._fail_first = fail_first
        self.AsyncConnection = self._AsyncConnectionNS(self)

    class _AsyncConnectionNS:
        def __init__(self, outer: _FakePsycopg) -> None:
            self._outer = outer

        async def connect(self, dsn: str) -> _FakeUnderlyingConn:
            outer = self._outer
            outer.attempts += 1
            if outer.attempts <= outer._fail_first:
                raise _OperationalError("connection refused (proxy not up yet)")
            c = _FakeUnderlyingConn()
            outer.conns.append(c)
            return c


def _helper() -> Any:
    """The composition-root helper under test.

    Imported lazily inside each test so the RED run reports a clear
    AttributeError on the missing symbol rather than an import-time collection
    error that hides the other cases.
    """
    from chora_ai_kernel_orchestrator.adapter.pubsub import qgen_crew_wiring

    return qgen_crew_wiring.build_qgen_db_conn


class TestQGenConnectionSelfHeals:
    @pytest.mark.asyncio
    async def test_returns_a_reconnecting_wrapper_not_a_raw_connection(self) -> None:
        pg = _FakePsycopg()
        conn = _helper()(pg, "postgres://x/y")
        assert isinstance(conn, ReconnectingAsyncConnection), (
            "qgen db_conn must be a ReconnectingAsyncConnection -- a raw "
            "AsyncConnection strands the outbox after any DB restart"
        )

    @pytest.mark.asyncio
    async def test_reconnects_after_the_underlying_connection_dies(self) -> None:
        """The live failure, reproduced: AdminShutdown kills the connection,
        and the NEXT borrow must succeed on a fresh one."""
        pg = _FakePsycopg()
        conn = await _helper()(pg, "postgres://x/y").connect()

        async with conn.cursor() as cur:  # healthy drain
            await cur.execute("SELECT 1")
        assert pg.attempts == 1

        # Cloud SQL restart: psycopg flags the connection broken.
        pg.conns[0].broken = 1

        async with conn.cursor() as cur:  # the borrow that used to raise forever
            await cur.execute("SELECT 1")

        assert pg.attempts == 2, "expected exactly one reconnect"
        assert len(pg.conns) == 2
        assert pg.conns[1].broken == 0

    @pytest.mark.asyncio
    async def test_reconnect_inherits_the_cold_start_backoff(self) -> None:
        """A reconnect races the same cloudsql-proxy cold start as a cold boot,
        so the wrapper's connect callable must keep _connect_with_retry."""
        from chora_ai_kernel_orchestrator.adapter.pubsub import qgen_crew_wiring

        slept: list[float] = []

        async def _no_sleep(seconds: float) -> None:
            slept.append(seconds)

        orig = qgen_crew_wiring.asyncio.sleep
        qgen_crew_wiring.asyncio.sleep = _no_sleep  # type: ignore[assignment]
        try:
            pg = _FakePsycopg(fail_first=2)
            conn = await _helper()(pg, "postgres://x/y").connect()
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        finally:
            qgen_crew_wiring.asyncio.sleep = orig  # type: ignore[assignment]

        assert pg.attempts == 3, "connect must retry through the cold-start race"
        assert slept, "retry must back off between attempts"

    @pytest.mark.asyncio
    async def test_does_not_force_autocommit_on(self) -> None:
        """store.PostgresOutboxStore.fetch_pending releases its read
        transaction with an explicit rollback BECAUSE the qgen connection runs
        autocommit=OFF. Flipping it on here would silently change that
        contract, so the wrapper must leave psycopg's default alone."""
        pg = _FakePsycopg()
        conn = await _helper()(pg, "postgres://x/y").connect()
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
        assert pg.conns[0].autocommit_calls == [], (
            "qgen connection must stay autocommit=OFF -- store.py's rollback-on-empty release depends on it"
        )


class TestCompositionRootUsesTheHelper:
    def test_build_qgen_crew_from_env_does_not_bind_a_raw_connection(self) -> None:
        """Guards the regression at its source: a refactor that goes back to
        `db_conn = await _connect_with_retry(...)` re-strands the outbox."""
        from chora_ai_kernel_orchestrator.adapter.pubsub import qgen_crew_wiring

        src = inspect.getsource(qgen_crew_wiring.build_qgen_crew_from_env)
        assert "build_qgen_db_conn" in src, "composition root must build db_conn via build_qgen_db_conn"
        assert "db_conn = await _connect_with_retry(" not in src, (
            "raw AsyncConnection bound to db_conn -- outbox will strand on the next DB restart"
        )


def test_asyncio_is_imported_by_the_wiring_module() -> None:
    """Sanity anchor for the monkeypatch in the backoff test above."""
    from chora_ai_kernel_orchestrator.adapter.pubsub import qgen_crew_wiring

    assert qgen_crew_wiring.asyncio is asyncio
