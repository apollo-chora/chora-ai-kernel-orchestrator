"""ReconnectingAsyncConnection — self-healing psycopg AsyncConnection wrapper.

§4.1 resilience fix ([[project_oe_fullmatrix_e2e_2026_06_03]] infra finding 1).

The OE-grading composition root opens a SINGLE long-lived
``psycopg.AsyncConnection`` shared by the InboxIdempotencyStore + the OE outbox
writer + the outbox store (oe_grading_crew_wiring.build_oe_grading_crew_from_env).
After a cost-pause/resume — or ANY transient DB blip — that connection is
``closed``; the next ``conn.cursor()`` raised
``psycopg.OperationalError: the connection is closed``. Because that error
surfaced inside the Pub/Sub message handler, the grading message failed its
delivery, retried 5×, and dead-lettered to
``chora.dlq.delivery.grading.submission_requested.v1`` — a topic with NO
consumer, so the learner's submission was silently lost until a pod restart
gave a fresh connection.

This wrapper makes the shared connection self-healing: it (re)connects lazily on
first use and whenever the underlying connection reports ``closed``, re-applying
``autocommit`` on every reconnect. Wrapping the shared ``db_conn`` once fixes the
inbox, the outbox writer, and the outbox store together, because they all share
the same object.

Design notes
------------
* ``cursor()`` stays a *synchronous* call returning an async context manager (so
  callers keep ``async with conn.cursor() as cur:``). The reconnect check has to
  run inside ``__aenter__`` because reconnecting is awaitable.
* The reconnect trigger is ``underlying.closed`` OR ``underlying.broken``.
  psycopg3 sets ``.closed`` when the connection is cleanly closed and marks
  ``.broken`` when an operation left it in an unusable state (a lost socket /
  server restart) — checking both means a *silently* dead connection (``.closed``
  still 0 but ``.broken`` true after the failing op raised) is reconnected on the
  Pub/Sub redelivery rather than handing back the dead connection again.
  Combined with the inbox idempotency key (``ON CONFLICT DO NOTHING`` downstream)
  this converges without message loss.
* ``_ensure`` is guarded by an ``asyncio.Lock`` with a double-check so that when a
  DB blip flushes a backlog of Pub/Sub messages at once (each scheduled onto the
  loop via ``run_coroutine_threadsafe``), only ONE reconnect happens and every
  caller binds to the same fresh connection — no orphaned/leaked connections.
* ``connect`` is injectable so the composition root can wrap its cold-start
  backoff (``_connect_with_retry``) and tests can run SDK-free.

Hexagonal: no infrastructure imports at module load — ``psycopg`` is imported
lazily only by the default connect path.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

# Test-on-borrow idle gate. A Cloud SQL / cloudsql-proxy idle-close leaves
# psycopg's ``.closed``/``.broken`` clean until the first op hits the dead
# socket (KG-walk finding 2026-07-05: the weakness-analyser inbox SELECT ate
# ``server closed the connection unexpectedly`` after the orchestrator sat idle
# ~7h, so the diagnosis only self-healed on the next Pub/Sub redelivery — after
# the FE polling window gave up). When a borrow follows more idle than this, the
# wrapper actively probes (``SELECT 1``) and reconnects on failure so the FIRST
# attempt succeeds. Kept above any realistic burst gap so hot paths (qgen/oe
# share this wrapper) skip the probe entirely.
_DEFAULT_LIVENESS_IDLE_SECONDS = 30.0


def _is_closed(conn: Any) -> bool:
    """True when there is no usable underlying connection.

    psycopg3 ``AsyncConnection.closed`` is an int (0 = open) and ``.broken`` is a
    bool flagged when an error left the connection unusable. Checking BOTH covers
    the silent-death case where a lost socket leaves ``.closed`` 0 but ``.broken``
    true after the failing op raised. None = never connected.
    """
    if conn is None:
        return True
    return bool(getattr(conn, "closed", False)) or bool(getattr(conn, "broken", False))


class _ReconnectingCursorCM:
    """Async context manager that ensures a live connection, then defers to the
    underlying ``conn.cursor()`` context manager."""

    def __init__(
        self,
        owner: ReconnectingAsyncConnection,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._owner = owner
        self._args = args
        self._kwargs = kwargs
        self._cur_cm: Any = None

    async def __aenter__(self) -> Any:
        conn = await self._owner._ensure()
        self._cur_cm = conn.cursor(*self._args, **self._kwargs)
        return await self._cur_cm.__aenter__()

    async def __aexit__(self, *exc: Any) -> Any:
        if self._cur_cm is not None:
            return await self._cur_cm.__aexit__(*exc)
        return None


class _ReconnectingTransactionCM:
    """Async context manager that ensures a live connection, then defers to the
    underlying ``conn.transaction()`` context manager.

    Used by ``PostgresPromptOverrideRepository`` (ADR-197 M-B.2) so its
    ``set_config('chora.tenant_id', $1, TRUE)`` GUC holds for the SELECT even on
    the OE wiring's autocommit=True connection (SET LOCAL needs a tx block)."""

    def __init__(
        self,
        owner: ReconnectingAsyncConnection,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._owner = owner
        self._args = args
        self._kwargs = kwargs
        self._tx_cm: Any = None

    async def __aenter__(self) -> Any:
        conn = await self._owner._ensure()
        self._tx_cm = conn.transaction(*self._args, **self._kwargs)
        return await self._tx_cm.__aenter__()

    async def __aexit__(self, *exc: Any) -> Any:
        if self._tx_cm is not None:
            return await self._tx_cm.__aexit__(*exc)
        return None


class ReconnectingAsyncConnection:
    """A psycopg ``AsyncConnection`` proxy that reconnects on ``closed``.

    Exposes the slice of the connection API the OE-grading adapters use:
    ``cursor()``, ``transaction()``, ``commit()``, ``rollback()``,
    ``set_autocommit()``, ``close()`` and the ``closed`` property.
    """

    def __init__(
        self,
        dsn: str,
        *,
        autocommit: bool = False,
        connect: Callable[[str], Awaitable[Any]] | None = None,
        session_settings: Mapping[str, str] | None = None,
        liveness_idle_seconds: float = _DEFAULT_LIVENESS_IDLE_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._dsn = dsn
        self._autocommit = autocommit
        # Session GUCs re-applied on EVERY (re)connect, next to autocommit.
        # A reconnect is transparent to callers, so anything that lives in
        # session state and is NOT re-applied here is silently lost — see the
        # class docstring's sweeper-GUC note. Empty/None = apply nothing, so a
        # connection that does not opt in behaves exactly as before.
        self._session_settings: dict[str, str] = dict(session_settings or {})
        self._connect = connect or self._default_connect
        self._conn: Any = None
        # Serialises concurrent reconnects so a backlog of Pub/Sub messages
        # flushing after a DB blip triggers exactly ONE reconnect.
        self._lock = asyncio.Lock()
        # Test-on-borrow idle gate (see _DEFAULT_LIVENESS_IDLE_SECONDS).
        self._liveness_idle_seconds = liveness_idle_seconds
        self._monotonic = monotonic
        self._last_used_at: float | None = None

    @staticmethod
    async def _default_connect(dsn: str) -> Any:
        import psycopg  # lazy — keeps the test path SDK-free

        return await psycopg.AsyncConnection.connect(dsn)

    async def _ensure(self) -> Any:
        """Return a live underlying connection, reconnecting if needed.

        Double-checked under a lock: the fast path (connection flagged live AND
        used recently) skips the lock; only a needed reconnect/probe serialises.
        The second check inside the lock means a caller that waited on a peer's
        reconnect reuses the fresh connection instead of opening another.

        Flags alone (``closed``/``broken``) miss a server-side idle-close, which
        stays invisible until the first op raises. So when a borrow follows more
        idle than ``liveness_idle_seconds``, actively probe (``SELECT 1``) and
        reconnect on failure — the FIRST attempt then succeeds instead of eating
        the error and deferring to a Pub/Sub redelivery.
        """
        # Fast path: flagged-live AND used within the idle window — no lock.
        if not _is_closed(self._conn) and not self._probe_due():
            return self._conn
        async with self._lock:
            conn = self._conn
            need_reconnect = _is_closed(conn)
            if not need_reconnect and self._probe_due():
                if await self._is_live(conn):
                    self._mark_used()
                    return conn
                logger.warning(
                    "reconnecting_connection.stale_on_borrow",
                    extra={"reason": "liveness probe failed after idle"},
                )
                need_reconnect = True
            if need_reconnect:
                if conn is not None:
                    logger.warning(
                        "reconnecting_connection.reconnect",
                        extra={"reason": "underlying connection closed/broken"},
                    )
                self._conn = await self._connect(self._dsn)
                if self._autocommit:
                    await self._conn.set_autocommit(True)
                await self._apply_session_settings(self._conn)
            self._mark_used()
            return self._conn

    async def _apply_session_settings(self, conn: Any) -> None:
        """Re-apply the session GUCs on a freshly (re)connected connection.

        Runs in the same place ``autocommit`` is re-applied, and for the same
        reason: a reconnect is invisible to callers, so session state that is
        not restored here silently disappears mid-lane. For the kernel sweeper
        GUC that would mean a fail-CLOSED policy quietly returning 0 rows — a
        reaper that reaps nothing, which TESTS GREEN. Parameterised, never
        interpolated: a GUC name is not a trusted literal.

        ``is_local=False`` (session scope, not transaction scope) is deliberate:
        the sweeper connections are autocommit, where a tx-local setting would
        expire at the end of every implicit single-statement transaction.
        """
        if not self._session_settings:
            return
        async with conn.cursor() as cur:
            for name, value in self._session_settings.items():
                await cur.execute("SELECT set_config(%s, %s, %s)", (name, value, False))

    def _probe_due(self) -> bool:
        """True when the connection has been idle long enough that a silent
        server-side close could have killed it without flagging ``closed``/
        ``broken``. Never probes a never-connected wrapper (about to connect
        fresh) — that keeps lazy-connect + the burst fast path allocation-free."""
        if self._last_used_at is None:
            return False
        return (self._monotonic() - self._last_used_at) >= self._liveness_idle_seconds

    def _mark_used(self) -> None:
        self._last_used_at = self._monotonic()

    async def _is_live(self, conn: Any) -> bool:
        """Cheap liveness probe. Any failure (incl. the silent server-close that
        only surfaces on first use) counts as dead → caller reconnects."""
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001
            # fail-loud-exempt: same as the checkpointer probe. The failure IS the
            # return value ("reconnect"), the caller acts on it at once, and this
            # runs on every borrow of the connection.
            return False

    async def connect(self) -> ReconnectingAsyncConnection:
        """Eagerly establish the connection (fail loud at startup). Returns self
        so the composition root can ``db_conn = await Reconnecting(...).connect()``
        in one line if it wants the old eager-connect semantics."""
        await self._ensure()
        return self

    def cursor(self, *args: Any, **kwargs: Any) -> _ReconnectingCursorCM:
        return _ReconnectingCursorCM(self, args, kwargs)

    def transaction(self, *args: Any, **kwargs: Any) -> _ReconnectingTransactionCM:
        return _ReconnectingTransactionCM(self, args, kwargs)

    async def commit(self) -> None:
        conn = await self._ensure()
        await conn.commit()

    async def rollback(self) -> None:
        # Only a live connection can have an open transaction to roll back; if
        # the connection is gone there is nothing to undo (and reconnecting just
        # to roll back an empty tx would be pointless).
        if not _is_closed(self._conn):
            await self._conn.rollback()

    async def set_autocommit(self, value: bool) -> None:
        self._autocommit = bool(value)
        conn = await self._ensure()
        await conn.set_autocommit(value)

    @property
    def closed(self) -> bool:
        return _is_closed(self._conn)

    async def close(self) -> None:
        if not _is_closed(self._conn):
            await self._conn.close()


__all__ = ["ReconnectingAsyncConnection"]
