"""TransactionalDispatchSaver — ADR-253 D3a.

The property, stated plainly: **the LangGraph park and the intent to dispatch
the agent commit in one transaction, or neither commits.**

Why this class has to exist. ADR-253 1.4 measured the shape the refactor was
originally scoped on and found it false: ``LazyPostgresSaver`` defers to
``PostgresSaver.from_conn_string(dsn)`` and so opens its OWN connection, and
that connection is autocommit; the crew outbox runs on a separate connection.
Two connections in one database means nothing could enlist a checkpoint write
and an outbox insert together, and a graph node cannot enlist the checkpoint at
all because the runtime writes it on its own connection at the superstep
boundary. The owner ruled D3a on 2026-08-20: build the saver rather than pay a
split store's ordering-plus-idempotency price without having a split store.

Where the seam is, measured rather than assumed (spike 2026-08-20). A park is
recorded as a ``put_writes`` on channel ``__interrupt__``, NOT as a ``put``:

    put        checkpoint c1  (state after the previous superstep)
    put_writes __interrupt__  (Interrupt(value=...))   <- the park

That ``put_writes`` IS the durable fact "this thread is parked awaiting agent
X", so it is the write the dispatch row must commit with. Hooking ``aput``
instead would commit the dispatch against a checkpoint that does not yet know
the run is parked.

Three things this class deliberately does differently from the saver it sits
beside:

1. **Async all the way down.** ``LazyPostgresSaver.aput`` is an async method
   calling the SYNCHRONOUS ``PostgresSaver.put`` with no thread offload, so
   every checkpoint write briefly stalls the event loop. This design raises
   checkpoint frequency, so inheriting that shape would be worse than the
   library it replaces (ADR-253 1.4). The inner saver here is
   ``AsyncPostgresSaver`` and every method on the async surface awaits an async
   inner method — the defect is closed by construction, not patched with an
   offload.
2. **It owns the transaction explicitly.** ``AsyncPostgresSaver`` internally
   uses ``conn.pipeline()`` or ``conn.transaction()`` depending on what libpq
   supports, and ``conn.transaction()`` COMMITS on exit when it is outermost.
   Wrapping every call in our own ``conn.transaction()`` makes the inner one a
   savepoint and puts the commit boundary where this class can see it, on every
   psycopg build.
3. **Reads release with COMMIT, not ROLLBACK.** This connection is shared, and
   an empty-path rollback on a shared connection destroys whatever else was
   written on it.

⚠ Override discipline. ``BaseCheckpointSaver`` implements every method, and
the ones it does not really implement raise a BARE ``NotImplementedError``.
Overriding "the methods a saver obviously needs" is therefore not enough: the
first un-overridden one is a runtime crash. ``get_next_version`` is the trap —
it is called on every put and its default raises as soon as the version is a
string, which is what ``AsyncPostgresSaver`` uses. A test in
``test_transactional_dispatch_saver.py`` now enumerates the base class and
fails if any method is left falling through.

⚠ Not a substitute for receive-side idempotency. Pub/Sub is at-least-once no
matter how cleanly the publish side commits. D3a removes the crash window
between the park and the dispatch row; it does not remove redelivery. The agent
and the completion consumer both still dedupe on ``idempotency_key``.

⚠ Connection choice. This saver takes its OWN autocommit=False connection
rather than literally borrowing the crew's. The crew connection is autocommit
=True on purpose (``oe_grading_crew_wiring``: no dispatcher runs on it, so an
open transaction would never be flushed and the learner's submission would sit
in PENDING_OE_GRADING forever). Borrowing it would either destroy the
single-transaction property or silently retire that contract. A dedicated
connection into the same database and the same ``ai_kernel_outbox_events``
table delivers the property exactly, with no blast radius on the live lane.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from chora_ai_kernel_orchestrator.adapter.checkpointer.qgen_serde import (
    register_qgen_types,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    extract_dispatch_requests,
)

# ONE definition of "this connection is dead", shared with the Pub/Sub wrapper
# rather than copied: two copies of that predicate would drift, and the whole
# defect this liveness path closes was a second connection nobody kept in step.
from chora_ai_kernel_orchestrator.adapter.pubsub.reconnecting_connection import (
    _is_closed,
)

logger = logging.getLogger(__name__)

#: Test-on-borrow idle gate. Matches the Pub/Sub wrapper's default: a Cloud SQL
#: idle-close leaves psycopg's flags clean until the first op hits the dead
#: socket, so a borrow after this much idle is probed rather than trusted.
_DEFAULT_LIVENESS_IDLE_SECONDS = 30.0


class TransactionalDispatchSaver(BaseCheckpointSaver[str]):
    """Async checkpointer that commits agent dispatch with the park.

    ``inner``  — an ``AsyncPostgresSaver`` bound to ``conn``.
    ``conn``   — an autocommit=False async psycopg connection.
    ``outbox_writer`` — an ``AgentDispatchOutboxWriter`` on the SAME ``conn``.

    All three must share one connection or the transaction spans nothing.
    """

    def __init__(
        self,
        *,
        inner: Any,
        conn: Any,
        outbox_writer: Any,
        park_ledger_writer: Any = None,
        deadline_policy: Any = None,
        connect: Any = None,
        liveness_idle_seconds: float = _DEFAULT_LIVENESS_IDLE_SECONDS,
        monotonic: Any = time.monotonic,
    ) -> None:
        super().__init__()
        if inner is None or conn is None or outbox_writer is None:
            raise ValueError(
                "TransactionalDispatchSaver: inner, conn and outbox_writer are "
                "all required; a missing one would silently drop either the "
                "checkpoint or the dispatch"
            )
        # ADR-254 D5: the park ledger row (parked_at, deadline_at) commits in
        # this same transaction. Writer and policy come together or not at
        # all: a ledger without a deadline policy would stamp no deadline, and
        # a policy without a ledger would never be consulted.
        if (park_ledger_writer is None) != (deadline_policy is None):
            raise ValueError(
                "TransactionalDispatchSaver: park_ledger_writer and "
                "deadline_policy must be given together (ADR-254 D5: every "
                "parked run carries a deadline) or both omitted"
            )
        self._inner = inner
        self._conn = conn
        self._outbox = outbox_writer
        self._ledger = park_ledger_writer
        self._deadline_policy = deadline_policy
        # CHO-1649 liveness. `connect` is the SAME procedure that opened this
        # connection (agent_dispatch_wiring.open_saver_connection), so a
        # reconnected session is configured identically, GUC committed included.
        # None = no reconnect (unit tests with a fake conn): behaviour is then
        # byte-for-byte what it was before this change.
        self._connect = connect
        self._liveness_idle_seconds = liveness_idle_seconds
        self._monotonic = monotonic
        self._last_used_at: float | None = None
        # ⚠ Serialises every transaction on this connection. NOT optional.
        #
        # LangGraph runs graph tasks as concurrent asyncio tasks, and a psycopg
        # AsyncConnection is not safe for concurrent use: two coroutines each
        # opening conn.transaction() close them out of order and psycopg raises
        # OutOfOrderTransactionNesting. Live on 2026-08-20 that aborted the
        # exit-time checkpoint persistence, so LangGraph's _suppress_interrupt
        # raised instead of returning True, GraphInterrupt escaped ainvoke, and
        # the subscriber NACKed every message while the run never parked. The
        # failure presented as "the lane is dead", four layers from its cause.
        #
        # Safe against deadlock because no method here awaits another: each
        # acquires, does its work through the inner saver, and releases.
        self._lock = asyncio.Lock()
        inner_serde = getattr(inner, "serde", None)
        if inner_serde is not None:
            # AUDIT-G3: merge the qgen domain types in, so a parked qgen job stays
            # resumable once LangGraph enforces its msgpack allowlist.
            self.serde = register_qgen_types(inner_serde)

    # ---- connection liveness (CHO-1649) ----------------------------------
    #
    # ⚠ WHY THIS LIVES HERE AND NOT IN A WRAPPER. The obvious repair is to hand
    # this class a ReconnectingAsyncConnection. It is IMPOSSIBLE: langgraph's
    # `_ainternal.get_connection` is a hard type gate (isinstance AsyncConnection
    # or AsyncConnectionPool, else TypeError), and the wrapper subclasses
    # neither, so the inner saver raises on the first checkpoint read. The second
    # obvious repair, an AsyncConnectionPool, PASSES that gate and is worse:
    # every method here calls the inner saver INSIDE _tx(), and the inner saver
    # then acquires its OWN connection, so at max_size=1 it deadlocks and at
    # max_size>1 the checkpoint commits on a different connection from the outbox
    # row and the park stops being atomic. Both measured 2026-08-23. The single
    # dedicated connection is what makes the write atomic, so it stays, and the
    # liveness is added around it.

    def _holders(self) -> list[tuple[Any, str]]:
        """Every object holding this connection.

        ⚠ FOUR, not one. Re-pointing a subset is the same defect one layer down:
        the transaction would then span a live connection and a dead one.
        """
        holders: list[tuple[Any, str]] = [
            (self, "_conn"),
            (self._inner, "conn"),  # AsyncPostgresSaver keeps its own reference
            (self._outbox, "_conn"),
        ]
        if self._ledger is not None:
            holders.append((self._ledger, "_conn"))
        return holders

    def _participants(self) -> list[Any]:
        """Objects whose instance dict is scanned for a leaked reference to the
        OLD connection after a re-point. Superset of the objects in _holders()."""
        objs: list[Any] = [self, self._inner, self._outbox]
        if self._ledger is not None:
            objs.append(self._ledger)
        return objs

    async def _reconnect(self) -> None:
        """Replace the connection in EVERY holder, atomically from callers' view.

        Caller must hold ``self._lock``.
        """
        old = self._conn
        new = await self._connect()
        for obj, attr in self._holders():
            setattr(obj, attr, new)
        # Post-condition 1: every LISTED holder now points at the new connection.
        # Catches a holder that is in the list but could not be set (read-only
        # property, __slots__, a holder that copies rather than references).
        stale = [f"{type(obj).__name__}.{attr}" for obj, attr in self._holders() if getattr(obj, attr, None) is not new]
        # Post-condition 2: NO attribute of any participant still references the
        # OLD connection.
        #
        # ⚠ THIS IS KEYED ON THE OLD OBJECT, NOT ON `_holders()`, AND THAT IS THE
        # WHOLE POINT. A check that re-walks the same list the loop just wrote
        # cannot see a fifth attribute somebody adds and forgets to register:
        # the loop never touches it, the re-walk never looks at it, and the guard
        # passes green while the transaction splits across two connections. A
        # census keyed on the convention is blind to non-adopters; one keyed on
        # the artefact is not, and the old connection object cannot hide.
        #
        # Scope, stated so it is not over-trusted: this sees any attribute on a
        # PARTICIPANT (self, inner saver, outbox writer, park ledger writer). It
        # cannot see a collaborator that is not a participant at all. Adding one
        # that caches the connection still requires adding it to _participants().
        leaked = [
            f"{type(obj).__name__}.{attr}"
            for obj in self._participants()
            for attr, value in getattr(obj, "__dict__", {}).items()
            if value is old
        ]
        if stale or leaked:
            raise RuntimeError(
                "TransactionalDispatchSaver._reconnect: connection not fully "
                f"replaced. unset holders={stale} still-on-old={leaked}. "
                "park + outbox would no longer be atomic"
            )
        logger.warning(
            "transactional_dispatch_saver.reconnected",
            extra={"holders": len(self._holders())},
        )
        if old is not None:
            try:
                await old.close()
            except Exception as exc:  # noqa: BLE001
                # Closing an already-dead socket routinely raises and must not
                # mask the successful reconnect above. Logged, not suppressed:
                # a close failing for any OTHER reason is a leaked connection
                # and the next reader needs to see it.
                logger.warning(
                    "transactional_dispatch_saver.old_connection_close_failed",
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                )

    def _probe_due(self) -> bool:
        if self._last_used_at is None:
            return False
        return (self._monotonic() - self._last_used_at) >= self._liveness_idle_seconds

    async def _is_live(self) -> bool:
        """Cheap probe. ⚠ autocommit is OFF here, so the SELECT opens an implicit
        transaction that MUST be closed: leaving it open would make the real
        ``conn.transaction()`` below a nested savepoint instead of the outermost
        transaction, silently changing when the work commits."""
        try:
            async with self._conn.cursor() as cur:
                await cur.execute("SELECT 1")
            await self._conn.rollback()
            return True
        except Exception:  # noqa: BLE001
            # fail-loud-exempt: same as the Pub/Sub wrapper's probe. The failure
            # IS the return value ("reconnect"), the caller acts on it in the
            # very next line, and this runs on every borrow of the connection.
            return False

    async def _ensure_live(self) -> None:
        """Caller must hold ``self._lock``."""
        if self._connect is None:
            return
        if _is_closed(self._conn):
            await self._reconnect()
            return
        if self._probe_due() and not await self._is_live():
            await self._reconnect()

    @contextlib.asynccontextmanager
    async def _tx(self):
        """One serialised transaction on the shared connection, on a LIVE one.

        Retries only the BEGIN, never the body: at BEGIN nothing has executed
        yet, so reopening is safe, whereas retrying the body could double-apply
        a write that had already landed.
        """
        async with self._lock:
            await self._ensure_live()
            try:
                tx = self._conn.transaction()
                await tx.__aenter__()
            except Exception:
                if self._connect is None:
                    raise
                # Died between the probe and BEGIN. Nothing has run yet.
                await self._reconnect()
                tx = self._conn.transaction()
                await tx.__aenter__()
            try:
                yield
            except BaseException as exc:
                # __aexit__ may legitimately suppress (psycopg's Rollback).
                if not await tx.__aexit__(type(exc), exc, exc.__traceback__):
                    raise
            else:
                await tx.__aexit__(None, None, None)
            finally:
                self._last_used_at = self._monotonic()

    # ---- the load-bearing method -----------------------------------------

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Record the writes and, when one of them is an agent-dispatch park,
        the dispatch request — in ONE transaction."""
        requests = extract_dispatch_requests(writes)
        async with self._tx():
            await self._inner.aput_writes(config, writes, task_id, task_path)
            for request in requests:
                await self._outbox.queue_request(request)
                if self._ledger is not None:
                    # parked_at is the request's own occurred_at, so the ledger
                    # and the outbox row date the park identically.
                    parked_at = _dt.datetime.fromisoformat(str(request["envelope"]["occurred_at"]))
                    deadline_at = self._deadline_policy.deadline_for(request["body"]["agent_role"], parked_at=parked_at)
                    await self._ledger.queue_park(request, deadline_at=deadline_at)
        if requests:
            logger.info(
                "transactional_dispatch_saver.parked",
                extra={
                    "thread_id": config.get("configurable", {}).get("thread_id", ""),
                    "dispatches": [r.get("idempotency_key", "") for r in requests],
                },
            )

    # ---- everything else: delegate inside an explicit transaction ---------

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        async with self._tx():
            return await self._inner.aput(config, checkpoint, metadata, new_versions)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        async with self._tx():
            return await self._inner.aget_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002 — upstream signature
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # Materialise inside the transaction so the connection is released
        # before the caller starts consuming; an async generator that yields
        # across the transaction boundary would hold it open for the caller's
        # whole loop.
        async with self._tx():
            rows = [row async for row in self._inner.alist(config, filter=filter, before=before, limit=limit)]
        for row in rows:
            yield row

    async def adelete_thread(self, thread_id: str) -> None:
        async with self._tx():
            await self._inner.adelete_thread(thread_id)

    async def acopy_thread(self, *args: Any, **kwargs: Any) -> Any:
        async with self._tx():
            return await self._inner.acopy_thread(*args, **kwargs)

    async def adelete_for_runs(self, *args: Any, **kwargs: Any) -> Any:
        async with self._tx():
            return await self._inner.adelete_for_runs(*args, **kwargs)

    async def aprune(self, *args: Any, **kwargs: Any) -> Any:
        async with self._tx():
            return await self._inner.aprune(*args, **kwargs)

    # ---- channel versioning ----------------------------------------------

    def get_next_version(self, current: Any, channel: Any = None) -> Any:
        """Delegate to the inner saver. NOT optional, and not obviously a
        method a checkpointer "has" — which is why it was the one that bit.

        The runtime calls this on EVERY put to compute channel versions.
        ``BaseCheckpointSaver``'s default raises ``NotImplementedError`` the
        moment ``current`` is a string, and ``AsyncPostgresSaver`` uses string
        versions, so a saver that does not delegate dies on the second
        superstep of every run. It reached production on 2026-08-20 and
        surfaced only as a silently NACKing subscriber, because this platform's
        application logs are dropped at the sink.
        """
        return self._inner.get_next_version(current, channel)

    # ---- sync surface: refused, loudly -----------------------------------
    #
    # Every BaseCheckpointSaver method raises NotImplementedError by default,
    # so an un-overridden sync call would surface as that rather than as the
    # real problem. This saver is only correct on the async path (its inner is
    # AsyncPostgresSaver and its connection is an async connection); a sync
    # call means something compiled a graph and invoked it synchronously, and
    # silently doing nothing there would lose checkpoints AND dispatches.

    def _refuse_sync(self, method: str) -> None:
        raise NotImplementedError(
            f"TransactionalDispatchSaver.{method}: this saver is async-only "
            "(ADR-253 D3a — the dispatch row and the park share one async "
            "transaction). Invoke the graph with ainvoke/astream."
        )

    def put(self, *_: Any, **__: Any) -> Any:
        self._refuse_sync("put")

    def put_writes(self, *_: Any, **__: Any) -> Any:
        self._refuse_sync("put_writes")

    def get_tuple(self, *_: Any, **__: Any) -> Any:
        self._refuse_sync("get_tuple")

    def list(self, *_: Any, **__: Any) -> Any:
        self._refuse_sync("list")

    def delete_thread(self, *_: Any, **__: Any) -> Any:
        self._refuse_sync("delete_thread")

    def copy_thread(self, *_: Any, **__: Any) -> Any:
        self._refuse_sync("copy_thread")

    def delete_for_runs(self, *_: Any, **__: Any) -> Any:
        self._refuse_sync("delete_for_runs")

    def prune(self, *_: Any, **__: Any) -> Any:
        self._refuse_sync("prune")


__all__ = ["TransactionalDispatchSaver"]
