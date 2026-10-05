"""PostgresSaver factory + thread_id helper.

Per `langgraph-orchestrator-python` skill the checkpointer thread_id is
formed as `{tenant_id}:{workflow_id}:{run_id}` so checkpoints are
isolated per tenant + workflow + run. We do NOT embed any user
identifier in the thread_id since that would leak across tenants.

Per CLAUDE.md §6 + memory `feedback_no_inline_config`: the Postgres DSN
comes from one of:

- ``CHORA_AI_KERNEL_PG_DSN`` — direct DSN (dev override / explicit).
- ``CHORA_AI_KERNEL_PG_DSN_SECRET_ID`` — Secret Manager secret resource
  name resolved at boot via ``adapter.secrets.resolve_dsn``.

Falls through to InMemorySaver when neither is set — keeps unit tests +
local dev runnable without a live Postgres.

The PostgresSaver is built lazily: we instantiate the class but do NOT
open a connection at import time. The first checkpoint write triggers
the connection; LangGraph's PostgresSaver auto-creates the underlying
schema on first use (see migrations/0001_initial.sql notes).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, cast

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
from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

logger = logging.getLogger(__name__)

# libpq TCP keepalive params — kernel pings the socket every 60s and drops
# a dead connection after 3 retries × 10s = ~90s. Smaller than Cloud SQL's
# default ~10-min idle cutoff so the server-side drop never fires on
# otherwise-healthy connections. Documented per psycopg `connect()`
# passthrough to libpq:
# https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS
_LIBPQ_KEEPALIVE_PARAMS = (
    ("keepalives", "1"),
    ("keepalives_idle", "60"),
    ("keepalives_interval", "10"),
    ("keepalives_count", "3"),
)

# Test-on-borrow idle gate (CHO-2341 defect 1). Keepalives (Option A) defend the
# ~10-min Cloud SQL idle cutoff, but a cost-pause STOPs the instance entirely:
# the socket dies with psycopg's ``.closed``/``.broken`` both clean until the
# first op raises "server closed the connection unexpectedly", forcing a rollout
# restart. When a checkpoint borrow follows more idle than this, actively probe
# (``SELECT 1``) and reopen on failure so the FIRST borrow after resume succeeds.
# Kept above any realistic graph-run gap so hot paths skip the probe. Mirrors
# ReconnectingAsyncConnection's gate on the outbox path.
_DEFAULT_LIVENESS_IDLE_SECONDS = 30.0


def _append_libpq_keepalive(dsn: str) -> str:
    """Return ``dsn`` with libpq TCP keepalive params appended.

    Caller-supplied params are honoured + not duplicated (the comparison
    is purely on the key name so e.g. ``?keepalives=0`` already in the
    DSN is preserved as-is — operators have an explicit override). The
    function is pure + side-effect free; tests cover the no-query +
    with-query DSN shapes.
    """
    sep = "&" if "?" in dsn else "?"
    additions: list[str] = []
    for key, value in _LIBPQ_KEEPALIVE_PARAMS:
        # Look for ?key= or &key= to detect a caller-supplied value.
        if (f"?{key}=" in dsn) or (f"&{key}=" in dsn):
            continue
        additions.append(f"{key}={value}")
    if not additions:
        return dsn
    return dsn + sep + "&".join(additions)


def build_thread_id(*, tenant_id: str, workflow_id: str, run_id: str) -> str:
    """Build the canonical LangGraph thread_id.

    Format: ``{tenant_id}:{workflow_id}:{run_id}`` — three colon-separated
    segments; PostgresSaver indexes on the full string, so each
    {tenant, workflow, run} triple gets its own checkpoint thread.
    """
    if not tenant_id.strip():
        raise ValueError("tenant_id is required for thread_id")
    if not workflow_id.strip():
        raise ValueError("workflow_id is required for thread_id")
    if not run_id.strip():
        raise ValueError("run_id is required for thread_id")
    return f"{tenant_id}:{workflow_id}:{run_id}"


def build_weakness_thread_id(*, tenant_id: str, upload_id: str) -> str:
    """Build the DETERMINISTIC Growth-Edge crew thread_id (CHO-1973 Wave A).

    Format: ``{tenant_id}:{upload_id}`` — TWO colon-separated segments, NO
    run_id. The crew runs exactly one analysis per upload and a "reiterate" stays
    in-thread (it loops HITL_review → diagnose on the same checkpoint), so the FE
    resume route reconstructs the thread from tenant + upload ALONE — no run_id to
    carry on the wire (ADR-205 D4/D5 reconciliation). A run_id is still minted
    internally for cost attribution; it just does not key the checkpoint.

    Kept SEPARATE from ``build_thread_id`` (which stays 3-segment) so the qgen /
    OE / ai-assist crews — whose thread IS {tenant}:{workflow}:{run} — are
    unaffected.
    """
    if not tenant_id.strip():
        raise ValueError("tenant_id is required for weakness thread_id")
    if not upload_id.strip():
        raise ValueError("upload_id is required for weakness thread_id")
    return f"{tenant_id}:{upload_id}"


def build_checkpointer_from_env() -> Any:
    """Return a LangGraph checkpointer based on env config.

    - DSN resolved (either ``CHORA_AI_KERNEL_PG_DSN`` directly or via
      ``CHORA_AI_KERNEL_PG_DSN_SECRET_ID`` → Secret Manager) -> a
      PostgresSaver bound to the DSN. The saver is built via the
      upstream ``from_conn_string`` contextmanager and we adopt the
      inner saver (caller is responsible for closing the connection
      at process shutdown — wired into the FastAPI lifespan).
    - DSN unresolvable (both env vars unset / blank, or fetcher failure
      caught upstream) -> InMemorySaver fallback (unit tests, local dev).

    The synchronous ``PostgresSaver`` is returned by default. Async callers
    should use the async cousin via ``build_async_checkpointer_from_env``.
    """
    dsn = resolve_dsn()
    if not dsn:
        # ⚠ DO NOT DELETE THIS FALLBACK AS "unused in production".
        # It is reached at IMPORT TIME, not just at runtime: main.py builds the
        # app at module scope (`app = build_app_from_env()`), which calls this,
        # and no test sets a DSN globally. Removing it therefore breaks pytest
        # COLLECTION (tests/unit/test_fail_loud_guards.py imports main), not one
        # test. Measured 2026-08-23 by deleting it: the whole suite failed to
        # collect. A reference count says "2 sites"; the mutation says
        # "the entire suite".
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    # Postgres path: build the saver lazily via a thin proxy that opens the
    # upstream contextmanager on first checkpoint write. This avoids
    # connecting at import time so unit tests + readiness probes stay
    # cheap, while production callers can `enter()` explicitly inside the
    # FastAPI lifespan to surface connection errors at startup.
    from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415

    return LazyPostgresSaver(dsn=dsn, builder=PostgresSaver)


class LazyPostgresSaver(BaseCheckpointSaver[int]):
    # ⚠ DO NOT DELETE AS VESTIGIAL. Two dependencies, one of them invisible
    # from this file:
    #   1. PRODUCTION APP CHECKPOINTER. The live kennel sets
    #      CHORA_AI_KERNEL_PG_DSN_SECRET_ID, so build_checkpointer_from_env()
    #      returns THIS, and main.py's FastAPI lifespan opens it at startup and
    #      closes it at shutdown via `isinstance(saver, LazyPostgresSaver)`.
    #   2. IT PROVISIONS THE CHECKPOINT TABLES THE DISPATCH LANES ASSUME EXIST.
    #      adapter/pubsub/agent_dispatch_wiring.py builds each lane's own
    #      AsyncPostgresSaver and states in its comment that the tables "are
    #      created by the shared saver's setup already", and that shared saver
    #      is this one. Deleting it does not drop a wrapper, it removes the thing
    #      the five transactional lanes depend on. That dependency is expressed
    #      only in prose, in another file, pointing the opposite way.
    # Audited 2026-08-23 (G6) and closed as INVALID: "nothing in scope uses it"
    # was not "nothing uses it".
    """Lazy `BaseCheckpointSaver` proxy for `langgraph.checkpoint.postgres.PostgresSaver`.

    Defers the upstream PostgresSaver's `from_conn_string` contextmanager
    open until the first checkpoint write OR an explicit ``open()`` call
    (typically wired into the FastAPI lifespan). Keeps unit tests +
    readiness probes cheap by avoiding a TCP open at import time.

    Why subclass BaseCheckpointSaver?

    Per langgraph 0.6+ ``langgraph.types.ensure_valid_checkpointer`` the
    saver passed to ``graph.compile(checkpointer=...)`` MUST be an
    instance of ``BaseCheckpointSaver`` (or one of the literals
    ``True``/``False``/``None``). Subclassing keeps the proxy pattern
    while satisfying that isinstance gate — the alternative of passing
    ``self._inner`` post-open leaks the inner saver into the composition
    root and breaks the lazy contract.

    Per [[feedback-d6-resilience-first-class]] this is a P1 pod-death
    survival enabler — without it `qgen_crew.lifespan.startup_failed`
    crashes orchestrator startup before the StreamingPull subscriber
    can resume in-flight workflow runs from checkpoints.

    Method override discipline:
        Every method on ``BaseCheckpointSaver`` raises ``NotImplementedError``
        by default (they are non-``@abstractmethod`` so subclassing alone
        is enough for isinstance). We MUST therefore explicitly override
        each one (sync + async surface) — Python's MRO resolves the
        parent class methods before any ``__getattr__`` fallback could.

    Closed lifecycle: `close()` is idempotent; the
    `close_checkpointer()` module helper wraps it for FastAPI lifespan
    teardown.
    """

    def __init__(
        self,
        *,
        dsn: str,
        builder: Any,
        liveness_idle_seconds: float = _DEFAULT_LIVENESS_IDLE_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        # NOTE: do NOT call super().__init__() yet — when _inner is opened
        # we adopt the inner saver's serde. Before open we want the
        # BaseCheckpointSaver default (JsonPlusSerializer via the class
        # attribute) so probing code does not crash with AttributeError.
        super().__init__()
        self._dsn = dsn
        self._builder = builder
        self._cm: Any | None = None
        self._inner: Any | None = None
        # Test-on-borrow idle gate (CHO-2341 defect 1): a cost-pause/resume can
        # leave the raw socket dead with ``.closed``/``.broken`` still clean
        # until the first op raises. After more idle than this, actively probe
        # before handing the inner back. ``monotonic`` is injectable for tests.
        self._liveness_idle_seconds = liveness_idle_seconds
        self._monotonic = monotonic
        self._last_used_at: float | None = None

    # ---- proxy lifecycle ------------------------------------------------

    @property
    def opened(self) -> bool:
        return self._inner is not None

    def open(self) -> Any:
        """Open the underlying PostgresSaver contextmanager.

        Idempotent — repeated calls return the same inner saver. Adopts
        the inner saver's ``serde`` so encoding behaviour matches the
        upstream PostgresSaver once connected.

        Appends libpq TCP keepalive params to the DSN (Option A of the
        CHECKPOINTER-STALE-CONN fix combo, handoff doc 2026-05-17) so
        the kernel keeps the socket demonstrably alive to Cloud SQL's
        proxy and the server-side idle cutoff (~10 min) never fires
        on otherwise-healthy connections. Caller-supplied keepalive
        params are honoured + not duplicated.
        """
        if self._inner is not None:
            return self._inner
        dsn = _append_libpq_keepalive(self._dsn)
        cm = self._builder.from_conn_string(dsn)
        self._cm = cm
        self._inner = cm.__enter__()
        inner_serde = getattr(self._inner, "serde", None)
        if inner_serde is not None:
            # AUDIT-G3: merge the qgen domain types in, so a parked qgen job stays
            # resumable once LangGraph enforces its msgpack allowlist.
            self.serde = register_qgen_types(inner_serde)
        return self._inner

    def close(self) -> None:
        if self._cm is None:
            return
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            # Never re-raise from a close: the caller is already tearing down
            # and an exception here would mask whatever it was tearing down
            # for. Loud, though. A connection pool that failed to close is why
            # the NEXT boot finds sessions still open on the database.
            logger.exception("lazy_postgres_saver.close_failed")
        self._cm = None
        self._inner = None

    def _raw_conn(self) -> Any:
        """The inner PostgresSaver's raw psycopg connection, or None.

        psycopg's PostgresSaver wraps a raw connection exposed as either
        ``.conn`` (newer langgraph) or ``._conn`` (older). None when neither is
        present (InMemorySaver, unit-test fakes, or a future upstream refactor).
        """
        if self._inner is None:
            return None
        raw = getattr(self._inner, "conn", None)
        if raw is None:
            raw = getattr(self._inner, "_conn", None)
        return raw

    def _is_inner_live(self) -> bool:
        """Return whether the cached inner has a known-live raw connection.

        Checks BOTH ``.closed`` (a clean close) AND ``.broken`` (a lost socket /
        server restart left the connection unusable after a failing op). psycopg3
        can leave ``.closed`` 0 while ``.broken`` is True, so ``.closed`` alone is
        a false-negative for a server-side drop, the exact gap that let a
        cost-paused connection be handed back and throw. When there is no raw
        connection to introspect we assume live (the stale-detection contract is
        "only reopen on a KNOWN-dead signal"); the idle probe in ``_ensure_open``
        covers the silent case both flags miss.
        """
        if self._inner is None:
            return False
        raw = self._raw_conn()
        if raw is None:
            return True  # no introspection surface → assume live
        return not (bool(getattr(raw, "closed", False)) or bool(getattr(raw, "broken", False)))

    def _probe_due(self) -> bool:
        """True when the connection has been idle long enough that a silent
        server-side close could have killed the socket without flagging
        ``.closed``/``.broken``. Never probes before the first borrow (nothing
        has been used yet), which keeps lazy-connect + the hot path allocation-
        free."""
        if self._last_used_at is None:
            return False
        return (self._monotonic() - self._last_used_at) >= self._liveness_idle_seconds

    def _mark_used(self) -> None:
        self._last_used_at = self._monotonic()

    def _probe_live(self) -> bool:
        """Active test-on-borrow liveness probe: run ``SELECT 1`` on the raw sync
        connection. Any failure (including the silent server-close that only
        surfaces on first use) counts as dead so the caller reopens. No raw
        connection to probe (fakes / InMemorySaver) → assume live. The langgraph
        PostgresSaver connection is autocommit, so the probe leaves no open
        transaction behind."""
        raw = self._raw_conn()
        if raw is None:
            return True
        try:
            with raw.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001
            # fail-loud-exempt: the probe's whole job is to answer this boolean.
            # Any failure IS the answer ("reconnect"), the caller acts on it
            # immediately, and logging every miss on this hot path would bury
            # the failures that matter.
            return False

    def _ensure_open(self) -> Any:
        """Open lazily and return a live inner saver.

        All delegating methods route through here so the contract is in one
        place. Three reopen triggers, cheapest first:

          1. never opened -> ``open()``.
          2. the raw connection is flagged dead (``.closed`` OR ``.broken``,
             Option B) -> close + reopen.
          3. flags are clean BUT the connection has been idle past
             ``liveness_idle_seconds`` -> actively probe (``SELECT 1``) and
             reopen on failure (Option C, CHO-2341 defect 1). This is the only
             reliable signal for a Cloud SQL cost-pause/resume, where the socket
             dies silently and neither flag is set until the first real op
             raises. Mirrors ReconnectingAsyncConnection's test-on-borrow gate.

        The hot path (flagged-live AND used within the idle window) skips the
        probe entirely, so per-op overhead is a couple of cheap checks. Option A
        (libpq keepalives, applied in ``open()``) rounds out the combo.
        """
        if self._inner is None:
            self.open()
        elif not self._is_inner_live():
            self.close()
            self.open()
        elif self._probe_due() and not self._probe_live():
            logger.warning(
                "lazy_postgres_saver.stale_on_borrow",
                extra={"reason": "liveness probe failed after idle"},
            )
            self.close()
            self.open()
        self._mark_used()
        return self._inner

    # ---- sync BaseCheckpointSaver surface -------------------------------

    @property
    def config_specs(self) -> list[Any]:
        # Avoid forcing an open() just to probe config specs — pods that
        # haven't reached a checkpoint write yet should still be able to
        # introspect. Once opened, defer to the inner.
        if self._inner is None:
            return []
        return cast("list[Any]", self._inner.config_specs)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return cast("CheckpointTuple | None", self._ensure_open().get_tuple(config))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002 — mirrors base signature
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        return cast(
            "Iterator[CheckpointTuple]",
            self._ensure_open().list(config, filter=filter, before=before, limit=limit),
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return cast(
            "RunnableConfig",
            self._ensure_open().put(config, checkpoint, metadata, new_versions),
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._ensure_open().put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        self._ensure_open().delete_thread(thread_id)

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        self._ensure_open().delete_for_runs(run_ids)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self._ensure_open().copy_thread(source_thread_id, target_thread_id)

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        self._ensure_open().prune(thread_ids, strategy=strategy)

    # ---- async BaseCheckpointSaver surface ------------------------------
    #
    # The inner saver is a SYNC ``PostgresSaver`` (per
    # ``build_checkpointer_from_env`` line 85). The upstream PostgresSaver
    # does not implement the async ``a*`` methods — they inherit the
    # NotImplementedError stub from BaseCheckpointSaver. To keep LangGraph's
    # async graph engine happy without dragging in asyncpg + the async
    # variant's contextmanager refactor, each async wrapper here calls the
    # corresponding SYNC method on the inner saver. The sync call blocks
    # the asyncio event loop briefly per DB roundtrip — acceptable for
    # demo / single-tenant burst load + flagged as tech debt for the
    # AsyncPostgresSaver migration (see follow-up note in builder docstring
    # at build_async_checkpointer_from_env).

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return cast(
            "CheckpointTuple | None",
            self._ensure_open().get_tuple(config),
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002 — mirrors base signature
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        inner = self._ensure_open()
        # PostgresSaver.list is a sync generator. Iterate synchronously +
        # yield through into the async generator surface.
        for item in inner.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return cast(
            "RunnableConfig",
            self._ensure_open().put(config, checkpoint, metadata, new_versions),
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._ensure_open().put_writes(config, writes, task_id, task_path)

    # ---- async tail surface (defaults raise on base; delegate to sync) ---

    async def adelete_thread(self, thread_id: str) -> None:
        self._ensure_open().delete_thread(thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        self._ensure_open().delete_for_runs(run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self._ensure_open().copy_thread(source_thread_id, target_thread_id)

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        self._ensure_open().prune(thread_ids, strategy=strategy)

    # ---- version helper -------------------------------------------------

    def get_next_version(self, current: int | None, channel: None) -> int:
        # When opened, delegate so the version scheme matches the inner.
        # When unopened (rare — versions are usually queried mid-graph),
        # fall back to the BaseCheckpointSaver default impl.
        if self._inner is not None:
            return cast("int", self._inner.get_next_version(current, channel))
        return super().get_next_version(current, channel)


def close_checkpointer(saver: Any) -> None:
    """Close any LazyPostgresSaver wrapper (best-effort, and loud on failure)."""
    closer = getattr(saver, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:
        logger.exception("checkpointer.close_failed")


def build_async_checkpointer_from_env() -> Any:
    """Async cousin of `build_checkpointer_from_env`.

    Returns AsyncPostgresSaver when DSN is configured; InMemorySaver
    otherwise. Used by FastAPI handlers that want awaitable checkpoint
    writes alongside `graph.ainvoke`.

    Caller is responsible for the matching contextmanager exit (see
    FastAPI lifespan in `adapter/http/handlers.py`).
    """
    dsn = resolve_dsn()
    if not dsn:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415

    return AsyncPostgresSaver.from_conn_string(dsn)
