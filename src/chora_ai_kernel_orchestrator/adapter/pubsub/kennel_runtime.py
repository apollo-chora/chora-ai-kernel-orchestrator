"""KennelRuntime: the per-process singletons every lane shares (ADR-254 D5).

ONE of each, composed once in ``main.py``'s lifespan and started once:

  * the park ledger store (migration 0056) and the deadline policy;
  * the completion router (role -> runner) and the ONE completion loop over the
    union of every registered lane's completion subscriptions;
  * the park reaper: its scan loop (arms b and d) and the DLQ loop over every
    lane's dead-letter pull subscriptions (arms a and c);
  * the single outbox drain (two dispatchers on ``ai_kernel_outbox_events``
    double-publish: the claim is a non-durable ``SKIP LOCKED``).

Lanes REGISTER into it: name, crew, the dispatch roles they park on, and the
runner that resumes them. A role can be bound once; a lane that parks on a
role nobody resumes cannot start. Every lane reports ``mark_lane_started`` /
``mark_lane_failed`` so ``/readyz`` can tell a pod that consumes from one that
only looks healthy. Before this, ``main.py`` kept the single-drain invariant
with a four-deep chain of ``if not X_tasks`` guards and ``/readyz`` checked
only the guardrail while five lanes could degrade to ``None`` on startup.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from chora_ai_kernel_orchestrator.adapter.postgres.kernel_sweeper import (
    SWEEPER_SESSION_SETTINGS,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_completion_subscriber import (
    AgentCompletionSubscriber,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.dispatch_dlq_subscriber import (
    DispatchDlqSubscriber,
    dlq_pull_subscription_names,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.park_reaper import (
    DEFAULT_REAPER_BATCH,
    DEFAULT_REAPER_INTERVAL_S,
    ParkReaperScanLoop,
)

logger = logging.getLogger(__name__)

DEFAULT_DRAIN_INTERVAL_S = 2.0
DEFAULT_DRAIN_BATCH = 100


@dataclass
class LaneRecord:
    name: str
    crew: str
    roles: tuple[str, ...]
    started: bool = False
    failed: bool = False
    detail: str = ""


async def drain_outbox_forever(
    dispatcher: Any,
    *,
    interval_s: float = DEFAULT_DRAIN_INTERVAL_S,
    batch_size: int = DEFAULT_DRAIN_BATCH,
) -> None:
    """Drain ``ai_kernel_outbox_events`` on a cadence, forever.

    ``drain_once`` records per-row failures in the store (mark_failed /
    deadletter) and does not raise for them; a raise here is the drain itself
    failing (connection gone), which is logged with its traceback and retried
    next tick. Cancellation propagates.
    """
    while True:
        try:
            await dispatcher.drain_once(batch_size=batch_size)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("outbox_dispatcher.drain_once_failed")
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            raise


class KennelRuntime:
    def __init__(
        self,
        *,
        pubsub_project: str,
        deadline_policy: Any,
        ledger: Any,
        router: Any,
        reaper: Any,
        inbox: Any,
        dispatcher: Any,
        saver_factory: Callable[..., Awaitable[Any]],
        completion_loop_factory: Callable[[Any, list[str]], Any],
        dlq_loop_factory: Callable[[Any, list[str]], Any],
        scan_interval_s: float = DEFAULT_REAPER_INTERVAL_S,
        scan_batch: int = DEFAULT_REAPER_BATCH,
        drain_interval_s: float = DEFAULT_DRAIN_INTERVAL_S,
    ) -> None:
        project = (pubsub_project or "").strip()
        if not project:
            raise ValueError("KennelRuntime: pubsub_project is required")
        for name, value in (
            ("deadline_policy", deadline_policy),
            ("ledger", ledger),
            ("router", router),
            ("reaper", reaper),
            ("inbox", inbox),
            ("dispatcher", dispatcher),
            ("saver_factory", saver_factory),
            ("completion_loop_factory", completion_loop_factory),
            ("dlq_loop_factory", dlq_loop_factory),
        ):
            if value is None:
                raise ValueError(f"KennelRuntime: {name} is required")
        self._pubsub_project = project
        self._deadline_policy = deadline_policy
        self._ledger = ledger
        self._router = router
        self._reaper = reaper
        self._inbox = inbox
        self._dispatcher = dispatcher
        self._saver_factory = saver_factory
        self._completion_loop_factory = completion_loop_factory
        self._dlq_loop_factory = dlq_loop_factory
        self._scan_interval_s = float(scan_interval_s)
        self._scan_batch = int(scan_batch)
        self._drain_interval_s = float(drain_interval_s)

        self._lanes: dict[str, LaneRecord] = {}
        self._crew_for_role: dict[str, str] = {}
        self._closers: list[Callable[[], Awaitable[Any]]] = []
        self._completion_loop: Any = None
        self._dlq_loop: Any = None
        self._scan_loop: ParkReaperScanLoop | None = None
        self.scan_task: asyncio.Task[None] | None = None
        self.drain_task: asyncio.Task[None] | None = None
        self.started = False

    # ---- accessors ------------------------------------------------------

    @property
    def pubsub_project(self) -> str:
        return self._pubsub_project

    @property
    def deadline_policy(self) -> Any:
        return self._deadline_policy

    @property
    def ledger(self) -> Any:
        return self._ledger

    @property
    def router(self) -> Any:
        return self._router

    @property
    def reaper(self) -> Any:
        return self._reaper

    @property
    def crew_for_role(self) -> dict[str, str]:
        return dict(self._crew_for_role)

    @property
    def lanes(self) -> dict[str, LaneRecord]:
        return dict(self._lanes)

    # ---- registration ---------------------------------------------------

    def register_lane(
        self,
        name: str,
        *,
        crew: str,
        roles: Iterable[str],
        runner: Any,
    ) -> LaneRecord:
        lane = _require(name, "lane name")
        crew_name = _require(crew, "crew")
        role_list = [r.strip() for r in roles if (r or "").strip()]
        if lane in self._lanes:
            raise ValueError(f"KennelRuntime: lane {lane!r} is already registered")
        if role_list and runner is None:
            raise ValueError(
                f"KennelRuntime: lane {lane!r} dispatches {role_list} but registered "
                "no runner; a run parked on those roles could never be resumed"
            )
        # Bind (role, crew) first: a duplicate pair raises before the lane is
        # recorded. A role MAY be shared by several crews (ADR-254 D5); the
        # router then resolves a completion by the ledger row's crew.
        for role in role_list:
            self._router.register(role, runner, crew=crew_name)
        for role in role_list:
            # The backfill map (pre-0056 parks) keeps the FIRST crew that bound
            # the role: those legacy rows predate every shared-role lane.
            self._crew_for_role.setdefault(role, crew_name)
        record = LaneRecord(name=lane, crew=crew_name, roles=tuple(role_list))
        self._lanes[lane] = record
        logger.info(
            "kennel_runtime.lane_registered",
            extra={"lane": lane, "crew": crew_name, "roles": role_list},
        )
        return record

    async def transactional_saver(self, *, crew: str) -> Any:
        """The D3a saver for one crew: park + outbox row + ledger row, one tx."""
        return await self._saver_factory(crew=_require(crew, "crew"))

    def register_closer(self, closer: Callable[[], Awaitable[Any]]) -> None:
        """A resource (the runtime's own DB connection) to close on ``stop``."""
        self._closers.append(closer)

    def mark_lane_started(self, name: str) -> None:
        record = self._lane(name)
        record.started = True
        record.failed = False
        record.detail = ""

    def mark_lane_failed(self, name: str, detail: str) -> None:
        record = self._lane(name)
        record.started = False
        record.failed = True
        record.detail = (detail or "")[:500]

    # ---- lifecycle ------------------------------------------------------

    async def start(self) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("KennelRuntime: already started")
        backfilled = await self._ledger.backfill_from_outbox(
            crew_for_role=dict(self._crew_for_role),
            deadline_policy=self._deadline_policy,
        )
        roles = list(self._router.roles)
        dlq_subscriptions: list[str] = []
        if roles:
            self._completion_loop = self._completion_loop_factory(
                AgentCompletionSubscriber(runner=self._router, inbox=self._inbox),
                roles,
            )
            await self._completion_loop.start()
            dlq_subscriptions = dlq_pull_subscription_names(roles)
            self._dlq_loop = self._dlq_loop_factory(
                DispatchDlqSubscriber(reaper=self._reaper),
                dlq_subscriptions,
            )
            await self._dlq_loop.start()
        self._scan_loop = ParkReaperScanLoop(
            reaper=self._reaper,
            interval_s=self._scan_interval_s,
            batch=self._scan_batch,
        )
        self.scan_task = asyncio.create_task(
            self._scan_loop.run_forever(),
            name="kennel-reaper-scan",
        )
        self.drain_task = asyncio.create_task(
            drain_outbox_forever(self._dispatcher, interval_s=self._drain_interval_s),
            name="kennel-outbox-drain",
        )
        self.started = True
        summary = {
            "backfilled": backfilled,
            "roles": roles,
            "dlq_subscriptions": dlq_subscriptions,
            "lanes": sorted(self._lanes),
        }
        logger.info("kennel_runtime.started", extra=summary)
        return summary

    async def stop(self) -> None:
        for task in (self.drain_task, self.scan_task):
            if task is None or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("kennel_runtime.task_stop_failed", extra={"task": task.get_name()})
        for loop in (self._dlq_loop, self._completion_loop):
            if loop is None:
                continue
            try:
                await loop.stop()
            except Exception:
                logger.exception("kennel_runtime.loop_stop_failed")
        for closer in self._closers:
            try:
                await closer()
            except Exception:
                logger.exception("kennel_runtime.closer_failed")
        self.started = False
        logger.info("kennel_runtime.stopped")

    # ---- readiness ------------------------------------------------------

    def readiness(self) -> dict[str, Any]:
        reasons: list[str] = []
        if not self.started:
            reasons.append("runtime_not_started")
        for name, record in sorted(self._lanes.items()):
            if record.failed:
                reasons.append(f"lane_failed:{name}")
            elif not record.started:
                reasons.append(f"lane_not_started:{name}")
        if self.started:
            if self.drain_task is None or self.drain_task.done():
                reasons.append("drain_task_dead")
            if self.scan_task is None or self.scan_task.done():
                reasons.append("scan_task_dead")
            if self._scan_loop is not None and self._scan_loop.last_error:
                reasons.append(f"scan_error:{self._scan_loop.last_error}")
            if self._router.roles:
                if self._completion_loop is None or not getattr(self._completion_loop, "started", True):
                    reasons.append("completion_loop_missing")
                if self._dlq_loop is None or not getattr(self._dlq_loop, "started", True):
                    reasons.append("dlq_loop_missing")
        return {
            "ready": not reasons,
            "reasons": reasons,
            "roles": list(self._router.roles),
            "lanes": {
                name: {
                    "crew": r.crew,
                    "roles": list(r.roles),
                    "started": r.started,
                    "failed": r.failed,
                    "detail": r.detail,
                }
                for name, r in sorted(self._lanes.items())
            },
        }

    def _lane(self, name: str) -> LaneRecord:
        try:
            return self._lanes[name]
        except KeyError:
            raise ValueError(f"KennelRuntime: unknown lane {name!r}; registered: {sorted(self._lanes)}") from None


def require_lane_built(name: str, components: Any) -> Any:
    """The fail-loud startup rule (ADR-254 D5): an ENABLED lane whose builder
    returned ``None`` is a configuration error, not a degradation.

    The lane builders return ``None`` when a required env var or dependency is
    missing and log a ``<lane>_wiring.skipped`` line at INFO. Before this,
    ``main.py`` accepted that ``None`` and the pod came up READY with the lane
    simply absent (the 2026-07 growth-edge cutover that "ran" with no lane is
    this exact shape). Now the process refuses to start.
    """
    if components is None:
        raise RuntimeError(
            f"{name} is enabled but its builder returned None: a required env var "
            f"or dependency is missing (see the '{name}' wiring 'skipped' log line). "
            "Refusing to start a pod that would look ready and consume nothing."
        )
    return components


async def build_kennel_runtime_from_env(
    *,
    deadline_policy: Any | None = None,
) -> KennelRuntime | None:  # pragma: no cover - integration glue; the class is unit-tested
    """Compose the runtime from env.

    Returns ``None`` only when ``NATS_URL`` or the DSN is unset, in
    which case no lane can dispatch either (their builders refuse on the same
    inputs). A bad reaper cadence value raises: a typo must not silently pick
    the default.
    """
    import os

    nats_url = (os.getenv("NATS_URL") or "").strip()
    if not nats_url:
        logger.info("kennel_runtime.skipped: NATS_URL unset")
        return None
    # Provenance label stamped on the outbox envelope (source_project). The
    # The cloud project id is gone; the service name is the source now.
    project = "chora-ai-kernel-orchestrator"
    from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

    dsn = resolve_dsn()
    if not dsn:
        logger.info("kennel_runtime.skipped: CHORA_AI_KERNEL_PG_DSN unset")
        return None

    import psycopg

    from chora_ai_kernel_orchestrator.adapter.pubsub import (
        NatsPublisher,
        OutboxDispatcher,
        PostgresOutboxStore,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_completion_loop import (
        AgentCompletionPubsubLoop,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_outbox_writer import (
        AgentDispatchOutboxWriter,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_wiring import (
        build_transactional_saver,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.completion_router import (
        CompletionRouter,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.inbox_idempotency import (
        InboxIdempotencyStore,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.park_ledger import ParkLedgerStore
    from chora_ai_kernel_orchestrator.adapter.pubsub.park_reaper import (
        ENV_REAPER_BATCH,
        ENV_REAPER_INTERVAL,
        ParkReaper,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.reconnecting_connection import (
        ReconnectingAsyncConnection,
    )
    from chora_ai_kernel_orchestrator.domain.agent_dispatch.deadline_policy import (
        ParkDeadlinePolicy,
    )

    policy = deadline_policy if deadline_policy is not None else ParkDeadlinePolicy.from_env()

    raw_interval = (os.getenv(ENV_REAPER_INTERVAL) or "").strip()
    raw_batch = (os.getenv(ENV_REAPER_BATCH) or "").strip()
    try:
        interval_s = float(raw_interval) if raw_interval else DEFAULT_REAPER_INTERVAL_S
        batch = int(raw_batch) if raw_batch else DEFAULT_REAPER_BATCH
    except ValueError as exc:
        raise ValueError(
            f"kennel runtime: {ENV_REAPER_INTERVAL}={raw_interval!r} / {ENV_REAPER_BATCH}={raw_batch!r} are not numbers"
        ) from exc

    async def _connect(d: str) -> Any:
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                return await psycopg.AsyncConnection.connect(d)
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
                last_exc = exc
                if attempt == 5:
                    break
                logger.warning(
                    "kennel_runtime.db_connect.retry",
                    extra={"attempt": attempt, "delay_s": delay, "err": f"{exc.__class__.__name__}: {exc}"},
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
        assert last_exc is not None
        raise last_exc

    # The runtime's OWN autocommit connection (the OE pattern): the ledger
    # flips, the reap inbox marks and the run_failed outbox rows are durable the
    # moment they return, and the single drain's reads release with COMMIT.
    # Sweeper mode, EXPLICIT (G2): the reaper age scan, the completion router's
    # key lookups and the ledger backfill span every tenant by design, so this
    # connection opts in to the kernel-sweeper arm of the 0055/0056 policies
    # rather than relying on an unset tenant GUC matching everything. Re-applied
    # on every reconnect by the wrapper — a dropped opt-in would read 0 rows and
    # the reaper would reap nothing while testing green.
    conn = ReconnectingAsyncConnection(
        dsn,
        autocommit=True,
        connect=_connect,
        session_settings=SWEEPER_SESSION_SETTINGS,
    )
    await conn.connect()

    ledger = ParkLedgerStore(conn=conn)
    inbox = InboxIdempotencyStore(conn=conn)
    router = CompletionRouter(ledger=ledger)
    reaper = ParkReaper(
        ledger=ledger,
        router=router,
        inbox=inbox,
        outbox_writer=AgentDispatchOutboxWriter(conn=conn, source_project=project),
        source_project=project,
    )
    publisher = NatsPublisher(url=nats_url)
    store = PostgresOutboxStore(conn=conn, worker_id="kennel-runtime-drain-1")
    dispatcher = OutboxDispatcher(
        store=store,
        publisher=publisher,
        worker_id="kennel-runtime-drain-1",
    )

    async def _saver(*, crew: str) -> Any:
        return await build_transactional_saver(
            dsn=dsn,
            source_project=project,
            crew=crew,
            deadline_policy=policy,
        )

    def _completion_loop(subscriber: Any, roles: list[str]) -> Any:
        loop = AgentCompletionPubsubLoop.from_env(subscriber=subscriber, agent_roles=roles)
        if loop is None:
            raise RuntimeError("kennel runtime: completion loop could not be built")
        loop._url = nats_url  # noqa: SLF001 — explicit url; from_env already read it
        return loop

    def _dlq_loop(subscriber: Any, subscriptions: list[str]) -> Any:
        loop = AgentCompletionPubsubLoop(
            subscriber=subscriber,
            project=project,
            subscriptions=subscriptions,
        )
        loop._url = nats_url  # noqa: SLF001 — direct construction; set the url
        return loop

    runtime = KennelRuntime(
        pubsub_project=project,
        deadline_policy=policy,
        ledger=ledger,
        router=router,
        reaper=reaper,
        inbox=inbox,
        dispatcher=dispatcher,
        saver_factory=_saver,
        completion_loop_factory=_completion_loop,
        dlq_loop_factory=_dlq_loop,
        scan_interval_s=interval_s,
        scan_batch=batch,
    )
    runtime.register_closer(conn.close)
    logger.info(
        "kennel_runtime.built",
        extra={
            "project": project,
            "scan_interval_s": interval_s,
            "scan_batch": batch,
            "default_deadline_s": policy.default_seconds,
            "per_role_deadline_s": dict(policy.per_role_seconds),
        },
    )
    return runtime


def _require(value: str, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"KennelRuntime: {field} is required")
    return text


__all__ = [
    "DEFAULT_DRAIN_BATCH",
    "DEFAULT_DRAIN_INTERVAL_S",
    "KennelRuntime",
    "LaneRecord",
    "build_kennel_runtime_from_env",
    "drain_outbox_forever",
    "require_lane_built",
]
