"""ParkReaper: settles parks the agent never answers (ADR-254 D5).

Four arms, one settle. ``settle_failed`` is what every arm does:

  1. dedupe on the reap key through the inbox (``<key>.reaped.<arm>``; distinct
     from the request key and from the real completion's ``<key>.completed``);
  2. resume the parked thread with a synthesized FAILED completion through the
     SAME router a real completion takes, so the crew's own failure handling
     runs (``AgentDispatchError`` in the node) and the caller receives a status
     on its own result topic, never silence;
  3. flip the ledger row to 'reaped' (only if still 'parked');
  4. queue ``chora.ai_kernel.crew.run_failed.v1`` with the ``arm`` through the
     outbox, so the dispatcher publishes it like every other event.

The ORDER is load-bearing: the resume comes first, so a failed resume leaves
the row parked and the inbox unmarked, and the next tick (or redelivery)
retries it. The reaper's connection is autocommit (the OE pattern); its ledger
flip and outbox insert are durable the moment they return.

Arms (b) and (d) run from the ledger in ``scan_once``; arms (a) and (c) arrive
through ``DispatchDlqSubscriber`` and call ``settle_by_key``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from collections.abc import Callable
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    reraise_if_dispatch_park,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
    AgentDispatchError,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.park import ParkRecord
from chora_ai_kernel_orchestrator.domain.agent_dispatch.reaper import (
    ReaperArm,
    reaped_completion,
    reaped_inbox_key,
    run_failed_event,
)

logger = logging.getLogger(__name__)

# Matches the completion subscriber: 7 days outlives every redelivery of the
# message that triggered the reap.
REAP_INBOX_TTL = _dt.timedelta(days=7)

ENV_REAPER_INTERVAL = "AGENT_DISPATCH_REAPER_INTERVAL_SECONDS"
ENV_REAPER_BATCH = "AGENT_DISPATCH_REAPER_BATCH"
DEFAULT_REAPER_INTERVAL_S = 60.0
DEFAULT_REAPER_BATCH = 100

OUTCOME_UNKNOWN = "unknown"
OUTCOME_NOT_PARKED = "not_parked"
OUTCOME_REAPED = "reaped"


class _LedgerLike(Protocol):
    async def get(self, idempotency_key: str) -> ParkRecord | None: ...
    async def mark_reaped(self, idempotency_key: str, *, arm: ReaperArm) -> bool: ...
    async def fetch_expired(self, *, limit: int) -> list[ParkRecord]: ...
    async def fetch_outbox_dead_lettered(self, *, limit: int) -> list[ParkRecord]: ...


class _RouterLike(Protocol):
    async def resume_reaped(self, completion: dict[str, Any]) -> None: ...


class _InboxLike(Protocol):
    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool: ...


class _OutboxWriterLike(Protocol):
    async def queue_request(self, request: dict[str, Any]) -> str: ...


class ParkReaper:
    def __init__(
        self,
        *,
        ledger: _LedgerLike,
        router: _RouterLike,
        inbox: _InboxLike,
        outbox_writer: _OutboxWriterLike,
        source_project: str,
        clock: Callable[[], _dt.datetime] | None = None,
    ) -> None:
        project = (source_project or "").strip()
        if not project:
            raise ValueError("ParkReaper: source_project is required")
        if ledger is None or router is None or inbox is None or outbox_writer is None:
            raise ValueError("ParkReaper: ledger, router, inbox and outbox_writer are required")
        self._ledger = ledger
        self._router = router
        self._inbox = inbox
        self._outbox = outbox_writer
        self._source_project = project
        self._clock = clock or (lambda: _dt.datetime.now(tz=_dt.UTC))

    # ---- the one settle every arm runs ---------------------------------

    async def settle_failed(
        self,
        park: ParkRecord,
        *,
        arm: ReaperArm,
        reason: str,
        original_topic: str,
        delivery_attempt: int,
    ) -> bool:
        """Settle one parked run FAILED. Returns False on a dedupe hit."""
        arm = ReaperArm(arm)
        key = reaped_inbox_key(arm, park.idempotency_key)

        async def _settle() -> None:
            now = self._clock()
            completion = reaped_completion(park, arm=arm, reason=reason, reaped_at=now)
            settle_reason = reason
            try:
                await self._router.resume_reaped(completion)
            except AgentDispatchError as exc:
                # The crew surfaced the FAILED completion by RAISING out of the
                # graph instead of settling to a FAILED terminal. Propagating
                # would NACK a DLQ pull message that has no onward DLQ and
                # redeliver it forever while the row stayed parked. Settle the
                # park anyway, carry the raise on the bus and in the log: the
                # missing terminal is the crew's defect, loud here, never a
                # silent loop.
                logger.error(
                    "park_reaper.crew_raised_on_failed_resume",
                    extra={
                        "arm": arm.value,
                        "idempotency_key": park.idempotency_key,
                        "thread_id": park.thread_id,
                        "crew": park.crew,
                        "agent_role": park.agent_role,
                        "err": f"{type(exc).__name__}: {exc}"[:500],
                    },
                )
                settle_reason = (f"{reason}; crew raised on the FAILED resume: {type(exc).__name__}: {exc}")[:1000]
            except Exception as exc:
                # Transient (ledger unreachable, a bug): leave the row parked and
                # the inbox unmarked so the redelivery or the next tick retries.
                reraise_if_dispatch_park(exc)
                raise
            await self._ledger.mark_reaped(park.idempotency_key, arm=arm)
            event = run_failed_event(
                park,
                arm=arm,
                reason=settle_reason,
                original_topic=original_topic,
                delivery_attempt=delivery_attempt,
                reaped_at=now,
                source_project=self._source_project,
            )
            await self._outbox.queue_request(event)
            logger.warning(
                "park_reaper.reaped",
                extra={
                    "arm": arm.value,
                    "idempotency_key": park.idempotency_key,
                    "thread_id": park.thread_id,
                    "agent_role": park.agent_role,
                    "crew": park.crew,
                    "tenant_id": park.tenant_id,
                    "parked_at": park.parked_at.isoformat(),
                    "deadline_at": park.deadline_at.isoformat(),
                    "delivery_attempt": delivery_attempt,
                    "reason": reason,
                },
            )

        return await self._inbox.process(key=key, ttl=REAP_INBOX_TTL, fn=_settle)

    async def settle_by_key(
        self,
        idempotency_key: str,
        *,
        arm: ReaperArm,
        reason: str,
        original_topic: str,
        delivery_attempt: int,
    ) -> str:
        """Arms (a) and (c): settle the park behind a dead-lettered message."""
        park = await self._ledger.get(idempotency_key)
        if park is None:
            # A dead letter for a run this ledger never saw (parked before the
            # ledger existed and not backfilled, or a message that never
            # parked anything). Loud, counted, and NOT an error path that would
            # NACK: a DLQ pull subscription has no onward DLQ, so a NACK here
            # loops forever.
            logger.error(
                "park_reaper.unknown_park",
                extra={
                    "arm": ReaperArm(arm).value,
                    "idempotency_key": idempotency_key,
                    "original_topic": original_topic,
                    "delivery_attempt": delivery_attempt,
                },
            )
            return OUTCOME_UNKNOWN
        if not park.is_parked:
            logger.info(
                "park_reaper.already_settled",
                extra={
                    "arm": ReaperArm(arm).value,
                    "idempotency_key": idempotency_key,
                    "state": park.state.value,
                    "settled_by": park.settled_by,
                },
            )
            return OUTCOME_NOT_PARKED
        await self.settle_failed(
            park,
            arm=arm,
            reason=reason,
            original_topic=original_topic,
            delivery_attempt=delivery_attempt,
        )
        return OUTCOME_REAPED

    # ---- arms (b) and (d): the scan ------------------------------------

    async def scan_once(self, *, limit: int = DEFAULT_REAPER_BATCH) -> dict[str, int]:
        """One pass over the ledger: deadline-expired parks (b) and parks whose
        dispatch request dead-lettered in the outbox (d).

        One failing row must not stop the scan for every tenant: each settle is
        isolated, logged with its traceback and COUNTED in ``errors``; the row
        stays 'parked' and is retried next tick. That is a visible loop, not a
        swallow. A LangGraph park escaping the resume is re-raised (it would
        mean the saver contract broke; the scan must die loud, not tick on).
        """
        tallies = {"request_expired": 0, "outbox_dead_lettered": 0, "errors": 0}

        for park in await self._ledger.fetch_expired(limit=limit):
            reason = (
                f"deadline {park.deadline_at.isoformat()} passed without a completion "
                f"(parked {park.parked_at.isoformat()} on {park.agent_role})"
            )
            try:
                if await self.settle_failed(
                    park,
                    arm=ReaperArm.REQUEST_EXPIRED,
                    reason=reason,
                    original_topic=park.request_topic,
                    delivery_attempt=0,
                ):
                    tallies["request_expired"] += 1
            except Exception as exc:
                reraise_if_dispatch_park(exc)
                tallies["errors"] += 1
                logger.exception(
                    "park_reaper.scan.settle_failed",
                    extra={"arm": "request_expired", "idempotency_key": park.idempotency_key},
                )

        for park in await self._ledger.fetch_outbox_dead_lettered(limit=limit):
            reason = (
                f"dispatch request dead-lettered in the outbox before it was ever published to {park.request_topic}"
            )
            try:
                if await self.settle_failed(
                    park,
                    arm=ReaperArm.OUTBOX_DEAD_LETTERED,
                    reason=reason,
                    original_topic=park.request_topic,
                    delivery_attempt=0,
                ):
                    tallies["outbox_dead_lettered"] += 1
            except Exception as exc:
                reraise_if_dispatch_park(exc)
                tallies["errors"] += 1
                logger.exception(
                    "park_reaper.scan.settle_failed",
                    extra={"arm": "outbox_dead_lettered", "idempotency_key": park.idempotency_key},
                )

        if any(tallies.values()):
            logger.info("park_reaper.scan.summary", extra=tallies)
        return tallies


class ParkReaperScanLoop:
    """Runs ``scan_once`` forever on a cadence; the kennel lifespan owns it."""

    def __init__(
        self,
        *,
        reaper: ParkReaper,
        interval_s: float = DEFAULT_REAPER_INTERVAL_S,
        batch: int = DEFAULT_REAPER_BATCH,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("ParkReaperScanLoop: interval_s must be > 0")
        if batch < 1:
            raise ValueError("ParkReaperScanLoop: batch must be >= 1")
        self._reaper = reaper
        self._interval_s = float(interval_s)
        self._batch = int(batch)
        self.ticks = 0
        self.last_error: str = ""

    @property
    def interval_s(self) -> float:
        return self._interval_s

    async def run_forever(self) -> None:  # pragma: no cover - integration loop
        while True:
            try:
                await self._reaper.scan_once(limit=self._batch)
                self.last_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reraise_if_dispatch_park(exc)
                # The scan itself failed (ledger unreachable, for instance):
                # loud, recorded for /readyz, and retried next tick.
                self.last_error = f"{type(exc).__name__}: {exc}"[:500]
                logger.exception("park_reaper.scan.tick_failed")
            self.ticks += 1
            try:
                await asyncio.sleep(self._interval_s)
            except asyncio.CancelledError:
                raise


__all__ = [
    "DEFAULT_REAPER_BATCH",
    "DEFAULT_REAPER_INTERVAL_S",
    "ENV_REAPER_BATCH",
    "ENV_REAPER_INTERVAL",
    "OUTCOME_NOT_PARKED",
    "OUTCOME_REAPED",
    "OUTCOME_UNKNOWN",
    "REAP_INBOX_TTL",
    "ParkReaper",
    "ParkReaperScanLoop",
]
