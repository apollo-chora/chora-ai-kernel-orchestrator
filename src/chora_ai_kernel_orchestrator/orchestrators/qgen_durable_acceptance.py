"""DurableQGenAcceptance - the ADR-251 D4 acceptance/drive split (CHO-2398).

Sits between QGenCrewSubscriber and QGenRunnerRouter. ``handle_started``
performs ACCEPTANCE only: register the job in the in-flight registry (durable,
committed), spawn the graph run as a background asyncio task, and return so
the subscriber ACKs. The Pub/Sub callback timeout now bounds acceptance, not
the whole run; a 200-question job can no longer crawl through redelivery
windows into the DLQ half-done.

Fail-loud contract:
  * A registry failure RAISES out of ``handle_started`` -> the subscriber
    NACKs -> redelivery -> DLQ. Acceptance keeps delivery-resilience Pillar 3.
  * A drive failure is logged loudly and recorded in ``failed_drives``; the
    registry row REMAINS so the boot sweep (QGenResumeSweeper) retries it.
    Never a fabricated terminal, never a silent drop.
  * Registry removal on success is NOT this class's job: the terminal outbox
    writer deletes the row in the same transaction as the terminal event.

The in-process ``_driving`` map makes a duplicate start (redelivery racing
the ACK) and a concurrent resume idempotent: one drive per assist_id per
process. Cross-process duplication is already covered by the subscriber's
inbox dedupe + the checkpointer's thread semantics.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _RouterLike(Protocol):
    async def handle_started(self, event: dict[str, Any]) -> None: ...

    async def handle_completion(self, completion: dict[str, Any]) -> Any: ...


class _RegistryLike(Protocol):
    async def register(
        self,
        *,
        assist_id: str,
        tenant_id: str,
        author_gcid: str,
        started_event: dict[str, Any],
    ) -> None: ...


class DurableQGenAcceptance:
    """Acceptance/drive split around the qgen runner router."""

    def __init__(self, *, registry: _RegistryLike, router: _RouterLike) -> None:
        self._registry = registry
        self._router = router
        self._driving: dict[str, asyncio.Task[None]] = {}
        # assist_id -> repr(exception) of the last failed drive. Observability
        # for the sweep + tests; never consulted for control flow.
        self.failed_drives: dict[str, str] = {}

    # ---- Subscriber-facing ------------------------------------------------

    async def handle_started(self, event: dict[str, Any]) -> None:
        """ACCEPT the job durably, spawn the drive, return (caller ACKs)."""
        assist_id = str(event.get("assist_id") or event.get("job_id") or "").strip()
        if not assist_id:
            raise ValueError("durable acceptance: started event carries no assist_id")
        if assist_id in self._driving:
            logger.info(
                "qgen_durable_acceptance.duplicate_start_ignored",
                extra={"assist_id": assist_id},
            )
            return

        await self._registry.register(
            assist_id=assist_id,
            tenant_id=str(event.get("tenant_id") or ""),
            author_gcid=str(event.get("author_gcid") or event.get("gcid") or ""),
            started_event=event,
        )
        self.spawn_drive(assist_id, event)

    # ---- Completion-facing (ADR-254 D5) -------------------------------------

    async def handle_completion(self, completion: dict[str, Any]) -> None:
        """Resume the parked job INLINE (so a failed resume NACKs and redelivers)
        and, when the graph reached a terminal, run the SETTLE (terminal
        publish, decision + HITL emits, the composer) as a tracked background
        drive: nothing slower than a checkpoint round trip sits on the
        completion ack path, and the shutdown drain + the boot sweep cover a
        settle cut by a pod death exactly as they cover a cut start (the
        registry row outlives it; a re-drive of a finished thread settles
        without re-invoking)."""
        outcome = await self._router.handle_completion(completion)
        settle = getattr(outcome, "settle", None)
        if settle is None:
            return
        thread_id = str(completion.get("thread_id") or "").strip()
        assist_id = thread_id.split(":", 1)[0] or thread_id
        self._spawn(assist_id, settle(), kind="settle")

    # ---- Drive management -------------------------------------------------

    def spawn_drive(self, assist_id: str, event: dict[str, Any]) -> None:
        """Start the background graph run for an accepted or resumed job."""
        if assist_id in self._driving:
            return
        self._spawn(
            assist_id, self._router.handle_started(event), kind="drive", tenant_id=str(event.get("tenant_id") or "")
        )

    def _spawn(self, assist_id: str, coro: Any, *, kind: str, tenant_id: str = "") -> None:
        if assist_id in self._driving:
            coro.close()
            logger.info(
                "qgen_durable_acceptance.duplicate_%s_ignored",
                kind,
                extra={"assist_id": assist_id},
            )
            return
        task = asyncio.create_task(
            self._run(assist_id, coro, kind=kind, tenant_id=tenant_id),
            name=f"qgen-{kind}-{assist_id}",
        )
        self._driving[assist_id] = task

    def is_driving(self, assist_id: str) -> bool:
        return assist_id in self._driving

    async def wait_idle(self, timeout: float = 30.0) -> None:  # noqa: ASYNC109 — timeout is the public param name callers use
        """Await every in-flight drive (shutdown + tests)."""
        tasks = list(self._driving.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)

    async def _run(self, assist_id: str, coro: Any, *, kind: str, tenant_id: str = "") -> None:
        try:
            await coro
            self.failed_drives.pop(assist_id, None)
        except Exception as exc:
            # Loud, attributed, and durable-by-omission: the registry row is
            # still there, so the next boot sweep resumes this job from its
            # checkpoint (a cut settle re-settles from the finished thread).
            # Swallowing here would be a silent drop; re-raising would kill an
            # unrelated task group.
            self.failed_drives[assist_id] = f"{exc.__class__.__name__}: {exc}"
            logger.exception(
                "qgen_durable_acceptance.%s_failed",
                kind,
                extra={"assist_id": assist_id, "tenant_id": tenant_id},
            )
        finally:
            self._driving.pop(assist_id, None)


__all__ = ["DurableQGenAcceptance"]
