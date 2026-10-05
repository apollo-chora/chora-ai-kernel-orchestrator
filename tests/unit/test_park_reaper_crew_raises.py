"""RED: a crew that RAISES on a FAILED resume must not make the reaper loop.

The reaper resumes a parked thread with a synthesized FAILED completion so the
crew's own failure handling settles the run (a FAILED terminal to the caller).
A crew whose node raises ``AgentDispatchError`` out of the graph instead of
settling leaves the kennel with a choice: propagate (the DLQ message NACKs and,
on a DLQ pull subscription with no onward DLQ, redelivers forever while the
ledger row stays 'parked') or settle the park anyway. The reaper settles: the
row flips to 'reaped', run_failed.v1 carries the crew's raise in its reason,
and the message ACKs. The crew's missing terminal is the crew's defect (fixed
separately in the OE crew), loud in the log and on the bus, never a silent
loop. Any OTHER exception (ledger down, bug) still propagates: that is a
transient and a retry is right.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.park_reaper import ParkReaper
from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
    AgentDispatchError,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.park import (
    ParkRecord,
    ParkState,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.reaper import ReaperArm

_NOW = _dt.datetime(2026, 8, 22, 16, 0, tzinfo=_dt.UTC)


def _park() -> ParkRecord:
    return ParkRecord(
        idempotency_key="agent_dispatch.oe_evaluate.sub-9:q1:1",
        workflow_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        thread_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        crew="oe_grading",
        tenant_id="11111111-1111-7111-8111-111111111111",
        gcid="00000000-0000-7000-8000-000000001999",
        agent_role="oe_evaluate",
        request_topic="chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
        completion_topic="chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1",
        traceparent="",
        tracestate="",
        parked_at=_NOW - _dt.timedelta(days=8),
        deadline_at=_NOW - _dt.timedelta(days=1),
        state=ParkState.PARKED,
    )


class _Ledger:
    def __init__(self) -> None:
        self.reaped: list[tuple[str, str]] = []

    async def mark_reaped(self, key: str, *, arm: ReaperArm) -> bool:
        self.reaped.append((key, arm.value))
        return True


class _RaisingRouter:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    async def resume_reaped(self, completion: dict[str, Any]) -> None:
        self.calls += 1
        raise self._exc


class _Inbox:
    def __init__(self) -> None:
        self.marked: list[str] = []

    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool:
        await fn()
        self.marked.append(key)
        return True


class _Outbox:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def queue_request(self, event: dict[str, Any]) -> str:
        self.events.append(event)
        return "row"


@pytest.mark.asyncio
async def test_a_crew_that_raises_agent_dispatch_error_is_still_settled_loudly() -> None:
    ledger, inbox, outbox = _Ledger(), _Inbox(), _Outbox()
    router = _RaisingRouter(AgentDispatchError("oe_evaluate dispatch x returned status=FAILED"))
    reaper = ParkReaper(
        ledger=ledger,
        router=router,
        inbox=inbox,
        outbox_writer=outbox,
        source_project="chora-489812",
        clock=lambda: _NOW,
    )

    ran = await reaper.settle_failed(
        _park(),
        arm=ReaperArm.REQUEST_DEAD_LETTERED,
        reason="request dead-lettered after 5 delivery attempts",
        original_topic="chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
        delivery_attempt=5,
    )

    assert ran is True
    assert router.calls == 1
    assert ledger.reaped == [("agent_dispatch.oe_evaluate.sub-9:q1:1", "request_dead_lettered")]
    assert len(outbox.events) == 1
    reason = outbox.events[0]["body"]["reason"]
    assert "request dead-lettered after 5 delivery attempts" in reason
    assert "crew raised on the FAILED resume" in reason
    assert "status=FAILED" in reason
    assert inbox.marked, "the reap must be marked so a redelivery dedupes"


@pytest.mark.asyncio
async def test_any_other_resume_failure_still_propagates_and_settles_nothing() -> None:
    ledger, inbox, outbox = _Ledger(), _Inbox(), _Outbox()
    router = _RaisingRouter(RuntimeError("ledger connection lost"))
    reaper = ParkReaper(
        ledger=ledger,
        router=router,
        inbox=inbox,
        outbox_writer=outbox,
        source_project="chora-489812",
        clock=lambda: _NOW,
    )
    with pytest.raises(RuntimeError, match="ledger connection lost"):
        await reaper.settle_failed(
            _park(),
            arm=ReaperArm.REQUEST_EXPIRED,
            reason="deadline passed",
            original_topic="t",
            delivery_attempt=0,
        )
    assert ledger.reaped == []
    assert outbox.events == []
    assert inbox.marked == []
