"""RED: ADR-254 D5, the park reaper settles parks the agent never answers.

Every arm runs the same settle: dedupe on the reap key (the inbox), resume the
parked thread with a synthesized FAILED completion through the SAME router a
real completion takes, flip the ledger row to 'reaped', and queue
``chora.ai_kernel.crew.run_failed.v1`` with the arm. The scan arms (b, d) run
from the ledger; the DLQ arms (a, c) arrive through the subscriber.

The ORDER matters and is asserted: resume first (a failed resume must leave
the row parked so the next tick retries), then the ledger flip, then the
terminal event.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.park_reaper import ParkReaper
from chora_ai_kernel_orchestrator.domain.agent_dispatch.park import (
    ParkRecord,
    ParkState,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.reaper import (
    RUN_FAILED_TOPIC,
    ReaperArm,
    reaped_inbox_key,
)

_NOW = _dt.datetime(2026, 8, 22, 15, 0, tzinfo=_dt.UTC)


def _park(key: str = "agent_dispatch.oe_evaluate.sub-1:q1:1", state: ParkState = ParkState.PARKED) -> ParkRecord:
    return ParkRecord(
        idempotency_key=key,
        workflow_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        thread_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        crew="oe_grading",
        tenant_id="11111111-1111-7111-8111-111111111111",
        gcid="00000000-0000-7000-8000-000000001999",
        agent_role="oe_evaluate",
        request_topic="chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
        completion_topic="chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1",
        traceparent="00-aa-bb-01",
        tracestate="",
        parked_at=_NOW - _dt.timedelta(days=8),
        deadline_at=_NOW - _dt.timedelta(days=1),
        state=state,
    )


class _Journal:
    def __init__(self) -> None:
        self.entries: list[str] = []


class _FakeLedger:
    def __init__(self, journal: _Journal, parks: dict[str, ParkRecord] | None = None) -> None:
        self._journal = journal
        self.parks = parks or {}
        self.expired: list[ParkRecord] = []
        self.outbox_dead: list[ParkRecord] = []
        self.reaped: list[tuple[str, str]] = []

    async def get(self, key: str) -> ParkRecord | None:
        return self.parks.get(key)

    async def mark_reaped(self, key: str, *, arm: ReaperArm) -> bool:
        self._journal.entries.append(f"mark_reaped:{arm.value}")
        self.reaped.append((key, arm.value))
        return True

    async def fetch_expired(self, *, limit: int) -> list[ParkRecord]:
        return list(self.expired)[:limit]

    async def fetch_outbox_dead_lettered(self, *, limit: int) -> list[ParkRecord]:
        return list(self.outbox_dead)[:limit]


class _FakeRouter:
    def __init__(self, journal: _Journal, fail_for: set[str] | None = None) -> None:
        self._journal = journal
        self.completions: list[dict[str, Any]] = []
        self._fail_for = fail_for or set()

    async def resume_reaped(self, completion: dict[str, Any]) -> None:
        if completion["idempotency_key"] in self._fail_for:
            raise RuntimeError("resume exploded")
        self._journal.entries.append("resume")
        self.completions.append(completion)


class _FakeInbox:
    """Mirrors InboxIdempotencyStore.process: seen -> skip; else run fn, mark
    only when fn returned."""

    def __init__(self, seen: set[str] | None = None) -> None:
        self.seen = set(seen or ())
        self.keys: list[str] = []

    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool:
        if key in self.seen:
            return False
        await fn()
        self.seen.add(key)
        self.keys.append(key)
        return True


class _FakeOutbox:
    def __init__(self, journal: _Journal) -> None:
        self._journal = journal
        self.events: list[dict[str, Any]] = []

    async def queue_request(self, event: dict[str, Any]) -> str:
        self._journal.entries.append("queue_run_failed")
        self.events.append(event)
        return "row-1"


def _reaper(
    journal: _Journal,
    ledger: _FakeLedger,
    router: _FakeRouter,
    inbox: _FakeInbox,
    outbox: _FakeOutbox,
) -> ParkReaper:
    return ParkReaper(
        ledger=ledger,
        router=router,
        inbox=inbox,
        outbox_writer=outbox,
        source_project="chora-489812",
        clock=lambda: _NOW,
    )


@pytest.mark.asyncio
async def test_settle_failed_resumes_flips_and_emits_in_that_order() -> None:
    journal = _Journal()
    park = _park()
    ledger = _FakeLedger(journal, {park.idempotency_key: park})
    router, inbox, outbox = _FakeRouter(journal), _FakeInbox(), _FakeOutbox(journal)
    reaper = _reaper(journal, ledger, router, inbox, outbox)

    ran = await reaper.settle_failed(
        park,
        arm=ReaperArm.REQUEST_DEAD_LETTERED,
        reason="request dead-lettered after 5 delivery attempts",
        original_topic=park.request_topic,
        delivery_attempt=5,
    )

    assert ran is True
    assert journal.entries == ["resume", "mark_reaped:request_dead_lettered", "queue_run_failed"]
    completion = router.completions[0]
    assert completion["status"] == "FAILED"
    assert completion["thread_id"] == park.thread_id
    assert completion["idempotency_key"] == park.idempotency_key
    assert completion["reaper_arm"] == "request_dead_lettered"
    event = outbox.events[0]
    assert event["topic"] == RUN_FAILED_TOPIC
    assert event["body"]["arm"] == "request_dead_lettered"
    assert event["body"]["delivery_attempt"] == 5
    assert event["envelope"]["source_project"] == "chora-489812"
    assert inbox.keys == [reaped_inbox_key(ReaperArm.REQUEST_DEAD_LETTERED, park.idempotency_key)]


@pytest.mark.asyncio
async def test_settle_failed_is_idempotent_on_redelivery() -> None:
    journal = _Journal()
    park = _park()
    key = reaped_inbox_key(ReaperArm.REQUEST_EXPIRED, park.idempotency_key)
    ledger, router = _FakeLedger(journal), _FakeRouter(journal)
    inbox, outbox = _FakeInbox(seen={key}), _FakeOutbox(journal)
    reaper = _reaper(journal, ledger, router, inbox, outbox)

    ran = await reaper.settle_failed(
        park,
        arm=ReaperArm.REQUEST_EXPIRED,
        reason="deadline passed",
        original_topic=park.request_topic,
        delivery_attempt=0,
    )

    assert ran is False
    assert journal.entries == []


@pytest.mark.asyncio
async def test_a_failed_resume_leaves_the_row_parked_and_the_inbox_unmarked() -> None:
    journal = _Journal()
    park = _park()
    ledger = _FakeLedger(journal)
    router = _FakeRouter(journal, fail_for={park.idempotency_key})
    inbox, outbox = _FakeInbox(), _FakeOutbox(journal)
    reaper = _reaper(journal, ledger, router, inbox, outbox)

    with pytest.raises(RuntimeError, match="resume exploded"):
        await reaper.settle_failed(
            park,
            arm=ReaperArm.REQUEST_EXPIRED,
            reason="deadline passed",
            original_topic=park.request_topic,
            delivery_attempt=0,
        )

    assert ledger.reaped == []
    assert outbox.events == []
    assert inbox.keys == []


@pytest.mark.asyncio
async def test_settle_by_key_reports_unknown_not_parked_and_reaped() -> None:
    journal = _Journal()
    parked = _park()
    settled = _park(key="agent_dispatch.oe_evaluate.sub-2:q1:1", state=ParkState.COMPLETED)
    ledger = _FakeLedger(journal, {parked.idempotency_key: parked, settled.idempotency_key: settled})
    router, inbox, outbox = _FakeRouter(journal), _FakeInbox(), _FakeOutbox(journal)
    reaper = _reaper(journal, ledger, router, inbox, outbox)

    assert (
        await reaper.settle_by_key(
            "agent_dispatch.oe_evaluate.nobody:1",
            arm=ReaperArm.REQUEST_DEAD_LETTERED,
            reason="r",
            original_topic="t",
            delivery_attempt=5,
        )
        == "unknown"
    )
    assert (
        await reaper.settle_by_key(
            settled.idempotency_key,
            arm=ReaperArm.COMPLETION_DEAD_LETTERED,
            reason="r",
            original_topic="t",
            delivery_attempt=5,
        )
        == "not_parked"
    )
    assert (
        await reaper.settle_by_key(
            parked.idempotency_key,
            arm=ReaperArm.REQUEST_DEAD_LETTERED,
            reason="r",
            original_topic="t",
            delivery_attempt=5,
        )
        == "reaped"
    )
    assert [c["idempotency_key"] for c in router.completions] == [parked.idempotency_key]


@pytest.mark.asyncio
async def test_scan_once_reaps_expired_and_outbox_dead_lettered_parks() -> None:
    journal = _Journal()
    a, b = _park(key="agent_dispatch.oe_evaluate.a:1"), _park(key="agent_dispatch.oe_evaluate.b:1")
    d = _park(key="agent_dispatch.oe_evaluate.d:1", state=ParkState.PARKED)
    ledger = _FakeLedger(journal)
    ledger.expired = [a, b]
    ledger.outbox_dead = [d]
    router, inbox, outbox = _FakeRouter(journal), _FakeInbox(), _FakeOutbox(journal)
    reaper = _reaper(journal, ledger, router, inbox, outbox)

    tallies = await reaper.scan_once(limit=100)

    assert tallies == {"request_expired": 2, "outbox_dead_lettered": 1, "errors": 0}
    assert [r for _, r in ledger.reaped] == ["request_expired", "request_expired", "outbox_dead_lettered"]
    reasons = [e["body"]["reason"] for e in outbox.events]
    assert "deadline" in reasons[0] and a.deadline_at.isoformat() in reasons[0]
    assert "outbox" in reasons[2]
    assert outbox.events[2]["body"]["original_topic"] == d.request_topic


@pytest.mark.asyncio
async def test_scan_once_isolates_one_failing_row_and_counts_it() -> None:
    journal = _Journal()
    good, bad = _park(key="agent_dispatch.oe_evaluate.good:1"), _park(key="agent_dispatch.oe_evaluate.bad:1")
    ledger = _FakeLedger(journal)
    ledger.expired = [bad, good]
    router = _FakeRouter(journal, fail_for={bad.idempotency_key})
    inbox, outbox = _FakeInbox(), _FakeOutbox(journal)
    reaper = _reaper(journal, ledger, router, inbox, outbox)

    tallies = await reaper.scan_once(limit=100)

    assert tallies == {"request_expired": 1, "outbox_dead_lettered": 0, "errors": 1}
    assert [k for k, _ in ledger.reaped] == [good.idempotency_key]


def test_reaper_refuses_a_blank_source_project() -> None:
    journal = _Journal()
    with pytest.raises(ValueError):
        ParkReaper(
            ledger=_FakeLedger(journal),
            router=_FakeRouter(journal),
            inbox=_FakeInbox(),
            outbox_writer=_FakeOutbox(journal),
            source_project="  ",
        )
