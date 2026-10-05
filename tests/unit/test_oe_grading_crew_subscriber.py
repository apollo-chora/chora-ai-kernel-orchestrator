"""OEGradingCrewSubscriber unit tests — RED→GREEN per [[feedback-strict-tdd]].

Exercises the Pub/Sub message → event projection: JSON decode, envelope-on-
attributes merge (tenant_id/gcid/traceparent ride attributes, NOT the body, for
chora-delivery's PublishCustom path), dedup-key extraction, and ack-after-
processing. NO live Pub/Sub.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.oe_grading_crew_subscriber import (
    OEGradingCrewSubscriber,
)


@dataclass
class _FakeMsg:
    data: bytes
    attributes: dict[str, str]
    acked: bool = False
    nacked: bool = False

    def ack(self) -> None:
        self.acked = True

    def nack(self) -> None:
        self.nacked = True


@dataclass
class _FakeRunner:
    events: list[dict[str, Any]] = field(default_factory=list)

    async def handle_requested(self, event: dict[str, Any]) -> None:
        self.events.append(event)


@dataclass
class _FakeInbox:
    """Minimal idempotent.Store double — runs fn once per key."""

    seen: set[str] = field(default_factory=set)

    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool:
        if key in self.seen:
            return False
        self.seen.add(key)
        await fn()
        return True


def _body() -> bytes:
    # submission_requested.v1 body — NOTE: no tenant_id (rides the attribute).
    return json.dumps(
        {
            "submission_id": "sub-1",
            "assessment_id": "assess-1",
            "learner_gcid": "learner-1",
            "total_points_possible": 5,
            "questions": [],
        }
    ).encode("utf-8")


@pytest.mark.asyncio
async def test_merges_tenant_id_from_attributes() -> None:
    runner, inbox = _FakeRunner(), _FakeInbox()
    sub = OEGradingCrewSubscriber(runner=runner, inbox=inbox)
    msg = _FakeMsg(data=_body(), attributes={"tenant_id": "tenant-9", "event_id": "e1"})

    await sub.handle_message(msg)

    assert msg.acked and not msg.nacked
    assert len(runner.events) == 1
    # The bug fix: tenant_id (attribute-only) must be merged into the event so
    # parse_requested → state.tenant_id is non-empty.
    assert runner.events[0]["tenant_id"] == "tenant-9"


@pytest.mark.asyncio
async def test_threads_traceparent_from_attributes() -> None:
    runner, inbox = _FakeRunner(), _FakeInbox()
    sub = OEGradingCrewSubscriber(runner=runner, inbox=inbox)
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    msg = _FakeMsg(data=_body(), attributes={"tenant_id": "t", "event_id": "e1", "traceparent": tp})

    await sub.handle_message(msg)
    assert runner.events[0]["traceparent"] == tp


@pytest.mark.asyncio
async def test_missing_dedup_key_nacks() -> None:
    runner, inbox = _FakeRunner(), _FakeInbox()
    sub = OEGradingCrewSubscriber(runner=runner, inbox=inbox)
    msg = _FakeMsg(data=_body(), attributes={"tenant_id": "t"})  # no event_id/idempotency_key

    await sub.handle_message(msg)
    assert msg.nacked and not msg.acked
    assert runner.events == []


@pytest.mark.asyncio
async def test_decode_failure_nacks() -> None:
    runner, inbox = _FakeRunner(), _FakeInbox()
    sub = OEGradingCrewSubscriber(runner=runner, inbox=inbox)
    msg = _FakeMsg(data=b"{not json", attributes={"tenant_id": "t", "event_id": "e1"})

    await sub.handle_message(msg)
    assert msg.nacked and not msg.acked
    assert runner.events == []


@pytest.mark.asyncio
async def test_duplicate_event_runs_once_both_ack() -> None:
    runner, inbox = _FakeRunner(), _FakeInbox()
    sub = OEGradingCrewSubscriber(runner=runner, inbox=inbox)
    attrs = {"tenant_id": "t", "event_id": "dup-1"}

    m1 = _FakeMsg(data=_body(), attributes=attrs)
    m2 = _FakeMsg(data=_body(), attributes=attrs)
    await sub.handle_message(m1)
    await sub.handle_message(m2)

    assert m1.acked and m2.acked  # dedupe-hit still ACKs
    assert len(runner.events) == 1  # runner invoked once
