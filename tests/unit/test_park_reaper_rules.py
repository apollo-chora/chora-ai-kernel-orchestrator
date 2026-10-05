"""RED: ADR-254 D5: the park reaper's pure rules.

Four arms, one terminal, one synthesized completion. Everything here is pure:
given a ``ParkRecord`` and a clock, produce (a) the FAILED completion that
resumes the parked thread through the SAME router a real completion takes, so
the crew's own failure handling runs and the caller receives a status rather
than silence, and (b) the ``chora.ai_kernel.crew.run_failed.v1`` event (REUSED
D6 P2 layer-4 name, ADR-254 D4) with the ``arm`` discriminator.
"""

from __future__ import annotations

import datetime as _dt
from uuid import UUID

import pytest

from chora_ai_kernel_orchestrator.domain.agent_dispatch.park import (
    ParkRecord,
    ParkState,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.reaper import (
    RUN_FAILED_EVENT_TYPE,
    RUN_FAILED_TOPIC,
    ReaperArm,
    reaped_completion,
    reaped_inbox_key,
    run_failed_event,
    run_failed_idempotency_key,
)

_NOW = _dt.datetime(2026, 8, 22, 15, 0, 0, tzinfo=_dt.UTC)
_KEY = "agent_dispatch.oe_evaluate.01a02062-e5b4-7870-8fca-53ce363cd542:tsq-1:1"


def _park(**over: object) -> ParkRecord:
    base: dict[str, object] = dict(
        idempotency_key=_KEY,
        workflow_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        thread_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        crew="oe_grading",
        tenant_id="11111111-1111-7111-8111-111111111111",
        gcid="00000000-0000-7000-8000-000000001999",
        agent_role="oe_evaluate",
        request_topic="chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
        completion_topic="chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1",
        traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        tracestate="",
        parked_at=_NOW - _dt.timedelta(hours=2),
        deadline_at=_NOW - _dt.timedelta(minutes=1),
        state=ParkState.PARKED,
    )
    base.update(over)
    return ParkRecord(**base)  # type: ignore[arg-type]


def test_arms_are_exactly_the_adr_254_values() -> None:
    assert {a.value for a in ReaperArm} == {
        "request_dead_lettered",
        "request_expired",
        "completion_dead_lettered",
        "outbox_dead_lettered",
    }


def test_the_terminal_is_the_reused_d6_name() -> None:
    assert RUN_FAILED_TOPIC == "chora.ai_kernel.crew.run_failed.v1"
    assert RUN_FAILED_EVENT_TYPE == "ai_kernel.crew.run_failed"


def test_keys_are_deterministic_per_arm_and_request() -> None:
    assert run_failed_idempotency_key(ReaperArm.REQUEST_EXPIRED, _KEY) == f"crew.run_failed.request_expired.{_KEY}"
    # The inbox key must differ from BOTH the request key and the real
    # completion's "<key>.completed", or the dedupe table swallows one of them.
    assert reaped_inbox_key(ReaperArm.REQUEST_EXPIRED, _KEY) == f"{_KEY}.reaped.request_expired"


def test_reaped_completion_is_a_failed_completion_for_the_parked_thread() -> None:
    park = _park()
    completion = reaped_completion(
        park,
        arm=ReaperArm.REQUEST_DEAD_LETTERED,
        reason="request dead-lettered after 5 delivery attempts",
        reaped_at=_NOW,
    )
    assert completion["status"] == "FAILED"
    assert completion["thread_id"] == park.thread_id
    assert completion["workflow_id"] == park.workflow_id
    assert completion["idempotency_key"] == park.idempotency_key
    assert completion["agent_role"] == park.agent_role
    assert completion["tenant_id"] == park.tenant_id
    assert completion["gcid"] == park.gcid
    assert completion["execution_id"] == "01a02062-e5b4-7870-8fca-53ce363cd542:tsq-1:1"
    assert completion["traceparent"] == park.traceparent
    assert completion["reaper_arm"] == "request_dead_lettered"
    assert "request_dead_lettered" in completion["error_message"]
    assert "after 5 delivery attempts" in completion["error_message"]
    assert completion["completed_at"] == _NOW.isoformat()
    # No fabricated output: a reaped run carries nothing an agent produced.
    assert completion["output_payload"] == ""


def test_reaped_completion_refuses_an_empty_reason() -> None:
    with pytest.raises(ValueError):
        reaped_completion(_park(), arm=ReaperArm.REQUEST_EXPIRED, reason="  ", reaped_at=_NOW)


@pytest.mark.parametrize("state", [ParkState.COMPLETED, ParkState.REAPED])
def test_reaped_completion_refuses_a_settled_park(state: ParkState) -> None:
    """Resuming a thread that already settled would double-settle the run."""
    with pytest.raises(ValueError):
        reaped_completion(_park(state=state), arm=ReaperArm.REQUEST_EXPIRED, reason="x", reaped_at=_NOW)


def test_run_failed_event_carries_the_mandatory_envelope_and_the_arm() -> None:
    park = _park()
    event = run_failed_event(
        park,
        arm=ReaperArm.REQUEST_DEAD_LETTERED,
        reason="request dead-lettered after 5 delivery attempts",
        original_topic=park.request_topic,
        delivery_attempt=5,
        reaped_at=_NOW,
        source_project="chora-489812",
    )
    assert event["topic"] == RUN_FAILED_TOPIC
    assert event["event_type"] == RUN_FAILED_EVENT_TYPE
    assert event["workflow_id"] == park.workflow_id
    assert event["tenant_id"] == park.tenant_id
    assert event["gcid"] == park.gcid
    assert event["idempotency_key"] == run_failed_idempotency_key(ReaperArm.REQUEST_DEAD_LETTERED, park.idempotency_key)

    envelope = event["envelope"]
    for field in (
        "event_id",
        "idempotency_key",
        "tenant_id",
        "gcid",
        "occurred_at",
        "published_at",
        "traceparent",
        "tracestate",
        "source_project",
        "source_service",
        "schema_version",
    ):
        assert field in envelope, field
    assert envelope["event_topic"] == RUN_FAILED_TOPIC
    assert envelope["source_project"] == "chora-489812"
    assert envelope["source_service"] == "chora-ai-kernel-orchestrator"
    assert envelope["traceparent"] == park.traceparent
    assert envelope["tenant_id"] == park.tenant_id
    assert envelope["occurred_at"] == _NOW.isoformat()
    assert UUID(envelope["event_id"]).version == 7

    body = event["body"]
    assert body["arm"] == "request_dead_lettered"
    assert body["workflow_id"] == park.workflow_id
    assert body["thread_id"] == park.thread_id
    assert body["tenant_id"] == park.tenant_id
    assert body["gcid"] == park.gcid
    assert body["agent_role"] == park.agent_role
    assert body["crew"] == park.crew
    assert body["original_topic"] == park.request_topic
    assert body["delivery_attempt"] == 5
    assert body["parked_at"] == park.parked_at.isoformat()
    assert body["deadline_at"] == park.deadline_at.isoformat()
    assert body["reaped_at"] == _NOW.isoformat()
    assert body["reason"] == "request dead-lettered after 5 delivery attempts"
    assert body["dispatch_idempotency_key"] == park.idempotency_key


def test_run_failed_event_refuses_a_blank_source_project() -> None:
    with pytest.raises(ValueError):
        run_failed_event(
            _park(),
            arm=ReaperArm.REQUEST_EXPIRED,
            reason="deadline passed",
            original_topic="t",
            delivery_attempt=0,
            reaped_at=_NOW,
            source_project="",
        )


def test_park_record_state_roundtrips_through_its_string_value() -> None:
    assert ParkState("parked") is ParkState.PARKED
    assert {s.value for s in ParkState} == {"parked", "completed", "reaped"}
    assert _park().is_parked is True
    assert _park(state=ParkState.REAPED).is_parked is False
