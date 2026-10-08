"""RED: ADR-254 D5, reaper arms (a) and (c) consume the dispatch DLQ subjects.

Every agent-dispatch lane dead-letters to ``_dlq.chora.ai_kernel.agent_dispatch.
{role}_{requested|completed}.v1`` — the platform DLQ convention
(``chora-common/eventbus/bus.go`` ``DLQSubject``), captured by the root Compose
CHORA_DLQ stream's ``_dlq.>`` filter. This subscriber reads them from the
orchestrator and settles the parked run FAILED through the reaper.

Which arm a message belongs to is read from the ``Chora-Dlq-*`` headers the Go
eventbus stamps on a dead letter (``Chora-Dlq-Source-Subject`` is the original
subject, ``Chora-Dlq-Delivery-Count`` the attempt count), with the Pub/Sub
attribute names kept as a migration fallback and the request/completion
``event_topic`` attribute as the last resort; never from the body's shape. A
message that carries none is logged loud and ACKed: a DLQ subject has no onward
DLQ, so a NACK would loop forever.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.dispatch_dlq_subscriber import (
    DispatchDlqSubscriber,
    dlq_pull_subscription_names,
    dlq_subject,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.reaper import ReaperArm

_KEY = "agent_dispatch.oe_evaluate.sub-1:q1:1"


class _FakeReaper:
    def __init__(self, outcome: str = "reaped", raise_exc: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._outcome = outcome
        self._raise = raise_exc

    async def settle_by_key(
        self,
        key: str,
        *,
        arm: ReaperArm,
        reason: str,
        original_topic: str,
        delivery_attempt: int,
    ) -> str:
        if self._raise is not None:
            raise self._raise
        self.calls.append(
            {
                "key": key,
                "arm": arm,
                "reason": reason,
                "original_topic": original_topic,
                "delivery_attempt": delivery_attempt,
            }
        )
        return self._outcome


class _Msg:
    def __init__(self, body: Any, attributes: dict[str, str]) -> None:
        self.data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.attributes = attributes
        self.acked = 0
        self.nacked = 0

    def ack(self) -> None:
        self.acked += 1

    def nack(self) -> None:
        self.nacked += 1


_REQUEST_BODY = {"agent_role": "oe_evaluate", "idempotency_key": _KEY, "thread_id": "t", "reply_topic": "x"}
_COMPLETION_BODY = {"agent_role": "oe_evaluate", "idempotency_key": _KEY, "thread_id": "t", "status": "OK"}


def test_pull_subscription_names_use_the_platform_dlq_convention_for_both_sides() -> None:
    """``_dlq.<original subject>`` per chora-common/eventbus/bus.go DLQSubject —
    the shape the root Compose CHORA_DLQ stream captures via ``_dlq.>``. The
    legacy ``chora.dlq.*`` + ``.pull`` form is a Pub/Sub subscription id, not a
    NATS subject, so a consumer bound to it waits forever."""
    names = dlq_pull_subscription_names(["oe_evaluate", "companion_chat"])
    assert names == [
        "_dlq.chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
        "_dlq.chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1",
        "_dlq.chora.ai_kernel.agent_dispatch.companion_chat_requested.v1",
        "_dlq.chora.ai_kernel.agent_dispatch.companion_chat_completed.v1",
    ]
    with pytest.raises(ValueError):
        dlq_pull_subscription_names([])


def test_dlq_subject_prefixes_the_original_subject_only() -> None:
    assert dlq_subject("oe_evaluate", side="requested") == (
        "_dlq.chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1"
    )
    assert dlq_subject("oe_evaluate", side="completed") == (
        "_dlq.chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1"
    )
    with pytest.raises(ValueError):
        dlq_subject("  ", side="requested")


@pytest.mark.asyncio
async def test_a_dead_lettered_request_is_arm_a() -> None:
    reaper = _FakeReaper()
    msg = _Msg(
        _REQUEST_BODY,
        {
            "Chora-Dlq-Source-Subject": "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
            "Chora-Dlq-Delivery-Count": "5",
            "Chora-Dlq-Consumer": "chora-oe-evaluator.agent-dispatch-oe-evaluate-requested",
            "Chora-Dlq-Reason": "context deadline exceeded",
        },
    )
    await DispatchDlqSubscriber(reaper=reaper).handle_message(msg)
    assert msg.acked == 1 and msg.nacked == 0
    call = reaper.calls[0]
    assert call["key"] == _KEY
    assert call["arm"] is ReaperArm.REQUEST_DEAD_LETTERED
    assert call["delivery_attempt"] == 5
    assert call["original_topic"] == "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1"
    assert "5" in call["reason"]


@pytest.mark.asyncio
async def test_a_dead_lettered_completion_is_arm_c() -> None:
    reaper = _FakeReaper()
    msg = _Msg(
        _COMPLETION_BODY,
        {
            "Chora-Dlq-Source-Subject": "chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1",
            "Chora-Dlq-Delivery-Count": "5",
            "status": "OK",
        },
    )
    await DispatchDlqSubscriber(reaper=reaper).handle_message(msg)
    assert msg.acked == 1
    call = reaper.calls[0]
    assert call["arm"] is ReaperArm.COMPLETION_DEAD_LETTERED
    assert call["original_topic"] == "chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1"
    assert "OK" in call["reason"]


@pytest.mark.asyncio
async def test_pubsub_attributes_still_classify_when_the_nats_headers_are_absent() -> None:
    """Migration fallback: a producer still on the Pub/Sub path stamps the
    CloudPubSubDeadLetter* attributes instead of the Chora-Dlq-* headers."""
    reaper = _FakeReaper()
    msg = _Msg(
        _REQUEST_BODY,
        {
            "CloudPubSubDeadLetterSourceSubscription": "chora-oe-evaluator.agent-dispatch-oe-evaluate-requested",
            "CloudPubSubDeadLetterSourceDeliveryCount": "5",
            "event_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
        },
    )
    await DispatchDlqSubscriber(reaper=reaper).handle_message(msg)
    assert msg.acked == 1
    call = reaper.calls[0]
    assert call["arm"] is ReaperArm.REQUEST_DEAD_LETTERED
    assert call["delivery_attempt"] == 5
    assert call["original_topic"] == "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1"
    assert "chora-oe-evaluator.agent-dispatch-oe-evaluate-requested" in call["reason"]


@pytest.mark.asyncio
async def test_a_missing_delivery_count_reads_as_zero_not_as_an_error() -> None:
    reaper = _FakeReaper()
    msg = _Msg(
        _REQUEST_BODY,
        {"Chora-Dlq-Source-Subject": "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1"},
    )
    await DispatchDlqSubscriber(reaper=reaper).handle_message(msg)
    assert msg.acked == 1
    assert reaper.calls[0]["delivery_attempt"] == 0


@pytest.mark.asyncio
async def test_event_topic_is_the_fallback_when_pubsub_stamps_are_absent() -> None:
    reaper = _FakeReaper()
    msg = _Msg(_REQUEST_BODY, {"event_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1"})
    await DispatchDlqSubscriber(reaper=reaper).handle_message(msg)
    assert msg.acked == 1
    assert reaper.calls[0]["arm"] is ReaperArm.REQUEST_DEAD_LETTERED
    assert reaper.calls[0]["delivery_attempt"] == 0


@pytest.mark.asyncio
async def test_a_message_with_no_arm_evidence_is_acked_loudly_not_looped() -> None:
    reaper = _FakeReaper()
    msg = _Msg(_REQUEST_BODY, {})
    await DispatchDlqSubscriber(reaper=reaper).handle_message(msg)
    assert msg.acked == 1 and msg.nacked == 0
    assert reaper.calls == []


@pytest.mark.asyncio
async def test_a_body_without_a_request_key_is_acked_loudly() -> None:
    reaper = _FakeReaper()
    msg = _Msg(
        {"agent_role": "oe_evaluate"},
        {"event_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1"},
    )
    await DispatchDlqSubscriber(reaper=reaper).handle_message(msg)
    assert msg.acked == 1
    assert reaper.calls == []


@pytest.mark.asyncio
async def test_undecodable_data_is_acked_loudly() -> None:
    reaper = _FakeReaper()
    msg = _Msg(b"\xff\xfe not json", {"event_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1"})
    await DispatchDlqSubscriber(reaper=reaper).handle_message(msg)
    assert msg.acked == 1
    assert reaper.calls == []


@pytest.mark.asyncio
async def test_a_reaper_failure_nacks_for_redelivery_and_never_raises() -> None:
    reaper = _FakeReaper(raise_exc=RuntimeError("db down"))
    msg = _Msg(_REQUEST_BODY, {"event_topic": "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1"})
    await DispatchDlqSubscriber(reaper=reaper).handle_message(msg)
    assert msg.nacked == 1 and msg.acked == 0
