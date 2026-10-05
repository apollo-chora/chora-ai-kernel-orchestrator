"""DispatchDlqSubscriber: reaper arms (a) and (c) over the dispatch DLQ pull subs.

Every agent-dispatch lane dead-letters to
``chora.dlq.ai_kernel.agent_dispatch.{role}_{requested|completed}.v1`` after
five delivery attempts, and each DLQ topic carries a ``.pull`` subscription
(``agent_dispatch_lanes.tf``) that nothing consumed until now. The kennel
consumes them all (the orchestrator SA holds subscriber on every one, ADR-254
D3) through the same StreamingPull wrapper the completion lane uses, and
settles the parked run FAILED through the reaper.

Which arm a message belongs to is read from what Pub/Sub stamps on a forwarded
dead letter (``CloudPubSubDeadLetterSourceSubscription`` ends in ``-requested``
or ``-completed``; ``CloudPubSubDeadLetterSourceDeliveryCount`` is the attempt
count), with the producer's ``event_topic`` attribute as the fallback. Never
from the body's shape.

Ack policy, stated because it differs from the request lanes: a message that
cannot be classified or carries no request key is logged LOUD and ACKed. A DLQ
pull subscription has no onward DLQ, so a NACK would redeliver the same
unprocessable message forever. A reaper failure (ledger unreachable, runner
error) NACKs: that is transient and a retry is right.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.domain.agent_dispatch.reaper import ReaperArm

logger = logging.getLogger(__name__)

_DLQ_PREFIX = "chora.dlq.ai_kernel.agent_dispatch"
_PULL_SUFFIX = ".pull"

ATTR_SOURCE_SUBSCRIPTION = "CloudPubSubDeadLetterSourceSubscription"
ATTR_SOURCE_DELIVERY_COUNT = "CloudPubSubDeadLetterSourceDeliveryCount"
ATTR_EVENT_TOPIC = "event_topic"


def dlq_pull_subscription_names(agent_roles: Iterable[str]) -> list[str]:
    """The ``.pull`` subscription on each lane's request AND completion DLQ."""
    roles = [r.strip() for r in agent_roles if (r or "").strip()]
    if not roles:
        raise ValueError(
            "dlq_pull_subscription_names: at least one role is required; a reaper listening on nothing reaps nothing"
        )
    names: list[str] = []
    for role in roles:
        names.append(f"{_DLQ_PREFIX}.{role}_requested.v1{_PULL_SUFFIX}")
        names.append(f"{_DLQ_PREFIX}.{role}_completed.v1{_PULL_SUFFIX}")
    return names


class _ReaperLike(Protocol):
    async def settle_by_key(
        self,
        idempotency_key: str,
        *,
        arm: ReaperArm,
        reason: str,
        original_topic: str,
        delivery_attempt: int,
    ) -> str: ...


class _MessageLike(Protocol):
    data: bytes
    attributes: dict[str, str]

    def ack(self) -> None: ...
    def nack(self) -> None: ...


class DispatchDlqSubscriber:
    def __init__(self, *, reaper: _ReaperLike) -> None:
        if reaper is None:
            raise ValueError("DispatchDlqSubscriber: reaper is required")
        self._reaper = reaper

    async def handle_message(self, msg: _MessageLike) -> None:
        """Never raises: a raise tears down the StreamingPull for every lane
        sharing the client."""
        attrs = dict(msg.attributes or {})
        try:
            decoded = json.loads(msg.data.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("expected a JSON object body")
        except (ValueError, UnicodeDecodeError) as exc:
            logger.error(
                "dispatch_dlq.undecodable_acked",
                extra={"err": str(exc), "attributes": attrs},
            )
            msg.ack()
            return

        arm, original_topic = _classify(attrs)
        if arm is None:
            logger.error(
                "dispatch_dlq.unclassifiable_acked",
                extra={"attributes": attrs, "idempotency_key": decoded.get("idempotency_key", "")},
            )
            msg.ack()
            return

        key = str(decoded.get("idempotency_key") or "").strip()
        if not key:
            logger.error(
                "dispatch_dlq.missing_request_key_acked",
                extra={"arm": arm.value, "attributes": attrs},
            )
            msg.ack()
            return

        attempts = _int(attrs.get(ATTR_SOURCE_DELIVERY_COUNT))
        source_sub = attrs.get(ATTR_SOURCE_SUBSCRIPTION, "")
        side = "request" if arm is ReaperArm.REQUEST_DEAD_LETTERED else "completion"
        reason = f"{side} dead-lettered after {attempts} delivery attempts"
        if source_sub:
            reason += f" (source subscription {source_sub})"
        if arm is ReaperArm.COMPLETION_DEAD_LETTERED:
            status = str(attrs.get("status") or decoded.get("status") or "").strip()
            if status:
                reason += f"; completion status {status}"

        try:
            outcome = await self._reaper.settle_by_key(
                key,
                arm=arm,
                reason=reason,
                original_topic=original_topic,
                delivery_attempt=attempts,
            )
        except Exception:
            logger.exception(
                "dispatch_dlq.settle_failed_nacked",
                extra={"arm": arm.value, "idempotency_key": key},
            )
            msg.nack()
            return

        msg.ack()
        logger.info(
            "dispatch_dlq.acked",
            extra={
                "arm": arm.value,
                "idempotency_key": key,
                "outcome": outcome,
                "delivery_attempt": attempts,
                "original_topic": original_topic,
            },
        )


def _classify(attrs: dict[str, str]) -> tuple[ReaperArm | None, str]:
    source_sub = (attrs.get(ATTR_SOURCE_SUBSCRIPTION) or "").strip()
    topic = (attrs.get(ATTR_EVENT_TOPIC) or "").strip()
    original_topic = topic or source_sub
    if source_sub.endswith("-requested"):
        return ReaperArm.REQUEST_DEAD_LETTERED, original_topic
    if source_sub.endswith("-completed"):
        return ReaperArm.COMPLETION_DEAD_LETTERED, original_topic
    if topic.endswith("_requested.v1"):
        return ReaperArm.REQUEST_DEAD_LETTERED, original_topic
    if topic.endswith("_completed.v1"):
        return ReaperArm.COMPLETION_DEAD_LETTERED, original_topic
    return None, original_topic


def _int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "ATTR_EVENT_TOPIC",
    "ATTR_SOURCE_DELIVERY_COUNT",
    "ATTR_SOURCE_SUBSCRIPTION",
    "DispatchDlqSubscriber",
    "dlq_pull_subscription_names",
]
