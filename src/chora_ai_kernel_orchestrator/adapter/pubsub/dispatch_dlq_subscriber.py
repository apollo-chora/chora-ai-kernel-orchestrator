"""DispatchDlqSubscriber: reaper arms (a) and (c) over the dispatch DLQ subjects.

Every agent-dispatch lane dead-letters to ``_dlq.chora.ai_kernel.agent_dispatch.
{role}_{requested|completed}.v1`` — the platform DLQ convention
(``chora-common/eventbus/bus.go`` ``DLQSubject``: the transport prefix ``_dlq.``
on the ORIGINAL subject), which the root Compose CHORA_DLQ stream captures via
``_dlq.>``. The kennel consumes them all (the orchestrator SA holds subscriber
on every one, ADR-254 D3) through the same pull-consumer wrapper the completion
lane uses, and settles the parked run FAILED through the reaper.

Which arm a message belongs to is read from the ``Chora-Dlq-*`` headers the Go
eventbus stamps on a dead letter (``Chora-Dlq-Source-Subject`` is the original
subject, ``Chora-Dlq-Delivery-Count`` the attempt count), with the Pub/Sub
attribute names kept as a migration fallback and the producer's ``event_topic``
attribute as the last resort. Never from the body's shape.

Ack policy, stated because it differs from the request lanes: a message that
cannot be classified or carries no request key is logged LOUD and ACKed. A DLQ
subject has no onward DLQ, so a NACK would redeliver the same unprocessable
message forever. A reaper failure (ledger unreachable, runner error) NACKs:
that is transient and a retry is right.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.domain.agent_dispatch.reaper import ReaperArm

logger = logging.getLogger(__name__)

#: The original (pre-DLQ) dispatch subject family. The DLQ subject is this
#: prefixed with ``_dlq.`` — see ``dlq_subject``.
_DISPATCH_SUBJECT_PREFIX = "chora.ai_kernel.agent_dispatch"

#: Transport prefix ``chora-common/eventbus/bus.go`` ``DLQSubject`` puts on the
#: original subject. An infrastructure address, not a domain event name.
_DLQ_SUBJECT_PREFIX = "_dlq."

# What the Go eventbus stamps on a dead letter (chora-common/eventbus/jetstream.go
# ``dlqHeaders``). These are the PRIMARY classify inputs: every NATS producer in
# the monorepo dead-letters through that path.
HDR_DLQ_SOURCE_SUBJECT = "Chora-Dlq-Source-Subject"
HDR_DLQ_DELIVERY_COUNT = "Chora-Dlq-Delivery-Count"

# Pub/Sub-era attribute names. Kept as a migration fallback only: nothing in the
# monorepo publishes to ``chora.dlq.*`` any more, so no NATS producer stamps them.
ATTR_SOURCE_SUBSCRIPTION = "CloudPubSubDeadLetterSourceSubscription"
ATTR_SOURCE_DELIVERY_COUNT = "CloudPubSubDeadLetterSourceDeliveryCount"
ATTR_EVENT_TOPIC = "event_topic"


def dlq_subject(agent_role: str, *, side: str) -> str:
    """The DLQ subject an agent role's ``side`` dispatch lane dead-letters to."""
    role = (agent_role or "").strip()
    if not role:
        raise ValueError("dlq_subject: agent_role required")
    return f"{_DLQ_SUBJECT_PREFIX}{_DISPATCH_SUBJECT_PREFIX}.{role}_{side}.v1"


def dlq_pull_subscription_names(agent_roles: Iterable[str]) -> list[str]:
    """The DLQ subject of each lane's request AND completion side.

    The ``.pull`` suffix is deliberately gone: these are NATS subjects, not
    Pub/Sub subscription ids, and the CHORA_DLQ stream captures ``_dlq.>``.
    """
    roles = [r.strip() for r in agent_roles if (r or "").strip()]
    if not roles:
        raise ValueError(
            "dlq_pull_subscription_names: at least one role is required; a reaper listening on nothing reaps nothing"
        )
    names: list[str] = []
    for role in roles:
        names.append(dlq_subject(role, side="requested"))
        names.append(dlq_subject(role, side="completed"))
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

        attempts = _delivery_count(attrs)
        source_sub = _source_subject(attrs)
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
    """Pick the reaper arm from the dead letter's provenance headers.

    ``Chora-Dlq-Source-Subject`` is the original subject the Go eventbus
    dead-lettered (``chora.ai_kernel.agent_dispatch.{role}_{side}.v1``), so it
    carries the arm directly. The Pub/Sub attribute and the producer's
    ``event_topic`` are fallbacks for a producer still on the old path.
    """
    source_sub = _source_subject(attrs)
    topic = (attrs.get(ATTR_EVENT_TOPIC) or "").strip()
    # ``event_topic`` is the producer's actual topic; the Pub/Sub subscription
    # id is not one, so it is the last resort for the reported original topic.
    original_topic = topic or source_sub
    for candidate in (source_sub, topic):
        if candidate.endswith("_requested.v1") or candidate.endswith("-requested"):
            return ReaperArm.REQUEST_DEAD_LETTERED, original_topic
        if candidate.endswith("_completed.v1") or candidate.endswith("-completed"):
            return ReaperArm.COMPLETION_DEAD_LETTERED, original_topic
    return None, original_topic


def _int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _source_subject(attrs: dict[str, str]) -> str:
    """The original subject, preferring the NATS ``Chora-Dlq-*`` header."""
    return (attrs.get(HDR_DLQ_SOURCE_SUBJECT) or attrs.get(ATTR_SOURCE_SUBSCRIPTION) or "").strip()


def _delivery_count(attrs: dict[str, str]) -> int:
    """How many deliveries the dead letter represents.

    ``Chora-Dlq-Delivery-Count`` is stamped by the Go eventbus and is the only
    source a NATS producer provides; the Pub/Sub attribute is the fallback.
    """
    return _int(attrs.get(HDR_DLQ_DELIVERY_COUNT) or attrs.get(ATTR_SOURCE_DELIVERY_COUNT))


__all__ = [
    "ATTR_EVENT_TOPIC",
    "ATTR_SOURCE_DELIVERY_COUNT",
    "ATTR_SOURCE_SUBSCRIPTION",
    "HDR_DLQ_DELIVERY_COUNT",
    "HDR_DLQ_SOURCE_SUBJECT",
    "DispatchDlqSubscriber",
    "dlq_pull_subscription_names",
    "dlq_subject",
]
