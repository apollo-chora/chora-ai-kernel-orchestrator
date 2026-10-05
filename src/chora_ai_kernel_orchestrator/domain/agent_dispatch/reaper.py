"""Park reaper rules: ADR-254 D5, pure.

Four arms settle a park the agent never answered:

  request_dead_lettered     the request landed in its DLQ (agent NACKed it out)
  request_expired           the deadline passed (retention by default: the
                            request is certainly gone from the subscription)
  completion_dead_lettered  the agent answered but the kennel could not consume
                            the completion and it dead-lettered
  outbox_dead_lettered      the request never left the outbox (publish failed
                            past the retry budget)

Every arm does the same two things: resume the parked thread with a FAILED
completion through the SAME router a real completion takes, so the crew's own
failure handling runs and the caller gets a status instead of silence; and
publish ``chora.ai_kernel.crew.run_failed.v1`` (the REUSED D6 P2 layer-4
terminal, ADR-254 D4) with the ``arm`` discriminator. This module builds both
payloads; the adapters move them.
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import Any

import uuid_utils as _uuid_utils

from chora_ai_kernel_orchestrator.domain.agent_dispatch.park import ParkRecord

RUN_FAILED_TOPIC = "chora.ai_kernel.crew.run_failed.v1"
RUN_FAILED_EVENT_TYPE = "ai_kernel.crew.run_failed"
SCHEMA_VERSION = 1
SOURCE_SERVICE = "chora-ai-kernel-orchestrator"
# IMDA D3 (ADR-141 label): a reaped run is a safety-and-robustness signal, the
# monitoring record that a workload did not complete.
IMDA_DIMENSION_SAFETY_AND_ROBUSTNESS = "safety_and_robustness"

STATUS_FAILED = "FAILED"


class ReaperArm(StrEnum):
    REQUEST_DEAD_LETTERED = "request_dead_lettered"
    REQUEST_EXPIRED = "request_expired"
    COMPLETION_DEAD_LETTERED = "completion_dead_lettered"
    OUTBOX_DEAD_LETTERED = "outbox_dead_lettered"


def run_failed_idempotency_key(arm: ReaperArm, request_key: str) -> str:
    return f"crew.run_failed.{ReaperArm(arm).value}.{_require(request_key, 'request_key')}"


def reaped_inbox_key(arm: ReaperArm, request_key: str) -> str:
    """The inbox dedupe key for one reap. Distinct from the request key AND
    from the real completion's ``<key>.completed``: all three land in the one
    ``idempotency_keys`` table, so any overlap would swallow one of them."""
    return f"{_require(request_key, 'request_key')}.reaped.{ReaperArm(arm).value}"


def reaped_completion(
    park: ParkRecord,
    *,
    arm: ReaperArm,
    reason: str,
    reaped_at: _dt.datetime,
) -> dict[str, Any]:
    """The FAILED completion that resumes the parked thread.

    Shaped like an agent completion (ADR-254 D6) so ``PubSubAgentExecutor``'s
    ``_map_completion`` raises ``AgentDispatchError`` in the node exactly as it
    would for an agent-reported failure. ``output_payload`` is empty on purpose:
    a reaped run carries nothing an agent produced.
    """
    arm = ReaperArm(arm)
    text = _require(reason, "reason")
    if not park.is_parked:
        raise ValueError(
            f"refusing to synthesize a completion for a park in state "
            f"{park.state.value!r} ({park.idempotency_key}); only a parked thread resumes"
        )
    _require_aware(reaped_at, "reaped_at")
    return {
        # ADR-254 D5: a shared role routes by crew (the park row knows it).
        "crew": park.crew,
        "agent_role": park.agent_role,
        "execution_id": park.execution_id,
        "thread_id": park.thread_id,
        "workflow_id": park.workflow_id,
        "idempotency_key": park.idempotency_key,
        "tenant_id": park.tenant_id,
        "gcid": park.gcid,
        "status": STATUS_FAILED,
        "output_payload": "",
        "error_message": f"{arm.value}: {text}",
        "reaper_arm": arm.value,
        "traceparent": park.traceparent,
        "tracestate": park.tracestate,
        "completed_at": reaped_at.isoformat(),
    }


def run_failed_event(
    park: ParkRecord,
    *,
    arm: ReaperArm,
    reason: str,
    original_topic: str,
    delivery_attempt: int,
    reaped_at: _dt.datetime,
    source_project: str,
    source_service: str = SOURCE_SERVICE,
) -> dict[str, Any]:
    """Build the ``run_failed.v1`` outbox row: mandatory envelope + body + routing.

    Returns the same shape ``agent_dispatch.build_dispatch_request`` does so
    the same outbox writer pattern carries it.
    """
    arm = ReaperArm(arm)
    text = _require(reason, "reason")
    project = _require(source_project, "source_project")
    service = _require(source_service, "source_service")
    _require_aware(reaped_at, "reaped_at")
    idempotency_key = run_failed_idempotency_key(arm, park.idempotency_key)
    occurred = reaped_at.isoformat()
    envelope: dict[str, Any] = {
        "event_id": str(_uuid_utils.uuid7()),
        "idempotency_key": idempotency_key,
        "tenant_id": park.tenant_id,
        "gcid": park.gcid,
        "occurred_at": occurred,
        "published_at": occurred,
        "traceparent": park.traceparent,
        "tracestate": park.tracestate,
        "source_project": project,
        "source_service": service,
        "schema_version": SCHEMA_VERSION,
        "chora_imda_dimension": IMDA_DIMENSION_SAFETY_AND_ROBUSTNESS,
        # The publisher client reserves the attribute name "topic".
        "event_topic": RUN_FAILED_TOPIC,
    }
    body: dict[str, Any] = {
        "workflow_id": park.workflow_id,
        "thread_id": park.thread_id,
        "tenant_id": park.tenant_id,
        "gcid": park.gcid,
        "crew": park.crew,
        "agent_role": park.agent_role,
        "arm": arm.value,
        "original_topic": original_topic,
        "delivery_attempt": int(delivery_attempt),
        "dispatch_idempotency_key": park.idempotency_key,
        "parked_at": park.parked_at.isoformat(),
        "deadline_at": park.deadline_at.isoformat(),
        "reaped_at": occurred,
        "reason": text,
    }
    return {
        "topic": RUN_FAILED_TOPIC,
        "event_type": RUN_FAILED_EVENT_TYPE,
        "envelope": envelope,
        "body": body,
        "idempotency_key": idempotency_key,
        "tenant_id": park.tenant_id,
        "gcid": park.gcid,
        "workflow_id": park.workflow_id,
    }


def _require(value: str, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"park reaper: {field} is required")
    return text


def _require_aware(value: _dt.datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"park reaper: {field} must be timezone-aware")


__all__ = [
    "IMDA_DIMENSION_SAFETY_AND_ROBUSTNESS",
    "RUN_FAILED_EVENT_TYPE",
    "RUN_FAILED_TOPIC",
    "SCHEMA_VERSION",
    "STATUS_FAILED",
    "ReaperArm",
    "reaped_completion",
    "reaped_inbox_key",
    "run_failed_event",
    "run_failed_idempotency_key",
]
