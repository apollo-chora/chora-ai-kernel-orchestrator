"""PromptActivationAuditEmitter — transactional-outbox emitter for the
prompt-override ACTIVATION audit (ADR-197 M-C.1, IMDA D1 accountability).

Mirrors :class:`AgentDecisionLogOutboxWriter`: INSERTs an event row into
``ai_kernel_outbox_events`` (migration ``0003_outbox.sql``) on the canonical
governance topic ``chora.governance.audit.recorded.v1`` with a binary-proto
``chora.governance.v1.AuditEntryRecorded`` payload. The ``OutboxDispatcher``
(separate asyncio task) drains the table + publishes to NATS JetStream — this
emitter does NOT publish directly.

Why outbox + binary proto (same reasoning as the agent_decision writer):

* **Audit-first accountability (ADR-197 §Decision IMDA D1)** — the
  ``PromptPromotionService`` calls this BEFORE the status write. A raise here
  aborts activation, so an activation can never happen un-audited.
* **Pod-death survival / exactly-once** — the audit row + the activation share
  ``chora_ai_kernel``; a deterministic ``idempotency_key`` +
  ``ON CONFLICT (idempotency_key) DO NOTHING`` make the audit exactly-once
  across retries / checkpoint resumes.
* **BINARY schema** — the topic is Schema-Registry-bound (encoding=BINARY); JSON
  publishes are rejected at the publish hop. ``encode_audit_entry_recorded``
  emits canonical proto3 wire bytes the chora-governance consumer decodes.

The emitter does NOT own the connection lifecycle and does NOT commit — the
caller wraps the audit-emit + activation in its own unit (same convention as the
sibling outbox writers).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from typing import Any
from uuid import uuid4

import uuid_utils as _uuid_utils

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    AUDIT_RESULT_ALLOWED,
    encode_audit_entry_recorded,
)

logger = logging.getLogger(__name__)

# Canonical governance audit topic (per chora-contracts §2 topic taxonomy +
# proto/events/governance/audit.proto). Tested as a constant so a copy-paste
# swap is caught at unit time.
TOPIC_AUDIT_RECORDED = "chora.governance.audit.recorded.v1"

# Event type — matches the topic's domain.aggregate.event_type segments.
_EVENT_TYPE = "governance.audit.recorded"

# IMDA dimension per ADR-141 — an activation audit is D1 accountability
# evidence. The chora-governance projector routes by this attribute.
IMDA_DIM_ACCOUNTABILITY = "accountability"

# Schema version — chora-contracts/proto/events/governance/audit.proto v1.
SCHEMA_VERSION = "1"

# UPPER_SNAKE_CASE action verb per audit.proto §action.
_ACTION_ACTIVATE = "ACTIVATE"

# Resource URI prefix for a prompt-override plan (audit.proto §target_resource_uri
# convention "chora.<domain>/<aggregate>:<id>").
_RESOURCE_PREFIX = "chora.ai_kernel/prompt_plan:"

# Tenant a PLATFORM-scope activation is audited under. A platform plan has no
# tenant of its own, but ai_kernel_outbox_events.tenant_id is UUID NOT NULL, so
# the audit must name one; the platform tenant is where the O+ operator reviews
# these gates. Overridable per deployment.
ENV_PLATFORM_TENANT_ID = "CHORA_PLATFORM_TENANT_ID"
PLATFORM_TENANT_ID = "00000000-0000-7000-8000-000000000001"

_INSERT_SQL = """
INSERT INTO ai_kernel_outbox_events (
    id, workflow_id, tenant_id, gcid, event_type, topic,
    payload, envelope, idempotency_key, occurred_at, status
) VALUES (
    %(id)s, %(workflow_id)s, %(tenant_id)s, %(gcid)s, %(event_type)s, %(topic)s,
    %(payload)s, %(envelope)s, %(idempotency_key)s, %(occurred_at)s, 'pending'
)
ON CONFLICT (idempotency_key) DO NOTHING
""".strip()


class PromptActivationAuditEmitter:
    """Writes the prompt-plan ACTIVATE audit to ``ai_kernel_outbox_events``.

    Implements the ``ActivationAuditEmitter`` Protocol the
    ``PromptPromotionService`` depends on.
    """

    def __init__(
        self,
        *,
        conn: Any,
        source_project: str,
        source_service: str = "chora-ai-kernel-orchestrator",
        platform_tenant_id: str | None = None,
    ) -> None:
        self._platform_tenant = (platform_tenant_id or os.getenv(ENV_PLATFORM_TENANT_ID) or PLATFORM_TENANT_ID).strip()
        if not (source_project or "").strip():
            raise ValueError("source_project required")
        if not (source_service or "").strip():
            raise ValueError("source_service required")
        self._conn = conn
        self._source_project = source_project
        self._source_service = source_service

    @property
    def source_project(self) -> str:
        return self._source_project

    @property
    def source_service(self) -> str:
        return self._source_service

    async def emit_activation(
        self,
        *,
        plan_id: str,
        actor_gcid: str,
        tenant_id: str,
        scope: str = "",
        annotation: str = "",
        traceparent: str = "",
        tracestate: str = "",
    ) -> str:
        """Persist a ``chora.governance.audit.recorded.v1`` row for the
        activation of ``plan_id`` by ``actor_gcid``.

        Returns the outbox row id (UUIDv4 — distinct from the deterministic
        idempotency_key). Idempotency key:
        ``prompt_activation.{plan_id}.{actor_gcid}`` — one audit per
        (plan, actor); a retry / checkpoint resume re-INSERTs the same key,
        which the unique index swallows.
        """
        plan = (plan_id or "").strip()
        if not plan:
            raise ValueError("emit_activation requires a plan_id")
        actor = (actor_gcid or "").strip()
        if not actor:
            raise ValueError("emit_activation requires an actor_gcid")
        # A PLATFORM-scope plan carries no tenant, but
        # ai_kernel_outbox_events.tenant_id is UUID NOT NULL: emitting "" raises
        # 22P02, poisons the consumer's transaction, and every approval NACKs
        # (CHO-2368 P2, found on the first live walk). Platform activations are
        # therefore audited under the platform tenant.
        tenant = (tenant_id or "").strip() or self._platform_tenant

        occurred_at_dt = _dt.datetime.now(tz=_dt.UTC)
        occurred_at = occurred_at_dt.isoformat()
        published_at = occurred_at  # system-emitted: same instant

        row_id = str(uuid4())
        event_id = str(_uuid_utils.uuid7())
        audit_id = str(_uuid_utils.uuid7())
        idempotency_key = f"prompt_activation.{plan}.{actor}"

        envelope: dict[str, str] = {
            "event_id": event_id,
            "idempotency_key": idempotency_key,
            "tenant_id": tenant,
            "gcid": actor,
            "occurred_at": occurred_at,
            "published_at": published_at,
            "traceparent": traceparent,
            "tracestate": tracestate,
            "source_project": self._source_project,
            "source_service": self._source_service,
            "schema_version": SCHEMA_VERSION,
            "chora_imda_dimension": IMDA_DIM_ACCOUNTABILITY,
            # Extra index attrs — the action + scope ride through to Pub/Sub
            # message attributes so governance subscribers can route without
            # decoding the payload.
            "action": _ACTION_ACTIVATE,
            "scope": scope,
        }

        body: dict[str, Any] = {
            "audit_id": audit_id,
            "actor_gcid": actor,
            "target_resource_uri": f"{_RESOURCE_PREFIX}{plan}",
            "action": _ACTION_ACTIVATE,
            "result": AUDIT_RESULT_ALLOWED,
            "annotation": annotation,
            "occurred_at": occurred_at,
        }

        payload = encode_audit_entry_recorded(envelope, body)

        params = {
            "id": row_id,
            "workflow_id": plan,
            "tenant_id": tenant,
            "gcid": actor,
            "event_type": _EVENT_TYPE,
            "topic": TOPIC_AUDIT_RECORDED,
            "payload": payload,
            "envelope": json.dumps(envelope),
            "idempotency_key": idempotency_key,
            "occurred_at": occurred_at_dt,
        }

        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        logger.info(
            "prompt_activation_audit.queued",
            extra={
                "row_id": row_id,
                "plan_id": plan,
                "actor_gcid": actor,
                "tenant_id": tenant,
                "topic": TOPIC_AUDIT_RECORDED,
                "idempotency_key": idempotency_key,
            },
        )
        return row_id


__all__ = [
    "IMDA_DIM_ACCOUNTABILITY",
    "PromptActivationAuditEmitter",
    "SCHEMA_VERSION",
    "TOPIC_AUDIT_RECORDED",
]
