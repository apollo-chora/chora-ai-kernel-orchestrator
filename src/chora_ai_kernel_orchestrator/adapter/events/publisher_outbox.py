"""TransactionalOutboxPublisher — B.6.2.a producer-side durable emission
for the AI Kernel LangGraph orchestrator.

Implements ``AIKernelEventPublisher`` by writing each emitted event to
``chora_ai_kernel.ai_kernel_outbox_events`` (the table created by
migration ``0003_outbox.sql``). A background dispatcher (see
``adapter/pubsub/dispatcher.py``) polls ``status='pending'`` rows +
publishes to NATS JetStream + marks rows ``status='published'``.

Per ``feedback_d6_resilience_first_class``: durable emission survives
orchestrator pod-death. Composes with the LangGraph PostgresSaver
checkpoint write (same DB, same connection pool); a workflow node that
emits an event AND advances the workflow state will land BOTH writes in
``chora_ai_kernel`` before returning.

Per CLAUDE.md cross-cutting rule, each row carries the 11 mandatory
envelope fields. Per the D6.3 multi-tenant chaos directive (user
2026-05-12), ``tenant_id`` is a top-level column for queryable
isolation, not just an envelope field.

POC scope:
* Payload serialization is JSON (Protobuf swap at M14 per
  ``chora-contracts/proto/events/ai_kernel/*.proto``).
* Same-connection-as-PostgresSaver atomicity is NOT enforced at the
  adapter layer — the orchestrator node calls ``publisher.publish_*``
  then returns + LangGraph writes its checkpoint. Both writes hit the
  same Cloud SQL instance; an outbox-row-without-checkpoint orphan is
  acceptable (dispatcher will still publish; the workflow state is
  reconstructed from PostgresSaver on resume).
* idempotency_key is the event_id (UUIDv7). Workflow-resume re-emission
  is collapsed by the unique index on ``idempotency_key`` (POC:
  re-emission raises IntegrityError; M14: ON CONFLICT DO NOTHING wrap).
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

import uuid_utils as _uuid_utils

from chora_ai_kernel_orchestrator.adapter.events.payloads import (
    AgentTerminated,
    CrewComposed,
    CrewExecuted,
    GuardrailEvaluated,
    ModelInvocationCompleted,
    ModelInvocationFailed,
    ModelInvoked,
)

SCHEMA_VERSION = "1"


_INSERT_SQL = """
INSERT INTO ai_kernel_outbox_events (
    id, workflow_id, tenant_id, gcid, event_type, topic,
    payload, envelope, idempotency_key, occurred_at, status
) VALUES (
    %(id)s, %(workflow_id)s, %(tenant_id)s, %(gcid)s, %(event_type)s, %(topic)s,
    %(payload)s, %(envelope)s, %(idempotency_key)s, %(occurred_at)s, %(status)s
)
""".strip()


class TransactionalOutboxPublisher:
    """Durable-emission ``AIKernelEventPublisher`` backed by
    ``ai_kernel_outbox_events`` in ``chora_ai_kernel``.

    The dispatcher (separate concern) reads pending rows + publishes.
    """

    def __init__(
        self,
        *,
        conn: Any,
        source_project: str,
        source_service: str,
    ) -> None:
        if not source_project:
            raise ValueError("source_project required")
        if not source_service:
            raise ValueError("source_service required")
        self._conn = conn
        self._source_project = source_project
        self._source_service = source_service

    # --------------------------------------------------------------
    # AIKernelEventPublisher protocol
    # --------------------------------------------------------------

    async def publish_model_invoked(self, e: ModelInvoked) -> None:
        await self._write(
            event_type="ai_kernel.invocation.invoked",
            topic="chora.ai_kernel.invocation.invoked.v1",
            workflow_id=e.workflow_id,
            tenant_id=e.tenant_id,
            gcid=e.gcid,
            occurred_at=e.invoked_at,
            traceparent=e.traceparent,
            tracestate=e.tracestate,
            body={
                "invocation_id": e.invocation_id,
                "workflow_id": e.workflow_id,
                "tenant_id": e.tenant_id,
                "gcid": e.gcid,
                "agid": e.agid,
                "model_id": e.model_id,
                "model_kind": e.model_kind,
                "prompt_hash": e.prompt_hash,
                "invoked_at": e.invoked_at.isoformat(),
            },
        )

    async def publish_model_invocation_completed(self, e: ModelInvocationCompleted) -> None:
        await self._write(
            event_type="ai_kernel.invocation.completed",
            topic="chora.ai_kernel.invocation.completed.v1",
            workflow_id=e.workflow_id,
            tenant_id=e.tenant_id,
            gcid=e.gcid,
            occurred_at=e.completed_at,
            traceparent=e.traceparent,
            tracestate=e.tracestate,
            body={
                "invocation_id": e.invocation_id,
                "workflow_id": e.workflow_id,
                "tenant_id": e.tenant_id,
                "gcid": e.gcid,
                "agid": e.agid,
                "model_id": e.model_id,
                "model_kind": e.model_kind,
                "prompt_hash": e.prompt_hash,
                "response_hash": e.response_hash,
                "latency_ms": e.latency_ms,
                "tokens_in": e.tokens_in,
                "tokens_out": e.tokens_out,
                "completed_at": e.completed_at.isoformat(),
            },
        )

    async def publish_model_invocation_failed(self, e: ModelInvocationFailed) -> None:
        await self._write(
            event_type="ai_kernel.invocation.failed",
            topic="chora.ai_kernel.invocation.failed.v1",
            workflow_id=e.workflow_id,
            tenant_id=e.tenant_id,
            gcid=e.gcid,
            occurred_at=e.failed_at,
            traceparent=e.traceparent,
            tracestate=e.tracestate,
            body={
                "invocation_id": e.invocation_id,
                "workflow_id": e.workflow_id,
                "tenant_id": e.tenant_id,
                "gcid": e.gcid,
                "agid": e.agid,
                "model_id": e.model_id,
                "model_kind": e.model_kind,
                "prompt_hash": e.prompt_hash,
                "error_code": e.error_code,
                "error_detail": e.error_detail,
                "latency_ms": e.latency_ms,
                "failed_at": e.failed_at.isoformat(),
            },
        )

    async def publish_crew_composed(self, e: CrewComposed) -> None:
        await self._write(
            event_type="ai_kernel.crew.composed",
            topic="chora.ai_kernel.crew.composed.v1",
            workflow_id=e.workflow_id,
            tenant_id=e.tenant_id,
            gcid=e.gcid,
            occurred_at=e.composed_at,
            traceparent=e.traceparent,
            tracestate=e.tracestate,
            body={
                "crew_id": e.crew_id,
                "workflow_id": e.workflow_id,
                "tenant_id": e.tenant_id,
                "gcid": e.gcid,
                "crew_kind": e.crew_kind,
                "member_agids": list(e.member_agids),
                "composed_at": e.composed_at.isoformat(),
            },
        )

    async def publish_crew_executed(self, e: CrewExecuted) -> None:
        await self._write(
            event_type="ai_kernel.crew.executed",
            topic="chora.ai_kernel.crew.executed.v1",
            workflow_id=e.workflow_id,
            tenant_id=e.tenant_id,
            gcid=e.gcid,
            occurred_at=e.executed_at,
            traceparent=e.traceparent,
            tracestate=e.tracestate,
            body={
                "crew_id": e.crew_id,
                "workflow_id": e.workflow_id,
                "tenant_id": e.tenant_id,
                "gcid": e.gcid,
                "crew_kind": e.crew_kind,
                "duration_ms": e.duration_ms,
                "outcome": e.outcome,
                "member_outcomes": dict(e.member_outcomes),
                "executed_at": e.executed_at.isoformat(),
            },
        )

    async def publish_guardrail_evaluated(self, e: GuardrailEvaluated) -> None:
        await self._write(
            event_type="ai_kernel.guardrail.evaluated",
            topic="chora.ai_kernel.guardrail.evaluated.v1",
            workflow_id=e.workflow_id,
            tenant_id=e.tenant_id,
            gcid=e.gcid,
            occurred_at=e.evaluated_at,
            traceparent=e.traceparent,
            tracestate=e.tracestate,
            body={
                "invocation_id": e.invocation_id,
                "workflow_id": e.workflow_id,
                "tenant_id": e.tenant_id,
                "gcid": e.gcid,
                "agid": e.agid,
                "tier": e.tier,
                "outcome": e.outcome,
                "detail": e.detail,
                "evaluated_at": e.evaluated_at.isoformat(),
            },
        )

    async def publish_agent_terminated(self, e: AgentTerminated) -> None:
        """Emit ``chora.ai_kernel.agent.terminated.v1`` via the outbox.

        Reuses ``workflow_id`` column as the execution-correlation key
        (LangGraph thread_id / ADK session_id). The dispatcher routes by
        ``topic``; subscribers correlate by ``workflow_id`` and the proto's
        ``execution_id``.
        """
        await self._write(
            event_type="ai_kernel.agent.terminated",
            topic="chora.ai_kernel.agent.terminated.v1",
            workflow_id=e.execution_id,
            tenant_id=e.tenant_id,
            gcid=e.gcid,
            occurred_at=e.terminated_at,
            traceparent=e.traceparent,
            tracestate=e.tracestate,
            body={
                "agent_id": e.agent_id,
                "agent_agid": e.agent_agid,
                "execution_id": e.execution_id,
                "runtime": e.runtime,
                "termination_code": e.termination_code,
                "crew_id": e.crew_id,
                "crew_pattern": e.crew_pattern,
                "context": {
                    "last_state_node": e.last_state_node,
                    "last_tool_name": e.last_tool_name,
                    "last_error_message": e.last_error_message,
                    "iteration_count": e.iteration_count,
                    "partial_state": dict(e.partial_state),
                    "current_span_id": e.current_span_id,
                },
                "terminated_at": e.terminated_at.isoformat(),
            },
        )

    # --------------------------------------------------------------
    # Internal write
    # --------------------------------------------------------------

    async def _write(
        self,
        *,
        event_type: str,
        topic: str,
        workflow_id: str,
        tenant_id: str,
        gcid: str,
        occurred_at: _dt.datetime,
        body: dict[str, Any],
        traceparent: str = "",
        tracestate: str = "",
    ) -> None:
        event_id = str(_uuid_utils.uuid7())
        envelope = {
            "event_id": event_id,
            "idempotency_key": event_id,  # POC: same as event_id; future: caller-supplied
            "tenant_id": tenant_id,
            "gcid": gcid,
            "occurred_at": occurred_at.isoformat(),
            "traceparent": traceparent,
            "tracestate": tracestate,
            "source_project": self._source_project,
            "source_service": self._source_service,
            "schema_version": SCHEMA_VERSION,
        }
        payload = json.dumps(body, default=str).encode("utf-8")
        async with self._conn.cursor() as cur:
            await cur.execute(
                _INSERT_SQL,
                {
                    "id": event_id,
                    "workflow_id": workflow_id,
                    "tenant_id": tenant_id,
                    "gcid": gcid,
                    "event_type": event_type,
                    "topic": topic,
                    "payload": payload,
                    "envelope": json.dumps(envelope),
                    "idempotency_key": event_id,
                    "occurred_at": occurred_at,
                    "status": "pending",
                },
            )


__all__ = ["TransactionalOutboxPublisher", "SCHEMA_VERSION"]
