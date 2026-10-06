"""AgentDispatchOutboxWriter — queues an agent dispatch request (ADR-253 D3a).

Writes one ``ai_kernel_outbox_events`` row per dispatch; the existing
``OutboxDispatcher`` drains it to the role's request topic. Deliberately the
SAME table and the SAME dispatcher as every other event on this platform, so
there is one publishing path rather than two, and the invocation leaves a
durable pre-publish record (ADR-253 1.3.3) instead of existing only as a trace
span.

⚠ This writer does NOT own its transaction and does NOT commit. It is called
from inside ``TransactionalDispatchSaver``'s transaction, on that saver's
connection, so the row and the LangGraph park commit together or neither does.
Calling it outside such a transaction on an autocommit=False connection leaves
the row uncommitted and the dispatcher will never see it.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_INSERT_SQL = """
INSERT INTO ai_kernel_outbox_events (
    id, workflow_id, tenant_id, gcid, event_type, topic,
    payload, envelope, idempotency_key, occurred_at, status
) VALUES (
    %(id)s, %(workflow_id)s, %(tenant_id)s, %(gcid)s, %(event_type)s, %(topic)s,
    %(payload)s, %(envelope)s, %(idempotency_key)s, %(occurred_at)s, 'pending'
)
ON CONFLICT (idempotency_key) DO NOTHING
"""


def _synthetic_traceparent(event_id: str) -> str:
    """A valid W3C traceparent derived from an event id, for a dispatch this
    service generates rather than receives (mirrors
    ``orchestrators.single_agent_workflow.synthetic_traceparent``; duplicated
    here to avoid an adapter → orchestrator import cycle)."""
    hexid = (event_id or "").replace("-", "")
    trace_id = (hexid + "0" * 32)[:32]
    parent_id = (hexid + "0" * 16)[:16]
    return f"00-{trace_id}-{parent_id}-01"


class AgentDispatchOutboxWriter:
    """INSERTs a dispatch request row on the caller's open transaction."""

    def __init__(
        self,
        *,
        conn: Any,
        source_project: str,
        source_service: str = "chora-ai-kernel-orchestrator",
    ) -> None:
        if not (source_project or "").strip():
            raise ValueError("source_project required")
        self._conn = conn
        self._source_project = source_project
        self._source_service = source_service

    @property
    def source_project(self) -> str:
        return self._source_project

    @property
    def source_service(self) -> str:
        return self._source_service

    async def queue_request(self, request: dict[str, Any]) -> str:
        """Queue one dispatch request built by ``agent_dispatch.build_dispatch_request``.

        ``ON CONFLICT (idempotency_key) DO NOTHING`` makes a re-queue a no-op,
        which matters because LangGraph re-executes a node from the top when a
        park resumes: the node rebuilds the same request, and the deterministic
        idempotency key means the second insert cannot produce a second
        dispatch.
        """
        envelope = dict(request["envelope"])
        # Stamp the writer's provenance last so a caller cannot fabricate it.
        envelope["source_project"] = self._source_project
        envelope["source_service"] = self._source_service
        # The Go eventbus dead-letters a message whose envelope it cannot
        # rebuild, and traceparent is mandatory there. A dispatch is generated
        # here, not received, so it carries no upstream trace context — derive
        # a valid one from the event id rather than shipping an empty field.
        if not str(envelope.get("traceparent") or "").strip():
            envelope["traceparent"] = _synthetic_traceparent(str(envelope.get("event_id") or ""))
        row_id = str(uuid4())
        params = {
            "id": row_id,
            "workflow_id": request["workflow_id"],
            "tenant_id": request["tenant_id"],
            "gcid": request.get("gcid", ""),
            "event_type": request["event_type"],
            "topic": request["topic"],
            "payload": json.dumps(request["body"], separators=(",", ":")).encode("utf-8"),
            "envelope": json.dumps(envelope),
            "idempotency_key": request["idempotency_key"],
            "occurred_at": envelope["occurred_at"],
        }
        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        logger.info(
            "agent_dispatch_outbox.queued",
            extra={
                "row_id": row_id,
                "topic": request["topic"],
                "idempotency_key": request["idempotency_key"],
                "thread_id": request["workflow_id"],
            },
        )
        return row_id


__all__ = ["AgentDispatchOutboxWriter"]
