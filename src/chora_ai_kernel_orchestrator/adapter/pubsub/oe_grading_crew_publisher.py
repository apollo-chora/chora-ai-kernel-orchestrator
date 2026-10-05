"""OEGradingCrewOutboxWriter — JSON outbox emitter for ADR-172.

Writes chora.delivery.grading.submission_completed.v1 to ai_kernel_outbox_events
(migration 0003_outbox.sql); the OutboxDispatcher drains it to Pub/Sub. Mirrors
QGenCrewTerminalOutboxWriter but JSON-only — the submission_completed topic is a
JSON topic (NO Pub/Sub Schema Registry binary schema attached), which chora-
delivery's grading inbox decodes via json.Unmarshal (decodeSubmissionCompleted).

Idempotency: idempotency_key = grading.submission_completed.{submission_id};
ON CONFLICT (idempotency_key) DO NOTHING makes the outbox exactly-once even when
a graph resume re-runs the terminal node (D6).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

TOPIC_SUBMISSION_COMPLETED = "chora.delivery.grading.submission_completed.v1"
SCHEMA_VERSION = "1"
IMDA_DIM_TRANSPARENCY = "transparency"

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


class OEGradingCrewOutboxWriter:
    """Writes the OE grading submission_completed.v1 event to the outbox."""

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

    async def publish_submission_completed(
        self,
        *,
        submission_id: str,
        assessment_id: str,
        tenant_id: str,
        learner_gcid: str,
        graded: list[dict[str, Any]],
        overall_comment: str,
        overall_comment_model_id: str,
        outcome: str,
        grading_job_id: str = "",
        overall_comment_response_id: str = "",
        failure_message: str = "",
        traceparent: str = "",
        tracestate: str = "",
    ) -> str:
        now = _dt.datetime.now(tz=_dt.UTC)
        row_id = str(uuid4())
        event_id = str(uuid4())
        idempotency_key = f"grading.submission_completed.{submission_id}"
        envelope: dict[str, str] = {
            "event_id": event_id,
            "idempotency_key": idempotency_key,
            "tenant_id": tenant_id,
            "gcid": learner_gcid,
            "occurred_at": now.isoformat(),
            "published_at": now.isoformat(),
            "traceparent": traceparent,
            "tracestate": tracestate,
            "source_project": self._source_project,
            "source_service": self._source_service,
            "schema_version": SCHEMA_VERSION,
            "chora_imda_dimension": IMDA_DIM_TRANSPARENCY,
            # delivery's grading inbox routes by the message's subject. The
            # NATS publisher reserves the "subject" name (it would collide with
            # the positional subject arg and raise TypeError — stranding the
            # row), so carry the routing topic under the non-reserved
            # "event_topic" key. delivery's
            # dispatchGradingInbox falls back to "event_topic" when the "topic"
            # attribute is absent (Go producers like the oe_batch path can set
            # "topic" directly; this Python producer cannot).
            "event_topic": TOPIC_SUBMISSION_COMPLETED,
        }
        body: dict[str, Any] = {
            "submission_id": submission_id,
            # Echoed for completion↔request correlation (the originating
            # submission_requested.v1 carries grading_job_id); delivery may use it
            # to disambiguate concurrent re-grades of the same submission.
            "grading_job_id": grading_job_id,
            "assessment_id": assessment_id,
            "tenant_id": tenant_id,
            "learner_gcid": learner_gcid,
            "graded": graded,
            "overall_comment": overall_comment,
            "overall_comment_model_id": overall_comment_model_id,
            # IMDA D2 provenance for the overall comment (delivery maps this to
            # SubmissionGrading.OverallCommentResponseID).
            "overall_comment_response_id": overall_comment_response_id,
            "outcome": outcome,
            "failure_message": failure_message,
            "traceparent": traceparent,
            "completed_at": now.isoformat(),
        }
        params = {
            "id": row_id,
            "workflow_id": submission_id,
            "tenant_id": tenant_id,
            "gcid": learner_gcid,
            "event_type": "delivery.grading.submission_completed",
            "topic": TOPIC_SUBMISSION_COMPLETED,
            "payload": json.dumps(body, separators=(",", ":")).encode("utf-8"),
            "envelope": json.dumps(envelope),
            "idempotency_key": idempotency_key,
            "occurred_at": now,
        }
        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        logger.info(
            "oe_grading_outbox.queued",
            extra={"row_id": row_id, "submission_id": submission_id, "topic": TOPIC_SUBMISSION_COMPLETED},
        )
        return row_id


__all__ = ["OEGradingCrewOutboxWriter", "TOPIC_SUBMISSION_COMPLETED"]
