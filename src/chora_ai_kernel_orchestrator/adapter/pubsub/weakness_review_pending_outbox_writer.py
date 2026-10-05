"""WeaknessReviewPendingOutboxWriter — BINARY transactional-outbox emitter for
``chora.consumption.weakness.review_pending.v1`` (CHO-1973 Wave A, ADR-205 D4/D5).

When the graduated Growth-Edge crew reaches its in-graph HITL ``interrupt()``, the
graph subscriber emits ONE ``review_pending`` event so chora-consumption can
surface the bounded review panel to the A+ FE. This writer persists ONE
``ai_kernel_outbox_events`` row whose ``payload`` is the canonical proto3 wire
bytes from ``review_pending_proto_encoder.encode_weakness_review_pending``. The
existing ``OutboxDispatcher`` + ``NatsPublisher`` drain it to Pub/Sub.

Mirrors ``weakness_analyzed_outbox_writer`` exactly (binary topic, ``event_topic``
routing key workaround, deterministic idempotency_key, ON CONFLICT DO NOTHING),
so the same drain machinery handles it with no new dispatcher.

Differences from the analyzed writer:
  * topic / event_type / idempotency_key prefix are ``review_pending``;
  * the IMDA dimension is ``fairness_and_human_oversight`` (D4) — the review IS
    the human-oversight checkpoint (ADR-141 / ADR-205 D4), not model attribution;
  * the writer stamps ``pending_at`` = emission moment into the panel so the FE +
    consumption see a real "review opened" timestamp.

D6 (per [[feedback-d6-resilience-first-class]]):
  * P2 idempotency — ``idempotency_key = weakness.review_pending.{upload_id}`` +
    ``ON CONFLICT (idempotency_key) DO NOTHING``; one upload → one review_pending.
  * P3 delivery — the dispatcher retries/dead-letters; the subscriber acks only
    after this row is durably written.
  * P4 OTel — traceparent/tracestate flow into the envelope → Pub/Sub attributes.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any
from uuid import uuid4

import uuid_utils as _uuid_utils

from chora_ai_kernel_orchestrator.adapter.pubsub.review_pending_proto_encoder import (
    encode_weakness_review_pending,
)

logger = logging.getLogger(__name__)

# Topic name — chora.consumption.weakness.review_pending.v1 (binary Schema-Registry topic).
TOPIC_WEAKNESS_REVIEW_PENDING = "chora.consumption.weakness.review_pending.v1"

# IMDA dimension per ADR-141 — the bounded HITL review IS the human-oversight
# checkpoint (D4), distinct from the analyzed event's model attribution (D2).
IMDA_DIM_OVERSIGHT = "fairness_and_human_oversight"

# Schema version — matches weakness.proto WeaknessReviewPending v1.
SCHEMA_VERSION = 1

# Outbox INSERT — same shape + ON CONFLICT (idempotency_key) DO NOTHING as the
# analyzed writer (exactly-once at the outbox layer).
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


class WeaknessReviewPendingOutboxWriter:
    """Writes ``weakness.review_pending.v1`` (binary) to ``ai_kernel_outbox_events``.

    Connection is caller-owned (main.py lifespan holds the autocommit
    ReconnectingAsyncConnection); this writer does NOT manage the lifecycle.
    """

    def __init__(
        self,
        *,
        conn: Any,
        source_project: str,
        source_service: str = "chora-ai-kernel-orchestrator",
    ) -> None:
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

    async def publish_review_pending(
        self,
        *,
        panel: dict[str, Any],
        traceparent: str = "",
        tracestate: str = "",
    ) -> str:
        """Persist one ``weakness.review_pending.v1`` event row.

        ``panel`` is the dict from ``build_review_panel`` — keys: upload_id,
        tenant_id, learner_gcid, familiar, proposed_edges[], candidate_struggles[],
        available_outputs[]. An empty ``proposed_edges`` list is valid (nothing
        weak surfaced) — the event still fires so consumption can open the review.

        Returns the outbox row id (UUIDv4 — distinct from the deterministic
        idempotency_key the dispatcher leaves untouched).
        """
        upload_id = str(panel.get("upload_id", "")).strip()
        if not upload_id:
            raise ValueError("WeaknessReviewPendingOutboxWriter: panel missing upload_id")
        tenant_id = str(panel.get("tenant_id", ""))
        learner_gcid = str(panel.get("learner_gcid", ""))

        now = _dt.datetime.now(tz=_dt.UTC)
        now_iso = now.isoformat()
        row_id = str(uuid4())
        event_id = str(_uuid_utils.uuid7())
        idempotency_key = f"weakness.review_pending.{upload_id}"

        # Stamp the review-opened moment onto the panel (proto field 9) so the FE
        # + consumption see a real pending_at, not a fabricated/zero timestamp.
        panel_with_ts = {**panel, "pending_at": now_iso}

        envelope: dict[str, Any] = {
            "event_id": event_id,
            "idempotency_key": idempotency_key,
            "tenant_id": tenant_id,
            "gcid": learner_gcid,
            "occurred_at": now_iso,
            "published_at": now_iso,
            "traceparent": traceparent,
            "tracestate": tracestate,
            "source_project": self._source_project,
            "source_service": self._source_service,
            "schema_version": SCHEMA_VERSION,
            "chora_imda_dimension": IMDA_DIM_OVERSIGHT,
            # Routing hint for chora-consumption's push handler — the publisher
            # strips the reserved "topic" attribute, so carry it here.
            "event_topic": TOPIC_WEAKNESS_REVIEW_PENDING,
        }

        payload = encode_weakness_review_pending(envelope, panel_with_ts)

        params = {
            "id": row_id,
            "workflow_id": upload_id,
            "tenant_id": tenant_id,
            "gcid": learner_gcid,
            "event_type": "consumption.weakness.review_pending",
            "topic": TOPIC_WEAKNESS_REVIEW_PENDING,
            "payload": payload,
            "envelope": json.dumps(envelope),
            "idempotency_key": idempotency_key,
            "occurred_at": now,
        }
        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        logger.info(
            "weakness_review_pending_outbox.queued",
            extra={
                "row_id": row_id,
                "upload_id": upload_id,
                "tenant_id": tenant_id,
                "proposed_edge_count": len(panel.get("proposed_edges") or []),
                "topic": TOPIC_WEAKNESS_REVIEW_PENDING,
            },
        )
        return row_id


__all__ = [
    "IMDA_DIM_OVERSIGHT",
    "SCHEMA_VERSION",
    "TOPIC_WEAKNESS_REVIEW_PENDING",
    "WeaknessReviewPendingOutboxWriter",
]
