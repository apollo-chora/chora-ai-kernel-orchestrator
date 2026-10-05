"""WeaknessAnalyzedOutboxWriter — BINARY transactional-outbox emitter for
``chora.consumption.weakness.analyzed.v1`` (W2-glue of the Growth-Edge layer).

The weakness-analyser crew distils an uploaded weakness document into N
``ExtractedGrowthEdge`` records (see ``domain/weakness_analyser_crew/runner.py``)
and this writer persists ONE ``ai_kernel_outbox_events`` row (migration
0003_outbox.sql) whose ``payload`` is the canonical proto3 wire bytes from
``weakness_proto_encoder.encode_weakness_analyzed``. The existing
``OutboxDispatcher`` + ``NatsPublisher`` drain it to Pub/Sub.

Why binary (the binary-proto trap): the analyzed.v1 topic carries a Pub/Sub
Schema Registry BINARY schema (flat
``proto/events-flat/consumption/weakness/analyzed.proto``); a JSON publish
dead-letters with INVALID_BINARY_PROTO_MESSAGE. chora-consumption (Go) decodes
the binary cleanly via ``proto.Unmarshal`` — no Python ``_pb2`` skew on that leg.

Why ``event_topic`` in the envelope: the Python ``NatsPublisher``
strips the reserved ``topic`` attribute (it would collide with the publish()
positional arg), and chora-consumption's push handler routes by the message
``topic`` attribute. So — exactly like the OE grading publisher — the routing
topic rides the non-reserved ``event_topic`` key; consumption falls back to it.

D6 (per [[feedback-d6-resilience-first-class]]):
  * P2 idempotency — ``idempotency_key = weakness.analyzed.{upload_id}`` +
    ``ON CONFLICT (idempotency_key) DO NOTHING`` makes re-delivery a no-op.
  * P3 delivery — the dispatcher retries/dead-letters; the subscriber acks only
    after this row is durably written.
  * P4 OTel — traceparent/tracestate land in the envelope → Pub/Sub attributes.

Per CLAUDE.md cross-cutting rule, the envelope carries the mandatory fields plus
``chora_imda_dimension = transparency`` (ADR-141; D2 model attribution).

Kept SEPARATE from the qgen/OE writers so the concurrent batch-qgen + OE tracks
never edit the same file in the shared worktree.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any
from uuid import uuid4

import uuid_utils as _uuid_utils

from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_proto_encoder import (
    encode_weakness_analyzed,
)

logger = logging.getLogger(__name__)

# Topic name — chora.consumption.weakness.analyzed.v1 (binary Schema-Registry topic).
TOPIC_WEAKNESS_ANALYZED = "chora.consumption.weakness.analyzed.v1"

# IMDA dimension per ADR-141 — model attribution + cost evidence on analysis (D2).
IMDA_DIM_TRANSPARENCY = "transparency"

# Schema version — matches weakness.proto WeaknessAnalyzed v1.
SCHEMA_VERSION = 1

# Outbox INSERT — ON CONFLICT (idempotency_key) DO NOTHING so a re-delivered or
# re-analysed upload re-emits without raising (exactly-once at the outbox layer).
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


class WeaknessAnalyzedOutboxWriter:
    """Writes ``weakness.analyzed.v1`` (binary) to ``ai_kernel_outbox_events``.

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

    async def publish_weakness_analyzed(
        self,
        *,
        body: dict[str, Any],
        traceparent: str = "",
        tracestate: str = "",
    ) -> str:
        """Persist one ``weakness.analyzed.v1`` event row.

        ``body`` is the dict returned by ``WeaknessAnalyserRunner.analyze`` —
        keys: upload_id, tenant_id, learner_gcid, model_used, input_token_count,
        output_token_count, analyzed_at (RFC3339), edges[]. An empty ``edges``
        list is valid (nothing weak detected) — the event still fires so
        consumption can correlate the upload.

        Returns the outbox row id (UUIDv4 — distinct from the deterministic
        idempotency_key the dispatcher leaves untouched).
        """
        upload_id = str(body.get("upload_id", "")).strip()
        if not upload_id:
            raise ValueError("WeaknessAnalyzedOutboxWriter: body missing upload_id")
        tenant_id = str(body.get("tenant_id", ""))
        learner_gcid = str(body.get("learner_gcid", ""))

        now = _dt.datetime.now(tz=_dt.UTC)
        row_id = str(uuid4())
        event_id = str(_uuid_utils.uuid7())
        idempotency_key = f"weakness.analyzed.{upload_id}"
        # occurred_at = the analysis moment (the runner stamps analyzed_at);
        # published_at = emission moment. The Go subscriber uses env.OccurredAt
        # as the Growth Edge's last_evidenced_at, so prefer the true analysis ts.
        occurred_at_iso = str(body.get("analyzed_at", "")).strip() or now.isoformat()

        envelope: dict[str, Any] = {
            "event_id": event_id,
            "idempotency_key": idempotency_key,
            "tenant_id": tenant_id,
            "gcid": learner_gcid,
            "occurred_at": occurred_at_iso,
            "published_at": now.isoformat(),
            "traceparent": traceparent,
            "tracestate": tracestate,
            "source_project": self._source_project,
            "source_service": self._source_service,
            "schema_version": SCHEMA_VERSION,
            "chora_imda_dimension": IMDA_DIM_TRANSPARENCY,
            # Routing hint for chora-consumption's push handler — the publisher
            # strips the reserved "topic" attribute, so carry it here.
            "event_topic": TOPIC_WEAKNESS_ANALYZED,
        }

        payload = encode_weakness_analyzed(envelope, body)

        params = {
            "id": row_id,
            "workflow_id": upload_id,
            "tenant_id": tenant_id,
            "gcid": learner_gcid,
            "event_type": "consumption.weakness.analyzed",
            "topic": TOPIC_WEAKNESS_ANALYZED,
            "payload": payload,
            "envelope": json.dumps(envelope),
            "idempotency_key": idempotency_key,
            "occurred_at": now,
        }
        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        logger.info(
            "weakness_analyzed_outbox.queued",
            extra={
                "row_id": row_id,
                "upload_id": upload_id,
                "tenant_id": tenant_id,
                "edge_count": len(body.get("edges") or []),
                "topic": TOPIC_WEAKNESS_ANALYZED,
            },
        )
        return row_id


__all__ = [
    "IMDA_DIM_TRANSPARENCY",
    "SCHEMA_VERSION",
    "TOPIC_WEAKNESS_ANALYZED",
    "WeaknessAnalyzedOutboxWriter",
]
