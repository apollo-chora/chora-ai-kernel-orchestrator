"""WeaknessOutputsGeneratedOutboxWriter - BINARY transactional-outbox emitter for
``chora.consumption.weakness.outputs_generated.v1`` (WS-7 delivery).

The weakness-analyser crew generates the learner-selected metered artifacts
(study_aids prose + a critic-gated practice_test) in a DETACHED background task
(``orchestrators/weakness_analyser_crew.py::_run_detached_generation``). Until
this writer existed those artifacts were produced, metered against the learner's
mana, and then reached no learner surface at all: the WS-7 delivery gap.

This writer persists ONE ``ai_kernel_outbox_events`` row (migration
0003_outbox.sql) whose ``payload`` is the canonical proto3 wire bytes from
``weakness_proto_encoder.encode_weakness_outputs_generated``. The existing
``OutboxDispatcher`` + ``NatsPublisher`` drain it to Pub/Sub, and
chora-consumption projects it into ``growth_edge_outputs`` for the A+ surface.

Why a SEPARATE event from weakness.analyzed.v1: analysis publishes first so the
HITL resume returns without waiting on up to 4 slow LLM generations (ADR-205
defect 3, reordered in `7d0fb71eb`). The artifacts land whenever they land, so
they need their own event rather than a field on the analysis.

Why binary (the binary-proto trap): the topic carries a Pub/Sub Schema Registry
BINARY schema (flat
``proto/events-flat/consumption/weakness/outputs_generated.proto``); a JSON
publish dead-letters with INVALID_BINARY_PROTO_MESSAGE.

Why ``event_topic`` in the envelope: the Python ``NatsPublisher``
strips the reserved ``topic`` attribute (it collides with the publish()
positional arg), and chora-consumption's push handler routes by the message
``topic`` attribute. So, exactly like the analyzed writer, the routing topic
rides the non-reserved ``event_topic`` key.

D6 (per [[feedback-d6-resilience-first-class]]):
  * P2 idempotency - ``idempotency_key = weakness.outputs_generated.{upload_id}``
    + ``ON CONFLICT (idempotency_key) DO NOTHING`` makes re-delivery a no-op.
  * P3 delivery - the dispatcher retries/dead-letters; the artifacts are durable
    in the outbox before any publish is attempted.
  * P4 OTel - traceparent/tracestate land in the envelope, then in the Pub/Sub
    attributes.

Kept SEPARATE from the analyzed writer so the two Growth-Edge tracks never edit
the same file in the shared worktree.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any
from uuid import uuid4

import uuid_utils as _uuid_utils

from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_proto_encoder import (
    encode_weakness_outputs_generated,
)

logger = logging.getLogger(__name__)

# Topic name - chora.consumption.weakness.outputs_generated.v1 (binary schema).
TOPIC_WEAKNESS_OUTPUTS_GENERATED = "chora.consumption.weakness.outputs_generated.v1"

# IMDA dimension per ADR-141 - model attribution on generated learner content.
IMDA_DIM_TRANSPARENCY = "transparency"

# Schema version - matches weakness.proto WeaknessOutputsGenerated v1.
SCHEMA_VERSION = 1

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


class WeaknessOutputsGeneratedOutboxWriter:
    """Writes ``weakness.outputs_generated.v1`` (binary) to the outbox.

    Connection is caller-owned (the crew wiring holds the autocommit
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

    async def publish_outputs_generated(
        self,
        *,
        body: dict[str, Any],
        traceparent: str = "",
        tracestate: str = "",
    ) -> str:
        """Persist one ``weakness.outputs_generated.v1`` event row.

        ``body`` keys: upload_id, tenant_id, learner_gcid, generated_at
        (RFC3339), and ``outputs`` (the list ``CrewOutputGenerator.generate``
        returns, each ``{"type", "content", "metered"}``).

        An EMPTY ``outputs`` list is valid and still emits: every kind screened
        out or soft-failed is a real outcome, and the event firing is what lets
        consumption distinguish "generated nothing" from "not generated yet".

        Returns the outbox row id (UUIDv4, distinct from the deterministic
        idempotency_key the dispatcher leaves untouched).
        """
        upload_id = str(body.get("upload_id", "")).strip()
        if not upload_id:
            raise ValueError("WeaknessOutputsGeneratedOutboxWriter: body missing upload_id")
        tenant_id = str(body.get("tenant_id", ""))
        learner_gcid = str(body.get("learner_gcid", ""))

        now = _dt.datetime.now(tz=_dt.UTC)
        row_id = str(uuid4())
        idempotency_key = f"weakness.outputs_generated.{upload_id}"
        # occurred_at = the generation moment; published_at = emission moment.
        occurred_at_iso = str(body.get("generated_at", "")).strip() or now.isoformat()

        envelope: dict[str, Any] = {
            "event_id": str(_uuid_utils.uuid7()),
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
            # Routing hint for chora-consumption's push handler - the publisher
            # strips the reserved "topic" attribute, so carry it here.
            "event_topic": TOPIC_WEAKNESS_OUTPUTS_GENERATED,
        }

        payload = encode_weakness_outputs_generated(envelope, body)

        params = {
            "id": row_id,
            "workflow_id": upload_id,
            "tenant_id": tenant_id,
            "gcid": learner_gcid,
            "event_type": "consumption.weakness.outputs_generated",
            "topic": TOPIC_WEAKNESS_OUTPUTS_GENERATED,
            "payload": payload,
            "envelope": json.dumps(envelope),
            "idempotency_key": idempotency_key,
            "occurred_at": now,
        }
        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        logger.info(
            "weakness_outputs_outbox.queued",
            extra={
                "row_id": row_id,
                "upload_id": upload_id,
                "tenant_id": tenant_id,
                "output_count": len(body.get("outputs") or []),
                "kinds": [str(o.get("type", "")) for o in (body.get("outputs") or [])],
                "topic": TOPIC_WEAKNESS_OUTPUTS_GENERATED,
            },
        )
        return row_id


__all__ = [
    "IMDA_DIM_TRANSPARENCY",
    "SCHEMA_VERSION",
    "TOPIC_WEAKNESS_OUTPUTS_GENERATED",
    "WeaknessOutputsGeneratedOutboxWriter",
]
