"""QGenCrewTerminalOutboxWriter — transactional-outbox emitter for the
qgen 2-agent crew's terminal events.

Implements the ``_TerminalPublisher`` Protocol from
``orchestrators/qgen_crew_runner.py`` by INSERTing event rows into
``ai_kernel_outbox_events`` (migration ``0003_outbox.sql``). The
``OutboxDispatcher`` (separate asyncio task in ``main.py``) drains the
table + publishes to NATS JetStream via ``NatsPublisher``.

Mirrors the canonical pattern in
``services/chora-closure-orchestrator/src/chora_closure_orchestrator/adapter/events/publisher_outbox.py``
(B.6.2.a producer-side durable emission per [[feedback-d6-resilience-first-class]]).

Why outbox instead of direct publish:

* **Pod-death survival (D6 P1)** — the LangGraph PostgresSaver checkpoint
  + outbox row write hit the same database (``chora_ai_kernel``).
  If the pod dies between checkpoint and Pub/Sub publish, the next pod
  resumes from the checkpoint AND finds the pending outbox row to
  publish. Direct-publish would silently drop the event.
* **Idempotency (D6 P2)** — deterministic ``idempotency_key`` per
  (assist_id, topic_suffix). ``ON CONFLICT (idempotency_key) DO NOTHING``
  swallows duplicate INSERTs from graph re-runs (resume from checkpoint
  re-executes the terminal node), so the outbox stays exactly-once.
* **DLQ (D6 P3)** — the existing ``OutboxDispatcher`` writes to
  ``ai_kernel_outbox_dead_letters`` after max_retries exhausted.
* **OTel (D6 P4)** — traceparent + tracestate land in the envelope and
  are propagated as Pub/Sub message attributes by the dispatcher's
  ``NatsPublisher``.

Per CLAUDE.md cross-cutting rule, the envelope carries the 11 mandatory
fields plus ``chora_imda_dimension`` per ADR-141.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any
from uuid import uuid4

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    encode_ai_assist_chunk_completed,
    encode_ai_assist_completed,
    encode_ai_assist_progress,
    encode_ai_assist_refused,
)

logger = logging.getLogger(__name__)

# Topic names — chora-infra/terraform/modules/m10-data-plane/main.tf:185-187.
TOPIC_AI_ASSIST_COMPLETED = "chora.creation.ai_assist.completed.v1"
TOPIC_AI_ASSIST_REFUSED = "chora.creation.ai_assist.refused.v1"
# Mid-run live-trace event (CHO Phase-1 streaming) — emitted once per qgen
# graph node, consumed ONLY by chora-creation to stream the partial trace.
TOPIC_AI_ASSIST_PROGRESS = "chora.creation.ai_assist.progress.v1"

# ADR-251 D5 (CHO-2398) - one per finished chunk of a multi-chunk set job.
# Domain data (the chunk's published-shape candidates), DISTINCT from the
# trace-only progress lane; consumed ONLY by chora-creation. NOT terminal:
# publishing it never touches the in-flight registry.
TOPIC_AI_ASSIST_CHUNK_COMPLETED = "chora.creation.ai_assist.chunk_completed.v1"

# IMDA dimension labels per ADR-141 + ai_assist.proto §line 31.
IMDA_DIM_ACCOUNTABILITY = "accountability"
IMDA_DIM_SAFETY_AND_ROBUSTNESS = "safety_and_robustness"

# Schema version — matches ai_assist.proto §AiAssistCompleted v1.
SCHEMA_VERSION = "1"

# Outbox INSERT — ON CONFLICT (idempotency_key) DO NOTHING so a graph
# re-run from checkpoint re-emits the terminal event without raising.
# UNIQUE index on idempotency_key (per migration 0003_outbox.sql) makes
# this exactly-once at the outbox layer.
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


class QGenCrewTerminalOutboxWriter:
    """Writes qgen 2-agent crew terminal events to ``ai_kernel_outbox_events``.

    Implements the ``_TerminalPublisher`` Protocol declared in
    ``qgen_crew_runner.py``. The actual Pub/Sub publish hop is the
    ``OutboxDispatcher``'s concern (separate adapter; existing).

    Connection management: the caller passes a psycopg ``AsyncConnection``
    (or a duck-typed equivalent for tests). The writer does NOT own the
    connection lifecycle — the orchestrator main.py holds it.
    """

    def __init__(
        self,
        *,
        conn: Any,
        source_project: str,
        source_service: str = "chora-ai-kernel-orchestrator",
        inflight_registry: Any | None = None,
    ) -> None:
        if not (source_project or "").strip():
            raise ValueError("source_project required")
        if not (source_service or "").strip():
            raise ValueError("source_service required")
        self._conn = conn
        self._source_project = source_project
        self._source_service = source_service
        # ADR-251 D4 (CHO-2398): when wired, a TERMINAL publish deletes the
        # job's ai_assist_inflight_jobs row inside the SAME transaction as
        # the outbox INSERT (registry.delete_in_tx executes on this shared
        # conn without committing; the single commit below covers both).
        # None keeps every publish byte-identical to the pre-D4 writer.
        self._inflight_registry = inflight_registry

    # ---- Properties exposed for introspection ------------------------------

    @property
    def source_project(self) -> str:
        return self._source_project

    @property
    def source_service(self) -> str:
        return self._source_service

    # ---- _TerminalPublisher Protocol ---------------------------------------

    async def publish_completed(
        self,
        *,
        assist_id: str,
        tenant_id: str,
        author_gcid: str,
        candidate_payload_json: str,
        pipeline_trace_json: str,
        quality_warning: bool,
        attempt_count: int,
        critic_notes: str,
        mana_charged: int,
        traceparent: str = "",
        tracestate: str = "",
        generated_count: int = 0,
        generation_summary: dict[str, Any] | None = None,
    ) -> str:
        """Persist a completed.v1 event row in the outbox.

        Returns the outbox row id (UUIDv4 — distinct from the
        deterministic idempotency_key). The dispatcher uses the id for
        per-worker SKIP LOCKED selection.

        ``generated_count`` (typed proto field 3) + ``generation_summary``
        (typed proto field 16, CHO-1819) carry the honest mixed-type /
        strict-shortfall accounting. Both default to the legacy single-path
        no-op (0 / omitted) so the live single-candidate publish is unchanged;
        the set-native batch runner populates them. The binary encoder already
        serialises both fields (proto_wire_encoder.py).
        """
        body: dict[str, Any] = {
            "assist_id": assist_id,
            "tenant_id": tenant_id,
            "author_gcid": author_gcid,
            "candidate_payload_json": candidate_payload_json,
            "pipeline_trace_json": pipeline_trace_json,
            "quality_warning": quality_warning,
            "attempt_count": attempt_count,
            "critic_notes": critic_notes,
            "mana_charged": mana_charged,
            "generated_count": generated_count,
        }
        if generation_summary:
            body["generation_summary"] = generation_summary
        return await self._write(
            assist_id=assist_id,
            tenant_id=tenant_id,
            gcid=author_gcid,
            event_type="creation.ai_assist.completed",
            topic=TOPIC_AI_ASSIST_COMPLETED,
            topic_suffix="completed",
            body=body,
            traceparent=traceparent,
            tracestate=tracestate,
            imda_dimension=IMDA_DIM_ACCOUNTABILITY,
            extra_envelope={},
        )

    async def publish_chunk_completed(
        self,
        *,
        assist_id: str,
        tenant_id: str,
        author_gcid: str,
        chunk_index: int,
        chunk_count: int,
        candidates_payload_json: str,
        candidate_count: int,
        warned_count: int = 0,
        images_rendered: int = 0,
        images_dropped: int = 0,
        images_failed: int = 0,
        images_skipped: int = 0,
        traceparent: str = "",
        tracestate: str = "",
    ) -> str:
        """Persist one chunk_completed.v1 row (ADR-251 D5, CHO-2398).

        NOT a terminal: the in-flight registry row survives (the same-tx
        delete above keys on the terminal topics only). Idempotency rides
        ``ai_assist.chunk_completed.{chunk_index}.{assist_id}`` via the
        topic_suffix, so a checkpoint-resumed re-run of publish_chunk
        re-emits harmlessly (ON CONFLICT DO NOTHING).
        """
        body: dict[str, Any] = {
            "assist_id": assist_id,
            "tenant_id": tenant_id,
            "author_gcid": author_gcid,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "candidates_payload_json": candidates_payload_json,
            "candidate_count": candidate_count,
            "warned_count": warned_count,
            "images_rendered": images_rendered,
            "images_dropped": images_dropped,
            "images_failed": images_failed,
            "images_skipped": images_skipped,
        }
        return await self._write(
            assist_id=assist_id,
            tenant_id=tenant_id,
            gcid=author_gcid,
            event_type="creation.ai_assist.chunk_completed",
            topic=TOPIC_AI_ASSIST_CHUNK_COMPLETED,
            topic_suffix=f"chunk_completed.{chunk_index}",
            body=body,
            traceparent=traceparent,
            tracestate=tracestate,
            imda_dimension=IMDA_DIM_ACCOUNTABILITY,
            extra_envelope={},
        )

    async def publish_refused(
        self,
        *,
        assist_id: str,
        tenant_id: str,
        author_gcid: str,
        refusal_reason: str,
        model_armor_verdict: str,
        user_facing_message: str,
        last_candidate_payload_json: str,
        pipeline_trace_json: str,
        attempt_count: int,
        mana_charged: int,
        traceparent: str = "",
        tracestate: str = "",
    ) -> str:
        body: dict[str, Any] = {
            "assist_id": assist_id,
            "tenant_id": tenant_id,
            "author_gcid": author_gcid,
            "refusal_reason": refusal_reason,
            "model_armor_verdict": model_armor_verdict or "armor:unspecified",
            "user_facing_message": user_facing_message,
            "last_candidate_payload_json": last_candidate_payload_json,
            "pipeline_trace_json": pipeline_trace_json,
            "attempt_count": attempt_count,
            "mana_charged": mana_charged,
        }
        # Extra envelope attrs flow through to Pub/Sub message attributes
        # via the OutboxDispatcher → NatsPublisher hop. D3
        # governance subscribers index without decoding the payload.
        extra = {
            "refusal_reason": refusal_reason,
            "model_armor_verdict": model_armor_verdict or "armor:unspecified",
        }
        return await self._write(
            assist_id=assist_id,
            tenant_id=tenant_id,
            gcid=author_gcid,
            event_type="creation.ai_assist.refused",
            topic=TOPIC_AI_ASSIST_REFUSED,
            topic_suffix="refused",
            body=body,
            traceparent=traceparent,
            tracestate=tracestate,
            imda_dimension=IMDA_DIM_SAFETY_AND_ROBUSTNESS,
            extra_envelope=extra,
        )

    async def publish_progress(
        self,
        *,
        assist_id: str,
        tenant_id: str,
        author_gcid: str,
        pipeline_trace_json: str,
        step_index: int,
        step_name: str = "",
        step_status: str = "",
        traceparent: str = "",
        tracestate: str = "",
    ) -> str:
        """Persist a progress.v1 outbox row — the mid-run live-trace event,
        emitted once per qgen graph node with the cumulative-so-far trace.

        ``topic_suffix=f"progress.{step_index}"`` makes the idempotency_key
        PER-STEP unique (``ai_assist.progress.{step_index}.{assist_id}``) — the
        per-job suffix used by completed/refused would collapse every progress
        emit to one row via ON CONFLICT DO NOTHING. ``step_index`` is the
        monotonic cursor (== len(trace)) the consumer uses to drop stale /
        out-of-order partials.
        """
        body: dict[str, Any] = {
            "assist_id": assist_id,
            "tenant_id": tenant_id,
            "author_gcid": author_gcid,
            "pipeline_trace_json": pipeline_trace_json,
            "step_index": step_index,
            "step_name": step_name,
            "step_status": step_status,
        }
        return await self._write(
            assist_id=assist_id,
            tenant_id=tenant_id,
            gcid=author_gcid,
            event_type="creation.ai_assist.progress",
            topic=TOPIC_AI_ASSIST_PROGRESS,
            topic_suffix=f"progress.{step_index}",
            body=body,
            traceparent=traceparent,
            tracestate=tracestate,
            imda_dimension=IMDA_DIM_ACCOUNTABILITY,
            extra_envelope={},
        )

    # ---- Internal write ----------------------------------------------------

    async def _write(
        self,
        *,
        assist_id: str,
        tenant_id: str,
        gcid: str,
        event_type: str,
        topic: str,
        topic_suffix: str,
        body: dict[str, Any],
        traceparent: str,
        tracestate: str,
        imda_dimension: str,
        extra_envelope: dict[str, str],
    ) -> str:
        """Build envelope + INSERT into ai_kernel_outbox_events."""
        now = _dt.datetime.now(tz=_dt.UTC)
        # Outbox row id — UUIDv4 for the row (NOT the event_id). The
        # event_id lives in the envelope and is what subscribers see.
        row_id = str(uuid4())
        event_id = str(uuid4())
        idempotency_key = f"ai_assist.{topic_suffix}.{assist_id}"

        envelope: dict[str, str] = {
            "event_id": event_id,
            "idempotency_key": idempotency_key,
            "tenant_id": tenant_id,
            "gcid": gcid,
            "occurred_at": now.isoformat(),
            "published_at": now.isoformat(),
            "traceparent": traceparent,
            "tracestate": tracestate,
            "source_project": self._source_project,
            "source_service": self._source_service,
            "schema_version": SCHEMA_VERSION,
            "chora_imda_dimension": imda_dimension,
            # Routing hint for consumption's push dispatch — the outbox
            # publisher strips the reserved "topic" attribute, so carry the
            # non-reserved alias (the weakness/oe-grading writer precedent).
            # Without it every qgen terminal is ACK-DROPPED downstream and
            # campaign question sets stay `requested` forever (CHO-2087).
            "event_topic": topic,
        }
        envelope.update(extra_envelope)

        # The NATS subject schema attaches a BINARY-encoded Protobuf
        # schema to the two AiAssist terminal topics, so JSON publishes
        # were failing at validation with INVALID_BINARY_PROTO_MESSAGE
        # (5/5 outbox rows status=failed prior to this fix — see
        # E2E-BE-MCQ-AI-ASSIST-ORCH-CONSUME). Dispatch to the binary
        # encoder per topic; fall back to JSON for any topic the writer
        # later grows to publish that isn't Schema-Registry-attached.
        if topic == TOPIC_AI_ASSIST_COMPLETED:
            payload = encode_ai_assist_completed(envelope, body)
        elif topic == TOPIC_AI_ASSIST_REFUSED:
            payload = encode_ai_assist_refused(envelope, body)
        elif topic == TOPIC_AI_ASSIST_PROGRESS:
            payload = encode_ai_assist_progress(envelope, body)
        elif topic == TOPIC_AI_ASSIST_CHUNK_COMPLETED:
            payload = encode_ai_assist_chunk_completed(envelope, body)
        else:
            payload = json.dumps(body, separators=(",", ":")).encode("utf-8")

        params = {
            "id": row_id,
            "workflow_id": assist_id,
            "tenant_id": tenant_id,
            "gcid": gcid,
            "event_type": event_type,
            "topic": topic,
            "payload": payload,
            "envelope": json.dumps(envelope),
            "idempotency_key": idempotency_key,
            "occurred_at": now,
        }

        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        # ADR-251 D4: a TERMINAL event removes the in-flight registry row in
        # the SAME transaction that makes the terminal durable, so "registry
        # row present" always means "job unfinished" to the boot sweep. The
        # chunk_completed / progress lanes are NOT terminal and never delete.
        if self._inflight_registry is not None and topic in (
            TOPIC_AI_ASSIST_COMPLETED,
            TOPIC_AI_ASSIST_REFUSED,
        ):
            await self._inflight_registry.delete_in_tx(assist_id)
        # COMMIT OUR OWN WRITE - "queued" must mean DURABLE.
        #
        # This connection is shared (qgen_crew_wiring) with the outbox
        # DISPATCHER, which runs as a concurrent asyncio task and releases its
        # own read transaction between our INSERT and any later commit. The
        # previous shape relied on someone else's commit ("the connection
        # auto-commits if not in a transaction" - it does not: autocommit is
        # deliberately OFF, see build_qgen_db_conn). On 2026-08-14 the
        # dispatcher's empty-path rollback landed between this INSERT and the
        # next commit and destroyed the terminal completed event for job
        # fd80f4e9; the runner then logged set_completed and the subscriber
        # ACKed, so the event could never be replayed and the job sat at
        # status=running forever. A lane that ACKs must carry its own committer
        # ([[reusable_gotcha_an_ack_is_not_a_commit_lane_without_committer]]).
        #
        # Fail-LOUD: a commit failure raises to the caller, which NACKs, rather
        # than logging a "queued" line for a row that does not exist.
        await self._conn.commit()
        logger.info(
            "qgen_crew_outbox.queued",
            extra={
                "row_id": row_id,
                "assist_id": assist_id,
                "tenant_id": tenant_id,
                "topic": topic,
                "idempotency_key": idempotency_key,
            },
        )
        return row_id


__all__ = [
    "IMDA_DIM_ACCOUNTABILITY",
    "IMDA_DIM_SAFETY_AND_ROBUSTNESS",
    "QGenCrewTerminalOutboxWriter",
    "SCHEMA_VERSION",
    "TOPIC_AI_ASSIST_CHUNK_COMPLETED",
    "TOPIC_AI_ASSIST_COMPLETED",
    "TOPIC_AI_ASSIST_PROGRESS",
    "TOPIC_AI_ASSIST_REFUSED",
]
