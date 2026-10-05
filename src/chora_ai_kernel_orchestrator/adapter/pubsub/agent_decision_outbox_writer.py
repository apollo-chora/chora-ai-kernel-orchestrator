"""AgentDecisionLogOutboxWriter — transactional-outbox emitter for the
qgen 2-agent crew's AgentDecisionLog events (Gate #8).

Implements the ``_AgentDecisionLogEmitter`` Protocol from
``orchestrators/qgen_crew_runner.py`` by INSERTing event rows into
``ai_kernel_outbox_events`` (migration ``0003_outbox.sql``) on the
canonical topic ``chora.observability.agent_decision.logged.v1``. The
``OutboxDispatcher`` (separate asyncio task in ``main.py``) drains the
table + publishes to NATS JetStream via ``NatsPublisher``.

Mirrors the canonical pattern in
``services/chora-ai-kernel-orchestrator/src/chora_ai_kernel_orchestrator/
adapter/pubsub/qgen_crew_publisher.py`` (the qgen terminal writer).

Why outbox instead of direct publish:

* **Pod-death survival (D6 P1)** — the LangGraph PostgresSaver checkpoint
  + outbox row write hit the same database (``chora_ai_kernel``).
  If the pod dies between the terminal emit + the dispatcher's Pub/Sub
  publish, the next pod resumes from the checkpoint AND the pending
  outbox row stays for the dispatcher to drain. Direct-publish would
  silently drop the AccountabilityEvidence event.
* **Idempotency (D6 P2)** — deterministic ``idempotency_key`` per
  (tenant_id, assist_id). ``ON CONFLICT (idempotency_key) DO NOTHING``
  swallows duplicate INSERTs from graph re-runs (resume from checkpoint
  re-executes the terminal node), so the outbox stays exactly-once.
* **DLQ (D6 P2)** — the existing ``OutboxDispatcher`` writes to
  ``ai_kernel_outbox_dead_letters`` after max_retries exhausted; in
  prod the consumer's subscription also has a Pub/Sub dead_letter_policy
  (see ``chora-infra/terraform/modules/m10-pubsub-dlq``).
* **Multi-tenant isolation (D6 P3)** — ``tenant_id`` lives in both the
  envelope (as a Pub/Sub message attribute) AND in the payload body —
  the consumer prefers the attribute but falls back to the payload field
  if attributes get stripped by an intermediate proxy.
* **OTel (D6 P4)** — ``traceparent`` + ``tracestate`` land in the
  envelope and are propagated as Pub/Sub message attributes by the
  dispatcher's ``NatsPublisher``.

Per CLAUDE.md §6 cross-cutting rule, the envelope carries the 11
mandatory fields plus ``chora_imda_dimension`` per ADR-141 ("accountability"
— D1 evidence per the IMDA Model AI Governance Framework).

Topic-existence ground-check per [[feedback-arch-ground-in-deployed-
reality]]: the OE-AI-ASSIST plan §Step 6 named
``chora.ai_kernel.agent_decided.v1`` but that topic does NOT exist in
deployed terraform. The canonical provisioned topic for AgentDecisionLog
events is ``chora.observability.agent_decision.logged.v1`` per
``chora-infra/terraform/modules/m10-data-plane/main.tf:472`` — same
semantic intent (D1 accountability), BigQuery sink + IAM grants already
wired.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any
from uuid import uuid4

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    encode_agent_decision_logged,
)

logger = logging.getLogger(__name__)

# Canonical provisioned topic for AgentDecisionLog events.
# Source: chora-infra/terraform/modules/m10-data-plane/main.tf:472.
TOPIC_AGENT_DECISION_LOGGED = "chora.observability.agent_decision.logged.v1"

# IMDA dimension label per ADR-141 — agent decisions are D1 accountability
# evidence. The chora-governance projector routes by this attribute into
# accountability_evidence (services/chora-governance/internal/domain/
# projector/projector.go §routeD1).
IMDA_DIM_ACCOUNTABILITY = "accountability"

# Schema version — matches
# chora-contracts/proto/events/observability/agent_decision.proto
# §AgentDecisionLogged v1.
SCHEMA_VERSION = "1"

# Event type for this topic. Matches the topic's middle segments so the
# consumer can validate the type without re-parsing the topic string.
_EVENT_TYPE = "observability.agent_decision.logged"

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


class AgentDecisionLogOutboxWriter:
    """Writes AgentDecisionLog events to ``ai_kernel_outbox_events``.

    Implements the ``_AgentDecisionLogEmitter`` Protocol declared in
    ``qgen_crew_runner.py``. The actual Pub/Sub publish hop is the
    ``OutboxDispatcher``'s concern (separate adapter; existing).

    Connection management: the caller passes a psycopg ``AsyncConnection``
    (or a duck-typed equivalent for tests). The writer does NOT own the
    connection lifecycle — the orchestrator main.py lifespan owns it,
    same as the qgen_crew terminal writer sibling.
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

    # ---- Properties exposed for introspection ------------------------------

    @property
    def source_project(self) -> str:
        return self._source_project

    @property
    def source_service(self) -> str:
        return self._source_service

    # ---- _AgentDecisionLogEmitter Protocol ---------------------------------

    async def emit(
        self,
        *,
        assist_id: str,
        agid: str,
        tenant_id: str,
        gcid: str,
        decision: str,
        attempt_count: int,
        max_retries: int,
        critic_notes: str,
        quality_warning: bool,
        chora_imda_dimension: str,
        occurred_at: str,
        question_type: str = "",
        traceparent: str = "",
        tracestate: str = "",
        crew_name: str = "",
        crew_id: str = "",
        is_resume: bool = False,
        is_eval_run: bool = False,
        adapter_version: str = "",
        guardrail_outcome: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        input_hash: str = "",
        output_hash: str = "",
        prompt_conditions: dict[str, str] | None = None,
    ) -> str:
        """Persist a chora.observability.agent_decision.logged.v1 event
        row in the outbox.

        Returns the outbox row id (UUIDv4 — distinct from the
        deterministic idempotency_key). The dispatcher uses the id for
        per-worker SKIP LOCKED selection.

        Idempotency key formula:
        ``agent_decision.{tenant_id}.{assist_id}.{agid}`` — one event per
        (terminal run, agent). The ``agid`` suffix is load-bearing: a single
        generation emits a qgen_question decision AND a qgen_critic decision
        (and OE grading emits oe_evaluator + oe_moderator), so without the
        per-agent suffix the second row would collide on the unique index and
        ``ON CONFLICT DO NOTHING`` would silently drop it. A graph re-run from
        PostgresSaver checkpoint re-INSERTs the same (run, agent) key, which the
        unique index swallows.

        Extension fields (2026-05-26 — per atomic-napping-spring O+ hydration
        plan + chora-contracts proto/events/observability/agent_decision.proto
        fields 12-20):

          - ``crew_name`` — feeds /o/agents Crews+Agents hierarchy.
          - ``crew_id`` — UUIDv7 of the crew instance / orchestration.
          - ``is_resume`` / ``is_eval_run`` — orchestration-state flags.
          - ``adapter_version`` — LoRA adapter (empty when base model).
          - ``guardrail_outcome`` — Cloud Model Armor verdict.
          - ``prompt_tokens`` / ``completion_tokens`` / ``cached_tokens``
            — gen_ai.usage.* counts.

        ADR-197 M-A.3 field:

          - ``prompt_conditions`` — the prompt-shaping discriminants the
            orchestrator holds (e.g. ``{"intent": "new_question",
            "question_type": "oe"}``). Threaded into the body so the encoder
            rides each entry in the field-21 attributes map under
            ``prompt_conditions.<key>``. ``None`` ⇒ no condition attributes.
        """
        # agid is load-bearing — it routes the /o/agents tile + suffixes the
        # idempotency key. Refuse a blank agid loudly rather than landing an
        # unattributed row (the legacy hardcoded "qgen_crew" is exactly the
        # bug we are closing: it matched no registry tile).
        agid = (agid or "").strip()
        if not agid:
            raise ValueError("agid required (per-agent registry id)")

        # Coerce blank dimension to the canonical accountability label so
        # the D1 evidence row is never blank in the projector.
        dimension = (chora_imda_dimension or "").strip()
        if not dimension:
            dimension = IMDA_DIM_ACCOUNTABILITY

        # NOTE: the writer does NOT re-stamp occurred_at — the caller
        # (runner._emit_agent_decision_log) stamps it at emit-call time
        # and BigQuery uses this as the partition key. Re-stamping here
        # would push the row into the wrong partition by the dispatcher's
        # drain delay.
        published_at = _dt.datetime.now(tz=_dt.UTC).isoformat()

        # Outbox row id — UUIDv4 for the row (NOT the event_id). The
        # event_id lives in the envelope and is what subscribers see.
        row_id = str(uuid4())
        event_id = str(uuid4())
        idempotency_key = f"agent_decision.{tenant_id}.{assist_id}.{agid}"

        # Body matches the AgentDecisionLogPayload struct in
        # services/chora-governance/internal/adapter/events/
        # agent_decision_consumer.go §AgentDecisionLogPayload. The
        # extension fields (crew_name through cached_tokens) are
        # additive per chora-contracts/proto/events/observability/
        # agent_decision.proto fields 12-20 — additive within v1.
        body: dict[str, Any] = {
            "assist_id": assist_id,
            # agid → proto field 4 (the registry agent id the /o/agents tile
            # keys on); question_type → attributes["question_type"] ("mcq"|"oe").
            "agid": agid,
            "question_type": question_type,
            "tenant_id": tenant_id,
            "gcid": gcid,
            "decision": decision,
            "attempt_count": attempt_count,
            "max_retries": max_retries,
            "critic_notes": critic_notes,
            "quality_warning": quality_warning,
            "chora_imda_dimension": dimension,
            "occurred_at": occurred_at,
            "traceparent": traceparent,
            "tracestate": tracestate,
            "crew_name": crew_name,
            "crew_id": crew_id,
            "is_resume": is_resume,
            "is_eval_run": is_eval_run,
            "adapter_version": adapter_version,
            "guardrail_outcome": guardrail_outcome,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            # Citation hashes (sha256 hex over the agent's input/output) — the
            # encoder rides them in the proto attributes map; the O+ reasoning
            # panel renders them as a PII-safe citation. Empty → encoder omits
            # the attribute (consumer keeps the zero sentinel).
            "input_hash": input_hash,
            "output_hash": output_hash,
            # ADR-197 M-A.3 — prompt-shaping condition discriminants (the runner
            # mirrors the Go *Conditions extractors, omitting blanks/false). The
            # encoder rides each entry in the field-21 attributes map under
            # ``prompt_conditions.<key>``; an empty map emits nothing.
            "prompt_conditions": prompt_conditions or {},
        }

        # Envelope = 11 mandatory CLAUDE.md §6 fields + IMDA dimension +
        # the decision attr (consumer's AgentDecisionEnvelopeAttrs prefers
        # attribute values for envelope-mandatory fields).
        envelope: dict[str, str] = {
            "event_id": event_id,
            "idempotency_key": idempotency_key,
            "tenant_id": tenant_id,
            "gcid": gcid,
            "occurred_at": occurred_at,
            "published_at": published_at,
            "traceparent": traceparent,
            "tracestate": tracestate,
            "source_project": self._source_project,
            "source_service": self._source_service,
            "schema_version": SCHEMA_VERSION,
            "chora_imda_dimension": dimension,
            # Extra envelope attrs flow through to Pub/Sub message
            # attributes via the OutboxDispatcher →
            # NatsPublisher hop. D1 governance subscribers
            # can dedupe + index by decision + crew without decoding
            # the payload.
            "decision": decision,
            "crew_name": crew_name,
        }

        # occurred_at in the SQL row goes in as the parsed datetime, not
        # the ISO string — the column is TIMESTAMP WITH TIME ZONE. Compute it
        # FIRST so a malformed caller value resolves to a usable timestamp
        # for BOTH the SQL row AND the proto payload (the BINARY proto cannot
        # carry an un-parseable Timestamp).
        try:
            occurred_at_dt = _dt.datetime.fromisoformat(occurred_at)
        except ValueError:
            # Caller bug — runner always stamps a valid ISO. Fall back
            # to publish time so the INSERT still lands; the JSON envelope
            # column keeps the (malformed) caller value for forensic
            # continuity, but the proto payload uses the sanitised value.
            occurred_at_dt = _dt.datetime.now(tz=_dt.UTC)

        # ADR-167: the topic is Schema-Registry-bound with encoding=BINARY.
        # Serialise the body to canonical proto3 wire bytes
        # (chora.observability.v1.AgentDecisionLogged) — JSON publishes were
        # rejected with INVALID_BINARY_PROTO_MESSAGE at the publish hop. The
        # encoder reads the envelope for field-1 EventEnvelope + maps the qgen
        # body per the canonical mapping (qgen verdict + counts ride the
        # attributes map). decided_at == occurred_at — pass the SANITISED ISO
        # so a malformed caller value still yields a valid proto Timestamp.
        # The dispatcher passes the bytes through unchanged.
        proto_envelope = dict(envelope)
        proto_envelope["occurred_at"] = occurred_at_dt.isoformat()
        payload = encode_agent_decision_logged(proto_envelope, body)

        params = {
            "id": row_id,
            "workflow_id": assist_id,
            "tenant_id": tenant_id,
            "gcid": gcid,
            "event_type": _EVENT_TYPE,
            "topic": TOPIC_AGENT_DECISION_LOGGED,
            "payload": payload,
            "envelope": json.dumps(envelope),
            "idempotency_key": idempotency_key,
            "occurred_at": occurred_at_dt,
        }

        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        # COMMIT OUR OWN WRITE - "queued" must mean DURABLE. Same convention as
        # QGenCrewTerminalOutboxWriter: this connection is SHARED with the
        # concurrently-draining outbox dispatcher, so a row left uncommitted can
        # be discarded by the dispatcher's transaction release before anything
        # commits it (live loss 2026-08-14, job fd80f4e9).
        await self._conn.commit()
        logger.info(
            "agent_decision_outbox.queued",
            extra={
                "row_id": row_id,
                "assist_id": assist_id,
                "agid": agid,
                "tenant_id": tenant_id,
                "decision": decision,
                "topic": TOPIC_AGENT_DECISION_LOGGED,
                "idempotency_key": idempotency_key,
                "crew_name": crew_name,
                "crew_id": crew_id,
                "is_eval_run": is_eval_run,
            },
        )
        return row_id


__all__ = [
    "AgentDecisionLogOutboxWriter",
    "IMDA_DIM_ACCOUNTABILITY",
    "SCHEMA_VERSION",
    "TOPIC_AGENT_DECISION_LOGGED",
]
