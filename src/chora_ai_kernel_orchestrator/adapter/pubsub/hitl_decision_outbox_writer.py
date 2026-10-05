"""HITLDecisionOutboxWriter — transactional-outbox emitter for the
Human-Oversight (HITL) gate event that feeds the O+ Human-Oversight queue.

Root cause this closes: the O+ Human-Oversight queue was always empty
because nothing escalated to a HITL gate. The qgen crew's
MAX-RETRIES-EXHAUSTED terminal (``completed_with_warning`` — a genuinely
low-quality candidate) is the escalation signal; this writer emits the gate
event the chora-governance projector consumes.

Implements the ``_HITLDecisionEmitter`` Protocol from
``orchestrators/qgen_crew_runner.py`` by INSERTing event rows into
``ai_kernel_outbox_events`` (migration ``0003_outbox.sql``) on the topic
``chora.governance.hitl.requested.v1``. The ``OutboxDispatcher`` (separate
asyncio task in ``main.py``) drains the table + publishes to NATS JetStream
via ``NatsPublisher``.

Body shape — the governance projector's ``projector.IncomingEvent`` D4
routing contract (services/chora-governance/internal/domain/projector/
projector.go §routeD4 → AppendHITLDecision + the synchronous
``/v1/governance/evidence/project`` entry which the projector doc note
(projector.go L47-49) says "shares this shape with the Pub/Sub subscriber
wiring"):

    imda_dimension = "fairness_and_human_oversight"   (routes to routeD4)
    event_type contains "hitl"                        (routeD4 → HITL branch)
    decision_id / run_id / agent_id / operator_gcid / hitl_verdict /
    autonomy_level / edit_payload + crew_name

so the gateway ``mapHITLItem`` (services/chora-gateway/internal/adapter/
http/handlers_oplus.go) renders a pending gate from the projected
hitl_decision_log row.

Unlike the AgentDecisionLog writer (whose ``chora.observability.
agent_decision.logged.v1`` topic is Schema-Registry-bound BINARY proto per
ADR-167), the HITL gate topic is NOT yet Schema-Registry-bound — so the
body is JSON. The body is self-describing (carries imda_dimension +
event_type) AND those same routing fields are mirrored onto the envelope so
the Pub/Sub message attributes carry them (NatsPublisher maps
the whole envelope dict to message attributes — see adapter/pubsub/
publisher.py L37).

Why outbox instead of direct publish (mirrors agent_decision_outbox_writer.py):

* **Pod-death survival (D6 P1)** — the LangGraph PostgresSaver checkpoint +
  outbox row write hit the same database (``chora_ai_kernel``). A pod death
  between the terminal emit + the dispatcher's Pub/Sub publish leaves the
  pending row for the dispatcher to drain; direct-publish would drop the gate.
* **Idempotency (D6 P2)** — deterministic ``idempotency_key`` per
  (tenant_id, decision_id). ``ON CONFLICT (idempotency_key) DO NOTHING``
  swallows duplicate INSERTs from a graph re-run (resume re-executes the
  terminal node), so the queue never double-enqueues the same gate.
* **DLQ (D6 P2/P3)** — the existing ``OutboxDispatcher`` writes to
  ``ai_kernel_outbox_dead_letters`` after max_retries; the consumer
  subscription also has a Pub/Sub dead_letter_policy.
* **Multi-tenant isolation (D6 P3)** — ``tenant_id`` lives in both the
  envelope (→ Pub/Sub attribute) AND the payload body.
* **OTel (D6 P4)** — ``traceparent`` + ``tracestate`` land in the envelope
  and propagate as Pub/Sub attributes via the dispatcher.

FLAGS (out of scope here — coordinator/parallel-agent owned):
  * The provisioned Pub/Sub topic ``chora.governance.hitl.requested.v1`` +
    DLQ + a chora-governance subscriber that decodes this body into
    projector.IncomingEvent and calls Project() do NOT yet exist (governance
    + chora-infra + possibly chora-contracts changes). This writer is the
    PRODUCER half; the consumer half lands separately.
  * Persisting a *pending* hitl_decision_log row also depends on the C1
    governance RLS-context fix AND a governance-side pending-enqueue path
    (NewHITLDecision currently requires a resolved approve|reject|edit
    verdict + non-blank operator_gcid; a pending gate carries neither).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Pub/Sub topic for the HITL gate-requested event. Governance-domain topic
# (chora.governance.*) since the gate lives in chora_governance's
# hitl_decision_log. FLAGGED: not yet in chora-infra/topics.yaml or
# terraform — coordinator sequences the topic + subscriber provisioning.
TOPIC_HITL_REQUESTED = "chora.governance.hitl.requested.v1"

# IMDA dimension label per ADR-141 — HITL escalations are D4
# (fairness_and_human_oversight) evidence. The chora-governance projector
# routes by this dimension + an event_type containing "hitl" into
# hitl_decision_log via routeD4 → AppendHITLDecision.
IMDA_DIM_FAIRNESS_HUMAN_OVERSIGHT = "fairness_and_human_oversight"

# Event type — MUST contain "hitl" so the projector's routeD4 switch
# (strings.Contains(et, "hitl")) routes to the HITL branch rather than bias.
_EVENT_TYPE = "governance.hitl.requested"

# Verdict sentinel for a *pending* gate. The gateway's hitlPending query
# treats an empty / "pending" verdict as not-yet-decided (oplus_handlers.go
# §hitlStatusMatches: stored == "" || stored == "pending"). We emit
# "pending" explicitly so a downstream consumer can route it without
# inferring intent from a blank string.
HITL_VERDICT_PENDING = "pending"

# Schema version for the envelope (additive within v1).
SCHEMA_VERSION = "1"

# Outbox INSERT — ON CONFLICT (idempotency_key) DO NOTHING so a graph re-run
# from checkpoint re-emits the gate without raising. UNIQUE index on
# idempotency_key (migration 0003_outbox.sql) makes this exactly-once.
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


class HITLDecisionOutboxWriter:
    """Writes HITL gate-requested events to ``ai_kernel_outbox_events``.

    Implements the ``_HITLDecisionEmitter`` Protocol declared in
    ``qgen_crew_runner.py``. The actual Pub/Sub publish hop is the
    ``OutboxDispatcher``'s concern (separate adapter; existing).

    Connection management: the caller passes a psycopg ``AsyncConnection``
    (or a duck-typed equivalent for tests). The writer does NOT own the
    connection lifecycle — the orchestrator main.py lifespan owns it, same
    as the qgen_crew terminal + agent-decision writer siblings.
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

    # ---- _HITLDecisionEmitter Protocol -------------------------------------

    async def emit(
        self,
        *,
        decision_id: str,
        run_id: str,
        tenant_id: str,
        gcid: str,
        agent_id: str,
        autonomy_level: str,
        summary: str,
        occurred_at: str,
        crew_name: str = "",
        operator_gcid: str = "",
        hitl_verdict: str = HITL_VERDICT_PENDING,
        edit_payload: dict[str, Any] | None = None,
        traceparent: str = "",
        tracestate: str = "",
    ) -> str:
        """Persist a ``chora.governance.hitl.requested.v1`` event row in the
        outbox carrying the D4 routing fields a governance subscriber maps to
        ``projector.IncomingEvent`` for routeD4 → AppendHITLDecision.

        Returns the outbox row id (UUIDv4 — distinct from the deterministic
        idempotency_key the dispatcher dedupes on).

        Idempotency key formula: ``hitl.{tenant_id}.{decision_id}`` — one
        gate per terminal escalation (decision_id == the qgen assist_id /
        invocation correlation key). A graph re-run from checkpoint
        re-executes the terminal node + re-INSERTs, but the unique index
        swallows the duplicate so the Human-Oversight queue is not flooded.
        """
        published_at = _dt.datetime.now(tz=_dt.UTC).isoformat()

        row_id = str(uuid4())
        event_id = str(uuid4())
        idempotency_key = f"hitl.{tenant_id}.{decision_id}"

        # Body = the governance projector.IncomingEvent D4 routing shape. The
        # JSON tags mirror projector.IncomingEvent's snake_case json tags so a
        # governance subscriber can json.Unmarshal it straight into
        # IncomingEvent and call Project(). A *pending* gate carries the
        # pending verdict sentinel + no operator (operator_gcid is set later
        # by the O+ claim/verdict flow, not at escalation time).
        body: dict[str, Any] = {
            "event_id": event_id,
            "tenant_id": tenant_id,
            "imda_dimension": IMDA_DIM_FAIRNESS_HUMAN_OVERSIGHT,
            "lifecycle_stage": "runtime",
            "event_type": _EVENT_TYPE,
            "traceparent": traceparent,
            # D4 hitl_decision_log fields (projector.IncomingEvent §D4).
            "decision_id": decision_id,
            "run_id": run_id,
            "agent_id": agent_id,
            "operator_gcid": operator_gcid,
            "hitl_verdict": hitl_verdict,
            "autonomy_level": autonomy_level,
            "edit_payload": edit_payload or {},
            # mapHITLItem prefers summary, falls back to reason — surface WHY.
            "summary": summary,
            "reason": summary,
            # crew_name lets /o/agents group the gate under the Crews+Agents
            # hierarchy (same convention as the AgentDecisionLog path).
            "crew_name": crew_name,
            "created_at": occurred_at,
        }

        # Envelope = 11 mandatory CLAUDE.md §6 fields + the D4 routing fields
        # mirrored as attributes so a subscriber can filter/route without
        # decoding the body (NatsPublisher maps the whole
        # envelope dict → Pub/Sub message attributes).
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
            "chora_imda_dimension": IMDA_DIM_FAIRNESS_HUMAN_OVERSIGHT,
            "event_type": _EVENT_TYPE,
            "decision_id": decision_id,
        }

        # occurred_at column is TIMESTAMP WITH TIME ZONE — parse the ISO
        # string. A malformed caller value falls back to publish time so the
        # INSERT still lands (forensic continuity: the body keeps the original
        # created_at value).
        try:
            occurred_at_dt = _dt.datetime.fromisoformat(occurred_at)
        except ValueError:
            occurred_at_dt = _dt.datetime.now(tz=_dt.UTC)

        # JSON payload (this topic is NOT Schema-Registry BINARY-bound). The
        # payload BYTEA column accepts UTF-8 JSON bytes; the dispatcher passes
        # row.payload through unchanged as the Pub/Sub message data.
        payload = json.dumps(body)

        params = {
            "id": row_id,
            "workflow_id": run_id,
            "tenant_id": tenant_id,
            "gcid": gcid,
            "event_type": _EVENT_TYPE,
            "topic": TOPIC_HITL_REQUESTED,
            "payload": payload,
            "envelope": json.dumps(envelope),
            "idempotency_key": idempotency_key,
            "occurred_at": occurred_at_dt,
        }

        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        # COMMIT OUR OWN WRITE - "queued" must mean DURABLE (same convention as
        # the agent-decision + qgen terminal writers). The shared connection is
        # drained by a concurrent dispatcher task whose transaction release
        # would otherwise discard this uncommitted row.
        await self._conn.commit()
        logger.info(
            "hitl_decision_outbox.queued",
            extra={
                "row_id": row_id,
                "decision_id": decision_id,
                "run_id": run_id,
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "autonomy_level": autonomy_level,
                "topic": TOPIC_HITL_REQUESTED,
                "idempotency_key": idempotency_key,
                "crew_name": crew_name,
            },
        )
        return row_id


__all__ = [
    "HITL_VERDICT_PENDING",
    "IMDA_DIM_FAIRNESS_HUMAN_OVERSIGHT",
    "SCHEMA_VERSION",
    "TOPIC_HITL_REQUESTED",
    "HITLDecisionOutboxWriter",
]
