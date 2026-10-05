"""Agent-dispatch wire contract (ADR-253 D1/D4).

The orchestrator-to-agent hop is a Pub/Sub **request and completion pair per
agent role**. This module owns the three things both sides of that wire must
agree on, and nothing else:

  * ``DISPATCH_INTERRUPT_KEY`` — the discriminator that tells an agent-dispatch
    park apart from a human-in-the-loop park. The growth-edge crew already parks
    on ``interrupt()`` for human review (ADR-205 WS-2), and a saver that
    mistook one for the other would publish a request no agent can answer. The
    key is checked, never inferred from shape.
  * the topic names, derived from the agent role so the orchestrator couples to
    a topic name rather than to a host, a port and a REST dialect (ADR-253 1.3.4).
  * the mandatory envelope (CLAUDE.md), including the traceparent, which is
    load-bearing here: it rides the HTTP session state today, so on Pub/Sub it
    must ride BOTH the envelope and the message attributes or one run stops
    being one trace (ADR-253 D4).

⚠ Deliberately NOT here: publishing. The dispatch request reaches its topic as
an ``ai_kernel_outbox_events`` row written in the same transaction as the park
(ADR-253 D3a), so there is exactly one publishing path and the invocation leaves
a durable pre-publish record (ADR-253 1.3.3).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

from langgraph.errors import GraphBubbleUp

# The key an agent-dispatch interrupt payload is wrapped in. A park whose value
# does not carry this key is somebody else's interrupt and is left alone.
DISPATCH_INTERRUPT_KEY = "__chora_agent_dispatch__"

# chora.{domain}.{aggregate}.{event_type}.v{N} — domain ai_kernel, aggregate
# agent_dispatch, one pair per role per ADR-253 D4.
_TOPIC_PREFIX = "chora.ai_kernel.agent_dispatch"
SCHEMA_VERSION = 1

# IMDA D1: an agent invocation is accountability evidence (ADR-141 labels).
IMDA_DIMENSION_ACCOUNTABILITY = "accountability"

# Completion status discriminator. ADR-253 D4 carries the outcome INSIDE the
# completion payload rather than on separate success/failure topics.
STATUS_OK = "OK"
STATUS_FAILED = "FAILED"


def request_topic(agent_role: str) -> str:
    """Topic an ``agent_role`` dispatch request is published to."""
    role = _require(agent_role, "agent_role")
    return f"{_TOPIC_PREFIX}.{role}_requested.v{SCHEMA_VERSION}"


def completion_topic(agent_role: str) -> str:
    """Topic the agent publishes its answer to."""
    role = _require(agent_role, "agent_role")
    return f"{_TOPIC_PREFIX}.{role}_completed.v{SCHEMA_VERSION}"


def dispatch_idempotency_key(*, agent_role: str, execution_id: str) -> str:
    """Deterministic key for one dispatch.

    ``execution_id`` is already unique per hop within a run (the crew nodes
    build it from submission / question / attempt), so this is stable across a
    redelivery AND across a node re-execution — which matters because LangGraph
    re-executes a node from the top when a park resumes.
    """
    return f"agent_dispatch.{_require(agent_role, 'agent_role')}.{_require(execution_id, 'execution_id')}"


def _require_uuid(value: str, field: str) -> str:
    """Refuse a non-UUID where the outbox column is ``UUID NOT NULL``.

    ``migrations/0003_outbox.sql:39`` types both ``workflow_id`` and ``gcid`` as
    UUID. A bad value fails at the INSERT, inside the same transaction as the
    park, so the dispatch never publishes and the run parks forever, with
    nothing in the logs naming the cause. Refusing at the request turns a silent
    lane into a loud, located error.
    """
    value = _require(value, field)
    try:
        UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(
            f"{field}={value!r} is not a UUID, but ai_kernel_outbox_events."
            f"{field} is UUID NOT NULL. A non-UUID here fails at the INSERT "
            f"inside the park transaction, so the dispatch would never publish "
            f"and the run would never resume. Pass the aggregate root's id "
            f"explicitly (the growth-edge lane uses upload_id)."
        ) from exc
    return value


def build_dispatch_request(
    *,
    agent_role: str,
    execution_id: str,
    tenant_id: str,
    gcid: str,
    thread_id: str,
    input_payload: str,
    workflow_id: str = "",
    prompt_template_id: str = "",
    agid: str = "",
    traceparent: str = "",
    tracestate: str = "",
    source_project: str = "",
    source_service: str = "chora-ai-kernel-orchestrator",
) -> dict[str, Any]:
    """Build one dispatch request: mandatory envelope + body + routing.

    Fails loud on a missing ``tenant_id`` or ``thread_id``. A dispatch with no
    tenant cannot be screened, metered or resumed, and a never-fed lane meets
    FORCE RLS on its first delivery — so the tenant is stamped at the DISPATCH,
    not left for the receiver to guess.

    ``workflow_id`` keys the outbox row; ``thread_id`` keys the CHECKPOINT. They
    are the same string on the OE lane only by accident of naming (thread_id =
    submission_id, already a UUID), so it defaults to ``thread_id`` and every
    existing caller is unaffected. A lane whose thread key is not a UUID (the
    growth-edge crew's is ``{tenant_id}:{upload_id}``) MUST pass its own.
    """
    tenant_id = _require(tenant_id, "tenant_id")
    thread_id = _require(thread_id, "thread_id")
    workflow_id = _require_uuid(workflow_id.strip() or thread_id, "workflow_id")
    execution_id = _require(execution_id, "execution_id")
    agent_role = _require(agent_role, "agent_role")

    now = _dt.datetime.now(tz=_dt.UTC)
    idempotency_key = dispatch_idempotency_key(agent_role=agent_role, execution_id=execution_id)
    envelope: dict[str, Any] = {
        "event_id": str(uuid4()),
        "idempotency_key": idempotency_key,
        "tenant_id": tenant_id,
        "gcid": gcid,
        "occurred_at": now.isoformat(),
        "published_at": now.isoformat(),
        "traceparent": traceparent,
        "tracestate": tracestate,
        "source_project": source_project,
        "source_service": source_service,
        "schema_version": SCHEMA_VERSION,
        "chora_imda_dimension": IMDA_DIMENSION_ACCOUNTABILITY,
        # The publisher client reserves the attribute name "topic"; the
        # platform's producers carry routing under "event_topic".
        "event_topic": request_topic(agent_role),
    }
    body: dict[str, Any] = {
        "agent_role": agent_role,
        "execution_id": execution_id,
        "tenant_id": tenant_id,
        "gcid": gcid,
        # The agent echoes these two back untouched. They are how the completion
        # consumer finds the parked run without keeping any state of its own.
        "thread_id": thread_id,
        "idempotency_key": idempotency_key,
        "agid": agid,
        "prompt_template_id": prompt_template_id,
        "input_payload": input_payload,
        "reply_topic": completion_topic(agent_role),
        "traceparent": traceparent,
        "tracestate": tracestate,
        "requested_at": now.isoformat(),
    }
    return {
        "topic": request_topic(agent_role),
        "event_type": f"ai_kernel.agent_dispatch.{agent_role}_requested",
        "envelope": envelope,
        "body": body,
        "idempotency_key": idempotency_key,
        "tenant_id": tenant_id,
        "gcid": gcid,
        "workflow_id": workflow_id,
    }


def wrap_for_interrupt(request: dict[str, Any]) -> dict[str, Any]:
    """Wrap a dispatch request as an ``interrupt()`` value.

    The wrapper is what makes an agent-dispatch park self-identifying to the
    saver; see ``DISPATCH_INTERRUPT_KEY``.
    """
    return {DISPATCH_INTERRUPT_KEY: request}


def reraise_if_dispatch_park(exc: BaseException) -> None:
    """Re-raise a LangGraph park so a best-effort ``except`` cannot swallow it.

    Call this as the FIRST statement of any ``except Exception`` that wraps an
    agent dispatch.

    Why it has to exist. Under the HTTP transport of ADR-169 ``execute`` returned
    a response, so a best-effort guard around a non-critical hop was correct: a
    failed hop should not sink the whole run. Under ADR-253 the same call no
    longer returns: it PARKS the run via ``interrupt()``, and LangGraph signals
    a park by raising ``GraphInterrupt``, which subclasses
    ``GraphBubbleUp(Exception)``. So every pre-existing best-effort guard
    silently became a park-eater: the interrupt never reaches LangGraph, the
    dispatch outbox row written in the same transaction as the park (D3a) rolls
    back with it, and the graph runs on to its terminal as though the hop had
    merely failed.

    That is not hypothetical. It is how the whole-assessment overall comment was
    lost on the live OE lane: only ``assess_summary_node`` carried such a guard,
    and zero ``:summary`` dispatch rows were ever written while per-question rows
    were fine.

    A park is control flow, not an error. Anything else is returned to the caller
    untouched, so genuine transport failures keep their existing behaviour.
    """
    if isinstance(exc, GraphBubbleUp):
        raise exc


def extract_dispatch_requests(
    writes: Sequence[tuple[str, Any]],
) -> list[dict[str, Any]]:
    """Pull every agent-dispatch request out of a LangGraph ``put_writes`` batch.

    Measured, not assumed (spike 2026-08-20): a park is recorded as a
    ``put_writes`` on channel ``__interrupt__`` whose value is a tuple of
    ``Interrupt`` objects — NOT as a ``put``. That write is the durable fact
    "this thread is parked awaiting this agent", which is why it is the write
    the dispatch row must commit with.

    Anything that is not an agent-dispatch interrupt yields nothing. No shape
    sniffing: the discriminator key is required.
    """
    found: list[dict[str, Any]] = []
    for channel, value in writes or ():
        if channel != "__interrupt__":
            continue
        items = value if isinstance(value, (tuple, list)) else (value,)
        for item in items:
            payload = getattr(item, "value", None)
            if not isinstance(payload, dict):
                continue
            request = payload.get(DISPATCH_INTERRUPT_KEY)
            if isinstance(request, dict):
                found.append(request)
    return found


def _require(value: str, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"agent dispatch: {field} is required")
    return text


__all__ = [
    "DISPATCH_INTERRUPT_KEY",
    "IMDA_DIMENSION_ACCOUNTABILITY",
    "SCHEMA_VERSION",
    "STATUS_FAILED",
    "STATUS_OK",
    "build_dispatch_request",
    "completion_topic",
    "dispatch_idempotency_key",
    "extract_dispatch_requests",
    "request_topic",
    "wrap_for_interrupt",
]
