"""Generic single-agent workflow (ADR-254 D5): ONE code path for every lane that
consumes a request, validates + stamps it, parks on ONE agent dispatch, maps the
completion and emits a result through the outbox.

Today it serves the two fog folds (``kg_exploration_workflow`` on
``concept_suggestion.requested.v1`` -> ``kg_explore``; ``companion_reflection_
workflow`` on ``goal_knowledge.synthesis_requested.v1`` -> ``companion_chat``
``turn_kind=reflect``); the five single-agent crews (recommend, moderate,
duel_atoms, profile_conjure, typed companion_chat) are further ``LaneContract``
instances. One engine, one test suite, one AST park guard.

A lane is a ``LaneContract``: pure functions the engine calls at three points.

  * ``decode(body, attrs)`` -> the initial state (MUST carry ``event_id``,
    ``tenant_id``, ``gcid``, ``traceparent``, ``tracestate``, ``idempotency_key``
    plus whatever the lane needs), or ``Skipped(reason)`` for a request the lane
    answers by doing nothing (ACKed, nothing published), or RAISE for a request
    the producer got wrong (NACKed -> redelivered -> the caller's DLQ).
  * ``dispatch(state)`` -> ``DispatchSpec`` (execution id, ``input_payload``
    object, the UUID ``workflow_id`` the outbox keys on).
  * ``result(state, completion)`` -> ``ResultEvent`` (emitted through the outbox
    with the mandatory envelope, idempotent on its key) or ``None`` (the lane
    decided, loudly, that nothing is published: a contract violation by the
    agent, an unattributable decline; the caller's claim TTL closes the loop).
  * optional ``rejected(state)`` + ``tenant_inflight_cap``: the per-tenant
    backpressure of ADR-254 D5 (closes ADR-253 open question 1). When the park
    ledger already holds ``cap`` parked runs for (tenant, role) the request is
    answered with the lane's REJECTED result and never dispatched.

The engine owns: the park (``PubSubAgentExecutor.execute`` inside a LangGraph
node, so the D3a saver commits park + dispatch row + ledger row together), the
``reraise_if_dispatch_park`` discipline, the idempotent emit (``result_key`` in
state; the outbox INSERT is ON CONFLICT DO NOTHING besides), the inbox dedupe
keyed on the request's business key, ack-after-park / NACK-on-raise.

Thread id = ``{lane}:{event_id}`` (the request's event id is the run identity);
``workflow_id`` must be a UUID (``ai_kernel_outbox_events.workflow_id``), so a
lane whose event id is not one derives a stable UUIDv5 from it.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    DISPATCH_INTERRUPT_KEY,
    reraise_if_dispatch_park,
)

logger = logging.getLogger(__name__)

STATUS_OK = "OK"
STATUS_FAILED = "FAILED"

OUTCOME_PARKED = "parked"
OUTCOME_COMPLETED = "completed"
OUTCOME_SKIPPED = "skipped"
OUTCOME_REJECTED = "rejected"

DEFAULT_INBOX_TTL = _dt.timedelta(days=7)
SOURCE_SERVICE = "chora-ai-kernel-orchestrator"

# A stable namespace for deriving a UUID workflow id from a non-UUID event id.
_WORKFLOW_NS = uuid.UUID("6b1f2d3e-4a5b-4c6d-8e7f-90a1b2c3d4e5")


# --------------------------------------------------------------------------- #
# contract types
# --------------------------------------------------------------------------- #


class Skipped:
    """``decode`` returns this for a request the lane answers by doing nothing."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Skipped({self.reason!r})"


@dataclass(frozen=True)
class DispatchSpec:
    """What the lane asks the agent: one park."""

    execution_id: str
    input_payload: dict[str, Any]
    workflow_id: str
    agid: str = ""
    prompt_template_id: str = ""


@dataclass(frozen=True)
class ResultEvent:
    """One outbox row: the lane's result on its caller-facing topic."""

    topic: str
    event_type: str
    idempotency_key: str
    body: dict[str, Any]
    tenant_id: str
    gcid: str
    workflow_id: str
    traceparent: str = ""
    tracestate: str = ""
    imda_dimension: str = "transparency"
    schema_version: int = 1

    def to_outbox_request(
        self, *, source_project: str, source_service: str = SOURCE_SERVICE, now: _dt.datetime | None = None
    ) -> dict[str, Any]:
        """The dict ``AgentDispatchOutboxWriter.queue_request`` persists: the
        mandatory envelope (CLAUDE.md) in ``envelope``, the JSON body, routing
        on ``event_topic`` (the publisher reserves ``topic``)."""
        ts = (now or _dt.datetime.now(tz=_dt.UTC)).isoformat()
        event_id = str(uuid.uuid4())
        envelope = {
            "event_id": event_id,
            "idempotency_key": self.idempotency_key,
            "tenant_id": self.tenant_id,
            "gcid": self.gcid,
            "occurred_at": ts,
            "published_at": ts,
            "traceparent": self.traceparent or synthetic_traceparent(event_id),
            "tracestate": self.tracestate,
            "source_project": source_project,
            "source_service": source_service,
            "schema_version": self.schema_version,
            "chora_imda_dimension": self.imda_dimension,
            "event_topic": self.topic,
        }
        return {
            "topic": self.topic,
            "event_type": self.event_type,
            "envelope": envelope,
            "body": dict(self.body),
            "idempotency_key": self.idempotency_key,
            "tenant_id": self.tenant_id,
            "gcid": self.gcid,
            "workflow_id": workflow_uuid(self.workflow_id),
        }


@dataclass(frozen=True)
class LaneContract:
    """One lane = one role + three pure functions (+ optional backpressure)."""

    name: str
    role: str
    request_kind: str
    decode: Callable[[dict[str, Any], dict[str, str]], dict[str, Any] | Skipped]
    dispatch: Callable[[dict[str, Any]], DispatchSpec]
    result: Callable[[dict[str, Any], dict[str, Any]], ResultEvent | None]
    rejected: Callable[[dict[str, Any]], ResultEvent | None] | None = None
    tenant_inflight_cap: int | None = None

    def __post_init__(self) -> None:
        for attr in ("name", "role", "request_kind"):
            if not (getattr(self, attr) or "").strip():
                raise ValueError(f"LaneContract: {attr} is required")
        if self.tenant_inflight_cap is not None:
            if self.tenant_inflight_cap < 1:
                raise ValueError("LaneContract: tenant_inflight_cap must be >= 1")
            if self.rejected is None:
                raise ValueError(
                    f"LaneContract {self.name!r}: a tenant_inflight_cap needs a rejected() "
                    "result, or a capped request would be dropped in silence"
                )


def synthetic_traceparent(event_id: str) -> str:
    """A valid W3C traceparent derived from an event id, for a request that
    arrived without one (consumption's validator refuses an empty field)."""
    hexid = (event_id or "").replace("-", "")
    trace_id = (hexid + "0" * 32)[:32]
    parent_id = (hexid + "0" * 16)[:16]
    return f"00-{trace_id}-{parent_id}-01"


def envelope_scope(body: dict[str, Any], attrs: dict[str, str], *, gcid_body_key: str) -> dict[str, str]:
    """The mandatory envelope fields a lane's ``decode`` must return, read from
    the Pub/Sub ATTRIBUTES with the body as fallback.

    Attributes win because that is what the broker carried and what routing and
    the DLQ stamps agree with; a body copy can be stale. Fields the publisher
    omits when empty (``tracestate``) read as "" rather than raising, so an
    ordinary request is not treated as a decorated one.
    """
    return {
        "event_id": (attrs.get("event_id") or str(body.get("event_id", "") or "")).strip(),
        "idempotency_key": (attrs.get("idempotency_key") or "").strip(),
        "tenant_id": (attrs.get("tenant_id") or str(body.get("tenant_id", "") or "")).strip(),
        "gcid": (attrs.get("gcid") or str(body.get(gcid_body_key, "") or "")).strip(),
        "traceparent": (attrs.get("traceparent") or str(body.get("traceparent", "") or "")).strip(),
        "tracestate": (attrs.get("tracestate") or str(body.get("tracestate", "") or "")).strip(),
    }


def loads_json_object(text: str) -> dict[str, Any] | None:
    """Parse an agent's answer as a JSON object, tolerating the ```json fence
    the model adds anyway. ``None`` for anything that is not an object, so the
    caller decides loudly what a contract violation means for its lane."""
    s = (text or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1 :] if nl != -1 else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    try:
        parsed = json.loads(s)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def workflow_uuid(value: str) -> str:
    """``ai_kernel_outbox_events.workflow_id`` is UUID NOT NULL: pass a UUID
    through, derive a stable UUIDv5 from anything else (never guess blindly)."""
    text = (value or "").strip()
    if not text:
        raise ValueError("workflow id is required")
    try:
        return str(uuid.UUID(text))
    except ValueError:
        derived = str(uuid.uuid5(_WORKFLOW_NS, text))
        logger.info("single_agent_workflow.workflow_id_derived", extra={"from": text[:64], "uuid": derived})
        return derived


# --------------------------------------------------------------------------- #
# ports
# --------------------------------------------------------------------------- #


class _ExecutorLike(Protocol):
    async def execute(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        agid: str,
        agent_role: str,
        input_payload: str,
        workflow_id: str = ...,
        prompt_template_id: str = ...,
    ) -> Any: ...


class _OutboxWriterLike(Protocol):
    async def queue_request(self, request: dict[str, Any]) -> str: ...


class _LedgerLike(Protocol):
    async def count_parked(self, *, tenant_id: str, agent_role: str) -> int: ...


# --------------------------------------------------------------------------- #
# state + nodes
# --------------------------------------------------------------------------- #


class SingleAgentState(TypedDict, total=False):
    event_id: str
    idempotency_key: str
    tenant_id: str
    gcid: str
    traceparent: str
    tracestate: str
    request: dict[str, Any]
    completion: dict[str, Any]
    result_key: str
    trace: list[dict[str, Any]]


def _trace(state: dict[str, Any], node: str, **fields: Any) -> list[dict[str, Any]]:
    return [*(state.get("trace") or []), {"node": node, **fields}]


async def dispatch_node(state: dict[str, Any], *, contract: LaneContract, executor: _ExecutorLike) -> dict[str, Any]:
    """ONE park. A FAILED completion (the executor raises) becomes a FAILED
    ``completion`` in state for the contract to map; a park propagates."""
    spec = contract.dispatch(state)
    try:
        response = await executor.execute(
            execution_id=spec.execution_id,
            tenant_id=state["tenant_id"],
            agid=spec.agid,
            agent_role=contract.role,
            input_payload=json.dumps(spec.input_payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False),
            workflow_id=workflow_uuid(spec.workflow_id),
            prompt_template_id=spec.prompt_template_id,
        )
    except Exception as exc:  # noqa: BLE001
        # A PARK IS NOT A FAILURE (ADR-253): re-raise it FIRST.
        reraise_if_dispatch_park(exc)
        logger.error(
            "single_agent_workflow.dispatch_failed",
            extra={
                "lane": contract.name,
                "role": contract.role,
                "event_id": state.get("event_id", ""),
                "err": f"{exc.__class__.__name__}: {exc}",
            },
        )
        return {
            "completion": {"status": STATUS_FAILED, "output_payload": "", "error_message": str(exc)},
            "trace": _trace(state, "dispatch", status=STATUS_FAILED),
        }
    return {
        "completion": {
            "status": STATUS_OK,
            "output_payload": str(getattr(response, "output_payload", "") or ""),
            "error_message": "",
        },
        "trace": _trace(state, "dispatch", status=STATUS_OK),
    }


async def emit_node(
    state: dict[str, Any], *, contract: LaneContract, outbox_writer: _OutboxWriterLike, source_project: str
) -> dict[str, Any]:
    """Map the completion to the lane's result and queue it, idempotently."""
    if state.get("result_key"):
        return {"trace": _trace(state, "emit", skipped="already_emitted")}
    event = contract.result(state, dict(state.get("completion") or {}))
    if event is None:
        logger.warning(
            "single_agent_workflow.no_result_event",
            extra={
                "lane": contract.name,
                "event_id": state.get("event_id", ""),
                "completion_status": (state.get("completion") or {}).get("status", ""),
            },
        )
        return {"trace": _trace(state, "emit", emitted=False)}
    row_id = await outbox_writer.queue_request(event.to_outbox_request(source_project=source_project))
    return {
        "result_key": event.idempotency_key,
        "trace": _trace(state, "emit", emitted=True, row_id=row_id, topic=event.topic),
    }


def build_single_agent_graph(
    *,
    contract: LaneContract,
    executor: _ExecutorLike,
    outbox_writer: _OutboxWriterLike,
    source_project: str,
    checkpointer: Any | None = None,
) -> Any:
    sg: StateGraph = StateGraph(SingleAgentState)
    sg.add_node("dispatch", partial(dispatch_node, contract=contract, executor=executor))
    sg.add_node(
        "emit", partial(emit_node, contract=contract, outbox_writer=outbox_writer, source_project=source_project)
    )
    sg.add_edge(START, "dispatch")
    sg.add_edge("dispatch", "emit")
    sg.add_edge("emit", END)
    compiled = sg.compile(checkpointer=checkpointer) if checkpointer is not None else sg.compile()
    return compiled.with_config({"recursion_limit": 10})


# --------------------------------------------------------------------------- #
# runner + subscriber
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RequestOutcome:
    outcome: str
    thread_id: str = ""
    detail: str = ""
    result_key: str = ""


class SingleAgentWorkflowRunner:
    """Drives one lane's graph: start a run (to its park), resume on completion."""

    def __init__(
        self,
        *,
        graph: Any,
        contract: LaneContract,
        outbox_writer: _OutboxWriterLike,
        source_project: str,
        ledger: _LedgerLike | None = None,
    ) -> None:
        self._graph = graph
        self._contract = contract
        self._outbox = outbox_writer
        self._source_project = source_project
        self._ledger = ledger
        if contract.tenant_inflight_cap is not None and ledger is None:
            raise ValueError(
                f"SingleAgentWorkflowRunner {contract.name!r}: a tenant_inflight_cap needs the park ledger"
            )

    @property
    def contract(self) -> LaneContract:
        return self._contract

    async def handle_request(self, body: dict[str, Any], attrs: dict[str, str]) -> RequestOutcome:
        decoded = self._contract.decode(dict(body or {}), dict(attrs or {}))
        if isinstance(decoded, Skipped):
            logger.info("single_agent_workflow.skipped", extra={"lane": self._contract.name, "reason": decoded.reason})
            return RequestOutcome(OUTCOME_SKIPPED, detail=decoded.reason)
        state = dict(decoded)
        for required in ("event_id", "tenant_id", "gcid"):
            if not str(state.get(required, "") or "").strip():
                raise ValueError(f"single_agent_workflow {self._contract.name!r}: decode() returned no {required}")
        thread_id = f"{self._contract.name}:{state['event_id']}"
        cap = self._contract.tenant_inflight_cap
        if cap is not None and self._ledger is not None:
            parked = await self._ledger.count_parked(tenant_id=state["tenant_id"], agent_role=self._contract.role)
            if parked >= cap:
                event = self._contract.rejected(state) if self._contract.rejected else None
                if event is not None:
                    await self._outbox.queue_request(event.to_outbox_request(source_project=self._source_project))
                logger.warning(
                    "single_agent_workflow.rejected_tenant_in_flight_cap",
                    extra={"lane": self._contract.name, "tenant_id": state["tenant_id"], "parked": parked, "cap": cap},
                )
                return RequestOutcome(
                    OUTCOME_REJECTED,
                    thread_id=thread_id,
                    detail="tenant_in_flight_cap",
                    result_key=event.idempotency_key if event else "",
                )
        terminal = await self._graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}})
        return self._outcome(terminal, thread_id)

    async def handle_completion(self, completion: dict[str, Any]) -> RequestOutcome:
        thread_id = str(completion.get("thread_id") or "").strip()
        if not thread_id:
            raise ValueError(
                f"single_agent_workflow {self._contract.name!r}: completion carries no thread_id; "
                "refusing to resume a guessed thread"
            )
        terminal = await self._graph.ainvoke(
            Command(resume=dict(completion)), config={"configurable": {"thread_id": thread_id}}
        )
        return self._outcome(terminal, thread_id)

    @staticmethod
    def _outcome(terminal: dict[str, Any], thread_id: str) -> RequestOutcome:
        interrupts = terminal.get("__interrupt__") or []
        for item in interrupts:
            value = getattr(item, "value", None)
            if isinstance(value, dict) and DISPATCH_INTERRUPT_KEY in value:
                return RequestOutcome(OUTCOME_PARKED, thread_id=thread_id)
        if interrupts:  # pragma: no cover - this graph has no human gate
            return RequestOutcome(OUTCOME_PARKED, thread_id=thread_id, detail="interrupt")
        return RequestOutcome(OUTCOME_COMPLETED, thread_id=thread_id, result_key=str(terminal.get("result_key") or ""))


class _InboxLike(Protocol):
    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool: ...


class _MessageLike(Protocol):
    data: bytes
    attributes: dict[str, str]

    def ack(self) -> None: ...
    def nack(self) -> None: ...


class SingleAgentRequestSubscriber:
    """Pub/Sub handler for a lane's request subscription: JSON body + envelope
    attributes -> inbox dedupe on the request's business key -> start the run
    (to its park) -> ACK. NACK on unreadable JSON, a missing key, or a request
    the contract refuses (the caller's DLQ after five attempts)."""

    def __init__(
        self,
        *,
        runner: SingleAgentWorkflowRunner,
        inbox: _InboxLike,
        contract: LaneContract,
        inbox_ttl: _dt.timedelta = DEFAULT_INBOX_TTL,
    ) -> None:
        self._runner = runner
        self._inbox = inbox
        self._contract = contract
        self._ttl = inbox_ttl

    async def handle_message(self, msg: _MessageLike) -> None:
        try:
            decoded = json.loads(msg.data.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("expected JSON object body")
            body: dict[str, Any] = decoded
        except (ValueError, UnicodeDecodeError) as exc:
            logger.exception(
                "single_agent_subscriber.decode_failed", extra={"lane": self._contract.name, "err": str(exc)}
            )
            msg.nack()
            return
        attrs = dict(msg.attributes or {})
        business_key = (
            attrs.get("idempotency_key") or attrs.get("event_id") or str(body.get("event_id", "") or "")
        ).strip()
        if not business_key:
            logger.warning(
                "single_agent_subscriber.missing_request_key", extra={"lane": self._contract.name, "attributes": attrs}
            )
            msg.nack()
            return
        dedup_key = f"{self._contract.request_kind}:{business_key}"

        async def _invoke() -> None:
            outcome = await self._runner.handle_request(body, attrs)
            logger.info(
                "single_agent_subscriber.handled",
                extra={
                    "lane": self._contract.name,
                    "outcome": outcome.outcome,
                    "thread_id": outcome.thread_id,
                    "detail": outcome.detail,
                },
            )

        try:
            ran = await self._inbox.process(key=dedup_key, ttl=self._ttl, fn=_invoke)
        except Exception:
            logger.exception(
                "single_agent_subscriber.run_failed", extra={"lane": self._contract.name, "dedup_key": dedup_key}
            )
            msg.nack()
            return
        msg.ack()
        logger.info(
            "single_agent_subscriber.acked", extra={"lane": self._contract.name, "dedup_key": dedup_key, "ran": ran}
        )


__all__ = [
    "DEFAULT_INBOX_TTL",
    "OUTCOME_COMPLETED",
    "OUTCOME_PARKED",
    "OUTCOME_REJECTED",
    "OUTCOME_SKIPPED",
    "SOURCE_SERVICE",
    "STATUS_FAILED",
    "STATUS_OK",
    "DispatchSpec",
    "LaneContract",
    "RequestOutcome",
    "ResultEvent",
    "SingleAgentRequestSubscriber",
    "SingleAgentState",
    "SingleAgentWorkflowRunner",
    "Skipped",
    "build_single_agent_graph",
    "envelope_scope",
    "loads_json_object",
    "synthetic_traceparent",
    "workflow_uuid",
]
