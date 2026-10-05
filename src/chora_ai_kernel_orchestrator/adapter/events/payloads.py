"""AI Kernel event payloads — typed record carriers for the Pub/Sub topics.

Mirrors ``chora-contracts/proto/events/ai_kernel/{invocation,crew,guardrail}.proto``
(the canonical wire shape; Protobuf binary swap handled at M14 production
wiring per ``feedback_d6_resilience_first_class``).

Each payload carries:

* ``workflow_id`` — UUID of the orchestrator workflow run (the aggregate
  root referenced by the outbox row).
* ``tenant_id`` / ``gcid`` — D6.3 multi-tenant chaos isolation columns.
* ``traceparent`` / ``tracestate`` — W3C trace context propagated across
  Pub/Sub per CLAUDE.md cross-cutting rule.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field


@dataclass
class ModelInvoked:
    """``chora.ai_kernel.invocation.invoked.v1`` payload."""

    invocation_id: str
    workflow_id: str
    tenant_id: str
    gcid: str
    agid: str
    model_id: str
    model_kind: str
    prompt_hash: str
    invoked_at: _dt.datetime
    traceparent: str = ""
    tracestate: str = ""


@dataclass
class ModelInvocationCompleted:
    """``chora.ai_kernel.invocation.completed.v1`` payload."""

    invocation_id: str
    workflow_id: str
    tenant_id: str
    gcid: str
    agid: str
    model_id: str
    model_kind: str
    prompt_hash: str
    response_hash: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    completed_at: _dt.datetime
    traceparent: str = ""
    tracestate: str = ""


@dataclass
class ModelInvocationFailed:
    """``chora.ai_kernel.invocation.failed.v1`` payload."""

    invocation_id: str
    workflow_id: str
    tenant_id: str
    gcid: str
    agid: str
    model_id: str
    model_kind: str
    prompt_hash: str
    error_code: str
    error_detail: str
    latency_ms: int
    failed_at: _dt.datetime
    traceparent: str = ""
    tracestate: str = ""


@dataclass
class CrewComposed:
    """``chora.ai_kernel.crew.composed.v1`` payload."""

    crew_id: str
    workflow_id: str
    tenant_id: str
    gcid: str
    crew_kind: str
    member_agids: list[str]
    composed_at: _dt.datetime
    traceparent: str = ""
    tracestate: str = ""


@dataclass
class CrewExecuted:
    """``chora.ai_kernel.crew.executed.v1`` payload."""

    crew_id: str
    workflow_id: str
    tenant_id: str
    gcid: str
    crew_kind: str
    duration_ms: int
    outcome: str
    executed_at: _dt.datetime
    member_outcomes: dict[str, str] = field(default_factory=dict)
    traceparent: str = ""
    tracestate: str = ""


@dataclass
class GuardrailEvaluated:
    """``chora.ai_kernel.guardrail.evaluated.v1`` payload."""

    invocation_id: str
    workflow_id: str
    tenant_id: str
    gcid: str
    agid: str
    tier: str
    outcome: str
    evaluated_at: _dt.datetime
    detail: str = ""
    traceparent: str = ""
    tracestate: str = ""


@dataclass
class AgentTerminated:
    """``chora.ai_kernel.agent.terminated.v1`` payload.

    Mirrors ``chora.ai_kernel.v1.AgentTerminated`` in
    ``chora-contracts/proto/events/ai_kernel/agent.proto``. Emitted at
    the Python orchestrator entry boundary on BOTH success and failure
    paths (failure mode is captured via ``termination_code``).

    Per the W3 foundation audit (2026-05-12): structured terminate-event
    is foundational for multi-crew chaos forensics + Cloud Trace span
    correlation + per-tenant attribution.
    """

    agent_id: str
    execution_id: str
    termination_code: str
    runtime: str
    tenant_id: str
    gcid: str
    terminated_at: _dt.datetime
    agent_agid: str = ""
    crew_id: str = ""
    crew_pattern: str = ""
    last_state_node: str = ""
    last_tool_name: str = ""
    last_error_message: str = ""
    iteration_count: int = 0
    partial_state: dict[str, str] = field(default_factory=dict)
    current_span_id: str = ""
    traceparent: str = ""
    tracestate: str = ""


__all__ = [
    "AgentTerminated",
    "CrewComposed",
    "CrewExecuted",
    "GuardrailEvaluated",
    "ModelInvocationCompleted",
    "ModelInvocationFailed",
    "ModelInvoked",
]
