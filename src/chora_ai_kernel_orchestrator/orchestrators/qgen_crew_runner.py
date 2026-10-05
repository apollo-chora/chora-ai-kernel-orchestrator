"""qgen_crew runner — wires the LangGraph state machine to Pub/Sub.

Step 3b of docs/m13/ack-oe-ai-assist-plan-2026-05-17.md. Connects:

    chora.creation.ai_assist.started.v1  (Pub/Sub subscriber)
            │
            ▼
    build_qgen_crew_graph + checkpointer  (LangGraph state machine,
                                           shipped in qgen_crew.py)
            │
            ▼
    chora.creation.ai_assist.completed.v1   (terminal: COMPLETED)
    chora.creation.ai_assist.refused.v1     (terminal: REFUSED)

The state machine itself + its unit tests live in
``orchestrators/qgen_crew.py`` (Step 3a). This module is the *runner*: the
load-bearing piece that drives the graph and publishes. ADR-254 D5: on the
Pub/Sub dispatch transport a run ends at its first agent PARK; the runners
drive until the park (``handle_started``), resume from each completion
(``handle_completion``, returning a settle thunk when a terminal is reached)
and settle idempotently whichever call got there (``_settle``).

Per [[feedback-agentic-pubsub-only]] the chora-creation handler does
NOT call the Vertex AI Reasoning Engine directly. It publishes
started.v1 + waits for the terminal event via this runner.

D6 4-pillar resilience per [[agentic-resilience-d6]]:
  - **Pod-death survival** — PostgresSaver checkpoint per job_id;
    on pod restart, ainvoke against the same thread_id resumes from
    the last persisted state. (Default thread_id = job_id.)
  - **Idempotent** — re-emit of the same started.v1 yields the same
    terminal state (LangGraph checkpoint is keyed on thread_id).
  - **DLQ** — handler errors → exception propagated to the Pub/Sub
    client which NACKs; subscription redelivers up to N times then
    dead-letters per topic-side DLQ config.
  - **OTel** — every node in the graph emits a span; the W3C
    traceparent carried in the started.v1 envelope continues the
    trace from the chora-creation publish span.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os as _os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from langgraph.types import Command

from chora_ai_kernel_orchestrator.adapter.gcs.source_uri_policy import (
    SourceUriNotPermittedError,
    require_tenant_scoped_source_uri,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    DISPATCH_INTERRUPT_KEY,
)
from chora_ai_kernel_orchestrator.domain.prompt_registry import (
    EMBEDDED_PROMPT_VERSIONS,
    SOURCE_EMBEDDED,
)
from chora_ai_kernel_orchestrator.domain.qgen_crew import (
    DEFAULT_MAX_RETRIES,
    CandidatePayload,
    QGenCrewState,
)
from chora_ai_kernel_orchestrator.observability.agent_trace import (
    agent_span_traceparent,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    DEFAULT_MAX_REGEN_ROUNDS,
    _chunk_size,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_grounding import (
    deterministic_draft_id,
    effective_source_files,
    normalise_candidate_citations,
    normalise_source_files,
    split_source_files,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_set_plan import (
    build_type_plan,
)

logger = logging.getLogger(__name__)


def _tracer() -> Any:
    """Return the OTel tracer for the runner. Lazy import keeps the
    test path SDK-free; degrades to a no-op when OTel isn't installed."""
    try:
        from opentelemetry import trace as _trace

        return _trace.get_tracer("chora_kernel.qgen_crew_runner")
    except ImportError:  # pragma: no cover — SDK should always be present in prod

        class _NoOpSpan:
            def __enter__(self) -> _NoOpSpan:
                return self

            def __exit__(self, *_: Any) -> None:
                return None

            def set_attribute(self, *_: Any, **__: Any) -> None:
                return None

        class _NoOpTracer:
            def start_as_current_span(self, *_: Any, **__: Any) -> _NoOpSpan:
                return _NoOpSpan()

        return _NoOpTracer()


def _continue_trace_context(traceparent: str) -> Any:
    """Convert a W3C ``traceparent`` string into an OTel ``Context`` so the
    runner's spans become children of the publisher's span tree (the
    chora-creation handler that published started.v1). Returns ``None``
    when traceparent is empty or OTel isn't installed — caller falls back
    to current ambient context.
    """
    if not traceparent:
        return None
    try:
        from opentelemetry import propagate

        carrier = {"traceparent": traceparent}
        return propagate.extract(carrier)
    except ImportError:  # pragma: no cover
        return None


# Pub/Sub topic names — mirror chora-contracts/proto/events/creation/
# ai_assist.proto §AiAssistCompleted, §AiAssistRefused. The STARTED copy was
# removed in CHO-2399: it was referenced only by __all__ and had silently
# drifted to the retired .v1 while the deployed topic is .v2 — a dead
# duplicate of a wire constant is exactly how that class of rot hides. The
# inbound topic's authoritative name lives with the subscriber wiring
# (adapter/pubsub/qgen_crew_loop.py names the -v2 subscription).
TOPIC_AI_ASSIST_COMPLETED = "chora.creation.ai_assist.completed.v1"
TOPIC_AI_ASSIST_REFUSED = "chora.creation.ai_assist.refused.v1"

# NOTE: the orchestrator no longer produces the observability
# token-usage-recorded event. That lane was RETIRED 2026-07-23 -- see
# tests/unit/test_token_usage_emitter_retired.py for the full incident
# write-up. Per ADR-163 chora-model-gateway is the SOLE producer; the qgen
# agents reach the LLM through it, so emitting here double-counted every
# call and did so with an empty, unpriceable model_id.

# Canonical AgentDecisionLog topic per chora-contracts/proto/events/
# observability/agent_decision.proto §AgentDecisionLogged + Tier 3 IMDA D1
# (accountability) per ADR-141. The Observability supporting domain OWNS the
# decision_log; AI Kernel is the PRODUCER per imda-governance-4-dimensions
# skill.
#
# NOTE: docs/m13/oe-ai-assist-session-close-2026-05-17.md §Step 6 names the
# topic chora.ai_kernel.agent_decided.v1, but that topic does NOT exist in
# deployed terraform. Per [[feedback-arch-ground-in-deployed-reality]] we
# emit on the canonical provisioned topic
# chora.observability.agent_decision.logged.v1 — same semantic intent,
# same accountability dimension, has BigQuery sink + IAM grants wired.
TOPIC_OBSERVABILITY_AGENT_DECISION = "chora.observability.agent_decision.logged.v1"

# IMDA dimension label per ADR-141 for accountability evidence (D1).
IMDA_DIMENSION_ACCOUNTABILITY = "accountability"

# Crew identifier for the qgen 2-agent (qgen_question + qgen_critic) crew
# that drives the MCQ-AI-Assist authoring flow on /a/atoms/{id}/edit. The
# crew_name is the stable snake_case identifier used by O+ to render the
# Crews + Agents hierarchy on /o/agents per the 2026-05-26 O+ hydration
# plan (atomic-napping-spring). Agents are reusable across crews — the
# same qgen_critic agid may emit decisions tagged with different
# crew_name values in future waves (e.g. `oe_ai_assist`).
MCQ_AI_ASSIST_CREW_NAME = "mcq_ai_assist"

# Per-agent registry ids (agid) for the qgen 2-agent crew. These MUST match the
# deployed GKE agent names (chora-qgen-question / chora-qgen-critic) + the
# /o/agents registry tiles. The producer emits ONE decision per agent per
# generation so each tile counts the runs its agent participated in. NEVER
# "qgen_crew" (the legacy crew-level hardcode that matched no tile).
AGID_QGEN_QUESTION = "qgen_question"
AGID_QGEN_CRITIC = "qgen_critic"

# Maps each qgen agid to the pipeline_trace LLM-hop whose token counts it owns,
# so the gen_ai.usage.* counts are split per agent (no cross-tile double-count).
_AGID_TO_LLM_HOP: dict[str, str] = {
    AGID_QGEN_QUESTION: "generate",
    AGID_QGEN_CRITIC: "critique",
}

# Eval-run segregation toggle per
# .claude/skills/mlops-agent-eval/SKILL.md. When CHORA_IS_EVAL_RUN=true
# the emitted AgentDecisionLog rows carry is_eval_run=True so the
# chora-governance projector + downstream dashboards segregate eval
# traffic from production posture.
ENV_CHORA_IS_EVAL_RUN = "CHORA_IS_EVAL_RUN"

# Canonical topic for the Human-Oversight (HITL) gate-requested event. The
# gate lives in chora_governance's hitl_decision_log, so this is a
# governance-domain topic. The chora-governance projector routes by
# chora_imda_dimension=fairness_and_human_oversight + an event_type
# containing "hitl" into hitl_decision_log via routeD4 → AppendHITLDecision;
# the gateway mapHITLItem then renders the pending gate on
# GET /api/hitl/pending. FLAGGED: topic provisioning + the governance HITL
# subscriber are out of scope here (coordinator-sequenced).
TOPIC_GOVERNANCE_HITL_REQUESTED = "chora.governance.hitl.requested.v1"

# IMDA dimension label per ADR-141 for the HITL escalation (D4). The
# AI Assist Reporter's HITL escalation is fairness_and_human_oversight
# evidence — distinct from the per-run AgentDecisionLog which is D1
# accountability.
IMDA_DIMENSION_FAIRNESS_HUMAN_OVERSIGHT = "fairness_and_human_oversight"

# Autonomy level stamped on the qgen HITL gate (ADR-141 Level 0-2; Level 3
# PROHIBITED). The qgen reporter escalation is a single-pass approve gate =
# HITL Level 0. Matches the governance evidence.AutonomyLevel value space
# (services/chora-governance/internal/domain/evidence/evidence.go).
_QGEN_HITL_AUTONOMY_LEVEL = "hitl_l0"

# Env knob (per `feedback_no_inline_config`) — when truthy (default), the
# runner escalates a MAX-RETRIES-EXHAUSTED terminal (decision=
# completed_with_warning, quality_warning=True) to a HITL gate. Set to
# "false"/"0"/"no" to disable the escalation without a redeploy.
ENV_HITL_ESCALATE_ON_QUALITY_WARNING = "QGEN_HITL_ESCALATE_ON_QUALITY_WARNING"

# Trace-row "name" values that mark an LLM-issuing dispatch hop.
# (_TRACE_NAME_TO_ROLE was removed 2026-07-23 with the token-usage lane --
# it had no other reader.)
_LLM_HOP_TRACE_NAMES = ("generate", "critique")

# Mapping from final quality_gate trace-row status → AgentDecisionLog
# decision string. The semantics mirror quality_gate_node in
# orchestrators/qgen_crew.py §480-536.
_QUALITY_GATE_STATUS_TO_DECISION: dict[str, str] = {
    "ACCEPTED": "accepted",
    "RETRY": "retry",
    "QUALITY_WARNING": "completed_with_warning",
    "REJECTED": "rejected",
    "FAILED": "rejected",
}


# -----------------------------------------------------------------------------
# Inbound payload — mirror of the started.v1 wire shape from
# chora-creation's outbox publish in
# services/chora-creation/internal/adapter/http/ai_assist_async_handler.go
# §aiAssistAsync `startedPayload`.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AiAssistStartedPayload:
    """Decoded chora.creation.ai_assist.started.v1 wire payload."""

    assist_id: str
    tenant_id: str
    author_gcid: str
    content_type: str  # legacy proto field — same value space as question_type
    question_type: str  # added 2026-05-17 — Phyllis scope: mcq | oe
    prompt: str
    metadata: dict[str, str]
    max_retries: int
    # Optional trace context (W3C) — propagated into the LangGraph span.
    traceparent: str = ""
    tracestate: str = ""
    # W8 author per-image opt-in (author-SELECTED). Both default False ⇒ no
    # image, live MCQ loop unchanged. These land on started.v1 in a parallel
    # chora-contracts layer; read defensively (absent ⇒ False).
    image_for_stem: bool = False
    image_for_answer: bool = False
    # -- EPIC-1a batch + grounding (additive; single-path defaults keep the live
    # single-candidate flow byte-for-byte unchanged) --
    #   job_kind: "" | "single" → QGenCrewRunner; "batch" → QGenBatchRunner.
    job_kind: str = ""
    requested_count: int = 1
    grounding_mode: str = ""
    source_blob_uri: str = ""
    source_mime_type: str = ""
    target_growth_edges: tuple[str, ...] = ()
    # -- Lane 1c multi-file + rubric grounding (CHO-1703, additive) --
    # Canonical role-tagged grounding files (f20). Empty tuple on pre-1c
    # events; f17/18 above mirror source_files[0] per the W0 contract.
    # Entries are {"blob_uri", "mime_type", "role"} dicts (role ∈
    # source|rubric, normalised by qgen_grounding.normalise_source_files).
    source_files: tuple[dict[str, str], ...] = ()
    # -- Mixed-type batch (CHO-1819, additive) --
    # Typed type_plan quotas (f21) decoded by proto_wire.decode_ai_assist_started
    # as [{question_type, count, max_images}]. EMPTY ⇒ legacy single-type batch
    # (the proven N-loop path, byte-for-byte unchanged + pre-P1c-agent safe);
    # NON-EMPTY ⇒ the set-native single-pass lane. content_type == "mixed".
    type_plan: tuple[dict[str, Any], ...] = ()
    # -- Review image regenerate (CHO-1819 P3, additive) --
    # The decoded regen spec {draft_id, placement, prompt, mode} (proto f22,
    # surfaced by proto_wire.decode_ai_assist_started). None ⇒ non-regen job
    # (the live single/batch paths are byte/behaviour-unchanged).
    regen: dict[str, Any] | None = None
    # -- ADR-195 WS7 (D7) compose model (.v2 ai_assist.started) --
    # The .v2 event DROPS job_kind and carries {operation, intent, input_kind}
    # explicitly (proto f23/24/25). Defaulted to "" so a v1 event (which carries
    # none of them) reads the attributes without a KeyError. The router PREFERS
    # these when present and falls back to job_kind for a v1 wire.
    operation: str = ""
    intent: str = ""
    input_kind: str = ""
    # -- CHO-1658 — model_answer_fill author content (.v2 existing_question_json,
    # proto f26, decoded by proto_wire into a dict). None on every non-fill event.
    # The runner threads it into QGenCrewState so generate_node forwards it to the
    # qgen_question executor, which surfaces the author_stem / author_options /
    # author_rubric / model_answer fill keys.
    existing_question: dict[str, Any] | None = None

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> AiAssistStartedPayload:
        """Parse the Pub/Sub-decoded event dict into a typed payload.

        The chora-creation handler emits a flat dict — but if a future
        wave swaps to the proto-encoded Envelope shape, this is the
        single place to update.
        """
        # Fall back to content_type when question_type missing (the proto
        # was extended additively 2026-05-17 — old publishers might still
        # emit only content_type).
        question_type = str(event.get("question_type") or event.get("content_type") or "")
        return cls(
            assist_id=str(event.get("assist_id") or event.get("job_id") or ""),
            tenant_id=str(event.get("tenant_id") or ""),
            author_gcid=str(event.get("author_gcid") or event.get("gcid") or ""),
            content_type=str(event.get("content_type") or ""),
            question_type=question_type,
            prompt=str(event.get("prompt") or ""),
            metadata=dict(event.get("metadata") or {}),
            max_retries=int(event.get("max_retries") or DEFAULT_MAX_RETRIES),
            traceparent=str(event.get("traceparent") or ""),
            tracestate=str(event.get("tracestate") or ""),
            # W8 — defensive read (field added in a parallel contracts layer;
            # absent on older publishers ⇒ False).
            image_for_stem=bool(event.get("image_for_stem") or False),
            image_for_answer=bool(event.get("image_for_answer") or False),
            # EPIC-1a batch + grounding — additive, defensively defaulted.
            job_kind=str(event.get("job_kind") or ""),
            requested_count=int(event.get("requested_count") or 1),
            grounding_mode=str(event.get("grounding_mode") or ""),
            source_blob_uri=str(event.get("source_blob_uri") or ""),
            source_mime_type=str(event.get("source_mime_type") or ""),
            target_growth_edges=tuple(str(e) for e in (event.get("target_growth_edges") or [])),
            # Lane 1c — defensive parse (garbage entries dropped; absent ⇒ ()).
            source_files=tuple(normalise_source_files(event.get("source_files"))),
            # CHO-1819 — typed type_plan quotas (proto_wire surfaces them as
            # [{question_type, count, max_images}] dicts). Absent ⇒ () ⇒ the
            # legacy single-type batch loop.
            type_plan=tuple(q for q in (event.get("type_plan") or []) if isinstance(q, dict)),
            # CHO-1819 P3 — regen spec (proto_wire surfaces it as a dict).
            # Absent / non-dict ⇒ None ⇒ non-regen job.
            regen=(dict(event["regen"]) if isinstance(event.get("regen"), dict) else None),
            # ADR-195 WS7 (D7) — compose model on the .v2 wire (absent on v1).
            operation=str(event.get("operation") or ""),
            intent=str(event.get("intent") or ""),
            input_kind=str(event.get("input_kind") or ""),
            # CHO-1658 — defensive: only a dict (proto_wire's parsed JSON) is kept;
            # absent / non-dict ⇒ None ⇒ the executor's `or {}` degrades safely.
            existing_question=(
                dict(event["existing_question"]) if isinstance(event.get("existing_question"), dict) else None
            ),
        )


# -----------------------------------------------------------------------------
# Terminal publisher — duck-typed Protocol for testability
# -----------------------------------------------------------------------------


class _TerminalPublisher(Protocol):
    """Publishes chora.creation.ai_assist.{completed,refused}.v1.

    Production: a NatsPublisher-backed wrapper that builds
    the canonical envelope (tenant_id + traceparent + tracestate +
    chora_imda_dimension etc.) and publishes.

    Tests: an in-memory recorder.
    """

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
        generated_count: int = 0,
        generation_summary: dict[str, Any] | None = None,
    ) -> str: ...

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
    ) -> str: ...


# -----------------------------------------------------------------------------
# AgentDecisionLog emitter — duck-typed Protocol per
# [[imda-governance-4-dimensions]] (Tier 3 IMDA D1 accountability evidence
# per ADR-141). Canonical billing-grade decision log writes go via the
# transactional outbox; the dispatcher publishes to Pub/Sub topic
# `chora.observability.agent_decision.logged.v1`.
#
# Production wiring: an AgentDecisionLogOutboxWriter (adapter — companion to
# QGenCrewTerminalOutboxWriter) that INSERTs one ai_kernel_outbox_events row
# per emit.
#
# Tests: an in-memory recorder.
# -----------------------------------------------------------------------------


class _AgentDecisionLogEmitter(Protocol):
    """Emits ONE chora.observability.agent_decision.logged.v1 event PER CALL,
    carrying the per-agent ``agid`` + question_type + the quality_gate decision
    + IMDA D1 dimension (accountability). The qgen runner invokes this twice per
    terminal run — once for ``agid="qgen_question"`` and once for
    ``agid="qgen_critic"`` — so each /o/agents tile counts the runs its agent
    participated in (the legacy single hardcoded "qgen_crew" event matched no
    tile). Best-effort — transient emit failures MUST NOT suppress the terminal
    publish (the runner logs + proceeds).

    Extension fields (2026-05-26 — per atomic-napping-spring O+ hydration
    plan + chora-contracts proto/events/observability/agent_decision.proto
    fields 12-20):

      - ``crew_name`` — stable snake_case crew identifier (e.g.
        ``mcq_ai_assist``). Feeds /o/agents Crews+Agents hierarchy.
      - ``crew_id`` — UUIDv7 of the crew instance / orchestration the
        decision belongs to. The runner uses assist_id as the
        orchestration identifier since the qgen runner is invocation-
        scoped.
      - ``is_resume`` — True when this decision occurred during a resumed
        orchestration (continuation after an interrupt). Defaults False.
      - ``is_eval_run`` — True when emitted under the eval golden-pipeline
        (per .claude/skills/mlops-agent-eval/SKILL.md). Eval-tagged rows
        must be segregated from production dashboards.
      - ``adapter_version`` — LoRA adapter version used by this decision
        (per gemma-lora-tenant SKILL). Empty string when base model used.
      - ``guardrail_outcome`` — Cloud Model Armor verdict (``pass`` |
        ``block`` | ``redact``). Empty string when guardrails not
        invoked.
      - ``prompt_tokens`` / ``completion_tokens`` / ``cached_tokens`` —
        gen_ai.usage.* aggregated across the terminal pipeline_trace LLM
        hops. Zero when no LLM was issued.
    """

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
    ) -> str: ...


# -----------------------------------------------------------------------------
# HITLDecision emitter — duck-typed Protocol for the Human-Oversight gate.
#
# Closes the "queue always empty" root cause: nothing escalated to a HITL
# gate. When the qgen terminal is MAX-RETRIES-EXHAUSTED
# (decision=completed_with_warning, quality_warning=True — a genuinely
# low-quality candidate), the runner emits ONE gate event carrying the
# chora-governance projector's D4 routing fields
# (chora_imda_dimension=fairness_and_human_oversight + event_type containing
# "hitl" + decision_id/run_id/agent_id/summary/autonomy_level/created_at).
#
# Production wiring: a HITLDecisionOutboxWriter (adapter — companion to
# QGenCrewTerminalOutboxWriter) that INSERTs one ai_kernel_outbox_events row
# per emit on chora.governance.hitl.requested.v1.
#
# Tests: an in-memory recorder.
# -----------------------------------------------------------------------------


class _HITLDecisionEmitter(Protocol):
    """Emits ONE chora.governance.hitl.requested.v1 event when the terminal
    run escalates to Human-In-The-Loop review. Best-effort — a transient emit
    failure MUST NOT suppress the terminal publish (the runner logs +
    proceeds)."""

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
        hitl_verdict: str = "pending",
        edit_payload: dict[str, Any] | None = None,
        traceparent: str = "",
        tracestate: str = "",
    ) -> str: ...


# -----------------------------------------------------------------------------
# Compiled-graph protocol — what we need from the LangGraph CompiledStateGraph
# -----------------------------------------------------------------------------


class _CompiledGraphLike(Protocol):
    """Duck-typed CompiledStateGraph from langgraph.

    The real CompiledStateGraph has many more methods; we need ainvoke for
    run-to-terminal AND astream for per-super-step streaming (the latter
    drives the mid-run progress emission — each ``stream_mode="values"`` chunk
    is the cumulative state, and the LAST chunk is byte-identical to ainvoke).
    """

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    def astream(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        *,
        stream_mode: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]: ...


class _PromptResolverLike(Protocol):
    """ADR-197 M-B.2 — duck-typed PromptResolver port.

    Production: ``domain.prompt_registry.PromptResolver`` bound to a
    ``PostgresPromptOverrideRepository`` over the shared psycopg conn (wired in
    the crew composition root). Tests: an in-memory fake. ``resolve`` returns a
    ``Resolved(segments, version, source)``; no active override → the embedded
    default (``segments={}``). Errors MUST surface (the runner does NOT swallow
    them — a resolve failure NACKs the message).
    """

    async def resolve(self, tenant_id: str, agent_id: str) -> Any: ...


class _ProgressPublisher(Protocol):
    """Mid-run progress emitter — publishes one
    chora.creation.ai_assist.progress.v1 per qgen graph node with the
    cumulative-so-far pipeline_trace. Satisfied by QGenCrewTerminalOutboxWriter
    (which also satisfies _TerminalPublisher); injected as None on the M11
    baseline (no streaming).
    """

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
    ) -> str: ...


# ---------------------------------------------------------------------------
# Park-aware graph driving (ADR-254 D5): start, re-drive, resume
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeOutcome:
    """What a completion resume did: the graph PARKED again (``settle`` None)
    or reached a terminal, in which case ``settle`` is the coroutine factory
    that publishes it. The settle is handed back rather than awaited inline so
    the acceptance layer can run it as a tracked background drive: nothing
    slower than a checkpoint round trip sits on the completion ack path."""

    parked: bool
    settle: Callable[[], Awaitable[None]] | None = None


_ProgressHook = Callable[[list[dict[str, Any]], str, str], Awaitable[None]]
_SkipPredicate = Callable[[str], bool]

# Recursion budgets. The single lane: up to 4 attempts x 4 nodes + the render
# loop (<= 2 images) + lane overhead (~25 supersteps); LangGraph's default of
# 25 was already tight, and a resume never raises the budget. The set lane is
# sized from the plan (chunks x (clean chunk + one regen round + 2 images per
# question)); the loop is structurally bounded by chunk_count regardless.
_SINGLE_RECURSION_LIMIT = 60


def _set_recursion_limit(count: int) -> int:
    size = max(1, _chunk_size())
    chunks = max(1, -(-max(1, count) // size))
    return 30 + chunks * (13 + 2 * size)


def _skip_publish_prefix(name: str) -> bool:
    """Single lane: terminal publish_* rows are not streamed (completed.v1
    carries the full final trace)."""
    return name.startswith("publish_")


def _skip_terminal_publishes(name: str) -> bool:
    """Set lane: only the two terminals are skipped; publish_chunk is a mid-run
    chunk boundary (ADR-251 D1) the live pipeline view wants."""
    return name in ("publish_completed", "publish_refused")


def _interrupted(result: Mapping[str, Any]) -> bool:
    return bool(result.get("__interrupt__"))


def _dispatch_key_of(interrupt_value: Any) -> str:
    """The dispatch idempotency key a park carries (``wrap_for_interrupt``)."""
    if isinstance(interrupt_value, Mapping):
        request = interrupt_value.get(DISPATCH_INTERRUPT_KEY)
        if isinstance(request, Mapping):
            return str(request.get("idempotency_key") or "")
    return ""


async def _thread_status(graph: Any, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """``("fresh" | "parked" | "running" | "finished", values)`` for a thread.

    A thread with pending interrupts is PARKED (its completion resumes it;
    re-driving it would dispatch again, a second paid model call); one with
    work queued but no interrupt was CUT mid-superstep (continue it); one with
    values and nothing queued is FINISHED (settle it, never re-invoke); anything
    else is fresh. A graph compiled without a checkpointer has no snapshot and
    is always fresh (stateless dev graphs).
    """
    try:
        snapshot = await graph.aget_state(config)
    except ValueError:
        return "fresh", {}
    values = dict(getattr(snapshot, "values", None) or {})
    if getattr(snapshot, "next", ()):
        if any(getattr(t, "interrupts", ()) for t in (getattr(snapshot, "tasks", ()) or ())):
            return "parked", values
        return "running", values
    if values and getattr(snapshot, "created_at", None) is not None:
        return "finished", values
    return "fresh", values


async def _run_graph(
    graph: Any,
    input_: Any,
    config: dict[str, Any],
    *,
    progress: _ProgressHook | None,
    skip: _SkipPredicate,
    already_emitted: int,
) -> dict[str, Any]:
    """ainvoke, or astream(values) with one progress emit per NEW trace row
    (the live-trace streaming contract; the LAST chunk is byte-identical to
    ainvoke's return so the caller reads it the same way)."""
    if progress is None:
        return await graph.ainvoke(input_, config=config)
    terminal: dict[str, Any] = {}
    last_emitted = already_emitted
    async for chunk in graph.astream(input_, config=config, stream_mode="values"):
        terminal = chunk
        trace = chunk.get("pipeline_trace") or []
        if len(trace) > last_emitted:
            latest = trace[-1] if trace else {}
            name = str(latest.get("name") or "")
            if not skip(name):
                last_emitted = len(trace)
                await progress(trace, name, str(latest.get("status") or ""))
    return terminal


async def _drive(
    graph: Any,
    state: Any,
    config: dict[str, Any],
    *,
    progress: _ProgressHook | None = None,
    skip: _SkipPredicate = _skip_publish_prefix,
) -> tuple[dict[str, Any], bool]:
    """Start or re-drive a thread idempotently -> ``(values, parked)``.

    The boot sweep re-drives every in-flight job from its started payload; a
    job parked on a dispatch is left for its completion, a finished one settles
    from its checkpoint (the pod died between the graph's end and the
    terminal publish), a cut one continues, a fresh one starts."""
    kind, values = await _thread_status(graph, config)
    if kind == "parked":
        return values, True
    if kind == "finished":
        return values, False
    input_ = None if kind == "running" else state
    already = len(values.get("pipeline_trace") or []) if kind == "running" else 0
    result = await _run_graph(graph, input_, config, progress=progress, skip=skip, already_emitted=already)
    return result, _interrupted(result)


async def _resume(
    graph: Any,
    config: dict[str, Any],
    completion: Mapping[str, Any],
    *,
    progress: _ProgressHook | None = None,
    skip: _SkipPredicate = _skip_publish_prefix,
) -> tuple[dict[str, Any], bool]:
    """Resume a parked thread from one completion -> ``(values, parked)``.

    The completion is matched to ITS interrupt by the dispatch idempotency key
    the park carries, and resumed by interrupt id; a single pending interrupt
    without a key match resumes the plain way (the reaper's synthesized
    completion carries the same key, so it matches too). A thread with no
    pending interrupt and nothing queued is already terminal: a redelivered
    completion after a cut settle; its values are returned to settle again
    (idempotent publish). Two pending interrupts and no match is refused,
    never guessed: resuming the wrong one would hand one dispatch's answer to
    another's node."""
    try:
        snapshot = await graph.aget_state(config)
    except ValueError as exc:
        raise RuntimeError("qgen resume needs a checkpointed graph") from exc
    pending = [i for t in (getattr(snapshot, "tasks", ()) or ()) for i in getattr(t, "interrupts", ())]
    values = dict(getattr(snapshot, "values", None) or {})
    already = len(values.get("pipeline_trace") or [])
    if not pending:
        if getattr(snapshot, "next", ()):
            result = await _run_graph(graph, None, config, progress=progress, skip=skip, already_emitted=already)
            return result, _interrupted(result)
        return values, False
    key = str(completion.get("idempotency_key") or "")
    matching = [i for i in pending if key and _dispatch_key_of(i.value) == key]
    if matching:
        resume_input: Any = Command(resume={matching[0].id: dict(completion)})
    elif len(pending) == 1:
        resume_input = Command(resume=dict(completion))
    else:
        raise ValueError(
            f"qgen resume: {len(pending)} interrupts pending on thread "
            f"{config.get('configurable', {}).get('thread_id', '')!r} and completion key {key!r} "
            "matches none; refusing to guess which dispatch this answers"
        )
    result = await _run_graph(graph, resume_input, config, progress=progress, skip=skip, already_emitted=already)
    return result, _interrupted(result)


def _require_thread_id(completion: Mapping[str, Any], who: str) -> str:
    thread_id = str(completion.get("thread_id") or "").strip()
    if not thread_id:
        raise ValueError(f"{who}: completion carries no thread_id; refusing to resume a guessed thread")
    return thread_id


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


class QGenCrewRunner:
    """Drives the single-candidate qgen graph for one
    chora.creation.ai_assist.started event and publishes its terminal.

    ADR-254 D5: on the dispatch transport a run ends at its first agent park.
    ``handle_started`` drives until the park (or a terminal) and
    ``handle_completion`` resumes from each completion; the terminal is
    published by ``_settle`` whichever call reached it. Both are idempotent
    against a re-drive (the boot sweep) and a redelivered completion.

    Construction: pass the compiled graph + a terminal publisher. The graph is
    compiled ONCE per process (via build_qgen_crew_graph in qgen_crew.py) and
    threaded through every runner invocation. Thread id for LangGraph
    checkpointing: the assist_id (== chora-creation's job_id).
    """

    def __init__(
        self,
        *,
        graph: _CompiledGraphLike,
        publisher: _TerminalPublisher,
        thread_id_for: Callable[[AiAssistStartedPayload], str] | None = None,
        agent_decision_emitter: _AgentDecisionLogEmitter | None = None,
        hitl_decision_emitter: _HITLDecisionEmitter | None = None,
        progress_emitter: _ProgressPublisher | None = None,
        prompt_resolver: _PromptResolverLike | None = None,
    ) -> None:
        self._graph = graph
        self._publisher = publisher
        self._thread_id_for = thread_id_for or (lambda p: p.assist_id)
        # Optional: when None, NO mid-run progress events are emitted and the
        # graph executes via the single-shot ainvoke (M11 baseline, byte-stable).
        # When wired (QGEN_PROGRESS_STREAM_ENABLED), the drive switches to
        # astream + publishes one progress.v1 per node with the partial trace.
        self._progress_emitter = progress_emitter
        # Optional: when None, no AgentDecisionLog emission happens
        # (M11-baseline backwards-compat shape).
        self._agent_decision_emitter = agent_decision_emitter
        # Optional: when None, no HITL gate emission happens. Same
        # M11-baseline backwards-compat shape. When wired, a MAX-RETRIES-
        # EXHAUSTED terminal escalates to a Human-Oversight gate (env-gated).
        self._hitl_decision_emitter = hitl_decision_emitter
        # ADR-197 M-B.2: optional PromptResolver. When None (no shared psycopg
        # conn / not wired) NO prompt-override resolution happens and the run is
        # byte-identical to the pre-registry baseline. When wired, the start
        # resolves the active override per agent role (qgen_question +
        # qgen_critic) and threads it into the executor payload + durable record.
        self._prompt_resolver = prompt_resolver

    # ---- start / re-drive -------------------------------------------------

    async def handle_started(self, event: dict[str, Any]) -> None:
        """Top-level message handler (also the boot sweep's re-drive).

        Errors propagate so the caller can NACK; a re-drive of a parked thread
        returns without dispatching again, a re-drive of a finished thread
        settles it without re-invoking.
        """
        payload = AiAssistStartedPayload.from_event(event)
        self._validate(payload)
        state = self._build_state(payload)
        # ADR-197 M-B.2: resolve the active prompt override per agent role
        # BEFORE the graph runs so the segment map rides every generate/critique
        # executor call + the durable AgentDecisionLog. No-op when no resolver is
        # wired OR no active override applies (byte-identical baseline).
        await self._apply_prompt_overrides(state, payload.tenant_id)

        thread_id = self._thread_id_for(payload)
        run_config = self._run_config(thread_id)
        logger.info(
            "qgen_crew_runner.started",
            extra={
                "assist_id": payload.assist_id,
                "tenant_id": payload.tenant_id,
                "question_type": payload.question_type,
                "max_retries": payload.max_retries,
                "thread_id": thread_id,
                "traceparent": payload.traceparent,
            },
        )
        # Continue the W3C trace from the chora-creation publisher span so the
        # orchestrator's per-graph-run span tree lands under the same Cloud
        # Trace trace_id as the FE-originated POST.
        parent_ctx = _continue_trace_context(payload.traceparent)
        with _tracer().start_as_current_span("qgen_crew.handle_started", context=parent_ctx) as span:
            self._stamp_span(span, payload, thread_id)
            terminal, parked = await _drive(
                self._graph,
                state,
                run_config,
                progress=self._progress_hook(payload),
                skip=_skip_publish_prefix,
            )
        if parked:
            logger.info(
                "qgen_crew_runner.parked",
                extra={"assist_id": payload.assist_id, "thread_id": thread_id},
            )
            return
        await self._settle(payload, terminal)

    # ---- completion resume ------------------------------------------------

    async def handle_completion(self, completion: Mapping[str, Any], *, started_event: dict[str, Any]) -> ResumeOutcome:
        """Resume the parked thread from one agent completion (ADR-254 D5).

        ``started_event`` is the job's durable started payload (the in-flight
        registry row), needed to settle with the same publisher fields a
        single-call run would have used. Returns the settle thunk when the
        graph reached a terminal; the caller (the acceptance layer) runs it as a
        tracked background drive."""
        payload = AiAssistStartedPayload.from_event(started_event)
        thread_id = _require_thread_id(completion, "qgen_crew_runner.handle_completion")
        run_config = self._run_config(thread_id)
        logger.info(
            "qgen_crew_runner.completion",
            extra={
                "assist_id": payload.assist_id,
                "thread_id": thread_id,
                "agent_role": completion.get("agent_role"),
                "status": completion.get("status"),
            },
        )
        parent_ctx = _continue_trace_context(payload.traceparent)
        with _tracer().start_as_current_span("qgen_crew.handle_completion", context=parent_ctx) as span:
            self._stamp_span(span, payload, thread_id)
            terminal, parked = await _resume(
                self._graph,
                run_config,
                completion,
                progress=self._progress_hook(payload),
                skip=_skip_publish_prefix,
            )
        if parked:
            return ResumeOutcome(parked=True)

        async def _settle_later() -> None:
            await self._settle(payload, terminal)

        return ResumeOutcome(parked=False, settle=_settle_later)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _run_config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": _SINGLE_RECURSION_LIMIT}

    @staticmethod
    def _stamp_span(span: Any, payload: AiAssistStartedPayload, thread_id: str) -> None:
        # OpenInference semantic conventions (per [[ai-observability-
        # cloud-trace]]): surface the qgen_crew run as an orchestrator-level
        # chain so Cloud Trace renders the agent call tree correctly. Mirror
        # service.name from the Resource onto the span itself (the Python
        # google-cloud-trace-exporter does NOT surface Resource.service.name as
        # a span label).
        span.set_attribute("openinference.span.kind", "chain")
        span.set_attribute("service.name", "chora-ai-kernel-orchestrator")
        span.set_attribute("chora.assist_id", payload.assist_id)
        span.set_attribute("chora.tenant_id", payload.tenant_id)
        span.set_attribute("chora.gcid", payload.author_gcid)
        span.set_attribute("chora.thread_id", thread_id)
        span.set_attribute("chora.question_type", payload.question_type or "")

    @staticmethod
    def _build_state(payload: AiAssistStartedPayload) -> QGenCrewState:
        return {
            "job_id": payload.assist_id,
            "tenant_id": payload.tenant_id,
            "gcid": payload.author_gcid,
            "prompt": payload.prompt,
            "question_type": payload.question_type or payload.content_type,
            "metadata": payload.metadata,
            "max_retries": payload.max_retries,
            # W8: author per-image opt-in threaded into generate_node's payload.
            "image_for_stem": payload.image_for_stem,
            "image_for_answer": payload.image_for_answer,
            # CHO-1658: compose intent + the author's existing question (model_
            # answer_fill). generate_node forwards both to the executor ONLY when
            # set, so the live new_question loop stays byte-stable.
            "intent": payload.intent,
            "existing_question": payload.existing_question or {},
            "pipeline_trace": [],
            "errors": [],
        }

    def _progress_hook(self, payload: AiAssistStartedPayload) -> _ProgressHook | None:
        emitter = self._progress_emitter
        if emitter is None:
            return None

        async def _hook(trace: list[dict[str, Any]], name: str, status: str) -> None:
            await self._emit_progress(payload, trace, name=name, status=status)

        return _hook

    async def _apply_prompt_overrides(self, state: QGenCrewState, tenant_id: str) -> None:
        """ADR-197 M-B.2: resolve + store the per-agent prompt override.

        For each qgen agent role (qgen_question + qgen_critic) ask the resolver
        for the active override for ``(tenant_id, agid)`` and, ONLY when an
        override actually applied (``segments`` non-empty), store the winning
        segment map + version + source under per-agent state keys. The graph
        nodes thread these into the executor payload; ``_emit_agent_decision_log``
        stamps the version + source onto the durable record.

        No resolver wired => no-op. Embedded default (no active override) => no
        keys set => byte-identical pre-registry behaviour. Resolver errors are NOT
        swallowed (fail-loud; the caller NACKs).
        """
        if self._prompt_resolver is None:
            return
        for agid, suffix in (
            (AGID_QGEN_QUESTION, "question"),
            (AGID_QGEN_CRITIC, "critic"),
        ):
            resolved = await self._prompt_resolver.resolve(tenant_id, agid)
            segments = getattr(resolved, "segments", None)
            if not segments:
                continue
            state[f"prompt_overrides_{suffix}"] = dict(segments)  # type: ignore[literal-required]
            state[f"prompt_version_{suffix}"] = str(getattr(resolved, "version", "") or "")  # type: ignore[literal-required]
            state[f"prompt_source_{suffix}"] = str(getattr(resolved, "source", "") or "")  # type: ignore[literal-required]

    @staticmethod
    def _validate(payload: AiAssistStartedPayload) -> None:
        """Fail-loud guards on the inbound payload per
        [[feedback-no-stubs-real-wiring]]. The graph's validate_input
        node also guards; this duplicates so we don't waste a graph run
        on obviously-malformed input.
        """
        missing: list[str] = []
        for name, value in (
            ("assist_id", payload.assist_id),
            ("tenant_id", payload.tenant_id),
            ("author_gcid", payload.author_gcid),
            ("prompt", payload.prompt),
        ):
            if not value.strip():
                missing.append(name)
        if missing:
            raise ValueError(f"qgen_crew_runner: malformed started.v1 — missing: {', '.join(missing)}")
        qt = (payload.question_type or payload.content_type).strip().lower()
        if qt not in {"mcq", "oe"}:
            raise ValueError(f"qgen_crew_runner: unsupported question_type {qt!r} (Phyllis scope is mcq | oe)")

    async def _emit_progress(
        self,
        payload: AiAssistStartedPayload,
        pipeline_trace: list[dict[str, Any]],
        *,
        name: str,
        status: str,
    ) -> None:
        """Best-effort mid-run progress publish (live-trace streaming).

        Publishes ONE chora.creation.ai_assist.progress.v1 carrying the
        cumulative-so-far trace. NEVER raises — a failed progress emit must not
        abort the graph run or suppress the terminal completed.v1 (swallow-and-log
        resilience).
        """
        emitter = self._progress_emitter
        if emitter is None:
            return
        try:
            await emitter.publish_progress(
                assist_id=payload.assist_id,
                tenant_id=payload.tenant_id,
                author_gcid=payload.author_gcid,
                pipeline_trace_json=json.dumps(pipeline_trace),
                step_index=len(pipeline_trace),
                step_name=name,
                step_status=status,
                traceparent=payload.traceparent,
                tracestate=payload.tracestate,
            )
        except Exception:  # noqa: BLE001 — best-effort; never abort the run
            logger.warning(
                "qgen_crew_runner.progress_emit_failed",
                extra={
                    "assist_id": payload.assist_id,
                    "step_index": len(pipeline_trace),
                    "step_name": name,
                },
                exc_info=True,
            )

    async def _emit_agent_decision_log(
        self,
        payload: AiAssistStartedPayload,
        terminal: dict[str, Any],
    ) -> None:
        """Emit TWO AgentDecisionLog events per terminal run (Gate #8) — one
        for ``agid="qgen_question"`` + one for ``agid="qgen_critic"`` — so the
        O+ /o/agents tiles populate per agent. The run-level verdict is shared
        across both; only token counts are split per agent's LLM hop.

        Inspects the terminal state's pipeline_trace for the FINAL
        `quality_gate` row + maps its status → decision string:

          - ``ACCEPTED``         → ``accepted``
          - ``RETRY``            → ``retry`` (rare — runners normally terminate
                                   on the loop's next quality_gate row)
          - ``QUALITY_WARNING``  → ``completed_with_warning``
          - ``REJECTED``/``FAILED`` → ``rejected``

        When NO quality_gate row exists (e.g. guardrail-pre block terminates
        before the gate ever runs), the decision derives from refusal_reason:
          - present → ``refused``
          - absent  → ``accepted`` (terminal happy path that bypassed the gate)

        Per ADR-141 the event is stamped with ``chora_imda_dimension =
        "accountability"`` (D1). The chora-governance projector routes by
        this dimension into accountability_evidence.

        Best-effort: a transient emit failure MUST NOT suppress the
        terminal publish per [[feedback-d6-resilience-first-class]].

        CHO-2364: delegates to the module-level
        ``_emit_qgen_agent_decisions`` so the batch runner emits per-item
        decisions through the SAME logic + condition builders (the batch
        lane previously bypassed this seam entirely and emitted nothing).
        """
        if self._agent_decision_emitter is None:
            return
        await _emit_qgen_agent_decisions(self._agent_decision_emitter, payload, terminal)

    async def _emit_hitl_gate(
        self,
        payload: AiAssistStartedPayload,
        terminal: dict[str, Any],
    ) -> None:
        """Escalate a MAX-RETRIES-EXHAUSTED terminal to a Human-Oversight gate.

        This is the missing piece that left the O+ Human-Oversight queue
        permanently empty: nothing escalated. A ``quality_warning=True``
        terminal (the qgen graph sets it ONLY on the genuinely low-quality
        outcome — the critic kept rejecting until retries were exhausted, OR
        the evaluator self-rejected on the last attempt) is a defensible
        escalation signal: a candidate that cleared the per-stage guardrails
        but is still low-confidence warrants human review (the AI Assist
        Reporter's HITL_LEVEL_0 autonomy per ADR-141).

        Escalation is env-gated via QGEN_HITL_ESCALATE_ON_QUALITY_WARNING
        (default true) per [[feedback-no-inline-config]]. A refused terminal
        is NEVER escalated (a guardrail/validation block is not a quality
        gate). When no HITL emitter is injected this is a no-op
        (M11-baseline backwards-compat).

        The event carries the chora-governance projector's D4 routing fields
        (chora_imda_dimension=fairness_and_human_oversight + event_type
        containing "hitl" + decision_id/run_id/agent_id/summary/
        autonomy_level/created_at) so routeD4 → AppendHITLDecision + gateway
        mapHITLItem render the pending gate.

        Best-effort: a transient emit failure MUST NOT suppress the terminal
        publish per [[feedback-d6-resilience-first-class]].
        """
        if self._hitl_decision_emitter is None:
            return
        if not _coerce_bool_env_default_true(ENV_HITL_ESCALATE_ON_QUALITY_WARNING):
            return

        refusal_reason = str(terminal.get("refusal_reason") or "")
        if refusal_reason:
            # A guardrail/validation block is not a quality escalation.
            return
        quality_warning = bool(terminal.get("quality_warning") or False)
        if not quality_warning:
            return

        attempt_count = int(terminal.get("attempt_count") or 0)
        max_retries = int(payload.max_retries or 0)

        # Summary surfaces WHY the gate fired — mapHITLItem renders it as the
        # FE `summary`. Prefer the most-recent critic verdict when present.
        critic_notes = ""
        completed_candidate = terminal.get("completed_candidate")
        if isinstance(completed_candidate, CandidatePayload):
            critic_notes = completed_candidate.critic_notes or ""
        summary = (
            f"AI Assist candidate flagged for review: quality_warning after "
            f"{attempt_count} attempt(s) (exhausted max_retries={max_retries})."
        )
        if critic_notes:
            summary = f"{summary} Last critic note: {critic_notes[:200]}"

        occurred_at = _dt.datetime.now(tz=_dt.UTC).isoformat()

        try:
            await self._hitl_decision_emitter.emit(
                # decision_id == run_id == assist_id: the qgen runner is
                # invocation-scoped, so the assist invocation IS the gate's
                # correlation key. The gateway maps run_id → workflow_id.
                decision_id=payload.assist_id,
                run_id=payload.assist_id,
                tenant_id=payload.tenant_id,
                gcid=payload.author_gcid,
                agent_id="qgen_crew",
                autonomy_level=_QGEN_HITL_AUTONOMY_LEVEL,
                summary=summary,
                occurred_at=occurred_at,
                crew_name=MCQ_AI_ASSIST_CREW_NAME,
                traceparent=payload.traceparent,
                tracestate=payload.tracestate,
            )
        except Exception:
            # Best-effort. Log and continue so the terminal publish still
            # fires — the outbox row (if the INSERT itself succeeded) stays
            # pending for the dispatcher; a wedged emitter must not abort
            # the whole job.
            logger.exception(
                "qgen_crew_runner.hitl_gate_emit_failed",
                extra={
                    "assist_id": payload.assist_id,
                    "attempt_count": attempt_count,
                    "max_retries": max_retries,
                },
            )

    async def _settle(
        self,
        payload: AiAssistStartedPayload,
        terminal: dict[str, Any],
    ) -> None:
        """Inspect the terminal state and emit the matching event (idempotent:
        the outbox idempotency key is deterministic per assist_id)."""
        refusal_reason = str(terminal.get("refusal_reason") or "")
        completed_candidate = terminal.get("completed_candidate")
        pipeline_trace = terminal.get("pipeline_trace") or []
        attempt_count = int(terminal.get("attempt_count") or 0)

        pipeline_trace_json = json.dumps(pipeline_trace)

        # AgentDecisionLog emit (Gate #8) — one event per terminal run,
        # IMDA D1 accountability dimension per ADR-141. Same rationale as
        # the token-usage emit above: write to outbox BEFORE terminal
        # publish so a mid-run crash leaves the decision row pending for
        # the dispatcher rather than dropped.
        await self._emit_agent_decision_log(payload, terminal)

        # HITL gate emit (Human-Oversight queue feed) — IMDA D4
        # fairness_and_human_oversight. Same write-to-outbox-BEFORE-publish
        # rationale: a mid-run crash leaves the gate pending for the
        # dispatcher rather than dropped. Fires only on the genuinely
        # low-quality (quality_warning) terminal, env-gated.
        await self._emit_hitl_gate(payload, terminal)

        if refusal_reason:
            # REFUSED terminal — guardrail or validation block.
            last_candidate_json = _candidate_to_json(terminal.get("last_candidate_on_refusal"))
            await self._publisher.publish_refused(
                assist_id=payload.assist_id,
                tenant_id=payload.tenant_id,
                author_gcid=payload.author_gcid,
                refusal_reason=refusal_reason,
                model_armor_verdict=str(terminal.get("refusal_armor_verdict") or ""),
                user_facing_message=str(terminal.get("refusal_user_facing_message") or ""),
                last_candidate_payload_json=last_candidate_json,
                pipeline_trace_json=pipeline_trace_json,
                attempt_count=attempt_count,
                mana_charged=0,  # finalised by mana_quoter; orchestrator-time 0
                traceparent=payload.traceparent,
            )
            logger.info(
                "qgen_crew_runner.refused",
                extra={
                    "assist_id": payload.assist_id,
                    "refusal_reason": refusal_reason,
                    "attempt_count": attempt_count,
                },
            )
            return

        # COMPLETED terminal — critic accepted OR retries exhausted.
        candidate_json = _candidate_to_json(completed_candidate)
        critic_notes = ""
        if isinstance(completed_candidate, CandidatePayload):
            critic_notes = completed_candidate.critic_notes
        quality_warning = bool(terminal.get("quality_warning") or False)
        await self._publisher.publish_completed(
            assist_id=payload.assist_id,
            tenant_id=payload.tenant_id,
            author_gcid=payload.author_gcid,
            candidate_payload_json=candidate_json,
            pipeline_trace_json=pipeline_trace_json,
            quality_warning=quality_warning,
            attempt_count=attempt_count,
            critic_notes=critic_notes,
            mana_charged=0,  # finalised post-publish by chora-tenancy mana_quoter
            traceparent=payload.traceparent,
        )
        logger.info(
            "qgen_crew_runner.completed",
            extra={
                "assist_id": payload.assist_id,
                "attempt_count": attempt_count,
                "quality_warning": quality_warning,
            },
        )

    # Back-compat name (tests + the pre-ADR-254 call sites).
    _publish_terminal = _settle


async def _emit_qgen_agent_decisions(
    emitter: Any,
    payload: AiAssistStartedPayload,
    terminal: dict[str, Any],
    *,
    decision_assist_id: str = "",
) -> None:
    """Emit TWO AgentDecisionLog events for ONE qgen graph run - one for
    ``agid="qgen_question"`` + one for ``agid="qgen_critic"`` - shared by
    the single runner (via ``QGenCrewRunner._emit_agent_decision_log``)
    and the batch runner (per N-loop item / per set run, CHO-2364) so both
    lanes ride the SAME decision mapping, condition builders, per-agent
    token split, marker-span minting and prompt-provenance stamp.

    ``decision_assist_id`` is the identity the decision rows carry (the
    outbox idempotency key ``agent_decision.{tenant}.{assist_id}.{agid}``
    dedupes on it). Blank -> ``payload.assist_id`` (single runner + set
    lane); the batch N-loop passes its per-candidate thread id
    ``"{job}:{i}"`` so per-item rows never collide on the unique index.
    ``crew_id`` always stays the job's ``payload.assist_id``.

    Best-effort per agent: each emit is independently guarded so a
    failure on one agent never suppresses the other (or the caller's
    terminal publish) per [[feedback-d6-resilience-first-class]].
    """
    pipeline_trace = terminal.get("pipeline_trace") or []
    refusal_reason = str(terminal.get("refusal_reason") or "")
    attempt_count = int(terminal.get("attempt_count") or 0)
    max_retries = int(payload.max_retries or 0)
    quality_warning = bool(terminal.get("quality_warning") or False)

    # Find the FINAL quality_gate row (there may be many across retries).
    final_gate_row: dict[str, Any] | None = None
    for row in pipeline_trace:
        if not isinstance(row, dict):
            continue
        if str(row.get("name") or "") == "quality_gate":
            final_gate_row = row

    if refusal_reason:
        decision = "refused"
        notes = refusal_reason
    elif final_gate_row is not None:
        gate_status = str(final_gate_row.get("status") or "")
        decision = _QUALITY_GATE_STATUS_TO_DECISION.get(gate_status, "accepted")
        notes = str(final_gate_row.get("notes") or "")
        # Prefer the gate row's own attempt index when present — it
        # represents which attempt the gate was evaluating, even when
        # state's attempt_count drifts.
        gate_attempt = _coerce_int(final_gate_row.get("attempt"))
        if gate_attempt > 0:
            attempt_count = gate_attempt
    else:
        # No gate row + no refusal — bare happy path (e.g. tests with
        # passthrough graphs). Default to accepted.
        decision = "accepted"
        notes = ""

    # critic_notes: prefer the terminal candidate's notes (which carry
    # the most-recent critic verdict), fall back to gate row notes for
    # forensic continuity.
    critic_notes = ""
    completed_candidate = terminal.get("completed_candidate")
    if isinstance(completed_candidate, CandidatePayload):
        critic_notes = completed_candidate.critic_notes or ""
    if not critic_notes:
        critic_notes = notes

    occurred_at = _dt.datetime.now(tz=_dt.UTC).isoformat()

    # is_eval_run sourced from CHORA_IS_EVAL_RUN env (truthy ∈
    # {"1", "true", "yes"} — case-insensitive). Eval pipelines flip
    # this on at job startup so projected rows segregate cleanly
    # from production posture per
    # .claude/skills/mlops-agent-eval/SKILL.md.
    is_eval_run = _coerce_bool_env(ENV_CHORA_IS_EVAL_RUN)

    # guardrail_outcome derived from the pipeline trace's guardrail
    # rows. The qgen_crew graph stamps `guardrail_pre` / `guardrail_post`
    # trace rows with a verdict; we surface the FIRST non-pass verdict
    # (block wins over pass per a layered-policy convention), else
    # "pass" when guardrails ran, else "" when they didn't. Run-level —
    # both agent decisions carry the same verdict.
    guardrail_outcome = _extract_guardrail_outcome(pipeline_trace, refusal_reason)

    # adapter_version reflects the LoRA adapter the LLM call used
    # (per gemma-lora-tenant SKILL). qgen_crew today routes to the
    # base Gemini family; per-tenant LoRA wiring lands in M15. The
    # adapter version, when present, lives on the LLM-hop trace
    # rows as `adapter_version`.
    adapter_version = _extract_adapter_version(pipeline_trace)

    # question_type ("mcq" | "oe") rides every qgen event's attributes so
    # the O+ consumer can split tiles by content type. By the time the
    # terminal emits, _validate has already guaranteed it is one of
    # {mcq, oe} (a graph run on an unsupported type never starts).
    question_type = (payload.question_type or payload.content_type).strip().lower()

    # Citation hashes (PII-safe sha256 hex) for the O+ reasoning-panel
    # citation (IMDA D2): input_hash over the author prompt + output_hash
    # over the generated candidate's canonical JSON. Raw content NEVER
    # leaves the orchestrator — only the one-way hashes ride the event, so
    # an auditor can verify the decision concerned a specific input/output
    # without the panel storing the raw question. Run-level — both agents
    # cite the same input/output (mirrors the run-level verdict + counts).
    # On refusal there is no accepted candidate → fall back to the last
    # candidate seen (else hash of "").
    candidate_for_hash: Any = completed_candidate
    if not isinstance(candidate_for_hash, CandidatePayload):
        candidate_for_hash = terminal.get("last_candidate_on_refusal")
    input_hash = _sha256_hex(payload.prompt)
    output_hash = _sha256_hex(_candidate_to_json(candidate_for_hash))

    # Emit ONE decision per agent (qgen_question + qgen_critic) — the
    # producer previously emitted a single hardcoded "qgen_crew" decision
    # that matched no /o/agents registry tile, so every qgen tile read 0.
    # The run-level verdict (decision / attempt_count / quality_warning /
    # guardrail_outcome / adapter_version) is the crew's terminal outcome,
    # attributed to each member; only the gen_ai.usage.* token counts are
    # split per agent (qgen_question = generate hop, qgen_critic = critique
    # hop) so the two tiles do not double-count the run's tokens. Each emit
    # is best-effort + INDEPENDENTLY guarded so a failure on one agent does
    # NOT suppress the other (or the terminal publish) per
    # [[feedback-d6-resilience-first-class]].
    #
    # ADR-197 M-A.3 — build each agent's prompt-shaping condition map once
    # (pure helpers mirroring the Go QuestionConditions/CriticConditions
    # extractors). The qgen_critic attempt_index is 0-based (attempt_count
    # is 1-based); has_prior_notes is true when a prior critique round ran
    # (attempt_count > 1 ⇒ a re-generation produced notes the critic read).
    question_prompt_conditions = _qgen_question_conditions(payload)
    critic_prompt_conditions = _qgen_critic_conditions(
        question_type=question_type,
        attempt_index=attempt_count - 1,
        has_prior_notes=attempt_count > 1,
        # CHO-2364 - set_mode rides when the job ran the SET lane. ADR-254 D2:
        # every batch does (a plan-less legacy batch gets a synthesised plan),
        # so the run's own set_mode is the honest source, with the wire's plan
        # as the fallback for passthrough terminals; request_surface is the
        # producer-stamped surface hint (the consumption dose lane sends
        # "campaign").
        request_surface=_request_surface(payload),
        set_mode=bool(terminal.get("set_mode") or payload.type_plan),
    )
    # ADR-197 M-B.2 + CHO-2364 - stamp the prompt version + source onto each
    # agent's condition map so the durable AgentDecisionLog records WHICH
    # prompt shaped the decision (rides the M-A.3 field-21 chain - no proto
    # change). The runner stored the resolver values on the initial state
    # before ainvoke; they persist to `terminal` as declared LangGraph
    # channels. EVERY decision stamps: resolver values on the override path,
    # else EMBEDDED_PROMPT_VERSIONS[agid] + "embedded".
    _stamp_prompt_override(question_prompt_conditions, terminal, "question")
    _stamp_prompt_override(critic_prompt_conditions, terminal, "critic")
    for agid, hop in _AGID_TO_LLM_HOP.items():
        prompt_tokens, completion_tokens, cached_tokens = _aggregate_tokens_for(pipeline_trace, hop)
        # Select the agent's condition discriminants (qgen_question vs
        # qgen_critic) so each durable record carries the SAME map the live
        # span stamped for that agent (ADR-197 M-A.3).
        prompt_conditions = question_prompt_conditions if agid == AGID_QGEN_QUESTION else critic_prompt_conditions
        # Per-agent marker span: give THIS agent's decision a DISTINCT span
        # id within the shared run trace so the O+ /o/agents 'View in Cloud
        # Trace' deep-link lands on the agent's OWN span (previously every
        # agent shared payload.traceparent → identical span id, so the link
        # resolved to the crew trace, not the agent). The span ALSO carries
        # the decision EVIDENCE — the verdict + a bounded reasoning summary
        # (§9) — so an auditor who deep-links from O+ Decision-Traces reads
        # WHAT the agent decided + WHY on the span (IMDA D2). PII discipline:
        # only the verdict + critic_notes (the agent's own reasoning) ride
        # the span — never the raw candidate question. Best-effort — returns
        # payload.traceparent unchanged on any failure.
        agent_tp = agent_span_traceparent(
            agid,
            payload.traceparent,
            attributes={
                "chora.question_type": question_type,
                "chora.guardrail_outcome": guardrail_outcome,
                "chora.attempt_count": str(attempt_count),
                "chora.max_retries": str(max_retries),
                "chora.quality_warning": str(quality_warning).lower(),
            },
            decision=decision,
            reasoning_summary=critic_notes,
        )
        try:
            await emitter.emit(
                assist_id=decision_assist_id or payload.assist_id,
                agid=agid,
                question_type=question_type,
                tenant_id=payload.tenant_id,
                gcid=payload.author_gcid,
                decision=decision,
                attempt_count=attempt_count,
                max_retries=max_retries,
                critic_notes=critic_notes,
                quality_warning=quality_warning,
                chora_imda_dimension=IMDA_DIMENSION_ACCOUNTABILITY,
                occurred_at=occurred_at,
                # agent_tp = this agent's own marker-span traceparent (NOT
                # the shared payload.traceparent) → agent-specific deep-link.
                traceparent=agent_tp,
                tracestate=payload.tracestate,
                crew_name=MCQ_AI_ASSIST_CREW_NAME,
                # crew_id = the job's assist_id. Single lane: equals the
                # decision assist_id. Batch lane (CHO-2364): the per-item
                # decision assist_id is "{job}:{i}" while crew_id stays
                # the whole job's id.
                # When a per-instance crew UUIDv7 is introduced (multi-
                # crew composition), bump this to a distinct field.
                crew_id=payload.assist_id,
                # qgen 2-agent crew does not yet support resume-from-
                # interrupt; LangGraph checkpointer resume on pod death
                # is graph-internal (no observable resume hop). Hardcode
                # False until the runner exposes a resume signal.
                is_resume=False,
                is_eval_run=is_eval_run,
                adapter_version=adapter_version,
                guardrail_outcome=guardrail_outcome,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                input_hash=input_hash,
                output_hash=output_hash,
                prompt_conditions=prompt_conditions,
            )
        except Exception:
            # Best-effort. Log and continue so the OTHER agent's decision
            # + the terminal publish still fire.
            logger.exception(
                "qgen_crew_runner.agent_decision_emit_failed",
                extra={
                    "assist_id": decision_assist_id or payload.assist_id,
                    "agid": agid,
                    "decision": decision,
                    "attempt_count": attempt_count,
                },
            )


class QGenBatchRunner:
    """EPIC-1a / CHO-1819 / ADR-251 D1: the batch lane. Every batch job rides
    the chunked SET-native graph and publishes ONE
    chora.creation.ai_assist.completed.v1 whose candidate_payload_json is the
    object wrapper ``{"candidates": [...]}`` (or the composed
    ``{"candidates": [...], "proposed_test_set": {...}}``), with the typed
    ``generated_count`` + honest ``generation_summary`` on the event.

    ADR-254 D2 retired the legacy per-candidate N-loop (thread per candidate,
    bare-array payload): it could not park per dispatch without being
    re-architected, it was unchunked against ADR-251 D1, and chora-creation's
    v2 producer always stamps a type_plan (``NewComposePlan`` builds a
    single-type plan for single-type batches). A plan-less legacy batch is
    given the SAME single-quota plan the producer stamps for count>1 (image
    opt-ins included) and runs chunked like any set.

    Reuses the SAME compiled graph as :class:`QGenCrewRunner`; differences are
    the set state + the set terminal. Per CHO-2364 it emits ONE
    qgen_question + ONE qgen_critic AgentDecisionLog per set run through the
    shared :func:`_emit_qgen_agent_decisions`.

    No-partial-success: a refusal (guardrail / validation block) refuses the
    WHOLE batch (``publish_refused``); the creation side then fails the job +
    refunds. Quality-warning candidates are VALID and included.

    Checkpoint thread = the assist_id. ADR-254 D5: the run parks on every
    agent dispatch; ``handle_completion`` resumes it and the terminal settles
    whichever call reached it.
    """

    # ADR-251 D1 (owner-ruled 2026-08-16): 50 -> 200, kept in lockstep with
    # chora-creation's aiassist.MaxBatchCount (the edge clamp) and
    # validate_type_plan's default max_batch (the graph's defence-in-depth
    # refusal). The chunk loop bounds every generate call regardless, so this
    # ceiling is a product/cost brake, not a token-limit guard.
    _MAX_BATCH = 200

    def __init__(
        self,
        *,
        graph: _CompiledGraphLike,
        publisher: _TerminalPublisher,
        compose_enabled: bool = False,
        agent_decision_emitter: Any | None = None,
        progress_emitter: _ProgressPublisher | None = None,
    ) -> None:
        self._graph = graph
        self._publisher = publisher
        # Live-trace streaming for the SET lane. None (default / flag off) keeps
        # the single-shot ainvoke. Wired => astream + one progress.v1 per node.
        self._progress_emitter = progress_emitter
        # CHO-2364: optional AgentDecisionLogOutboxWriter (Gate #8). When None
        # (unit fixtures / legacy callers) the batch lane emits no decisions.
        self._agent_decision_emitter = agent_decision_emitter
        # Lane 1c (CHO-1703) / ADR-254 D2: the test-set composer is a graph node
        # (compose_test_set, ONE mode=compose dispatch on the qgen_generate lane
        # after finalize_set). This flag threads QGEN_TESTSET_COMPOSE_ENABLED
        # into the set state; off => the batch publishes {"candidates": [...]},
        # on => the BatchCandidatePayload OBJECT with proposed_test_set.
        self._compose_enabled = bool(compose_enabled)

    # ---- start / re-drive -------------------------------------------------

    async def handle_started(self, event: dict[str, Any]) -> None:
        payload = AiAssistStartedPayload.from_event(event)
        self._validate(payload)
        count = self._count(payload)
        files = self._files(payload)
        type_plan = self._type_plan_for(payload, count)
        state = self._build_set_state(payload, type_plan=type_plan, files=files)
        logger.info(
            "qgen_batch_runner.set_started",
            extra={
                "assist_id": payload.assist_id,
                "tenant_id": payload.tenant_id,
                "requested_count": count,
                "type_plan_quotas": len(type_plan),
                "grounding_mode": payload.grounding_mode,
                "source_files": len(files),
                "synthesized_plan": not payload.type_plan,
            },
        )
        terminal, parked = await _drive(
            self._graph,
            state,
            self._run_config(payload, count),
            progress=self._progress_hook(payload),
            skip=_skip_terminal_publishes,
        )
        if parked:
            logger.info("qgen_batch_runner.parked", extra={"assist_id": payload.assist_id})
            return
        await self._settle_set(payload, terminal, count=count, files=files)

    # ---- completion resume ------------------------------------------------

    async def handle_completion(self, completion: Mapping[str, Any], *, started_event: dict[str, Any]) -> ResumeOutcome:
        payload = AiAssistStartedPayload.from_event(started_event)
        thread_id = _require_thread_id(completion, "qgen_batch_runner.handle_completion")
        count = self._count(payload)
        files = self._files(payload)
        run_config = self._run_config(payload, count)
        run_config["configurable"]["thread_id"] = thread_id
        logger.info(
            "qgen_batch_runner.completion",
            extra={
                "assist_id": payload.assist_id,
                "thread_id": thread_id,
                "agent_role": completion.get("agent_role"),
                "status": completion.get("status"),
            },
        )
        terminal, parked = await _resume(
            self._graph,
            run_config,
            completion,
            progress=self._progress_hook(payload),
            skip=_skip_terminal_publishes,
        )
        if parked:
            return ResumeOutcome(parked=True)

        async def _settle_later() -> None:
            await self._settle_set(payload, terminal, count=count, files=files)

        return ResumeOutcome(parked=False, settle=_settle_later)

    # ---- settle -------------------------------------------------------------

    async def _settle_set(
        self,
        payload: AiAssistStartedPayload,
        terminal: dict[str, Any],
        *,
        count: int,
        files: list[dict[str, str]],
    ) -> None:
        """Publish ONE terminal for the set run (idempotent: the outbox
        idempotency key is deterministic per assist_id)."""
        pipeline_trace = terminal.get("pipeline_trace") or []

        # CHO-2364: ONE qgen_question + ONE qgen_critic decision for the set
        # run, through the SAME emitter + condition builders as the single
        # runner. Emitted BEFORE the refusal branch so a refused set's decision
        # is recorded too.
        if self._agent_decision_emitter is not None:
            await _emit_qgen_agent_decisions(self._agent_decision_emitter, payload, terminal)

        refusal_reason = str(terminal.get("refusal_reason") or "")
        if refusal_reason:
            # No-partial-success: a guardrail/validation block fails the batch.
            await self._publisher.publish_refused(
                assist_id=payload.assist_id,
                tenant_id=payload.tenant_id,
                author_gcid=payload.author_gcid,
                refusal_reason=refusal_reason,
                model_armor_verdict=str(terminal.get("refusal_armor_verdict") or ""),
                user_facing_message=str(terminal.get("refusal_user_facing_message") or ""),
                last_candidate_payload_json=_candidate_to_json(terminal.get("last_candidate_on_refusal")),
                pipeline_trace_json=json.dumps(pipeline_trace),
                attempt_count=int(terminal.get("attempt_count") or 0),
                mana_charged=0,
                traceparent=payload.traceparent,
            )
            logger.info(
                "qgen_batch_runner.set_refused",
                extra={"assist_id": payload.assist_id, "refusal_reason": refusal_reason},
            )
            return

        # Stamp the deterministic draft_id + normalise citations over the FINAL
        # accepted set (the graph already deduped + reconciled it).
        accepted = terminal.get("accepted_set") or []
        candidates: list[Any] = []
        for i, obj in enumerate(accepted):
            if isinstance(obj, dict):
                obj = normalise_candidate_citations(obj, files)
                obj["draft_id"] = deterministic_draft_id(payload.assist_id, i)
            candidates.append(obj)

        summary = terminal.get("generation_summary") or {}
        quality_warning = bool(terminal.get("quality_warning") or False)

        # Set-mode payload is always the OBJECT wrapper ({"candidates": [...]} or
        # the composed {"candidates": [...], "proposed_test_set": {...}}); the
        # honest summary rides the TYPED field 16, NOT this JSON wrapper. The
        # proposal (and its compose_test_set trace row) come from the graph's
        # compose_test_set node (ADR-254 D2); its draft ids are the same
        # deterministic ids stamped above.
        proposal = terminal.get("proposed_test_set")
        if isinstance(proposal, dict) and proposal:
            candidate_payload_json = json.dumps({"candidates": candidates, "proposed_test_set": proposal})
        else:
            candidate_payload_json = json.dumps({"candidates": candidates})

        await self._publisher.publish_completed(
            assist_id=payload.assist_id,
            tenant_id=payload.tenant_id,
            author_gcid=payload.author_gcid,
            candidate_payload_json=candidate_payload_json,
            pipeline_trace_json=json.dumps(pipeline_trace),
            quality_warning=quality_warning,
            attempt_count=count,
            critic_notes="",
            mana_charged=0,
            traceparent=payload.traceparent,
            generated_count=int(summary.get("generated_total") or len(candidates)),
            generation_summary=summary,
        )
        logger.info(
            "qgen_batch_runner.set_completed",
            extra={
                "assist_id": payload.assist_id,
                "requested_total": summary.get("requested_total"),
                "generated_total": summary.get("generated_total"),
                "shortfall": bool(summary.get("shortfall_reason")),
                "quality_warning": quality_warning,
            },
        )

    # ---- helpers ------------------------------------------------------------

    def _count(self, payload: AiAssistStartedPayload) -> int:
        return max(1, min(self._MAX_BATCH, int(payload.requested_count or 1)))

    @staticmethod
    def _files(payload: AiAssistStartedPayload) -> list[dict[str, str]]:
        # Lane 1c (CHO-1703): the job's EFFECTIVE grounding-file list: source_files
        # (f20) canonical, f17/18 single-file fallback, [] when ungrounded.
        return effective_source_files(
            source_files=list(payload.source_files),
            source_blob_uri=payload.source_blob_uri,
            source_mime_type=payload.source_mime_type,
        )

    @staticmethod
    def _type_plan_for(payload: AiAssistStartedPayload, count: int) -> list[dict[str, Any]]:
        """The resolved type_plan. EMPTY on the wire (a pre-compose legacy batch)
        => the SAME single-quota plan chora-creation stamps for count>1
        (``question_subscriber.go``: {question_type, count, max_images: 0,
        image_for_stem, image_for_answer}), so the author's per-image opt-ins
        survive onto the set lane exactly as on the producer path."""
        question_type = (payload.question_type or payload.content_type or "mcq").strip().lower()
        quotas: list[Any] = list(payload.type_plan)
        if not quotas:
            entry: dict[str, Any] = {"question_type": question_type, "count": count, "max_images": 0}
            if payload.image_for_stem:
                entry["image_for_stem"] = True
            if payload.image_for_answer:
                entry["image_for_answer"] = True
            quotas = [entry]
        return build_type_plan(quotas, question_type=question_type, requested_count=count)

    @staticmethod
    def _run_config(payload: AiAssistStartedPayload, count: int) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": payload.assist_id},
            # ADR-251 D1 / ADR-254 D12: the chunk loop plus the per-image render
            # loop multiply node transitions well past LangGraph's default of 25.
            "recursion_limit": _set_recursion_limit(count),
        }

    def _progress_hook(self, payload: AiAssistStartedPayload) -> _ProgressHook | None:
        if self._progress_emitter is None:
            return None

        async def _hook(trace: list[dict[str, Any]], name: str, status: str) -> None:
            await self._emit_progress(payload, trace, name=name, status=status)

        return _hook

    def _build_set_state(
        self,
        payload: AiAssistStartedPayload,
        *,
        type_plan: list[dict[str, Any]],
        files: list[dict[str, str]],
    ) -> QGenCrewState:
        """Build the SET-mode graph state (CHO-1819). Generalises
        :meth:`_build_state` — same grounding resolution — plus the set keys
        (set_mode + type_plan + max_regen_rounds) that gate the set-native lane.
        """
        source_blob_uri = payload.source_blob_uri
        source_mime_type = payload.source_mime_type
        if files and not source_blob_uri:
            sources, _rubric = split_source_files(files)
            primary = sources[0] if sources else None
            if primary is not None:
                source_blob_uri = primary["blob_uri"]
                source_mime_type = primary["mime_type"]
        state: QGenCrewState = {
            "job_id": payload.assist_id,
            "tenant_id": payload.tenant_id,
            "gcid": payload.author_gcid,
            "prompt": payload.prompt,
            # Mixed-type batch carries content_type="mixed"; validate_input_node
            # accepts it under set_mode (the quotas validate per-type downstream).
            "question_type": payload.question_type or payload.content_type or "mixed",
            "metadata": payload.metadata,
            "max_retries": payload.max_retries,
            "set_mode": True,
            "type_plan": type_plan,
            "max_regen_rounds": DEFAULT_MAX_REGEN_ROUNDS,
            "compose_enabled": self._compose_enabled,
            "source_blob_uri": source_blob_uri,
            "source_mime_type": source_mime_type,
            "grounding_mode": payload.grounding_mode,
            "target_growth_edges": list(payload.target_growth_edges),
            "pipeline_trace": [],
            "errors": [],
        }
        if files:
            state["source_files"] = files
        return state

    async def _emit_progress(
        self,
        payload: AiAssistStartedPayload,
        pipeline_trace: list[dict[str, Any]],
        *,
        name: str,
        status: str,
    ) -> None:
        """Best-effort mid-run progress publish for the SET lane.

        Same contract as QGenCrewRunner._emit_progress: NEVER raises — a failed
        progress emit must not abort the set run or suppress the terminal
        completed.v1 (swallow-and-log resilience).
        """
        emitter = self._progress_emitter
        if emitter is None:
            return
        try:
            await emitter.publish_progress(
                assist_id=payload.assist_id,
                tenant_id=payload.tenant_id,
                author_gcid=payload.author_gcid,
                pipeline_trace_json=json.dumps(pipeline_trace),
                step_index=len(pipeline_trace),
                step_name=name,
                step_status=status,
                traceparent=payload.traceparent,
                tracestate=payload.tracestate,
            )
        except Exception:  # noqa: BLE001 — best-effort; never abort the run
            logger.warning(
                "qgen_batch_runner.progress_emit_failed",
                extra={
                    "assist_id": payload.assist_id,
                    "step_index": len(pipeline_trace),
                    "step_name": name,
                },
                exc_info=True,
            )

    @staticmethod
    def _validate(payload: AiAssistStartedPayload) -> None:
        missing: list[str] = []
        for name, value in (
            ("assist_id", payload.assist_id),
            ("tenant_id", payload.tenant_id),
            ("author_gcid", payload.author_gcid),
            ("prompt", payload.prompt),
        ):
            if not value.strip():
                missing.append(name)
        if missing:
            raise ValueError(f"qgen_batch_runner: malformed started.v1 — missing: {', '.join(missing)}")
        qt = (payload.question_type or payload.content_type).strip().lower()
        # Every batch rides the set lane (ADR-254 D2): a typed type_plan carries
        # content_type="mixed"; a plan-less legacy batch is single-type. The
        # per-type quotas are validated by the set graph (parse_set_response).
        if qt not in {"mcq", "oe", "mixed"}:
            raise ValueError(f"qgen_batch_runner: unsupported question_type {qt!r} (mcq | oe)")
        if qt == "mixed" and not payload.type_plan:
            # A mixed set has no single type to synthesise a quota for: the
            # per-type quotas MUST ride the wire (the v2 producer stamps them).
            raise ValueError("qgen_batch_runner: content_type 'mixed' requires a type_plan (per-type quotas)")


_GCS_EXT_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "svg": "image/svg+xml",
}


def _gcs_uri_mime(gs_uri: str) -> str:
    """Infer an image MIME from a gs:// object's extension (default image/png).
    The render pipeline keys object names with a content-type-aligned suffix."""
    tail = gs_uri.rsplit(".", 1)
    if len(tail) == 2:
        return _GCS_EXT_MIME.get(tail[1].lower(), "image/png")
    return "image/png"


class ImageRegenRunner:
    """CHO-1822 / ADR-210 / ADR-254 D2: re-authors + re-renders ONE image
    (stem|answer) for a single candidate draft of a parent batch job, THROUGH
    the qgen agents on the dispatch lanes.

    Flow (``orchestrators/image_regen_graph.py``): dispatch the qgen generator
    with ``intent=image_regen`` + the author's CURRENT (edited) stem (+ model
    answer for answer placement) + the refinement instruction -> the agent
    authors ONE ``image_spec {mode, source}`` -> render it (Mermaid: Kroki + the
    kennel's upload; scene: ONE qgen_render dispatch carrying the CURRENT image
    by reference for an EDIT, the returned gs:// signed here) -> publish
    ai_assist.completed.v1 whose ``candidate_payload_json`` is a 1-element
    image-patch array ``[{draft_id, placement, image_url, image_gcs_uri}]`` the
    creation terminal subscriber applies to the parent candidate.

    Fail-soft (never strand the job): a malformed regen, unwired clients, a
    missing original (ADR-210 D3, checked BEFORE any dispatch), an agent/author
    error, an empty authored spec, or a render error publishes refused.v1.
    Park-aware like the other runners: the job parks on the author dispatch
    (and on a scene render) and ``handle_completion`` resumes it.
    """

    def __init__(
        self,
        *,
        graph: Any,
        publisher: Any,
        kroki: Any,
        gcs: Any,
        image_downloader: Any = None,
    ) -> None:
        self._graph = graph
        self._publisher = publisher
        self._kroki = kroki
        self._gcs = gcs
        # ADR-210 image-to-image: the CURRENT image is EDITED by the renderer,
        # which reads it by gs:// reference. FAIL-LOUD (D3): when an original
        # WAS requested (regen.original_image_gcs_uri set) but is gone (aged out
        # of the transient bucket TTL / IAM / missing), the runner REFUSES with
        # an explicit author message BEFORE dispatching; never a silent
        # text-redraw of a different image.
        self._image_downloader = image_downloader

    async def handle_started(self, event: dict[str, Any]) -> None:
        payload = AiAssistStartedPayload.from_event(event)
        regen = payload.regen or {}
        draft_id = str(regen.get("draft_id") or "").strip()
        placement = str(regen.get("placement") or "").strip()
        refinement = str(regen.get("prompt") or "").strip()
        if not draft_id or placement not in ("stem", "answer") or not refinement:
            await self._refuse(payload, "image_regen_bad_request", "The image regenerate request was malformed.")
            return
        if self._kroki is None or self._gcs is None:
            await self._refuse(payload, "image_regen_unwired", "Image generation is not configured.")
            return
        original_uri = str(regen.get("original_image_gcs_uri") or "").strip()
        if original_uri:
            # The URI is echoed from the CLIENT's request body, and the renderer
            # that would read it holds project-level storage.objectViewer, so an
            # unconstrained one is a cross-tenant read with an image-to-image
            # output channel. Pin it to this run's own bucket and tenant prefix
            # BEFORE anything else touches it.
            #
            # The tenant is safe to key on precisely because it does NOT share
            # the URI's provenance: chora-creation stamps tenant_id from the
            # authenticated request context before publishing. The attacker
            # shapes the URI; they do not choose the tenant they run as.
            try:
                original_uri = require_tenant_scoped_source_uri(
                    original_uri,
                    bucket=str(getattr(self._gcs, "bucket_name", "") or ""),
                    tenant_id=payload.tenant_id,
                )
            except SourceUriNotPermittedError as exc:
                # No URI in the log line either: keep the blast radius of a
                # probing attempt to the ids that identify the run.
                logger.warning(
                    "image_regen_runner.source_uri_refused",
                    extra={
                        "assist_id": payload.assist_id,
                        "draft_id": draft_id,
                        "placement": placement,
                        "reason": str(exc),
                    },
                )
                await self._refuse(
                    payload,
                    "image_regen_source_not_permitted",
                    "That image cannot be used as the source for this edit. Please regenerate the image from scratch.",
                )
                return
        # ORDER IS THE POINT: the existence probe runs only on a URI already
        # pinned to the caller's own prefix. Run the other way round and the
        # probe confirms readability of an attacker-shaped object, which is a
        # discovery oracle on top of the read. Kept rather than dropped because
        # the bucket has a 7-day TTL, so "your original aged out" is a real and
        # common case that deserves its own message instead of a render failure.
        if original_uri and not await self._original_available(original_uri):
            logger.warning(
                "image_regen_runner.original_image_unavailable",
                extra={"assist_id": payload.assist_id, "draft_id": draft_id, "placement": placement},
            )
            await self._refuse(
                payload,
                "image_regen_original_unavailable",
                "The original image is no longer available to edit. Please regenerate it from scratch.",
            )
            return
        state: dict[str, Any] = {
            "assist_id": payload.assist_id,
            "tenant_id": payload.tenant_id,
            "gcid": payload.author_gcid,
            "question_type": payload.question_type or "mcq",
            "traceparent": payload.traceparent,
            "tracestate": payload.tracestate,
            "draft_id": draft_id,
            "placement": placement,
            "refinement_prompt": refinement,
            "current_stem": str(regen.get("current_stem") or ""),
            "current_model_answer": str(regen.get("current_model_answer") or ""),
            "original_mode": str(regen.get("mode") or ""),
            "original_source": str(regen.get("original_source") or ""),
            "original_image_gcs_uri": original_uri,
            "original_image_mime": _gcs_uri_mime(original_uri) if original_uri else "",
        }
        terminal, parked = await _drive(self._graph, state, self._run_config(payload))
        if parked:
            logger.info("image_regen_runner.parked", extra={"assist_id": payload.assist_id})
            return
        await self._settle(payload, terminal)

    async def handle_completion(self, completion: Mapping[str, Any], *, started_event: dict[str, Any]) -> ResumeOutcome:
        payload = AiAssistStartedPayload.from_event(started_event)
        thread_id = _require_thread_id(completion, "image_regen_runner.handle_completion")
        run_config = self._run_config(payload)
        run_config["configurable"]["thread_id"] = thread_id
        terminal, parked = await _resume(self._graph, run_config, completion)
        if parked:
            return ResumeOutcome(parked=True)

        async def _settle_later() -> None:
            await self._settle(payload, terminal)

        return ResumeOutcome(parked=False, settle=_settle_later)

    async def _settle(self, payload: AiAssistStartedPayload, terminal: Mapping[str, Any]) -> None:
        reason = str(terminal.get("refusal_reason") or "")
        if reason:
            await self._refuse(
                payload,
                reason,
                str(terminal.get("refusal_message") or "The image could not be regenerated. Please try again."),
            )
            return
        # ADR-210: carry the NEW durable gs:// object path alongside the signed
        # display URL so a SUBSEQUENT regen edits THIS result (iterative
        # image-to-image), not the stale original.
        candidate_payload = json.dumps(
            [
                {
                    "draft_id": str(terminal.get("draft_id") or ""),
                    "placement": str(terminal.get("placement") or ""),
                    "image_url": str(terminal.get("image_url") or ""),
                    "image_gcs_uri": str(terminal.get("image_gcs_uri") or ""),
                }
            ]
        )
        await self._publisher.publish_completed(
            assist_id=payload.assist_id,
            tenant_id=payload.tenant_id,
            author_gcid=payload.author_gcid,
            candidate_payload_json=candidate_payload,
            pipeline_trace_json="{}",
            quality_warning=False,
            attempt_count=1,
            critic_notes="",
            mana_charged=0,
            traceparent=payload.traceparent,
            tracestate=payload.tracestate,
        )

    async def _original_available(self, gs_uri: str) -> bool:
        if self._image_downloader is None:
            return False
        try:
            return bool(await self._image_downloader.exists(gs_uri))
        except Exception:
            logger.exception("image_regen_runner.original_check_failed", extra={"gs_uri": gs_uri[:200]})
            return False

    @staticmethod
    def _run_config(payload: AiAssistStartedPayload) -> dict[str, Any]:
        return {"configurable": {"thread_id": payload.assist_id}}

    async def _refuse(self, payload: AiAssistStartedPayload, reason: str, message: str) -> None:
        await self._publisher.publish_refused(
            assist_id=payload.assist_id,
            tenant_id=payload.tenant_id,
            author_gcid=payload.author_gcid,
            refusal_reason=reason,
            model_armor_verdict="armor:unspecified",
            user_facing_message=message,
            last_candidate_payload_json="[]",
            pipeline_trace_json="{}",
            attempt_count=1,
            mana_charged=0,
            traceparent=payload.traceparent,
            tracestate=payload.tracestate,
        )


# Compose-model route lanes — the three qgen runner destinations.
_ROUTE_SINGLE = "single"
_ROUTE_BATCH = "batch"
_ROUTE_IMAGE_REGEN = "image_regen"


def _route_for_started(event: dict[str, Any]) -> str:
    """Resolve the qgen runner lane for an ai_assist.started event,
    version-agnostically (ADR-195 WS7 D7).

    v1 carries the legacy ``job_kind`` discriminant
    ("" | "single" | "batch" | "image_regen"). v2 DROPS job_kind and carries the
    compose model {operation, intent, input_kind}; batch-vs-single is then derived
    from the orthogonal seed fields (input_kind / requested_count / type_plan) —
    EXACTLY the conditions under which the chora-creation producer set
    ``job_kind=batch``: source files → RAG batch; count>1 or a non-empty type_plan
    → prompt-batch; otherwise the single-candidate path. Both wires are published
    during the cutover, so v1 routing MUST stay byte-for-byte identical.

    Returns one of ``_ROUTE_IMAGE_REGEN`` | ``_ROUTE_BATCH`` | ``_ROUTE_SINGLE``.
    """
    intent = str(event.get("intent") or "").strip().lower()
    job_kind = str(event.get("job_kind") or "").strip().lower()

    # image_regen — a post-generation parent-patch, never the full qgen graph.
    # v2 carries intent=image_regen; v1 carries job_kind=image_regen.
    if intent == _ROUTE_IMAGE_REGEN or job_kind == _ROUTE_IMAGE_REGEN:
        return _ROUTE_IMAGE_REGEN

    # v1 — the explicit job_kind discriminant is authoritative when present.
    if job_kind:
        return _ROUTE_BATCH if job_kind == _ROUTE_BATCH else _ROUTE_SINGLE

    # v2 (no job_kind) — derive batch-vs-single from the compose seed the producer
    # used. requested_count / type_plan are orthogonal to the discriminant and
    # STAY on the v2 wire; input_kind=source_files marks the RAG batch.
    input_kind = str(event.get("input_kind") or "").strip().lower()
    requested_count = _coerce_int(event.get("requested_count"))
    type_plan = event.get("type_plan") or []
    has_type_plan = isinstance(type_plan, (list, tuple)) and len(type_plan) > 0
    if input_kind == "source_files" or requested_count > 1 or has_type_plan:
        return _ROUTE_BATCH
    return _ROUTE_SINGLE


class QGenRunnerRouter:
    """Routes an ai_assist.started event to the single-candidate
    :class:`QGenCrewRunner`, the :class:`QGenBatchRunner`, or the
    :class:`ImageRegenRunner`, and a dispatch COMPLETION back to the runner
    that parked it (ADR-254 D5).

    Version-agnostic (ADR-195 WS7 D7): a v1 event routes on the legacy
    ``job_kind`` discriminant; a v2 event (no job_kind) routes on the explicit
    compose model {intent, input_kind} + the orthogonal seed fields
    (requested_count / type_plan), preserving single/batch/image_regen parity (see
    :func:`_route_for_started`). The agent layer is UNCHANGED (ADR-195 D2).

    A completion carries only its thread id; the job's started payload is read
    back from the in-flight registry (the durable copy that outlives the drive)
    and routed EXACTLY as its start was, so the same runner settles it with the
    same publisher fields. A job that is no longer in flight cannot be resumed
    and is refused loudly (the completion NACKs, redelivers, then dead-letters;
    the reaper settles the park). An image_regen route with no image_regen
    runner wired falls back to the single path (unchanged from before).
    """

    def __init__(
        self,
        *,
        single: Any,
        batch: Any,
        image_regen: Any = None,
        inflight_registry: Any = None,
    ) -> None:
        self._single = single
        self._batch = batch
        self._image_regen = image_regen
        self._registry = inflight_registry

    def _runner_for(self, event: dict[str, Any]) -> Any:
        route = _route_for_started(event)
        if route == _ROUTE_IMAGE_REGEN and self._image_regen is not None:
            return self._image_regen
        if route == _ROUTE_BATCH:
            return self._batch
        return self._single

    async def handle_started(self, event: dict[str, Any]) -> None:
        await self._runner_for(event).handle_started(event)

    async def handle_completion(self, completion: Mapping[str, Any]) -> ResumeOutcome:
        thread_id = _require_thread_id(completion, "qgen_runner_router.handle_completion")
        # Every qgen lane parks on a thread keyed by the assist id; the legacy
        # per-candidate ``{assist_id}:{i}`` shape is tolerated for any thread
        # that predates the N-loop's retirement.
        assist_id = thread_id.split(":", 1)[0]
        if self._registry is None:
            raise RuntimeError(
                "qgen_runner_router: an in-flight registry is required to resume a completion "
                "(the started payload is read back from it); refusing to guess the job"
            )
        row = await self._registry.get(assist_id)
        if row is None:
            raise LookupError(
                f"qgen_runner_router: completion for thread {thread_id!r} names assist {assist_id!r}, "
                "which is not in flight (already settled, or never accepted); refusing to resume"
            )
        event = dict(row.get("started_payload") or {})
        return await self._runner_for(event).handle_completion(completion, started_event=event)


def _coerce_int(value: Any) -> int:
    """Defensive int coercion for trace-row fields. The trace_row dict comes
    from LangGraph state which is a TypedDict — at runtime, values can be
    int / str / None. Anything not convertible falls back to 0.
    """
    if value is None:
        return 0
    if isinstance(value, bool):  # bool is a subclass of int; reject explicitly
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_bool_env(env_var: str) -> bool:
    """Read an env var and coerce to bool. Truthy ∈ {"1", "true", "yes"}
    case-insensitive — matches the convention used by chora-go-common's
    BoolFromEnv on the Go side so a deployment toggling CHORA_IS_EVAL_RUN
    behaves identically across the Python orchestrator + Go executors.
    """
    raw = (_os.environ.get(env_var) or "").strip().lower()
    return raw in ("1", "true", "yes")


def _coerce_bool_env_default_true(env_var: str) -> bool:
    """Like ``_coerce_bool_env`` but DEFAULTS TRUE when the env var is unset
    or blank. Used for opt-OUT toggles (the HITL escalation is on by default;
    operators set "false"/"0"/"no" to disable it). An unrecognised value
    other than the explicit falsy set is treated as truthy (fail-on).
    """
    raw = (_os.environ.get(env_var) or "").strip().lower()
    if raw == "":
        return True
    return raw not in ("0", "false", "no", "off")


def _aggregate_tokens_for(
    pipeline_trace: list[dict[str, Any]],
    hop_name: str,
) -> tuple[int, int, int]:
    """Sum ``input_tokens`` / ``output_tokens`` / ``cached_tokens`` across the
    trace rows for ONE LLM hop (``generate`` OR ``critique``).

    Returns ``(prompt_tokens, completion_tokens, cached_tokens)`` — the proto's
    gen_ai.usage.* triple, scoped to a single agent so the qgen_question +
    qgen_critic decisions carry their OWN token counts (no cross-tile
    double-count). A hop may appear on multiple rows across retries; they sum.
    Falls back to 0 when the executor doesn't break out per-hop tokens (M11
    baseline: only tokens_consumed_total is emitted).
    """
    prompt = 0
    completion = 0
    cached = 0
    for row in pipeline_trace:
        if not isinstance(row, dict):
            continue
        if str(row.get("name") or "") != hop_name:
            continue
        prompt += _coerce_int(row.get("input_tokens"))
        completion += _coerce_int(row.get("output_tokens"))
        cached += _coerce_int(row.get("cached_tokens"))
    return prompt, completion, cached


def _aggregate_tokens(
    pipeline_trace: list[dict[str, Any]],
) -> tuple[int, int, int]:
    """Run-level total: sum ``input_tokens`` / ``output_tokens`` /
    ``cached_tokens`` across ALL LLM-issuing trace rows (``generate`` +
    ``critique``). Retained for run-level cost views; the per-agent decision
    emit uses ``_aggregate_tokens_for`` to attribute counts per agent.
    """
    prompt = 0
    completion = 0
    cached = 0
    for hop in _LLM_HOP_TRACE_NAMES:
        p, c, cache = _aggregate_tokens_for(pipeline_trace, hop)
        prompt += p
        completion += c
        cached += cache
    return prompt, completion, cached


def _extract_guardrail_outcome(
    pipeline_trace: list[dict[str, Any]],
    refusal_reason: str,
) -> str:
    """Derive the per-decision Cloud Model Armor verdict surface ∈
    ``{"pass", "block", "redact", ""}`` from the terminal pipeline_trace.

    Empty string when guardrails were not invoked (non-LLM-issuing
    agents — e.g. synthetic test traces that omit guardrail rows). The
    qgen_crew graph stamps ``guardrail_pre`` / ``guardrail_post`` trace
    rows with status ∈ {ACCEPTED, REFUSED, FAILED}; a refused row OR a
    refusal_reason ∈ {GUARDRAIL_PRE, GUARDRAIL_POST} yields ``"block"``.
    Otherwise ``"pass"`` when at least one guardrail row ran.
    """
    if refusal_reason in ("GUARDRAIL_PRE", "GUARDRAIL_POST"):
        return "block"
    saw_guardrail_row = False
    for row in pipeline_trace:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if not name.startswith("guardrail_"):
            continue
        saw_guardrail_row = True
        status = str(row.get("status") or "").upper()
        if status in ("REFUSED", "BLOCK", "BLOCKED", "REJECTED"):
            return "block"
        if status in ("REDACT", "REDACTED"):
            return "redact"
    return "pass" if saw_guardrail_row else ""


def _extract_adapter_version(pipeline_trace: list[dict[str, Any]]) -> str:
    """Read the first non-empty ``adapter_version`` stamped on an LLM-
    hop trace row. Empty string when the base model was used (the
    current production default until M15 per-tenant LoRA wiring lands).
    """
    for row in pipeline_trace:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if name not in _LLM_HOP_TRACE_NAMES:
            continue
        adapter = str(row.get("adapter_version") or "").strip()
        if adapter:
            return adapter
    return ""


# suffix used in the per-agent state keys -> the agent's registry id (agid),
# which keys EMBEDDED_PROMPT_VERSIONS for the always-stamp fallback below.
_PROMPT_SUFFIX_TO_AGID: dict[str, str] = {
    "question": AGID_QGEN_QUESTION,
    "critic": AGID_QGEN_CRITIC,
}


def _stamp_prompt_override(conditions: dict[str, str], terminal: dict[str, Any], suffix: str) -> None:
    """ADR-197 M-B.2 + CHO-2364 - stamp ``prompt_version`` + ``prompt_source``
    onto a prompt condition map IN PLACE for EVERY decision.

    Reads the per-agent keys the runner stored on state before the graph ran
    (``prompt_version_<suffix>`` / ``prompt_source_<suffix>`` - persisted to the
    terminal state as declared LangGraph channels). When the resolver provided a
    version (override path) the resolver values are kept verbatim. When it did
    not (no resolver wired OR the embedded default won), the stamp falls back to
    ``EMBEDDED_PROMPT_VERSIONS[agid]`` + source ``"embedded"`` so the durable
    record always carries prompt provenance - a decision row is never
    version-blank.
    """
    version = str(terminal.get(f"prompt_version_{suffix}") or "")
    source = str(terminal.get(f"prompt_source_{suffix}") or "")
    if not version:
        version = EMBEDDED_PROMPT_VERSIONS[_PROMPT_SUFFIX_TO_AGID[suffix]]
        source = SOURCE_EMBEDDED
    conditions["prompt_version"] = version
    if source:
        conditions["prompt_source"] = source


def _request_surface(payload: AiAssistStartedPayload) -> str:
    """CHO-2364 - the originating surface hint the producer stamped on
    ``metadata["surface"]`` (the consumption dose lane sends ``"campaign"``;
    authoring stamps none). Blank when absent or whitespace - callers OMIT the
    discriminant then (never fabricated).
    """
    return str((payload.metadata or {}).get("surface") or "").strip()


def _qgen_question_conditions(payload: AiAssistStartedPayload) -> dict[str, str]:
    """ADR-197 M-A.3 — the qgen_question prompt-shaping condition discriminants
    the orchestrator genuinely holds, mirroring the Go ``QuestionConditions``
    extractor (``agents/qgen_adk_go/internal/agent/conditions.go``) key-for-key so
    the durable record + the live agent span agree.

    Keys: ``intent`` (defaults to ``new_question``); ``set_mode`` (``"true"``,
    omits ``question_type``) when a typed type_plan drives the pass, else
    ``question_type`` (lowercased); ``subject_hint`` / ``cognitive_level_hint``
    / ``difficulty_hint`` from the author metadata (the SAME map the executor's
    ``_stamp_metadata_hints`` reads); ``image_for_stem`` / ``image_for_answer``
    (``"true"`` only when the author opted in); ``request_surface`` (CHO-2364)
    from ``metadata["surface"]`` - the consumption dose lane stamps
    ``"campaign"``, authoring stamps none. Blanks/false are OMITTED - every
    emitted value is non-empty (no fabricated discriminants).
    """
    conditions: dict[str, str] = {
        "intent": (payload.intent or "").strip() or "new_question",
    }
    # In set mode the per-type plan is the source of truth; the single
    # question_type is moot, so surface the mode instead (mirrors the Go).
    if payload.type_plan:
        conditions["set_mode"] = "true"
    else:
        conditions["question_type"] = (payload.question_type or payload.content_type).strip().lower()
    metadata = payload.metadata or {}
    for src_key, dst_key in (
        ("subject", "subject_hint"),
        ("cognitive_level", "cognitive_level_hint"),
        ("difficulty", "difficulty_hint"),
    ):
        val = str(metadata.get(src_key) or "").strip()
        if val:
            conditions[dst_key] = val
    if payload.image_for_stem:
        conditions["image_for_stem"] = "true"
    if payload.image_for_answer:
        conditions["image_for_answer"] = "true"
    surface = _request_surface(payload)
    if surface:
        conditions["request_surface"] = surface
    return conditions


def _qgen_critic_conditions(
    *,
    question_type: str,
    attempt_index: int,
    has_prior_notes: bool,
    request_surface: str = "",
    set_mode: bool = False,
) -> dict[str, str]:
    """ADR-197 M-A.3 — the qgen_critic prompt-shaping condition discriminants,
    mirroring the Go ``CriticConditions`` extractor. ``question_type`` is always
    present; ``attempt_index`` is the 0-based attempt counter (the Go critic's
    ``tc.AttemptIndex`` convention — 0 = first attempt); ``has_prior_notes`` is
    emitted (``"true"``) only when a prior critique round produced notes the
    critic conditioned on (i.e. a re-generation occurred).

    CHO-2364 additions: ``set_mode`` (``"true"`` only when the job ran with a
    typed type_plan - the batch SET lane) and ``request_surface`` (the
    producer-stamped ``metadata["surface"]``, e.g. ``"campaign"`` from the
    consumption dose lane). Both default off/blank and are then OMITTED, so
    single-run records are byte-identical to the pre-CHO-2364 shape.
    """
    conditions: dict[str, str] = {
        "question_type": question_type,
        "attempt_index": str(max(attempt_index, 0)),
    }
    if has_prior_notes:
        conditions["has_prior_notes"] = "true"
    if set_mode:
        conditions["set_mode"] = "true"
    request_surface = (request_surface or "").strip()
    if request_surface:
        conditions["request_surface"] = request_surface
    return conditions


def _sha256_hex(s: str) -> str:
    """Lowercase hex sha256 of ``s`` (utf-8). Deterministic; same-input →
    same-hash. Mirrors chora-observability ``decision.sha256Hex`` so the
    citation hashes the producer emits are byte-identical to what the audit
    store would compute over the same canonical text. ``None`` → hash of "".
    """
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _candidate_to_json(candidate: Any) -> str:
    """Serialise a CandidatePayload to its OpenAPI AiAssistCandidate
    JSON shape. None → empty string (the empty case is meaningful
    on REFUSED pre-screens where no candidate was produced).
    """
    if candidate is None:
        return ""
    if isinstance(candidate, CandidatePayload):
        # CandidatePayload.payload_json is already the full
        # AiAssistCandidate JSON (including stem + mcq_payload OR
        # oe_payload + critic_notes). It was produced by the generate
        # node (which writes whatever qgen_question's output_payload
        # was). Re-emit verbatim.
        return candidate.payload_json
    if isinstance(candidate, dict):
        return json.dumps(candidate)
    return json.dumps({"stem": "", "question_type": "", "raw": str(candidate)})


def _candidate_to_obj(candidate: Any) -> Any:
    """Parse a CandidatePayload into its AiAssistCandidate dict for inclusion in
    a batch candidate ARRAY. Falls back to a minimal object on None / unparseable
    so json.dumps(list) never raises and the array stays well-formed."""
    if isinstance(candidate, CandidatePayload):
        try:
            return json.loads(candidate.payload_json)
        except (ValueError, TypeError):
            return {"stem": candidate.stem, "question_type": candidate.question_type}
    if isinstance(candidate, dict):
        return candidate
    return {"stem": "", "question_type": ""}


__all__ = [
    "AiAssistStartedPayload",
    "ImageRegenRunner",
    "QGenBatchRunner",
    "TOPIC_AI_ASSIST_COMPLETED",
    "TOPIC_AI_ASSIST_REFUSED",
    "QGenRunnerRouter",
    "ENV_CHORA_IS_EVAL_RUN",
    "ENV_HITL_ESCALATE_ON_QUALITY_WARNING",
    "IMDA_DIMENSION_ACCOUNTABILITY",
    "IMDA_DIMENSION_FAIRNESS_HUMAN_OVERSIGHT",
    "MCQ_AI_ASSIST_CREW_NAME",
    "QGenCrewRunner",
    "TOPIC_GOVERNANCE_HITL_REQUESTED",
    "TOPIC_OBSERVABILITY_AGENT_DECISION",
]
