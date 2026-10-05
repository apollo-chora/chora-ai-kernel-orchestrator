"""oe_grading_crew_runner unit tests — RED→GREEN per [[feedback-strict-tdd]].

Covers parse_requested (submission_requested.v1 → OEGradingState) + the runner's
handle_requested → graph.ainvoke(thread_id=submission_id) → publisher.publish_
submission_completed path. NO Pub/Sub, NO gRPC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.domain.oe_grading_crew.state import (
    GradedOEQuestion,
)
from chora_ai_kernel_orchestrator.orchestrators.oe_grading_crew_runner import (
    OEGradingCrewRunner,
    parse_requested,
)

# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


@dataclass
class _FakeGraph:
    terminal: dict[str, Any]
    invocations: list[dict[str, Any]] = field(default_factory=list)

    async def ainvoke(self, state: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.invocations.append({"state": state, "config": config})
        return self.terminal


@dataclass
class _FakePublisher:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def publish_submission_completed(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "row-1"


@dataclass
class _FakeAgentDecisionLogEmitter:
    """Records each emitted AgentDecisionLog event in memory."""

    emitted: list[dict[str, Any]] = field(default_factory=list)

    async def emit(self, **kwargs: Any) -> str:
        self.emitted.append(kwargs)
        return f"adl-{len(self.emitted)}"


def _graded(qid: str, points: float, flagged: bool = False) -> GradedOEQuestion:
    return GradedOEQuestion(
        test_set_question_id=f"tsq-{qid}",
        question_id=f"q-{qid}",
        points_earned=points,
        points_possible=5,
        criterion_scores_json=json.dumps([{"criterion_id": "c1", "score": 4.0}]),
        comment="ok",
        grading_model_id="gemini-3.1-pro-preview",
        grading_response_id="resp-1",
        quality_flagged=flagged,
        attempt_count=1,
    )


def _terminal(*, outcome: str = "SUCCESS", graded: list[GradedOEQuestion] | None = None) -> dict[str, Any]:
    return {
        "submission_id": "sub-1",
        "assessment_id": "assess-1",
        "tenant_id": "tenant-1",
        "gcid": "learner-1",
        "graded": graded if graded is not None else [_graded("a", 4.0)],
        "overall_comment": "Overall 80%; strong work.",
        "overall_comment_model_id": "gemini-3.1-pro-preview",
        "outcome": outcome,
        "failure_message": "",
    }


def _event() -> dict[str, Any]:
    # Mirrors chora-delivery emitSubmissionRequestedEvent — UPPERCASE question_type.
    return {
        "grading_job_id": "job-1",
        "submission_id": "sub-1",
        "assessment_id": "assess-1",
        "tenant_id": "tenant-1",
        "learner_gcid": "learner-1",
        "passing_threshold_percent": 50,
        "model_tier": "T1",
        "per_question_feedback_enabled": True,
        "subject": "science",
        "total_points_possible": 10,
        "mcq_points_earned": 4.0,
        "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "questions": [
            {
                "test_set_question_id": "tsq-a",
                "question_id": "q-a",
                "question_type": "OE",
                "points_possible": 5,
                "prompt": "Explain photosynthesis.",
                "subject": "science",
                "topic": "photosynthesis",
                "oe_response_text": "Plants convert light to energy.",
                "rubric_json": json.dumps([{"criterion_id": "c1", "weight": 1.0}]),
                "model_answer": "Reference.",
            },
            {
                "test_set_question_id": "tsq-m",
                "question_id": "q-m",
                "question_type": "MCQ",
                "points_possible": 5,
                "mcq_correct": True,
                "mcq_points_earned": 4.0,
            },
        ],
    }


# -----------------------------------------------------------------------------
# parse_requested
# -----------------------------------------------------------------------------


def test_parse_requested_lifts_oe_and_mcq_uppercase() -> None:
    state = parse_requested(_event())
    assert state["submission_id"] == "sub-1"
    assert state["tenant_id"] == "tenant-1"
    assert state["gcid"] == "learner-1"
    # UPPERCASE "OE"/"MCQ" from delivery must still be projected (case-fold fix).
    assert len(state["oe_questions"]) == 1
    assert state["oe_questions"][0].test_set_question_id == "tsq-a"
    assert state["oe_questions"][0].learner_response == "Plants convert light to energy."
    assert len(state["mcq_results"]) == 1
    assert state["mcq_results"][0].correct is True
    assert state["mcq_points_earned"] == pytest.approx(4.0)
    assert state["total_points_possible"] == 10


def test_parse_requested_lowercase_also_works() -> None:
    ev = _event()
    for q in ev["questions"]:
        q["question_type"] = q["question_type"].lower()
    state = parse_requested(ev)
    assert len(state["oe_questions"]) == 1
    assert len(state["mcq_results"]) == 1


# -----------------------------------------------------------------------------
# handle_requested
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_requested_invokes_graph_and_publishes() -> None:
    graph = _FakeGraph(terminal=_terminal())
    pub = _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub)

    await runner.handle_requested(_event())

    # thread_id == submission_id (D6 P1 pod-death recovery key).
    assert len(graph.invocations) == 1
    assert graph.invocations[0]["config"]["configurable"]["thread_id"] == "sub-1"

    assert len(pub.calls) == 1
    c = pub.calls[0]
    assert c["submission_id"] == "sub-1"
    assert c["assessment_id"] == "assess-1"
    assert c["learner_gcid"] == "learner-1"
    assert c["overall_comment"] == "Overall 80%; strong work."
    assert c["outcome"] == "SUCCESS"
    assert len(c["graded"]) == 1
    assert c["graded"][0]["test_set_question_id"] == "tsq-a"
    assert c["graded"][0]["points_earned"] == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_handle_requested_propagates_partial_outcome() -> None:
    graph = _FakeGraph(
        terminal=_terminal(
            outcome="PARTIAL",
            graded=[_graded("a", 0.0, flagged=True), _graded("b", 3.0)],
        )
    )
    pub = _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub)

    await runner.handle_requested(_event())

    assert pub.calls[0]["outcome"] == "PARTIAL"
    assert len(pub.calls[0]["graded"]) == 2
    assert any(g["quality_flagged"] for g in pub.calls[0]["graded"])


@pytest.mark.asyncio
async def test_handle_requested_missing_submission_id_returns_without_publish() -> None:
    graph = _FakeGraph(terminal=_terminal())
    pub = _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub)
    bad = _event()
    bad["submission_id"] = ""

    # Committed runner logs + returns (does NOT raise); never invokes graph/publish.
    await runner.handle_requested(bad)
    assert graph.invocations == []
    assert pub.calls == []


# -----------------------------------------------------------------------------
# Per-agent AgentDecisionLog emission (oe_evaluator + oe_moderator)
#
# O+ /o/agents OE-grading tiles read agent_decision_log rows keyed on agid. The
# OE runner emits ONE oe_evaluator (the grade) + ONE oe_moderator (moderation)
# decision per submission, carrying the REAL per-request tenant_id (NEVER the
# legacy "qgen_crew"). Best-effort — a transient emit failure MUST NOT suppress
# the terminal submission_completed publish.
# -----------------------------------------------------------------------------


def _terminal_with_trace(*, any_flagged: bool = False) -> dict[str, Any]:
    """A terminal with a pipeline_trace carrying per-hop token counts so the
    per-agent token split (evaluator = evaluate + assess_summary; moderator =
    moderate) is exercised."""
    t = _terminal(graded=[_graded("a", 5.0, flagged=any_flagged), _graded("b", 3.0)])
    t["pipeline_trace"] = [
        {"name": "validate_input", "status": "OK"},
        {"name": "guardrail_pre", "status": "ALLOW", "index": 0},
        {"name": "evaluate", "status": "OK", "index": 0, "attempt": 1, "input_tokens": 120, "output_tokens": 40},
        {"name": "guardrail_post", "status": "ALLOW", "index": 0},
        {"name": "moderate", "status": "ACCEPT", "index": 0, "attempt": 1, "input_tokens": 60, "output_tokens": 20},
        {"name": "assess_summary", "status": "OK", "input_tokens": 30, "output_tokens": 50},
        {"name": "publish_completed", "status": "OK", "graded": 2},
    ]
    t["max_iterations"] = 2
    return t


@pytest.mark.asyncio
async def test_handle_requested_emits_oe_evaluator_and_moderator() -> None:
    graph = _FakeGraph(terminal=_terminal_with_trace())
    pub = _FakePublisher()
    adl = _FakeAgentDecisionLogEmitter()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub, agent_decision_emitter=adl)

    await runner.handle_requested(_event())

    assert len(adl.emitted) == 2
    by_agid = {e["agid"]: e for e in adl.emitted}
    assert set(by_agid) == {"oe_evaluator", "oe_moderator"}
    assert "qgen_crew" not in by_agid
    for ev in adl.emitted:
        # Real per-request tenant preserved (NOT normalized to platform).
        assert ev["tenant_id"] == "tenant-1"
        assert ev["gcid"] == "learner-1"
        assert ev["assist_id"] == "sub-1"
        assert ev["crew_id"] == "sub-1"
        assert ev["crew_name"] == "oe_grading"
        assert ev["chora_imda_dimension"] == "accountability"
        assert ev["question_type"] == "oe"
    # Clean grade (nothing flagged) → both accept.
    assert by_agid["oe_evaluator"]["decision"] == "accepted"
    assert by_agid["oe_moderator"]["decision"] == "accepted"
    # Tokens split per agent: evaluator = evaluate + assess_summary hops;
    # moderator = moderate hop (no cross-tile double-count).
    assert by_agid["oe_evaluator"]["prompt_tokens"] == 150  # 120 + 30
    assert by_agid["oe_evaluator"]["completion_tokens"] == 90  # 40 + 50
    assert by_agid["oe_moderator"]["prompt_tokens"] == 60
    assert by_agid["oe_moderator"]["completion_tokens"] == 20
    # ADR-197 M-A.3 — prompt-shaping conditions ride the durable record,
    # mirroring the Go EvaluatorConditions / ModeratorConditions extractors.
    ec = by_agid["oe_evaluator"]["prompt_conditions"]
    assert ec["mode"] == "evaluate"  # OE questions graded → evaluate mode
    assert ec["attempt_index"] == "0"  # 0-based (Go convention); no re-grade
    assert ec["subject"] == "science"
    assert "has_prior_moderator_feedback" not in ec  # no re-grade → no prior fb
    mc = by_agid["oe_moderator"]["prompt_conditions"]
    assert mc["attempt_index"] == "0"
    assert mc["subject"] == "science"
    # Terminal publish still fires.
    assert len(pub.calls) == 1


@pytest.mark.asyncio
async def test_handle_requested_flagged_grade_yields_warning_decisions() -> None:
    graph = _FakeGraph(terminal=_terminal_with_trace(any_flagged=True))
    pub = _FakePublisher()
    adl = _FakeAgentDecisionLogEmitter()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub, agent_decision_emitter=adl)

    await runner.handle_requested(_event())

    by_agid = {e["agid"]: e for e in adl.emitted}
    assert by_agid["oe_evaluator"]["decision"] == "completed_with_warning"
    assert by_agid["oe_moderator"]["decision"] == "completed_with_warning"
    assert by_agid["oe_evaluator"]["quality_warning"] is True
    assert by_agid["oe_moderator"]["quality_warning"] is True


@pytest.mark.asyncio
async def test_per_agent_decisions_carry_distinct_trace_spans() -> None:
    """Each OE agent's AgentDecisionLog must carry a DISTINCT span id within the
    SAME run trace so the O+ /o/agents 'View in Cloud Trace' deep-link resolves
    to oe_evaluator vs oe_moderator individually (not the shared crew trace). A
    real OTel SDK is installed so the per-agent marker spans mint real span ids;
    conftest's _reset_global_tracer_provider restores the global provider after."""
    pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace_api.set_tracer_provider(provider)

    # _event()'s inbound traceparent (the shared run trace).
    inbound_trace_id = "0af7651916cd43dd8448eb211c80319c"
    inbound_span_id = "b7ad6b7169203331"

    graph = _FakeGraph(terminal=_terminal_with_trace())
    pub = _FakePublisher()
    adl = _FakeAgentDecisionLogEmitter()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub, agent_decision_emitter=adl)

    await runner.handle_requested(_event())

    assert len(adl.emitted) == 2
    by_agid = {e["agid"]: e for e in adl.emitted}
    assert set(by_agid) == {"oe_evaluator", "oe_moderator"}

    tp_e = by_agid["oe_evaluator"]["traceparent"]
    tp_m = by_agid["oe_moderator"]["traceparent"]

    def _trace_id(tp: str) -> str:
        return tp.split("-")[1]

    def _span_id(tp: str) -> str:
        return tp.split("-")[2]

    # Both stay within the inbound run trace ...
    assert _trace_id(tp_e) == inbound_trace_id
    assert _trace_id(tp_m) == inbound_trace_id
    # ... but each carries a DISTINCT span id — from each other AND from the
    # shared inbound span.
    assert _span_id(tp_e) != _span_id(tp_m)
    assert _span_id(tp_e) != inbound_span_id
    assert _span_id(tp_m) != inbound_span_id

    # A per-agent marker span tagged chora.agent_id was emitted for each agent.
    marker_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("agent.")]
    tagged = {s.attributes.get("chora.agent_id") for s in marker_spans}
    assert {"oe_evaluator", "oe_moderator"} <= tagged

    # §9 — each marker span ALSO carries the decision EVIDENCE (the verdict), so
    # the O+ Decision-Traces → Cloud Trace deep-link surfaces WHAT each grading
    # agent decided when an auditor lands on its span (IMDA D2). Happy path →
    # both "accepted".
    by_name = {s.name: s for s in marker_spans}
    assert by_name["agent.oe_evaluator"].attributes["chora.decision"] == "accepted"
    assert by_name["agent.oe_moderator"].attributes["chora.decision"] == "accepted"


@pytest.mark.asyncio
async def test_handle_requested_no_decision_emitter_means_no_emit() -> None:
    graph = _FakeGraph(terminal=_terminal())
    pub = _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub)  # no emitter wired

    await runner.handle_requested(_event())

    # Behaviour unchanged from the pre-emission baseline.
    assert len(pub.calls) == 1


@pytest.mark.asyncio
async def test_handle_requested_decision_emit_failure_does_not_break_publish() -> None:
    @dataclass
    class _FailingEmitter:
        async def emit(self, **_: Any) -> str:
            raise RuntimeError("simulated emit failure")

    graph = _FakeGraph(terminal=_terminal_with_trace())
    pub = _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub, agent_decision_emitter=_FailingEmitter())

    await runner.handle_requested(_event())

    # Best-effort: emit failure MUST NOT suppress the terminal publish.
    assert len(pub.calls) == 1


# -----------------------------------------------------------------------------
# ADR-197 M-A.3 — pure condition-extractor helpers (mirror the Go
# EvaluatorConditions / ModeratorConditions in
# agents/oe_grading_adk_go/internal/agent/conditions.go). Direct unit tests lock
# the pinned contract + cover every branch (blanks/false omitted).
# -----------------------------------------------------------------------------


from chora_ai_kernel_orchestrator.orchestrators.oe_grading_crew_runner import (  # noqa: E402
    _oe_evaluator_conditions,
    _oe_moderator_conditions,
)


def test_oe_evaluator_conditions_evaluate_with_prior_feedback() -> None:
    # A re-grade ran (max_attempt 2 → 0-based index 1) → has_prior_moderator_
    # feedback="true"; subject present.
    assert _oe_evaluator_conditions(
        mode="evaluate",
        attempt_index=1,
        subject="Physics",
        has_prior_moderator_feedback=True,
    ) == {
        "mode": "evaluate",
        "attempt_index": "1",
        "subject": "Physics",
        "has_prior_moderator_feedback": "true",
    }


def test_oe_evaluator_conditions_assess_summary_blank_subject() -> None:
    # MCQ-only submission → assess_summary mode; blank subject omitted; no prior
    # feedback; negative index guard clamps to 0.
    assert _oe_evaluator_conditions(
        mode="assess_summary",
        attempt_index=-1,
        subject="",
        has_prior_moderator_feedback=False,
    ) == {"mode": "assess_summary", "attempt_index": "0"}


def test_oe_moderator_conditions_branches() -> None:
    assert _oe_moderator_conditions(attempt_index=0, subject="science") == {
        "attempt_index": "0",
        "subject": "science",
    }
    # Blank subject omitted.
    assert _oe_moderator_conditions(attempt_index=2, subject="  ") == {
        "attempt_index": "2",
    }


# -----------------------------------------------------------------------------
# ADR-197 M-B.2 — PromptResolver wiring into the OE grading crew.
#
# When a resolver is injected, handle_requested resolves the active override
# per agent role (oe_evaluator + oe_moderator) and:
#   1. threads prompt_overrides_json / resolved_prompt_version / prompt_source
#      into the matching grading-node executor payload, AND
#   2. stamps prompt_version / prompt_source onto the durable AgentDecisionLog
#      prompt_conditions map (rides the M-A.3 chain — no proto change).
#
# Executor-neutral: resolver=None OR embedded default → NO override keys on
# the threaded graph state (the grading LLM calls stay byte-identical).
# Durable record (CHO-2364): EVERY decision stamps prompt_version +
# prompt_source - resolver values on the override path, else
# EMBEDDED_PROMPT_VERSIONS[agid] + "embedded".
# Fail-loud: a resolver error surfaces (the run NACKs); never swallowed.
# -----------------------------------------------------------------------------


from chora_ai_kernel_orchestrator.domain.prompt_registry import (  # noqa: E402
    EMBEDDED_PROMPT_VERSIONS,
    Resolved,
)


@dataclass
class _FakeResolver:
    by_agent: dict[str, Resolved] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def resolve(self, tenant_id: str, agent_id: str) -> Resolved:
        self.calls.append((tenant_id, agent_id))
        return self.by_agent.get(agent_id, Resolved(segments={}, version=None, source="embedded"))


@pytest.mark.asyncio
async def test_resolver_stamps_prompt_overrides_on_decisions() -> None:
    resolver = _FakeResolver(
        by_agent={
            "oe_evaluator": Resolved(
                segments={"task": "TENANT EVAL TASK"},
                version="plan-tenant-7",
                source="tenant_override",
            ),
            "oe_moderator": Resolved(
                segments={"role": "PLATFORM MOD ROLE"},
                version="plan-plat-3",
                source="platform_override",
            ),
        }
    )
    graph = _FakeGraph(terminal=_terminal_with_trace())
    pub = _FakePublisher()
    adl = _FakeAgentDecisionLogEmitter()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub, agent_decision_emitter=adl, prompt_resolver=resolver)

    await runner.handle_requested(_event())

    # Resolved per agent role with the real per-request tenant.
    assert ("tenant-1", "oe_evaluator") in resolver.calls
    assert ("tenant-1", "oe_moderator") in resolver.calls

    # The resolver-stored override keys land on the state passed to the graph
    # (the grading nodes thread them into the executor payload).
    threaded_state = graph.invocations[0]["state"]
    assert threaded_state["prompt_overrides_evaluator"] == {"task": "TENANT EVAL TASK"}
    assert threaded_state["prompt_version_evaluator"] == "plan-tenant-7"
    assert threaded_state["prompt_source_evaluator"] == "tenant_override"
    assert threaded_state["prompt_overrides_moderator"] == {"role": "PLATFORM MOD ROLE"}
    assert threaded_state["prompt_version_moderator"] == "plan-plat-3"
    assert threaded_state["prompt_source_moderator"] == "platform_override"

    # Durable stamp — each agent carries its own prompt_version + prompt_source.
    by_agid = {e["agid"]: e for e in adl.emitted}
    ec = by_agid["oe_evaluator"]["prompt_conditions"]
    assert ec["prompt_version"] == "plan-tenant-7"
    assert ec["prompt_source"] == "tenant_override"
    mc = by_agid["oe_moderator"]["prompt_conditions"]
    assert mc["prompt_version"] == "plan-plat-3"
    assert mc["prompt_source"] == "platform_override"
    # M-A.3 discriminants still ride alongside the new stamp.
    assert ec["mode"] == "evaluate"
    assert mc["attempt_index"] == "0"

    assert len(pub.calls) == 1


@pytest.mark.asyncio
async def test_resolver_none_executor_neutral_but_decision_stamps_embedded() -> None:
    """No resolver → nothing threaded into the graph state (grading LLM calls
    unchanged), but every durable decision stamps the embedded provenance
    (CHO-2364): prompt_version = EMBEDDED_PROMPT_VERSIONS[agid] + prompt_source
    = "embedded"."""
    graph = _FakeGraph(terminal=_terminal_with_trace())
    pub = _FakePublisher()
    adl = _FakeAgentDecisionLogEmitter()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub, agent_decision_emitter=adl)

    await runner.handle_requested(_event())

    threaded_state = graph.invocations[0]["state"]
    assert "prompt_overrides_evaluator" not in threaded_state
    assert "prompt_overrides_moderator" not in threaded_state
    assert len(adl.emitted) == 2
    for ev in adl.emitted:
        conditions = ev["prompt_conditions"]
        assert conditions["prompt_version"] == EMBEDDED_PROMPT_VERSIONS[ev["agid"]]
        assert conditions["prompt_source"] == "embedded"


@pytest.mark.asyncio
async def test_resolver_embedded_default_executor_neutral_decision_stamped() -> None:
    """Resolver wired but no active override → nothing threaded into the graph
    state, while the durable record stamps the embedded provenance (CHO-2364)."""
    resolver = _FakeResolver(by_agent={})  # every agent → embedded
    graph = _FakeGraph(terminal=_terminal_with_trace())
    pub = _FakePublisher()
    adl = _FakeAgentDecisionLogEmitter()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub, agent_decision_emitter=adl, prompt_resolver=resolver)

    await runner.handle_requested(_event())

    # Resolver consulted ...
    assert ("tenant-1", "oe_evaluator") in resolver.calls
    assert ("tenant-1", "oe_moderator") in resolver.calls
    # ... but the embedded default threads nothing into the graph state.
    threaded_state = graph.invocations[0]["state"]
    assert "prompt_overrides_evaluator" not in threaded_state
    assert "prompt_source_moderator" not in threaded_state
    # The durable record still stamps the embedded provenance (CHO-2364).
    assert len(adl.emitted) == 2
    for ev in adl.emitted:
        conditions = ev["prompt_conditions"]
        assert conditions["prompt_version"] == EMBEDDED_PROMPT_VERSIONS[ev["agid"]]
        assert conditions["prompt_source"] == "embedded"


@pytest.mark.asyncio
async def test_resolver_error_surfaces_and_does_not_swallow() -> None:
    @dataclass
    class _BoomResolver:
        async def resolve(self, tenant_id: str, agent_id: str) -> Resolved:
            raise RuntimeError("resolver boom")

    graph = _FakeGraph(terminal=_terminal_with_trace())
    pub = _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub, prompt_resolver=_BoomResolver())

    with pytest.raises(RuntimeError, match="resolver boom"):
        await runner.handle_requested(_event())
    # Aborted before graph invoke + publish — nothing fabricated.
    assert graph.invocations == []
    assert pub.calls == []
