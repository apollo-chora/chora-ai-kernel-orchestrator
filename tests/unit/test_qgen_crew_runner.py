"""qgen_crew runner unit tests — RED→GREEN per [[feedback-strict-tdd]].

End-to-end wiring tests for QGenCrewRunner: started.v1 event → ainvoke
the qgen_crew graph → publish completed.v1 OR refused.v1 with the
correct fields forwarded verbatim.

Uses MemorySaver + fake executor (duck-typed) from
test_qgen_crew_graph.py-style and a fake _TerminalPublisher.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    GuardrailScreenInput,
    ScreenResult,
    Verdict,
)
from chora_ai_kernel_orchestrator.domain.qgen_crew import CandidatePayload
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_CRITIQUE,
    ROLE_GENERATE,
    build_qgen_crew_graph,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
    TOPIC_AI_ASSIST_COMPLETED,
    TOPIC_AI_ASSIST_REFUSED,
    AiAssistStartedPayload,
    QGenCrewRunner,
)

# -----------------------------------------------------------------------------
# Fakes (smaller variants of the test_qgen_crew_graph.py fakes)
# -----------------------------------------------------------------------------


@dataclass
class _FakeExecutor:
    responses: dict[str, list[str]]
    calls: list[dict[str, Any]] = field(default_factory=list)
    # The concrete model the agent reported for the hop (proto field 6) —
    # the real executor's map_agent_response lifts it off the agent JSON.
    model_id: str = ""

    async def execute(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        agid: str,
        agent_role: str,
        input_payload: str,
        workflow_id: str = "",
        prompt_template_id: str = "",
        context_window: list[dict[str, str]] | None = None,
        available_tools: list[str] | None = None,
    ) -> AgentExecutorResponse:
        self.calls.append({"execution_id": execution_id, "agent_role": agent_role, "input_payload": input_payload})
        queue = self.responses.get(agent_role, [])
        if not queue:
            raise RuntimeError(f"_FakeExecutor: no response for {agent_role}")
        out = queue.pop(0)
        return AgentExecutorResponse(
            execution_id=execution_id,
            output_payload=out,
            tokens_consumed_total=42,
            cost_micros_total=100,
            final_state="EXECUTION_FINAL_STATE_SUCCESS",
            model_id=self.model_id,
        )


@dataclass
class _FakeGuardrail:
    """Duck-typed ModelArmorGuardrailPort (ADR-169) — screen(payload) →
    ScreenResult; the qgen_crew nodes adapt that into GuardrailResult."""

    block_substrings: list[str] = field(default_factory=list)

    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
        for sub in self.block_substrings:
            if sub in payload.content:
                return ScreenResult(verdict=Verdict.BLOCK, reason="pii_high_risk_block")
        return ScreenResult(verdict=Verdict.ALLOW, reason="clean")


@dataclass
class _FakePublisher:
    completed: list[dict[str, Any]] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)
    progress: list[dict[str, Any]] = field(default_factory=list)
    # When True, publish_progress raises — exercises the best-effort guard
    # (a failed progress emit must NOT abort the run / suppress the terminal).
    fail_progress: bool = False

    async def publish_completed(self, **kwargs: Any) -> str:
        self.completed.append(kwargs)
        return f"msg-id-c{len(self.completed)}"

    async def publish_refused(self, **kwargs: Any) -> str:
        self.refused.append(kwargs)
        return f"msg-id-r{len(self.refused)}"

    async def publish_progress(self, **kwargs: Any) -> str:
        if self.fail_progress:
            raise RuntimeError("progress emit boom")
        self.progress.append(kwargs)
        return f"msg-id-p{len(self.progress)}"


def _good_mcq_payload() -> str:
    return json.dumps(
        {
            "stem": "What gas do plants release as a byproduct of photosynthesis?",
            "question_type": "mcq",
            "mcq_payload": {
                "options": [
                    {"option_id": "a", "label": "A", "text": "Oxygen", "is_correct": True, "explainer": "O2"},
                    {
                        "option_id": "b",
                        "label": "B",
                        "text": "Carbon dioxide",
                        "is_correct": False,
                        "explainer": "input",
                    },
                    {"option_id": "c", "label": "C", "text": "Nitrogen", "is_correct": False, "explainer": "n/a"},
                    {"option_id": "d", "label": "D", "text": "Methane", "is_correct": False, "explainer": "anaerobic"},
                ],
                "scoring_mode": "single_correct",
            },
        }
    )


def _good_oe_payload() -> str:
    return json.dumps(
        {
            "stem": "Explain chlorophyll in 2-3 sentences.",
            "question_type": "oe",
            "oe_payload": {
                "model_answer": "Chlorophyll absorbs photons in the light-dependent reactions...",
                "rubric": [
                    {"criterion_id": "a", "title": "mechanism", "description": "names photon absorption", "weight": 40},
                    {"criterion_id": "b", "title": "carriers", "description": "names ATP + NADPH", "weight": 30},
                    {"criterion_id": "c", "title": "outcome", "description": "describes Calvin cycle", "weight": 30},
                ],
                "grader_tier": "T2",
            },
        }
    )


def _critic_accept() -> str:
    return json.dumps({"accepted": True, "critique_notes": "", "suggested_revisions": []})


def _critic_reject(notes: str = "stem ambiguous") -> str:
    return json.dumps({"accepted": False, "critique_notes": notes, "suggested_revisions": ["clarify"]})


def _started_event(**overrides: Any) -> dict[str, Any]:
    """A canonical started.v1 event matching chora-creation's outbox publish
    in ai_assist_async_handler.go §aiAssistAsync `startedPayload`."""
    base: dict[str, Any] = {
        "assist_id": "job-abc",
        "tenant_id": "tenant-test",
        "author_gcid": "gcid-test",
        "content_type": "oe",
        "question_type": "oe",
        "prompt": "Explain chlorophyll in 2-3 sentences",
        "metadata": {"subject": "Biology"},
        "max_retries": 3,
        "traceparent": "00-1234-5678-01",
    }
    base.update(overrides)
    return base


def _build_runner(executor: Any, guardrail: Any, publisher: Any) -> QGenCrewRunner:
    graph = build_qgen_crew_graph(executor=executor, guardrail=guardrail, checkpointer=MemorySaver())
    return QGenCrewRunner(graph=graph, publisher=publisher)


# -----------------------------------------------------------------------------
# from_event parser tests
# -----------------------------------------------------------------------------


def test_payload_from_event_happy() -> None:
    p = AiAssistStartedPayload.from_event(_started_event())
    assert p.assist_id == "job-abc"
    assert p.tenant_id == "tenant-test"
    assert p.author_gcid == "gcid-test"
    assert p.question_type == "oe"
    assert p.max_retries == 3
    assert p.metadata == {"subject": "Biology"}
    assert p.traceparent == "00-1234-5678-01"


def test_payload_from_event_falls_back_to_content_type() -> None:
    # When question_type missing (old publishers), fall back to content_type
    # — the additive field landed 2026-05-17.
    event = _started_event(question_type="")
    p = AiAssistStartedPayload.from_event(event)
    assert p.question_type == "oe"  # from content_type


def test_payload_from_event_aliases() -> None:
    # Accept job_id alias for assist_id, gcid alias for author_gcid.
    event = {
        "job_id": "j",
        "tenant_id": "t",
        "gcid": "g",
        "question_type": "mcq",
        "prompt": "x",
        "max_retries": 0,
    }
    p = AiAssistStartedPayload.from_event(event)
    assert p.assist_id == "j"
    assert p.author_gcid == "g"


def test_payload_from_event_captures_existing_question() -> None:
    """CHO-1658 — from_event surfaces existing_question (the author's current
    question content) for the model_answer_fill path; None on every other event."""
    existing = {"stem": "Q", "oe_rubric": [{"criterion": "x", "weight": 1.0}]}
    p = AiAssistStartedPayload.from_event(_started_event(intent="model_answer_fill", existing_question=existing))
    assert p.intent == "model_answer_fill"
    assert p.existing_question == existing
    # Absent ⇒ None (the live new_question path; byte/behaviour-compatible).
    assert AiAssistStartedPayload.from_event(_started_event()).existing_question is None
    # A non-dict existing_question is ignored (defensive).
    assert AiAssistStartedPayload.from_event(_started_event(existing_question="garbage")).existing_question is None


@pytest.mark.asyncio
async def test_handle_started_model_answer_fill_threads_existing_question_to_executor() -> None:
    """CHO-1658 end-to-end — a model_answer_fill started event must thread intent +
    existing_question through from_event → QGenCrewState → generate_node into the
    qgen_question executor input, so the executor surfaces the author's stem /
    rubric fill keys (it reads input_obj['intent'] + input_obj['existing_question']).
    """
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_good_oe_payload()], ROLE_CRITIQUE: [_critic_accept()]})
    runner = _build_runner(executor, _FakeGuardrail(), _FakePublisher())
    existing = {
        "stem": "Explain chlorophyll.",
        "oe_rubric": [{"criterion": "names photon absorption", "weight": 0.4}],
        "model_answer": "placeholder",
    }
    await runner.handle_started(_started_event(intent="model_answer_fill", existing_question=existing))
    gen = next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)
    payload = json.loads(gen["input_payload"])
    assert payload["intent"] == "model_answer_fill"
    assert payload["existing_question"] == existing


# -----------------------------------------------------------------------------
# Happy path — runner → graph → publish_completed
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_started_happy_path_publishes_completed() -> None:
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    publisher = _FakePublisher()
    runner = _build_runner(executor, guardrail, publisher)

    await runner.handle_started(_started_event())

    assert len(publisher.completed) == 1
    assert len(publisher.refused) == 0
    c = publisher.completed[0]
    assert c["assist_id"] == "job-abc"
    assert c["tenant_id"] == "tenant-test"
    assert c["author_gcid"] == "gcid-test"
    assert c["quality_warning"] is False
    assert c["attempt_count"] == 1
    # candidate_payload_json must be the canonical AiAssistCandidate JSON.
    assert c["candidate_payload_json"]
    decoded = json.loads(c["candidate_payload_json"])
    assert decoded["question_type"] == "oe"
    assert decoded["stem"]
    # pipeline_trace_json is the graph's IMDA D2 evidence.
    trace = json.loads(c["pipeline_trace_json"])
    names = [r["name"] for r in trace]
    assert "generate" in names
    assert "critique" in names
    assert "publish_completed" in names
    # traceparent forwarded.
    assert c["traceparent"] == "00-1234-5678-01"


# -----------------------------------------------------------------------------
# Live-trace streaming — progress_emitter wired → astream + per-node progress
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_started_streams_progress_per_node() -> None:
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(
        graph=graph,
        publisher=publisher,
        progress_emitter=publisher,
    )

    await runner.handle_started(_started_event())

    # Exactly one terminal completed, no refusal.
    assert len(publisher.completed) == 1
    assert len(publisher.refused) == 0

    # Progress was streamed: at least the generate + critique nodes emitted.
    assert len(publisher.progress) >= 2, publisher.progress
    step_indices = [p["step_index"] for p in publisher.progress]
    # Monotonically strictly increasing (the runner emits only on growth).
    assert step_indices == sorted(step_indices)
    assert len(set(step_indices)) == len(step_indices)
    # Each progress carries assist_id/tenant + a partial cumulative trace whose
    # length equals its step_index, and NONE include the terminal publish_* row.
    for p in publisher.progress:
        assert p["assist_id"] == "job-abc"
        assert p["tenant_id"] == "tenant-test"
        assert p["traceparent"] == "00-1234-5678-01"
        partial = json.loads(p["pipeline_trace_json"])
        assert len(partial) == p["step_index"]
        names = [r["name"] for r in partial]
        assert "publish_completed" not in names
        assert "publish_refused" not in names
    # The LAST progress carries strictly fewer steps than the terminal trace
    # (the terminal publish_completed row is only on the completed.v1 event).
    final_trace = json.loads(publisher.completed[0]["pipeline_trace_json"])
    assert step_indices[-1] < len(final_trace)
    assert "publish_completed" in [r["name"] for r in final_trace]


@pytest.mark.asyncio
async def test_handle_started_no_progress_emitter_streams_nothing() -> None:
    # Default (no progress_emitter) keeps the byte-stable single-shot ainvoke
    # path — the terminal still publishes, but zero progress events fire.
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    runner = _build_runner(executor, _FakeGuardrail(), publisher)

    await runner.handle_started(_started_event())

    assert len(publisher.completed) == 1
    assert publisher.progress == []


@pytest.mark.asyncio
async def test_handle_started_progress_emit_failure_does_not_break_terminal() -> None:
    # A raising progress emit is swallowed (best-effort) — the terminal
    # completed.v1 still publishes, mirroring the token-usage emitter guard.
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher(fail_progress=True)
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(
        graph=graph,
        publisher=publisher,
        progress_emitter=publisher,
    )

    await runner.handle_started(_started_event())

    assert len(publisher.completed) == 1
    assert publisher.progress == []  # all emits raised + were swallowed


# -----------------------------------------------------------------------------
# Retry exhausted — quality_warning=True forwarded
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_started_quality_warning_propagated() -> None:
    # max_retries=1 → up to 2 attempts; both critic-rejected → quality_warning.
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload(), _good_oe_payload()],
            ROLE_CRITIQUE: [_critic_reject("attempt 1"), _critic_reject("attempt 2")],
        }
    )
    guardrail = _FakeGuardrail()
    publisher = _FakePublisher()
    runner = _build_runner(executor, guardrail, publisher)

    await runner.handle_started(_started_event(max_retries=1))

    assert len(publisher.completed) == 1
    assert publisher.completed[0]["quality_warning"] is True
    assert publisher.completed[0]["attempt_count"] == 2
    assert len(publisher.refused) == 0  # NOT refused per user-locked semantics


# -----------------------------------------------------------------------------
# Guardrail pre block → publish_refused with reason GUARDRAIL_PRE
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_started_guardrail_pre_block_publishes_refused() -> None:
    executor = _FakeExecutor(responses={ROLE_GENERATE: [], ROLE_CRITIQUE: []})
    guardrail = _FakeGuardrail(block_substrings=["SSN"])
    publisher = _FakePublisher()
    runner = _build_runner(executor, guardrail, publisher)

    await runner.handle_started(_started_event(prompt="Generate question about SSN 123-45-6789"))

    assert len(publisher.completed) == 0
    assert len(publisher.refused) == 1
    r = publisher.refused[0]
    assert r["refusal_reason"] == "GUARDRAIL_PRE"
    assert r["model_armor_verdict"] == "armor:pii_high_risk_block"
    assert r["user_facing_message"]
    # Pipeline trace must be present (even on early-refuse).
    trace = json.loads(r["pipeline_trace_json"])
    assert any(row["name"] == "publish_refused" for row in trace)


# -----------------------------------------------------------------------------
# Guardrail post block → publish_refused with reason GUARDRAIL_POST
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_started_guardrail_post_block_publishes_refused() -> None:
    # Pre allows; post blocks on the candidate. "Calvin cycle" appears in
    # the OE model_answer but NOT in the prompt — so pre-screen allows
    # but post-screen blocks.
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_good_oe_payload()], ROLE_CRITIQUE: []})
    guardrail = _FakeGuardrail(block_substrings=["Calvin cycle"])
    publisher = _FakePublisher()
    runner = _build_runner(executor, guardrail, publisher)

    await runner.handle_started(_started_event())

    assert len(publisher.refused) == 1
    assert publisher.refused[0]["refusal_reason"] == "GUARDRAIL_POST"
    # last_candidate_payload_json populated (the generator did produce one).
    assert publisher.refused[0]["last_candidate_payload_json"]


# -----------------------------------------------------------------------------
# Fail-loud validation guards
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_started_rejects_missing_assist_id() -> None:
    runner = _build_runner(_FakeExecutor(responses={}), _FakeGuardrail(), _FakePublisher())
    with pytest.raises(ValueError, match="assist_id"):
        await runner.handle_started(_started_event(assist_id=""))


@pytest.mark.asyncio
async def test_handle_started_rejects_missing_prompt() -> None:
    runner = _build_runner(_FakeExecutor(responses={}), _FakeGuardrail(), _FakePublisher())
    with pytest.raises(ValueError, match="prompt"):
        await runner.handle_started(_started_event(prompt=""))


@pytest.mark.asyncio
async def test_handle_started_rejects_unsupported_question_type() -> None:
    runner = _build_runner(_FakeExecutor(responses={}), _FakeGuardrail(), _FakePublisher())
    with pytest.raises(ValueError, match="unsupported question_type"):
        await runner.handle_started(_started_event(question_type="flashcard", content_type=""))


# -----------------------------------------------------------------------------
# thread_id customisation — D6 P1 pod-death recovery seam
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_started_continues_inbound_traceparent() -> None:
    """Regression guard for 2026-05-17 FE-coord
    E2E-BE-AI-ASSIST-TRACE-EXPORT-PERM Leg 3 (Pub/Sub trace continuation):
    the orchestrator must extract the W3C traceparent from the inbound
    started.v1 envelope and start a span whose trace_id matches the
    publisher's. Without this, agent-side spans (qgen_question /
    qgen_critic) land under a DIFFERENT trace_id than the FE-originated
    POST trace_id and the auditor can't reconstruct the request.
    """
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

    # Canonical W3C traceparent — trace_id = 0af7651916cd43dd8448eb211c80319c
    inbound_tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    runner = _build_runner(
        _FakeExecutor({ROLE_GENERATE: _good_oe_payload(), ROLE_CRITIQUE: _critic_accept()}),
        _FakeGuardrail(),
        _FakePublisher(),
    )
    await runner.handle_started(_started_event(traceparent=inbound_tp))

    spans = exporter.get_finished_spans()
    handle_started_spans = [s for s in spans if s.name == "qgen_crew.handle_started"]
    assert handle_started_spans, f"expected qgen_crew.handle_started span; got names: {[s.name for s in spans]!r}"
    span = handle_started_spans[0]
    got_trace_id = format(span.context.trace_id, "032x")
    assert got_trace_id == "0af7651916cd43dd8448eb211c80319c", (
        f"trace_id continuation broken: got {got_trace_id!r}, want "
        f"0af7651916cd43dd8448eb211c80319c (the inbound traceparent's trace_id)"
    )


@pytest.mark.asyncio
async def test_handle_started_uses_assist_id_as_default_thread_id() -> None:
    # Validates that the runner threads thread_id=assist_id into the
    # graph config (default behaviour). Tested via the executor's
    # execution_id which the graph builds from job_id.
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    publisher = _FakePublisher()
    runner = _build_runner(executor, guardrail, publisher)
    await runner.handle_started(_started_event(assist_id="thread-test-id"))

    # Both executor calls should carry the assist_id in their execution_id.
    assert all("thread-test-id" in c["execution_id"] for c in executor.calls)


# -----------------------------------------------------------------------------
# Topic constants — explicit re-export check
# -----------------------------------------------------------------------------


def test_topic_constants() -> None:
    assert TOPIC_AI_ASSIST_COMPLETED == "chora.creation.ai_assist.completed.v1"
    assert TOPIC_AI_ASSIST_REFUSED == "chora.creation.ai_assist.refused.v1"


# -----------------------------------------------------------------------------
# TokenUsageLedger emit (Gate #7) — REMOVED 2026-07-23.
#
# The orchestrator no longer produces token-usage events; chora-model-gateway
# is the sole producer per ADR-163. The retirement is pinned by
# tests/unit/test_token_usage_emitter_retired.py.
#
# The tests that lived here asserted emission and all passed while the lane
# was emitting an EMPTY model_id in production, because
# test_handle_started_engine_resource_propagates hand-filled engine_resource
# into the trace row it fed the runner. No production node ever stamped one.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# AgentDecisionLog emit (Gate #8 — per
# docs/m13/oe-ai-assist-session-close-2026-05-17.md §Step 6)
#
# The runner SHOULD emit ONE chora.observability.agent_decision.logged.v1
# event per terminal run, derived from the FINAL `quality_gate` trace row
# (when present). The event carries:
#
#   - tenant_id, gcid, assist_id
#   - decision (accepted | retry | completed_with_warning | rejected | refused)
#   - attempt_count, max_retries
#   - critic_notes (forwarded when the critic stamped them)
#   - quality_warning bool
#   - chora_imda_dimension == "accountability" (D1) per ADR-141
#   - occurred_at (ISO-8601 UTC)
#   - traceparent / tracestate forwarded
#
# Emission goes via the transactional outbox per
# [[feedback-d6-resilience-first-class]]. Wired through a SEPARATE port
# (`_AgentDecisionLogEmitter`) so existing call sites that DO NOT inject
# one keep behaving exactly as today.
# -----------------------------------------------------------------------------


from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (  # noqa: E402
    ENV_CHORA_IS_EVAL_RUN,
    IMDA_DIMENSION_ACCOUNTABILITY,
    MCQ_AI_ASSIST_CREW_NAME,
    TOPIC_OBSERVABILITY_AGENT_DECISION,
)


@dataclass
class _FakeAgentDecisionLogEmitter:
    """Records each emitted AgentDecisionLog event in memory."""

    emitted: list[dict[str, Any]] = field(default_factory=list)

    async def emit(self, **kwargs: Any) -> str:
        self.emitted.append(kwargs)
        return f"adl-{len(self.emitted)}"


def _by_agid(emitted: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index emitted AgentDecisionLog events by agid. The runner emits exactly
    one decision per agent per run (qgen_question + qgen_critic) — NEVER the
    legacy single "qgen_crew" event."""
    return {e["agid"]: e for e in emitted}


@pytest.mark.asyncio
async def test_handle_started_emits_agent_decision_log_on_accepted() -> None:
    """Happy path: critic accepts → ONE AgentDecisionLog event with
    decision=accepted + accountability dimension."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=guardrail, checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event())

    # Per-agent attribution — ONE qgen_question + ONE qgen_critic decision
    # (was a single hardcoded "qgen_crew" event before; that matched no
    # /o/agents tile so every qgen tile read 0).
    assert len(adl_emitter.emitted) == 2
    by_agid = _by_agid(adl_emitter.emitted)
    assert set(by_agid) == {"qgen_question", "qgen_critic"}
    # question_type rides every qgen event's attributes (read from payload).
    assert all(e["question_type"] == "oe" for e in adl_emitter.emitted)
    ev = by_agid["qgen_question"]
    assert ev["assist_id"] == "job-abc"
    assert ev["tenant_id"] == "tenant-test"
    assert ev["gcid"] == "gcid-test"
    assert ev["decision"] == "accepted"
    assert ev["attempt_count"] == 1
    assert ev["max_retries"] == 3
    assert ev["chora_imda_dimension"] == "accountability"
    assert ev["quality_warning"] is False
    assert ev["traceparent"] == "00-1234-5678-01"
    assert ev["occurred_at"]
    # Extension fields (atomic-napping-spring Phase A2) — feed O+ hydration.
    assert ev["crew_name"] == MCQ_AI_ASSIST_CREW_NAME
    assert ev["crew_name"] == "mcq_ai_assist"
    # crew_id == assist_id since the qgen runner is invocation-scoped.
    assert ev["crew_id"] == "job-abc"
    assert ev["is_resume"] is False
    assert ev["is_eval_run"] is False
    # Adapter version empty when base model used (M11 baseline — pre-M15
    # per-tenant LoRA wiring).
    assert ev["adapter_version"] == ""
    # Guardrail invoked on the happy path → "pass" verdict surface.
    assert ev["guardrail_outcome"] == "pass"
    # C3 (CR qgen 2026-06-01) — the executor now emits a per-hop input/output
    # token split, and generate_node/critique_node stamp it onto the trace
    # rows (no longer hardcoded 0). _FakeExecutor sets tokens_consumed_total=42
    # with no explicit split, so each LLM hop's input_tokens falls back to the
    # total (42) and output_tokens stays 0. The runner aggregates
    # prompt_tokens = Σ input_tokens (generate 42 + critique 42 = 84) and
    # completion_tokens = Σ output_tokens (0 + 0 = 0).
    # Tokens are split per agent: qgen_question carries the generate-hop count,
    # qgen_critic the critique-hop count — no double-count across the two tiles.
    assert ev["prompt_tokens"] == 42  # generate hop (input_tokens via total-fallback)
    assert ev["completion_tokens"] == 0  # _FakeExecutor emits no output_tokens split
    assert ev["cached_tokens"] == 0
    assert by_agid["qgen_critic"]["prompt_tokens"] == 42  # critique hop
    assert by_agid["qgen_critic"]["completion_tokens"] == 0
    assert by_agid["qgen_critic"]["crew_name"] == MCQ_AI_ASSIST_CREW_NAME
    # Terminal publish still fires (additive).
    assert len(publisher.completed) == 1


@pytest.mark.asyncio
async def test_agent_decision_carries_model_id_from_agent_report() -> None:
    """The concrete model the agent reported for the hop rides the
    pipeline_trace rows → the emit's model_id (proto field 6) so
    chora-observability can price the per-hop token counts per model.
    A blank model_id zeroes the cost attribution."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        },
        model_id="gemini-3.1-pro-preview",
    )
    guardrail = _FakeGuardrail()
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=guardrail, checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event())

    assert len(adl_emitter.emitted) == 2
    assert all(e["model_id"] == "gemini-3.1-pro-preview" for e in adl_emitter.emitted)


@pytest.mark.asyncio
async def test_agent_decision_model_id_defaults_empty() -> None:
    """No model reported → blank model_id (proto3-default-omit downstream;
    the consumer keeps the zero sentinel rather than a fabricated model)."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        },
    )
    guardrail = _FakeGuardrail()
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=guardrail, checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event())

    assert len(adl_emitter.emitted) == 2
    assert all(e["model_id"] == "" for e in adl_emitter.emitted)


@pytest.mark.asyncio
async def test_agent_decision_carries_prompt_conditions() -> None:
    """ADR-197 M-A.3 — each qgen AgentDecisionLog carries the prompt-shaping
    condition discriminants the orchestrator genuinely holds, mirroring the Go
    QuestionConditions / CriticConditions extractors so the durable record + the
    live agent span agree. Blanks/false are omitted per the contract."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=guardrail, checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event(metadata={"subject": "Biology", "cognitive_level": "apply"}))

    by_agid = _by_agid(adl_emitter.emitted)
    # qgen_question — intent defaults to new_question; question_type lowercased;
    # subject_hint + cognitive_level_hint from metadata; difficulty omitted
    # (blank); image flags omitted (author opted out); request_surface omitted
    # (no metadata.surface). CHO-2364: the embedded prompt provenance is ALWAYS
    # stamped (no override resolved here).
    q = by_agid["qgen_question"]["prompt_conditions"]
    assert q == {
        "intent": "new_question",
        "question_type": "oe",
        "subject_hint": "Biology",
        "cognitive_level_hint": "apply",
        "prompt_version": EMBEDDED_PROMPT_VERSIONS["qgen_question"],
        "prompt_source": "embedded",
    }
    # qgen_critic — question_type always present; attempt_index is 0-based (Go
    # convention); has_prior_notes omitted on a first-attempt accept; set_mode
    # omitted (no type_plan on a single run).
    c = by_agid["qgen_critic"]["prompt_conditions"]
    assert c["question_type"] == "oe"
    assert c["attempt_index"] == "0"
    assert "has_prior_notes" not in c
    assert "set_mode" not in c
    assert "request_surface" not in c
    assert c["prompt_version"] == EMBEDDED_PROMPT_VERSIONS["qgen_critic"]
    assert c["prompt_source"] == "embedded"


@pytest.mark.asyncio
async def test_agent_decisions_carry_request_surface_from_metadata() -> None:
    """CHO-2364 - a started.v1 whose metadata carries surface="campaign" (the
    consumption dose lane stamps it) yields decisions whose conditions carry
    request_surface="campaign" on BOTH agents."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=_FakePublisher(), agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event(metadata={"subject": "Biology", "surface": "campaign"}))

    assert len(adl_emitter.emitted) == 2
    for ev in adl_emitter.emitted:
        assert ev["prompt_conditions"]["request_surface"] == "campaign"


@pytest.mark.asyncio
async def test_agent_decision_carries_real_citation_hashes() -> None:
    """Each AgentDecisionLog carries real sha256 citation hashes (IMDA D2):
    input_hash over the author prompt + output_hash over the generated
    candidate. Closes the all-zeros sentinel the O+ reasoning panel showed —
    raw content never leaves the orchestrator (one-way hash only). Run-level:
    both agents cite the same input/output (mirrors the run-level verdict)."""
    import hashlib

    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event())

    assert len(adl_emitter.emitted) == 2
    expected_input = hashlib.sha256(b"Explain chlorophyll in 2-3 sentences").hexdigest()
    zero = "0" * 64
    for ev in adl_emitter.emitted:
        # input_hash is deterministic over the known author prompt.
        assert ev["input_hash"] == expected_input
        # output_hash is a real 64-hex sha256 over the candidate — NOT the
        # all-zeros sentinel the panel showed before.
        assert len(ev["output_hash"]) == 64
        assert ev["output_hash"] != zero
        assert ev["input_hash"] != zero


@pytest.mark.asyncio
async def test_handle_started_emits_per_agent_decisions_mcq() -> None:
    """Contract: an MCQ generation emits exactly TWO agent_decision events —
    agid="qgen_question" + agid="qgen_critic", NEVER "qgen_crew" — and each
    carries question_type="mcq" in attributes (read from payload.question_type)
    + the real per-request tenant (not normalized to platform)."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event(content_type="mcq", question_type="mcq"))

    assert len(adl_emitter.emitted) == 2
    by_agid = _by_agid(adl_emitter.emitted)
    assert set(by_agid) == {"qgen_question", "qgen_critic"}
    assert "qgen_crew" not in by_agid
    assert by_agid["qgen_question"]["question_type"] == "mcq"
    assert by_agid["qgen_critic"]["question_type"] == "mcq"
    assert by_agid["qgen_question"]["tenant_id"] == "tenant-test"
    assert by_agid["qgen_critic"]["tenant_id"] == "tenant-test"


@pytest.mark.asyncio
async def test_per_agent_decisions_carry_distinct_trace_spans() -> None:
    """Each agent's AgentDecisionLog must carry a DISTINCT span id within the
    SAME run trace, so the O+ /o/agents 'View in Cloud Trace' deep-link resolves
    to the agent's OWN span (qgen_question vs qgen_critic) rather than the shared
    crew trace. A real OTel SDK is installed so the per-agent marker spans mint
    real span ids; conftest's _reset_global_tracer_provider restores the global
    provider afterwards."""
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

    # Canonical inbound W3C traceparent (the shared run trace).
    inbound_trace_id = "0af7651916cd43dd8448eb211c80319c"
    inbound_span_id = "b7ad6b7169203331"
    inbound_tp = f"00-{inbound_trace_id}-{inbound_span_id}-01"

    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event(content_type="mcq", question_type="mcq", traceparent=inbound_tp))

    assert len(adl_emitter.emitted) == 2
    by_agid = _by_agid(adl_emitter.emitted)
    assert set(by_agid) == {"qgen_question", "qgen_critic"}

    tp_q = by_agid["qgen_question"]["traceparent"]
    tp_c = by_agid["qgen_critic"]["traceparent"]

    def _trace_id(tp: str) -> str:
        return tp.split("-")[1]

    def _span_id(tp: str) -> str:
        return tp.split("-")[2]

    # Both decisions stay within the inbound run trace ...
    assert _trace_id(tp_q) == inbound_trace_id
    assert _trace_id(tp_c) == inbound_trace_id
    # ... but each carries a DISTINCT span id — from each other AND from the
    # shared inbound span (the core fix: no more identical span ids per agent).
    assert _span_id(tp_q) != _span_id(tp_c)
    assert _span_id(tp_q) != inbound_span_id
    assert _span_id(tp_c) != inbound_span_id

    # A per-agent marker span tagged chora.agent_id was emitted for each agent.
    marker_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("agent.")]
    tagged = {s.attributes.get("chora.agent_id") for s in marker_spans}
    assert {"qgen_question", "qgen_critic"} <= tagged

    # §9 — each marker span ALSO carries the decision EVIDENCE (the verdict),
    # so the O+ Decision-Traces → Cloud Trace deep-link surfaces WHAT the agent
    # decided when an auditor lands on the span. Critic accepts here → both the
    # generator + critic decisions are "accepted".
    by_name = {s.name: s for s in marker_spans}
    assert by_name["agent.qgen_question"].attributes["chora.decision"] == "accepted"
    assert by_name["agent.qgen_critic"].attributes["chora.decision"] == "accepted"
    # question_type still rides alongside the verdict.
    assert by_name["agent.qgen_question"].attributes["chora.question_type"] == "mcq"


@pytest.mark.asyncio
async def test_handle_started_emits_agent_decision_log_on_quality_warning() -> None:
    """Retry exhausted (critic rejects each attempt) → AgentDecisionLog
    event with decision=completed_with_warning + quality_warning=True."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload(), _good_oe_payload()],
            ROLE_CRITIQUE: [_critic_reject("a1"), _critic_reject("a2")],
        }
    )
    guardrail = _FakeGuardrail()
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=guardrail, checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event(max_retries=1))

    assert len(adl_emitter.emitted) == 2
    assert {e["agid"] for e in adl_emitter.emitted} == {"qgen_question", "qgen_critic"}
    ev = _by_agid(adl_emitter.emitted)["qgen_critic"]
    assert ev["decision"] == "completed_with_warning"
    assert ev["quality_warning"] is True
    assert ev["attempt_count"] == 2
    assert ev["max_retries"] == 1
    # critic_notes propagates from the final attempt.
    assert ev["critic_notes"]
    # Extension fields — crew identity stays stamped even on retry-exhausted.
    assert ev["crew_name"] == MCQ_AI_ASSIST_CREW_NAME
    assert ev["crew_id"] == "job-abc"
    # Multi-attempt path still ran guardrails on each generate hop → pass.
    assert ev["guardrail_outcome"] == "pass"
    assert ev["is_eval_run"] is False


@pytest.mark.asyncio
async def test_handle_started_emits_agent_decision_log_on_refused() -> None:
    """Guardrail-pre block → AgentDecisionLog event with decision=refused.
    No quality_gate row exists, so decision derives from refusal_reason."""
    executor = _FakeExecutor(responses={ROLE_GENERATE: [], ROLE_CRITIQUE: []})
    guardrail = _FakeGuardrail(block_substrings=["SSN"])
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=guardrail, checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event(prompt="Generate question about SSN 123-45-6789"))

    # Both agents still emit on a guardrail-pre refusal (the run is attributed
    # to qgen_question + qgen_critic even though neither LLM hop executed).
    assert len(adl_emitter.emitted) == 2
    assert {e["agid"] for e in adl_emitter.emitted} == {"qgen_question", "qgen_critic"}
    ev = _by_agid(adl_emitter.emitted)["qgen_question"]
    assert ev["decision"] == "refused"
    assert ev["chora_imda_dimension"] == "accountability"
    # Extension fields — guardrail BLOCK on pre-screen yields "block" verdict
    # surface (the load-bearing signal for /o/governance Decision Traces).
    assert ev["guardrail_outcome"] == "block"
    assert ev["crew_name"] == MCQ_AI_ASSIST_CREW_NAME
    assert ev["crew_id"] == "job-abc"
    # Refused publish STILL fires.
    assert len(publisher.refused) == 1


@pytest.mark.asyncio
async def test_handle_started_no_decision_emitter_means_no_emit() -> None:
    """Backwards-compat: without an emitter, the runner DOES NOT crash + DOES
    NOT emit. Existing terminal publish path stays the only side effect."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    runner = _build_runner(executor, _FakeGuardrail(), publisher)

    await runner.handle_started(_started_event())

    # No decision emitter wired → behaviour unchanged from the M11 baseline.
    assert len(publisher.completed) == 1


@pytest.mark.asyncio
async def test_handle_started_decision_emit_failure_does_not_break_terminal_publish() -> None:
    """Best-effort: a transient emit error MUST NOT suppress the terminal
    publish. The runner logs + proceeds."""

    @dataclass
    class _FailingDecisionEmitter:
        async def emit(self, **_: Any) -> str:
            raise RuntimeError("simulated emit failure")

    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    runner = QGenCrewRunner(
        graph=build_qgen_crew_graph(
            executor=executor,
            guardrail=_FakeGuardrail(),
            checkpointer=MemorySaver(),
        ),
        publisher=publisher,
        agent_decision_emitter=_FailingDecisionEmitter(),
    )

    await runner.handle_started(_started_event())

    assert len(publisher.completed) == 1


def test_agent_decision_topic_constant_is_canonical() -> None:
    # Per chora-contracts/proto/events/observability/agent_decision.proto + IMDA D1 (ADR-141).
    # Topic chora.ai_kernel.agent_decided.v1 was the planned name in
    # docs/m13/oe-ai-assist-session-close-2026-05-17.md but the canonical
    # provisioned topic is chora.observability.agent_decision.logged.v1
    # (see chora-infra/terraform/modules/m10-data-plane/main.tf:472 +
    # bigquery sink). We use the canonical topic per
    # [[feedback-arch-ground-in-deployed-reality]].
    assert TOPIC_OBSERVABILITY_AGENT_DECISION == "chora.observability.agent_decision.logged.v1"
    assert IMDA_DIMENSION_ACCOUNTABILITY == "accountability"


@pytest.mark.asyncio
async def test_handle_started_decision_from_quality_gate_row() -> None:
    """When the FINAL quality_gate trace row stamps a decision, the emit
    uses that row's status + notes. Exercised via a passthrough graph that
    returns a hand-rolled terminal state."""

    @dataclass
    class _PassthroughGraph:
        async def aget_state(self, config: dict[str, Any]) -> Any:
            raise ValueError("no checkpointer")  # a stateless graph is always fresh

        terminal: dict[str, Any]

        async def ainvoke(
            self,
            _state: dict[str, Any],
            config: dict[str, Any] | None = None,
            **_: Any,
        ) -> dict[str, Any]:
            return self.terminal

    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    runner = QGenCrewRunner(
        graph=_PassthroughGraph(
            terminal={
                "pipeline_trace": [
                    {"name": "validate_input", "status": "ACCEPTED"},
                    {"name": "generate", "status": "COMPLETED", "attempt": 1},
                    {"name": "critique", "status": "ACCEPTED", "attempt": 1},
                    {
                        "name": "quality_gate",
                        "status": "ACCEPTED",
                        "attempt": 1,
                        "notes": "critic accepted",
                    },
                    {"name": "publish_completed", "status": "COMPLETED"},
                ],
                "completed_candidate": CandidatePayload(
                    stem="x",
                    question_type="oe",
                    payload_json=_good_oe_payload(),
                    critic_notes="some notes",
                ),
                "attempt_count": 1,
                "quality_warning": False,
            }
        ),
        publisher=publisher,
        agent_decision_emitter=adl_emitter,
    )

    await runner.handle_started(_started_event())

    assert len(adl_emitter.emitted) == 2
    assert {e["agid"] for e in adl_emitter.emitted} == {"qgen_question", "qgen_critic"}
    ev = _by_agid(adl_emitter.emitted)["qgen_question"]
    assert ev["decision"] == "accepted"
    assert ev["critic_notes"] == "some notes"
    assert ev["quality_warning"] is False
    # The quality_gate row's attempt index propagates as attempt_count.
    assert ev["attempt_count"] == 1


@pytest.mark.asyncio
async def test_handle_started_decision_for_retry_exhausted_quality_gate() -> None:
    """When the FINAL quality_gate row's status is QUALITY_WARNING (retries
    exhausted), the emit's decision is `completed_with_warning`."""

    @dataclass
    class _PassthroughGraph:
        async def aget_state(self, config: dict[str, Any]) -> Any:
            raise ValueError("no checkpointer")  # a stateless graph is always fresh

        terminal: dict[str, Any]

        async def ainvoke(
            self,
            _state: dict[str, Any],
            config: dict[str, Any] | None = None,
            **_: Any,
        ) -> dict[str, Any]:
            return self.terminal

    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    runner = QGenCrewRunner(
        graph=_PassthroughGraph(
            terminal={
                "pipeline_trace": [
                    {"name": "generate", "status": "COMPLETED", "attempt": 1},
                    {"name": "critique", "status": "REJECTED", "attempt": 1},
                    {
                        "name": "quality_gate",
                        "status": "RETRY",
                        "attempt": 1,
                        "notes": "critic rejected on attempt 1",
                    },
                    {"name": "generate", "status": "COMPLETED", "attempt": 2},
                    {"name": "critique", "status": "REJECTED", "attempt": 2},
                    {
                        "name": "quality_gate",
                        "status": "QUALITY_WARNING",
                        "attempt": 2,
                        "notes": "exhausted max_retries",
                    },
                    {"name": "publish_completed", "status": "COMPLETED"},
                ],
                "completed_candidate": CandidatePayload(
                    stem="x",
                    question_type="oe",
                    payload_json=_good_oe_payload(),
                    critic_notes="final attempt notes",
                ),
                "attempt_count": 2,
                "quality_warning": True,
            }
        ),
        publisher=publisher,
        agent_decision_emitter=adl_emitter,
    )

    await runner.handle_started(_started_event(max_retries=1))

    assert len(adl_emitter.emitted) == 2
    assert {e["agid"] for e in adl_emitter.emitted} == {"qgen_question", "qgen_critic"}
    ev = _by_agid(adl_emitter.emitted)["qgen_critic"]
    assert ev["decision"] == "completed_with_warning"
    # Uses FINAL quality_gate row, not the first one.
    assert ev["attempt_count"] == 2
    assert ev["quality_warning"] is True


# -----------------------------------------------------------------------------
# Extension fields — atomic-napping-spring Phase A2 (2026-05-26)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_started_emits_is_eval_run_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CHORA_IS_EVAL_RUN=true → is_eval_run=True on the emitted event.
    Eval-tagged rows must segregate from production posture (per
    .claude/skills/mlops-agent-eval/SKILL.md)."""
    monkeypatch.setenv(ENV_CHORA_IS_EVAL_RUN, "true")
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=guardrail, checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event())

    assert len(adl_emitter.emitted) == 2
    assert all(e["is_eval_run"] is True for e in adl_emitter.emitted)


@pytest.mark.asyncio
async def test_handle_started_emits_is_eval_run_false_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CHORA_IS_EVAL_RUN unset → is_eval_run=False (production default)."""
    monkeypatch.delenv(ENV_CHORA_IS_EVAL_RUN, raising=False)
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(
        executor=_FakeExecutor(
            responses={
                ROLE_GENERATE: [_good_oe_payload()],
                ROLE_CRITIQUE: [_critic_accept()],
            }
        ),
        guardrail=_FakeGuardrail(),
        checkpointer=MemorySaver(),
    )
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)
    await runner.handle_started(_started_event())
    assert len(adl_emitter.emitted) == 2
    assert all(e["is_eval_run"] is False for e in adl_emitter.emitted)


@pytest.mark.asyncio
async def test_handle_started_aggregates_tokens_across_llm_hops() -> None:
    """prompt_tokens + completion_tokens + cached_tokens sum across all
    LLM-hop trace rows. Per-hop token counts are zero on the M11 baseline
    (executor returns only tokens_consumed_total) — we synthesise a
    terminal state with per-hop counts to exercise the aggregation."""

    @dataclass
    class _PassthroughGraph:
        async def aget_state(self, config: dict[str, Any]) -> Any:
            raise ValueError("no checkpointer")  # a stateless graph is always fresh

        terminal: dict[str, Any]

        async def ainvoke(
            self,
            _state: dict[str, Any],
            config: dict[str, Any] | None = None,
            **_: Any,
        ) -> dict[str, Any]:
            return self.terminal

    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    runner = QGenCrewRunner(
        graph=_PassthroughGraph(
            terminal={
                "pipeline_trace": [
                    {
                        "name": "generate",
                        "status": "COMPLETED",
                        "attempt": 1,
                        "input_tokens": 120,
                        "output_tokens": 75,
                        "cached_tokens": 10,
                        "adapter_version": "lora-tenant-acme-v3",
                    },
                    {
                        "name": "critique",
                        "status": "ACCEPTED",
                        "attempt": 1,
                        "input_tokens": 60,
                        "output_tokens": 25,
                        "cached_tokens": 5,
                    },
                    {
                        "name": "quality_gate",
                        "status": "ACCEPTED",
                        "attempt": 1,
                        "notes": "critic accepted",
                    },
                    {"name": "publish_completed", "status": "COMPLETED"},
                ],
                "completed_candidate": CandidatePayload(
                    stem="x",
                    question_type="oe",
                    payload_json=_good_oe_payload(),
                ),
                "attempt_count": 1,
            }
        ),
        publisher=publisher,
        agent_decision_emitter=adl_emitter,
    )

    await runner.handle_started(_started_event())

    assert len(adl_emitter.emitted) == 2
    by_agid = _by_agid(adl_emitter.emitted)
    # Tokens are split per agent (no cross-tile double-count): qgen_question
    # carries the generate-hop counts, qgen_critic the critique-hop counts.
    q = by_agid["qgen_question"]
    assert q["prompt_tokens"] == 120
    assert q["completion_tokens"] == 75
    assert q["cached_tokens"] == 10
    c = by_agid["qgen_critic"]
    assert c["prompt_tokens"] == 60
    assert c["completion_tokens"] == 25
    assert c["cached_tokens"] == 5
    # Adapter version surfaced from the first LLM-hop row that stamped it
    # (M15 per-tenant LoRA seam — currently empty on the production path).
    assert q["adapter_version"] == "lora-tenant-acme-v3"
    # Guardrail rows absent in this synthetic trace → outcome empty (both agents).
    assert q["guardrail_outcome"] == ""
    assert c["guardrail_outcome"] == ""


@pytest.mark.asyncio
async def test_handle_started_guardrail_block_yields_block_outcome() -> None:
    """guardrail_outcome=block when refusal_reason ∈ {GUARDRAIL_PRE,
    GUARDRAIL_POST}. This is the canonical signal /o/governance Decision
    Traces uses to surface refused decisions distinctly from accepted
    ones."""
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_good_oe_payload()], ROLE_CRITIQUE: []})
    guardrail = _FakeGuardrail(block_substrings=["Calvin cycle"])
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=guardrail, checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event())

    assert len(adl_emitter.emitted) == 2
    assert {e["agid"] for e in adl_emitter.emitted} == {"qgen_question", "qgen_critic"}
    ev = _by_agid(adl_emitter.emitted)["qgen_question"]
    assert ev["guardrail_outcome"] == "block"
    assert ev["decision"] == "refused"


# -----------------------------------------------------------------------------
# W8 — author per-image opt-in propagates from started.v1 → generate payload
# -----------------------------------------------------------------------------


def test_started_payload_reads_image_opt_in() -> None:
    """from_event reads image_for_stem / image_for_answer (defensive: absent
    ⇒ False, present ⇒ coerced bool)."""
    p = AiAssistStartedPayload.from_event(_started_event(image_for_stem=True, image_for_answer=False))
    assert p.image_for_stem is True
    assert p.image_for_answer is False
    # Absent on the legacy publisher ⇒ both default False.
    p2 = AiAssistStartedPayload.from_event(_started_event())
    assert p2.image_for_stem is False
    assert p2.image_for_answer is False


@pytest.mark.asyncio
async def test_runner_threads_image_opt_in_to_generate_payload() -> None:
    """End-to-end: a started.v1 carrying image_for_stem=True reaches the
    generate executor payload (author opt-in → generation agent)."""
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_good_oe_payload()], ROLE_CRITIQUE: [_critic_accept()]})
    publisher = _FakePublisher()
    runner = _build_runner(executor, _FakeGuardrail(), publisher)

    await runner.handle_started(_started_event(image_for_stem=True, image_for_answer=True))

    gen = next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)
    payload = json.loads(gen["input_payload"])
    assert payload["image_for_stem"] is True
    assert payload["image_for_answer"] is True


# -----------------------------------------------------------------------------
# HITL gate escalation (Human-Oversight queue feed)
#
# Root cause closed: the O+ Human-Oversight queue was always empty because
# nothing escalated to a HITL gate. The qgen crew's MAX-RETRIES-EXHAUSTED
# terminal (decision=completed_with_warning, quality_warning=True — the
# scout-observed in-flight outcome) is a genuinely low-quality candidate
# that warrants human review. When a `_HITLDecisionEmitter` is injected the
# runner emits ONE HITL gate event carrying the D4 routing fields the
# governance projector routeD4 → AppendHITLDecision + gateway mapHITLItem
# expect (decision_id, run_id, agent_id, summary/reason, autonomy_level,
# created_at) with chora_imda_dimension=fairness_and_human_oversight +
# event_type containing "hitl".
#
# Escalation is env-gated (QGEN_HITL_ESCALATE_ON_QUALITY_WARNING, default
# true) per [[feedback-no-inline-config]]. Emission is via the SEPARATE
# `_HITLDecisionEmitter` port (outbox) so older paths that DON'T inject one
# keep behaving exactly as today.
# -----------------------------------------------------------------------------


from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (  # noqa: E402
    ENV_HITL_ESCALATE_ON_QUALITY_WARNING,
    IMDA_DIMENSION_FAIRNESS_HUMAN_OVERSIGHT,
    TOPIC_GOVERNANCE_HITL_REQUESTED,
)


@dataclass
class _FakeHITLDecisionEmitter:
    """Records each emitted HITL gate event in memory."""

    emitted: list[dict[str, Any]] = field(default_factory=list)

    async def emit(self, **kwargs: Any) -> str:
        self.emitted.append(kwargs)
        return f"hitl-{len(self.emitted)}"


@pytest.mark.asyncio
async def test_handle_started_emits_hitl_gate_on_quality_warning() -> None:
    """Retry exhausted (critic rejects each attempt) → ONE HITL gate event
    with the D4 routing fields so a pending gate lands in the queue."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload(), _good_oe_payload()],
            ROLE_CRITIQUE: [_critic_reject("a1 ambiguous"), _critic_reject("a2 ambiguous")],
        }
    )
    publisher = _FakePublisher()
    hitl_emitter = _FakeHITLDecisionEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, hitl_decision_emitter=hitl_emitter)

    await runner.handle_started(_started_event(max_retries=1))

    # Terminal publish still fires (additive — HITL emit does not suppress it).
    assert len(publisher.completed) == 1
    assert publisher.completed[0]["quality_warning"] is True

    assert len(hitl_emitter.emitted) == 1
    ev = hitl_emitter.emitted[0]
    # decision_id == assist_id (the qgen runner is invocation-scoped); run_id
    # is the correlation id the gateway maps to workflow_id.
    assert ev["decision_id"] == "job-abc"
    assert ev["run_id"] == "job-abc"
    assert ev["tenant_id"] == "tenant-test"
    assert ev["gcid"] == "gcid-test"
    assert ev["agent_id"] == "qgen_crew"
    assert ev["crew_name"] == "mcq_ai_assist"
    # autonomy_level is a valid Level 0-2 tag (ADR-141; Level 3 prohibited).
    assert ev["autonomy_level"] in {"hitl_l0", "hitl_l1", "hitl_l2"}
    # summary/reason surfaces WHY the gate fired (mapHITLItem renders it).
    assert "max_retries" in ev["summary"] or "quality" in ev["summary"].lower()
    assert ev["occurred_at"]
    assert ev["traceparent"] == "00-1234-5678-01"


@pytest.mark.asyncio
async def test_handle_started_does_not_emit_hitl_gate_on_clean_accept() -> None:
    """Happy path (critic accepts) is NOT low-quality → no HITL gate. The
    Human-Oversight queue must not fill with healthy runs."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    hitl_emitter = _FakeHITLDecisionEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, hitl_decision_emitter=hitl_emitter)

    await runner.handle_started(_started_event())

    assert len(publisher.completed) == 1
    assert len(hitl_emitter.emitted) == 0


@pytest.mark.asyncio
async def test_handle_started_no_hitl_emitter_means_no_emit() -> None:
    """Backwards-compat: without a HITL emitter the runner DOES NOT crash +
    DOES NOT emit even on quality_warning."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload(), _good_oe_payload()],
            ROLE_CRITIQUE: [_critic_reject("a1"), _critic_reject("a2")],
        }
    )
    publisher = _FakePublisher()
    runner = _build_runner(executor, _FakeGuardrail(), publisher)

    await runner.handle_started(_started_event(max_retries=1))

    assert len(publisher.completed) == 1  # unchanged M11 baseline


@pytest.mark.asyncio
async def test_handle_started_hitl_emit_failure_does_not_break_terminal_publish() -> None:
    """The HITL gate emit is best-effort. A transient emit error MUST NOT
    suppress the terminal publish; the runner logs and proceeds."""

    @dataclass
    class _FailingHITLEmitter:
        async def emit(self, **_: Any) -> str:
            raise RuntimeError("simulated hitl emit failure")

    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload(), _good_oe_payload()],
            ROLE_CRITIQUE: [_critic_reject("a1"), _critic_reject("a2")],
        }
    )
    publisher = _FakePublisher()
    runner = QGenCrewRunner(
        graph=build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver()),
        publisher=publisher,
        hitl_decision_emitter=_FailingHITLEmitter(),
    )

    await runner.handle_started(_started_event(max_retries=1))

    assert len(publisher.completed) == 1


@pytest.mark.asyncio
async def test_handle_started_hitl_escalation_disabled_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env knob QGEN_HITL_ESCALATE_ON_QUALITY_WARNING=false disables the
    escalation per [[feedback-no-inline-config]] — operators can turn the
    gate off without a redeploy."""
    monkeypatch.setenv(ENV_HITL_ESCALATE_ON_QUALITY_WARNING, "false")
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload(), _good_oe_payload()],
            ROLE_CRITIQUE: [_critic_reject("a1"), _critic_reject("a2")],
        }
    )
    publisher = _FakePublisher()
    hitl_emitter = _FakeHITLDecisionEmitter()
    runner = QGenCrewRunner(
        graph=build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver()),
        publisher=publisher,
        hitl_decision_emitter=hitl_emitter,
    )

    await runner.handle_started(_started_event(max_retries=1))

    assert len(publisher.completed) == 1
    assert len(hitl_emitter.emitted) == 0  # escalation disabled


def test_hitl_gate_topic_and_dimension_constants() -> None:
    assert TOPIC_GOVERNANCE_HITL_REQUESTED == "chora.governance.hitl.requested.v1"
    assert IMDA_DIMENSION_FAIRNESS_HUMAN_OVERSIGHT == "fairness_and_human_oversight"


# -----------------------------------------------------------------------------
# ADR-197 M-A.3 — pure condition-extractor helpers (mirror the Go *Conditions
# extractors in agents/qgen_adk_go/internal/agent/conditions.go). Direct unit tests
# lock the pinned contract + cover every branch (blanks/false omitted).
# -----------------------------------------------------------------------------


from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (  # noqa: E402
    _qgen_critic_conditions,
    _qgen_question_conditions,
)


def test_qgen_question_conditions_full_metadata_and_images() -> None:
    payload = AiAssistStartedPayload.from_event(
        _started_event(
            question_type="mcq",
            content_type="mcq",
            intent="model_answer_fill",
            metadata={
                "subject": "Algebra",
                "cognitive_level": "evaluate",
                "difficulty": "advanced",
            },
            image_for_stem=True,
            image_for_answer=True,
        )
    )
    assert _qgen_question_conditions(payload) == {
        "intent": "model_answer_fill",
        "question_type": "mcq",
        "subject_hint": "Algebra",
        "cognitive_level_hint": "evaluate",
        "difficulty_hint": "advanced",
        "image_for_stem": "true",
        "image_for_answer": "true",
    }


def test_qgen_question_conditions_defaults_and_blanks_omitted() -> None:
    # No intent → defaults to new_question; empty metadata → all hints omitted;
    # images default False → omitted.
    payload = AiAssistStartedPayload.from_event(_started_event(question_type="OE", content_type="OE", metadata={}))
    assert _qgen_question_conditions(payload) == {
        "intent": "new_question",
        "question_type": "oe",  # lowercased
    }


def test_qgen_question_conditions_set_mode_omits_question_type() -> None:
    # A non-empty type_plan ⇒ set_mode="true" + question_type omitted (mirrors Go).
    payload = AiAssistStartedPayload.from_event(
        _started_event(
            content_type="mixed",
            question_type="mixed",
            type_plan=[{"question_type": "mcq", "count": 2}],
            metadata={},
        )
    )
    c = _qgen_question_conditions(payload)
    assert c["set_mode"] == "true"
    assert "question_type" not in c


def test_qgen_critic_conditions_branches() -> None:
    # First attempt (attempt_count=1 → 0-based index 0), no prior notes.
    assert _qgen_critic_conditions(question_type="mcq", attempt_index=0, has_prior_notes=False) == {
        "question_type": "mcq",
        "attempt_index": "0",
    }
    # Re-generation (attempt_count=3 → index 2) with prior critic notes.
    assert _qgen_critic_conditions(question_type="oe", attempt_index=2, has_prior_notes=True) == {
        "question_type": "oe",
        "attempt_index": "2",
        "has_prior_notes": "true",
    }
    # Negative guard → clamps to 0.
    assert _qgen_critic_conditions(question_type="mcq", attempt_index=-1, has_prior_notes=False)["attempt_index"] == "0"


def test_qgen_question_conditions_request_surface() -> None:
    """CHO-2364 - metadata["surface"] (the consumption dose lane stamps
    "campaign") rides the question conditions as request_surface; blank or
    absent is OMITTED (authoring sends no surface - never fabricated)."""
    with_surface = AiAssistStartedPayload.from_event(
        _started_event(metadata={"subject": "Biology", "surface": "campaign"})
    )
    assert _qgen_question_conditions(with_surface)["request_surface"] == "campaign"
    # Absent key → omitted.
    without = AiAssistStartedPayload.from_event(_started_event(metadata={}))
    assert "request_surface" not in _qgen_question_conditions(without)
    # Blank value → omitted (no fabricated discriminants).
    blank = AiAssistStartedPayload.from_event(_started_event(metadata={"surface": "  "}))
    assert "request_surface" not in _qgen_question_conditions(blank)


def test_qgen_critic_conditions_request_surface_and_set_mode() -> None:
    """CHO-2364 - the critic conditions gain the same request_surface plus
    set_mode="true" when the job ran with a type_plan; both omitted otherwise
    (defaults keep every existing call site byte-identical)."""
    c = _qgen_critic_conditions(
        question_type="mixed",
        attempt_index=0,
        has_prior_notes=False,
        request_surface="campaign",
        set_mode=True,
    )
    assert c["set_mode"] == "true"
    assert c["request_surface"] == "campaign"
    assert c["question_type"] == "mixed"
    # Blank surface + set_mode=False → both omitted.
    c2 = _qgen_critic_conditions(
        question_type="mcq",
        attempt_index=0,
        has_prior_notes=False,
        request_surface="",
        set_mode=False,
    )
    assert "set_mode" not in c2
    assert "request_surface" not in c2


# -----------------------------------------------------------------------------
# ADR-197 M-B.2 — PromptResolver wiring into the qgen crew.
#
# When a resolver is injected (production wiring binds it to the shared
# psycopg AsyncConnection), handle_started resolves the active override per
# agent role (qgen_question + qgen_critic) and:
#   1. threads prompt_overrides_json / resolved_prompt_version / prompt_source
#      into the matching executor input dict (the Go composer's
#      overrideOr(segment_id, embedded) merge seam), AND
#   2. stamps prompt_version / prompt_source onto the durable AgentDecisionLog
#      prompt_conditions map (rides the M-A.3 chain — no proto change).
#
# Executor-neutral guarantees (the LLM request path):
#   * resolver=None  → NO override keys on the executor payload (byte-identical
#     M11 baseline for the agent call).
#   * resolver returns the embedded default (segments={}) → same.
# Durable record (CHO-2364): EVERY decision stamps prompt_version +
# prompt_source. Override path keeps the resolver values; the embedded path
# stamps EMBEDDED_PROMPT_VERSIONS[agid] + source "embedded" so provenance is
# never blank.
# Fail-loud: a resolver error surfaces (the run NACKs); it is never swallowed.
# -----------------------------------------------------------------------------


from chora_ai_kernel_orchestrator.domain.prompt_registry import (  # noqa: E402
    EMBEDDED_PROMPT_VERSIONS,
    Resolved,
)


@dataclass
class _FakeResolver:
    """In-memory ``PromptResolver`` returning a fixed ``Resolved`` per agent_id.

    Unknown agent_ids fall back to the embedded default (segments={}).
    """

    by_agent: dict[str, Resolved] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def resolve(self, tenant_id: str, agent_id: str) -> Resolved:
        self.calls.append((tenant_id, agent_id))
        return self.by_agent.get(agent_id, Resolved(segments={}, version=None, source="embedded"))


@pytest.mark.asyncio
async def test_resolver_threads_overrides_into_executor_and_stamps_decision() -> None:
    resolver = _FakeResolver(
        by_agent={
            "qgen_question": Resolved(
                segments={"role": "TENANT ROLE OVERRIDE"},
                version="plan-tenant-1",
                source="tenant_override",
            ),
            "qgen_critic": Resolved(
                segments={"task": "PLATFORM CRITIC TASK"},
                version="plan-plat-9",
                source="platform_override",
            ),
        }
    )
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(
        graph=graph,
        publisher=publisher,
        agent_decision_emitter=adl_emitter,
        prompt_resolver=resolver,
    )

    await runner.handle_started(_started_event())

    # Resolved per agent role with the real per-request tenant.
    assert ("tenant-test", "qgen_question") in resolver.calls
    assert ("tenant-test", "qgen_critic") in resolver.calls

    # generate executor call carries the qgen_question override (PINNED contract).
    gen = next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)
    gen_input = json.loads(gen["input_payload"])
    assert json.loads(gen_input["prompt_overrides_json"]) == {"role": "TENANT ROLE OVERRIDE"}
    assert gen_input["resolved_prompt_version"] == "plan-tenant-1"
    assert gen_input["prompt_source"] == "tenant_override"

    # critique executor call carries the qgen_critic override.
    crit = next(c for c in executor.calls if c["agent_role"] == ROLE_CRITIQUE)
    crit_input = json.loads(crit["input_payload"])
    assert json.loads(crit_input["prompt_overrides_json"]) == {"task": "PLATFORM CRITIC TASK"}
    assert crit_input["resolved_prompt_version"] == "plan-plat-9"
    assert crit_input["prompt_source"] == "platform_override"

    # Durable stamp — each agent's prompt_conditions carries its own
    # prompt_version + prompt_source (rides the M-A.3 field-21 chain).
    by_agid = _by_agid(adl_emitter.emitted)
    q_cond = by_agid["qgen_question"]["prompt_conditions"]
    assert q_cond["prompt_version"] == "plan-tenant-1"
    assert q_cond["prompt_source"] == "tenant_override"
    c_cond = by_agid["qgen_critic"]["prompt_conditions"]
    assert c_cond["prompt_version"] == "plan-plat-9"
    assert c_cond["prompt_source"] == "platform_override"
    # The M-A.3 discriminants still ride alongside the new override stamp.
    assert q_cond["intent"] == "new_question"
    assert c_cond["question_type"] == "oe"

    # Terminal publish unaffected (additive).
    assert len(publisher.completed) == 1


@pytest.mark.asyncio
async def test_resolver_none_executor_neutral_but_decision_stamps_embedded() -> None:
    """No resolver injected → NO override keys on the executor payload (the
    LLM request path stays byte-identical M11), but the durable record now
    ALWAYS stamps provenance (CHO-2364): each agent's prompt_conditions carries
    prompt_version = EMBEDDED_PROMPT_VERSIONS[agid] + prompt_source =
    "embedded" so no decision row is ever version-blank."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl_emitter)

    await runner.handle_started(_started_event())

    gen_input = json.loads(next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)["input_payload"])
    assert "prompt_overrides_json" not in gen_input
    assert "resolved_prompt_version" not in gen_input
    assert "prompt_source" not in gen_input
    assert len(adl_emitter.emitted) == 2
    for ev in adl_emitter.emitted:
        conditions = ev["prompt_conditions"]
        assert conditions["prompt_version"] == EMBEDDED_PROMPT_VERSIONS[ev["agid"]]
        assert conditions["prompt_source"] == "embedded"


@pytest.mark.asyncio
async def test_resolver_embedded_default_executor_neutral_decision_stamped() -> None:
    """Resolver wired but returns the embedded default (no active override) →
    NO override keys on the executor payload, while the durable record stamps
    the embedded provenance (CHO-2364). Closes the 'wired but no overrides'
    path."""
    resolver = _FakeResolver(by_agent={})  # every agent → embedded
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    adl_emitter = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(
        graph=graph,
        publisher=publisher,
        agent_decision_emitter=adl_emitter,
        prompt_resolver=resolver,
    )

    await runner.handle_started(_started_event())

    # Resolver WAS consulted (both agents) ...
    assert ("tenant-test", "qgen_question") in resolver.calls
    assert ("tenant-test", "qgen_critic") in resolver.calls
    # ... but the embedded default threads nothing into the executor call.
    gen_input = json.loads(next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)["input_payload"])
    assert "prompt_overrides_json" not in gen_input
    assert "prompt_source" not in gen_input
    # The durable record still stamps the embedded provenance (CHO-2364).
    assert len(adl_emitter.emitted) == 2
    for ev in adl_emitter.emitted:
        conditions = ev["prompt_conditions"]
        assert conditions["prompt_version"] == EMBEDDED_PROMPT_VERSIONS[ev["agid"]]
        assert conditions["prompt_source"] == "embedded"


@pytest.mark.asyncio
async def test_resolver_error_surfaces_and_does_not_swallow() -> None:
    """arch-clean / fail-loud: a resolver error propagates so the Pub/Sub
    handler NACKs — it is NEVER swallowed into a silent embedded fallback."""

    @dataclass
    class _BoomResolver:
        async def resolve(self, tenant_id: str, agent_id: str) -> Resolved:
            raise RuntimeError("resolver boom")

    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_oe_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    publisher = _FakePublisher()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher, prompt_resolver=_BoomResolver())

    with pytest.raises(RuntimeError, match="resolver boom"):
        await runner.handle_started(_started_event())
    # The run aborted before the terminal publish — nothing fabricated.
    assert len(publisher.completed) == 0
