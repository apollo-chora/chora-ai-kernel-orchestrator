"""oe_grading_crew StateGraph unit tests — RED→GREEN per [[feedback-strict-tdd]].

Covers the per-submission evaluator→moderator quality loop (ADR-172 §D2/§D5)
against the REAL graph API: build_oe_grading_crew_graph(executor=, guardrail=,
checkpointer=). The composite is computed in Python (scoring.compute_composite)
from the LLM's per-criterion sub-scores + the rubric weights — the evaluator
response carries NO points_earned.

Loop bound (committed quality_gate): DEFAULT_MAX_ITERATIONS=2 ⇒ up to 3 total
grades (1 initial + 2 re-grades) before shipping quality_flagged.

Fakes only — NO gRPC, NO Vertex AI, NO Pub/Sub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    ROLE_OE_EVALUATE,
    ROLE_OE_MODERATE,
)
from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    GuardrailScreenInput,
    ScreenResult,
    Verdict,
)
from chora_ai_kernel_orchestrator.domain.oe_grading_crew import (
    OEGradingState,
    OEQuestionInput,
)
from chora_ai_kernel_orchestrator.orchestrators.oe_grading_crew import (
    build_oe_grading_crew_graph,
    guardrail_post_node,
    guardrail_pre_node,
)

# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


@dataclass
class _FakeExecutor:
    """Duck-typed _ExecutorLike. Routes execute() to a queued response
    by (agent_role, payload.mode): evaluate / summary / moderate."""

    responses: dict[str, list[str]]  # "evaluate"|"summary"|"moderate" -> payloads
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def execute(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        agid: str,
        agent_role: str,
        input_payload: str,
        **_: Any,
    ) -> AgentExecutorResponse:
        payload = json.loads(input_payload)
        mode = payload.get("mode", "")
        self.calls.append({"agent_role": agent_role, "mode": mode, "input_payload": input_payload})
        if agent_role == ROLE_OE_EVALUATE and mode == "assess_summary":
            key = "summary"
        elif agent_role == ROLE_OE_EVALUATE:
            key = "evaluate"
        else:
            key = "moderate"
        queue = self.responses.get(key, [])
        if not queue:
            raise RuntimeError(f"_FakeExecutor: no queued response for {key} (mode={mode})")
        return AgentExecutorResponse(
            execution_id=execution_id,
            output_payload=queue.pop(0),
            input_tokens=11,
            output_tokens=7,
        )


@dataclass
class _FakeGuardrail:
    """Duck-typed ModelArmorGuardrailPort. Allows by default; substrings BLOCK."""

    block_substrings: list[str] = field(default_factory=list)
    calls: list[GuardrailScreenInput] = field(default_factory=list)

    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
        self.calls.append(payload)
        for sub in self.block_substrings:
            if sub and sub in payload.content:
                return ScreenResult(verdict=Verdict.BLOCK, reason="pii_high_risk_block")
        return ScreenResult(verdict=Verdict.ALLOW, reason="")


# -----------------------------------------------------------------------------
# Builders — canonical rubric shape (bare array of {criterion_id, weight})
# -----------------------------------------------------------------------------


def _eval_payload(score: float, comment: str = "Good answer.") -> str:
    """Evaluator output — criterion_scores ONLY (no points_earned; Python derives)."""
    return json.dumps(
        {
            "criterion_scores": [
                {
                    "criterion_id": "c1",
                    "title": "Accuracy",
                    "score": score,
                    "max_score": 5.0,
                    "feedback": "evidence-grounded",
                },
            ],
            "comment": comment,
        }
    )


def _mod_payload(accepted: bool, feedback: str = "") -> str:
    return json.dumps({"accepted": accepted, "feedback": feedback})


def _summary_payload(text: str = "Overall 80%; revisit photosynthesis.") -> str:
    return json.dumps({"overall_comment": text})


def _question(qid: str, response: str = "Photosynthesis converts light.") -> OEQuestionInput:
    return OEQuestionInput(
        test_set_question_id=f"tsq-{qid}",
        question_id=f"q-{qid}",
        prompt=f"Explain {qid}.",
        rubric_json=json.dumps([{"criterion_id": "c1", "title": "Accuracy", "weight": 1.0}]),
        model_answer="Reference answer.",
        learner_response=response,
        points_possible=5,
        subject="science",
        topic=qid,
    )


def _base_state(questions: list[OEQuestionInput]) -> OEGradingState:
    return {
        "grading_job_id": "job-1",
        "submission_id": "sub-1",
        "assessment_id": "assess-1",
        "tenant_id": "tenant-1",
        "gcid": "learner-1",
        "passing_threshold_percent": 50,
        "model_tier": "T1",
        "per_question_feedback_enabled": True,
        "subject": "science",
        "total_points_possible": 5 * max(len(questions), 1),
        "mcq_points_earned": 0.0,
        "oe_questions": questions,
        "mcq_results": [],
        "pipeline_trace": [],
        "errors": [],
    }


def _cfg(thread_id: str = "sub-1") -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _eval_calls(ex: _FakeExecutor) -> list[dict[str, Any]]:
    return [c for c in ex.calls if c["agent_role"] == ROLE_OE_EVALUATE and c["mode"] == "evaluate"]


def _summary_calls(ex: _FakeExecutor) -> list[dict[str, Any]]:
    return [c for c in ex.calls if c["mode"] == "assess_summary"]


def _mod_calls(ex: _FakeExecutor) -> list[dict[str, Any]]:
    return [c for c in ex.calls if c["agent_role"] == ROLE_OE_MODERATE]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accept_first_pass_single_answer() -> None:
    ex = _FakeExecutor(
        responses={
            "evaluate": [_eval_payload(4.0)],  # 4/5 * weight1 * 5pts = 4.0
            "moderate": [_mod_payload(True)],
            "summary": [_summary_payload()],
        }
    )
    graph = build_oe_grading_crew_graph(executor=ex, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())

    terminal = await graph.ainvoke(_base_state([_question("a")]), config=_cfg())

    graded = terminal["graded"]
    assert len(graded) == 1
    assert graded[0].attempt_count == 1
    assert graded[0].quality_flagged is False
    assert graded[0].points_earned == pytest.approx(4.0)  # Python-derived composite
    assert graded[0].comment == "Good answer."
    assert terminal["outcome"] == "SUCCESS"
    assert terminal["overall_comment"].startswith("Overall 80%")
    assert len(_eval_calls(ex)) == 1
    assert len(_mod_calls(ex)) == 1
    assert len(_summary_calls(ex)) == 1


@pytest.mark.asyncio
async def test_reject_then_regrade_within_loop() -> None:
    ex = _FakeExecutor(
        responses={
            "evaluate": [_eval_payload(2.0), _eval_payload(5.0)],
            "moderate": [_mod_payload(False, "criterion c1 under-scored vs evidence"), _mod_payload(True)],
            "summary": [_summary_payload()],
        }
    )
    graph = build_oe_grading_crew_graph(executor=ex, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())

    terminal = await graph.ainvoke(_base_state([_question("a")]), config=_cfg())

    graded = terminal["graded"]
    assert len(graded) == 1
    assert graded[0].attempt_count == 2  # initial + 1 re-grade
    assert graded[0].quality_flagged is False
    assert graded[0].points_earned == pytest.approx(5.0)  # last evaluator output (5/5*5)
    ecalls = _eval_calls(ex)
    assert len(ecalls) == 2
    # Moderator feedback threaded into the re-grade dispatch as prior_moderator_feedback.
    assert "under-scored" in ecalls[1]["input_payload"]


@pytest.mark.asyncio
async def test_exhaustion_ships_quality_flagged_three_grades() -> None:
    # DEFAULT_MAX_ITERATIONS=2 → 3 total grades before forced record.
    ex = _FakeExecutor(
        responses={
            "evaluate": [_eval_payload(1.0), _eval_payload(1.5), _eval_payload(2.0)],
            "moderate": [
                _mod_payload(False, "still off"),
                _mod_payload(False, "still off"),
                _mod_payload(False, "still off"),
            ],
            "summary": [_summary_payload()],
        }
    )
    graph = build_oe_grading_crew_graph(executor=ex, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())

    terminal = await graph.ainvoke(_base_state([_question("a")]), config=_cfg())

    graded = terminal["graded"]
    assert len(graded) == 1
    assert graded[0].quality_flagged is True
    assert graded[0].attempt_count == 3  # bounded — no infinite loop
    assert graded[0].points_earned == pytest.approx(2.0)  # last output shipped (2/5*5)
    assert len(_eval_calls(ex)) == 3
    assert len(_mod_calls(ex)) == 3
    assert terminal["outcome"] == "PARTIAL"  # any flagged → PARTIAL


@pytest.mark.asyncio
async def test_assess_summary_runs_exactly_once_after_last_answer() -> None:
    ex = _FakeExecutor(
        responses={
            "evaluate": [_eval_payload(4.0), _eval_payload(3.0)],
            "moderate": [_mod_payload(True), _mod_payload(True)],
            "summary": [_summary_payload("Two-answer summary.")],
        }
    )
    graph = build_oe_grading_crew_graph(executor=ex, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())

    terminal = await graph.ainvoke(_base_state([_question("a"), _question("b")]), config=_cfg())

    assert len(terminal["graded"]) == 2
    scalls = _summary_calls(ex)
    assert len(scalls) == 1
    assert scalls[0]["agent_role"] == ROLE_OE_EVALUATE  # summary is an evaluator mode, never moderator
    assert terminal["overall_comment"] == "Two-answer summary."
    # results_digest carries both answers' outcomes.
    assert "results_digest" in json.loads(scalls[0]["input_payload"])


@pytest.mark.asyncio
async def test_criterion_scores_preserved_on_graded() -> None:
    ex = _FakeExecutor(
        responses={
            "evaluate": [_eval_payload(4.0)],
            "moderate": [_mod_payload(True)],
            "summary": [_summary_payload()],
        }
    )
    graph = build_oe_grading_crew_graph(executor=ex, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())

    terminal = await graph.ainvoke(_base_state([_question("a")]), config=_cfg())

    g = terminal["graded"][0]
    criteria = json.loads(g.criterion_scores_json)
    assert isinstance(criteria, list)
    assert criteria[0]["criterion_id"] == "c1"
    assert g.points_possible == 5


@pytest.mark.asyncio
async def test_guardrail_pre_block_fails_only_that_answer() -> None:
    bad = _question("a", response="my SSN is 123-45-6789 LEAKTOKEN")
    good = _question("b")
    ex = _FakeExecutor(
        responses={
            "evaluate": [_eval_payload(4.0)],  # only the good answer reaches the evaluator
            "moderate": [_mod_payload(True)],
            "summary": [_summary_payload()],
        }
    )
    graph = build_oe_grading_crew_graph(
        executor=ex,
        guardrail=_FakeGuardrail(block_substrings=["LEAKTOKEN"]),
        checkpointer=MemorySaver(),
    )

    terminal = await graph.ainvoke(_base_state([bad, good]), config=_cfg())

    graded = terminal["graded"]
    assert len(graded) == 2
    blocked = next(g for g in graded if g.test_set_question_id == "tsq-a")
    ok = next(g for g in graded if g.test_set_question_id == "tsq-b")
    assert blocked.quality_flagged is True
    assert blocked.points_earned == pytest.approx(0.0)
    assert blocked.grading_model_id == "guardrail_block"
    assert ok.quality_flagged is False
    assert ok.points_earned == pytest.approx(4.0)
    assert terminal["outcome"] == "PARTIAL"
    assert len(_eval_calls(ex)) == 1  # blocked answer never hit the evaluator


@pytest.mark.asyncio
async def test_no_oe_questions_still_runs_summary() -> None:
    ex = _FakeExecutor(responses={"summary": [_summary_payload("MCQ-only summary.")]})
    state = _base_state([])
    state["mcq_points_earned"] = 8.0
    state["total_points_possible"] = 10
    graph = build_oe_grading_crew_graph(executor=ex, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())

    terminal = await graph.ainvoke(state, config=_cfg())

    assert terminal["graded"] == []
    assert terminal["overall_comment"] == "MCQ-only summary."
    assert terminal["outcome"] == "SUCCESS"


def test_build_rejects_none_guardrail() -> None:
    """ADR-250 D1: a nil guardrail port is a construction failure, never a
    silent pass-through. A grading crew that scores a learner's open-ended
    answer is not the workload to fail open on: qgen already refuses to run when
    its mapping is unreadable (adapter/pubsub/qgen_crew_wiring.py), and this
    crew now fails the same way."""
    ex = _FakeExecutor(responses={})

    with pytest.raises(ValueError, match="guardrail"):
        build_oe_grading_crew_graph(executor=ex, guardrail=None, checkpointer=MemorySaver())


def test_build_requires_an_explicit_guardrail_argument() -> None:
    """ADR-250 D1: the port has no default, so forgetting to wire it is a
    TypeError at construction rather than an unscreened run."""
    ex = _FakeExecutor(responses={})

    with pytest.raises(TypeError):
        build_oe_grading_crew_graph(executor=ex, checkpointer=MemorySaver())  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_guardrail_pre_screens_nothing_when_there_is_no_question() -> None:
    """The only remaining early return is "there is nothing to screen", never
    "there is no screener" (ADR-250 D1): with no current question the port is
    not called at all, and the graph routes on to the summary."""
    gr = _FakeGuardrail()

    out = await guardrail_pre_node(_base_state([]), guardrail=gr)

    assert out["guardrail_pre_result"].allowed is True
    assert gr.calls == []


@pytest.mark.asyncio
async def test_guardrail_post_screens_nothing_when_there_is_no_evaluation() -> None:
    """Same invariant on the post leg: no evaluation means no grader comment
    exists yet, so there is no content to screen and the port is not called."""
    gr = _FakeGuardrail()

    out = await guardrail_post_node(_base_state([]), guardrail=gr)

    assert out["guardrail_post_result"].allowed is True
    assert gr.calls == []


@pytest.mark.asyncio
async def test_each_crew_member_screens_under_its_own_agent_id() -> None:
    """ADR-250 D2: the pre-screen (learner answer bound for the evaluator) is
    attributed to oe_evaluator; the post-screen (grader comment on its way into
    the moderator) is attributed to oe_moderator, so per-agent guardrail
    telemetry can separate the two members of the crew.

    Attribution only: both ids are declared balanced in
    chora-contracts/yaml/agent-guardrail-mapping.yaml and neither is raised to
    strict by the gateway switch, so no tier moves."""
    ex = _FakeExecutor(
        responses={
            "evaluate": [_eval_payload(4.0)],
            "moderate": [_mod_payload(True)],
            "summary": [_summary_payload()],
        }
    )
    gr = _FakeGuardrail()
    graph = build_oe_grading_crew_graph(executor=ex, guardrail=gr, checkpointer=MemorySaver())

    await graph.ainvoke(_base_state([_question("a")]), config=_cfg())

    attribution = [(c.direction, c.agent_id) for c in gr.calls]
    assert attribution == [("input", "oe_evaluator"), ("output", "oe_moderator")]


@pytest.mark.asyncio
async def test_idempotent_state_replay_same_thread() -> None:
    def make_ex() -> _FakeExecutor:
        return _FakeExecutor(
            responses={
                "evaluate": [_eval_payload(4.0)],
                "moderate": [_mod_payload(True)],
                "summary": [_summary_payload()],
            }
        )

    cp = MemorySaver()
    t1 = await build_oe_grading_crew_graph(
        executor=make_ex(),
        guardrail=_FakeGuardrail(),
        checkpointer=cp,
    ).ainvoke(_base_state([_question("a")]), config=_cfg("sub-X"))

    # Re-invoke same thread_id: checkpointer holds the terminal state.
    t2 = await build_oe_grading_crew_graph(
        executor=make_ex(),
        guardrail=_FakeGuardrail(),
        checkpointer=cp,
    ).ainvoke(None, config=_cfg("sub-X"))

    assert t2["outcome"] == t1["outcome"]
    assert len(t2["graded"]) == len(t1["graded"])
    assert t2["graded"][0].points_earned == pytest.approx(t1["graded"][0].points_earned)
