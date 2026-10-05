"""RED: ADR-254 D5, the OE crew SETTLES a FAILED agent dispatch.

Under ADR-253 the executor raises ``AgentDispatchError`` inside the node when
the completion carries ``status=FAILED`` (an agent failure, or the park reaper
settling a run the agent never answered). Today ``evaluate_node`` and
``moderate_node`` let that raise escape the graph: the run ends with no
terminal, chora-delivery never hears, and the reaper's DLQ message would
redeliver forever. The contract pinned here: a FAILED dispatch becomes
``outcome=FAILED`` + ``failure_message`` in state, the graph routes straight to
the terminal (no further dispatch), ``publish_completed`` keeps FAILED, and the
runner publishes ``submission_completed`` with ``outcome=FAILED`` so the caller
gets a status, never silence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    GuardrailScreenInput,
    ScreenResult,
    Verdict,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
    AgentDispatchError,
)
from chora_ai_kernel_orchestrator.domain.oe_grading_crew import (
    EvaluationResult,
    OEQuestionInput,
)
from chora_ai_kernel_orchestrator.orchestrators.oe_grading_crew import (
    build_oe_grading_crew_graph,
    evaluate_node,
    moderate_node,
    publish_completed_node,
    route_after_evaluate,
    route_after_moderate,
)
from chora_ai_kernel_orchestrator.orchestrators.oe_grading_crew_runner import (
    OEGradingCrewRunner,
)

_SUBMISSION = "01a02062-e5b4-7870-8fca-53ce363cd542"


@dataclass
class _FailingExecutor:
    """Every dispatch comes back FAILED (the reaper's synthesized completion)."""

    calls: list[str] = field(default_factory=list)

    async def execute(self, *, execution_id: str, agent_role: str, **_: Any) -> Any:
        self.calls.append(f"{agent_role}:{execution_id}")
        raise AgentDispatchError(
            f"{agent_role} dispatch {execution_id} returned status=FAILED: "
            "request_dead_lettered: request dead-lettered after 5 delivery attempts"
        )


@dataclass
class _FakeGuardrail:
    calls: list[GuardrailScreenInput] = field(default_factory=list)

    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
        self.calls.append(payload)
        return ScreenResult(verdict=Verdict.ALLOW, reason="")


@dataclass
class _FakePublisher:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def publish_submission_completed(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "row-1"


def _question() -> OEQuestionInput:
    return OEQuestionInput(
        test_set_question_id="tsq-1",
        question_id="q-1",
        prompt="Explain osmosis.",
        rubric_json=json.dumps([{"criterion": "accuracy", "weight": 1.0}]),
        model_answer="Water moves across a membrane.",
        learner_response="Water moves.",
        points_possible=5,
        subject="Biology",
        topic="cells",
    )


def _state() -> dict[str, Any]:
    return {
        "grading_job_id": "job-1",
        "submission_id": _SUBMISSION,
        "assessment_id": "a-1",
        "tenant_id": "11111111-1111-7111-8111-111111111111",
        "gcid": "00000000-0000-7000-8000-000000001999",
        "passing_threshold_percent": 50,
        "model_tier": "T1",
        "per_question_feedback_enabled": True,
        "subject": "Biology",
        "total_points_possible": 5,
        "mcq_points_earned": 0.0,
        "traceparent": "",
        "tracestate": "",
        "max_iterations": 2,
        "oe_questions": [_question()],
        "mcq_results": [],
        "graded": [],
        "pipeline_trace": [],
        "errors": [],
        "current_index": 0,
        "attempt": 1,
    }


def _event() -> dict[str, Any]:
    """The inbound submission_requested.v1 body the runner parses itself."""
    return {
        "grading_job_id": "job-1",
        "submission_id": _SUBMISSION,
        "assessment_id": "a-1",
        "tenant_id": "11111111-1111-7111-8111-111111111111",
        "learner_gcid": "00000000-0000-7000-8000-000000001999",
        "passing_threshold_percent": 50,
        "total_points_possible": 5,
        "mcq_points_earned": 0,
        "per_question_feedback_enabled": True,
        "subject": "Biology",
        "questions": [
            {
                "question_type": "OE",
                "test_set_question_id": "tsq-1",
                "question_id": "q-1",
                "prompt": "Explain osmosis.",
                "rubric_json": json.dumps([{"criterion": "accuracy", "weight": 1.0}]),
                "model_answer": "Water moves across a membrane.",
                "oe_response_text": "Water moves.",
                "points_possible": 5,
                "subject": "Biology",
                "topic": "cells",
            }
        ],
    }


@pytest.mark.asyncio
async def test_evaluate_node_settles_a_failed_dispatch_as_outcome_failed() -> None:
    ex = _FailingExecutor()
    result = await evaluate_node(_state(), executor=ex)
    assert ex.calls, "the node must reach the dispatch"
    assert result["outcome"] == "FAILED"
    assert "request_dead_lettered" in result["failure_message"]
    assert "evaluate" in result["failure_message"]
    assert result["pipeline_trace"][-1]["status"] == "DISPATCH_FAILED"


@pytest.mark.asyncio
async def test_moderate_node_settles_a_failed_dispatch_as_outcome_failed() -> None:
    state = _state()
    state["current_evaluation"] = EvaluationResult(
        points_earned=4.0,
        points_possible=5,
        criterion_scores_json="[]",
        comment="ok",
        grading_model_id="m",
        grading_response_id="r",
    )
    result = await moderate_node(state, executor=_FailingExecutor())
    assert result["outcome"] == "FAILED"
    assert "moderate" in result["failure_message"]


def test_routes_send_a_failed_outcome_straight_to_the_terminal() -> None:
    assert route_after_evaluate({"outcome": "FAILED"}) == "failed"
    assert route_after_evaluate({"outcome": ""}) == "screen"
    assert route_after_evaluate({}) == "screen"
    assert route_after_moderate({"outcome": "FAILED"}) == "failed"
    assert route_after_moderate({}) == "gate"


@pytest.mark.asyncio
async def test_publish_completed_keeps_a_failed_outcome() -> None:
    result = await publish_completed_node({**_state(), "outcome": "FAILED", "failure_message": "x"})
    assert result.get("outcome", "FAILED") == "FAILED"
    assert result["pipeline_trace"][-1]["status"] == "FAILED"


@pytest.mark.asyncio
async def test_the_graph_reaches_the_terminal_with_failed_and_no_second_dispatch() -> None:
    ex = _FailingExecutor()
    graph = build_oe_grading_crew_graph(executor=ex, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    terminal = await graph.ainvoke(_state(), config={"configurable": {"thread_id": _SUBMISSION}})
    assert not terminal.get("__interrupt__")
    assert terminal["outcome"] == "FAILED"
    assert "request_dead_lettered" in terminal["failure_message"]
    assert terminal["graded"] == []
    assert ex.calls == [f"oe_evaluate:{_SUBMISSION}:tsq-1:1"], ex.calls


@pytest.mark.asyncio
async def test_the_runner_publishes_the_failed_terminal_to_the_caller() -> None:
    ex = _FailingExecutor()
    graph = build_oe_grading_crew_graph(executor=ex, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    publisher = _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=publisher)
    await runner.handle_requested(_event())
    assert len(publisher.calls) == 1
    call = publisher.calls[0]
    assert call["submission_id"] == _SUBMISSION
    assert call["outcome"] == "FAILED"
    assert "request_dead_lettered" in call["failure_message"]
    assert call["graded"] == []
