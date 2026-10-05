"""RED — ADR-253 D2: the OE grading runner parks and resumes.

The behavioural change this file pins down: ``handle_requested`` no longer runs
a submission to completion. It drives the graph to its FIRST agent dispatch and
returns while the run is parked. The terminal ``submission_completed.v1`` must
therefore move to wherever the graph actually finishes, which is the LAST
completion event to arrive — not the request handler.

Getting this wrong is silent and expensive in both directions: publishing on the
parked return grades a learner on an empty result set, and never publishing on
the resume leaves the submission in PENDING_OE_GRADING forever.
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
)


class _Interrupt:
    def __init__(self, value: Any) -> None:
        self.value = value


@dataclass
class _ScriptedGraph:
    """Returns each scripted result in turn, recording how it was invoked."""

    results: list[dict[str, Any]]
    invocations: list[dict[str, Any]] = field(default_factory=list)

    async def ainvoke(self, state: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.invocations.append({"state": state, "config": config})
        idx = min(len(self.invocations) - 1, len(self.results) - 1)
        return self.results[idx]


@dataclass
class _FakePublisher:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def publish_submission_completed(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "row-1"


def _graded() -> GradedOEQuestion:
    return GradedOEQuestion(
        test_set_question_id="tsq-1",
        question_id="q-1",
        points_earned=4.0,
        points_possible=5,
        criterion_scores_json=json.dumps([{"score": 4}]),
        comment="ok",
        grading_model_id="m",
        grading_response_id="r",
        quality_flagged=False,
        attempt_count=1,
    )


def _base_state() -> dict[str, Any]:
    return {
        "submission_id": "01a02062-e5b4-7870-8fca-53ce363cd542",
        "grading_job_id": "job-1",
        "assessment_id": "a-1",
        "tenant_id": "11111111-1111-7111-8111-111111111111",
        "gcid": "g-1",
        "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01",
        "tracestate": "",
        "subject": "Physics",
        "max_iterations": 2,
    }


def _parked() -> dict[str, Any]:
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
        DISPATCH_INTERRUPT_KEY,
        build_dispatch_request,
    )

    req = build_dispatch_request(
        agent_role="oe_evaluate",
        execution_id="01a02062-e5b4-7870-8fca-53ce363cd542:tsq-1:1",
        tenant_id="11111111-1111-7111-8111-111111111111",
        gcid="g-1",
        thread_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        input_payload="{}",
        source_project="chora-489812",
    )
    return {**_base_state(), "__interrupt__": [_Interrupt({DISPATCH_INTERRUPT_KEY: req})]}


def _terminal() -> dict[str, Any]:
    return {
        **_base_state(),
        "outcome": "SUCCESS",
        "graded": [_graded()],
        "overall_comment": "good work",
        "overall_comment_model_id": "m",
        "overall_comment_response_id": "r",
        "pipeline_trace": [],
    }


_EVENT = {
    "submission_id": "01a02062-e5b4-7870-8fca-53ce363cd542",
    "grading_job_id": "job-1",
    "assessment_id": "a-1",
    "tenant_id": "11111111-1111-7111-8111-111111111111",
    "learner_gcid": "g-1",
    "questions": [
        {
            "question_type": "OE",
            "test_set_question_id": "tsq-1",
            "question_id": "q-1",
            "prompt": "Explain",
            "rubric_json": "[]",
            "oe_response_text": "because",
            "points_possible": 5,
        }
    ],
    "total_points_possible": 5,
    "passing_threshold_percent": 50,
}


# ---------------------------------------------------------------------------


async def test_a_parked_run_does_not_publish_a_terminal() -> None:
    """The single most expensive mistake available here: publishing on the
    parked return would tell chora-delivery the submission is graded when not
    one agent has answered."""
    graph, pub = _ScriptedGraph([_parked()]), _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub)

    await runner.handle_requested(dict(_EVENT))

    assert pub.calls == [], "published a terminal while the run was still parked"


async def test_a_completion_that_finishes_the_run_publishes_the_terminal() -> None:
    graph, pub = _ScriptedGraph([_terminal()]), _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub)

    await runner.handle_completion(
        {
            "thread_id": "01a02062-e5b4-7870-8fca-53ce363cd542",
            "status": "OK",
            "output_payload": "{}",
            "idempotency_key": "agent_dispatch.oe_evaluate.01a02062-e5b4-7870-8fca-53ce363cd542:tsq-1:1",
        }
    )

    assert len(pub.calls) == 1
    assert pub.calls[0]["submission_id"] == "01a02062-e5b4-7870-8fca-53ce363cd542"
    assert pub.calls[0]["outcome"] == "SUCCESS"
    # Read off the RESUMED terminal state, not off a request the resume path
    # never saw — this is the field most likely to silently go blank.
    assert pub.calls[0]["grading_job_id"] == "job-1"
    assert pub.calls[0]["traceparent"] == "00-" + "a" * 32 + "-" + "b" * 16 + "-01"


async def test_a_completion_that_re_parks_publishes_nothing() -> None:
    """A submission has many dispatches. Only the last completion finishes it."""
    graph, pub = _ScriptedGraph([_parked()]), _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub)

    await runner.handle_completion(
        {
            "thread_id": "01a02062-e5b4-7870-8fca-53ce363cd542",
            "status": "OK",
            "output_payload": "{}",
        }
    )

    assert pub.calls == []


async def test_the_completion_resumes_the_thread_it_names() -> None:
    """The consumer keeps no state of its own; the thread key rides the event."""
    from langgraph.types import Command

    graph, pub = _ScriptedGraph([_terminal()]), _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub)

    await runner.handle_completion(
        {"thread_id": "01a02062-72e3-7da4-aeb5-c441ed92d9c8", "status": "OK", "output_payload": "{}"}
    )

    inv = graph.invocations[0]
    assert inv["config"]["configurable"]["thread_id"] == "01a02062-72e3-7da4-aeb5-c441ed92d9c8"
    assert isinstance(inv["state"], Command), "resume must use Command(resume=...)"
    assert inv["state"].resume["status"] == "OK"


async def test_a_completion_with_no_thread_id_is_refused() -> None:
    """Resuming a guessed thread would corrupt an unrelated run."""
    graph, pub = _ScriptedGraph([_terminal()]), _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub)

    with pytest.raises(ValueError, match="thread_id"):
        await runner.handle_completion({"status": "OK", "output_payload": "{}"})
    assert graph.invocations == []


async def test_a_run_that_terminates_without_parking_still_publishes() -> None:
    """A validate_input failure never reaches a dispatch. That path must keep
    publishing its terminal or a malformed submission hangs forever."""
    failed = {
        **_base_state(),
        "outcome": "FAILED",
        "failure_message": "missing submission_id",
        "graded": [],
        "pipeline_trace": [],
    }
    graph, pub = _ScriptedGraph([failed]), _FakePublisher()
    runner = OEGradingCrewRunner(graph=graph, publisher=pub)

    await runner.handle_requested(dict(_EVENT))

    assert len(pub.calls) == 1
    assert pub.calls[0]["outcome"] == "FAILED"
