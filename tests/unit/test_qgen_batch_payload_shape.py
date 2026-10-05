"""QGenBatchRunner payload-shape tests (Lane 1c W2 slice 4, CHO-1703; ADR-254 D2).

The completed.v1 ``candidate_payload_json`` contract (BatchCandidatePayload,
creation-questions.yaml v1.5.0):

* compose ENABLED (QGEN_TESTSET_COMPOSE_ENABLED) -> the set lane's
  compose_test_set node dispatches ONE mode=compose call on the qgen_generate
  lane and the payload is the JSON OBJECT ``{"candidates": [...],
  "proposed_test_set": {...}}``; parsers branch on first non-space char ``{``.
* compose DISABLED -> the object wrapper WITHOUT proposed_test_set
  (``{"candidates": [...]}``); the legacy bare array retired with the N-loop.
* the single-candidate path emits the lone candidate object VERBATIM,
  untouched by this lane (regression-pinned).
* a compose failure (a FAILED dispatch, an unusable answer) must NEVER fail the
  batch: deterministic fallback proposal, trace row FALLBACK.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_CRITIQUE,
    ROLE_GENERATE,
    build_qgen_crew_graph,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
    QGenBatchRunner,
    QGenCrewRunner,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_grounding import (
    deterministic_draft_id,
)
from tests.integration.test_qgen_crew_set import _FakeSetExecutor, _mcq, _wrapper
from tests.unit.test_qgen_batch_multifile import _multifile_event
from tests.unit.test_qgen_crew_runner import (
    _critic_accept,
    _FakeExecutor,
    _FakeGuardrail,
    _FakePublisher,
    _good_mcq_payload,
    _started_event,
)


def _runner_with(compose: Any, *, count: int = 2) -> tuple[QGenBatchRunner, _FakeSetExecutor, _FakePublisher]:
    """``compose``: False => feature off; True => the fake's default answer;
    a dict => that answer; an Exception => a FAILED compose dispatch."""
    executor = _FakeSetExecutor(
        generate_queue=[_wrapper([_mcq(f"Q{i}") for i in range(count)])],
        compose_answer=None if compose is True else compose,
    )
    publisher = _FakePublisher()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenBatchRunner(graph=graph, publisher=publisher, compose_enabled=compose is not False)
    return runner, executor, publisher


@pytest.mark.asyncio
async def test_compose_enabled_emits_batch_candidate_payload_object() -> None:
    d0 = deterministic_draft_id("job-1c", 0)
    d1 = deterministic_draft_id("job-1c", 1)
    answer = {
        "proposed_test_set": {
            "title": "Mock Paper",
            "description": "From the uploaded exam.",
            "order": [d1, d0],
            "points": {d0: 5, d1: 10},
        }
    }
    runner, executor, publisher = _runner_with(answer)

    await runner.handle_started(_multifile_event(requested_count=2))

    raw = publisher.completed[0]["candidate_payload_json"]
    assert raw.lstrip()[0] == "{"  # parsers branch on first char
    obj = json.loads(raw)
    assert set(obj.keys()) == {"candidates", "proposed_test_set"}
    assert [c["draft_id"] for c in obj["candidates"]] == [d0, d1]
    assert obj["proposed_test_set"]["order"] == [d1, d0]
    assert obj["proposed_test_set"]["points"] == {d0: 5, d1: 10}
    # ONE compose dispatch on the generate lane, carrying the SAME candidates
    # (with the SAME deterministic draft ids) the payload publishes, the
    # author's prompt, and the source files BY REFERENCE for the agent.
    assert len(executor.compose_calls) == 1
    sent = executor.compose_calls[0]
    assert sent["mode"] == "compose"
    assert [c["draft_id"] for c in sent["candidates"]] == [d0, d1]
    assert sent["candidates"] == obj["candidates"]
    assert sent["author_prompt"] == "Create a test set like the uploaded paper."
    assert [f["gs_uri"] for f in sent["source_files"]] == [
        "gs://b/exam.pdf",
        "gs://b/fig.png",
        "gs://b/marks.pdf",
    ]
    assert [f["role"] for f in sent["source_files"]] == ["source", "source", "rubric"]
    assert sent["grounding_mode"] == "strict"
    compose_calls = [c for c in executor.calls if c["execution_id"].endswith(":compose")]
    assert compose_calls and compose_calls[0]["agent_role"] == ROLE_GENERATE


@pytest.mark.asyncio
async def test_compose_disabled_emits_the_candidates_wrapper_without_a_proposal() -> None:
    # ADR-254 D2: the legacy bare array went with the N-loop; with the feature
    # off the set lane publishes the object wrapper WITHOUT proposed_test_set
    # (the parser branches on the first char and is object-aware since
    # v1.5.0) and dispatches NO compose call.
    runner, executor, publisher = _runner_with(False)

    await runner.handle_started(_multifile_event(requested_count=2))

    raw = publisher.completed[0]["candidate_payload_json"]
    assert raw.lstrip()[0] == "{"
    obj = json.loads(raw)
    assert set(obj.keys()) == {"candidates"} and len(obj["candidates"]) == 2
    assert executor.compose_calls == []
    trace = json.loads(publisher.completed[0]["pipeline_trace_json"])
    assert not [r for r in trace if r.get("name") == "compose_test_set"]


@pytest.mark.asyncio
async def test_failed_compose_dispatch_falls_back_deterministically() -> None:
    # D4: a FAILED compose completion (the agent could not shape an answer, a
    # reaped park) must never fail the batch: the deterministic proposal ships.
    runner, _, publisher = _runner_with(
        RuntimeError("qgen_generate dispatch returned status=FAILED: compose_unusable_answer")
    )

    await runner.handle_started(_multifile_event(requested_count=2))

    obj = json.loads(publisher.completed[0]["candidate_payload_json"])
    proposal = obj["proposed_test_set"]
    d0 = deterministic_draft_id("job-1c", 0)
    d1 = deterministic_draft_id("job-1c", 1)
    assert proposal["order"] == [d0, d1]  # submission order
    assert proposal["points"] == {d0: 10, d1: 10}  # uniform default
    assert proposal["title"].startswith("Test set")


@pytest.mark.asyncio
async def test_unusable_compose_answer_falls_back_deterministically() -> None:
    # The agent answered, but without a proposed_test_set object: the kennel
    # never guesses; the deterministic proposal ships and the row says FALLBACK.
    runner, _, publisher = _runner_with({"nonsense": 1})

    await runner.handle_started(_multifile_event(requested_count=2))

    obj = json.loads(publisher.completed[0]["candidate_payload_json"])
    d0 = deterministic_draft_id("job-1c", 0)
    d1 = deterministic_draft_id("job-1c", 1)
    assert obj["proposed_test_set"]["order"] == [d0, d1]
    trace = json.loads(publisher.completed[0]["pipeline_trace_json"])
    rows = [r for r in trace if r.get("name") == "compose_test_set"]
    assert rows and rows[0]["status"] == "FALLBACK"
    assert "no proposed_test_set" in rows[0]["notes"]


@pytest.mark.asyncio
async def test_compose_trace_row_appended_when_enabled() -> None:
    runner, _, publisher = _runner_with(True, count=1)

    await runner.handle_started(_multifile_event(requested_count=1))

    trace = json.loads(publisher.completed[0]["pipeline_trace_json"])
    rows = [r for r in trace if r.get("name") == "compose_test_set"]
    assert len(rows) == 1
    assert rows[0]["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_compose_trace_row_marks_fallback() -> None:
    runner, _, publisher = _runner_with(RuntimeError("boom"), count=1)

    await runner.handle_started(_multifile_event(requested_count=1))

    trace = json.loads(publisher.completed[0]["pipeline_trace_json"])
    rows = [r for r in trace if r.get("name") == "compose_test_set"]
    assert rows and rows[0]["status"] == "FALLBACK"


@pytest.mark.asyncio
async def test_refused_batch_never_dispatches_compose() -> None:
    executor = _FakeSetExecutor(generate_queue=[_wrapper([_mcq("Q0")])])
    publisher = _FakePublisher()
    graph = build_qgen_crew_graph(
        executor=executor,
        guardrail=_FakeGuardrail(block_substrings=["FORBIDDEN"]),
        checkpointer=MemorySaver(),
    )
    runner = QGenBatchRunner(graph=graph, publisher=publisher, compose_enabled=True)

    await runner.handle_started(_multifile_event(requested_count=1, prompt="contains FORBIDDEN text"))

    assert publisher.refused and not publisher.completed
    assert executor.compose_calls == []


@pytest.mark.asyncio
async def test_single_path_payload_untouched_by_composer_feature() -> None:
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_good_mcq_payload()], ROLE_CRITIQUE: [_critic_accept()]})
    publisher = _FakePublisher()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=publisher)

    await runner.handle_started(_started_event(content_type="mcq", question_type="mcq", prompt="plain"))

    raw = publisher.completed[0]["candidate_payload_json"]
    obj = json.loads(raw)
    # The lone candidate object VERBATIM — no draft_id, no envelope keys.
    assert "candidates" not in obj
    assert "proposed_test_set" not in obj
    assert "draft_id" not in obj
    assert obj["stem"]
