"""QGenBatchRunner SET-native lane unit tests (CHO-1819 P1b).

Every batch rides the set lane (ADR-254 D2): a typed ``type_plan`` is used as
sent; an EMPTY one is synthesised as the producer's single-quota plan. The
completed.v1 carries the object wrapper + typed generated_count +
generation_summary. These tests pin the set lane.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    build_qgen_crew_graph,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
    AiAssistStartedPayload,
    QGenBatchRunner,
)
from tests.integration.test_qgen_crew_set import (
    _FakeGuardrail,
    _FakeSetExecutor,
    _mcq,
    _oe,
    _wrapper,
)
from tests.unit.test_qgen_crew_runner import _FakePublisher, _started_event

_MIXED = [
    {"question_type": "mcq", "count": 8, "max_images": 0},
    {"question_type": "oe", "count": 2, "max_images": 0},
]


def _set_event(**overrides: Any) -> dict[str, Any]:
    base = _started_event(
        assist_id="job-set-batch",
        content_type="mixed",
        question_type="mixed",
        prompt="Generate a mixed set from the source.",
        job_kind="batch",
        requested_count=10,
        type_plan=[dict(q) for q in _MIXED],
    )
    base.update(overrides)
    return base


def _set_runner(executor: Any, publisher: Any, **kwargs: Any) -> QGenBatchRunner:
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    return QGenBatchRunner(graph=graph, publisher=publisher, **kwargs)


# -----------------------------------------------------------------------------
# from_event + _validate — type_plan parsing
# -----------------------------------------------------------------------------


def test_from_event_parses_type_plan() -> None:
    p = AiAssistStartedPayload.from_event(_set_event())
    assert p.type_plan == (
        {"question_type": "mcq", "count": 8, "max_images": 0},
        {"question_type": "oe", "count": 2, "max_images": 0},
    )


def test_from_event_empty_type_plan_defaults_to_tuple() -> None:
    p = AiAssistStartedPayload.from_event(_started_event())
    assert p.type_plan == ()


def test_validate_accepts_mixed_only_with_type_plan() -> None:
    runner = _set_runner(_FakeSetExecutor(generate_queue=[]), _FakePublisher())
    # mixed WITH a type_plan is valid.
    runner._validate(AiAssistStartedPayload.from_event(_set_event()))
    # mixed WITHOUT a type_plan is rejected (no quotas to validate against).
    with pytest.raises(ValueError):
        runner._validate(
            AiAssistStartedPayload.from_event(_set_event(type_plan=[], content_type="mixed", question_type="mixed"))
        )


# -----------------------------------------------------------------------------
# Set lane — ONE completed.v1 with the object wrapper + typed summary
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_batch_publishes_object_wrapper_with_summary() -> None:
    cands = [_mcq(f"MCQ {i}") for i in range(8)] + [_oe(f"OE {i}") for i in range(2)]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    publisher = _FakePublisher()
    runner = _set_runner(executor, publisher)

    await runner.handle_started(_set_event())

    assert len(publisher.refused) == 0
    assert len(publisher.completed) == 1
    ev = publisher.completed[0]
    # Object wrapper (NOT the legacy bare array).
    payload = json.loads(ev["candidate_payload_json"])
    assert isinstance(payload, dict)
    assert len(payload["candidates"]) == 10
    # Every candidate got a deterministic draft_id stamped.
    assert all(c.get("draft_id") for c in payload["candidates"])
    # Typed honest accounting rides the dedicated kwargs (proto fields 3 + 16).
    assert ev["generated_count"] == 10
    assert ev["generation_summary"]["requested_total"] == 10
    assert ev["generation_summary"]["generated_per_type"] == {"mcq": 8, "oe": 2}
    assert ev["generation_summary"]["shortfall_reason"] == ""


@pytest.mark.asyncio
async def test_set_batch_strict_shortfall_surfaces_generated_count() -> None:
    cands = [_mcq(f"MCQ {i}") for i in range(6)] + [_oe("OE 0")]
    reason = "Source supported 6 MCQ + 1 OE."
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands, shortfall_reason=reason)])
    publisher = _FakePublisher()
    runner = _set_runner(executor, publisher)

    await runner.handle_started(_set_event(grounding_mode="strict"))

    ev = publisher.completed[0]
    assert ev["generated_count"] == 7
    assert ev["generation_summary"]["generated_total"] == 7
    assert ev["generation_summary"]["shortfall_reason"] == reason


@pytest.mark.asyncio
async def test_set_batch_refusal_fails_whole_batch() -> None:
    # Pre-screen blocks the prompt → the whole set is refused (no partial).
    cands = [_mcq("a"), _mcq("b")]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    graph = build_qgen_crew_graph(
        executor=executor,
        guardrail=_FakeGuardrail(block_substrings=["FORBIDDEN"]),
        checkpointer=MemorySaver(),
    )
    publisher = _FakePublisher()
    runner = QGenBatchRunner(graph=graph, publisher=publisher)

    await runner.handle_started(
        _set_event(
            prompt="contains FORBIDDEN text",
            type_plan=[{"question_type": "mcq", "count": 2, "max_images": 0}],
            requested_count=2,
        )
    )

    assert len(publisher.completed) == 0
    assert len(publisher.refused) == 1
    assert publisher.refused[0]["refusal_reason"] in ("GUARDRAIL_PRE", "GUARDRAIL_POST")


@pytest.mark.asyncio
async def test_set_batch_non_strict_short_set_refuses() -> None:
    # Agent returns 7 of 10 with NO shortfall reason + non-strict ⇒ fail loud.
    cands = [_mcq(f"MCQ {i}") for i in range(7)]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    publisher = _FakePublisher()
    runner = _set_runner(executor, publisher)

    await runner.handle_started(_set_event(grounding_mode="starting_point"))

    assert len(publisher.completed) == 0
    assert len(publisher.refused) == 1
    assert publisher.refused[0]["refusal_reason"] == "VALIDATION"


@pytest.mark.asyncio
async def test_set_batch_with_compose_enabled_emits_proposed_test_set() -> None:
    from chora_ai_kernel_orchestrator.orchestrators.qgen_grounding import deterministic_draft_id

    cands = [_mcq("A"), _mcq("B")]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])  # default compose answer: reversed order
    publisher = _FakePublisher()
    runner = _set_runner(executor, publisher, compose_enabled=True)

    await runner.handle_started(
        _set_event(
            type_plan=[{"question_type": "mcq", "count": 2, "max_images": 0}],
            requested_count=2,
        )
    )

    ev = publisher.completed[0]
    payload = json.loads(ev["candidate_payload_json"])
    assert "proposed_test_set" in payload
    d0, d1 = deterministic_draft_id("job-set-batch", 0), deterministic_draft_id("job-set-batch", 1)
    assert payload["proposed_test_set"]["order"] == [d1, d0]
    assert payload["proposed_test_set"]["points"] == {d0: 7, d1: 7}
    assert len(executor.compose_calls) == 1
    # Typed summary still rides the dedicated kwargs (NOT the wrapper).
    assert ev["generated_count"] == 2


# -----------------------------------------------------------------------------
# AgentDecisionLog emission on the SET lane - CHO-2364. ONE graph run per set
# job -> ONE qgen_question + ONE qgen_critic decision, through the SAME
# emitter + condition builders as the single runner, so set_mode +
# request_surface ride the durable record. (This lane carried the daily-dose
# type_plan traffic that emitted ZERO decisions since 2026-07-04.)
# -----------------------------------------------------------------------------


from chora_ai_kernel_orchestrator.domain.prompt_registry import (  # noqa: E402
    EMBEDDED_PROMPT_VERSIONS,
)
from tests.unit.test_qgen_crew_runner import (  # noqa: E402
    _FakeAgentDecisionLogEmitter,
)


@pytest.mark.asyncio
async def test_set_batch_emits_decisions_with_set_mode_and_surface() -> None:
    cands = [_mcq(f"MCQ {i}") for i in range(8)] + [_oe(f"OE {i}") for i in range(2)]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    publisher = _FakePublisher()
    adl = _FakeAgentDecisionLogEmitter()
    runner = _set_runner(executor, publisher, agent_decision_emitter=adl)

    # The consumption dose lane stamps metadata.surface="campaign" (see
    # chora-consumption dose_campaign_wiring.go) - dose decisions must carry
    # request_surface="campaign".
    await runner.handle_started(_set_event(metadata={"subject": "Biology", "surface": "campaign"}))

    assert len(adl.emitted) == 2  # ONE pair for the single set-graph run
    by_agid = {e["agid"]: e for e in adl.emitted}
    assert set(by_agid) == {"qgen_question", "qgen_critic"}
    for ev in adl.emitted:
        assert ev["assist_id"] == "job-set-batch"
        assert ev["crew_id"] == "job-set-batch"
        assert ev["decision"] == "accepted"
        conditions = ev["prompt_conditions"]
        assert conditions["set_mode"] == "true"
        assert conditions["request_surface"] == "campaign"
        # Embedded prompt provenance rides the set lane too (CHO-2364).
        assert conditions["prompt_version"] == EMBEDDED_PROMPT_VERSIONS[ev["agid"]]
        assert conditions["prompt_source"] == "embedded"
    # Question conditions mirror the Go extractor: set_mode omits question_type.
    assert "question_type" not in by_agid["qgen_question"]["prompt_conditions"]
    # The critic's question_type is the payload's genuine value ("mixed").
    assert by_agid["qgen_critic"]["prompt_conditions"]["question_type"] == "mixed"
    # Terminal publish unaffected (additive).
    assert len(publisher.completed) == 1


@pytest.mark.asyncio
async def test_set_batch_dose_without_surface_omits_request_surface() -> None:
    # Authoring sends no metadata.surface -> request_surface must NOT be
    # fabricated on the decision record.
    cands = [_mcq(f"MCQ {i}") for i in range(8)] + [_oe(f"OE {i}") for i in range(2)]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    adl = _FakeAgentDecisionLogEmitter()
    runner = _set_runner(executor, _FakePublisher(), agent_decision_emitter=adl)

    await runner.handle_started(_set_event(metadata={}))

    assert len(adl.emitted) == 2
    for ev in adl.emitted:
        assert "request_surface" not in ev["prompt_conditions"]


@pytest.mark.asyncio
async def test_set_batch_refusal_emits_refused_decisions() -> None:
    cands = [_mcq("a"), _mcq("b")]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    graph = build_qgen_crew_graph(
        executor=executor,
        guardrail=_FakeGuardrail(block_substrings=["FORBIDDEN"]),
        checkpointer=MemorySaver(),
    )
    publisher = _FakePublisher()
    adl = _FakeAgentDecisionLogEmitter()
    runner = QGenBatchRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl)

    await runner.handle_started(
        _set_event(
            prompt="contains FORBIDDEN text",
            type_plan=[{"question_type": "mcq", "count": 2, "max_images": 0}],
            requested_count=2,
        )
    )

    assert len(publisher.refused) == 1
    assert len(adl.emitted) == 2
    for ev in adl.emitted:
        assert ev["decision"] == "refused"
        assert ev["prompt_conditions"]["set_mode"] == "true"


# -----------------------------------------------------------------------------
# Legacy lane untouched — empty type_plan still uses the bare-array N-loop
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_less_batch_rides_the_set_lane_with_a_synthesized_plan() -> None:
    """ADR-254 D2: the legacy N-loop is retired. A batch that arrives WITHOUT
    a type_plan is given the SAME single-quota plan the producer stamps for
    count>1 and runs the set lane: ONE generate dispatch, the object wrapper,
    the typed generated_count."""
    executor = _FakeSetExecutor(generate_queue=[_wrapper([_mcq(f"legacy {i}") for i in range(3)])])
    publisher = _FakePublisher()
    runner = _set_runner(executor, publisher)

    await runner.handle_started(
        _started_event(
            assist_id="job-legacy",
            content_type="mcq",
            question_type="mcq",
            job_kind="batch",
            requested_count=3,
        )
    )

    ev = publisher.completed[0]
    payload = json.loads(ev["candidate_payload_json"])
    assert isinstance(payload, dict) and len(payload["candidates"]) == 3
    assert ev["generated_count"] == 3
    gen_calls = [c for c in executor.calls if c["agent_role"] == "qgen_question"]
    assert len(gen_calls) == 1
    assert json.loads(gen_calls[0]["input_payload"])["type_plan"] == [
        {"question_type": "mcq", "count": 3, "max_images": 0}
    ]


def _mcq_json() -> str:
    return json.dumps(_mcq("legacy stem"))


# -----------------------------------------------------------------------------
# Live-trace streaming on the SET lane — progress_emitter wired → astream +
# one progress.v1 per node; unwired → byte-stable single-shot ainvoke.
# The canvas's mixed-type batch (job_kind batch + typed type_plan) rides THIS
# lane, and it emitted zero progress while the single lane streamed.
# -----------------------------------------------------------------------------


def _full_set() -> list[dict[str, Any]]:
    return [_mcq(f"MCQ {i}") for i in range(8)] + [_oe(f"OE {i}") for i in range(2)]


@pytest.mark.asyncio
async def test_set_lane_streams_progress_per_node() -> None:
    executor = _FakeSetExecutor(generate_queue=[_wrapper(_full_set())])
    publisher = _FakePublisher()
    runner = _set_runner(executor, publisher, progress_emitter=publisher)

    await runner.handle_started(_set_event())

    assert len(publisher.completed) == 1, publisher.refused
    # At least two mid-run nodes stream before the terminal publishes.
    assert len(publisher.progress) >= 2, publisher.progress
    step_indices = [p["step_index"] for p in publisher.progress]
    assert step_indices == sorted(step_indices)
    names = [p["step_name"] for p in publisher.progress]
    for p in publisher.progress:
        assert p["assist_id"] == "job-set-batch"
        # Only the TERMINALS are excluded from the live stream; the ADR-251 D1
        # chunk boundary (publish_chunk) is a mid-run row and MUST stream.
        assert p["step_name"] not in ("publish_completed", "publish_refused")
        trace = json.loads(p["pipeline_trace_json"])
        assert len(trace) == p["step_index"]
    assert "publish_chunk" in names, names


@pytest.mark.asyncio
async def test_set_lane_progress_failure_never_aborts_the_run() -> None:
    executor = _FakeSetExecutor(generate_queue=[_wrapper(_full_set())])
    publisher = _FakePublisher(fail_progress=True)
    runner = _set_runner(executor, publisher, progress_emitter=publisher)

    await runner.handle_started(_set_event())

    # Best-effort guard: every progress emit failed, the terminal still lands.
    assert len(publisher.completed) == 1
    assert publisher.progress == []


@pytest.mark.asyncio
async def test_set_lane_no_emitter_streams_nothing() -> None:
    executor = _FakeSetExecutor(generate_queue=[_wrapper(_full_set())])
    publisher = _FakePublisher()
    runner = _set_runner(executor, publisher)

    await runner.handle_started(_set_event())

    assert len(publisher.completed) == 1
    assert publisher.progress == []
