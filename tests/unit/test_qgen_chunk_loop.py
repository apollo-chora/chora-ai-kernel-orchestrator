"""ADR-251 D1 (CHO-2396) — the set lane becomes a GRAPH-structured chunk loop.

RED→GREEN per [[feedback-strict-tdd]].

A big set was ONE generate call: the token ceiling and the (now 120s) write
window bound N, and validate_type_plan's 50-cap refused large sets outright.
ADR-251 D1: ``chunk_type_plan`` is wired through a graph-structured loop
(plan_chunks → per-chunk generate/critique/gate/render/publish_chunk →
finalize_set), max_batch rises to 200, and QGEN_SET_MAX_PER_CALL (default 10)
bounds each generate call. The loop is IN THE GRAPH so every chunk stage
checkpoints at a node transition — a node-internal loop was rejected as a
shortcut (owner directive 2026-08-16): LangGraph persists at node transitions,
so in-node loop progress dies with the pod.

Graph-level proof through the REAL build_qgen_crew_graph + QGenBatchRunner with
the established set-lane fakes; no gRPC, no Pub/Sub.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_GENERATE,
    build_qgen_crew_graph,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
    QGenBatchRunner,
)
from tests.integration.test_qgen_crew_set import (
    _REJECT_MARKER,
    _FakeGcs,
    _FakeGuardrail,
    _FakeKroki,
    _FakeSetExecutor,
    _mcq,
    _wrapper,
)
from tests.unit.test_qgen_crew_runner import _FakePublisher, _started_event


def _mcq_plan(count: int, *, max_images: int = 0, **flags: Any) -> list[dict[str, Any]]:
    entry: dict[str, Any] = {
        "question_type": "mcq",
        "count": count,
        "max_images": max_images,
    }
    entry.update(flags)
    return [entry]


def _set_event(count: int, plan: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    base = _started_event(
        assist_id=f"job-chunk-{count}",
        content_type="mixed",
        question_type="mixed",
        prompt="Generate a mixed set from the source.",
        job_kind="batch",
        requested_count=count,
        type_plan=[dict(q) for q in plan],
    )
    base.update(overrides)
    return base


def _runner(
    executor: Any,
    publisher: Any,
    *,
    kroki: Any = None,
    gcs: Any = None,
) -> QGenBatchRunner:
    graph = build_qgen_crew_graph(
        executor=executor,
        guardrail=_FakeGuardrail(),
        checkpointer=MemorySaver(),
        kroki=kroki,
        gcs=gcs,
    )
    return QGenBatchRunner(graph=graph, publisher=publisher)


def _generate_plans(executor: _FakeSetExecutor) -> list[list[dict[str, Any]]]:
    """The type_plan each ROLE_GENERATE call was asked to satisfy, in order."""
    plans: list[list[dict[str, Any]]] = []
    for call in executor.calls:
        if call["agent_role"] != ROLE_GENERATE:
            continue
        payload = json.loads(call["input_payload"])
        plans.append(payload.get("type_plan") or [])
    return plans


def _counts(plans: list[list[dict[str, Any]]]) -> list[int]:
    return [sum(int(q.get("count") or 0) for q in p) for p in plans]


# -----------------------------------------------------------------------------
# Chunked dispatch
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_25_question_plan_runs_three_chunk_calls() -> None:
    """AC 'Graph-structured loop': 25 questions at the default chunk size 10
    dispatch exactly 3 generate calls (10/10/5) and publish ONE completed.v1
    carrying all 25."""
    queue = [
        _wrapper([_mcq(f"q{i}") for i in range(10)]),
        _wrapper([_mcq(f"q{10 + i}") for i in range(10)]),
        _wrapper([_mcq(f"q{20 + i}") for i in range(5)]),
    ]
    executor = _FakeSetExecutor(generate_queue=queue)
    publisher = _FakePublisher()

    await _runner(executor, publisher).handle_started(_set_event(25, _mcq_plan(25)))

    assert len(publisher.refused) == 0, publisher.refused
    assert _counts(_generate_plans(executor)) == [10, 10, 5]
    assert len(publisher.completed) == 1
    ev = publisher.completed[0]
    payload = json.loads(ev["candidate_payload_json"])
    assert len(payload["candidates"]) == 25
    assert ev["generated_count"] == 25
    assert ev["generation_summary"]["requested_total"] == 25
    assert ev["generation_summary"]["generated_per_type"] == {"mcq": 25}


@pytest.mark.asyncio
async def test_chunk_size_env_override_and_invalid_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 'Env default': QGEN_SET_MAX_PER_CALL drives the chunk size; an
    invalid value falls back to 10 loudly."""
    monkeypatch.setenv("QGEN_SET_MAX_PER_CALL", "7")
    queue = [
        _wrapper([_mcq(f"a{i}") for i in range(7)]),
        _wrapper([_mcq(f"b{i}") for i in range(7)]),
    ]
    executor = _FakeSetExecutor(generate_queue=queue)
    publisher = _FakePublisher()
    await _runner(executor, publisher).handle_started(_set_event(14, _mcq_plan(14)))
    assert _counts(_generate_plans(executor)) == [7, 7]

    monkeypatch.setenv("QGEN_SET_MAX_PER_CALL", "not-a-number")
    queue2 = [
        _wrapper([_mcq(f"c{i}") for i in range(10)]),
        _wrapper([_mcq(f"d{i}") for i in range(2)]),
    ]
    executor2 = _FakeSetExecutor(generate_queue=queue2)
    publisher2 = _FakePublisher()
    await _runner(executor2, publisher2).handle_started(_set_event(12, _mcq_plan(12)))
    assert _counts(_generate_plans(executor2)) == [10, 2]


@pytest.mark.asyncio
async def test_small_set_single_chunk_parity() -> None:
    """AC 'Small-set parity': a plan at or under the chunk size runs exactly
    one generate call and publishes the same candidates + summary as today."""
    executor = _FakeSetExecutor(generate_queue=[_wrapper([_mcq(f"q{i}") for i in range(5)])])
    publisher = _FakePublisher()
    await _runner(executor, publisher).handle_started(_set_event(5, _mcq_plan(5)))

    assert _counts(_generate_plans(executor)) == [5]
    ev = publisher.completed[0]
    payload = json.loads(ev["candidate_payload_json"])
    assert len(payload["candidates"]) == 5
    assert ev["generation_summary"]["generated_total"] == 5
    assert ev["generation_summary"]["shortfall_reason"] == ""
    assert not ev.get("quality_warning")


# -----------------------------------------------------------------------------
# Per-chunk regen + exhausted semantics
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_chunk_regen_regenerates_only_that_chunks_rejects() -> None:
    """AC 'Per-chunk regen': chunk 2's rejects regenerate against a round plan
    scoped to chunk 2; chunk 1 is untouched and nothing re-generates it."""
    queue = [
        _wrapper([_mcq(f"clean-a{i}") for i in range(10)]),
        _wrapper([_mcq(f"clean-b{i}") for i in range(8)] + [_mcq(f"{_REJECT_MARKER} dup {i}") for i in range(2)]),
        _wrapper([_mcq(f"regen-{i}") for i in range(2)]),  # chunk 2's regen round
    ]
    executor = _FakeSetExecutor(generate_queue=queue)
    publisher = _FakePublisher()

    await _runner(executor, publisher).handle_started(_set_event(20, _mcq_plan(20)))

    plans = _generate_plans(executor)
    assert _counts(plans) == [10, 10, 2], "regen round must be scoped to the chunk"
    ev = publisher.completed[0]
    payload = json.loads(ev["candidate_payload_json"])
    stems = [c["stem"] for c in payload["candidates"]]
    assert len(stems) == 20
    assert not any(_REJECT_MARKER in s for s in stems), "rejects were replaced"
    assert not ev.get("quality_warning")
    assert ev["generation_summary"]["generated_total"] == 20


@pytest.mark.asyncio
async def test_exhausted_chunk_publishes_warned_others_stay_clean() -> None:
    """Locked exhausted semantics per chunk: a candidate still rejected after
    the chunk's regen budget ships WITH quality_warning + critic_notes on
    completed.v1 (never refused.v1); clean chunks stay clean; the job-level
    warning is sticky through finalize."""
    queue = [
        _wrapper([_mcq(f"clean-a{i}") for i in range(10)]),
        _wrapper([_mcq(f"clean-b{i}") for i in range(9)] + [_mcq(f"{_REJECT_MARKER} x")]),
        _wrapper([_mcq(f"{_REJECT_MARKER} still bad")]),  # regen also rejected
    ]
    executor = _FakeSetExecutor(generate_queue=queue)
    publisher = _FakePublisher()

    await _runner(executor, publisher).handle_started(_set_event(20, _mcq_plan(20)))

    ev = publisher.completed[0]
    payload = json.loads(ev["candidate_payload_json"])
    warned = [c for c in payload["candidates"] if c.get("quality_warning")]
    assert len(payload["candidates"]) == 20
    assert len(warned) == 1
    assert warned[0].get("critic_notes")
    assert ev.get("quality_warning") is True
    assert ev["generation_summary"]["generated_total"] == 20


# -----------------------------------------------------------------------------
# Image budgets + CHO-2395 skip survive chunking
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forced_images_render_across_chunks_none_dropped_over_budget() -> None:
    """AC 'Image budgets survive chunking': image_for_stem forces ONE image per
    question (CHO-1825); the flag must survive onto every chunk plan so no
    forced image is dropped as over-budget in later chunks."""
    queue = [
        _wrapper([_mcq(f"img-a{i}", image=True) for i in range(10)]),
        _wrapper([_mcq(f"img-b{i}", image=True) for i in range(2)]),
    ]
    executor = _FakeSetExecutor(generate_queue=queue)
    publisher = _FakePublisher()
    gcs = _FakeGcs()

    await _runner(
        executor,
        publisher,
        kroki=_FakeKroki(),
        gcs=gcs,
    ).handle_started(_set_event(12, _mcq_plan(12, image_for_stem=True)))

    assert gcs.signed == 12, "every author-forced image must render, no cap drops"
    ev = publisher.completed[0]
    payload = json.loads(ev["candidate_payload_json"])
    assert all(c.get("image_url") for c in payload["candidates"])
    assert all("image_specs" not in c for c in payload["candidates"])
    assert not ev.get("quality_warning")


# -----------------------------------------------------------------------------
# Cap 200 + recursion headroom
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_200_question_plan_runs_twenty_chunks_to_one_terminal() -> None:
    """AC 'Cap raised to 200' + the loop's recursion headroom: 20 chunks of 10
    complete to ONE terminal with all 200 candidates (the default LangGraph
    recursion limit of 25 would abort this run)."""
    queue = [_wrapper([_mcq(f"q{c}-{i}") for i in range(10)]) for c in range(20)]
    executor = _FakeSetExecutor(generate_queue=queue)
    publisher = _FakePublisher()

    await _runner(executor, publisher).handle_started(_set_event(200, _mcq_plan(200)))

    assert len(publisher.refused) == 0, publisher.refused
    assert _counts(_generate_plans(executor)) == [10] * 20
    ev = publisher.completed[0]
    payload = json.loads(ev["candidate_payload_json"])
    assert len(payload["candidates"]) == 200
    assert ev["generation_summary"]["requested_total"] == 200


@pytest.mark.asyncio
async def test_201_question_plan_is_refused_with_the_count_named() -> None:
    """AC 'Cap raised to 200': one over the ceiling refuses (VALIDATION) with
    both numbers in the message, defence-in-depth behind chora-creation's own
    MaxBatchCount."""
    executor = _FakeSetExecutor(generate_queue=[])
    publisher = _FakePublisher()

    await _runner(executor, publisher).handle_started(_set_event(201, _mcq_plan(201)))

    assert len(publisher.completed) == 0
    assert len(publisher.refused) == 1
    ev = publisher.refused[0]
    assert ev["refusal_reason"] == "VALIDATION"
    msg = str(ev.get("user_facing_message") or "")
    assert "201" in msg and "200" in msg, msg
