"""QGenBatchRunner unit tests (EPIC-1a, ADR-254 D2).

Every batch rides the chunked SET lane (the per-candidate N-loop is retired):
the runner drives the qgen_crew graph ONCE in set mode and publishes ONE
chora.creation.ai_assist.completed.v1 whose candidate_payload_json is the
object wrapper ``{"candidates": [...]}`` with the typed generated_count. Any
guardrail/validation refusal fails the WHOLE batch (no partial success),
publishing refused.v1.
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
    AiAssistStartedPayload,
    QGenBatchRunner,
)
from tests.integration.test_qgen_crew_set import _FakeSetExecutor, _mcq, _wrapper
from tests.unit.test_qgen_crew_runner import (
    _critic_accept,
    _FakeExecutor,
    _FakeGuardrail,
    _FakePublisher,
    _good_mcq_payload,
    _started_event,
)


def _set_executor(n: int) -> _FakeSetExecutor:
    """ADR-254 D2: every batch rides the SET lane; the generator answers ONE
    wrapper of n distinct candidates and the critic accepts them."""
    return _FakeSetExecutor(generate_queue=[_wrapper([_mcq(f"Q{i}") for i in range(n)])])


def _candidates(ev: dict[str, Any]) -> list[dict[str, Any]]:
    payload = json.loads(ev["candidate_payload_json"])
    assert isinstance(payload, dict), "the set lane publishes the object wrapper"
    return list(payload["candidates"])


def _batch_event(**overrides: Any) -> dict[str, Any]:
    base = _started_event(
        assist_id="job-batch",
        content_type="mcq",
        question_type="mcq",
        prompt="Generate MCQs from the source material.",
        job_kind="batch",
        requested_count=3,
        grounding_mode="strict",
        source_blob_uri="gs://chora-batch-uploads/t/job-batch/material.pdf",
        source_mime_type="application/pdf",
        target_growth_edges=["fractions", "ratios"],
    )
    base.update(overrides)
    return base


def _batch_runner(executor: Any, guardrail: Any, publisher: Any) -> QGenBatchRunner:
    graph = build_qgen_crew_graph(executor=executor, guardrail=guardrail, checkpointer=MemorySaver())
    return QGenBatchRunner(graph=graph, publisher=publisher)


class _RecordingRunner:
    """A handle_started double that records the events it receives — lets the
    router tests assert which runner a given event was dispatched to."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def handle_started(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _v2_event(*, intent: str = "new_question", input_kind: str = "prompt", **overrides: Any) -> dict[str, Any]:
    """An ADR-195 WS7 (D7) .v2 ai_assist.started event — carries the explicit
    compose model {operation, intent, input_kind} and NO legacy job_kind (the v2
    encoder drops it). Mirrors what proto_wire.decode_ai_assist_started surfaces
    for a v2 wire message."""
    base = _started_event(
        assist_id="job-v2",
        content_type="mcq",
        question_type="mcq",
        prompt="Generate MCQs.",
        operation="compose",
        intent=intent,
        input_kind=input_kind,
    )
    base.update(overrides)
    base.pop("job_kind", None)  # v2 drops the legacy discriminant
    return base


def _make_router(single: Any, batch: Any, image_regen: Any) -> Any:
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
        QGenRunnerRouter,
    )

    return QGenRunnerRouter(single=single, batch=batch, image_regen=image_regen)


# -----------------------------------------------------------------------------
# AiAssistStartedPayload — batch field parsing
# -----------------------------------------------------------------------------


def test_payload_from_event_batch_fields() -> None:
    p = AiAssistStartedPayload.from_event(_batch_event())
    assert p.job_kind == "batch"
    assert p.requested_count == 3
    assert p.grounding_mode == "strict"
    assert p.source_blob_uri.endswith("material.pdf")
    assert p.source_mime_type == "application/pdf"
    assert tuple(p.target_growth_edges) == ("fractions", "ratios")


def test_payload_from_event_single_defaults() -> None:
    # A non-batch (single) event leaves the batch fields at safe defaults so the
    # live single-candidate path is byte-for-byte unchanged.
    p = AiAssistStartedPayload.from_event(_started_event())
    assert p.job_kind == ""
    assert p.requested_count == 1
    assert p.grounding_mode == ""
    assert p.source_blob_uri == ""
    assert p.source_mime_type == ""
    assert tuple(p.target_growth_edges) == ()


def test_payload_from_event_v2_compose_model() -> None:
    # ADR-195 WS7 (D7) — a .v2 event surfaces the explicit compose model on the
    # typed payload so the decoded view faithfully mirrors the v2 wire.
    p = AiAssistStartedPayload.from_event(_v2_event(intent="image_regen", input_kind="prompt"))
    assert p.operation == "compose"
    assert p.intent == "image_regen"
    assert p.input_kind == "prompt"


def test_payload_from_event_v1_compose_model_defaults() -> None:
    # A v1 event (no operation/intent/input_kind) defaults them to "" so a
    # consumer always reads the attribute without a KeyError.
    p = AiAssistStartedPayload.from_event(_started_event())
    assert p.operation == ""
    assert p.intent == ""
    assert p.input_kind == ""


# -----------------------------------------------------------------------------
# QGenBatchRunner
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_runner_publishes_one_completed_wrapper() -> None:
    executor = _set_executor(3)
    publisher = _FakePublisher()
    runner = _batch_runner(executor, _FakeGuardrail(), publisher)

    await runner.handle_started(_batch_event(requested_count=3))

    assert len(publisher.refused) == 0
    assert len(publisher.completed) == 1, "batch must publish exactly ONE completed.v1"
    ev = publisher.completed[0]
    assert ev["assist_id"] == "job-batch"
    cands = _candidates(ev)
    assert len(cands) == 3
    for c in cands:
        assert c.get("stem"), "each batch candidate must carry a stem"
    assert ev["generated_count"] == 3
    # ONE generate dispatch for the whole (single-chunk) set, not one per item.
    assert sum(1 for c in executor.calls if c["agent_role"] == ROLE_GENERATE) == 1
    sent = json.loads(executor.calls[0]["input_payload"])
    assert sent["set_mode"] is True
    assert sent["type_plan"] == [{"question_type": "mcq", "count": 3, "max_images": 0}]


@pytest.mark.asyncio
async def test_batch_runner_clamps_count_to_at_least_one() -> None:
    executor = _set_executor(1)
    publisher = _FakePublisher()
    runner = _batch_runner(executor, _FakeGuardrail(), publisher)

    await runner.handle_started(_batch_event(requested_count=0))  # -> clamp to 1

    assert len(publisher.completed) == 1
    assert len(_candidates(publisher.completed[0])) == 1


@pytest.mark.asyncio
async def test_batch_runner_refusal_fails_whole_batch() -> None:
    # The pre-screen blocks the prompt → the first candidate refuses → the whole
    # batch is refused (no partial success), publishing refused.v1 (not completed).
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload(), _good_mcq_payload(), _good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept(), _critic_accept(), _critic_accept()],
        }
    )
    guardrail = _FakeGuardrail(block_substrings=["FORBIDDEN"])
    publisher = _FakePublisher()
    runner = _batch_runner(executor, guardrail, publisher)

    await runner.handle_started(_batch_event(requested_count=3, prompt="contains FORBIDDEN text"))

    assert len(publisher.completed) == 0
    assert len(publisher.refused) == 1
    assert publisher.refused[0]["refusal_reason"] in ("GUARDRAIL_PRE", "GUARDRAIL_POST")


@pytest.mark.asyncio
async def test_batch_runner_rejects_missing_assist_id() -> None:
    runner = _batch_runner(_FakeExecutor(responses={}), _FakeGuardrail(), _FakePublisher())
    with pytest.raises(ValueError):
        await runner.handle_started(_batch_event(assist_id=""))


# -----------------------------------------------------------------------------
# AgentDecisionLog emission - CHO-2364.
#
# The batch lane previously emitted NO AgentDecisionLogged at all (its
# handle_started published terminals directly, bypassing the single runner's
# _emit_agent_decision_log seam, and the wiring passed it no emitter) - live
# evidence: zero qgen decision rows since 2026-07-04 while daily-dose batch
# generation ran. The batch runner now emits per-item qgen_question +
# qgen_critic decisions through the SAME emitter + condition builders as the
# single runner. Per-item identity: assist_id = "{job}:{i}" (the same
# per-candidate id the checkpointer uses) so the outbox idempotency key
# agent_decision.{tenant}.{assist_id}.{agid} never collides across items,
# while crew_id stays the batch job id.
# -----------------------------------------------------------------------------


from chora_ai_kernel_orchestrator.domain.prompt_registry import (  # noqa: E402
    EMBEDDED_PROMPT_VERSIONS,
)
from tests.unit.test_qgen_crew_runner import (  # noqa: E402
    _FakeAgentDecisionLogEmitter,
)


@pytest.mark.asyncio
async def test_batch_runner_emits_one_decision_pair_per_set_run() -> None:
    executor = _set_executor(3)
    publisher = _FakePublisher()
    adl = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenBatchRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl)

    await runner.handle_started(_batch_event(requested_count=3))

    # ADR-254 D2: the batch is ONE set run, so ONE qgen_question + ONE
    # qgen_critic decision (the set lane's CHO-2364 shape); the job id is both
    # the decision identity and the crew id.
    assert len(adl.emitted) == 2
    assert {e["assist_id"] for e in adl.emitted} == {"job-batch"}
    agids = sorted(e["agid"] for e in adl.emitted)
    assert agids == ["qgen_critic", "qgen_question"]
    for ev in adl.emitted:
        assert ev["crew_id"] == "job-batch"
        assert ev["crew_name"] == "mcq_ai_assist"
        assert ev["decision"] == "accepted"
        assert ev["question_type"] == "mcq"
        conditions = ev["prompt_conditions"]
        # Embedded prompt provenance rides the batch lane too (CHO-2364).
        assert conditions["prompt_version"] == EMBEDDED_PROMPT_VERSIONS[ev["agid"]]
        assert conditions["prompt_source"] == "embedded"
    # Terminal publish unaffected (additive).
    assert len(publisher.completed) == 1


@pytest.mark.asyncio
async def test_batch_runner_refusal_emits_refused_decisions_for_the_set() -> None:
    executor = _set_executor(3)
    publisher = _FakePublisher()
    adl = _FakeAgentDecisionLogEmitter()
    graph = build_qgen_crew_graph(
        executor=executor,
        guardrail=_FakeGuardrail(block_substrings=["FORBIDDEN"]),
        checkpointer=MemorySaver(),
    )
    runner = QGenBatchRunner(graph=graph, publisher=publisher, agent_decision_emitter=adl)

    await runner.handle_started(_batch_event(requested_count=3, prompt="contains FORBIDDEN text"))

    # The pre-screen refuses the whole set (no-partial-success): the decision
    # pair lands as decision="refused" and nothing was generated.
    assert len(publisher.refused) == 1
    assert len(adl.emitted) == 2
    for ev in adl.emitted:
        assert ev["assist_id"] == "job-batch"
        assert ev["decision"] == "refused"
    assert executor.calls == []


@pytest.mark.asyncio
async def test_batch_runner_no_emitter_emits_nothing() -> None:
    # Constructor default (None) keeps the pre-CHO-2364 behaviour for callers
    # that do not wire an emitter (unit fixtures) - publish still fires.
    executor = _set_executor(1)
    publisher = _FakePublisher()
    runner = _batch_runner(executor, _FakeGuardrail(), publisher)

    await runner.handle_started(_batch_event(requested_count=1))

    assert len(publisher.completed) == 1


@pytest.mark.asyncio
async def test_batch_runner_decision_emit_failure_does_not_break_publish() -> None:
    class _BoomEmitter:
        async def emit(self, **kwargs: Any) -> str:
            raise RuntimeError("decision emit boom")

    executor = _set_executor(1)
    publisher = _FakePublisher()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenBatchRunner(graph=graph, publisher=publisher, agent_decision_emitter=_BoomEmitter())

    await runner.handle_started(_batch_event(requested_count=1))

    # Best-effort per [[feedback-d6-resilience-first-class]] - the terminal
    # publish must never be suppressed by a decision-emit failure.
    assert len(publisher.completed) == 1


# -----------------------------------------------------------------------------
# Grounding passthrough — generate_node forwards the blob to the qgen_question
# agent (the agent inlines it as Gemini multimodal; renderer is agent-side).
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_runner_forwards_grounding_to_generator() -> None:
    executor = _set_executor(1)
    publisher = _FakePublisher()
    runner = _batch_runner(executor, _FakeGuardrail(), publisher)

    await runner.handle_started(
        _batch_event(
            requested_count=1,
            source_blob_uri="gs://b/material.pdf",
            source_mime_type="application/pdf",
            grounding_mode="strict",
            target_growth_edges=["fractions"],
        )
    )

    gen_calls = [c for c in executor.calls if c["agent_role"] == ROLE_GENERATE]
    assert gen_calls, "generator must be invoked"
    sent = json.loads(gen_calls[0]["input_payload"])
    assert sent["source_blob_uri"] == "gs://b/material.pdf"
    assert sent["source_mime_type"] == "application/pdf"
    assert sent["grounding_mode"] == "strict"
    assert sent["target_growth_edges"] == ["fractions"]


@pytest.mark.asyncio
async def test_single_path_generate_payload_has_no_grounding_keys() -> None:
    # The live single-candidate generate payload must be byte-for-byte unchanged
    # — no grounding keys leak in when there is no uploaded material.
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
        QGenCrewRunner,
    )

    executor = _FakeExecutor(responses={ROLE_GENERATE: [_good_mcq_payload()], ROLE_CRITIQUE: [_critic_accept()]})
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=_FakePublisher())

    await runner.handle_started(_started_event(content_type="mcq", question_type="mcq", prompt="plain"))

    gen_calls = [c for c in executor.calls if c["agent_role"] == ROLE_GENERATE]
    sent = json.loads(gen_calls[0]["input_payload"])
    assert "source_blob_uri" not in sent
    assert "grounding_mode" not in sent
    assert "target_growth_edges" not in sent


# -----------------------------------------------------------------------------
# QGenRunnerRouter — job_kind dispatch (subscriber stays unchanged)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_routes_batch_vs_single() -> None:
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
        QGenRunnerRouter,
    )

    class _Recording:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        async def handle_started(self, event: dict[str, Any]) -> None:
            self.events.append(event)

    single = _Recording()
    batch = _Recording()
    router = QGenRunnerRouter(single=single, batch=batch)

    await router.handle_started(_started_event())  # no job_kind → single
    await router.handle_started(_batch_event())  # job_kind=batch → batch
    await router.handle_started(_started_event(job_kind="single"))  # explicit single

    assert len(single.events) == 2
    assert len(batch.events) == 1
    assert batch.events[0]["job_kind"] == "batch"


# -----------------------------------------------------------------------------
# QGenRunnerRouter — ADR-195 WS7 (D7) v2 routing on intent/input_kind.
#
# The .v2 ai_assist.started event DROPS job_kind and carries the explicit compose
# model {operation, intent, input_kind}. The router must select the SAME runner
# the v1 job_kind selected, derived from the orthogonal compose fields:
#   intent=image_regen                       → image_regen runner
#   input_kind=source_files (RAG batch)      → batch runner
#   requested_count>1 / non-empty type_plan  → batch runner (prompt-batch)
#   else (new_question + prompt, count<=1)   → single runner
# v1 events (job_kind present) keep routing exactly as before (back-compat while
# both wires are published during the cutover).
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_v2_source_files_routes_to_batch() -> None:
    single, batch, image_regen = _RecordingRunner(), _RecordingRunner(), _RecordingRunner()
    router = _make_router(single, batch, image_regen)

    # RAG batch: new_question seeded by source files (the producer set job_kind=
    # batch in v1; v2 carries input_kind=source_files instead).
    await router.handle_started(_v2_event(intent="new_question", input_kind="source_files"))

    assert len(batch.events) == 1
    assert len(single.events) == 0
    assert len(image_regen.events) == 0


@pytest.mark.asyncio
async def test_router_v2_count_gt1_routes_to_batch() -> None:
    single, batch, image_regen = _RecordingRunner(), _RecordingRunner(), _RecordingRunner()
    router = _make_router(single, batch, image_regen)

    # Prompt-batch: a pure prompt with count>1 (the count=5 fix). requested_count
    # is orthogonal to the discriminant and STAYS on the v2 wire.
    await router.handle_started(_v2_event(input_kind="prompt", requested_count=5))

    assert len(batch.events) == 1
    assert len(single.events) == 0


@pytest.mark.asyncio
async def test_router_v2_type_plan_routes_to_batch() -> None:
    single, batch, image_regen = _RecordingRunner(), _RecordingRunner(), _RecordingRunner()
    router = _make_router(single, batch, image_regen)

    # A non-empty type_plan is a set generation → batch, even if requested_count
    # were not surfaced.
    await router.handle_started(_v2_event(input_kind="prompt", type_plan=[{"question_type": "mcq", "count": 3}]))

    assert len(batch.events) == 1
    assert len(single.events) == 0


@pytest.mark.asyncio
async def test_router_v2_prompt_single_routes_to_single() -> None:
    single, batch, image_regen = _RecordingRunner(), _RecordingRunner(), _RecordingRunner()
    router = _make_router(single, batch, image_regen)

    # The genuine single case: new_question, pure prompt, count<=1, no type_plan.
    await router.handle_started(_v2_event(intent="new_question", input_kind="prompt"))

    assert len(single.events) == 1
    assert len(batch.events) == 0
    assert len(image_regen.events) == 0


@pytest.mark.asyncio
async def test_router_v2_image_regen_routes_to_image_regen() -> None:
    single, batch, image_regen = _RecordingRunner(), _RecordingRunner(), _RecordingRunner()
    router = _make_router(single, batch, image_regen)

    await router.handle_started(_v2_event(intent="image_regen", input_kind="prompt"))

    assert len(image_regen.events) == 1
    assert len(single.events) == 0
    assert len(batch.events) == 0


@pytest.mark.asyncio
async def test_router_v1_back_compat_preserved() -> None:
    # v1 events (job_kind present, no intent/input_kind) keep routing as before —
    # both wires are published during the cutover, so the v1 path must be intact.
    single, batch, image_regen = _RecordingRunner(), _RecordingRunner(), _RecordingRunner()
    router = _make_router(single, batch, image_regen)

    await router.handle_started(_started_event())  # no job_kind → single
    await router.handle_started(_batch_event())  # job_kind=batch → batch
    await router.handle_started(
        _started_event(job_kind="image_regen", regen={"draft_id": "d1", "placement": "stem", "prompt": "p"})
    )  # job_kind=image_regen → image_regen

    assert len(single.events) == 1
    assert len(batch.events) == 1
    assert len(image_regen.events) == 1


@pytest.mark.asyncio
async def test_batch_runner_rejects_unsupported_question_type() -> None:
    runner = _batch_runner(_FakeExecutor(responses={}), _FakeGuardrail(), _FakePublisher())
    with pytest.raises(ValueError):
        await runner.handle_started(_batch_event(question_type="flashcard", content_type=""))


def test_candidate_to_obj_variants() -> None:
    from chora_ai_kernel_orchestrator.domain.qgen_crew import CandidatePayload
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
        _candidate_to_obj,
    )

    cp = CandidatePayload(stem="Q", question_type="mcq", payload_json='{"stem":"Q","question_type":"mcq"}')
    assert _candidate_to_obj(cp)["stem"] == "Q"
    # Unparseable payload_json → minimal fallback object.
    bad = CandidatePayload(stem="S", question_type="oe", payload_json="not json{")
    assert _candidate_to_obj(bad) == {"stem": "S", "question_type": "oe"}
    # A bare dict passes through; None / other → minimal object.
    assert _candidate_to_obj({"a": 1}) == {"a": 1}
    assert _candidate_to_obj(None) == {"stem": "", "question_type": ""}
