"""QGenBatchRunner multi-file + rubric grounding tests (Lane 1c W2, CHO-1703).

Covers the started.v1 → graph-state → generate_node wiring for the new
``source_files`` (f20) shape:

* ``AiAssistStartedPayload.from_event`` parses ``source_files`` defensively
  (absent ⇒ empty tuple — 1a/single events unchanged).
* The batch runner threads the EFFECTIVE file list into per-candidate graph
  state; the scalar f17/18 mirror keys keep feeding the deployed agent's
  single-FileData groundingplugin (back-compat) and are DERIVED from the
  first role="source" file when only f20 is stamped.
* ``generate_node`` forwards ``source_files`` to the qgen_question agent and
  appends the grounding+citations prompt block for grounded batch jobs (the
  1a single-file grounded batch gains the citations demand too — D9 applies
  to every grounded batch job).
* Ungrounded batch + the live single-candidate path stay byte-for-byte
  unchanged (no new keys, no prompt suffix).
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

_SOURCE_1 = {"blob_uri": "gs://b/exam.pdf", "mime_type": "application/pdf", "role": "source"}
_SOURCE_2 = {"blob_uri": "gs://b/fig.png", "mime_type": "image/png", "role": "source"}
_RUBRIC = {"blob_uri": "gs://b/marks.pdf", "mime_type": "application/pdf", "role": "rubric"}


def _multifile_event(**overrides: Any) -> dict[str, Any]:
    base = _started_event(
        assist_id="job-1c",
        content_type="mcq",
        question_type="mcq",
        prompt="Create a test set like the uploaded paper.",
        job_kind="batch",
        requested_count=1,
        grounding_mode="strict",
        # f17/18 mirror source_files[0] per the W0 contract.
        source_blob_uri=_SOURCE_1["blob_uri"],
        source_mime_type=_SOURCE_1["mime_type"],
        source_files=[_SOURCE_1, _SOURCE_2, _RUBRIC],
    )
    base.update(overrides)
    return base


def _runner(executor: Any, publisher: Any) -> QGenBatchRunner:
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    return QGenBatchRunner(graph=graph, publisher=publisher)


def _one_shot_executor() -> _FakeExecutor:
    """The SINGLE lane's fake (one candidate, one verdict)."""
    return _FakeExecutor(responses={ROLE_GENERATE: [_good_mcq_payload()], ROLE_CRITIQUE: [_critic_accept()]})


def _set_executor(candidates: list[dict[str, Any]] | None = None) -> _FakeSetExecutor:
    """The BATCH lane's fake (ADR-254 D2: every batch rides the set lane): ONE
    wrapper answers the generate dispatch, the critic accepts."""
    return _FakeSetExecutor(generate_queue=[_wrapper(candidates or [_mcq("Q0")])])


def _gen_input(executor: Any) -> dict[str, Any]:
    gen_calls = [c for c in executor.calls if c["agent_role"] == ROLE_GENERATE]
    assert gen_calls, "generator must be invoked"
    return json.loads(gen_calls[0]["input_payload"])


# -----------------------------------------------------------------------------
# Payload parsing
# -----------------------------------------------------------------------------


def test_payload_from_event_parses_source_files() -> None:
    p = AiAssistStartedPayload.from_event(_multifile_event())
    assert list(p.source_files) == [_SOURCE_1, _SOURCE_2, _RUBRIC]


def test_payload_from_event_source_files_default_empty() -> None:
    p = AiAssistStartedPayload.from_event(_started_event())
    assert p.source_files == ()


def test_payload_from_event_source_files_garbage_tolerant() -> None:
    p = AiAssistStartedPayload.from_event(
        _multifile_event(source_files=["junk", {"mime_type": "application/pdf"}, None])
    )
    assert p.source_files == ()


# -----------------------------------------------------------------------------
# Batch runner — state threading + generate_node forwarding
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multifile_batch_forwards_source_files_to_generator() -> None:
    executor = _set_executor()
    runner = _runner(executor, _FakePublisher())

    await runner.handle_started(_multifile_event())

    sent = _gen_input(executor)
    assert sent["source_files"] == [_SOURCE_1, _SOURCE_2, _RUBRIC]
    # Scalar mirror keys keep feeding the deployed agent's single-FileData
    # grounding plugin.
    assert sent["source_blob_uri"] == _SOURCE_1["blob_uri"]
    assert sent["source_mime_type"] == _SOURCE_1["mime_type"]
    assert sent["grounding_mode"] == "strict"


@pytest.mark.asyncio
async def test_multifile_batch_derives_scalar_mirror_when_f17_absent() -> None:
    # A publisher that stamps ONLY f20 (post-rollout) — the runner derives the
    # f17/18 mirror from the first role="source" file so the deployed agent
    # still grounds by-reference.
    executor = _set_executor()
    runner = _runner(executor, _FakePublisher())

    await runner.handle_started(_multifile_event(source_blob_uri="", source_mime_type=""))

    sent = _gen_input(executor)
    assert sent["source_blob_uri"] == _SOURCE_1["blob_uri"]
    assert sent["source_mime_type"] == _SOURCE_1["mime_type"]


@pytest.mark.asyncio
async def test_multifile_batch_appends_citation_block_to_prompt() -> None:
    executor = _set_executor()
    runner = _runner(executor, _FakePublisher())

    await runner.handle_started(_multifile_event())

    sent = _gen_input(executor)
    prompt = sent["prompt"]
    assert prompt.startswith("Create a test set like the uploaded paper.")
    assert '"citations"' in prompt
    assert '"source_file"' in prompt
    assert '"excerpt"' in prompt
    # Rubric mark-scheme alignment instructions ride the generation prompt.
    assert "gs://b/marks.pdf" in prompt
    assert "[5 marks]" in prompt


@pytest.mark.asyncio
async def test_legacy_1a_single_file_batch_gains_citations_block() -> None:
    # An EPIC-1a event (f17/18 only, no f20) is STILL a grounded batch job —
    # D9 citations apply. The synthesized single-entry source_files list is
    # forwarded + the prompt carries the citation demand.
    executor = _set_executor()
    runner = _runner(executor, _FakePublisher())

    await runner.handle_started(_multifile_event(source_files=[], grounding_mode="starting_point"))

    sent = _gen_input(executor)
    assert sent["source_files"] == [_SOURCE_1]
    assert sent["source_blob_uri"] == _SOURCE_1["blob_uri"]
    assert '"citations"' in sent["prompt"]
    # No rubric file on a 1a event → no mark-scheme instructions.
    assert "[5 marks]" not in sent["prompt"]


@pytest.mark.asyncio
async def test_ungrounded_batch_prompt_and_keys_unchanged() -> None:
    executor = _set_executor()
    runner = _runner(executor, _FakePublisher())

    await runner.handle_started(
        _multifile_event(
            source_files=[],
            source_blob_uri="",
            source_mime_type="",
            grounding_mode="",
            prompt="Five MCQs about photosynthesis.",
        )
    )

    sent = _gen_input(executor)
    assert sent["prompt"] == "Five MCQs about photosynthesis."
    assert "source_files" not in sent
    assert "source_blob_uri" not in sent


# -----------------------------------------------------------------------------
# Per-candidate draft_id stamping + citation normalisation (Slice 3)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_candidates_carry_deterministic_draft_ids() -> None:
    from chora_ai_kernel_orchestrator.orchestrators.qgen_grounding import (
        deterministic_draft_id,
    )

    executor = _set_executor([_mcq("Q0"), _mcq("Q1")])
    publisher = _FakePublisher()
    runner = _runner(executor, publisher)

    await runner.handle_started(_multifile_event(requested_count=2))

    arr = json.loads(publisher.completed[0]["candidate_payload_json"])["candidates"]
    assert [c["draft_id"] for c in arr] == [
        deterministic_draft_id("job-1c", 0),
        deterministic_draft_id("job-1c", 1),
    ]


@pytest.mark.asyncio
async def test_batch_grounded_citations_normalised_on_publish() -> None:
    cand = _mcq("cited")
    cand["citations"] = [
        {"source_file": "exam.pdf", "page": "2", "excerpt": "y" * 3000},
        "garbage",
    ]
    executor = _set_executor([cand])
    publisher = _FakePublisher()
    runner = _runner(executor, publisher)

    await runner.handle_started(_multifile_event(requested_count=1))

    arr = json.loads(publisher.completed[0]["candidate_payload_json"])["candidates"]
    cits = arr[0]["citations"]
    assert cits == [
        {
            "source_file": _SOURCE_1["blob_uri"],  # bare filename → blob_uri
            "page": 2,
            "excerpt": "y" * 1024,
        }
    ]


@pytest.mark.asyncio
async def test_batch_ungrounded_candidates_have_no_citations_key() -> None:
    cand = _mcq("ungrounded")
    cand["citations"] = [{"source_file": "hallucinated.pdf", "excerpt": "x"}]
    executor = _set_executor([cand])
    publisher = _FakePublisher()
    runner = _runner(executor, publisher)

    await runner.handle_started(
        _multifile_event(
            requested_count=1,
            source_files=[],
            source_blob_uri="",
            source_mime_type="",
            grounding_mode="",
        )
    )

    arr = json.loads(publisher.completed[0]["candidate_payload_json"])["candidates"]
    assert "citations" not in arr[0]
    assert "draft_id" in arr[0]  # draft ids stamp on every batch candidate


@pytest.mark.asyncio
async def test_single_path_stays_unchanged_with_multifile_decoder_live() -> None:
    # The live single-candidate runner NEVER threads source_files even if a
    # (mis-routed) single event carried them — its state builder is untouched.
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
        QGenCrewRunner,
    )

    executor = _one_shot_executor()
    graph = build_qgen_crew_graph(executor=executor, guardrail=_FakeGuardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=_FakePublisher())

    await runner.handle_started(
        _started_event(
            content_type="mcq",
            question_type="mcq",
            prompt="plain",
            source_files=[_SOURCE_1],
        )
    )

    sent = _gen_input(executor)
    assert sent["prompt"] == "plain"
    assert "source_files" not in sent
    assert "source_blob_uri" not in sent
