"""Focused production-path tests for generate_set_node's grounding wiring.

The autouse resolver double in conftest.py keeps the rest of the crew suite off
the network. That convenience must not become cover for an untested behaviour:
these tests OVERRIDE the double and pin the invariants the deployed path
depends on.

  * grounded material reaches the dispatch payload verbatim;
  * the resolved corpus is persisted into graph state so later chunks and regen
    rounds reuse it instead of re-fetching;
  * a retrieval failure refuses BEFORE dispatching — no executor call at all —
    and carries the grounding-specific copy, not the infrastructure copy.
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.creation.source_material import (
    REASON_NO_TEXT,
    SourceMaterialError,
)
from chora_ai_kernel_orchestrator.orchestrators import qgen_crew

TARGET = "chora_ai_kernel_orchestrator.orchestrators.qgen_crew.source_material_text_for_state"

MATERIAL = '<SOURCE file="paper.pdf" page="1">Photosynthesis.</SOURCE>'


class _Executor:
    """Records dispatch and returns a minimal valid set payload."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        class _Resp:
            output_payload = (
                '{"candidates":[{"draft_id":"d-1","stem":"Q?","question_type":"mcq"}],"generation_summary":{}}'
            )
            output_tokens = 5
            input_tokens = 5

        return _Resp()


def _state(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": "job-g1",
        "tenant_id": "tenant-t",
        "gcid": "gcid-a",
        "prompt": "Generate a set on photosynthesis",
        "type_plan": [{"question_type": "mcq", "count": 1, "max_images": 0}],
        "source_files": [{"blob_uri": "s3://b/jobs/job-g1/source", "mime_type": "application/pdf", "role": "source"}],
        "source_blob_uri": "s3://b/jobs/job-g1/source",
        "source_mime_type": "application/pdf",
    }
    base.update(extra)
    return base


@pytest.mark.asyncio
async def test_resolved_material_reaches_the_dispatch_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _resolved(state: Any, **_kw: Any) -> tuple[str, bool]:
        return MATERIAL, True

    monkeypatch.setattr(TARGET, _resolved)
    executor = _Executor()

    out = await qgen_crew.generate_set_node(_state(), executor=executor)

    assert executor.calls, "a grounded run must still dispatch"
    sent = __import__("json").loads(executor.calls[0]["input_payload"])
    assert sent["source_material_text"] == MATERIAL
    # The corpus is persisted so the NEXT chunk/round can reuse it.
    assert out["source_material_text"] == MATERIAL
    assert out["source_material_complete"] is True


@pytest.mark.asyncio
async def test_truncation_is_persisted_not_papered_over(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _resolved(state: Any, **_kw: Any) -> tuple[str, bool]:
        return MATERIAL, False

    monkeypatch.setattr(TARGET, _resolved)
    out = await qgen_crew.generate_set_node(_state(), executor=_Executor())
    # A partial corpus must not later report itself as whole.
    assert out["source_material_complete"] is False


@pytest.mark.asyncio
async def test_retrieval_failure_refuses_before_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(state: Any, **_kw: Any) -> tuple[str, bool]:
        raise SourceMaterialError(REASON_NO_TEXT, "no extractable text")

    monkeypatch.setattr(TARGET, _boom)
    executor = _Executor()

    out = await qgen_crew.generate_set_node(_state(), executor=executor)

    # The whole point of resolving here rather than at the model boundary.
    assert executor.calls == [], "must not spend a model call on ungroundable material"
    assert out["refusal_reason"] == "VALIDATION"
    assert out["refusal_user_facing_message"] == qgen_crew._GROUNDING_UNAVAILABLE_BY_REASON[REASON_NO_TEXT]
    assert out["refusal_user_facing_message"] != qgen_crew._EXECUTOR_FAILURE_USER_MESSAGE
    assert any("no extractable text" in e for e in out["errors"])
    assert "source_material_text" not in out


@pytest.mark.asyncio
async def test_ungrounded_job_never_calls_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    # The node always asks the resolver; the RESOLVER is what short-circuits on
    # "no material declared" (pinned directly in
    # test_creation_source_material.test_ungrounded_state_is_untouched_and_never_calls_creation).
    # Here we pin that the node tolerates that answer and sends no material.
    async def _nothing(state: Any, **_kw: Any) -> tuple[str, bool]:
        return "", True

    monkeypatch.setattr(TARGET, _nothing)
    executor = _Executor()
    state = _state()
    for key in ("source_files", "source_blob_uri", "source_mime_type"):
        state.pop(key)

    out = await qgen_crew.generate_set_node(state, executor=executor)

    assert executor.calls, "the prompt-only path must still dispatch"
    assert "source_material_text" not in __import__("json").loads(executor.calls[0]["input_payload"])
    assert out.get("source_material_text", "") == ""


@pytest.mark.parametrize("reason", sorted(qgen_crew._GROUNDING_UNAVAILABLE_BY_REASON))
def test_grounding_copy_is_specific_per_reason(reason: str) -> None:
    message = qgen_crew._grounding_unavailable_message(reason)
    assert message == qgen_crew._GROUNDING_UNAVAILABLE_BY_REASON[reason]
    assert message != qgen_crew._GROUNDING_UNAVAILABLE_USER_MESSAGE


def test_unknown_grounding_reason_falls_back_to_generic_copy() -> None:
    assert (
        qgen_crew._grounding_unavailable_message("a_reason_this_module_does_not_know")
        == qgen_crew._GROUNDING_UNAVAILABLE_USER_MESSAGE
    )


class _ContractThenValidExecutor:
    """Violates the set contract once, then answers properly."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        payload = (
            "{}"
            if len(self.calls) == 1
            else '{"candidates":[{"draft_id":"d-1","stem":"Q?","question_type":"mcq"}],"generation_summary":{}}'
        )

        class _Resp:
            output_payload = payload
            output_tokens = 5
            input_tokens = 5
            model_id = "test-model"

        return _Resp()


@pytest.mark.asyncio
async def test_contract_violation_is_repaired_once_with_a_distinct_execution_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _resolved(state: Any, **_kw: Any) -> tuple[str, bool]:
        return MATERIAL, True

    monkeypatch.setattr(TARGET, _resolved)
    executor = _ContractThenValidExecutor()

    out = await qgen_crew.generate_set_node(_state(), executor=executor)

    assert len(executor.calls) == 2, "a contract violation must earn exactly one repair"
    first_id = executor.calls[0]["execution_id"]
    repair_id = executor.calls[1]["execution_id"]
    assert repair_id != first_id, "the repair must not reuse the idempotency key it replaces"
    assert "generate_set_repair" in repair_id
    loads = __import__("json").loads
    assert qgen_crew._SET_REPAIR_INSTRUCTION not in loads(executor.calls[0]["input_payload"])["prompt"]
    assert qgen_crew._SET_REPAIR_INSTRUCTION in loads(executor.calls[1]["input_payload"])["prompt"]
    assert out["candidate_set"], "the repaired answer must be the one used"
    assert "refusal_reason" not in out


@pytest.mark.asyncio
async def test_repair_that_also_violates_refuses_naming_chunk_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _resolved(state: Any, **_kw: Any) -> tuple[str, bool]:
        return MATERIAL, True

    monkeypatch.setattr(TARGET, _resolved)
    executor = _Executor()

    async def _always_invalid(**kwargs: Any) -> Any:
        executor.calls.append(kwargs)

        class _Resp:
            output_payload = "{}"
            output_tokens = 5
            input_tokens = 5
            model_id = "test-model"

        return _Resp()

    executor.execute = _always_invalid  # type: ignore[method-assign]

    out = await qgen_crew.generate_set_node(_state(), executor=executor)

    assert len(executor.calls) == 2, "the repair is bounded; it must not loop"
    assert out["refusal_reason"] == "VALIDATION"
    assert out["refusal_user_facing_message"] == qgen_crew._SET_VALIDATION_USER_MESSAGE
    assert any("chunk=0" in e for e in out["errors"])
    assert any("test-model" in e for e in out["errors"])
