"""Failure CLASSIFICATION in the qgen crew — the author must never be told the
generator produced an invalid question set when no generator output existed.

A dispatch/executor failure happens BEFORE the agent returns anything, so the
"invalid question set" copy is a lie: it blames model output for an
infrastructure fault. The two cases must not share user-facing copy.

The wire refusal vocabulary stays ``GUARDRAIL_PRE | GUARDRAIL_POST |
VALIDATION`` (fixed by chora-creation), so only the author-facing message
diverges here.
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.orchestrators import qgen_crew


class _RaisingExecutor:
    """Executor whose dispatch fails — no agent output ever exists."""

    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc or RuntimeError("dispatch returned status=FAILED")

    async def execute(self, **kwargs: Any) -> Any:
        raise self._exc


class _EmptyPayloadExecutor:
    """Executor that returns a syntactically valid but contract-violating set."""

    class _Resp:
        output_payload = "{}"
        output_tokens = 0
        input_tokens = 0

    async def execute(self, **kwargs: Any) -> Any:
        return self._Resp()


def _executor_failure_message() -> str:
    msg = getattr(qgen_crew, "_EXECUTOR_FAILURE_USER_MESSAGE", None)
    assert msg, "qgen_crew must define _EXECUTOR_FAILURE_USER_MESSAGE"
    return str(msg)


def _set_state() -> dict[str, Any]:
    return {
        "job_id": "job-fail-1",
        "tenant_id": "tenant-test",
        "prompt": "Generate a set on photosynthesis",
        "type_plan": [{"question_type": "mcq", "count": 2, "max_images": 0}],
    }


def _single_state() -> dict[str, Any]:
    return {
        "job_id": "job-fail-2",
        "tenant_id": "tenant-test",
        "gcid": "gcid-author",
        "prompt": "Generate a question on photosynthesis byproducts",
        "question_type": "mcq",
        "metadata": {"subject": "Biology", "cognitive_level": "knowledge"},
        "max_retries": 3,
        "attempt_count": 0,
    }


@pytest.mark.asyncio
async def test_generate_set_executor_failure_uses_infrastructure_copy() -> None:
    """A failed dispatch must NOT be reported as an invalid question set."""
    out = await qgen_crew.generate_set_node(_set_state(), executor=_RaisingExecutor())

    message = out.get("refusal_user_facing_message") or ""
    assert message == _executor_failure_message()
    assert message != qgen_crew._SET_VALIDATION_USER_MESSAGE
    assert "invalid question set" not in message.lower()
    # The wire vocabulary is unchanged: this still ends the job as VALIDATION.
    assert out["refusal_reason"] == "VALIDATION"
    # The real cause stays available to operators.
    assert any("dispatch returned status=FAILED" in e for e in out["errors"])


@pytest.mark.asyncio
async def test_generate_set_contract_violation_keeps_validation_copy() -> None:
    """A genuinely malformed generator answer keeps the original copy."""
    out = await qgen_crew.generate_set_node(_set_state(), executor=_EmptyPayloadExecutor())

    assert out["refusal_reason"] == "VALIDATION"
    assert out["refusal_user_facing_message"] == qgen_crew._SET_VALIDATION_USER_MESSAGE


@pytest.mark.asyncio
async def test_generate_single_executor_failure_uses_infrastructure_copy() -> None:
    """The single-candidate lane classifies dispatch failures the same way."""
    out = await qgen_crew.generate_node(_single_state(), executor=_RaisingExecutor())

    message = out.get("refusal_user_facing_message") or ""
    assert message == _executor_failure_message()
    assert message != qgen_crew._SET_VALIDATION_USER_MESSAGE
    assert "invalid question set" not in message.lower()
    assert out["refusal_reason"] == "VALIDATION"
