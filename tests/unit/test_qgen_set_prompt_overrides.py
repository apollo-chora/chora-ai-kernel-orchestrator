"""CHO-2368 P2.a — the SET/batch lane must thread prompt overrides too.

The single-candidate lane threads the resolver's override into the executor
payload (qgen_crew.py generate_node / critique_node). The set lane historically
did NOT, while the runner stamps the resolver's version/source on EVERY
decision — so an active 1.1.0 would be claimed on batch decisions that actually
ran the embedded prompt (false provenance). These tests pin the fix: both set
hops carry the PINNED contract keys when (and only when) the runner resolved an
override.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_CRITIQUE,
    ROLE_GENERATE,
    critique_set_node,
    generate_set_node,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_set_plan import (
    AgentContractViolation,
)


class _Resp:
    def __init__(self) -> None:
        self.output_payload = "{}"
        self.output_tokens = 0
        self.input_tokens = 0


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> _Resp:
        self.calls.append(kwargs)
        return _Resp()


def _set_state(**extra: Any) -> dict[str, Any]:
    return {
        "job_id": "job-1",
        "tenant_id": "tenant-test",
        "prompt": "Generate a set on photosynthesis",
        "type_plan": [{"question_type": "mcq", "count": 2, "max_images": 0}],
        **extra,
    }


async def test_generate_set_threads_prompt_overrides() -> None:
    executor = _RecordingExecutor()
    state = _set_state(
        prompt_overrides_question={"role": "revised role body"},
        prompt_version_question="1.1.0",
        prompt_source_question="platform_override",
    )
    await generate_set_node(state, executor=executor)

    assert executor.calls, "generate_set_node must dispatch the executor"
    call = executor.calls[0]
    assert call["agent_role"] == ROLE_GENERATE
    gen_input = json.loads(call["input_payload"])
    assert json.loads(gen_input["prompt_overrides_json"]) == {"role": "revised role body"}
    assert gen_input["resolved_prompt_version"] == "1.1.0"
    assert gen_input["prompt_source"] == "platform_override"


async def test_generate_set_threads_nothing_without_override() -> None:
    executor = _RecordingExecutor()
    await generate_set_node(_set_state(), executor=executor)
    gen_input = json.loads(executor.calls[0]["input_payload"])
    assert "prompt_overrides_json" not in gen_input
    assert "resolved_prompt_version" not in gen_input


async def test_critique_set_threads_prompt_overrides() -> None:
    # The recording executor answers "{}", which the strict echo-contract
    # parse refuses AFTER the call is recorded; the pinned contract here is
    # the override keys on the ONE batched call's payload (CHO-2397).
    executor = _RecordingExecutor()
    state = _set_state(
        candidate_set=[{"stem": "What gas do plants absorb?", "question_type": "mcq"}],
        prompt_overrides_critic={"examples": "revised examples body"},
        prompt_version_critic="1.1.0",
        prompt_source_critic="platform_override",
    )
    with pytest.raises(AgentContractViolation):
        await critique_set_node(state, executor=executor)

    call = executor.calls[0]
    assert call["agent_role"] == ROLE_CRITIQUE
    crit_input = json.loads(call["input_payload"])
    assert json.loads(crit_input["prompt_overrides_json"]) == {"examples": "revised examples body"}
    assert crit_input["resolved_prompt_version"] == "1.1.0"
    assert crit_input["prompt_source"] == "platform_override"


async def test_critique_set_threads_nothing_without_override() -> None:
    executor = _RecordingExecutor()
    state = _set_state(candidate_set=[{"stem": "s", "question_type": "mcq"}])
    with pytest.raises(AgentContractViolation):
        await critique_set_node(state, executor=executor)
    crit_input = json.loads(executor.calls[0]["input_payload"])
    assert "prompt_overrides_json" not in crit_input
