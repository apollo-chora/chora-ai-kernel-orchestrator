"""C3 — generate_node + critique_node stamp REAL per-hop tokens + wall-clock.

RED→GREEN per [[feedback-strict-tdd]]. Closes the W6 eval C3/C4-marginal gap.

Before this change the nodes hardcoded ``output_tokens=0`` and called
``_now_iso()`` twice (started_at == completed_at == zero duration). The eval's
token-compounding + duration criteria therefore read 0.

After: each ``generate`` / ``critique`` trace row carries
  - ``input_tokens``  = the executor response's input_tokens (fallback to
    tokens_consumed_total for the M11 baseline shape)
  - ``output_tokens`` = the executor response's output_tokens (REAL, not 0)
  - ``started_at`` <= ``completed_at`` (real wall-clock captured around the
    ``executor.execute(...)`` call)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    critique_node,
    generate_node,
)


@dataclass
class _TokenSplitExecutor:
    """Duck-typed _ExecutorLike returning a configurable per-hop token split.

    ``delay_s`` sleeps inside execute so the captured started_at/completed_at
    span a measurable wall-clock interval (asserts strict monotonicity without
    flaking on a same-microsecond stamp)."""

    output_payload: str
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_consumed_total: int = 0
    delay_s: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def execute(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        agid: str,
        agent_role: str,
        input_payload: str,
        workflow_id: str = "",
        prompt_template_id: str = "",
        context_window: list[dict[str, str]] | None = None,
        available_tools: list[str] | None = None,
    ) -> AgentExecutorResponse:
        self.calls.append({"agent_role": agent_role})
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return AgentExecutorResponse(
            execution_id=execution_id,
            output_payload=self.output_payload,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            tokens_consumed_total=self.tokens_consumed_total,
            final_state="EXECUTION_FINAL_STATE_SUCCESS",
        )


def _good_mcq_payload() -> str:
    return json.dumps(
        {
            "stem": "What gas do plants release in photosynthesis?",
            "question_type": "mcq",
            "mcq_payload": {"options": [], "scoring_mode": "single_correct"},
        }
    )


def _critic_accept() -> str:
    return json.dumps({"accepted": True, "critique_notes": "", "suggested_revisions": []})


def _starting_state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": "job-c3",
        "tenant_id": "tenant-c3",
        "gcid": "gcid-c3",
        "prompt": "Generate a question on photosynthesis",
        "question_type": "mcq",
        "metadata": {},
        "max_retries": 3,
        "attempt_count": 0,
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------------
# generate_node — real tokens + duration
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_node_stamps_real_input_and_output_tokens() -> None:
    """The generate trace row carries the executor's REAL split (output_tokens
    is no longer hardcoded 0)."""
    executor = _TokenSplitExecutor(
        output_payload=_good_mcq_payload(),
        input_tokens=640,
        output_tokens=128,
        tokens_consumed_total=768,
    )
    delta = await generate_node(_starting_state(), executor=executor)
    row = delta["pipeline_trace"][-1]
    assert row["name"] == "generate"
    assert row["input_tokens"] == 640
    assert row["output_tokens"] == 128


@pytest.mark.asyncio
async def test_generate_node_input_tokens_falls_back_to_total() -> None:
    """M11-baseline back-compat: an executor that sets ONLY
    tokens_consumed_total (no split) still yields a non-zero input_tokens via
    the total fallback — keeps the live MCQ loop's existing trace shape green."""
    executor = _TokenSplitExecutor(
        output_payload=_good_mcq_payload(),
        input_tokens=0,
        output_tokens=0,
        tokens_consumed_total=42,
    )
    delta = await generate_node(_starting_state(), executor=executor)
    row = delta["pipeline_trace"][-1]
    assert row["input_tokens"] == 42  # fallback to total
    assert row["output_tokens"] == 0


@pytest.mark.asyncio
async def test_generate_node_stamps_monotonic_wall_clock() -> None:
    """started_at <= completed_at with a measurable interval (real wall-clock
    captured around executor.execute, not _now_iso() twice)."""
    import datetime as _dt

    executor = _TokenSplitExecutor(
        output_payload=_good_mcq_payload(),
        input_tokens=10,
        output_tokens=5,
        tokens_consumed_total=15,
        delay_s=0.02,
    )
    delta = await generate_node(_starting_state(), executor=executor)
    row = delta["pipeline_trace"][-1]
    started = _dt.datetime.fromisoformat(row["started_at"])
    completed = _dt.datetime.fromisoformat(row["completed_at"])
    assert started <= completed
    assert (completed - started).total_seconds() >= 0.01


# -----------------------------------------------------------------------------
# critique_node — real tokens + duration
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critique_node_stamps_real_input_and_output_tokens() -> None:
    """The critique trace row carries the executor's REAL split."""
    from chora_ai_kernel_orchestrator.domain.qgen_crew import CandidatePayload

    executor = _TokenSplitExecutor(
        output_payload=_critic_accept(),
        input_tokens=900,
        output_tokens=33,
        tokens_consumed_total=933,
    )
    state = _starting_state(
        attempt_count=1,
        current_candidate=CandidatePayload(stem="x", question_type="mcq", payload_json=_good_mcq_payload()),
    )
    delta = await critique_node(state, executor=executor)
    row = delta["pipeline_trace"][-1]
    assert row["name"] == "critique"
    assert row["input_tokens"] == 900
    assert row["output_tokens"] == 33


@pytest.mark.asyncio
async def test_critique_node_stamps_monotonic_wall_clock() -> None:
    import datetime as _dt

    from chora_ai_kernel_orchestrator.domain.qgen_crew import CandidatePayload

    executor = _TokenSplitExecutor(
        output_payload=_critic_accept(),
        input_tokens=10,
        output_tokens=5,
        tokens_consumed_total=15,
        delay_s=0.02,
    )
    state = _starting_state(
        attempt_count=1,
        current_candidate=CandidatePayload(stem="x", question_type="mcq", payload_json=_good_mcq_payload()),
    )
    delta = await critique_node(state, executor=executor)
    row = delta["pipeline_trace"][-1]
    started = _dt.datetime.fromisoformat(row["started_at"])
    completed = _dt.datetime.fromisoformat(row["completed_at"])
    assert started <= completed
    assert (completed - started).total_seconds() >= 0.01


@pytest.mark.asyncio
async def test_below_threshold_generate_row_stamps_real_output_tokens() -> None:
    """Evaluator self-rejection ({"scored": null}) still records the per-hop
    split on the REJECTED generate row (the below-threshold attempt consumed
    tokens too)."""
    executor = _TokenSplitExecutor(
        output_payload=json.dumps({"scored": None, "reason": "below_threshold"}),
        input_tokens=300,
        output_tokens=12,
        tokens_consumed_total=312,
    )
    delta = await generate_node(_starting_state(), executor=executor)
    row = delta["pipeline_trace"][-1]
    assert row["status"] == "REJECTED"
    assert row["input_tokens"] == 300
    assert row["output_tokens"] == 12
