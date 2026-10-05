"""RED: QGenDispatchAdapter, the kennel side of the qgen lanes (ADR-254 D2).

The qgen graph nodes keep their ``executor.execute(agent_role=qgen_question |
qgen_critic, ...)`` call shape. This adapter sits between them and the
``PubSubAgentExecutor``: it maps the agent id onto the lane role the topics
are keyed on, keeps the agid (Armor tier + decision log identity), builds the
session-state-shaped payload the subscriber-only binaries merge verbatim,
and requires the explicit ``workflow_id`` the outbox needs (qgen thread ids
are ``assist_id`` strings, not guaranteed UUIDs for the outbox's column).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    LANE_ROLE_CRITIQUE,
    LANE_ROLE_GENERATE,
    LANE_ROLE_RENDER,
)
from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_dispatch import (
    QGEN_LANE_ROLES,
    QGenDispatchAdapter,
)

_JOB = "0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"


@dataclass
class _Inner:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def execute(self, **kwargs: Any) -> AgentExecutorResponse:
        self.calls.append(kwargs)
        return AgentExecutorResponse(execution_id=kwargs["execution_id"], output_payload="{}")


def test_lane_roles_are_the_three_go_dispatch_roles() -> None:
    assert QGEN_LANE_ROLES == (LANE_ROLE_GENERATE, LANE_ROLE_CRITIQUE, LANE_ROLE_RENDER)


@pytest.mark.asyncio
async def test_generate_maps_role_keeps_agid_and_ships_session_state_shape() -> None:
    inner = _Inner()
    adapter = QGenDispatchAdapter(inner)
    payload = {
        "prompt": "P",
        "question_type": "mcq",
        "metadata": {"subject": "Maths"},
        "gcid": "g-1",
        "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01",
    }
    await adapter.execute(
        execution_id=f"{_JOB}:generate_set:c0:r0",
        tenant_id="t-1",
        agid="qgen_question",
        agent_role="qgen_question",
        input_payload=json.dumps(payload),
        prompt_template_id="qgen_crew::generate",
        workflow_id=_JOB,
    )
    call = inner.calls[0]
    assert call["agent_role"] == LANE_ROLE_GENERATE
    assert call["agid"] == "qgen_question"
    assert call["workflow_id"] == _JOB
    assert call["execution_id"] == f"{_JOB}:generate_set:c0:r0"
    assert call["prompt_template_id"] == "qgen_crew::generate"
    shipped = json.loads(call["input_payload"])
    assert shipped["input_payload"] == "P" and shipped["subject_hint"] == "Maths"
    # gcid + trace context stay in the body so the executor can lift them
    # into the envelope (the Go merge takes them from the envelope, not here).
    assert shipped["gcid"] == "g-1" and shipped["traceparent"].startswith("00-")


@pytest.mark.asyncio
async def test_critique_maps_onto_the_critique_lane() -> None:
    inner = _Inner()
    adapter = QGenDispatchAdapter(inner)
    await adapter.execute(
        execution_id=f"{_JOB}:critique:1",
        tenant_id="t-1",
        agid="qgen_critic",
        agent_role="qgen_critic",
        input_payload=json.dumps({"stem": "s", "author_prompt": "P", "attempt_index": 0}),
        prompt_template_id="qgen_crew::critique",
        workflow_id=_JOB,
    )
    call = inner.calls[0]
    assert call["agent_role"] == LANE_ROLE_CRITIQUE and call["agid"] == "qgen_critic"
    shipped = json.loads(call["input_payload"])
    assert shipped["job_id"] == f"{_JOB}:critique:1" and shipped["author_prompt"] == "P"
    assert json.loads(shipped["input_payload"])["stem"] == "s"


@pytest.mark.asyncio
async def test_render_role_passes_its_payload_verbatim_plus_identity() -> None:
    inner = _Inner()
    adapter = QGenDispatchAdapter(inner)
    await adapter.execute(
        execution_id=f"{_JOB}:render:c0:i0:stem",
        tenant_id="t-1",
        agid="qgen_renderer",
        agent_role=LANE_ROLE_RENDER,
        input_payload=json.dumps({"render_prompt": "apple", "mode": "scene", "job_id": _JOB, "gcid": "g"}),
        workflow_id=_JOB,
    )
    call = inner.calls[0]
    assert call["agent_role"] == LANE_ROLE_RENDER and call["agid"] == "qgen_renderer"
    shipped = json.loads(call["input_payload"])
    assert shipped["render_prompt"] == "apple" and shipped["mode"] == "scene"
    assert shipped["tenant_id"] == "t-1" and shipped["user_gcid"] == "g"


@pytest.mark.asyncio
async def test_unknown_role_is_refused_before_any_dispatch() -> None:
    inner = _Inner()
    adapter = QGenDispatchAdapter(inner)
    with pytest.raises(KeyError):
        await adapter.execute(
            execution_id="e",
            tenant_id="t",
            agid="x",
            agent_role="oe_evaluate",
            input_payload="{}",
            workflow_id=_JOB,
        )
    assert inner.calls == []


@pytest.mark.asyncio
async def test_missing_workflow_id_is_refused_before_any_dispatch() -> None:
    inner = _Inner()
    adapter = QGenDispatchAdapter(inner)
    with pytest.raises(ValueError):
        await adapter.execute(
            execution_id="e",
            tenant_id="t",
            agid="qgen_question",
            agent_role="qgen_question",
            input_payload="{}",
        )
    assert inner.calls == []


@pytest.mark.asyncio
async def test_context_window_and_tools_are_refused_like_the_executor_does() -> None:
    inner = _Inner()
    adapter = QGenDispatchAdapter(inner)
    with pytest.raises(ValueError):
        await adapter.execute(
            execution_id="e",
            tenant_id="t",
            agid="qgen_question",
            agent_role="qgen_question",
            input_payload="{}",
            workflow_id=_JOB,
            available_tools=["x"],
        )
    assert inner.calls == []
