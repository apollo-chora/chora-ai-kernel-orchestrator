"""Crash-resume integration test for a parked crew run.

Per the `langgraph-orchestrator-python` skill the PostgresSaver checkpoints
allow a kill-mid-execution plus restart to resume from the last completed
node. We exercise the same property using LangGraph's InMemorySaver here
because no live Postgres is available in CI; the contract (saver + thread_id +
node-state checkpoints) is identical for the production saver.

The test simulates a "crash" by tearing down the graph after the first
ainvoke (which parks on the agent dispatch), then rebuilds the graph with the
SAME saver and thread_id. The resume call hits the persisted checkpoint and
continues from the park, which is the proof that the checkpointer plus
thread_id contract holds.

2026-08-23: re-pointed from the deleted 6-agent ai_assist crew onto the qgen
crew, which is what actually parks in production now (ADR-253 D1/D2). The
property under test is unchanged; only the graph carrying it moved. The park
goes through the REAL ``interrupt()`` the PubSubAgentExecutor uses, so this
exercises LangGraph's suspend and resume rather than a stub.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    GuardrailScreenInput,
    ScreenResult,
    Verdict,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    STATUS_OK,
    build_dispatch_request,
    wrap_for_interrupt,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_CRITIQUE,
    ROLE_GENERATE,
    build_qgen_crew_graph,
)

_TENANT = "00000000-0000-7000-8000-000000000001"
_GCID = "00000000-0000-7000-8000-000000000002"
_JOB = "0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"


class _Guardrail:
    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
        return ScreenResult(verdict=Verdict.ALLOW, reason="clean")


def _mcq(stem: str) -> str:
    return json.dumps(
        {
            "stem": stem,
            "question_type": "mcq",
            "mcq_payload": {
                "options": [
                    {
                        "option_id": "a",
                        "label": "A",
                        "text": "x",
                        "is_correct": True,
                        "explainer": "x",
                    },
                    {
                        "option_id": "b",
                        "label": "B",
                        "text": "y",
                        "is_correct": False,
                        "explainer": "y",
                    },
                ],
                "scoring_mode": "single_correct",
            },
        }
    )


def _accept() -> str:
    return json.dumps({"accepted": True, "critique_notes": "fine", "suggested_revisions": []})


class _ParkingExecutor:
    """Parks on the generate dispatch through the real ``interrupt()``, exactly
    as PubSubAgentExecutor does; answers the critique inline so the run reaches
    terminal on a single resume."""

    def __init__(self) -> None:
        self.resumed: list[Any] = []

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
        context_window: Any = None,
        available_tools: Any = None,
    ) -> AgentExecutorResponse:
        if agent_role == ROLE_CRITIQUE:
            return AgentExecutorResponse(execution_id=execution_id, output_payload=_accept())
        assert agent_role == ROLE_GENERATE
        request = build_dispatch_request(
            agent_role=agent_role,
            execution_id=execution_id,
            tenant_id=tenant_id,
            gcid=_GCID,
            thread_id=workflow_id or _JOB,
            input_payload=input_payload,
            workflow_id=workflow_id,
            agid=agid,
            source_project="chora-489812",
        )
        completion = interrupt(wrap_for_interrupt(request))
        self.resumed.append(completion)
        return AgentExecutorResponse(
            execution_id=execution_id,
            output_payload=str(completion["output_payload"]),
        )


def _state() -> dict[str, Any]:
    return {
        "job_id": _JOB,
        "tenant_id": _TENANT,
        "gcid": _GCID,
        "prompt": "Photosynthesis",
        "question_type": "mcq",
        "metadata": {},
        "max_retries": 3,
        "pipeline_trace": [],
        "errors": [],
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_crash_resume_restores_from_checkpoint() -> None:
    """Run, simulate crash, rebuild graph, resume: same saver + thread_id."""
    saver = InMemorySaver()
    thread_id = f"{_TENANT}:qgen:{_JOB}"
    config = {"configurable": {"thread_id": thread_id}}

    # Phase 1: the initial run parks on the generate dispatch.
    executor1 = _ParkingExecutor()
    graph1 = build_qgen_crew_graph(executor=executor1, guardrail=_Guardrail(), checkpointer=saver)
    first = await graph1.ainvoke(_state(), config=config)
    assert first.get("__interrupt__"), "the run should be parked on its dispatch"
    snapshot1 = await graph1.aget_state(config)
    assert snapshot1.next, "graph should have a pending node while parked"
    trace_at_park = [t["name"] for t in (snapshot1.values.get("pipeline_trace") or [])]
    assert "guardrail_pre" in trace_at_park

    # Phase 2: discard graph1 (simulate process restart) and rebuild with the
    # SAME saver. Resume via Command. Production-saver-equivalent behaviour:
    # same thread_id + same saver means same state.
    del graph1, executor1
    executor2 = _ParkingExecutor()
    graph2 = build_qgen_crew_graph(executor=executor2, guardrail=_Guardrail(), checkpointer=saver)

    final = await graph2.ainvoke(
        Command(
            resume={
                "status": STATUS_OK,
                "output_payload": _mcq("Why is chlorophyll green?"),
                "agent_role": ROLE_GENERATE,
            }
        ),
        config=config,
    )
    assert not final.get("__interrupt__"), "the resumed run should reach terminal"
    assert not final.get("refusal_reason"), final.get("errors")
    assert final["completed_candidate"] is not None
    assert final["completed_candidate"].stem == "Why is chlorophyll green?"

    # The trace persisted across the "crash": the nodes that ran before the
    # park are still in the state the rebuilt graph resumed from.
    nodes = {t["name"] for t in (final.get("pipeline_trace") or [])}
    for stage in ("validate_input", "guardrail_pre", "generate", "critique"):
        assert stage in nodes, f"{stage} lost across the crash cycle: {nodes}"
