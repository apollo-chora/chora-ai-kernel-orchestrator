"""D6.3 multi-tenant + multi-workflow chaos test for a parking crew.

Per ``agentic-resilience-d6`` skill Pillar 3 + the closure-saga reference in
``docs/architecture/poc/adk-poc-results-week2.md`` section B.6.3, this test
exercises:

* N concurrent workflows x M tenants on a shared LangGraph saver
* Per-tenant isolation in the saver's `channel_values`
* "Pod death" via graph teardown
* Cross-engine resume (rebuild graph with same saver + thread_id)
* No cross-tenant bleed during chaos

InMemorySaver is used for CI determinism; the contract (thread_id + saver +
node-state checkpoints) is identical for the production PostgresSaver per the
closure-saga POC evidence.

2026-08-23: re-pointed from the deleted 6-agent ai_assist crew onto the qgen
crew, which is what parks in production now (ADR-253 D1/D2). Every property
above is unchanged; what moved is the graph carrying them, and the pause is
now a real agent-dispatch park through ``interrupt()`` rather than a HITL one.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.checkpointer import build_thread_id
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

WORKFLOW_ID = "qgen"


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
    """Parks on the generate dispatch through the real ``interrupt()``; answers
    the critique inline so one resume carries a run to terminal."""

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
            gcid="00000000-0000-7000-9e0a-000000000001",
            thread_id=workflow_id,
            input_payload=input_payload,
            workflow_id=workflow_id,
            agid=agid,
            source_project="chora-489812",
        )
        completion = interrupt(wrap_for_interrupt(request))
        return AgentExecutorResponse(
            execution_id=execution_id,
            output_payload=str(completion["output_payload"]),
        )


def _resume_command(stem: str) -> Command:
    return Command(
        resume={
            "status": STATUS_OK,
            "output_payload": _mcq(stem),
            "agent_role": ROLE_GENERATE,
        }
    )


def _state(*, job_id: str, tenant_id: str, gcid: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "tenant_id": tenant_id,
        "gcid": gcid,
        "prompt": "Photosynthesis",
        "question_type": "mcq",
        "metadata": {},
        "max_retries": 3,
        "pipeline_trace": [],
        "errors": [],
    }


# 3 tenants x 3 workflows each; per-tenant tenant_id is a distinct UUID.
TENANTS = [
    ("00000000-0000-7000-9d01-000000000001", "tenant-A"),
    ("00000000-0000-7000-9d02-000000000001", "tenant-B"),
    ("00000000-0000-7000-9d03-000000000001", "tenant-C"),
]
WORKFLOWS_PER_TENANT = 3


def _job_id(tenant_index: int, workflow_index: int) -> str:
    """A distinct UUID per (tenant, workflow).

    It has to be a real UUID: build_dispatch_request refuses a non-UUID
    workflow_id because ai_kernel_outbox_events.workflow_id is UUID NOT NULL,
    so a non-UUID would fail the INSERT inside the park transaction and the
    run would never resume. The guard fires before the park, which is how this
    test found out.
    """
    return f"0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f{tenant_index}{workflow_index:03d}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_tenant_concurrent_workflows_isolated() -> None:
    """Scenario (a) + (b): N workflows x M tenants concurrent on a shared
    saver. Each run's state stays isolated; no cross-tenant bleed.
    """
    saver = InMemorySaver()
    graph = build_qgen_crew_graph(executor=_ParkingExecutor(), guardrail=_Guardrail(), checkpointer=saver)

    plan: list[tuple[str, str, str, str]] = []
    for t_index, (tenant_id, _label) in enumerate(TENANTS):
        for i in range(WORKFLOWS_PER_TENANT):
            run_id = _job_id(t_index, i)
            gcid = f"00000000-0000-7000-9e0{i}-000000000001"
            thread_id = build_thread_id(tenant_id=tenant_id, workflow_id=WORKFLOW_ID, run_id=run_id)
            plan.append((tenant_id, gcid, run_id, thread_id))

    async def _run(tenant_id: str, gcid: str, run_id: str, thread_id: str) -> tuple[str, str, bool]:
        config = {"configurable": {"thread_id": thread_id}}
        out = await graph.ainvoke(_state(job_id=run_id, tenant_id=tenant_id, gcid=gcid), config=config)
        snap = await graph.aget_state(config=config)
        parked = bool(out.get("__interrupt__")) and bool(snap.next)
        return tenant_id, snap.values.get("tenant_id", ""), parked

    t0 = time.time()
    results = await asyncio.gather(*[_run(*p) for p in plan])
    wall = time.time() - t0

    # All 9 workflows kicked off + parked on their dispatch within bounded
    # contention.
    assert len(results) == 9
    # Every one of them actually PARKED. Without this the isolation assertions
    # below would pass just as happily on nine runs that refused at validation
    # and never suspended at all.
    assert all(parked for _, _, parked in results), "every run should be parked"
    # Pillar 3 (b): every run's state carries its OWN tenant_id; zero
    # cross-tenant bleed.
    for expected_tenant, actual_tenant, _parked in results:
        assert actual_tenant == expected_tenant, (
            f"cross-tenant bleed: expected {expected_tenant!r}, got {actual_tenant!r}"
        )

    # Pillar 3 (b.4): cross-tenant query attempt. Each thread_id only surfaces
    # its own tenant's state via aget_state.
    distinct_tenants = {t for t, _, _ in results}
    assert len(distinct_tenants) == 3, "expected 3 distinct tenants"

    # Pillar 3 (e): bounded contention. 9 concurrent should be sub-second on
    # InMemorySaver (the production PostgresSaver budget is wider; see closure
    # POC section B.6.3, 0.39s for 9 concurrent against live Cloud SQL).
    assert wall < 5.0, f"concurrent dispatch should be bounded; got {wall:.2f}s"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chaos_kill_mid_flight_all_workflows_resume_with_isolation() -> None:
    """Scenario (c) + (d): tear down the graph mid-flight (simulating pod
    death); rebuild the graph with the same saver; verify ALL workflows resume
    on the new graph with tenant isolation intact.
    """
    saver = InMemorySaver()
    graph1 = build_qgen_crew_graph(executor=_ParkingExecutor(), guardrail=_Guardrail(), checkpointer=saver)

    threads: list[tuple[str, str, str]] = []  # (tenant_id, gcid, thread_id)
    for t_index, (tenant_id, label) in enumerate(TENANTS):
        gcid = f"00000000-0000-7000-9e09-{label[-1].lower()}00000001"
        run_id = _job_id(t_index, 9)
        thread_id = build_thread_id(tenant_id=tenant_id, workflow_id=WORKFLOW_ID, run_id=run_id)
        threads.append((tenant_id, gcid, thread_id))

    # Pre-chaos: start each workflow, each parks on its generate dispatch.
    for tenant_id, gcid, thread_id in threads:
        await graph1.ainvoke(
            _state(job_id=thread_id.split(":")[-1], tenant_id=tenant_id, gcid=gcid),
            config={"configurable": {"thread_id": thread_id}},
        )

    # Snapshot pre-chaos state per thread.
    pre_states = {}
    for tenant_id, _, thread_id in threads:
        snap = await graph1.aget_state(config={"configurable": {"thread_id": thread_id}})
        assert snap.next, f"workflow {thread_id} should be parked"
        pre_states[thread_id] = (tenant_id, snap.values.get("tenant_id", ""))

    # CHAOS: tear down the graph and executor (simulates pod death).
    del graph1

    # Recovery: rebuild the graph with the SAME saver.
    graph2 = build_qgen_crew_graph(executor=_ParkingExecutor(), guardrail=_Guardrail(), checkpointer=saver)

    # Pillar 3 (c.i): cross-engine state retrieval. Each thread's snapshot is
    # identical to pre-chaos; tenant_id preserved.
    post_states = {}
    for tenant_id, _, thread_id in threads:
        snap = await graph2.aget_state(config={"configurable": {"thread_id": thread_id}})
        assert snap is not None, f"workflow {thread_id} lost state across chaos"
        post_states[thread_id] = (tenant_id, snap.values.get("tenant_id", ""))

    assert post_states == pre_states, f"state mismatch across chaos cycle: pre={pre_states} post={post_states}"

    # Pillar 3 (c.ii): resume each workflow concurrently on the rebuilt graph;
    # each run completes with its own tenant_id intact.
    async def _resume(thread_id: str) -> dict:
        return await graph2.ainvoke(
            _resume_command("Why is chlorophyll green?"),
            config={"configurable": {"thread_id": thread_id}},
        )

    finals = await asyncio.gather(*[_resume(thread_id) for _, _, thread_id in threads])

    # All resumed to terminal; tenant attribution preserved through the resume.
    for (tenant_id, _, _), final in zip(threads, finals, strict=True):
        assert not final.get("__interrupt__"), "run did not reach terminal"
        assert not final.get("refusal_reason"), final.get("errors")
        assert final["completed_candidate"] is not None
        assert final["tenant_id"] == tenant_id, "tenant_id lost on resume"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_resumes_do_not_corrupt_other_tenant_state() -> None:
    """Scenario (d), non-disruption: resuming one tenant's workflow must NOT
    corrupt or affect another tenant's parked workflow.
    """
    saver = InMemorySaver()
    graph = build_qgen_crew_graph(executor=_ParkingExecutor(), guardrail=_Guardrail(), checkpointer=saver)

    threads: list[tuple[str, str]] = []  # (tenant_id, thread_id)
    for t_index, (tenant_id, _label) in enumerate(TENANTS):
        thread_id = build_thread_id(
            tenant_id=tenant_id,
            workflow_id=WORKFLOW_ID,
            run_id=_job_id(t_index, 99),
        )
        threads.append((tenant_id, thread_id))
        await graph.ainvoke(
            _state(
                job_id=thread_id.split(":")[-1],
                tenant_id=tenant_id,
                gcid="00000000-0000-7000-9e0a-000000000001",
            ),
            config={"configurable": {"thread_id": thread_id}},
        )

    # Resume only tenant-A's workflow. The other two stay parked.
    tenant_a_thread = threads[0][1]
    await graph.ainvoke(
        _resume_command("Tenant A candidate"),
        config={"configurable": {"thread_id": tenant_a_thread}},
    )

    # Verify tenant-B and tenant-C are STILL parked: non-disruption.
    for tenant_id, thread_id in threads[1:]:
        snap = await graph.aget_state(config={"configurable": {"thread_id": thread_id}})
        assert snap.next, (
            f"tenant {tenant_id} workflow disrupted by tenant-A's resume; expected parked, got {snap.next=}"
        )
        assert snap.values.get("tenant_id") == tenant_id, "tenant_id bleed during single-tenant resume"
