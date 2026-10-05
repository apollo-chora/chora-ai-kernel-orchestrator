"""RED: after a confirm, the run PARKS on the first learner-output dispatch.

ADR-254 D5 moved study_aids / practice_test from a detached in-process
generation to parked dispatches that run AFTER publish_analyzed. So the HITL
resume (the gateway's synchronous POST /resume) now returns at an agent park
with the analysis already durable. Two things must hold for the FE:

  * the runner reports the reviewed edges from the analyzed body (the panel
    payload is None on a dispatch park, and before this the edges were read
    from the interrupt payload, so a confirm would have answered with []);
  * the resume route answers ``completed`` (the learner is done reviewing; the
    artifacts ride the outputs event), never ``awaiting_review``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chora_ai_kernel_orchestrator.adapter.http.weakness_router import build_weakness_router
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    DISPATCH_INTERRUPT_KEY,
)
from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew_runner import (
    RunResult,
    WeaknessAnalyserCrewRunner,
)

_TENANT = "11111111-1111-7111-8111-111111111111"
_UPLOAD = "01a0235d-2487-7263-8754-92993f0cd59e"
_THREAD = f"{_TENANT}:{_UPLOAD}"
_EDGES = [
    {
        "concept_key": "adding-fractions",
        "concept_label": "Adding fractions",
        "category": "concept",
        "tags": [],
        "confidence": 0.9,
        "strength": 0.4,
        "descriptor_json": "{}",
    }
]


class _Interrupt:
    def __init__(self, value: Any) -> None:
        self.value = value


def _outputs_park_terminal() -> dict[str, Any]:
    """What LangGraph hands back when the study_aids hop parks after the publish."""
    return {
        "__interrupt__": [
            _Interrupt(
                {
                    DISPATCH_INTERRUPT_KEY: {
                        "agent_role": "companion_diagnose",
                        "execution_id": f"{_UPLOAD}:study_aids:abc",
                    }
                }
            )
        ],
        "run_id": "run-1",
        "governance_status": "approved",
        "analyzed_body": {"upload_id": _UPLOAD, "edges": _EDGES},
        "reviewed_edges": _EDGES,
        "published_row_id": "row-1",
        "generated_outputs": [],
    }


@pytest.mark.asyncio
async def test_a_post_review_park_reports_the_reviewed_edges() -> None:
    runner = WeaknessAnalyserCrewRunner.__new__(WeaknessAnalyserCrewRunner)
    result = await runner._result(_outputs_park_terminal(), thread_id=_THREAD, fallback_run_id="run-1")

    assert result.awaiting_agent is True and result.interrupted is True
    assert result.review_panel is None and result.review_payload is None
    assert result.governance_status == "approved"
    assert result.published_row_id == "row-1"
    assert [e["concept_key"] for e in result.edges] == ["adding-fractions"], (
        "the analysis is published; a dispatch park after it must still answer with the edges"
    )


class _FakeRunner:
    async def resume(self, *, thread_id: str, decision: dict[str, Any]) -> RunResult:
        return RunResult(
            interrupted=True,
            thread_id=thread_id,
            run_id="run-1",
            governance_status="approved",
            review_payload=None,
            edges=_EDGES,
            published_row_id="row-1",
            generated_outputs=[],
            review_panel=None,
            awaiting_agent=True,
        )


def test_the_resume_route_answers_completed_on_a_post_review_park() -> None:
    app = FastAPI()
    app.include_router(build_weakness_router(runner=_FakeRunner()))
    client = TestClient(app)
    resp = client.post(
        f"/v1/orchestrator/weakness/{_UPLOAD}/resume",
        json={
            "action": "confirm",
            "edges": [{"proposed_edge_id": "pe-0", "decision": "accept"}],
            "added_struggles": [],
            "selected_outputs": ["study_aids"],
        },
        headers={"X-Tenant-Id": _TENANT},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed", body
    assert body["governance_status"] == "approved"
    assert body["interrupted"] is False, "an agent park is not a review pause for the FE"
    assert body["panel"] is None
    assert [e["concept_key"] for e in body["edges"]] == ["adding-fractions"]
    assert body["generated_outputs"] == []
