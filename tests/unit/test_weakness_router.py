"""CHO-1973 (ADR-205 D4/D5) — FE resume route for the Growth-Edge HITL review.

POST /v1/orchestrator/weakness/{upload_id}/resume carries the PANEL-SHAPE learner
decision (the A+ panel state IS the resume payload, ADR-205 D4):

    {action: "confirm"|"reiterate",
     edges: [{proposed_edge_id, decision, merge_into_id?, difficulty?}],
     added_struggles: [concept_key], selected_outputs: [output_kind]}

The route reconstructs the DETERMINISTIC thread_id from tenant + upload ALONE (no
run_id) and drives the checkpointed graph via the crew runner — returning a
job/panel shape: AWAITING_REVIEW + a refreshed panel on reiterate, COMPLETED (or
BLOCKED/FAILED) on confirm.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from chora_ai_kernel_orchestrator.adapter.http.weakness_router import build_weakness_router
from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew_runner import RunResult


def _panel() -> dict[str, Any]:
    return {
        "upload_id": "upload-1",
        "tenant_id": "tenant-1",
        "learner_gcid": "gcid-1",
        "familiar": {},
        "proposed_edges": [{"proposed_edge_id": "pe-0", "concept_label": "adding fractions"}],
        "candidate_struggles": [],
        "available_outputs": [{"kind": "focused_dose", "mana_price": 0, "default_selected": True}],
    }


class FakeRunner:
    def __init__(
        self,
        *,
        interrupted: bool = False,
        governance_status: str | None = "approved",
        panel: dict[str, Any] | None = None,
    ) -> None:
        self.resumed: list[dict[str, Any]] = []
        self._interrupted = interrupted
        self._gov = governance_status
        self._panel = panel

    async def resume(self, *, thread_id: str, decision: dict[str, Any]) -> RunResult:
        self.resumed.append({"thread_id": thread_id, "decision": decision})
        return RunResult(
            interrupted=self._interrupted,
            thread_id=thread_id,
            run_id="run-1",
            governance_status=None if self._interrupted else self._gov,
            edges=[]
            if self._interrupted
            else [{"concept_key": "adding-fractions", "concept_label": "adding fractions"}],
            generated_outputs=[]
            if self._interrupted
            else [{"type": "study_aids", "content": '{"advice":"keep going"}', "metered": True}],
            review_panel=self._panel if self._interrupted else None,
        )


def _client(runner: Any) -> TestClient:
    app = FastAPI()
    app.include_router(build_weakness_router(runner=runner))
    return TestClient(app)


def _confirm_body() -> dict[str, Any]:
    return {
        "action": "confirm",
        "edges": [{"proposed_edge_id": "pe-0", "decision": "accept"}],
        "added_struggles": [],
        "selected_outputs": ["focused_dose"],
    }


def test_confirm_completes_and_returns_edges() -> None:
    runner = FakeRunner(interrupted=False, governance_status="approved")
    client = _client(runner)
    resp = client.post(
        "/v1/orchestrator/weakness/upload-1/resume",
        headers={"X-Tenant-Id": "tenant-1"},
        json=_confirm_body(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert body["interrupted"] is False
    assert body["governance_status"] == "approved"
    assert body["upload_id"] == "upload-1"
    assert [e["concept_key"] for e in body["edges"]] == ["adding-fractions"]
    assert body["generated_outputs"][0]["type"] == "study_aids"
    assert body["panel"] is None  # completed -> no pending panel
    # DETERMINISTIC 2-segment thread (no run_id)
    assert runner.resumed[0]["thread_id"] == "tenant-1:upload-1"
    # the panel-shape decision is forwarded structured
    fwd = runner.resumed[0]["decision"]
    assert fwd["action"] == "confirm"
    assert fwd["edges"] == [{"proposed_edge_id": "pe-0", "decision": "accept"}]
    assert fwd["selected_outputs"] == ["focused_dose"]


def test_reiterate_returns_awaiting_review_with_panel() -> None:
    runner = FakeRunner(interrupted=True, panel=_panel())
    client = _client(runner)
    resp = client.post(
        "/v1/orchestrator/weakness/upload-1/resume",
        headers={"X-Tenant-Id": "tenant-1"},
        json={"action": "reiterate", "added_struggles": ["long-division"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "awaiting_review"
    assert body["interrupted"] is True
    # the FE re-enters review with the refreshed panel
    assert body["panel"]["proposed_edges"][0]["proposed_edge_id"] == "pe-0"
    assert runner.resumed[0]["decision"]["added_struggles"] == ["long-division"]


def test_blocked_run_reports_blocked_status() -> None:
    runner = FakeRunner(interrupted=False, governance_status="blocked")
    client = _client(runner)
    resp = client.post(
        "/v1/orchestrator/weakness/upload-1/resume",
        headers={"X-Tenant-Id": "tenant-1"},
        json=_confirm_body(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "blocked"


def test_resume_rejects_unknown_action() -> None:
    client = _client(FakeRunner())
    resp = client.post(
        "/v1/orchestrator/weakness/upload-1/resume",
        headers={"X-Tenant-Id": "tenant-1"},
        json={"action": "delete_everything"},
    )
    assert resp.status_code == 400


def test_unconfigured_runner_returns_503() -> None:
    client = _client(None)
    resp = client.post(
        "/v1/orchestrator/weakness/upload-1/resume",
        headers={"X-Tenant-Id": "tenant-1"},
        json=_confirm_body(),
    )
    assert resp.status_code == 503


def test_resume_resolves_runner_from_app_state() -> None:
    """The crew runner is built in the FastAPI lifespan AFTER the router is
    mounted, so a no-arg router resolves it at REQUEST time from
    ``request.app.state.weakness_crew_runner``."""
    app = FastAPI()
    app.include_router(build_weakness_router())  # no explicit runner
    runner = FakeRunner()
    app.state.weakness_crew_runner = runner
    client = TestClient(app)
    resp = client.post(
        "/v1/orchestrator/weakness/upload-1/resume",
        headers={"X-Tenant-Id": "tenant-1"},
        json=_confirm_body(),
    )
    assert resp.status_code == 200, resp.text
    assert runner.resumed[0]["thread_id"] == "tenant-1:upload-1"


def test_resume_503_when_app_state_runner_unset() -> None:
    app = FastAPI()
    app.include_router(build_weakness_router())  # no explicit runner, no app.state
    client = TestClient(app)
    resp = client.post(
        "/v1/orchestrator/weakness/upload-1/resume",
        headers={"X-Tenant-Id": "tenant-1"},
        json=_confirm_body(),
    )
    assert resp.status_code == 503
