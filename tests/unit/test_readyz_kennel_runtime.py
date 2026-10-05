"""RED: ADR-254 D5, ``/readyz`` reflects every lane, not only the guardrail.

``main.py``'s lifespan sets ``app.state.kennel_runtime`` before the app serves.
When it is set, readiness is the runtime's verdict: a pod whose lanes did not
start, whose drain or scan died, or whose completion loop is absent is NOT
ready, with the reasons in the body. The guardrail check stays in front of it.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from chora_ai_kernel_orchestrator.adapter.http.handlers import build_app
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    ModelArmorGuardrailPort,
    StubScreener,
    Verdict,
)


def _guardrail() -> ModelArmorGuardrailPort:
    return ModelArmorGuardrailPort.from_components(
        screener=StubScreener(force_verdict=Verdict.ALLOW),
        project="chora-test",
        location="us-central1",
        environment="dev",
    )


class _FakeRuntime:
    def __init__(self, verdict: dict[str, Any]) -> None:
        self._verdict = verdict

    def readiness(self) -> dict[str, Any]:
        return dict(self._verdict)


def test_readyz_is_200_when_the_kennel_runtime_is_ready() -> None:
    app = build_app(guardrail=_guardrail())
    app.state.kennel_runtime = _FakeRuntime(
        {"ready": True, "reasons": [], "lanes": {"oe_grading": {"started": True}}, "roles": ["oe_evaluate"]}
    )
    r = TestClient(app).get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["kennel"]["roles"] == ["oe_evaluate"]


def test_readyz_is_503_with_the_reasons_when_the_runtime_is_not_ready() -> None:
    app = build_app(guardrail=_guardrail())
    app.state.kennel_runtime = _FakeRuntime(
        {
            "ready": False,
            "reasons": ["lane_not_started:oe_grading", "scan_task_dead"],
            "lanes": {"oe_grading": {"started": False, "detail": ""}},
            "roles": ["oe_evaluate"],
        }
    )
    r = TestClient(app).get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["reason"] == "kennel_runtime_not_ready"
    assert body["detail"] == ["lane_not_started:oe_grading", "scan_task_dead"]


def test_readyz_guardrail_gate_still_comes_first() -> None:
    app = build_app(guardrail=None)
    app.state.kennel_runtime = _FakeRuntime({"ready": True, "reasons": [], "lanes": {}, "roles": []})
    r = TestClient(app).get("/readyz")
    assert r.status_code == 503
    assert r.json()["reason"] == "guardrail_unconfigured"
