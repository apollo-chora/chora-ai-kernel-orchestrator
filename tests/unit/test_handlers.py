"""FastAPI handler tests using TestClient + stub adapters.

The app's HTTP surface is the four paths the mesh admits. RULING A
(2026-08-23) deleted the rest, so the orchestrate / crews / agents /
ai-assist route tests went with the routes they covered; what remains is the
probe surface and the two mounted routers.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from chora_ai_kernel_orchestrator.adapter.http.handlers import build_app
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    ModelArmorGuardrailPort,
    StubScreener,
    Verdict,
)


def _port_with(verdict: Verdict) -> ModelArmorGuardrailPort:
    return ModelArmorGuardrailPort.from_components(
        screener=StubScreener(force_verdict=verdict),
        project="chora-test",
        location="us-central1",
        environment="dev",
    )


def test_healthz_ok() -> None:
    app = build_app(guardrail=_port_with(Verdict.ALLOW))
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_readyz_with_the_guardrail_returns_200() -> None:
    """Per ADR-152 + ADR-146 (Model Broker retired, chora-guardrail superseded
    by Cloud Model Armor) the guardrail port is the only mandatory adapter:
    every crew screens through it, and it is the source of the shared
    screener."""
    app = build_app(guardrail=_port_with(Verdict.ALLOW))
    client = TestClient(app)
    r = client.get("/readyz")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ready"


def test_readyz_without_the_guardrail_returns_503() -> None:
    """Guardrail absence MUST 503: the crews' pre and post LLM hops require
    Cloud Model Armor screening per ADR-152 Tier 3 D9. There is no inline
    fallback."""
    app = build_app(guardrail=None)
    client = TestClient(app)
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["reason"] == "guardrail_unconfigured"


def test_the_deleted_routes_are_gone() -> None:
    """RULING A. Every one of these was denied by the ns AuthorizationPolicy,
    so none had a caller that could reach it; serving them anyway was dead
    surface that read as live in the OpenAPI. A 404 here is the point."""
    app = build_app(guardrail=_port_with(Verdict.ALLOW))
    client = TestClient(app)
    assert client.get("/crews").status_code == 404
    assert client.get("/agents").status_code == 404
    assert client.get("/orchestrate/anything").status_code == 404
    assert client.post("/orchestrate", json={}).status_code == 404
    assert client.post("/v1/orchestrator/runs/ai-assist", json={}).status_code == 404


# --- Growth-Edge resume route mount (ADR-205 WS-2 / CHO-1954) -----------------


def test_build_app_initialises_weakness_crew_runner_state() -> None:
    """The composition root defines ``app.state.weakness_crew_runner`` (None
    until the lifespan builds it in graph mode) so the resume router resolves
    it at request time."""
    app = build_app(guardrail=None)
    assert app.state.weakness_crew_runner is None


def test_weakness_resume_route_mounted_returns_503_by_default() -> None:
    """The resume route is ALWAYS mounted (never 404); with no runner wired
    (single_shot / unconfigured) it returns 503."""
    app = build_app(guardrail=None)
    client = TestClient(app)
    r = client.post(
        "/v1/orchestrator/weakness/upload-1/resume",
        headers={"X-Tenant-Id": "tenant-1"},
        json={
            "action": "confirm",
            "edges": [{"proposed_edge_id": "pe-0", "decision": "accept"}],
        },
    )
    assert r.status_code == 503


def test_prompt_registry_route_mounted_returns_503_by_default() -> None:
    """CHO-2368: the O+ catalogue read API is mounted at build time and
    resolves its repository from app.state at request time, so it answers 503
    rather than 404 until the lifespan wires the repo."""
    app = build_app(guardrail=None)
    client = TestClient(app)
    r = client.get(
        "/v1/prompt-registry/agents/qgen_question/versions",
        headers={"X-Tenant-Id": "tenant-1"},
    )
    assert r.status_code == 503
