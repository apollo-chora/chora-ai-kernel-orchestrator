"""CHO-2368 P3 - the resolve read route.

The familiar's ADK session is created by chora-consumption, not the
orchestrator, so the ADR-197 session-state injection needs a network-reachable
resolve: GET /v1/prompt-registry/agents/{agent_id}/resolve[?tenant_id=] returns
the PromptResolver verdict ({segments, version, source}) off the same
catalogue repo the read API already rides. Embedded (no active override) is a
VALID 200 resolution - never 404; unconfigured repo is 503 like the sibling
catalogue routes.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from chora_ai_kernel_orchestrator.adapter.http.prompt_registry_router import (
    build_prompt_registry_router,
)
from chora_ai_kernel_orchestrator.domain.prompt_registry.models import ScopeOverride


class _FakeResolveRepo:
    """Catalogue repo double exposing only the resolve port."""

    def __init__(self, overrides: list[ScopeOverride]) -> None:
        self._overrides = overrides
        self.calls: list[tuple[str, str]] = []

    async def fetch_active_overrides(self, *, tenant_id: str, agent_id: str) -> list[ScopeOverride]:
        self.calls.append((tenant_id, agent_id))
        return self._overrides


def _client(repo: Any | None) -> TestClient:
    app = FastAPI()
    app.include_router(build_prompt_registry_router(repo=repo))
    return TestClient(app)


def test_resolve_returns_active_platform_override() -> None:
    repo = _FakeResolveRepo(
        [
            ScopeOverride(
                scope="platform",
                version="1.1.0",
                segments={"role_frame": "CRAFT", "task_frame": "MORE"},
            )
        ]
    )
    resp = _client(repo).get("/v1/prompt-registry/agents/familiar/resolve")
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_id"] == "familiar"
    assert body["version"] == "1.1.0"
    assert body["source"] == "platform_override"
    assert body["segments"] == {"role_frame": "CRAFT", "task_frame": "MORE"}


def test_resolve_embedded_overlays_known_embedded_version() -> None:
    """An embedded resolution for a KNOWN agent carries that agent's embedded
    version (EMBEDDED_PROMPT_VERSIONS) so the caller can stamp what actually
    runs - consumption injects it and the ritual stamp agrees with the agent's
    own fallback constant.

    The chat agent is addressed as `companion_chat`, the id consumption
    resolves under. It read `familiar` here until the ids were reconciled: that
    WAS the right id when the fixture was written, so this is a re-point rather
    than a masked regression, and under the old key this route answered
    version=null while the agent ran a real prompt version."""
    repo = _FakeResolveRepo([])
    resp = _client(repo).get("/v1/prompt-registry/agents/companion_chat/resolve")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "v1"
    assert body["source"] == "embedded"
    assert body["segments"] == {}


def test_resolve_embedded_unknown_agent_has_null_version() -> None:
    repo = _FakeResolveRepo([])
    resp = _client(repo).get("/v1/prompt-registry/agents/not-an-agent/resolve")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] is None
    assert body["source"] == "embedded"


def test_resolve_threads_tenant_id_query() -> None:
    repo = _FakeResolveRepo([])
    resp = _client(repo).get(
        "/v1/prompt-registry/agents/familiar/resolve",
        params={"tenant_id": "11111111-1111-7111-8111-111111111111"},
    )
    assert resp.status_code == 200
    assert repo.calls == [("11111111-1111-7111-8111-111111111111", "familiar")]


def test_resolve_unconfigured_repo_is_503() -> None:
    resp = _client(None).get("/v1/prompt-registry/agents/familiar/resolve")
    assert resp.status_code == 503
    assert resp.json()["code"] == "PROMPT_REGISTRY_UNAVAILABLE"
