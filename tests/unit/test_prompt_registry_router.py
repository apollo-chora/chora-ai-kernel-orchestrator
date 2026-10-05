"""CHO-2368 prompt-registry read API (the O+ catalogue's orchestrator half).

Mirrors the weakness/agent-registry router test idiom: a bare FastAPI app with
only the router mounted, a hand-rolled fake repo, and the unconfigured path
returning 503 (never 404 - the route is always mounted).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from chora_ai_kernel_orchestrator.adapter.http.prompt_registry_router import (
    build_prompt_registry_router,
)
from chora_ai_kernel_orchestrator.domain.prompt_registry.models import (
    CataloguePlanVersion,
    CatalogueSegmentView,
)

_BASE = CataloguePlanVersion(
    plan_id="p-base",
    plan_code="baseline-qgen_question-v1",
    agent_id="qgen_question",
    version_label="v1",
    kind="baseline",
    status="active",
    created_at="2026-07-27T00:00:00+00:00",
    activated_at="2026-07-27T00:00:00+00:00",
    approved_by=None,
    eval_run_id=None,
)

_OVERRIDE = CataloguePlanVersion(
    plan_id="p-110",
    plan_code="qgen_question-1.1.0",
    agent_id="qgen_question",
    version_label="1.1.0",
    kind="override",
    status="pending_eval",
    created_at="2026-07-27T01:00:00+00:00",
    activated_at=None,
    approved_by=None,
    eval_run_id="eval-42",
)

_SEGMENTS = [
    CatalogueSegmentView(
        segment_id="role", body="role body", locked=False, position=20, content_hash="h1", note="", version=1
    ),
    CatalogueSegmentView(
        segment_id="output_new_mcq",
        body="contract body",
        locked=True,
        position=90,
        content_hash="h2",
        note="Output contract; never override-eligible.",
        version=1,
    ),
]


class _FakeRepo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def list_agent_versions(self, *, agent_id: str) -> list[CataloguePlanVersion]:
        self.calls.append(("list", agent_id))
        return [_BASE, _OVERRIDE] if agent_id == "qgen_question" else []

    async def get_agent_version(
        self, *, agent_id: str, version_label: str
    ) -> tuple[CataloguePlanVersion, list[CatalogueSegmentView]] | None:
        self.calls.append(("get", (agent_id, version_label)))
        if agent_id == "qgen_question" and version_label == "v1":
            return _BASE, _SEGMENTS
        return None


def _client(repo: Any | None) -> TestClient:
    app = FastAPI()
    app.include_router(build_prompt_registry_router(repo=repo))
    return TestClient(app)


def test_list_versions_happy() -> None:
    client = _client(_FakeRepo())
    resp = client.get("/v1/prompt-registry/agents/qgen_question/versions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_id"] == "qgen_question"
    assert body["total"] == 2
    assert [v["version"] for v in body["versions"]] == ["v1", "1.1.0"]
    assert body["versions"][0]["kind"] == "baseline"
    assert body["versions"][1]["eval_run_id"] == "eval-42"


def test_list_versions_unknown_agent_404() -> None:
    client = _client(_FakeRepo())
    resp = client.get("/v1/prompt-registry/agents/nope/versions")
    assert resp.status_code == 404
    assert resp.json()["code"] == "PROMPT_REGISTRY_AGENT_NOT_FOUND"


def test_get_version_happy_with_locked_flags() -> None:
    client = _client(_FakeRepo())
    resp = client.get("/v1/prompt-registry/agents/qgen_question/versions/v1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "v1"
    assert body["kind"] == "baseline"
    segs = body["segments"]
    assert [s["segment_id"] for s in segs] == ["role", "output_new_mcq"]
    assert segs[0]["locked"] is False
    assert segs[1]["locked"] is True
    assert segs[1]["body"] == "contract body"


def test_get_version_unknown_404() -> None:
    client = _client(_FakeRepo())
    resp = client.get("/v1/prompt-registry/agents/qgen_question/versions/9.9.9")
    assert resp.status_code == 404
    assert resp.json()["code"] == "PROMPT_REGISTRY_VERSION_NOT_FOUND"


def test_unconfigured_repo_returns_503_not_404() -> None:
    client = _client(None)
    for path in (
        "/v1/prompt-registry/agents/qgen_question/versions",
        "/v1/prompt-registry/agents/qgen_question/versions/v1",
    ):
        resp = client.get(path)
        assert resp.status_code == 503
        assert resp.json()["code"] == "PROMPT_REGISTRY_UNAVAILABLE"
