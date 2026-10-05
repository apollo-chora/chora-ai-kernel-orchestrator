"""CHO-2368 prompt-registry read API - the O+ catalogue's orchestrator half.

Serves the per-agent version list and per-version segment content out of the
``chora_ai_kernel`` prompt override registry (ADR-197): the seeded ``baseline``
transcriptions plus every ``override`` plan moving through the six-state gate.

Reached ONLY via chora-gateway (``/bff/oplus/prompts/{agent_id}/versions`` under
the OPlusAuthorizedRoles auditor gate); the ai-kernel namespace default-deny
AuthorizationPolicy pins the mesh edge to the gateway service account with
exact paths (see chora-infra authz-allow-gateway.yaml).

Mirrors ``build_weakness_router``: a factory returning an ``APIRouter``; the
repository is built in the FastAPI lifespan, so the router resolves it at
request time:

  * an explicit ``repo`` passed to the factory wins (tests / direct wiring);
  * otherwise it reads ``request.app.state.prompt_catalogue_repo``.

When neither is configured every request returns 503 - never 404 (the route is
always mounted), matching the unconfigured-adapter convention.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...domain.prompt_registry.embedded_versions import EMBEDDED_PROMPT_VERSIONS
from ...domain.prompt_registry.models import (
    CataloguePlanVersion,
    CatalogueSegmentView,
)
from ...domain.prompt_registry.resolver import PromptResolver

logger = logging.getLogger(__name__)


class PromptVersionSummary(BaseModel):
    """One row of the per-agent version list."""

    version: str
    kind: str
    status: str
    plan_code: str
    created_at: str
    activated_at: str | None = None
    approved_by: str | None = None
    eval_run_id: str | None = None


class PromptVersionListResponse(BaseModel):
    agent_id: str
    versions: list[PromptVersionSummary]
    total: int


class PromptSegmentView(BaseModel):
    """One segment of a version, with its display metadata."""

    segment_id: str
    body: str
    locked: bool
    position: int
    content_hash: str | None = None
    note: str = ""


class PromptVersionDetailResponse(BaseModel):
    agent_id: str
    version: str
    kind: str
    status: str
    plan_code: str
    created_at: str
    activated_at: str | None = None
    approved_by: str | None = None
    eval_run_id: str | None = None
    segments: list[PromptSegmentView] = Field(default_factory=list)


class PromptResolveResponse(BaseModel):
    """The PromptResolver verdict for one (tenant, agent) - ADR-197 P3.

    ``embedded`` (empty segments, null version) is a VALID resolution: the
    caller renders its embedded baseline. Consumed by chora-consumption at
    familiar session build to fill the ``prompt_overrides_json`` session-state
    contract.
    """

    agent_id: str
    tenant_id: str = ""
    version: str | None = None
    source: str
    segments: dict[str, str] = Field(default_factory=dict)


def _summary(plan: CataloguePlanVersion) -> PromptVersionSummary:
    return PromptVersionSummary(
        version=plan.version_label,
        kind=plan.kind,
        status=plan.status,
        plan_code=plan.plan_code,
        created_at=plan.created_at,
        activated_at=plan.activated_at,
        approved_by=plan.approved_by,
        eval_run_id=plan.eval_run_id,
    )


def _unavailable() -> JSONResponse:
    return JSONResponse(
        {
            "code": "PROMPT_REGISTRY_UNAVAILABLE",
            "message": "prompt catalogue repository is not configured",
        },
        status_code=503,
    )


def build_prompt_registry_router(repo: Any | None = None) -> APIRouter:
    """Pure factory - no shared mutable state beyond the injected repo."""

    router = APIRouter()

    def _resolve_repo(request: Request) -> Any | None:
        if repo is not None:
            return repo
        return getattr(request.app.state, "prompt_catalogue_repo", None)

    @router.get("/v1/prompt-registry/agents/{agent_id}/versions")
    async def list_versions(
        request: Request,
        agent_id: str = Path(..., min_length=1),
    ) -> Any:
        catalogue = _resolve_repo(request)
        if catalogue is None:
            return _unavailable()
        versions: list[CataloguePlanVersion] = await catalogue.list_agent_versions(agent_id=agent_id)
        if not versions:
            return JSONResponse(
                {
                    "code": "PROMPT_REGISTRY_AGENT_NOT_FOUND",
                    "message": f"no catalogue versions for agent {agent_id!r}",
                },
                status_code=404,
            )
        return PromptVersionListResponse(
            agent_id=agent_id,
            versions=[_summary(p) for p in versions],
            total=len(versions),
        )

    @router.get("/v1/prompt-registry/agents/{agent_id}/resolve")
    async def resolve_agent(
        request: Request,
        agent_id: str = Path(..., min_length=1),
        tenant_id: str = Query(default=""),
    ) -> Any:
        catalogue = _resolve_repo(request)
        if catalogue is None:
            return _unavailable()
        resolver = PromptResolver(repository=catalogue)
        resolved = await resolver.resolve(tenant_id=tenant_id, agent_id=agent_id)
        version = resolved.version
        if version is None and resolved.source == "embedded":
            # Overlay the agent's embedded default (e.g. "v1") so the caller
            # can stamp what actually runs - consumption injects it into the
            # familiar session and the ritual stamp then agrees with the
            # agent's own fallback constant. Unknown agents stay null.
            version = EMBEDDED_PROMPT_VERSIONS.get(agent_id)
        return PromptResolveResponse(
            agent_id=agent_id,
            tenant_id=tenant_id,
            version=version,
            source=resolved.source,
            segments=resolved.segments,
        )

    @router.get("/v1/prompt-registry/agents/{agent_id}/versions/{version}")
    async def get_version(
        request: Request,
        agent_id: str = Path(..., min_length=1),
        version: str = Path(..., min_length=1),
    ) -> Any:
        catalogue = _resolve_repo(request)
        if catalogue is None:
            return _unavailable()
        got: tuple[CataloguePlanVersion, list[CatalogueSegmentView]] | None = await catalogue.get_agent_version(
            agent_id=agent_id, version_label=version
        )
        if got is None:
            return JSONResponse(
                {
                    "code": "PROMPT_REGISTRY_VERSION_NOT_FOUND",
                    "message": f"agent {agent_id!r} has no version {version!r}",
                },
                status_code=404,
            )
        plan, segments = got
        summary = _summary(plan)
        return PromptVersionDetailResponse(
            agent_id=agent_id,
            version=summary.version,
            kind=summary.kind,
            status=summary.status,
            plan_code=summary.plan_code,
            created_at=summary.created_at,
            activated_at=summary.activated_at,
            approved_by=summary.approved_by,
            eval_run_id=summary.eval_run_id,
            segments=[
                PromptSegmentView(
                    segment_id=s.segment_id,
                    body=s.body,
                    locked=s.locked,
                    position=s.position,
                    content_hash=s.content_hash,
                    note=s.note,
                )
                for s in segments
            ],
        )

    return router


__all__ = [
    "PromptSegmentView",
    "PromptVersionDetailResponse",
    "PromptVersionListResponse",
    "PromptVersionSummary",
    "build_prompt_registry_router",
]
