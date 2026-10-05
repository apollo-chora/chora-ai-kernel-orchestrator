"""FE resume route for the Growth-Edge analyser HITL review (ADR-205 / CHO-1973).

``POST /v1/orchestrator/weakness/{upload_id}/resume`` carries the learner's
PANEL-SHAPE review decision — the A+ panel state IS the resume payload (ADR-205
D4, zero free-form reprompt):

    {action: "confirm"|"reiterate",
     edges: [{proposed_edge_id, decision, merge_into_id?, difficulty?}],
     added_struggles: [concept_key], selected_outputs: [output_kind]}

The route reconstructs the DETERMINISTIC thread_id from tenant + upload ALONE (no
run_id — the crew runs one analysis per upload) and drives the checkpointed graph
via the crew runner, returning a job/panel shape: ``awaiting_review`` + a
refreshed panel on ``reiterate`` (the FE re-enters review), ``completed`` (or
``blocked`` / ``failed``) on ``confirm``.

Mirrors ``build_ai_assist_router``: a factory returning an ``APIRouter``. The
crew runner is built in the FastAPI lifespan, so the router resolves it at REQUEST
time:

  * an explicit ``runner`` passed to the factory wins (tests / direct wiring);
  * otherwise it reads ``request.app.state.weakness_crew_runner`` (graph mode).

When neither is configured (single_shot / unconfigured) every request returns
503 — never 404 (the route is always mounted) — so the weakness path's absence
never breaks the qgen / OE / ai-assist routes.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Header, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
    build_weakness_thread_id,
)

logger = logging.getLogger(__name__)

# Bounded review actions (ADR-205 D4) — never free-form.
_ALLOWED_ACTIONS = {"confirm", "reiterate"}


class WeaknessResumeEdge(BaseModel):
    """One bounded per-edge decision, addressing a proposed edge by its panel id."""

    proposed_edge_id: str = Field(min_length=1)
    decision: str  # "accept" | "reject" | "merge"
    merge_into_id: str | None = None
    difficulty: str | None = None  # "easier" | "standard" | "harder"


class WeaknessResumeRequest(BaseModel):
    """The bounded HITL panel state (CHO-1973). Everything is structured — never
    free text (``added_struggles`` are concept_keys, ``selected_outputs`` are
    output kinds; both bounded, not a reprompt)."""

    action: str
    edges: list[WeaknessResumeEdge] = Field(default_factory=list)
    added_struggles: list[str] = Field(default_factory=list)
    selected_outputs: list[str] = Field(default_factory=list)


class WeaknessResumeResponse(BaseModel):
    """Job/panel shape the FE re-enters review on (reiterate) or resolves on
    (confirm)."""

    upload_id: str
    status: str  # "awaiting_review" | "completed" | "blocked" | "failed"
    governance_status: str | None = None
    interrupted: bool = False
    # the refreshed review panel when interrupted (reiterate); None on completion.
    panel: dict[str, Any] | None = None
    # final reviewed edges on confirm.
    edges: list[dict[str, Any]] = Field(default_factory=list)
    # WS-7: orchestrator-produced outputs (study aids / practice test). Since
    # ADR-254 D5 they are dispatched after the publish and ride the
    # weakness.outputs_generated.v1 event, so this list is normally EMPTY on the
    # resume response; kept for the wire shape.
    generated_outputs: list[dict[str, Any]] = Field(default_factory=list)


def _status_for(result: Any) -> str:
    # ADR-254 D5: after a confirm the run parks on its learner-output
    # dispatches (an AGENT park, not a review). The analysis is already
    # durable and the learner is done reviewing: that is "completed" for the
    # FE; the artifacts ride weakness.outputs_generated.v1 whenever they land.
    if getattr(result, "interrupted", False) and not getattr(result, "awaiting_agent", False):
        return "awaiting_review"
    gov = str(getattr(result, "governance_status", "") or "").strip().lower()
    if gov == "blocked":
        return "blocked"
    if gov == "failed":
        return "failed"
    return "completed"  # approved / unknown terminal


def build_weakness_router(*, runner: Any | None = None) -> APIRouter:
    """Build the Growth-Edge resume sub-router.

    ``runner`` (when supplied) binds the route directly; otherwise the runner is
    resolved per-request from ``request.app.state.weakness_crew_runner`` (built in
    the lifespan, graph mode only). Unconfigured ⇒ 503.
    """
    router = APIRouter()

    def _active_runner(request: Request) -> Any | None:
        if runner is not None:
            return runner
        return getattr(request.app.state, "weakness_crew_runner", None)

    @router.post("/v1/orchestrator/weakness/{upload_id}/resume")
    async def resume(
        request: Request,
        body: WeaknessResumeRequest,
        upload_id: str = Path(..., min_length=1),
        x_tenant_id: str = Header(..., alias="X-Tenant-Id"),
    ) -> JSONResponse:
        active = _active_runner(request)
        if active is None:
            return JSONResponse(
                {
                    "error": "weakness_analyser_unconfigured",
                    "detail": "the Growth-Edge analyser crew is not wired in this deployment",
                },
                status_code=503,
            )
        if body.action not in _ALLOWED_ACTIONS:
            return JSONResponse(
                {"error": "invalid_action", "detail": f"action must be one of {sorted(_ALLOWED_ACTIONS)}"},
                status_code=400,
            )
        thread_id = build_weakness_thread_id(tenant_id=x_tenant_id, upload_id=upload_id)
        # Forward the panel-shape decision verbatim (the graph's hitl_review node
        # maps proposed_edge_id → candidate_edge — the orchestrator alone holds it).
        decision = {
            "action": body.action,
            "edges": [e.model_dump(exclude_none=True) for e in body.edges],
            "added_struggles": body.added_struggles,
            "selected_outputs": body.selected_outputs,
        }
        try:
            result = await active.resume(thread_id=thread_id, decision=decision)
        except Exception:  # noqa: BLE001 — surface a clean 502, never leak internals
            logger.exception("weakness_router.resume_failed", extra={"thread_id": thread_id})
            return JSONResponse(
                {"error": "resume_failed", "detail": "graph resume errored"},
                status_code=502,
            )
        awaiting_agent = bool(getattr(result, "awaiting_agent", False))
        return JSONResponse(
            WeaknessResumeResponse(
                upload_id=upload_id,
                status=_status_for(result),
                governance_status=result.governance_status,
                # an agent park is not a review pause for the FE
                interrupted=bool(result.interrupted) and not awaiting_agent,
                panel=None if awaiting_agent else result.review_panel,
                edges=result.edges,
                generated_outputs=result.generated_outputs,
            ).model_dump()
        )

    return router


__all__ = [
    "WeaknessResumeEdge",
    "WeaknessResumeRequest",
    "WeaknessResumeResponse",
    "build_weakness_router",
]
