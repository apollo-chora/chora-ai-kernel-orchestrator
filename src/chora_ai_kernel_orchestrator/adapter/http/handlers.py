"""FastAPI handlers for the orchestrator service.

The kennel is a Pub/Sub worker; its HTTP surface exists for the probes and the
two routes the mesh actually admits. RULING A (2026-08-23) deleted the rest:
the ns AuthorizationPolicy allowlist is exactly the four paths below, and every
other route the app served (POST /orchestrate, GET /orchestrate/{id},
GET /crews, GET /agents, /v1/orchestrator/runs/*, the exam-prep coach) was
denied at the mesh, so none of them had a caller that could reach it.

Endpoints:

- GET  /healthz                          liveness
- GET  /readyz                           readiness (kennel runtime verdict)
- POST /v1/orchestrator/weakness/{id}/resume   Growth-Edge HITL resume
- GET  /v1/prompt-registry/agents/*      O+ prompt catalogue read API
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from chora_ai_kernel_orchestrator.adapter.checkpointer import (
    build_checkpointer_from_env,
)
from chora_ai_kernel_orchestrator.adapter.http.prompt_registry_router import (
    build_prompt_registry_router,
)
from chora_ai_kernel_orchestrator.adapter.http.weakness_router import (
    build_weakness_router,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor import ModelArmorGuardrailPort

# --- App factory --------------------------------------------------------------


def build_app(
    *,
    guardrail: Any | None,
    checkpointer: Any | None = None,
) -> FastAPI:
    """Build the FastAPI app with the supplied adapters.

    - guardrail = None means readiness fails 503. It is the Cloud Model Armor
      port (ADR-152) and the source of the screener every crew shares, so an
      absent one is a pod that must not take traffic.
    - checkpointer is surfaced on app.state for the main.py lifespan, which
      opens it and UPSERTs the agent registry.
    """
    app = FastAPI(
        title="Chora AI Kernel Orchestrator",
        version="0.1.0",
        description="Hybrid kernel Python LangGraph orchestrator (Tier 2 D5).",
    )

    # Surface the checkpointer on app.state so the FastAPI lifespan (in
    # main.py) can open / close the PostgresSaver + UPSERT the agent registry
    # without hitting build_app_from_env again.
    app.state.checkpointer = checkpointer
    # Every crew shares the Cloud Model Armor screener. Surface it so
    # main.py's lifespan can hand it to each crew builder.
    app.state.screener = (
        guardrail._screener  # noqa: SLF001 — internal handoff, single owner
        if guardrail is not None and hasattr(guardrail, "_screener")
        else None
    )
    # Growth-Edge HITL resume runner (ADR-205 WS-2). Built in the main.py
    # lifespan when WEAKNESS_ANALYSER_ENABLED; None here so the resume route
    # resolves it at request time and returns 503 until the lifespan wires it.
    app.state.weakness_crew_runner = None
    # ADR-254 D5: set by main.py's lifespan once the kennel runtime (router,
    # reaper, single drain) is up; /readyz gates on it when present.
    app.state.kennel_runtime = None

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok", "time": _now_iso()})

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        # Post-ADR-146 (Model Broker service retired) + ADR-152 (chora-
        # guardrail superseded by Cloud Model Armor SDK-direct calls): the
        # only mandatory adapter is `guardrail` (the Cloud Model Armor port).
        # Per CLAUDE.md section 6 there is no inline fallback for it: fail
        # loud with 503.
        if guardrail is None:
            return JSONResponse(
                {"status": "not_ready", "reason": "guardrail_unconfigured"},
                status_code=503,
            )
        # ADR-254 D5: readiness reflects every lane. main.py's lifespan sets the
        # kennel runtime before the app serves; when it is present, its verdict
        # is the gate (lanes started, drain + reaper alive, completion loop up).
        # A pod whose lanes never started used to report ready and consume
        # nothing. Absent only in unit tests that build the app without the
        # lifespan.
        runtime = getattr(app.state, "kennel_runtime", None)
        if runtime is not None:
            verdict = runtime.readiness()
            if not verdict.get("ready"):
                return JSONResponse(
                    {
                        "status": "not_ready",
                        "reason": "kennel_runtime_not_ready",
                        "detail": list(verdict.get("reasons") or []),
                    },
                    status_code=503,
                )
            return JSONResponse(
                {
                    "status": "ready",
                    "time": _now_iso(),
                    "kennel": {
                        "roles": list(verdict.get("roles") or []),
                        "lanes": verdict.get("lanes") or {},
                    },
                }
            )
        return JSONResponse({"status": "ready", "time": _now_iso()})

    # ADR-205 WS-2: mount the Growth-Edge HITL resume route. The crew runner is
    # built in the main.py lifespan (graph mode), so the route resolves it from
    # app.state.weakness_crew_runner at request time; until then it returns 503
    # (no inline fallback — the weakness path's absence never breaks other routes).
    app.include_router(build_weakness_router())

    # CHO-2368 — mount the prompt-registry read API (O+ catalogue). The repo is
    # built in the main.py lifespan, so the route resolves it from
    # app.state.prompt_catalogue_repo at request time; until then it returns 503.
    app.include_router(build_prompt_registry_router())

    return app


def build_app_from_env() -> FastAPI:
    """Build the app reading adapter config from env vars.

    Per CLAUDE.md section 6 + memory `feedback_no_inline_config`: no inline
    fallback; if env vars are absent, /readyz returns 503.

    Guardrail adapter (per ADR-152, chora-guardrail superseded by Cloud Model
    Armor):
        CHORA_MODELARMOR_PROJECT  -> Model Armor templates project
        CHORA_MODELARMOR_LOCATION -> Model Armor templates location
        CHORA_ENVIRONMENT         -> appended to template names (dev/staging/prod)
        CHORA_AGENT_GUARDRAIL_MAPPING_PATH -> override the bundled YAML

    Checkpointer:
        CHORA_AI_KERNEL_PG_DSN       -> PostgresSaver (else InMemorySaver)
    """
    guardrail = ModelArmorGuardrailPort.from_env()
    checkpointer = build_checkpointer_from_env()
    return build_app(guardrail=guardrail, checkpointer=checkpointer)


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()
