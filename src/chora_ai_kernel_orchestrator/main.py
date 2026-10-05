"""FastAPI entrypoint for chora-ai-kernel-orchestrator.

Run: ``uvicorn chora_ai_kernel_orchestrator.main:app --host 0.0.0.0 --port 8080``

Startup sequence (ADR-254 D5, the kennel):
    1. Initialise OTLP tracing (Cloud Trace exporter) + the logging handler.
    2. Build the FastAPI app + adapters from env vars (no inline config).
    3. Open the shared Postgres checkpointer (the lanes still on it) and
       UPSERT the agent registry (best-effort, reporting only).
    4. Build the KENNEL RUNTIME once: the park ledger store, the completion
       router, the park reaper (scan + DLQ loops), the single outbox drain.
    5. Build every ENABLED lane against it. A lane that is enabled and cannot
       build aborts startup: a pod that looks ready and consumes nothing is a
       defect, not a degraded mode.
    6. Start the runtime (completion + DLQ loops, reaper scan, drain) FIRST,
       then each lane's request subscriber, marking the lane started so
       /readyz reflects it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
    LazyPostgresSaver,
    close_checkpointer,
)
from chora_ai_kernel_orchestrator.adapter.http.handlers import build_app_from_env
from chora_ai_kernel_orchestrator.adapter.pubsub.kennel_runtime import (
    build_kennel_runtime_from_env,
    require_lane_built,
)
from chora_ai_kernel_orchestrator.observability.logging_bootstrap import (
    configure_logging,
)
from chora_ai_kernel_orchestrator.observability.tracing import init_tracing

logger = logging.getLogger(__name__)

# Handler-before-anything: without this every app-level INFO breadcrumb
# (lifespan started/skipped, consumer acked/skip reasons) is invisible in
# kubectl logs - Python's lastResort emits WARNING+ only (CHO-2368).
configure_logging()
init_tracing()

_TRUTHY = {"1", "true", "yes"}


def _enabled(env_name: str) -> bool:
    return (os.getenv(env_name) or "").strip().lower() in _TRUTHY


async def _build_enabled_lane(name: str, build: Awaitable[Any]) -> Any:  # pragma: no cover - integration glue
    """Build one ENABLED lane, fail loud on any failure (ADR-254 D5)."""
    try:
        components = await build
    except Exception as exc:
        logger.exception("%s.lifespan.startup_failed", name)
        raise RuntimeError(f"{name} is enabled but failed to build: {exc}") from exc
    return require_lane_built(name, components)


async def _stop_quietly(what: str, coro: Awaitable[Any]) -> None:  # pragma: no cover - integration glue
    """Stop one loop during teardown, loudly on failure.

    Teardown must not abort: a loop that refuses to stop cannot be allowed to
    strand the rest of the shutdown, and re-raising here would do exactly that.
    But it must not vanish either. A loop that could not be stopped is why the
    next boot finds a subscription still attached, and `with suppress(Exception)`
    (what this replaces) left nothing behind to connect the two.
    """
    try:
        await coro
    except Exception:
        logger.exception("lifespan.stop_failed", extra={"what": what})


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:  # pragma: no cover - integration glue
    """FastAPI lifespan: runtime + lanes up, fail loud, one drain, then serve."""
    saver = getattr(app.state, "checkpointer", None)
    if isinstance(saver, LazyPostgresSaver):
        try:
            saver.open()
        except Exception:
            logger.exception("checkpointer open failed")

        # Best-effort agent registry UPSERT (reporting only).
        try:
            from chora_ai_kernel_orchestrator.adapter.registry import (
                persist_ai_assist_registry,
            )
            from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

            dsn = resolve_dsn()
            if dsn:
                import psycopg

                with psycopg.connect(dsn) as conn:
                    persist_ai_assist_registry(conn)
        except Exception:
            logger.exception("agent registry UPSERT failed")

    screener = getattr(app.state, "screener", None)

    # ---- ADR-254 D5: the kennel runtime, built ONCE, before any lane ------
    runtime = await build_kennel_runtime_from_env()
    app.state.kennel_runtime = runtime

    # ---- lanes: built against the runtime; enabled + unbuildable = abort ----
    qgen_components: Any = None
    if _enabled("QGEN_CREW_ENABLED"):
        from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_wiring import (
            build_qgen_crew_from_env,
        )

        qgen_components = await _build_enabled_lane(
            "qgen_crew",
            build_qgen_crew_from_env(screener=screener, runtime=runtime),
        )
    else:
        logger.info("qgen_crew.lifespan.disabled (QGEN_CREW_ENABLED off)")

    oe_components: Any = None
    if _enabled("OE_GRADING_CREW_ENABLED"):
        from chora_ai_kernel_orchestrator.adapter.pubsub.oe_grading_crew_wiring import (
            build_oe_grading_crew_from_env,
        )

        oe_components = await _build_enabled_lane(
            "oe_grading_crew",
            build_oe_grading_crew_from_env(
                checkpointer=saver,
                screener=screener,
                runtime=runtime,
            ),
        )
    else:
        logger.info("oe_grading_crew.lifespan.disabled (OE_GRADING_CREW_ENABLED off)")

    weakness_components: Any = None
    if _enabled("WEAKNESS_ANALYSER_ENABLED"):
        from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyser_crew_wiring import (
            build_weakness_analyser_crew_from_env,
        )

        weakness_components = await _build_enabled_lane(
            "weakness_analyser_crew",
            build_weakness_analyser_crew_from_env(screener=screener, runtime=runtime),
        )
        # Expose the crew runner so the HITL resume route (the one surviving
        # non-dispatch HTTP hop, ADR-254 D12) resolves it at request time.
        app.state.weakness_crew_runner = weakness_components.crew_runner
    else:
        logger.info("weakness_analyser_crew.lifespan.disabled (WEAKNESS_ANALYSER_ENABLED off)")

    # ---- ADR-254 D5/D13: the generic single-agent lanes --------------------
    single_agent_lane_components: list[Any] = []
    if runtime is not None:
        from chora_ai_kernel_orchestrator.adapter.pubsub.single_agent_lanes_wiring import (
            build_single_agent_lanes_from_env,
            single_agent_lane_specs_from_env,
        )

        if single_agent_lane_specs_from_env():
            single_agent_lane_components = await _build_enabled_lane(
                "single_agent_lanes",
                build_single_agent_lanes_from_env(runtime=runtime),
            )
        else:
            logger.info("single_agent_lanes.lifespan.disabled (no lane enable flag set)")

    prompt_audit_components: Any = None
    if _enabled("PROMPT_PROMOTION_AUDIT_ENABLED"):
        from chora_ai_kernel_orchestrator.adapter.pubsub.prompt_promotion_audit_wiring import (
            build_prompt_promotion_audit_from_env,
        )

        prompt_audit_components = await _build_enabled_lane(
            "prompt_promotion_audit",
            build_prompt_promotion_audit_from_env(),
        )
        if runtime is not None:
            runtime.register_lane(
                "prompt_promotion_audit",
                crew="prompt_promotion_audit",
                roles=(),
                runner=None,
            )
    else:
        logger.info("prompt_promotion_audit.lifespan.disabled (PROMPT_PROMOTION_AUDIT_ENABLED off)")

    built = [c for c in (qgen_components, oe_components, weakness_components, prompt_audit_components) if c is not None]
    if built and runtime is None:
        raise RuntimeError(
            "kennel runtime could not be built (CHORA_PUBSUB_PROJECT or the DSN is "
            "unset) while a lane is enabled; refusing to start lanes with no drain, "
            "no completion loop and no reaper"
        )

    # ---- prompt catalogue read repo (CHO-2368) -----------------------------
    catalogue_conn = None
    app.state.prompt_catalogue_repo = None
    try:
        from chora_ai_kernel_orchestrator.adapter.pg.prompt_override_repository import (
            PostgresPromptOverrideRepository,
        )

        if prompt_audit_components is not None:
            app.state.prompt_catalogue_repo = PostgresPromptOverrideRepository(conn=prompt_audit_components.db_conn)
            logger.info("prompt_catalogue.lifespan.repo_shared_with_audit_lane")
        else:
            from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

            catalogue_dsn = resolve_dsn()
            if catalogue_dsn:
                import psycopg

                catalogue_conn = await psycopg.AsyncConnection.connect(catalogue_dsn, autocommit=True)
                app.state.prompt_catalogue_repo = PostgresPromptOverrideRepository(conn=catalogue_conn)
                logger.info("prompt_catalogue.lifespan.repo_started_standalone")
            else:
                logger.info("prompt_catalogue.lifespan.skipped: CHORA_AI_KERNEL_PG_DSN unset")
    except Exception:
        logger.exception("prompt_catalogue.lifespan.startup_failed")
        app.state.prompt_catalogue_repo = None

    # ---- start: runtime FIRST, then the request lanes ------------------------
    # The runtime's completion + DLQ loops, reaper scan and the single outbox
    # drain are up before any lane can publish a dispatch, so nothing is ever
    # published before something is listening for its completion.
    if runtime is not None:
        summary = await runtime.start()
        logger.info("kennel_runtime.lifespan.started", extra=summary)

    if qgen_components is not None:
        # ADR-251 D4: resume-on-boot BEFORE the streaming pull, so a resumed
        # drive never races a redelivered duplicate of its own started event.
        # Fail-loud: a sweep failure aborts startup (a pod that cannot resume
        # must not ACK new work while silently stranding old work).
        if qgen_components.resume_sweeper is not None:
            tallies = await qgen_components.resume_sweeper.sweep_and_resume()
            logger.info("qgen_crew.lifespan.resume_sweep", extra=tallies)
        await qgen_components.pubsub_loop.start()
        if runtime is not None:
            runtime.mark_lane_started("qgen")
        logger.info("qgen_crew.lifespan.started")

    if oe_components is not None:
        await oe_components.pubsub_loop.start()
        # Legacy per-lane completion loop: only when no runtime routed it.
        if oe_components.completion_loop is not None:
            await oe_components.completion_loop.start()
        if runtime is not None:
            runtime.mark_lane_started("oe_grading")
        logger.info("oe_grading_crew.lifespan.started")

    if weakness_components is not None:
        await weakness_components.pubsub_loop.start()
        # completions resume through the runtime's single role-routed loop
        if runtime is not None and "weakness_analyser" in runtime.lanes:
            runtime.mark_lane_started("weakness_analyser")
        logger.info("weakness_analyser_crew.lifespan.started")

    for lane in single_agent_lane_components:
        await lane.pubsub_loop.start()
        if runtime is not None and lane.name in runtime.lanes:
            runtime.mark_lane_started(lane.name)
        logger.info("fold_lane.lifespan.started", extra={"lane": lane.name})

    if prompt_audit_components is not None:
        await prompt_audit_components.pubsub_loop.start()
        if runtime is not None:
            runtime.mark_lane_started("prompt_promotion_audit")
        logger.info("prompt_promotion_audit.lifespan.started")

    try:
        yield
    finally:
        # Stop the request lanes first so nothing new parks while the runtime
        # winds down; then the runtime (loops, scan, drain, its connection);
        # then each lane's connections; then the shared checkpointer.
        if qgen_components is not None:
            await _stop_quietly("qgen.pubsub_loop", qgen_components.pubsub_loop.stop())
            # ADR-251 D4: best-effort drain of in-flight background drives
            # before the DB connection closes. A drive that outlives the grace
            # window keeps its registry row and resumes on next boot.
            if qgen_components.acceptance is not None:
                try:
                    await qgen_components.acceptance.wait_idle(timeout=10.0)
                except TimeoutError:
                    # The DESIGNED outcome (ADR-251 D4), and the only one this
                    # message is true of: the drive keeps its registry row and
                    # the resume sweep picks it up on next boot.
                    logger.info("qgen_crew.lifespan.drives_deferred_to_resume_sweep")
                except Exception:
                    # Anything else is not a deferred drive, and reporting it as
                    # one would put a reassuring line in the log for a failure
                    # nobody then looks for.
                    logger.exception("qgen_crew.lifespan.drain_failed")
        if oe_components is not None:
            if oe_components.completion_loop is not None:
                await _stop_quietly("oe.completion_loop", oe_components.completion_loop.stop())
            await _stop_quietly("oe.pubsub_loop", oe_components.pubsub_loop.stop())
        if weakness_components is not None:
            app.state.weakness_crew_runner = None
            # completions resume through the runtime's single loop (stopped below)
            await _stop_quietly("weakness.pubsub_loop", weakness_components.pubsub_loop.stop())
        for lane in single_agent_lane_components:
            await _stop_quietly(f"fold_lane.{lane.name}", lane.pubsub_loop.stop())
        if prompt_audit_components is not None:
            await _stop_quietly(
                "prompt_promotion_audit.pubsub_loop",
                prompt_audit_components.pubsub_loop.stop(),
            )

        if runtime is not None:
            try:
                await runtime.stop()
            except Exception:
                logger.exception("kennel_runtime.lifespan.stop_failed")
            app.state.kennel_runtime = None

        for name, components in (
            ("qgen_crew", qgen_components),
            ("oe_grading_crew", oe_components),
            ("weakness_analyser_crew", weakness_components),
            ("prompt_promotion_audit", prompt_audit_components),
            *((f"fold_lane:{lane.name}", lane) for lane in single_agent_lane_components),
        ):
            if components is None:
                continue
            try:
                await components.aclose()
            except Exception:
                logger.exception("%s.lifespan.aclose_failed", name)

        app.state.prompt_catalogue_repo = None
        if catalogue_conn is not None:
            try:
                await catalogue_conn.close()
            except Exception:
                logger.exception("prompt_catalogue.lifespan.close_failed")

        if isinstance(saver, LazyPostgresSaver):
            close_checkpointer(saver)


app = build_app_from_env()
app.router.lifespan_context = _lifespan


def run() -> None:
    """Console-script entrypoint."""
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")  # noqa: S104


if __name__ == "__main__":  # pragma: no cover
    run()
