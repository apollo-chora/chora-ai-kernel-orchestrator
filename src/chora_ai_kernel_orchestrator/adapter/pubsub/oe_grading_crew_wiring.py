"""OE grading crew composition root (ADR-172).

Clone of qgen_crew_wiring.py — simpler (no TokenUsageLedger / HITL-decision
writers, no image-render clients), but it DOES wire an AgentDecisionLog writer
(Gate #8) so the per-agent oe_evaluator + oe_moderator decisions hydrate the O+
/o/agents tiles. Bundles every adapter the OE grading crew needs into a single
``OEGradingCrewComponents`` dataclass that ``main.py`` lifespan starts + stops.
Pulls config from env vars per [[secrets-and-env]].

Composed adapters:
  * ``PubSubAgentExecutor`` (ADR-253/254) - the ONLY dispatch path: the crew
    parks and the oe_evaluate / oe_moderate requests leave on the bus. The
    subscriber-only oe_evaluator + oe_moderator agents answer on the
    completion topics.
  * ``ModelArmorGuardrailPort`` — tier-mapped Cloud Model Armor (ADR-152); the
    graph screens the learner answer (pre) + grader comment (post).
  * ``build_oe_grading_crew_graph`` compiled with the SHARED PostgresSaver
    checkpointer (thread_id = submission_id; D6 P1 pod-death resume).
  * ``OEGradingCrewOutboxWriter`` — INSERT submission_completed.v1 (JSON) into
    ai_kernel_outbox_events.
  * ``AgentDecisionLogOutboxWriter`` — INSERT one
    observability.agent_decision.logged.v1 row per agent (oe_evaluator +
    oe_moderator) into ai_kernel_outbox_events (shared autocommit conn).
  * ``OEGradingCrewRunner`` — graph + submission_completed writer +
    AgentDecisionLog writer.
  * ``InboxIdempotencyStore`` — Pub/Sub message dedupe.
  * ``OEGradingCrewSubscriber`` + ``OEGradingCrewPubsubLoop`` — StreamingPull.
  * ``OutboxDispatcher`` — drains ai_kernel_outbox_events (see the
    one-dispatcher-per-process note in main.py: the dispatcher drains ALL rows
    regardless of crew, so qgen's and OE's must not BOTH loop on the table).

Env vars (if any are missing the orchestrator stays up but the OE subscriber does
NOT start; the qgen path is unaffected):

    CHORA_AI_KERNEL_PG_DSN (or _SECRET_ID) — psycopg DSN for chora_ai_kernel
    CHORA_PUBSUB_PROJECT — Pub/Sub host project
    OE_GRADING_CREW_SUBSCRIPTION — submission_requested.v1 subscription
    CHORA_MODELARMOR_PROJECT / _LOCATION / CHORA_ENVIRONMENT — guardrail tiers
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    ModelArmorGuardrailPort,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_decision_outbox_writer import (
    AgentDecisionLogOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.inbox_idempotency import (
    InboxIdempotencyStore,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.oe_grading_crew_loop import (
    OEGradingCrewPubsubLoop,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.oe_grading_crew_publisher import (
    OEGradingCrewOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.oe_grading_crew_subscriber import (
    OEGradingCrewSubscriber,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.reconnecting_connection import (
    ReconnectingAsyncConnection,
)

logger = logging.getLogger(__name__)


async def _connect_with_retry(
    psycopg_mod: Any,
    dsn: str,
    *,
    attempts: int,
    base_delay_s: float,
    max_delay_s: float,
) -> Any:
    """Connect with exponential backoff (cloudsql-proxy cold-start race). Mirrors
    qgen_crew_wiring._connect_with_retry. Raises the LAST exception when all
    attempts fail so the caller fails loud per [[no-stubs-real-wiring]]."""
    last_exc: Exception | None = None
    delay = base_delay_s
    for i in range(1, attempts + 1):
        try:
            conn = await psycopg_mod.AsyncConnection.connect(dsn)
            if i > 1:
                logger.info(
                    "oe_grading_crew_wiring.db_connect.recovered",
                    extra={"attempt": i, "attempts": attempts},
                )
            return conn
        except Exception as exc:  # noqa: BLE001 — cold-start race surface is broad
            last_exc = exc
            if i == attempts:
                break
            logger.warning(
                "oe_grading_crew_wiring.db_connect.retry",
                extra={"attempt": i, "attempts": attempts, "delay_s": delay, "err": f"{exc.__class__.__name__}: {exc}"},
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay_s)
    assert last_exc is not None
    raise last_exc


@dataclass
class OEGradingCrewComponents:
    """All the OE grading crew adapters main.py needs to start/stop together."""

    pubsub_loop: OEGradingCrewPubsubLoop
    outbox_dispatcher: Any  # OutboxDispatcher — lifecycle managed by main.py
    db_conn: Any
    subscriber: OEGradingCrewSubscriber
    runner: Any  # OEGradingCrewRunner
    inbox: InboxIdempotencyStore
    outbox_writer: OEGradingCrewOutboxWriter
    agent_decision_writer: AgentDecisionLogOutboxWriter
    executor: Any  # PubSubAgentExecutor
    graph: Any  # compiled LangGraph
    # ADR-253 D2: the lane that resumes parked runs.
    completion_loop: Any = None
    # The D3a saver's dedicated connection.
    dispatch_conn: Any = None

    async def aclose(self) -> None:  # pragma: no cover — main.py lifespan
        for name, conn in (("db_conn", self.db_conn), ("dispatch_conn", self.dispatch_conn)):
            if conn is None:
                continue
            try:
                await conn.close()
            except Exception:
                logger.exception(
                    "oe_grading_crew_components.db_close_failed",
                    extra={"conn": name},
                )


# The crew name the park ledger and the completion router bind the OE roles to.
OE_CREW = "oe_grading"


async def build_oe_grading_crew_from_env(
    *,
    checkpointer: Any | None,
    screener: Any | None,
    runtime: Any = None,
) -> OEGradingCrewComponents | None:  # pragma: no cover — integration glue; see component-level unit tests
    """Compose every OE grading crew adapter from env vars.

    Async because the shared ``psycopg.AsyncConnection`` is awaited inline so
    adapters receive a live connection. Returns None when env vars are missing
    (orchestrator stays up without the OE path) — mirrors build_qgen_crew_from_env.
    """
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_wiring import (
        OE_DISPATCH_ROLES,
    )

    if checkpointer is None or screener is None:
        logger.info("oe_grading_crew_wiring.skipped: checkpointer or screener missing")
        return None

    nats_url = (os.getenv("NATS_URL") or "").strip()
    if not nats_url:
        logger.info("oe_grading_crew_wiring.skipped: NATS_URL unset")
        return None
    # Provenance label for the outbox envelope (source_project).
    pubsub_project = "chora-ai-kernel-orchestrator"

    from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

    dsn = resolve_dsn()
    if not dsn:
        logger.info("oe_grading_crew_wiring.skipped: CHORA_AI_KERNEL_PG_DSN unset")
        return None

    # Lazy imports to keep the test path SDK-free.
    import psycopg

    from chora_ai_kernel_orchestrator.adapter.pubsub import (
        NatsPublisher,
        OutboxDispatcher,
        PostgresOutboxStore,
    )
    from chora_ai_kernel_orchestrator.orchestrators.oe_grading_crew import (
        build_oe_grading_crew_graph,
    )
    from chora_ai_kernel_orchestrator.orchestrators.oe_grading_crew_runner import (
        OEGradingCrewRunner,
    )

    # Single async psycopg connection shared by the outbox writer + inbox +
    # outbox store, wrapped in a ReconnectingAsyncConnection (§4.1) so a
    # cost-pause/resume or any DB blip self-heals instead of dead-lettering the
    # grading message ([[project_oe_fullmatrix_e2e_2026_06_03]] finding 1). The
    # wrapper reconnects lazily on a closed underlying connection and re-applies
    # autocommit on every reconnect. Cold-start retry (mirrors qgen) is threaded
    # through as the wrapper's connect callable so reconnects also back off.
    async def _connect(d: str) -> Any:
        return await _connect_with_retry(psycopg, d, attempts=5, base_delay_s=2.0, max_delay_s=8.0)

    # Put the OE connection in autocommit mode. This connection is WRITE-ONLY
    # from the dispatcher's perspective: main.py starts exactly ONE outbox drain
    # task per process and — because the qgen crew is enabled — that dispatcher
    # runs on the QGEN connection, NOT this one (the OE OutboxDispatcher below is
    # built but deliberately never started, to avoid the double-dispatch /
    # non-durable-claim hazard [[project_outbox_double_dispatch_dev_prod_2026_05_29]]).
    # With psycopg's default autocommit=False, the submission_completed INSERT +
    # the InboxIdempotencyStore dedupe write would sit in an open transaction that
    # NO committer on this connection ever flushes, so the qgen dispatcher (a
    # SEPARATE connection) never sees the row and the learner's submission is
    # stuck in PENDING_OE_GRADING forever. autocommit=True makes every write
    # durable + cross-connection-visible immediately and matches the
    # InboxIdempotencyStore contract ("each Pub/Sub message gets its own
    # auto-commit"). Unlike qgen, the OE write path has no co-located running
    # dispatcher whose mark_published commit would otherwise flush it.
    db_conn = ReconnectingAsyncConnection(dsn, autocommit=True, connect=_connect)
    # Eager connect (fail loud at startup if the DB is unreachable) + apply
    # autocommit; subsequent reconnects re-apply it automatically.
    await db_conn.connect()

    # Tier-mapped Cloud Model Armor port (ADR-152). The graph screens the learner
    # answer (pre) + grader comment (post). from_env returns None only when the
    # YAML mapping is unreadable — fail loud (skip the OE path).
    guardrail = ModelArmorGuardrailPort.from_env(screener=screener)
    if guardrail is None:
        logger.error(
            "oe_grading_crew_wiring.skipped: ModelArmorGuardrailPort.from_env "
            "returned None (agent-guardrail-mapping.yaml unreadable)"
        )
        return None

    # ---- ADR-253/254: the dispatch lane ---------------------------------
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_wiring import (
        build_transactional_saver,
        require_transactional_saver,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
        PubSubAgentExecutor,
    )

    executor: Any = PubSubAgentExecutor(
        source_project=pubsub_project,
        allowed_roles=OE_DISPATCH_ROLES,
    )
    # D3a: the park and the dispatch outbox row commit together, which needs a
    # checkpointer on a connection this process controls. Built here, not taken
    # from main.py's shared saver (see agent_dispatch_wiring). ADR-254 D5:
    # through the kennel runtime the saver also writes the park ledger row
    # (parked_at, deadline_at) in that same transaction.
    if runtime is not None:
        checkpointer = await runtime.transactional_saver(crew=OE_CREW)
    else:
        checkpointer = await build_transactional_saver(
            dsn=dsn,
            source_project=pubsub_project,
        )
    dispatch_conn = getattr(checkpointer, "_conn", None)
    # Refuses anything that is not the transactional saver. A silent fallback
    # would split the park from the dispatch and make the ratified property
    # quietly false.
    require_transactional_saver(checkpointer)
    logger.info(
        "oe_grading_crew_wiring.dispatch_lane",
        extra={"roles": list(OE_DISPATCH_ROLES)},
    )

    graph = build_oe_grading_crew_graph(
        executor=executor,
        guardrail=guardrail,
        checkpointer=checkpointer,
    )

    outbox_writer = OEGradingCrewOutboxWriter(
        conn=db_conn,
        source_project=pubsub_project,
    )

    # Gate #8 — AgentDecisionLog outbox writer (per-agent O+ /o/agents tile
    # hydration: oe_evaluator + oe_moderator). Shares the SAME autocommit
    # db_conn as the submission_completed writer so each decision row is
    # immediately durable + visible to the single (qgen) outbox dispatcher that
    # drains ALL crews' rows. The runner's _emit_agent_decisions is best-effort
    # per [[feedback-d6-resilience-first-class]].
    agent_decision_writer = AgentDecisionLogOutboxWriter(
        conn=db_conn,
        source_project=pubsub_project,
    )
    # ADR-197 M-B.2 — optional PromptResolver over the shared (reconnecting)
    # conn. The 0007 prompt_override_* tables live in chora_ai_kernel; the
    # repository wraps its SELECT in a tx so the set_config('chora.tenant_id')
    # GUC holds even on this autocommit=True conn. No active override for a
    # (tenant, agent) ⇒ the resolver returns the embedded default ⇒
    # byte-identical grading. The runner self-no-ops per-run when no override
    # applies.
    from chora_ai_kernel_orchestrator.adapter.pg.prompt_override_repository import (
        PostgresPromptOverrideRepository,
    )
    from chora_ai_kernel_orchestrator.domain.prompt_registry import (
        PromptResolver,
    )

    prompt_resolver = PromptResolver(PostgresPromptOverrideRepository(conn=db_conn))
    runner = OEGradingCrewRunner(
        graph=graph,
        publisher=outbox_writer,
        agent_decision_emitter=agent_decision_writer,
        prompt_resolver=prompt_resolver,
    )

    inbox = InboxIdempotencyStore(conn=db_conn)
    subscriber = OEGradingCrewSubscriber(runner=runner, inbox=inbox)
    pubsub_loop = OEGradingCrewPubsubLoop.from_env(subscriber=subscriber)
    if pubsub_loop is None:
        return None

    # ADR-253 D2: the completion lane. Without it a parked run is never
    # resumed, so its absence is fatal, not a degradation: every submission
    # would stop at its first dispatch.
    completion_loop = None
    if runtime is not None:
        # ADR-254 D5: ONE completion loop per process, routed by role. The
        # runtime binds the OE roles to this runner and starts the loop
        # (plus the reaper's DLQ loop) over every registered role.
        runtime.register_lane(
            OE_CREW,
            crew=OE_CREW,
            roles=OE_DISPATCH_ROLES,
            runner=runner,
        )
    else:
        from chora_ai_kernel_orchestrator.adapter.pubsub.agent_completion_loop import (
            AgentCompletionPubsubLoop,
        )
        from chora_ai_kernel_orchestrator.adapter.pubsub.agent_completion_subscriber import (  # noqa: E501
            AgentCompletionSubscriber,
        )

        completion_loop = AgentCompletionPubsubLoop.from_env(
            subscriber=AgentCompletionSubscriber(runner=runner, inbox=inbox),
            agent_roles=OE_DISPATCH_ROLES,
        )
        if completion_loop is None:
            raise RuntimeError(
                "oe_grading_crew_wiring: the agent completion loop could not be "
                "built; every run would park at its first agent dispatch and "
                "never resume"
            )

    # Outbox dispatcher — drains ai_kernel_outbox_events. NOTE: the dispatcher
    # drains ALL pending rows regardless of crew; main.py starts exactly ONE
    # drain task across qgen + OE to avoid the double-dispatch / non-durable-claim
    # hazard ([[project_outbox_double_dispatch_dev_prod_2026_05_29]]).
    publisher = NatsPublisher(url=nats_url)
    store = PostgresOutboxStore(conn=db_conn, worker_id="oe-grading-worker-1")
    dispatcher = OutboxDispatcher(
        store=store,
        publisher=publisher,
        worker_id="oe-grading-worker-1",
    )

    return OEGradingCrewComponents(
        pubsub_loop=pubsub_loop,
        outbox_dispatcher=dispatcher,
        db_conn=db_conn,
        subscriber=subscriber,
        runner=runner,
        inbox=inbox,
        outbox_writer=outbox_writer,
        agent_decision_writer=agent_decision_writer,
        executor=executor,
        graph=graph,
        completion_loop=completion_loop,
        dispatch_conn=dispatch_conn,
    )


__all__ = [
    "OEGradingCrewComponents",
    "build_oe_grading_crew_from_env",
]
