"""companion_diagnosis crew composition root (ex weakness analyser wiring), on the bus.

ADR-254 D2/D5/D12 (2026-08-22). ONE path now: the graduated governed StateGraph
(``orchestrators/weakness_analyser_crew.py``), checkpointed on the kennel
runtime's D3a transactional saver (park + dispatch outbox row + park-ledger row
in ONE transaction) and dispatching every model call over Pub/Sub:

    reserve_mana -> safesearch -> extract (companion_extract, by reference)
    -> screen_input -> diagnose (companion_diagnose, task_kind=diagnose)
    -> critic_verify -> screen_output -> HITL_review -> synthesize_edges
    -> publish_analyzed -> emit_evidence -> study_aids -> practice_test
    (companion_diagnose by task_kind, one park each) -> publish_outputs

Gone with the cut (ADR-254 section 6): the single-shot runner, the HTTP
``gke://`` diagnoser, the gateway extract, the in-process text-gateway output
generation, and the ``WEAKNESS_CREW_MODE`` / ``WEAKNESS_DISPATCH_TRANSPORT`` /
``WEAKNESS_DIAGNOSER_ENGINE_RESOURCE`` variables. No model-gateway client is
constructed here any more (deterministic kernel, D5).

Composed adapters:
  * ``PubSubAgentExecutor`` pinned to THIS lane's two roles; an executor allowed
    to dispatch any role would park a run on a topic pair that may not exist.
  * ``PubSubExtractorAdapter`` (companion_extract) + ``PubSubDiagnoserAdapter``
    (companion_diagnose: the Diagnoser port AND the OutputTaskRunner port).
  * ``ModelArmorScreenerAdapter`` (template ``weakness_analyser_crew:strict``
    from the platform mapping, no inline template) + ``CloudVisionSafeSearchAdapter``
    + ``GcsBlobDownloader`` (SafeSearch bytes only) + ``FreeTierMana``.
  * the binary outbox writers: analyzed, outputs_generated, review_pending, the
    D1 evidence emitter; ``InboxIdempotencyStore``; the graph subscriber + loop.
  * the kennel runtime (REQUIRED): the D3a saver with the park ledger, and the
    ONE completion loop per process, routed by role. This lane registers
    ``companion_extract`` + ``companion_diagnose``. The legacy
    ``weakness_diagnose`` registration was dropped on the coordinator's
    "drop it" (2026-08-23) after its request subscription read 0 undelivered:
    the drain window is over and the legacy lane is destroyed with it
    (ADR-254 D13 / G4).

Env vars (per [[secrets-and-env]]; a missing required one returns None and the
lifespan ABORTS the lane as enabled-but-unbuildable):

    CHORA_PUBSUB_PROJECT            Pub/Sub host project (chora-489812)
    WEAKNESS_ANALYSER_SUBSCRIPTION  weakness_doc.uploaded.v1 subscription
    WEAKNESS_DIAGNOSER_MODEL_ID     attribution only (analyzed.v1 model_used);
                                    the agent picks the model through the gateway
    WEAKNESS_PRACTICE_TEST_MAX_QUESTIONS  optional, default 8 (the agent's default)
    WEAKNESS_OUTPUT_PRICE_PRACTICE_TEST / _STUDY_AIDS  panel DISPLAY prices
                                    (unset -> 0/free + loud log; not the authority)
    CHORA_MODELARMOR_PROJECT / _LOCATION / CHORA_ENVIRONMENT  guardrail template
    CHORA_AI_KERNEL_PG_DSN (or _SECRET_ID)  psycopg DSN for chora_ai_kernel
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from chora_ai_kernel_orchestrator.adapter.gcs.weakness_downloader import (
    GcsBlobDownloader,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    ModelArmorGuardrailPort,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_decision_outbox_writer import (
    AgentDecisionLogOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.inbox_idempotency import (
    InboxIdempotencyStore,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.reconnecting_connection import (
    ReconnectingAsyncConnection,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyser_crew_graph_subscriber import (
    WeaknessAnalyserCrewGraphSubscriber,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyser_crew_loop import (
    WeaknessAnalyserCrewPubsubLoop,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyzed_outbox_writer import (
    WeaknessAnalyzedOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_evidence_emitter import (
    WeaknessEvidenceEmitter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_outputs_outbox_writer import (
    WeaknessOutputsGeneratedOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_review_pending_outbox_writer import (
    WeaknessReviewPendingOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.weakness.mana import FreeTierMana
from chora_ai_kernel_orchestrator.adapter.weakness.pubsub_dispatch import (
    ROLE_COMPANION_DIAGNOSE,
    ROLE_COMPANION_EXTRACT,
    PubSubDiagnoserAdapter,
    PubSubExtractorAdapter,
)
from chora_ai_kernel_orchestrator.adapter.weakness.safesearch import (
    LocalSafeSearchAdapter,
)
from chora_ai_kernel_orchestrator.adapter.weakness.screener_adapter import (
    ModelArmorScreenerAdapter,
)
from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew import (
    AGENT_ID,
    DEFAULT_PRACTICE_TEST_MAX_QUESTIONS,
    build_weakness_analyser_graph,
)
from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew_runner import (
    WeaknessAnalyserCrewRunner,
)

logger = logging.getLogger(__name__)

# Attribution only: the analyzed.v1 ``model_used`` field. The agent chooses the
# model through the gateway; the completion does not carry it back.
ENV_DIAGNOSER_MODEL_ID = "WEAKNESS_DIAGNOSER_MODEL_ID"
_DEFAULT_DIAGNOSER_MODEL_ID = "gemini-2.5-pro"

# ADR-254 D6 addendum: practice_test ``max_questions`` (default 8).
ENV_PRACTICE_TEST_MAX_QUESTIONS = "WEAKNESS_PRACTICE_TEST_MAX_QUESTIONS"

# CHO-1973: metered Growth-Edge output prices for the review panel (no inline
# config). Free outputs (focused_dose / familiar_coaching) are never priced here.
ENV_PRICE_PRACTICE_TEST = "WEAKNESS_OUTPUT_PRICE_PRACTICE_TEST"
ENV_PRICE_STUDY_AIDS = "WEAKNESS_OUTPUT_PRICE_STUDY_AIDS"
_PRICED_OUTPUTS = (("practice_test", ENV_PRICE_PRACTICE_TEST), ("study_aids", ENV_PRICE_STUDY_AIDS))

# Single outbox worker id (parity with the OE/qgen convention; main.py runs
# exactly one drain task per process).
_OUTBOX_WORKER_ID = "weakness-analyser-worker-1"

# The crew name the park ledger and the completion router bind this lane's
# roles to (ADR-254 D5). Renamed with the crew at the module rename.
WEAKNESS_CREW = "weakness_analyser"

# The roles this kennel DISPATCHES on (the executor is pinned to exactly these)
# and the roles whose COMPLETIONS resume this crew's runs. They are the same
# pair: the legacy weakness_diagnose registration was dropped once its drain
# window closed (2026-08-23), so nothing is bound that is never dispatched.
DISPATCH_ROLES: tuple[str, ...] = (ROLE_COMPANION_EXTRACT, ROLE_COMPANION_DIAGNOSE)
COMPLETION_ROLES: tuple[str, ...] = DISPATCH_ROLES


async def _connect_with_retry(
    psycopg_mod: Any,
    dsn: str,
    *,
    attempts: int,
    base_delay_s: float,
    max_delay_s: float,
) -> Any:  # pragma: no cover - live DB connect; mirrors oe/qgen, integration-tested
    """Connect with exponential backoff (cloudsql-proxy cold-start race). Raises
    the LAST exception when all attempts fail so the caller fails loud."""
    last_exc: Exception | None = None
    delay = base_delay_s
    for i in range(1, attempts + 1):
        try:
            conn = await psycopg_mod.AsyncConnection.connect(dsn)
            if i > 1:
                logger.info(
                    "weakness_analyser_wiring.db_connect.recovered",
                    extra={"attempt": i, "attempts": attempts},
                )
            return conn
        except Exception as exc:  # noqa: BLE001 - cold-start race surface is broad
            last_exc = exc
            if i == attempts:
                break
            logger.warning(
                "weakness_analyser_wiring.db_connect.retry",
                extra={"attempt": i, "attempts": attempts, "delay_s": delay, "err": f"{exc.__class__.__name__}: {exc}"},
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay_s)
    assert last_exc is not None
    raise last_exc


@dataclass
class WeaknessAnalyserCrewComponents:
    """All the companion_diagnosis adapters main.py starts/stops together.

    ``crew_runner`` backs the FE HITL resume route (the one non-dispatch HTTP
    hop that survives, ADR-254 D12) and the completion router binding.
    """

    pubsub_loop: WeaknessAnalyserCrewPubsubLoop
    outbox_dispatcher: Any  # OutboxDispatcher (lifecycle managed by main.py)
    db_conn: Any
    subscriber: WeaknessAnalyserCrewGraphSubscriber
    publisher: WeaknessAnalyzedOutboxWriter
    inbox: InboxIdempotencyStore
    crew_runner: WeaknessAnalyserCrewRunner
    graph: Any  # compiled LangGraph StateGraph

    async def aclose(self) -> None:  # pragma: no cover - main.py lifespan
        if self.db_conn is not None:
            try:
                await self.db_conn.close()
            except Exception:
                logger.exception("weakness_analyser_components.db_close_failed")


async def build_weakness_analyser_crew_from_env(
    *,
    screener: Any | None = None,
    runtime: Any = None,
) -> WeaknessAnalyserCrewComponents | None:
    """Compose the companion_diagnosis crew from env vars.

    Returns None when a required input is missing (the lifespan then ABORTS:
    an enabled lane that cannot be built is a defect, never a silent skip).
    ``runtime`` is the kennel runtime (ADR-254 D5): the D3a saver with the park
    ledger, and the completion router this lane registers its roles on; it is
    REQUIRED, because without it a parked run could never be resumed.
    """
    nats_url = (os.getenv("NATS_URL") or "").strip()
    if not nats_url:
        logger.info("weakness_analyser_wiring.skipped: NATS_URL unset")
        return None
    # Provenance label for the outbox envelope (source_project).
    pubsub_project = "chora-ai-kernel-orchestrator"
    if runtime is None:
        raise RuntimeError(
            "weakness_analyser_wiring: the kennel runtime is required (ADR-254 D5); "
            "without it the lane would park runs nothing resumes"
        )
    if screener is None:
        # ADR-250 D1: a workload that cannot resolve its guardrail does not dispatch.
        logger.error("weakness_analyser_wiring.skipped: Model Armor screener missing")
        return None

    # Resolve the strict Model Armor template (no inline config). Fail loud when
    # the YAML mapping is unreadable: never a bare template.
    guardrail_template = _resolve_weakness_guardrail_template(screener)
    if not guardrail_template:
        logger.error(
            "weakness_analyser_wiring.skipped: weakness_analyser_crew:strict "
            "Model Armor template unresolved (agent-guardrail-mapping.yaml unreadable)"
        )
        return None

    diagnoser_model_id = (os.getenv(ENV_DIAGNOSER_MODEL_ID) or "").strip() or _DEFAULT_DIAGNOSER_MODEL_ID
    practice_test_max_questions = _resolve_practice_test_max_questions()

    from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

    dsn = resolve_dsn()
    if not dsn:
        logger.info("weakness_analyser_wiring.skipped: CHORA_AI_KERNEL_PG_DSN unset")
        return None

    return await _build_graph_live(  # pragma: no cover - live SDK/DB path
        pubsub_project=pubsub_project,
        screener=screener,
        diagnoser_model_id=diagnoser_model_id,
        guardrail_template=guardrail_template,
        dsn=dsn,
        runtime=runtime,
        practice_test_max_questions=practice_test_max_questions,
        nats_url=nats_url,
    )


async def _build_graph_live(  # pragma: no cover - live SDK/DB glue; assembly unit-tested
    *,
    pubsub_project: str,
    screener: Any,
    diagnoser_model_id: str,
    guardrail_template: str,
    dsn: str,
    runtime: Any,
    nats_url: str,
    practice_test_max_questions: int,
) -> WeaknessAnalyserCrewComponents | None:
    # Lazy imports to keep the test path SDK-free (psycopg).
    import psycopg

    from chora_ai_kernel_orchestrator.adapter.pubsub import (
        NatsPublisher,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_wiring import (
        require_transactional_saver,
    )

    async def _connect(d: str) -> Any:
        return await _connect_with_retry(psycopg, d, attempts=5, base_delay_s=2.0, max_delay_s=8.0)

    # Autocommit + ReconnectingAsyncConnection (the OE pattern): the analyzed /
    # outputs / review_pending INSERTs and the inbox dedupe write must be
    # immediately durable + visible to the single outbox drain main.py runs.
    db_conn = ReconnectingAsyncConnection(dsn, autocommit=True, connect=_connect)
    await db_conn.connect()  # eager: fail loud at startup if the DB is unreachable

    out_publisher = NatsPublisher(url=nats_url)

    # D3a + ADR-254 D5: the park, its dispatch outbox row and its park-ledger
    # row commit TOGETHER on a connection this process controls; the runtime
    # builds that saver per crew. Refuse anything else: a silent fallback would
    # make the ratified atomicity quietly untrue.
    checkpointer = require_transactional_saver(
        await runtime.transactional_saver(crew=WEAKNESS_CREW),
        lane="growth edge crew",
    )

    components = _assemble_graph_components(
        pubsub_project=pubsub_project,
        db_conn=db_conn,
        screener=screener,
        checkpointer=checkpointer,
        diagnoser_model_id=diagnoser_model_id,
        guardrail_template=guardrail_template,
        out_publisher=out_publisher,
        runtime=runtime,
        practice_test_max_questions=practice_test_max_questions,
    )
    if components is None:
        await db_conn.close()
    return components


def _resolve_weakness_guardrail_template(screener: Any) -> str | None:
    """Resolve the full guardrail template name for the crew by
    REUSING the platform tier-mapping resolver (ADR-152): the canonical crew
    agent id ``AGENT_ID`` (``weakness_analyser_crew``, the SAME id the screener
    adapter screens under) -> tier=strict -> full template resource name.
    Returns None when the YAML mapping is unreadable so the caller fails loud
    rather than passing a bare/inline template ([[feedback_no_inline_config]])."""
    port = ModelArmorGuardrailPort.from_env(screener=screener)
    if port is None:
        return None
    return port.resolver.template_for(AGENT_ID)


def _resolve_practice_test_max_questions() -> int:
    """``WEAKNESS_PRACTICE_TEST_MAX_QUESTIONS``: unset -> the agent's default 8;
    a non-integer or non-positive value RAISES (a typo must not silently pick
    the default and look like a working override)."""
    raw = (os.getenv(ENV_PRACTICE_TEST_MAX_QUESTIONS) or "").strip()
    if not raw:
        return DEFAULT_PRACTICE_TEST_MAX_QUESTIONS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"weakness_analyser_wiring: {ENV_PRACTICE_TEST_MAX_QUESTIONS}={raw!r} is not an integer"
        ) from exc
    if value < 1:
        raise ValueError(f"weakness_analyser_wiring: {ENV_PRACTICE_TEST_MAX_QUESTIONS}={raw!r} must be >= 1")
    return value


def _resolve_output_prices() -> dict[str, int]:
    """Resolve the metered Growth-Edge output prices from env (no inline config).

    Free outputs (focused_dose / familiar_coaching) are never priced. An unset /
    blank / non-integer price is logged LOUDLY + OMITTED (the panel then encodes
    0 -> consumption treats the kind as free). The TRUE price authority is the
    upload-door mana reservation (chora-consumption PricePlanResolver, ADR-178);
    this env table is the panel's DISPLAY price only.
    """
    prices: dict[str, int] = {}
    for kind, env in _PRICED_OUTPUTS:
        raw = (os.getenv(env) or "").strip()
        if not raw:
            logger.warning(
                "weakness_analyser_wiring.output_price_unset",
                extra={"kind": kind, "env": env},
            )
            continue
        try:
            prices[kind] = int(raw)
        except ValueError:
            logger.error(
                "weakness_analyser_wiring.output_price_invalid",
                extra={"kind": kind, "env": env, "raw": raw},
            )
    return prices


def _assemble_graph_components(
    *,
    pubsub_project: str,
    db_conn: Any,
    screener: Any,
    checkpointer: Any,
    diagnoser_model_id: str,
    guardrail_template: str,
    out_publisher: Any,
    runtime: Any,
    practice_test_max_questions: int = DEFAULT_PRACTICE_TEST_MAX_QUESTIONS,
) -> WeaknessAnalyserCrewComponents | None:
    """Assemble the crew from already-connected live deps + resolved config.
    Pure of DB-connect / SDK-client construction (those live in
    ``_build_graph_live``) so the wiring is unit-testable with fakes.
    """
    if runtime is None:
        raise RuntimeError(
            "weakness_analyser_wiring: the kennel runtime is required to bind the "
            "completion roles; a run parked on them could never be resumed"
        )
    from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
        PubSubAgentExecutor,
    )

    # Pinned to THIS lane's roles. An executor allowed to dispatch any role
    # would park a run on a topic pair that may not exist, and a run parked
    # there can never be resumed by anything.
    executor = PubSubAgentExecutor(
        source_project=pubsub_project,
        allowed_roles=set(DISPATCH_ROLES),
    )
    extractor = PubSubExtractorAdapter(executor=executor)
    # ONE adapter fills both the Diagnoser port (task_kind=diagnose) and the
    # OutputTaskRunner port (study_aids / practice_test): same role, same agent.
    diagnoser = PubSubDiagnoserAdapter(executor=executor, model_id=diagnoser_model_id)

    screener_port = ModelArmorScreenerAdapter(screener=screener, template_name=guardrail_template)
    safesearch = LocalSafeSearchAdapter.from_env()
    downloader = GcsBlobDownloader()
    # WS-4 cutover seam: swap ``FreeTierMana()`` -> ``PaidMana(wallet=...)`` for
    # the upload-driven premium path once the mana-wallet adapter lands; the
    # graph already refunds on every fail-loud / BLOCK node.
    mana = FreeTierMana()
    publisher = WeaknessAnalyzedOutboxWriter(conn=db_conn, source_project=pubsub_project)
    outputs_publisher = WeaknessOutputsGeneratedOutboxWriter(conn=db_conn, source_project=pubsub_project)
    # WS-3 (CHO-1955): D1 accountability evidence for the diagnoser's decision,
    # ONE observability.agent_decision.logged.v1 row per analysis through the
    # shared AgentDecisionLogOutboxWriter + the single ai_kernel_outbox_events drain.
    evidence_emitter = WeaknessEvidenceEmitter(
        agent_decision_emitter=AgentDecisionLogOutboxWriter(conn=db_conn, source_project=pubsub_project)
    )

    graph = build_weakness_analyser_graph(
        mana=mana,
        downloader=downloader,
        extractor=extractor,
        safesearch=safesearch,
        screener=screener_port,
        diagnoser=diagnoser,
        publisher=publisher,
        task_runner=diagnoser,
        outputs_publisher=outputs_publisher,
        evidence_emitter=evidence_emitter,
        checkpointer=checkpointer,
        practice_test_max_questions=practice_test_max_questions,
    )

    # CHO-1973 Wave A: the bounded HITL review panel, emitted on the interrupt
    # via the review_pending binary outbox writer (same drain machinery).
    output_prices = _resolve_output_prices()
    review_pending_publisher = WeaknessReviewPendingOutboxWriter(conn=db_conn, source_project=pubsub_project)
    crew_runner = WeaknessAnalyserCrewRunner(
        graph=graph,
        output_prices=output_prices,
        review_pending_publisher=review_pending_publisher,
    )
    inbox = InboxIdempotencyStore(conn=db_conn)

    # ADR-254 D5: ONE completion loop per process, routed by role. The runtime
    # binds this crew's roles to the runner and starts the loop (plus the
    # reaper's DLQ loop) over every registered role.
    runtime.register_lane(
        WEAKNESS_CREW,
        crew=WEAKNESS_CREW,
        roles=COMPLETION_ROLES,
        runner=crew_runner,
    )

    subscriber = WeaknessAnalyserCrewGraphSubscriber(
        crew_runner=crew_runner,
        inbox=inbox,
        review_pending_publisher=review_pending_publisher,
    )
    pubsub_loop = WeaknessAnalyserCrewPubsubLoop.from_env(subscriber=subscriber)
    if pubsub_loop is None:
        # CHORA_PUBSUB_PROJECT unset cannot happen past the guard above, but
        # never start a half-wired loop.
        return None

    # OutboxDispatcher drains ai_kernel_outbox_events; main.py starts exactly one
    # drain task across qgen/OE/this (two on the same table double-publish).
    from chora_ai_kernel_orchestrator.adapter.pubsub import (
        OutboxDispatcher,
        PostgresOutboxStore,
    )

    store = PostgresOutboxStore(conn=db_conn, worker_id=_OUTBOX_WORKER_ID)
    dispatcher = OutboxDispatcher(store=store, publisher=out_publisher, worker_id=_OUTBOX_WORKER_ID)

    return WeaknessAnalyserCrewComponents(
        pubsub_loop=pubsub_loop,
        outbox_dispatcher=dispatcher,
        db_conn=db_conn,
        subscriber=subscriber,
        publisher=publisher,
        inbox=inbox,
        crew_runner=crew_runner,
        graph=graph,
    )


__all__ = [
    "COMPLETION_ROLES",
    "DISPATCH_ROLES",
    "ENV_DIAGNOSER_MODEL_ID",
    "ENV_PRACTICE_TEST_MAX_QUESTIONS",
    "WEAKNESS_CREW",
    "WeaknessAnalyserCrewComponents",
    "build_weakness_analyser_crew_from_env",
]
