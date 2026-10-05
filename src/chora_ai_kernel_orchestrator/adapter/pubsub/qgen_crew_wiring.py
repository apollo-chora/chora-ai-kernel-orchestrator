"""qgen crew composition root (ADR-254 D2: the qgen lanes on the bus).

Bundles every adapter the qgen crew needs into a single
``QGenCrewComponents`` dataclass that ``main.py`` lifespan starts +
stops together. Pulls config from env vars per [[secrets-and-env]].

Composed adapters:

* ``QGenDispatchAdapter`` over ``PubSubAgentExecutor``: every agent call
  (qgen_generate / qgen_critique / qgen_render) is a Pub/Sub dispatch that
  PARKS the run; the kennel runtime's completion loop resumes it. No HTTP
  executor, no engine resource: the lanes are the transport.
* ``ModelArmorGuardrailPort``: tier-mapped Cloud Model Armor guardrail
  port (ADR-169). Resolves the per-agent template tier from
  ``agent-guardrail-mapping.yaml`` (qgen_question + qgen_critic -> balanced);
  the qgen_crew nodes collapse its ``ScreenResult`` into the
  ``GuardrailResult`` they consume.
* ``build_qgen_crew_graph`` + ``build_image_regen_graph`` compiled with the
  runtime's TransactionalDispatchSaver for crew ``qgen`` (ADR-253 D3a: the
  park and the dispatch outbox row commit in ONE transaction).
* ``KrokiClient`` + ``GcsImageUploadAdapter``: Mermaid renders in-process,
  scene renders are qgen_render dispatches whose gs:// the kennel SIGNS.
* ``QGenCrewTerminalOutboxWriter``: INSERT into ai_kernel_outbox_events.
* ``QGenCrewRunner`` / ``QGenBatchRunner`` / ``ImageRegenRunner`` behind
  ``QGenRunnerRouter``, fronted by ``DurableQGenAcceptance`` (ACK after the
  durable registration; drives and settles run as tracked background work).
* ``InboxIdempotencyStore``: Pub/Sub message dedupe.
* ``QGenCrewSubscriber`` + ``QGenCrewPubsubLoop``: the started.v1 lane.
* ``OutboxDispatcher`` task: drains ai_kernel_outbox_events.
* the lane is registered on the kennel runtime (``register_lane("qgen",
  roles=QGEN_LANE_ROLES, runner=acceptance)``) so completions on the three
  roles resume the parked job through the acceptance.

Env vars (all required for the qgen path to come online; QGEN_CREW_ENABLED
is the gate main.py reads, and an enabled lane that cannot build aborts the
lifespan):

    CHORA_AI_KERNEL_PG_DSN        psycopg DSN for chora_ai_kernel
    CHORA_PUBSUB_PROJECT          Pub/Sub host project (also the dispatch
                                  envelope's source_project)
    QGEN_CREW_SUBSCRIPTION        full subscription resource (or short name;
                                  default
                                  chora-ai-kernel-orchestrator.qgen-crew-ai-assist-started)
    CHORA_MODELARMOR_PROJECT      Model Armor templates project
                                  (ModelArmorGuardrailPort.from_env)
    CHORA_MODELARMOR_LOCATION     Model Armor templates location
                                  (ModelArmorGuardrailPort.from_env)
    CHORA_ENVIRONMENT             appended to template names
                                  (ModelArmorGuardrailPort.from_env)
    CHORA_AGENT_GUARDRAIL_MAPPING_PATH  override the bundled tier YAML
                                  (ModelArmorGuardrailPort.from_env)
    KROKI_ENDPOINT / QGEN_IMAGE_GCS_BUCKET  the two image clients (optional:
                                  unset => render is a no-op until a candidate
                                  carries image_specs, then fail-loud)

Per ADR-169 the bare ``ARMOR_TEMPLATE_AI_ASSIST`` env var is RETIRED; the
per-agent template tier is resolved from agent-guardrail-mapping.yaml.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from chora_ai_kernel_orchestrator.adapter.gcs import (
    GcsImageUploadAdapter,
)
from chora_ai_kernel_orchestrator.adapter.gcs.weakness_downloader import (
    GcsBlobDownloader,
)
from chora_ai_kernel_orchestrator.adapter.kroki import (
    KrokiClient,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    ModelArmorGuardrailPort,
)
from chora_ai_kernel_orchestrator.adapter.postgres.kernel_sweeper import (
    SWEEPER_SESSION_SETTINGS,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_decision_outbox_writer import (
    AgentDecisionLogOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.hitl_decision_outbox_writer import (
    HITLDecisionOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.inbox_idempotency import (
    InboxIdempotencyStore,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_loop import (
    QGenCrewPubsubLoop,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_publisher import (
    QGenCrewTerminalOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_subscriber import (
    QGenCrewSubscriber,
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
    """Connect with exponential backoff.

    Closes the cold-start race where the cloudsql-proxy sidecar isn't yet
    listening when the orchestrator's lifespan fires. Previously
    surfaced as ``qgen_crew.lifespan.startup_failed`` once on first pod
    start; the workaround was ``kubectl delete pod`` to retry on a warm
    node. Mirrors chora-creation's Go-side
    ``CHORA_BOOTSTRAP_TIMEOUT_SECONDS=90`` retry semantics.

    Raises the LAST exception when all attempts fail so the caller can
    NACK / fail-loud per [[no-stubs-real-wiring]].
    """
    last_exc: Exception | None = None
    delay = base_delay_s
    for i in range(1, attempts + 1):
        try:
            conn = await psycopg_mod.AsyncConnection.connect(dsn)
            if i > 1:
                logger.info(
                    "qgen_crew_wiring.db_connect.recovered",
                    extra={"attempt": i, "attempts": attempts},
                )
            return conn
        except Exception as exc:  # noqa: BLE001 — cold-start race surface area is broad
            last_exc = exc
            if i == attempts:
                break
            logger.warning(
                "qgen_crew_wiring.db_connect.retry",
                extra={
                    "attempt": i,
                    "attempts": attempts,
                    "delay_s": delay,
                    "err": f"{exc.__class__.__name__}: {exc}",
                },
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay_s)
    assert last_exc is not None
    raise last_exc


def build_qgen_db_conn(psycopg_mod: Any, dsn: str) -> ReconnectingAsyncConnection:
    """The shared qgen connection, wrapped so a DB restart self-heals.

    This connection is shared by the InboxIdempotencyStore, every outbox writer,
    the prompt-override repo AND the PostgresOutboxStore. Critically, main.py
    starts exactly ONE outbox drain task per process and -- because the qgen crew
    is enabled -- that dispatcher runs HERE, not on the OE connection (see
    oe_grading_crew_wiring: the OE dispatcher is built but deliberately never
    started). So a dead connection on this object strands EVERY orchestrator
    producer's events, not just qgen's.

    Live incident 2026-07-23: a Cloud SQL restart sent ``AdminShutdown``, and the
    raw ``AsyncConnection`` this used to hold stayed dead for the pod's whole
    lifetime -- ``drain_once_failed`` every 2s, 0 successful drains, while
    ``/readyz`` returned 200 the entire time. Only a manual ``rollout restart``
    recovered it. ``oe_grading_crew_wiring`` and ``weakness_analyser_crew_wiring``
    already wrapped the identical pattern; qgen was the one left raw.

    Cold-start retry is threaded through as the wrapper's connect callable so a
    RECONNECT absorbs the same cloudsql-proxy race a cold boot does.

    autocommit is deliberately left at psycopg's default (OFF).
    ``PostgresOutboxStore.fetch_pending`` releases its read transaction with an
    explicit rollback on the empty path precisely because this connection runs
    autocommit=OFF; turning it on here would silently retire that contract.
    """

    async def _connect(d: str) -> Any:
        return await _connect_with_retry(psycopg_mod, d, attempts=5, base_delay_s=2.0, max_delay_s=8.0)

    # Sweeper mode, EXPLICIT (G2): InflightRegistry's boot sweep resumes every
    # unfinished assist across every tenant, so this connection opts in to the
    # kernel-sweeper arm of the 0055 policy instead of leaning on an unset
    # tenant GUC matching all rows. Re-applied on every reconnect by the wrapper.
    return ReconnectingAsyncConnection(
        dsn,
        connect=_connect,
        session_settings=SWEEPER_SESSION_SETTINGS,
    )


# ---------------------------------------------------------------------------
# Composed components
# ---------------------------------------------------------------------------


@dataclass
class QGenCrewComponents:
    """All the qgen crew adapters main.py needs to start/stop together."""

    pubsub_loop: QGenCrewPubsubLoop
    outbox_dispatcher: Any  # OutboxDispatcher — lifecycle managed by main.py
    db_conn: Any
    # Exposed for introspection + tests:
    subscriber: QGenCrewSubscriber
    runner: Any  # QGenCrewRunner — domain object, no need to type strictly
    inbox: InboxIdempotencyStore
    outbox_writer: QGenCrewTerminalOutboxWriter
    # Gate #8 — AgentDecisionLog outbox writer. Exposed for the
    # composition-root smoke test that asserts the runner has its
    # emitter wired (runner._agent_decision_emitter is not None).
    agent_decision_writer: AgentDecisionLogOutboxWriter
    # Human-Oversight gate writer — emits the HITL gate event (D4) that feeds
    # the O+ Human-Oversight queue when a terminal escalates (max-retries-
    # exhausted quality_warning). Shares the same DB connection so the gate
    # row + terminal publish land in one atomic unit (D6 P1).
    hitl_decision_writer: HITLDecisionOutboxWriter
    executor: Any  # QGenDispatchAdapter over PubSubAgentExecutor (ADR-254 D2)
    graph: Any  # compiled LangGraph
    # ADR-251 D4 (CHO-2398) - durable acceptance + boot resume. main.py runs
    # resume_sweeper.sweep_and_resume() BEFORE pubsub_loop.start() and awaits
    # acceptance.wait_idle(best-effort) at shutdown.
    acceptance: Any = None
    resume_sweeper: Any = None
    inflight_registry: Any = None

    async def aclose(self) -> None:  # pragma: no cover — main.py lifespan
        """Close the held DB connection. Caller stops loop + dispatcher
        first."""
        if self.db_conn is not None:
            try:
                await self.db_conn.close()
            except Exception:
                logger.exception("qgen_crew_components.db_close_failed")


async def build_qgen_crew_from_env(
    *,
    screener: Any | None,
    runtime: Any,
) -> QGenCrewComponents | None:  # pragma: no cover, integration glue; see component-level unit tests
    """Compose every qgen crew adapter from env vars.

    Async because the shared ``psycopg.AsyncConnection`` is awaited
    inline so adapters receive a live connection (not the unawaited
    coroutine that `connect(dsn)` returns), otherwise downstream
    `self._conn.cursor()` calls crash with ``AttributeError: 'coroutine'
    object has no attribute 'cursor'`` (regression caught 2026-05-17
    post-Gate-#7+#8 rollout).

    Returns:
        QGenCrewComponents when ALL required env vars are set + the
        Cloud Model Armor screener + the kennel runtime are supplied (the
        orchestrator's main lifespan).
        None when env vars are missing (main.py's require_lane_built then
        aborts an ENABLED lane loudly).

    Raises:
        RuntimeError when the kennel runtime is absent: the qgen lanes have
        no HTTP fallback (ADR-254 D2), so a qgen crew without the runtime's
        dispatch saver + completion loop could park and never resume.
    """
    if runtime is None:
        raise RuntimeError(
            "qgen_crew_wiring: the kennel runtime is required (CHORA_PUBSUB_PROJECT "
            "+ the chora_ai_kernel DSN); the qgen lanes dispatch on Pub/Sub and "
            "have no HTTP fallback (ADR-254 D2)"
        )
    if screener is None:
        logger.info("qgen_crew_wiring.skipped: screener missing")
        return None

    nats_url = (os.getenv("NATS_URL") or "").strip()
    if not nats_url:
        logger.info("qgen_crew_wiring.skipped: NATS_URL unset")
        return None
    # Provenance label for the outbox envelope (source_project). The cloud
    # project id is gone; the service name is the source now.
    pubsub_project = "chora-ai-kernel-orchestrator"

    # Resolve DSN via the same helper main.py uses for the checkpointer.
    from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

    dsn = resolve_dsn()
    if not dsn:
        logger.info("qgen_crew_wiring.skipped: CHORA_AI_KERNEL_PG_DSN unset")
        return None

    # Lazy import to keep the test path SDK-free.
    import psycopg

    from chora_ai_kernel_orchestrator.adapter.pubsub import (
        NatsPublisher,
        OutboxDispatcher,
        PostgresOutboxStore,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_wiring import (
        require_transactional_saver,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
        PubSubAgentExecutor,
    )
    from chora_ai_kernel_orchestrator.orchestrators.image_regen_graph import (
        build_image_regen_graph,
    )
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
        build_qgen_crew_graph,
    )
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
        ImageRegenRunner,
        QGenBatchRunner,
        QGenCrewRunner,
        QGenRunnerRouter,
    )
    from chora_ai_kernel_orchestrator.orchestrators.qgen_dispatch import (
        QGEN_LANE_ROLES,
        QGenDispatchAdapter,
    )
    from chora_ai_kernel_orchestrator.orchestrators.testset_composer import (
        ENV_TESTSET_COMPOSE_ENABLED,
    )

    # ADR-254 D2: the three qgen lanes. The adapter maps the crew's agent ids
    # (qgen_question / qgen_critic / the renderer) onto the lane roles and
    # ships the ADK session-state shape the Go agents merge verbatim; the
    # inner executor refuses any role the lanes do not serve BEFORE a park.
    executor = QGenDispatchAdapter(PubSubAgentExecutor(source_project=pubsub_project, allowed_roles=QGEN_LANE_ROLES))
    # ADR-253 D3a: the park + the dispatch outbox row + the park-ledger row
    # commit in ONE transaction, on the runtime's per-crew saver. A
    # non-transactional saver is refused here rather than quietly splitting
    # them.
    saver = require_transactional_saver(
        await runtime.transactional_saver(crew="qgen"),
        lane="qgen crew",
    )

    # Single async psycopg connection shared by:
    #   - QGenCrewTerminalOutboxWriter (INSERT outbox row)
    #   - InboxIdempotencyStore         (UPSERT / SELECT inbox dedupe)
    #   - PostgresOutboxStore           (drain pending rows)
    #   - AgentDecisionLogOutboxWriter  (Gate #8 per-terminal decision emit)
    #
    # Awaited inline so every adapter binds to the live connection
    # — not the unawaited coroutine that `connect()` returns. The
    # previous shape passed the coroutine to adapters and the caller
    # was expected to `await qgen_components.db_conn` post-hoc; that
    # left adapters holding a stale coroutine reference and crashed
    # downstream `.cursor()` calls in the outbox dispatcher tight loop.
    #
    # Self-healing (build_qgen_db_conn): this used to bind the RAW
    # AsyncConnection, so a Cloud SQL restart left it dead for the pod's
    # whole lifetime while /readyz stayed 200 (live 2026-07-23). The
    # wrapper reconnects on a closed/broken underlying connection and
    # keeps the cold-start backoff for reconnects too.
    #
    # Cold-start race retry: cloudsql-proxy sidecar may not be listening
    # on first orchestrator pod start (workaround was `kubectl delete pod`
    # to retry on a warm node). Retry with exponential backoff so the
    # lifespan succeeds on first boot. Mirrors chora-creation's Go-side
    # CHORA_BOOTSTRAP_TIMEOUT_SECONDS=90 pattern. Closes Debt 9 from
    # FE-coord E2E-BE-AI-ASSIST-TRACE-EXPORT-PERM debt audit 2026-05-17.
    #
    # Eager connect keeps the fail-loud startup: an unreachable DB raises
    # here rather than surfacing as a silent no-op drain later.
    db_conn = await build_qgen_db_conn(psycopg, dsn).connect()

    # ADR-169 — tier-mapped guardrail port. Resolves the per-agent template
    # tier from agent-guardrail-mapping.yaml (qgen_question + qgen_critic →
    # balanced) instead of threading a bare env template name. The shared
    # Cloud Model Armor screener (built by handlers.build_app for the legacy
    # 6-agent crew) is injected so both crews ride the same SDK transport.
    # from_env returns None only when the YAML mapping is unreadable — fail
    # loud (skip the qgen path) rather than silently mis-screen.
    guardrail = ModelArmorGuardrailPort.from_env(screener=screener)
    if guardrail is None:
        logger.error(
            "qgen_crew_wiring.skipped: ModelArmorGuardrailPort.from_env "
            "returned None (agent-guardrail-mapping.yaml unreadable)"
        )
        return None

    # W8 / ADR-254 D12 image-render clients (KrokiClient / GcsImageUploadAdapter).
    # Mermaid renders in-process on Kroki and the kennel uploads + signs; a
    # scene render is a qgen_render dispatch whose returned gs:// the kennel
    # only SIGNS (the same GCS adapter). Each is None when its env var is
    # unset: the render plan stays a no-op unless an accepted candidate
    # carries image_specs, and fails loud only then if a client is missing
    # (per [[secrets-and-env]]). NOT a gate on the qgen path coming online.
    kroki_client = KrokiClient.from_env()
    gcs_image_client = GcsImageUploadAdapter.from_env()
    if not (kroki_client and gcs_image_client):
        logger.info(
            "qgen_crew_wiring.image_render_partial: image clients "
            "(kroki=%s gcs=%s); render_image is a no-op until both are "
            "env-wired AND a candidate carries image_specs",
            kroki_client is not None,
            gcs_image_client is not None,
        )

    # ADR-251 D4 (CHO-2398) - the in-flight job registry rides the same shared
    # connection. The terminal writer deletes the registry row inside its own
    # commit (same-transaction boundary); acceptance + the boot sweep use the
    # registry's own committing methods.
    from chora_ai_kernel_orchestrator.adapter.postgres.inflight_registry import (
        InflightRegistry,
    )

    inflight_registry = InflightRegistry(conn=db_conn)

    outbox_writer = QGenCrewTerminalOutboxWriter(
        conn=db_conn,
        source_project=pubsub_project,
        inflight_registry=inflight_registry,
    )

    graph = build_qgen_crew_graph(
        executor=executor,
        guardrail=guardrail,
        checkpointer=saver,
        kroki=kroki_client,
        gcs=gcs_image_client,
        # ADR-251 D5 - publish_chunk emits chunk_completed.v1 through the
        # same outbox writer (deterministic per-chunk idempotency key).
        publisher=outbox_writer,
    )

    # Gate #8 — AgentDecisionLog outbox writer. Shares the same DB
    # connection so the AccountabilityEvidence row + terminal event lands
    # in the same atomic unit (D6 P1 pod-death survival). The runner's
    # _emit_agent_decision_log is best-effort per [[feedback-d6-
    # resilience-first-class]] — a transient emit failure here logs +
    # proceeds so the terminal publish is never suppressed.
    agent_decision_writer = AgentDecisionLogOutboxWriter(
        conn=db_conn,
        source_project=pubsub_project,
    )

    # Human-Oversight gate writer — emits chora.governance.hitl.requested.v1
    # on the genuinely low-quality (quality_warning) terminal so the O+
    # Human-Oversight queue is no longer permanently empty. Shares db_conn so
    # the gate row + terminal publish are atomic (D6 P1). Best-effort per
    # [[feedback-d6-resilience-first-class]] — a transient emit failure logs +
    # proceeds; the terminal publish is never suppressed.
    hitl_decision_writer = HITLDecisionOutboxWriter(
        conn=db_conn,
        source_project=pubsub_project,
    )

    # CHO live-trace streaming — when QGEN_PROGRESS_STREAM_ENABLED is truthy the
    # single runner switches to astream + publishes one ai_assist.progress.v1 per
    # graph node (the SAME outbox writer, which carries publish_progress). Default
    # off ⇒ byte-stable single-shot ainvoke (ships dark; flip on once chora-
    # creation's progress subscriber + the FE live-trace consumer are deployed).
    progress_stream_enabled = os.getenv("QGEN_PROGRESS_STREAM_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    logger.info(
        "qgen_crew_wiring.progress_stream_%s",
        "enabled" if progress_stream_enabled else "disabled",
    )
    # ADR-197 M-B.2 — optional PromptResolver over the shared psycopg conn
    # (the 0007 prompt_override_* tables live in chora_ai_kernel, same DB as the
    # outbox). Bound to the SAME db_conn so the override SELECT rides the same
    # connection as the rest of the crew. When no override exists for a
    # (tenant, agent) the resolver returns the embedded default → byte-identical
    # behaviour. Constructed unconditionally here because db_conn is always live
    # by this point; the runner self-no-ops per-run when no override applies.
    from chora_ai_kernel_orchestrator.adapter.pg.prompt_override_repository import (
        PostgresPromptOverrideRepository,
    )
    from chora_ai_kernel_orchestrator.domain.prompt_registry import (
        PromptResolver,
    )

    prompt_resolver = PromptResolver(PostgresPromptOverrideRepository(conn=db_conn))

    runner = QGenCrewRunner(
        graph=graph,
        publisher=outbox_writer,
        agent_decision_emitter=agent_decision_writer,
        hitl_decision_emitter=hitl_decision_writer,
        progress_emitter=outbox_writer if progress_stream_enabled else None,
        prompt_resolver=prompt_resolver,
    )

    # EPIC-1a — the batch runner reuses the SAME compiled graph + outbox writer;
    # it loops the graph requested_count times and publishes ONE completed.v1
    # with a JSON-array candidate_payload_json. The router dispatches on
    # job_kind so the subscriber stays unchanged (single → runner, batch →
    # batch_runner).
    #
    # Lane 1c (CHO-1703) / ADR-254 D2: the test-set composer is the set lane's
    # compose_test_set node (ONE mode=compose dispatch on qgen_generate). The
    # env flag is the only switch left: no gateway client, no target.
    compose_enabled = os.getenv(ENV_TESTSET_COMPOSE_ENABLED, "").strip().lower() in {"1", "true", "yes", "on"}
    logger.info(
        "qgen_crew_wiring.testset_compose_%s",
        "enabled" if compose_enabled else "disabled",
    )
    batch_runner = QGenBatchRunner(
        graph=graph,
        publisher=outbox_writer,
        compose_enabled=compose_enabled,
        # CHO-2364 - the SAME Gate #8 writer the single runner uses, so the
        # batch lane (incl. the daily-dose type_plan path) emits per-item
        # qgen_question/qgen_critic decisions into the same outbox. Absent
        # this, the batch lane emitted NO AgentDecisionLog at all (zero qgen
        # decision rows since 2026-07-04 while dose generation ran batch).
        agent_decision_emitter=agent_decision_writer,
        # Live-trace streaming for the SET lane, same flag + writer as the
        # single runner. The canvas's mixed-type batch rides the set lane and
        # published zero progress.v1 while the single lane streamed (topic
        # metric read 0 during a batch run, 2026-08-07); the FE canvas and the
        # creation progress consumer are already live, so wiring the emitter
        # is the whole remaining fix.
        progress_emitter=outbox_writer if progress_stream_enabled else None,
    )
    # CHO-1819 P3 / ADR-254 D2: review image regenerate, a small checkpointed
    # graph on the SAME dispatch adapter, saver, Kroki + GCS clients and outbox
    # writer. Constructed unconditionally (the image clients may be None; the
    # runner self-refuses if unwired) so the router always has the image_regen
    # branch.
    image_regen_runner = ImageRegenRunner(
        graph=build_image_regen_graph(
            executor=executor,
            kroki=kroki_client,
            gcs=gcs_image_client,
            checkpointer=saver,
        ),
        publisher=outbox_writer,
        kroki=kroki_client,
        gcs=gcs_image_client,
        # ADR-210 image-to-image: the CURRENT image is checked by reference
        # (exists) BEFORE any dispatch and handed to the renderer by gs://
        # reference so a refinement EDITS it. ADC auth (same as the weakness
        # downloader). The runner FAILS LOUD if an original is requested but
        # gone: never a silent text-redraw.
        image_downloader=GcsBlobDownloader(),
    )
    # ADR-254 D5: completions carry only the thread id; the router resolves
    # which runner owns a job from its durable started payload (the inflight
    # registry), so a pod that did not accept the job can still resume it.
    router = QGenRunnerRouter(
        single=runner,
        batch=batch_runner,
        image_regen=image_regen_runner,
        inflight_registry=inflight_registry,
    )

    # ADR-251 D4 - the durable acceptance/drive split. The subscriber now ACKs
    # after REGISTRATION (durable), not after the whole graph run; the drive
    # continues as a background task and the boot sweep (main.py lifespan)
    # resumes anything a pod death stranded. The router keeps its exact
    # handle_started contract underneath.
    from chora_ai_kernel_orchestrator.orchestrators.qgen_durable_acceptance import (
        DurableQGenAcceptance,
    )
    from chora_ai_kernel_orchestrator.orchestrators.qgen_resume_sweeper import (
        OutboxTerminalIndex,
        QGenResumeSweeper,
    )

    acceptance = DurableQGenAcceptance(registry=inflight_registry, router=router)
    resume_sweeper = QGenResumeSweeper(
        registry=inflight_registry,
        acceptance=acceptance,
        terminal_index=OutboxTerminalIndex(conn=db_conn),
        # ADR-254 D5: a job parked on an agent dispatch is resumed by its
        # completion, never re-driven by the boot sweep.
        park_ledger=runtime.ledger,
    )
    # ADR-254 D2/D5: the three qgen roles resume through the acceptance (the
    # completion ack path does a checkpoint round trip; the settle runs as a
    # tracked background drive). Tracked for /readyz.
    runtime.register_lane("qgen", crew="qgen", roles=QGEN_LANE_ROLES, runner=acceptance)

    inbox = InboxIdempotencyStore(conn=db_conn)
    subscriber = QGenCrewSubscriber(runner=acceptance, inbox=inbox)
    pubsub_loop = QGenCrewPubsubLoop.from_env(subscriber=subscriber)
    if pubsub_loop is None:
        return None

    # Build outbox dispatcher — drains ai_kernel_outbox_events.
    publisher = NatsPublisher(url=nats_url)
    store = PostgresOutboxStore(conn=db_conn, worker_id="qgen-crew-worker-1")
    dispatcher = OutboxDispatcher(
        store=store,
        publisher=publisher,
        worker_id="qgen-crew-worker-1",
    )

    return QGenCrewComponents(
        pubsub_loop=pubsub_loop,
        outbox_dispatcher=dispatcher,
        db_conn=db_conn,
        subscriber=subscriber,
        runner=runner,
        inbox=inbox,
        outbox_writer=outbox_writer,
        agent_decision_writer=agent_decision_writer,
        hitl_decision_writer=hitl_decision_writer,
        executor=executor,
        graph=graph,
        acceptance=acceptance,
        resume_sweeper=resume_sweeper,
        inflight_registry=inflight_registry,
    )


__all__ = [
    "QGenCrewComponents",
    "build_qgen_crew_from_env",
]
