"""Composition root for the generic single-agent lanes (ADR-254 D5): the two
fog folds (D13: ``kg_exploration_workflow`` + ``companion_reflection_workflow``)
and the caller-facing ``companion_turn`` lane (D1/D4/D8), whose contract lives
in ``orchestrators/consumption_lanes``.

One ``LaneSpec`` row per lane: its ``LaneContract``, the env flag that
enables it, the env var naming its request subscription (default = the
subscription the coordinator provisions at the fold cut). For each enabled lane:

  * its own autocommit ``ReconnectingAsyncConnection`` (the OE pattern);
  * the kennel runtime's D3a transactional saver for its crew (park + dispatch
    outbox row + park-ledger row in ONE transaction), refused if not D3a;
  * ONE ``PubSubAgentExecutor`` pinned to its role;
  * the shared ``AgentDispatchOutboxWriter`` shape for the result event;
  * the compiled graph, the runner, the inbox store, the request subscriber and
    a StreamingPull loop on exactly its request subscription;
  * the lane registered on the runtime (crew + role): the ONE role-routed
    completion loop resumes it, the reaper watches its DLQs.

No model-gateway client is constructed here (deterministic kernel, D5).

Env (per [[secrets-and-env]]):
    KG_EXPLORATION_ENABLED / KG_EXPLORATION_SUBSCRIPTION
        (default chora.consumption.concept_suggestion.requested.v1)
    COMPANION_REFLECTION_ENABLED / COMPANION_REFLECTION_SUBSCRIPTION
        (default chora.consumption.goal_knowledge.synthesis_requested.v1)
    COMPANION_TURN_ENABLED / COMPANION_TURN_SUBSCRIPTION
        (default chora.consumption.companion_turn.requested.v1)
    DOSE_RECOMMENDATION_ENABLED / DOSE_RECOMMENDATION_SUBSCRIPTION
        (default chora.consumption.dose_recommendation.requested.v1)
    COMPANION_TURN_TENANT_INFLIGHT_CAP
        optional per-tenant in-flight cap; unset = no cap, unreadable = RAISE
    CHORA_PUBSUB_PROJECT, CHORA_AI_KERNEL_PG_DSN (or _SECRET_ID)
An enabled lane that cannot be built returns None and the lifespan ABORTS.

The subscription env vars name a NATS SUBJECT, not a Pub/Sub subscription id: in
this codebase the subscription string is the NATS subject the pull consumer binds
to, and the live JetStream stream captures only ``chora.>``. Each default below
is therefore the canonical ``chora.consumption.*`` request topic the lane's own
contract module names — never the legacy Pub/Sub subscription id.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, replace
from typing import Any

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_completion_loop import (
    AgentCompletionPubsubLoop,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_outbox_writer import (
    AgentDispatchOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.inbox_idempotency import (
    InboxIdempotencyStore,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.reconnecting_connection import (
    ReconnectingAsyncConnection,
)
from chora_ai_kernel_orchestrator.orchestrators.consumption_lanes import (
    companion_turn_contract,
    dose_recommendation_contract,
)
from chora_ai_kernel_orchestrator.orchestrators.fold_lanes import (
    companion_reflection_contract,
    kg_exploration_contract,
)
from chora_ai_kernel_orchestrator.orchestrators.single_agent_workflow import (
    LaneContract,
    SingleAgentRequestSubscriber,
    SingleAgentWorkflowRunner,
    build_single_agent_graph,
)

logger = logging.getLogger(__name__)


#: Dispatch roles whose AGENT BINARY actually runs the agentdispatch subscriber,
#: so a parked run on that role can be completed and resumed.
#:
#: ⚠ THIS IS A CENSUS BY EFFECT, NOT BY NAMING CONVENTION. A crew having a
#: Deployment, a Service, a provisioned request subscription and a registry entry
#: proves NONE of it: chora-recommender has had all four since 2026-06-04 while
#: running the ADK web launcher with no AGENT_DISPATCH_* env, so nothing has ever
#: consumed its request topic. That shape (an estate provisioned for a consumer
#: nobody wrote) is what this set exists to make impossible to enable by accident.
#:
#: ADDING A ROLE HERE IS A CLAIM THAT ITS BINARY SUBSCRIBES. Verify it the way
#: the census was taken: read AGENT_DISPATCH_ENABLED / AGENT_DISPATCH_SUBSCRIPTION
#: off the live Deployment, not the manifest and not the registry.
#:
#: Deliberately ABSENT: `recommend`. The dose lane is committed and correct but
#: DORMANT by owner ruling (2026-08-23, the dose recommender is out of the
#: refactor deliverable); its agent half was never written.
ROLES_WITH_SUBSCRIBER_AGENTS: frozenset[str] = frozenset(
    {
        "companion_chat",
        "companion_diagnosis",
        "kg_explore",
        "oe_grading",
        "qgen",
    }
)


def resolve_tenant_inflight_cap(name: str) -> int | None:
    """Read a lane's per-tenant in-flight cap from the environment.

    Unset means no cap. A value that is set but unreadable RAISES rather than
    falling back to "no cap": a mis-typed cap that silently means unlimited is
    the failure this backpressure exists to prevent.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        cap = int(raw)
    except ValueError:
        raise ValueError(
            f"single_agent_lanes_wiring: {name}={raw!r} is not an integer; refusing to "
            "read an unparseable cap as no cap"
        ) from None
    if cap < 1:
        raise ValueError(f"single_agent_lanes_wiring: {name} must be >= 1, got {cap}")
    return cap


@dataclass(frozen=True)
class LaneSpec:
    contract: LaneContract
    enabled_env: str
    subscription_env: str
    default_subscription: str
    #: request-side per-message callback wait, seconds. NO default on purpose:
    #: AgentCompletionPubsubLoop.DEFAULT_CALLBACK_TIMEOUT_S is justified for a
    #: COMPLETION resume, so a lane reusing that loop on its REQUEST side must
    #: state its own number here rather than inherit that one silently.
    callback_timeout_s: float
    subscription: str = ""
    #: env var naming this lane's per-tenant in-flight cap; "" = the lane takes none
    cap_env: str = ""

    def resolved(self) -> LaneSpec:
        sub = (os.getenv(self.subscription_env) or "").strip() or self.default_subscription
        contract = self.contract
        if self.cap_env:
            cap = resolve_tenant_inflight_cap(self.cap_env)
            if cap != contract.tenant_inflight_cap:
                # LaneContract.__post_init__ re-validates, so a cap without a
                # rejected() result still cannot be constructed here.
                contract = replace(contract, tenant_inflight_cap=cap)
        return LaneSpec(
            contract=contract,
            enabled_env=self.enabled_env,
            subscription_env=self.subscription_env,
            default_subscription=self.default_subscription,
            subscription=sub,
            callback_timeout_s=self.callback_timeout_s,
            cap_env=self.cap_env,
        )


SINGLE_AGENT_LANES: tuple[LaneSpec, ...] = (
    LaneSpec(
        contract=kg_exploration_contract(),
        enabled_env="KG_EXPLORATION_ENABLED",
        subscription_env="KG_EXPLORATION_SUBSCRIPTION",
        default_subscription="chora.consumption.concept_suggestion.requested.v1",
        callback_timeout_s=60.0,
    ),
    LaneSpec(
        contract=companion_reflection_contract(),
        enabled_env="COMPANION_REFLECTION_ENABLED",
        subscription_env="COMPANION_REFLECTION_SUBSCRIPTION",
        default_subscription="chora.consumption.goal_knowledge.synthesis_requested.v1",
        callback_timeout_s=60.0,
    ),
    LaneSpec(
        contract=companion_turn_contract(),
        enabled_env="COMPANION_TURN_ENABLED",
        subscription_env="COMPANION_TURN_SUBSCRIPTION",
        default_subscription="chora.consumption.companion_turn.requested.v1",
        callback_timeout_s=60.0,
        cap_env="COMPANION_TURN_TENANT_INFLIGHT_CAP",
    ),
    LaneSpec(
        contract=dose_recommendation_contract(),
        enabled_env="DOSE_RECOMMENDATION_ENABLED",
        subscription_env="DOSE_RECOMMENDATION_SUBSCRIPTION",
        default_subscription="chora.consumption.dose_recommendation.requested.v1",
        # Stated rather than inherited, per the field's contract. 60s matches the
        # siblings and is deliberately NOT consumption's 30s dose deadline: this
        # wait covers decode + dispatch + PARK, which is fast, while the 30s
        # governs how long the learner's handler waits for the ANSWER. Tying the
        # two would abandon a park that had already succeeded.
        callback_timeout_s=60.0,
        cap_env="DOSE_RECOMMENDATION_TENANT_INFLIGHT_CAP",
    ),
)


def _truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def single_agent_lane_specs_from_env() -> list[LaneSpec]:
    """The enabled lanes, in table order, with their subscriptions resolved."""
    return [spec.resolved() for spec in SINGLE_AGENT_LANES if _truthy(spec.enabled_env)]


@dataclass
class SingleAgentLaneComponents:
    """One lane's adapters, started/stopped together by main.py."""

    name: str
    pubsub_loop: AgentCompletionPubsubLoop
    db_conn: Any
    subscriber: SingleAgentRequestSubscriber
    runner: SingleAgentWorkflowRunner
    graph: Any
    inbox: InboxIdempotencyStore

    async def aclose(self) -> None:  # pragma: no cover - main.py lifespan
        if self.db_conn is not None:
            try:
                await self.db_conn.close()
            except Exception:
                logger.exception("single_agent_lane.db_close_failed", extra={"lane": self.name})


async def _connect_with_retry(
    psycopg_mod: Any, dsn: str, *, attempts: int = 5, base_delay_s: float = 2.0, max_delay_s: float = 8.0
) -> Any:  # pragma: no cover
    last_exc: Exception | None = None
    delay = base_delay_s
    for i in range(1, attempts + 1):
        try:
            return await psycopg_mod.AsyncConnection.connect(dsn)
        except Exception as exc:  # noqa: BLE001 - cold-start race surface is broad
            last_exc = exc
            if i == attempts:
                break
            logger.warning(
                "single_agent_lane.db_connect.retry",
                extra={"attempt": i, "delay_s": delay, "err": f"{exc.__class__.__name__}: {exc}"},
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay_s)
    assert last_exc is not None
    raise last_exc


async def build_single_agent_lanes_from_env(*, runtime: Any) -> list[SingleAgentLaneComponents] | None:
    """Build every enabled fold lane. ``[]`` when none is enabled; ``None`` when
    an enabled lane cannot be built (the lifespan then aborts); RAISES without
    the kennel runtime (a lane that parks runs nothing resumes)."""
    specs = single_agent_lane_specs_from_env()
    if not specs:
        return []
    if runtime is None:
        raise RuntimeError(
            "single_agent_lanes_wiring: the kennel runtime is required (ADR-254 D5); "
            "without it a fold lane would park runs nothing resumes"
        )
    nats_url = (os.getenv("NATS_URL") or "").strip()
    if not nats_url:
        logger.error("single_agent_lanes_wiring.skipped: NATS_URL unset")
        return None
    # Provenance label for the outbox envelope (source_project).
    pubsub_project = "chora-ai-kernel-orchestrator"
    from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

    dsn = resolve_dsn()
    if not dsn:
        logger.error("single_agent_lanes_wiring.skipped: CHORA_AI_KERNEL_PG_DSN unset")
        return None
    built: list[SingleAgentLaneComponents] = []
    for spec in specs:
        components = await _build_lane_live(  # pragma: no cover - live SDK/DB glue
            spec=spec,
            pubsub_project=pubsub_project,
            dsn=dsn,
            runtime=runtime,
            nats_url=nats_url,
        )
        if components is None:
            return None
        built.append(components)
    return built


async def _build_lane_live(
    *, spec: LaneSpec, pubsub_project: str, dsn: str, runtime: Any, nats_url: str
) -> SingleAgentLaneComponents | None:  # pragma: no cover
    import psycopg

    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_wiring import (
        require_transactional_saver,
    )

    async def _connect(d: str) -> Any:
        return await _connect_with_retry(psycopg, d)

    db_conn = ReconnectingAsyncConnection(dsn, autocommit=True, connect=_connect)
    await db_conn.connect()  # eager: fail loud at startup if the DB is unreachable
    checkpointer = require_transactional_saver(
        await runtime.transactional_saver(crew=spec.contract.name),
        lane=f"{spec.contract.name} lane",
    )
    components = _assemble_single_agent_lane(
        contract=spec.contract,
        subscription=spec.subscription,
        pubsub_project=pubsub_project,
        db_conn=db_conn,
        checkpointer=checkpointer,
        runtime=runtime,
        callback_timeout_s=spec.callback_timeout_s,
        nats_url=nats_url,
    )
    if components is None:
        await db_conn.close()
    return components


def _assemble_single_agent_lane(
    *,
    contract: LaneContract,
    subscription: str,
    pubsub_project: str,
    db_conn: Any,
    checkpointer: Any,
    runtime: Any,
    callback_timeout_s: float,
    nats_url: str,
) -> SingleAgentLaneComponents:
    """Pure of DB-connect / SDK-client construction so it is unit-testable."""
    if runtime is None:
        raise RuntimeError(
            f"single_agent_lanes_wiring {contract.name!r}: the kennel runtime is required to bind "
            "the completion role; a run parked on it could never be resumed"
        )
    if contract.role not in ROLES_WITH_SUBSCRIBER_AGENTS:
        # Same failure as the runtime guard above, reached through the other
        # door: there the resume machinery is missing, here the ANSWERER is.
        raise RuntimeError(
            f"single_agent_lanes_wiring {contract.name!r}: role {contract.role!r} has "
            "no subscriber-capable agent, so this lane would park every request on a "
            "dispatch nothing can ever complete. Refusing to start. Add the role to "
            "ROLES_WITH_SUBSCRIBER_AGENTS only once its binary actually runs "
            "agentdispatch; setting the lane's ENABLED flag is NOT sufficient."
        )
    if not (subscription or "").strip():
        raise ValueError(f"single_agent_lanes_wiring {contract.name!r}: a request subscription is required")
    from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
        PubSubAgentExecutor,
    )

    executor = PubSubAgentExecutor(source_project=pubsub_project, allowed_roles={contract.role})
    outbox_writer = AgentDispatchOutboxWriter(conn=db_conn, source_project=pubsub_project)
    graph = build_single_agent_graph(
        contract=contract,
        executor=executor,
        outbox_writer=outbox_writer,
        source_project=pubsub_project,
        checkpointer=checkpointer,
    )
    runner = SingleAgentWorkflowRunner(
        graph=graph,
        contract=contract,
        outbox_writer=outbox_writer,
        source_project=pubsub_project,
        ledger=getattr(runtime, "ledger", None),
    )
    inbox = InboxIdempotencyStore(conn=db_conn)
    subscriber = SingleAgentRequestSubscriber(runner=runner, inbox=inbox, contract=contract)
    pubsub_loop = AgentCompletionPubsubLoop(
        subscriber=subscriber,
        project=pubsub_project,
        subscriptions=[subscription],
        callback_timeout_s=callback_timeout_s,
    )
    pubsub_loop._url = nats_url  # noqa: SLF001 — direct construction; set the url
    runtime.register_lane(contract.name, crew=contract.name, roles=(contract.role,), runner=runner)
    return SingleAgentLaneComponents(
        name=contract.name,
        pubsub_loop=pubsub_loop,
        db_conn=db_conn,
        subscriber=subscriber,
        runner=runner,
        graph=graph,
        inbox=inbox,
    )


__all__ = [
    "ROLES_WITH_SUBSCRIBER_AGENTS",
    "SINGLE_AGENT_LANES",
    "LaneSpec",
    "SingleAgentLaneComponents",
    "build_single_agent_lanes_from_env",
    "single_agent_lane_specs_from_env",
]
