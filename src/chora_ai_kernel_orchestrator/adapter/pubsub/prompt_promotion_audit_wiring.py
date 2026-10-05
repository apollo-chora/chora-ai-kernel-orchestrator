"""Composition root for the prompt-override HITL approval round-trip
(ADR-197 M-C.2).

Bundles every adapter the audit-recorded consumer needs into a single
``PromptPromotionAuditComponents`` dataclass that ``main.py`` lifespan
starts/stops together (mirrors ``build_qgen_crew_from_env`` /
``build_oe_grading_crew_from_env``). Pulls config from env vars per
[[secrets-and-env]] — no inline config.

Subscribes to ``chora.governance.audit.recorded.v1`` (provisioned via Terraform —
``chora-infra/terraform/environments/dev/main.tf`` subscription
``chora-ai-kernel-orchestrator.prompt-promotion-audit-recorded``; the orchestrator
reads its resource name from ``PROMPT_PROMOTION_AUDIT_SUBSCRIPTION``) and drives
the promotion transition on a human approve/reject.

BEHAVIOUR-NEUTRAL: this path comes online only when
``PROMPT_PROMOTION_AUDIT_ENABLED`` is truthy AND the env is fully wired. Until a
draft plan exists (M-D) and an approval actually flows, the consumer simply acks
+ ignores the high-volume audit traffic (strict prefix filter) — nothing
transitions.

Env vars (all required for the path to come online; missing any leaves the
orchestrator up without this consumer):

    CHORA_PUBSUB_PROJECT                  — Pub/Sub host project (chora-489812)
    CHORA_AI_KERNEL_PG_DSN                — psycopg DSN for chora_ai_kernel
    PROMPT_PROMOTION_AUDIT_SUBSCRIPTION   — full subscription resource (or short
                                            name; default below)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from chora_ai_kernel_orchestrator.adapter.pubsub.audit_recorded_consumer import (
    AuditRecordedConsumer,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.hitl_decision_outbox_writer import (
    HITLDecisionOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.inbox_idempotency import (
    InboxIdempotencyStore,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.prompt_activation_audit_emitter import (
    PromptActivationAuditEmitter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.prompt_hitl_request_emitter import (
    PromptHITLRequestEmitter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_loop import (
    QGenCrewPubsubLoop,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_wiring import (
    _connect_with_retry,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.reconnecting_connection import (
    ReconnectingAsyncConnection,
)
from chora_ai_kernel_orchestrator.domain.prompt_registry import (
    PromptPromotionApprovalHandler,
    PromptPromotionService,
)

logger = logging.getLogger(__name__)

# Matches the Terraform subscription key for this consumer.
DEFAULT_SUBSCRIPTION = "chora-ai-kernel-orchestrator.prompt-promotion-audit-recorded"
ENV_SUBSCRIPTION = "PROMPT_PROMOTION_AUDIT_SUBSCRIPTION"

_WORKER_ID = "prompt-promotion-audit-worker-1"


@dataclass
class PromptPromotionAuditComponents:
    """All the adapters main.py needs to start/stop together."""

    pubsub_loop: QGenCrewPubsubLoop
    outbox_dispatcher: Any  # OutboxDispatcher — lifecycle managed by main.py
    db_conn: Any
    consumer: AuditRecordedConsumer
    handler: PromptPromotionApprovalHandler
    service: PromptPromotionService

    async def aclose(self) -> None:  # pragma: no cover — main.py lifespan
        if self.db_conn is not None:
            try:
                await self.db_conn.close()
            except Exception:
                logger.exception("prompt_promotion_audit_components.db_close_failed")


def _make_db_conn(dsn: str) -> ReconnectingAsyncConnection:
    """The audit lane's connection: self-healing wrapper, autocommit=True.

    autocommit=True is LOAD-BEARING, not a convenience (mirrors the OE
    wiring's note): this lane's own OutboxDispatcher is built but never
    started (main.py single-drain invariant - the qgen dispatcher runs on its
    OWN connection), so nothing co-located ever commits here. With psycopg's
    default autocommit=False, the inbox's first bare SELECT opens an implicit
    transaction; every repo/emitter ``conn.transaction()`` block after it
    degrades to a SAVEPOINT inside that never-committed transaction, and the
    already-ACKed approval's activation + activation audit + inbox mark all
    roll back when the connection resets. Observed live 2026-07-27: an O+
    approve was consumed + acked and the plan stayed pending_hitl, the app
    backend idle with last query ROLLBACK. autocommit=True makes each write
    durable + cross-connection-visible immediately; ``transaction()`` blocks
    still give real BEGIN/COMMIT units (and tx-local set_config scoping)
    where the repo needs them.
    """

    async def _connect(d: str) -> Any:
        import psycopg

        return await _connect_with_retry(psycopg, d, attempts=5, base_delay_s=2.0, max_delay_s=8.0)

    return ReconnectingAsyncConnection(dsn, autocommit=True, connect=_connect)


def _assemble_components(
    *,
    db_conn: Any,
    pubsub_project: str,
    subscription: str,
    publisher: Any,
) -> PromptPromotionAuditComponents:
    """Pure assembly - thread the ONE lane connection through every adapter.

    No I/O here; the env/SDK glue lives in
    :func:`build_prompt_promotion_audit_from_env`.
    """
    from chora_ai_kernel_orchestrator.adapter.pg.prompt_override_repository import (
        PostgresPromptOverrideRepository,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub import (
        OutboxDispatcher,
        PostgresOutboxStore,
    )

    repo = PostgresPromptOverrideRepository(conn=db_conn)
    activation_audit = PromptActivationAuditEmitter(conn=db_conn, source_project=pubsub_project)
    hitl_writer = HITLDecisionOutboxWriter(conn=db_conn, source_project=pubsub_project)
    hitl_emitter = PromptHITLRequestEmitter(writer=hitl_writer)
    service = PromptPromotionService(
        repo=repo,
        audit_emitter=activation_audit,
        hitl_emitter=hitl_emitter,
    )
    handler = PromptPromotionApprovalHandler(transition_service=service, plan_reader=repo)
    inbox = InboxIdempotencyStore(conn=db_conn)
    consumer = AuditRecordedConsumer(handler=handler, inbox=inbox)

    # Reuse the generic StreamingPull wrapper (QGenCrewPubsubLoop is a thin,
    # subscriber-agnostic Pub/Sub loop — the consumer only needs handle_message).
    pubsub_loop = QGenCrewPubsubLoop(
        subscriber=consumer,
        project=pubsub_project,
        subscription=subscription,
    )

    store = PostgresOutboxStore(conn=db_conn, worker_id=_WORKER_ID)
    dispatcher = OutboxDispatcher(store=store, publisher=publisher, worker_id=_WORKER_ID)

    return PromptPromotionAuditComponents(
        pubsub_loop=pubsub_loop,
        outbox_dispatcher=dispatcher,
        db_conn=db_conn,
        consumer=consumer,
        handler=handler,
        service=service,
    )


async def build_prompt_promotion_audit_from_env() -> (
    PromptPromotionAuditComponents | None
):  # pragma: no cover - integration glue; component units cover the parts
    """Compose the audit-recorded consumer from env vars.

    Returns ``None`` (orchestrator stays up without this path) when any required
    env var is missing - fail-loud-by-skip, never a silent half-wire.
    """
    nats_url = (os.getenv("NATS_URL") or "").strip()
    if not nats_url:
        logger.warning("prompt_promotion_audit_wiring.skipped: NATS_URL unset")
        return None
    # Provenance label for the outbox envelope (source_project).
    pubsub_project = "chora-ai-kernel-orchestrator"

    from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

    dsn = resolve_dsn()
    if not dsn:
        logger.warning("prompt_promotion_audit_wiring.skipped: CHORA_AI_KERNEL_PG_DSN unset")
        return None

    subscription = (os.getenv(ENV_SUBSCRIPTION) or DEFAULT_SUBSCRIPTION).strip()

    from chora_ai_kernel_orchestrator.adapter.pubsub import (
        NatsPublisher,
    )

    # Own connection (separate from the qgen/oe crews) - all this path's writes
    # land in the shared ai_kernel_outbox_events table, drained by whatever
    # single OutboxDispatcher main.py started (single-drain invariant).
    db_conn = _make_db_conn(dsn)
    # Eager connect (fail loud at startup if the DB is unreachable) + apply
    # autocommit; subsequent reconnects re-apply it automatically.
    await db_conn.connect()

    publisher = NatsPublisher(url=nats_url)
    components = _assemble_components(
        db_conn=db_conn,
        pubsub_project=pubsub_project,
        subscription=subscription,
        publisher=publisher,
    )

    logger.info(
        "prompt_promotion_audit_wiring.built",
        extra={"subscription": subscription, "project": pubsub_project},
    )
    return components


__all__ = [
    "DEFAULT_SUBSCRIPTION",
    "ENV_SUBSCRIPTION",
    "PromptPromotionAuditComponents",
    "build_prompt_promotion_audit_from_env",
]
