"""QGenCrewSubscriber — Pub/Sub StreamingPull subscriber for the qgen
2-agent crew's inbound started.v2 events.

Wired in the chora-ai-kernel-orchestrator kennel as one of N per-crew
subscribers. Subscribes to ``chora.creation.ai_assist.started.v2`` and
routes each message through:

    inbox.process(dedup_key, ttl, fn=lambda: runner.handle_started(event))

If the inbox short-circuits (duplicate delivery), the message ACKs
immediately without invoking the runner — saves a wasted graph re-run
from PostgresSaver.

D6 4-pillar contract per [[agentic-resilience-d6]]:

* **P1 Pod-death survival** — fn := runner.handle_started, which runs
  graph.ainvoke under PostgresSaver checkpointing. On pod restart, a
  retry-delivered message resumes the graph from the last persisted
  state. The orchestrator-side outbox writer (QGenCrewTerminalOutboxWriter)
  is the durable boundary for terminal events.

* **P2 Idempotency** — InboxIdempotencyStore + outbox writer's
  ON CONFLICT (idempotency_key) DO NOTHING form a belt-and-braces
  dedupe. The inbox is the first line; the outbox UNIQUE constraint
  is the safety net.

* **P3 Delivery resilience (ack-after-processing + DLQ)** — ACK happens
  only after the runner returns nil. On error → NACK; Pub/Sub redelivers
  up to ``max_delivery_attempts`` then routes to the subscription's
  configured DLQ topic.

* **P4 OTel trace context** — traceparent from message attributes is
  threaded into the runner's event dict, which qgen_crew_runner uses
  to continue the trace tree.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire import (
    ProtoDecodeError,
    decode_ai_assist_started,
    looks_like_binary_proto,
)

logger = logging.getLogger(__name__)

# Default inbox TTL — matches chora-creation subscriber's AiAssistInboxTTL
# (7 days). Pub/Sub's max redelivery window is bounded; 7d covers it.
DEFAULT_INBOX_TTL = _dt.timedelta(days=7)


class _RunnerLike(Protocol):
    """Duck-typed QGenCrewRunner from
    ``orchestrators/qgen_crew_runner.py``."""

    async def handle_started(self, event: dict[str, Any]) -> None: ...


class _InboxLike(Protocol):
    """Duck-typed InboxIdempotencyStore from
    ``adapter/pubsub/inbox_idempotency.py``."""

    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool: ...


class _MessageLike(Protocol):
    """Duck-typed Pub/Sub message — ack/nack-able with data + attributes."""

    data: bytes
    attributes: dict[str, str]

    def ack(self) -> None: ...
    def nack(self) -> None: ...


class QGenCrewSubscriber:
    """Wraps QGenCrewRunner with inbox dedupe + ack-after-processing.

    Construction takes a runner + an inbox. The actual StreamingPull
    loop is wired in main.py — the subscriber exposes a single
    ``handle_message(msg)`` entry point that the loop invokes for each
    inbound Pub/Sub message.

    The subscriber does NOT swallow errors silently — NACK is the
    explicit signal to Pub/Sub that redelivery is wanted.
    """

    def __init__(
        self,
        *,
        runner: _RunnerLike,
        inbox: _InboxLike,
        inbox_ttl: _dt.timedelta = DEFAULT_INBOX_TTL,
    ) -> None:
        self._runner = runner
        self._inbox = inbox
        self._inbox_ttl = inbox_ttl

    @property
    def inbox_ttl(self) -> _dt.timedelta:
        return self._inbox_ttl

    async def handle_message(self, msg: _MessageLike) -> None:
        """Top-level Pub/Sub message handler.

        Always returns None. ACKs or NACKs the message internally:

        * ACK on (a) successful processing OR (b) inbox dedupe-hit.
        * NACK on (a) decoding failure, (b) missing dedup key, OR
          (c) runner exception.

        The exception is logged + the message NACKed; the caller (the
        StreamingPull loop) does NOT raise — that would tear down the
        subscriber.
        """
        # Step 1: decode the message body. chora-creation now publishes
        # binary-protobuf payloads on chora.creation.ai_assist.started.v2
        # because the topic carries a BINARY-encoding Schema Registry
        # schema (chora-creation-ai_assist-started-v2, per
        # chora-contracts/proto/events-flat/creation/ai_assist/started.proto).
        # Fall back to JSON for any non-proto producer (legacy tests +
        # synthetic events). Either path yields a flat dict that the
        # runner's AiAssistStartedPayload.from_event consumes.
        event: dict[str, Any]
        try:
            if looks_like_binary_proto(msg.data):
                event = decode_ai_assist_started(msg.data)
            else:
                decoded = json.loads(msg.data.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise ValueError("expected dict body")
                event = decoded
        except (ProtoDecodeError, ValueError, UnicodeDecodeError) as exc:
            logger.exception(
                "qgen_crew_subscriber.decode_failed",
                extra={"err": str(exc), "attributes": dict(msg.attributes)},
            )
            msg.nack()
            return

        # Step 2: extract dedup_key. Prefer Pub/Sub message attributes
        # (canonical Chora pattern set by libs/chora-go-common/pubsub
        # CloudPublisher.envelopeAttributes); fall back to the proto
        # Envelope nested in the body (chora-creation's job_event_bridge
        # synthesises an envelope in the proto but does not yet emit the
        # canonical Pub/Sub attributes — tracked as a follow-up). Either
        # field anchors the InboxIdempotencyStore key.
        env_in_body = event.get("envelope") if isinstance(event, dict) else None
        dedup_key = (
            msg.attributes.get("idempotency_key")
            or msg.attributes.get("event_id")
            or (env_in_body.get("idempotency_key") if isinstance(env_in_body, dict) else "")
            or (env_in_body.get("event_id") if isinstance(env_in_body, dict) else "")
            or ""
        )
        if not dedup_key:
            logger.warning(
                "qgen_crew_subscriber.missing_dedup_key",
                extra={"attributes": dict(msg.attributes)},
            )
            msg.nack()
            return

        # Step 2b: the envelope tenant CHECKS the body tenant.
        #
        # The body's tenant_id decides which tenant the whole run executes as,
        # and it is what the image-regen source-URI constraint pins against, so
        # a body the envelope contradicts would let a forged message choose its
        # own blast radius. Until 2026-08-23 the attribute was read here for
        # LOGGING ONLY (the acked / runner_failed lines below), so it
        # contributed nothing to authorisation.
        #
        # Disagreement is FATAL; absence is not. chora-creation stamps
        # tenant_id unconditionally for this topic (job_event_bridge.go
        # envelopeAttrs, on both publish paths), so the check is live in
        # production. But six principals hold project-level
        # roles/pubsub.publisher, so an unstamped producer is possible, and
        # making absence fatal would turn the first one into an outage on the
        # whole lane. An empty value counts as absent for the same reason.
        envelope_tenant = msg.attributes.get("tenant_id", "").strip()
        body_tenant = str(event.get("tenant_id") or "").strip()
        if not envelope_tenant:
            logger.warning(
                "qgen_crew_subscriber.envelope_tenant_absent",
                extra={"dedup_key": dedup_key, "body_tenant_id": body_tenant},
            )
        elif envelope_tenant != body_tenant:
            # NACK rather than ack-and-drop: redelivery will not fix a
            # deterministic mismatch, so this lands in the lane's dead-letter
            # where the message survives as evidence instead of vanishing.
            logger.error(
                "qgen_crew_subscriber.tenant_mismatch",
                extra={
                    "dedup_key": dedup_key,
                    "envelope_tenant_id": envelope_tenant,
                    "body_tenant_id": body_tenant,
                },
            )
            msg.nack()
            return

        # Step 3: thread the trace context. Prefer message attributes
        # (canonical W3C location); fall back to proto envelope where
        # chora-creation's protomarshal encoder writes them today.
        tp = msg.attributes.get("traceparent", "") or event.get("traceparent", "")
        if tp:
            event["traceparent"] = tp
        ts = msg.attributes.get("tracestate", "") or event.get("tracestate", "")
        if ts:
            event["tracestate"] = ts

        # Step 4: inbox dedupe → invoke runner → ack-after-processing.
        async def _invoke() -> None:
            await self._runner.handle_started(event)

        try:
            ran = await self._inbox.process(
                key=dedup_key,
                ttl=self._inbox_ttl,
                fn=_invoke,
            )
        except Exception:
            logger.exception(
                "qgen_crew_subscriber.runner_failed",
                extra={
                    "dedup_key": dedup_key,
                    "tenant_id": msg.attributes.get("tenant_id", ""),
                },
            )
            msg.nack()
            return

        msg.ack()
        logger.info(
            "qgen_crew_subscriber.acked",
            extra={
                "dedup_key": dedup_key,
                "ran": ran,  # False on dedupe-hit
                "tenant_id": msg.attributes.get("tenant_id", ""),
            },
        )


__all__ = ["DEFAULT_INBOX_TTL", "QGenCrewSubscriber"]
