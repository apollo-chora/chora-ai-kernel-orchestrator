"""AuditRecordedConsumer — Pub/Sub StreamingPull subscriber for the prompt-
override HITL approval round-trip (ADR-197 M-C.2).

Subscribes to ``chora.governance.audit.recorded.v1`` and routes each message
to the pure :class:`PromptPromotionApprovalHandler`. The topic is high-volume
(every auditable action across the platform lands here), so the consumer applies
a STRICT, cheap gate FIRST — ``handler.matches(decoded)`` — and acks + ignores
any audit that is not a prompt-override HITL decision WITHOUT touching the inbox
or the handler. Only a matching decision pays for inbox dedupe + the transition.

D6 4-pillar contract (mirrors :class:`QGenCrewSubscriber`):

* **P1 Pod-death survival** — the activation it triggers writes the activation
  audit + the status flip through the transactional outbox / guarded UPDATE in
  ``chora_ai_kernel``; a redelivery after a pod death re-runs idempotently.
* **P2 Idempotency** — InboxIdempotencyStore (first line, keyed on the audit
  event's idempotency_key / event_id / audit_id) + the handler's status
  pre-check + the guarded-transition StaleError (safety net).
* **P3 Delivery resilience** — ack-after-processing; NACK on decode failure /
  missing dedup key / handler exception so Pub/Sub redelivers (→ subscription
  DLQ after max_delivery_attempts).
* **P4 OTel** — traceparent/tracestate threaded from message attributes (or the
  decoded envelope) into the handler input.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire import (
    looks_like_binary_proto,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    ProtoDecodeError,
    decode_audit_entry_recorded,
)

logger = logging.getLogger(__name__)

# Match the qgen subscriber's inbox TTL (7d covers Pub/Sub's redelivery window).
DEFAULT_INBOX_TTL = _dt.timedelta(days=7)


class _HandlerLike(Protocol):
    """Duck-typed PromptPromotionApprovalHandler."""

    def matches(self, decoded: dict[str, Any]) -> bool: ...

    async def handle(self, decoded: dict[str, Any]) -> bool: ...


class _InboxLike(Protocol):
    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool: ...


class _MessageLike(Protocol):
    data: bytes
    attributes: dict[str, str]

    def ack(self) -> None: ...
    def nack(self) -> None: ...


class AuditRecordedConsumer:
    """Wraps the approval handler with decode + strict-filter + inbox dedupe +
    ack-after-processing. The StreamingPull loop is wired in the composition
    root; this class exposes a single ``handle_message(msg)`` entry point."""

    def __init__(
        self,
        *,
        handler: _HandlerLike,
        inbox: _InboxLike,
        inbox_ttl: _dt.timedelta = DEFAULT_INBOX_TTL,
    ) -> None:
        self._handler = handler
        self._inbox = inbox
        self._inbox_ttl = inbox_ttl

    @property
    def inbox_ttl(self) -> _dt.timedelta:
        return self._inbox_ttl

    async def handle_message(self, msg: _MessageLike) -> None:
        """Top-level Pub/Sub message handler. Always returns None; acks/nacks
        the message internally."""
        # Step 1 — decode. The topic is BINARY-proto Schema-Registry-bound, so
        # the canonical body is binary AuditEntryRecorded; fall back to JSON for
        # synthetic / legacy producers.
        try:
            if looks_like_binary_proto(msg.data):
                decoded = decode_audit_entry_recorded(msg.data)
            else:
                obj = json.loads(msg.data.decode("utf-8"))
                if not isinstance(obj, dict):
                    raise ValueError("expected dict body")
                decoded = obj
        except (ProtoDecodeError, ValueError, UnicodeDecodeError) as exc:
            logger.exception(
                "audit_recorded_consumer.decode_failed",
                extra={"err": str(exc), "attributes": dict(msg.attributes)},
            )
            msg.nack()
            return

        # Step 2 — STRICT filter. The vast majority of audit.recorded traffic is
        # not a prompt-promotion decision: ack + ignore WITHOUT an inbox write so
        # the high-volume path stays cheap. NEVER act on a non-matching event.
        if not self._handler.matches(decoded):
            msg.ack()
            return

        # Step 3 — dedup key. Prefer Pub/Sub attributes (canonical Chora
        # location), then the decoded envelope, then the stable audit_id.
        env = decoded.get("envelope") if isinstance(decoded, dict) else None
        dedup_key = (
            msg.attributes.get("idempotency_key")
            or msg.attributes.get("event_id")
            or (env.get("idempotency_key") if isinstance(env, dict) else "")
            or (env.get("event_id") if isinstance(env, dict) else "")
            or str(decoded.get("audit_id", ""))
            or ""
        )
        if not dedup_key:
            logger.warning(
                "audit_recorded_consumer.missing_dedup_key",
                extra={"attributes": dict(msg.attributes)},
            )
            msg.nack()
            return

        # Step 4 — thread the trace context (attributes win; else the decoded
        # envelope already surfaced traceparent/tracestate top-level).
        tp = msg.attributes.get("traceparent", "") or decoded.get("traceparent", "")
        if tp:
            decoded["traceparent"] = tp
        ts = msg.attributes.get("tracestate", "") or decoded.get("tracestate", "")
        if ts:
            decoded["tracestate"] = ts

        # Step 5 — inbox dedupe → handler → ack-after-processing.
        async def _invoke() -> None:
            await self._handler.handle(decoded)

        try:
            ran = await self._inbox.process(
                key=dedup_key,
                ttl=self._inbox_ttl,
                fn=_invoke,
            )
        except Exception:
            logger.exception(
                "audit_recorded_consumer.handler_failed",
                extra={
                    "dedup_key": dedup_key,
                    "tenant_id": decoded.get("tenant_id", ""),
                },
            )
            msg.nack()
            return

        msg.ack()
        logger.info(
            "audit_recorded_consumer.acked",
            extra={
                "dedup_key": dedup_key,
                "ran": ran,  # False on dedupe-hit
                "target_resource_uri": decoded.get("target_resource_uri", ""),
            },
        )


__all__ = ["DEFAULT_INBOX_TTL", "AuditRecordedConsumer"]
