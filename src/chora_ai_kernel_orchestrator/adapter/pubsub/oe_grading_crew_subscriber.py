"""OEGradingCrewSubscriber — Pub/Sub subscriber for ADR-172 submission_requested.v1.

Clone of qgen_crew_subscriber.py for the OE grading crew. Subscribes to
``chora.delivery.grading.submission_requested.v1`` and routes each message
through:

    inbox.process(dedup_key, ttl, fn=lambda: runner.handle_requested(event))

chora-delivery emits submission_requested.v1 as a JSON body (the assessment
handler builds a ``map[string]any`` and publishes via the delivery outbox), with
envelope fields (event_id / tenant_id / traceparent / …) on Pub/Sub message
ATTRIBUTES. So this subscriber JSON-decodes the body and reads the dedup key +
trace context from attributes (falling back to a body-nested envelope). Unlike
qgen there is no binary-proto path — a non-JSON body fails loud (NACK) rather
than being silently mis-decoded.

D6 4-pillar contract per [[agentic-resilience-d6]]:
  * P1 pod-death survival — runner.handle_requested runs graph.ainvoke under
    PostgresSaver (thread_id=submission_id); a retry-delivered message resumes.
  * P2 idempotency — InboxIdempotencyStore (first line) + the OE outbox writer's
    ON CONFLICT (idempotency_key) DO NOTHING (safety net).
  * P3 delivery resilience — ACK only after the runner returns; NACK on error →
    Pub/Sub redelivers up to max_delivery_attempts then routes to the DLQ.
  * P4 OTel trace — traceparent/tracestate from attributes thread into the event.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Default inbox TTL — matches the qgen subscriber (7 days; covers Pub/Sub's max
# redelivery window).
DEFAULT_INBOX_TTL = _dt.timedelta(days=7)


class _RunnerLike(Protocol):
    """Duck-typed OEGradingCrewRunner from orchestrators/oe_grading_crew_runner.py."""

    async def handle_requested(self, event: dict[str, Any]) -> None: ...


class _InboxLike(Protocol):
    """Duck-typed InboxIdempotencyStore."""

    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool: ...


class _MessageLike(Protocol):
    """Duck-typed Pub/Sub message — ack/nack-able with data + attributes."""

    data: bytes
    attributes: dict[str, str]

    def ack(self) -> None: ...
    def nack(self) -> None: ...


class OEGradingCrewSubscriber:
    """Wraps OEGradingCrewRunner with inbox dedupe + ack-after-processing."""

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

        ACKs on successful processing OR inbox dedupe-hit; NACKs on decode
        failure, missing dedup key, or runner exception. Never raises (that would
        tear down the StreamingPull subscriber).
        """
        # Step 1: JSON-decode the submission_requested.v1 body.
        try:
            decoded = json.loads(msg.data.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("expected JSON object body")
            event: dict[str, Any] = decoded
        except (ValueError, UnicodeDecodeError) as exc:
            logger.exception(
                "oe_grading_crew_subscriber.decode_failed",
                extra={"err": str(exc), "attributes": dict(msg.attributes)},
            )
            msg.nack()
            return

        # Step 2: extract dedup_key — prefer Pub/Sub attributes, fall back to a
        # body-nested envelope.
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
                "oe_grading_crew_subscriber.missing_dedup_key",
                extra={"attributes": dict(msg.attributes)},
            )
            msg.nack()
            return

        # Step 3a: merge envelope fields that ride Pub/Sub ATTRIBUTES, not the
        # body. chora-delivery emits submission_requested.v1 via the outbox
        # PublishCustom(payload, tenantID, gcid) path: tenant_id (+ gcid) live on
        # the message envelope/attributes, NOT in the JSON body. parse_requested
        # reads tenant_id + learner_gcid from the event dict, so without this
        # merge the crew would dispatch with an EMPTY tenant_id → the agent
        # manaplugin refuses the run + RLS/cost attribution breaks. (gcid is also
        # in the body as learner_gcid; the merge is defensive.)
        attr_tenant = msg.attributes.get("tenant_id", "")
        if attr_tenant and not event.get("tenant_id"):
            event["tenant_id"] = attr_tenant
        attr_gcid = msg.attributes.get("gcid", "")
        if attr_gcid and not event.get("learner_gcid"):
            event["learner_gcid"] = attr_gcid

        # Step 3b: thread the trace context (attributes first; delivery emits
        # traceparent in the body too, tracestate only on attributes).
        tp = msg.attributes.get("traceparent", "") or event.get("traceparent", "")
        if tp:
            event["traceparent"] = tp
        ts = msg.attributes.get("tracestate", "") or event.get("tracestate", "")
        if ts:
            event["tracestate"] = ts

        # Step 4: inbox dedupe → invoke runner → ack-after-processing.
        async def _invoke() -> None:
            await self._runner.handle_requested(event)

        try:
            ran = await self._inbox.process(
                key=dedup_key,
                ttl=self._inbox_ttl,
                fn=_invoke,
            )
        except Exception:
            logger.exception(
                "oe_grading_crew_subscriber.runner_failed",
                extra={
                    "dedup_key": dedup_key,
                    "tenant_id": msg.attributes.get("tenant_id", ""),
                },
            )
            msg.nack()
            return

        msg.ack()
        logger.info(
            "oe_grading_crew_subscriber.acked",
            extra={
                "dedup_key": dedup_key,
                "ran": ran,  # False on dedupe-hit
                "tenant_id": msg.attributes.get("tenant_id", ""),
            },
        )


__all__ = ["DEFAULT_INBOX_TTL", "OEGradingCrewSubscriber"]
