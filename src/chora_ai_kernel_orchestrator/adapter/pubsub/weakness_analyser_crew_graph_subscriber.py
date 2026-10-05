"""WeaknessAnalyserCrewGraphSubscriber — Pub/Sub subscriber for the GRADUATED
(graph-mode) Growth-Edge explicit-upload path (ADR-205 WS-2).

The single-shot ``WeaknessAnalyserCrewSubscriber`` analyses + publishes
synchronously on one message. The graduated crew is a checkpointed LangGraph with
an in-graph HITL ``interrupt()``: this subscriber STARTS the run
(``crew_runner.handle_uploaded(event)`` drives the graph to the interrupt — a
paused, PostgresSaver-checkpointed run) and ACKs. The ``weakness.analyzed.v1``
publish happens LATER, on the FE resume (the graph's publish_analyzed node), so
this subscriber holds NO publisher.

Same contract as the single-shot path otherwise:
  * JSON inbound (Go consumption emits the uploaded event as a JSON body +
    envelope-on-attributes; no inbound ``_pb2`` skew).
  * tenant_id / learner_gcid / traceparent / tracestate merge from attributes.
  * idempotency keyed on the UPLOAD (a re-delivered upload must NOT re-start the
    paused run; the FE resume is keyed on the deterministic thread_id, not on
    Pub/Sub redelivery).
  * ack-after-processing; NACK on decode failure / missing key / any
    ``handle_uploaded`` exception (incl. WeaknessCrewError on a malformed event).

D6 4-pillar contract per [[agentic-resilience-d6]]:
  * P1 pod-death survival — the run is checkpointed at the interrupt; a resume
    after a pod restart finds it via the deterministic thread_id (no in-memory
    run index to lose).
  * P2 idempotency — InboxIdempotencyStore (keyed on upload) + the graph's
    publish_analyzed ``ON CONFLICT`` net.
  * P3 delivery resilience — ACK only after the run reaches the interrupt; NACK →
    Pub/Sub redelivers → DLQ after max attempts.
  * P4 OTel trace — traceparent/tracestate from attributes thread into the event
    → the graph nodes → the analyzed event envelope.

Kept SEPARATE from the single-shot subscriber so the cut-over is a one-line
composition-root swap (no edit churn on the proven single-shot file).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Inbox TTL — 7 days, covering Pub/Sub's max redelivery window (matches the
# single-shot subscriber + qgen/OE).
DEFAULT_INBOX_TTL = _dt.timedelta(days=7)


class _CrewRunnerLike(Protocol):
    """Duck-typed WeaknessAnalyserCrewRunner (orchestrators/...crew_runner.py)."""

    async def handle_uploaded(self, event: dict[str, Any]) -> Any: ...


class _ReviewPendingPublisherLike(Protocol):
    """Duck-typed WeaknessReviewPendingOutboxWriter — emits the HITL review panel
    as ``chora.consumption.weakness.review_pending.v1`` (CHO-1973 Wave A)."""

    async def publish_review_pending(
        self, *, panel: dict[str, Any], traceparent: str = "", tracestate: str = ""
    ) -> str: ...


class _InboxLike(Protocol):
    """Duck-typed InboxIdempotencyStore."""

    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool: ...


class _MessageLike(Protocol):
    """Duck-typed Pub/Sub message — ack/nack-able with data + attributes."""

    data: bytes
    attributes: dict[str, str]

    def ack(self) -> None: ...
    def nack(self) -> None: ...


class WeaknessAnalyserCrewGraphSubscriber:
    """Starts the checkpointed crew run with inbox dedupe + ack-after-processing."""

    def __init__(
        self,
        *,
        crew_runner: _CrewRunnerLike,
        inbox: _InboxLike,
        review_pending_publisher: _ReviewPendingPublisherLike | None = None,
        inbox_ttl: _dt.timedelta = DEFAULT_INBOX_TTL,
    ) -> None:
        self._runner = crew_runner
        self._inbox = inbox
        self._review_pending_publisher = review_pending_publisher
        self._inbox_ttl = inbox_ttl

    @property
    def inbox_ttl(self) -> _dt.timedelta:
        return self._inbox_ttl

    async def handle_message(self, msg: _MessageLike) -> None:
        """Top-level Pub/Sub message handler. Never raises (that would tear down
        the StreamingPull subscriber)."""
        # Step 1: JSON-decode the weakness_doc.uploaded.v1 body.
        try:
            decoded = json.loads(msg.data.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("expected JSON object body")
            event: dict[str, Any] = decoded
        except (ValueError, UnicodeDecodeError) as exc:
            logger.exception(
                "weakness_graph_subscriber.decode_failed",
                extra={"err": str(exc), "attributes": dict(msg.attributes)},
            )
            msg.nack()
            return

        # Step 2: merge envelope fields that ride Pub/Sub ATTRIBUTES (tenant_id /
        # gcid / trace) so the run carries tenant context + trace — without this
        # the graph would dispatch with an EMPTY tenant_id (mana/cost attribution
        # + downstream RLS scoping break).
        attr_tenant = msg.attributes.get("tenant_id", "")
        if attr_tenant and not event.get("tenant_id"):
            event["tenant_id"] = attr_tenant
        attr_gcid = msg.attributes.get("gcid", "")
        if attr_gcid and not event.get("learner_gcid"):
            event["learner_gcid"] = attr_gcid
        tp = msg.attributes.get("traceparent", "") or event.get("traceparent", "")
        if tp:
            event["traceparent"] = tp
        ts = msg.attributes.get("tracestate", "") or event.get("tracestate", "")
        if ts:
            event["tracestate"] = ts

        # Step 3: dedup key — prefer the upload_id (the crew's natural idempotency
        # unit), fall back to the Pub/Sub envelope keys.
        env_in_body = event.get("envelope") if isinstance(event, dict) else None
        upload_id = str(event.get("upload_id", "")).strip()
        dedup_key = (
            f"weakness.uploaded:{upload_id}"
            if upload_id
            else (
                msg.attributes.get("idempotency_key")
                or msg.attributes.get("event_id")
                or (env_in_body.get("idempotency_key") if isinstance(env_in_body, dict) else "")
                or (env_in_body.get("event_id") if isinstance(env_in_body, dict) else "")
                or ""
            )
        )
        if not dedup_key:
            logger.warning(
                "weakness_graph_subscriber.missing_dedup_key",
                extra={"attributes": dict(msg.attributes)},
            )
            msg.nack()
            return

        # Step 4: inbox dedupe → start the checkpointed run → (on the HITL
        # interrupt) emit review_pending → ack-after-processing. The emit runs
        # INSIDE the dedupe guard so a re-delivered upload never re-emits, and a
        # publisher failure NACKs the whole start (never a half-emit + ack).
        async def _invoke() -> None:
            result = await self._runner.handle_uploaded(event)
            if not getattr(result, "interrupted", False):
                return  # BLOCK / fail-loud / completed terminal — no panel to emit
            if getattr(result, "awaiting_agent", False):
                # ADR-253: parked on an AGENT DISPATCH, not on a human. The run
                # is checkpointed and its dispatch outbox row committed in the
                # same transaction (D3a); the completion consumer resumes the
                # thread. There is no panel yet and there SHOULD not be one.
                #
                # Returning here means the inbound message is ACKed. That is the
                # point: a NACK would redeliver the upload and re-run the whole
                # graph from the top, including the paid extract call, which is
                # exactly what happened on 2026-08-21 (five dispatches for one
                # upload, stopping at maxDeliveryAttempts).
                #
                # Logged at INFO, not WARNING: on the Pub/Sub transport this is
                # the normal path for every diagnosis, and a warning that fires
                # on every run teaches everyone to ignore the one that matters.
                logger.info(
                    "weakness_graph_subscriber.parked_on_agent_dispatch",
                    extra={
                        "thread_id": getattr(result, "thread_id", ""),
                        "dedup_key": dedup_key,
                    },
                )
                return
            panel = getattr(result, "review_panel", None)
            if not panel:
                logger.warning(
                    "weakness_graph_subscriber.interrupt_without_panel",
                    extra={"tenant_id": event.get("tenant_id", ""), "dedup_key": dedup_key},
                )
                return
            if self._review_pending_publisher is None:
                logger.warning(
                    "weakness_graph_subscriber.review_pending_publisher_unconfigured "
                    "(graph mode reached the HITL interrupt but no review_pending writer "
                    "is wired — the FE will not receive the review panel event)",
                    extra={"tenant_id": event.get("tenant_id", ""), "dedup_key": dedup_key},
                )
                return
            await self._review_pending_publisher.publish_review_pending(
                panel=panel,
                traceparent=event.get("traceparent", ""),
                tracestate=event.get("tracestate", ""),
            )

        try:
            ran = await self._inbox.process(key=dedup_key, ttl=self._inbox_ttl, fn=_invoke)
        except Exception:
            logger.exception(
                "weakness_graph_subscriber.run_failed",
                extra={"dedup_key": dedup_key, "tenant_id": event.get("tenant_id", "")},
            )
            msg.nack()
            return

        msg.ack()
        logger.info(
            "weakness_graph_subscriber.acked",
            extra={
                "dedup_key": dedup_key,
                "ran": ran,  # False on dedupe-hit
                "tenant_id": event.get("tenant_id", ""),
            },
        )


__all__ = ["DEFAULT_INBOX_TTL", "WeaknessAnalyserCrewGraphSubscriber"]
