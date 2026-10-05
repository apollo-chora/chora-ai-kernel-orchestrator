"""AgentCompletionSubscriber — resumes a parked run from one agent completion
(ADR-253 D2).

The other half of the suspend-and-resume pair. The dispatching node parked the
graph and returned; this subscriber consumes the agent's answer and resumes that
thread by the key the completion carries, exactly as the growth-edge HITL
interrupt is resumed from the FE today.

It holds no state. The thread key rides the event because the agent echoes back
the ``thread_id`` the orchestrator stamped on the request, so a completion can be
resumed by ANY orchestrator pod — which is the whole point of parking rather
than blocking.

D6 4-pillar contract:
  * P1 pod-death survival — the run lives in the checkpoint, not in a process.
  * P2 idempotency — ``InboxIdempotencyStore`` first, plus the dispatch outbox's
    ``ON CONFLICT (idempotency_key) DO NOTHING`` behind it. Still mandatory even
    though the publish side commits in one transaction (ADR-253 D3a): Pub/Sub is
    at-least-once regardless of how cleanly the writer committed.
  * P3 delivery resilience — ACK after the resume returns; NACK on error, then
    the DLQ at 5 attempts.
  * P4 OTel trace — traceparent/tracestate off the attributes into the resume.

⚠ Verification for this lane cannot lean on logs: the `buildout-cost-pause`
exclusion on the `_Default` sink drops severity>=DEFAULT application logs
platform wide, so a subscriber that silently failed to start looks exactly like
a healthy one. Check ack counts and Cloud Trace, never the absence of a log line.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Matches the request-side subscriber: 7 days covers Pub/Sub's max redelivery
# window, so a dedupe entry outlives every retry of the message it guards.
DEFAULT_INBOX_TTL = _dt.timedelta(days=7)


class _RunnerLike(Protocol):
    async def handle_completion(self, completion: dict[str, Any]) -> None: ...


class _InboxLike(Protocol):
    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool: ...


class _MessageLike(Protocol):
    data: bytes
    attributes: dict[str, str]

    def ack(self) -> None: ...
    def nack(self) -> None: ...


class AgentCompletionSubscriber:
    """Wraps a crew runner's ``handle_completion`` with dedupe + ack policy."""

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
        """Never raises — a raise here tears down the StreamingPull subscriber
        for every lane sharing the client."""
        try:
            decoded = json.loads(msg.data.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("expected a JSON object body")
        except (ValueError, UnicodeDecodeError) as exc:
            logger.exception(
                "agent_completion_subscriber.decode_failed",
                extra={"err": str(exc), "attributes": dict(msg.attributes)},
            )
            msg.nack()
            return

        completion: dict[str, Any] = decoded

        # The completion's OWN key, not the request's. Both sides write to one
        # idempotency_keys table, so reusing the request key here would make the
        # first completion look like an already-seen request and swallow it.
        dedup_key = (
            msg.attributes.get("idempotency_key") or msg.attributes.get("event_id") or _completion_key(completion)
        )
        if not dedup_key:
            logger.warning(
                "agent_completion_subscriber.missing_dedup_key",
                extra={"attributes": dict(msg.attributes)},
            )
            msg.nack()
            return

        # Trace context: attributes win, body is the fallback. Without this the
        # resumed half of the run starts a new trace and one run stops being one
        # trace across the Pub/Sub hop.
        tp = msg.attributes.get("traceparent", "") or completion.get("traceparent", "")
        if tp:
            completion["traceparent"] = tp
        ts = msg.attributes.get("tracestate", "") or completion.get("tracestate", "")
        if ts:
            completion["tracestate"] = ts
        attr_tenant = msg.attributes.get("tenant_id", "")
        if attr_tenant and not completion.get("tenant_id"):
            completion["tenant_id"] = attr_tenant

        async def _invoke() -> None:
            await self._runner.handle_completion(completion)

        try:
            ran = await self._inbox.process(
                key=dedup_key,
                ttl=self._inbox_ttl,
                fn=_invoke,
            )
        except Exception:
            logger.exception(
                "agent_completion_subscriber.resume_failed",
                extra={
                    "dedup_key": dedup_key,
                    "thread_id": completion.get("thread_id", ""),
                    "agent_role": completion.get("agent_role", ""),
                },
            )
            msg.nack()
            return

        msg.ack()
        logger.info(
            "agent_completion_subscriber.acked",
            extra={
                "dedup_key": dedup_key,
                "ran": ran,  # False on a dedupe hit
                "thread_id": completion.get("thread_id", ""),
                "agent_role": completion.get("agent_role", ""),
                "status": completion.get("status", ""),
            },
        )


def _completion_key(completion: dict[str, Any]) -> str:
    """Body-derived fallback when the publisher set no attribute key."""
    request_key = str(completion.get("idempotency_key") or "").strip()
    return f"{request_key}.completed" if request_key else ""


__all__ = ["DEFAULT_INBOX_TTL", "AgentCompletionSubscriber"]
