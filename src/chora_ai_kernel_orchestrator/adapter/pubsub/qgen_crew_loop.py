"""QGenCrewLoop — NATS JetStream pull-consumer over the qgen started.v1 lane.

Replaces the ``QGenCrewPubsubLoop`` StreamingPull wrapper. Wires a NATS
JetStream pull consumer to a ``QGenCrewSubscriber``: the subscription name is
the NATS subject, the envelope headers ride as NATS message headers, and the
ack-after-processing + callback-timeout + nack-on-failure discipline is
unchanged from the StreamingPull loop it replaces.

Per [[secrets-and-env]] the subject + NATS URL come from env vars:

    NATS_URL                  — NATS connection URL (required)
    QGEN_CREW_SUBSCRIPTION    — NATS subject (default: the canonical
                                 chora.creation.ai_assist.started.v2)

Wire site: ``main.py`` lifespan starts the loop alongside the OutboxDispatcher
task. On shutdown both halt cleanly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.adapter.pubsub.nats import NatsConsumerLoop

logger = logging.getLogger(__name__)

# Canonical inbound subject per chora-contracts/asyncapi/creation/
# ai-assist-started-v2.yaml (ADR-195 WS7 D7): chora-creation publishes the
# job-started event on ``chora.creation.ai_assist.started.v2``. The legacy
# Pub/Sub subscription id is NOT a NATS subject (the live JetStream stream
# captures only ``chora.>``), so a consumer bound to it never receives a job.
DEFAULT_SUBSCRIPTION = "chora.creation.ai_assist.started.v2"

ENV_NATS_URL = "NATS_URL"
ENV_QGEN_CREW_SUBSCRIPTION = "QGEN_CREW_SUBSCRIPTION"
ENV_CALLBACK_TIMEOUT = "QGEN_CREW_CALLBACK_TIMEOUT_SECONDS"

# Per-message callback wait. The qgen crew runs a multi-attempt actor↔critic
# loop (max 4 attempts × {assurance + generation + evaluation + critic}); with
# the high tier (gemini-3.1-pro-preview generation+evaluation ~15s each) a
# 3-attempt run exceeds 120s wall-clock. The callback timeout is the real
# bound (NATS redelivers an un-acked message after its ack wait); 300s gives
# the looped high-tier run comfortable headroom. Tunable via
# QGEN_CREW_CALLBACK_TIMEOUT_SECONDS.
DEFAULT_CALLBACK_TIMEOUT_S = 300.0


def _callback_timeout_from_env() -> float:
    """Read QGEN_CREW_CALLBACK_TIMEOUT_SECONDS; fall back to the default on
    unset / non-numeric / non-positive."""
    raw = (os.getenv(ENV_CALLBACK_TIMEOUT) or "").strip()
    if not raw:
        return DEFAULT_CALLBACK_TIMEOUT_S
    try:
        val = float(raw)
    except ValueError:
        logger.warning(
            "qgen_crew_loop.invalid_callback_timeout",
            extra={"value": raw, "default": DEFAULT_CALLBACK_TIMEOUT_S},
        )
        return DEFAULT_CALLBACK_TIMEOUT_S
    return val if val > 0 else DEFAULT_CALLBACK_TIMEOUT_S


class _SubscriberLike(Protocol):
    """Duck-typed QGenCrewSubscriber."""

    async def handle_message(self, msg: Any) -> None: ...


class QGenCrewPubsubLoop:
    """NATS JetStream pull-consumer wrapper around QGenCrewSubscriber."""

    def __init__(
        self,
        *,
        subscriber: _SubscriberLike,
        project: str,
        subscription: str = DEFAULT_SUBSCRIPTION,
        callback_timeout_s: float = DEFAULT_CALLBACK_TIMEOUT_S,
    ) -> None:
        if not (project or "").strip():
            raise ValueError("project required")
        if not (subscription or "").strip():
            raise ValueError("subscription required")
        self._subscriber = subscriber
        self._project = project
        self._subscription = subscription
        self._callback_timeout_s = (
            callback_timeout_s if callback_timeout_s and callback_timeout_s > 0 else DEFAULT_CALLBACK_TIMEOUT_S
        )
        self._url: str | None = None
        self._impl: NatsConsumerLoop | None = None

    @classmethod
    def from_env(cls, *, subscriber: _SubscriberLike) -> QGenCrewPubsubLoop | None:
        """Build from env vars; returns None when NATS_URL is unset
        (orchestrator can still serve sync /orchestrate requests)."""
        url = (os.getenv(ENV_NATS_URL) or "").strip()
        if not url:
            return None
        subscription = (os.getenv(ENV_QGEN_CREW_SUBSCRIPTION) or DEFAULT_SUBSCRIPTION).strip()
        loop = cls(
            subscriber=subscriber,
            project="chora-ai-kernel",
            subscription=subscription,
            callback_timeout_s=_callback_timeout_from_env(),
        )
        loop._url = url
        return loop

    async def start(self) -> None:  # pragma: no cover — exercised in integration
        """Open the JetStream pull consumer. Non-blocking; returns once the
        consumer is registered."""
        url = self._url or (os.getenv(ENV_NATS_URL) or "").strip()
        if not url:
            raise RuntimeError("qgen_crew_loop: NATS_URL unset")
        self._impl = NatsConsumerLoop(
            url=url,
            subjects=[self._subscription],
            subscriber=self._subscriber,
            callback_timeout_s=self._callback_timeout_s,
        )
        await self._impl.start()
        logger.info(
            "qgen_crew_loop.started",
            extra={"subscription": self._subscription},
        )

    async def stop(self) -> None:  # pragma: no cover — exercised in integration
        """Stop the pull consumer + tear down the connection."""
        if self._impl is not None:
            await self._impl.stop()
            self._impl = None


__all__ = [
    "DEFAULT_CALLBACK_TIMEOUT_S",
    "DEFAULT_SUBSCRIPTION",
    "ENV_CALLBACK_TIMEOUT",
    "ENV_NATS_URL",
    "ENV_QGEN_CREW_SUBSCRIPTION",
    "QGenCrewPubsubLoop",
]
