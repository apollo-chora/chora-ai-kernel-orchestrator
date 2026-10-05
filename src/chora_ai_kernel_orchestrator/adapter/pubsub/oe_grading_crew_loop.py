"""OEGradingCrewLoop — NATS JetStream pull-consumer over the OE submission lane.

Replaces the ``OEGradingCrewPubsubLoop`` StreamingPull wrapper. The
subscription name is the NATS subject, the envelope headers ride as NATS
message headers, and the ack-after-processing + callback-timeout +
nack-on-failure discipline is unchanged.

Per [[secrets-and-env]] the subject + NATS URL come from env vars:

    NATS_URL                     — NATS connection URL (required)
    OE_GRADING_CREW_SUBSCRIPTION — NATS subject (default: the canonical
                                   submission-requested subject)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.adapter.pubsub.nats import NatsConsumerLoop

logger = logging.getLogger(__name__)

DEFAULT_SUBSCRIPTION = "chora-ai-kernel-orchestrator.oe-grading-submission-requested"

ENV_NATS_URL = "NATS_URL"
ENV_OE_CREW_SUBSCRIPTION = "OE_GRADING_CREW_SUBSCRIPTION"
ENV_CALLBACK_TIMEOUT = "OE_GRADING_CREW_CALLBACK_TIMEOUT_SECONDS"

# Per-message callback wait. The OE crew runs a per-OE-answer evaluator↔moderator
# loop (≤2 iterations) across multiple answers + an assess_summary pass; with the
# high-tier evaluator (gemini-3.1-pro-preview) a multi-answer looped run far
# exceeds the 120s default. The callback timeout is the real bound (NATS
# redelivers an un-acked message after its ack wait); 300s gives headroom.
# Tunable via OE_GRADING_CREW_CALLBACK_TIMEOUT_SECONDS.
DEFAULT_CALLBACK_TIMEOUT_S = 300.0


def _callback_timeout_from_env() -> float:
    """Read OE_GRADING_CREW_CALLBACK_TIMEOUT_SECONDS; fall back to default on
    unset / non-numeric / non-positive."""
    raw = (os.getenv(ENV_CALLBACK_TIMEOUT) or "").strip()
    if not raw:
        return DEFAULT_CALLBACK_TIMEOUT_S
    try:
        val = float(raw)
    except ValueError:
        logger.warning(
            "oe_grading_crew_loop.invalid_callback_timeout",
            extra={"value": raw, "default": DEFAULT_CALLBACK_TIMEOUT_S},
        )
        return DEFAULT_CALLBACK_TIMEOUT_S
    return val if val > 0 else DEFAULT_CALLBACK_TIMEOUT_S


class _SubscriberLike(Protocol):
    """Duck-typed OEGradingCrewSubscriber."""

    async def handle_message(self, msg: Any) -> None: ...


class OEGradingCrewPubsubLoop:
    """NATS JetStream pull-consumer wrapper around OEGradingCrewSubscriber."""

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
    def from_env(cls, *, subscriber: _SubscriberLike) -> OEGradingCrewPubsubLoop | None:
        """Build from env vars; returns None when NATS_URL is unset."""
        url = (os.getenv(ENV_NATS_URL) or "").strip()
        if not url:
            return None
        subscription = (os.getenv(ENV_OE_CREW_SUBSCRIPTION) or DEFAULT_SUBSCRIPTION).strip()
        loop = cls(
            subscriber=subscriber,
            project="chora-ai-kernel",
            subscription=subscription,
            callback_timeout_s=_callback_timeout_from_env(),
        )
        loop._url = url
        return loop

    async def start(self) -> None:  # pragma: no cover — exercised in integration
        """Open the JetStream pull consumer. Non-blocking."""
        url = self._url or (os.getenv(ENV_NATS_URL) or "").strip()
        if not url:
            raise RuntimeError("oe_grading_crew_loop: NATS_URL unset")
        self._impl = NatsConsumerLoop(
            url=url,
            subjects=[self._subscription],
            subscriber=self._subscriber,
            callback_timeout_s=self._callback_timeout_s,
        )
        await self._impl.start()
        logger.info(
            "oe_grading_crew_loop.started",
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
    "ENV_OE_CREW_SUBSCRIPTION",
    "OEGradingCrewPubsubLoop",
]
