"""AgentCompletionLoop — NATS JetStream pull-consumer over the agent completion
lanes (ADR-253 D2/D5).

Replaces the ``AgentCompletionPubsubLoop`` StreamingPull wrapper. One NATS
subject per agent role, so one pull consumer per role, all fed to one
``AgentCompletionSubscriber``. Pull on BOTH sides of the dispatch pair, which
is why this lane needs no gateway push route.

The callback runs on the asyncio loop (NATS is async-native, so no threadpool
marshalling is needed) under the callback timeout; a raise or timeout NACKs
so JetStream redelivers. The subscriber owns the happy-path ack/nack.

⚠ The per-message callback wait here is deliberately SHORT compared with the
request lanes'. A completion resume is a checkpoint read, a few supersteps and a
checkpoint write — it does not contain a model call, because that already
happened in the agent. If this timeout ever needs to be large, something has
put a model call back on the resume path.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.adapter.pubsub.nats import NatsConsumerLoop

logger = logging.getLogger(__name__)

ENV_NATS_URL = "NATS_URL"
ENV_COMPLETION_SUBSCRIPTIONS = "AGENT_COMPLETION_SUBSCRIPTIONS"
ENV_CALLBACK_TIMEOUT = "AGENT_COMPLETION_CALLBACK_TIMEOUT_SECONDS"

_SUBJECT_PREFIX = "chora-ai-kernel-orchestrator.agent-dispatch"

# Per-message callback wait. TWO CALLERS, and the justification differs:
#
#   * the COMPLETION side (this loop's own subjects, agent-dispatch-*-
#     completed). A resume is checkpoint-read → supersteps → checkpoint-write,
#     with no model call in it. 60s is generous for that and still short enough
#     that a wedged resume surfaces as a redelivery rather than a held message.
#   * the REQUEST side of the generic single-agent lanes, which reuse this loop
#     (single_agent_lanes_wiring). That callback is decode → inbox → graph-to-
#     park → ack; PubSubAgentExecutor PUBLISHES rather than calling a model, so
#     it is DB-bound too and 60s is likewise generous. Those lanes now DECLARE
#     the number per lane (LaneSpec.callback_timeout_s) instead of inheriting
#     this default silently.
#
# Note this constant, not the consumer ack wait, is the binding bound on the
# healthy path: JetStream redelivers an un-acked message after its ack wait, so
# this wait caps first. Reached via from_env() only; the generic lanes pass
# their own value.
DEFAULT_CALLBACK_TIMEOUT_S = 60.0


def completion_subscription_name(agent_role: str) -> str:
    """Subject this orchestrator consumes an agent role's completions from."""
    role = (agent_role or "").strip()
    if not role:
        raise ValueError("completion_subscription_name: agent_role required")
    return f"{_SUBJECT_PREFIX}-{role.replace('_', '-')}-completed"


class _SubscriberLike(Protocol):
    async def handle_message(self, msg: Any) -> None: ...


class AgentCompletionPubsubLoop:
    """NATS JetStream pull-consumer fanning N completion subjects into one handler."""

    def __init__(
        self,
        *,
        subscriber: _SubscriberLike,
        project: str,
        subscriptions: Iterable[str],
        callback_timeout_s: float = DEFAULT_CALLBACK_TIMEOUT_S,
    ) -> None:
        if not (project or "").strip():
            raise ValueError("project required")
        subs = [s.strip() for s in subscriptions if (s or "").strip()]
        if not subs:
            raise ValueError(
                "AgentCompletionPubsubLoop: at least one subscription is "
                "required; a completion loop with nothing to listen on starts "
                "clean and parks every run forever"
            )
        self._subscriber = subscriber
        self._project = project
        self._subscriptions = subs
        self._callback_timeout_s = (
            callback_timeout_s if callback_timeout_s and callback_timeout_s > 0 else DEFAULT_CALLBACK_TIMEOUT_S
        )
        self._url: str | None = None
        self._impl: NatsConsumerLoop | None = None

    @property
    def subscriptions(self) -> list[str]:
        return list(self._subscriptions)

    @property
    def project(self) -> str:
        return self._project

    @classmethod
    def from_env(
        cls,
        *,
        subscriber: _SubscriberLike,
        agent_roles: Iterable[str],
    ) -> AgentCompletionPubsubLoop | None:
        """Build from env; None when NATS_URL is unset.

        ``AGENT_COMPLETION_SUBSCRIPTIONS`` (comma-separated) overrides the
        per-role default so a deployment can repoint a lane without a code
        change ([[secrets-and-env]]).
        """
        url = (os.getenv(ENV_NATS_URL) or "").strip()
        if not url:
            return None
        override = (os.getenv(ENV_COMPLETION_SUBSCRIPTIONS) or "").strip()
        if override:
            subscriptions = [s.strip() for s in override.split(",") if s.strip()]
        else:
            subscriptions = [completion_subscription_name(r) for r in agent_roles]
        loop = cls(
            subscriber=subscriber,
            project="chora-ai-kernel",
            subscriptions=subscriptions,
            callback_timeout_s=_callback_timeout_from_env(),
        )
        loop._url = url
        return loop

    async def start(self) -> None:  # pragma: no cover — exercised in integration
        """Open the JetStream pull consumers. Non-blocking."""
        url = self._url or (os.getenv(ENV_NATS_URL) or "").strip()
        if not url:
            raise RuntimeError("agent_completion_loop: NATS_URL unset")
        self._impl = NatsConsumerLoop(
            url=url,
            subjects=self._subscriptions,
            subscriber=self._subscriber,
            callback_timeout_s=self._callback_timeout_s,
        )
        await self._impl.start()
        logger.info(
            "agent_completion_loop.started",
            extra={"subscriptions": self._subscriptions},
        )

    async def stop(self) -> None:  # pragma: no cover — exercised in integration
        """Stop the pull consumers + tear down the connection."""
        if self._impl is not None:
            await self._impl.stop()
            self._impl = None


def _callback_timeout_from_env() -> float:
    raw = (os.getenv(ENV_CALLBACK_TIMEOUT) or "").strip()
    if not raw:
        return DEFAULT_CALLBACK_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "agent_completion_loop.invalid_callback_timeout",
            extra={"value": raw, "default": DEFAULT_CALLBACK_TIMEOUT_S},
        )
        return DEFAULT_CALLBACK_TIMEOUT_S
    return value if value > 0 else DEFAULT_CALLBACK_TIMEOUT_S


__all__ = [
    "AgentCompletionPubsubLoop",
    "DEFAULT_CALLBACK_TIMEOUT_S",
    "ENV_COMPLETION_SUBSCRIPTIONS",
    "ENV_NATS_URL",
    "completion_subscription_name",
]
