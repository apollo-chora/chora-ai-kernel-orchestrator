"""Tests for OEGradingCrewPubsubLoop construction + from_env factory.

The actual ``start`` / ``stop`` lifecycle is exercised in integration
(production wiring lands in oe_grading_crew_wiring.py); these tests verify
the construction contract + the env-var factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.oe_grading_crew_loop import (
    DEFAULT_CALLBACK_TIMEOUT_S,
    DEFAULT_SUBSCRIPTION,
    ENV_CALLBACK_TIMEOUT,
    OEGradingCrewPubsubLoop,
)


@dataclass
class _FakeSubscriber:
    async def handle_message(self, msg: Any) -> None:  # pragma: no cover
        return None


class TestConstruction:
    def test_requires_project(self) -> None:
        with pytest.raises(ValueError, match="project"):
            OEGradingCrewPubsubLoop(
                subscriber=_FakeSubscriber(),
                project="",
                subscription="x",
            )

    def test_requires_subscription(self) -> None:
        with pytest.raises(ValueError, match="subscription"):
            OEGradingCrewPubsubLoop(
                subscriber=_FakeSubscriber(),
                project="chora-489812",
                subscription="",
            )


class TestFromEnv:
    def test_none_when_nats_url_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NATS_URL", raising=False)
        monkeypatch.delenv("OE_GRADING_CREW_SUBSCRIPTION", raising=False)
        assert OEGradingCrewPubsubLoop.from_env(subscriber=_FakeSubscriber()) is None

    def test_defaults_to_canonical_subscription(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.delenv("OE_GRADING_CREW_SUBSCRIPTION", raising=False)
        loop = OEGradingCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        # The canonical contract topic (chora-contracts/proto/events/delivery/
        # grading.proto §GradingSubmissionRequested) — NOT the legacy Pub/Sub
        # subscription id, which is outside the chora.> JetStream stream and
        # can never deliver.
        assert DEFAULT_SUBSCRIPTION == "chora.delivery.grading.submission_requested.v1"
        assert loop._subscription == DEFAULT_SUBSCRIPTION  # noqa: SLF001
        assert loop._project == "chora-ai-kernel"  # noqa: SLF001
        assert loop._url == "nats://nats:4222"  # noqa: SLF001

    def test_subscription_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv("OE_GRADING_CREW_SUBSCRIPTION", "chora-ai-kernel.custom-oe-sub")
        loop = OEGradingCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._subscription == (  # noqa: SLF001
            "chora-ai-kernel.custom-oe-sub"
        )


class TestCallbackTimeout:
    def test_default_is_generous(self) -> None:
        loop = OEGradingCrewPubsubLoop(
            subscriber=_FakeSubscriber(),
            project="chora-489812",
            subscription="x",
        )
        assert loop._callback_timeout_s == DEFAULT_CALLBACK_TIMEOUT_S  # noqa: SLF001

    def test_explicit_param(self) -> None:
        loop = OEGradingCrewPubsubLoop(
            subscriber=_FakeSubscriber(),
            project="chora-489812",
            subscription="x",
            callback_timeout_s=180.0,
        )
        assert loop._callback_timeout_s == 180.0  # noqa: SLF001

    def test_from_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv(ENV_CALLBACK_TIMEOUT, "420")
        loop = OEGradingCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._callback_timeout_s == 420.0  # noqa: SLF001

    def test_from_env_invalid_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv(ENV_CALLBACK_TIMEOUT, "not-a-number")
        loop = OEGradingCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._callback_timeout_s == DEFAULT_CALLBACK_TIMEOUT_S  # noqa: SLF001

    def test_from_env_nonpositive_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv(ENV_CALLBACK_TIMEOUT, "0")
        loop = OEGradingCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._callback_timeout_s == DEFAULT_CALLBACK_TIMEOUT_S  # noqa: SLF001
