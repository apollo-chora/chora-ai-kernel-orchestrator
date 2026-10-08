"""Tests for WeaknessAnalyserCrewPubsubLoop construction + from_env factory.

The ``start`` / ``stop`` StreamingPull lifecycle is exercised in integration
(production wiring lands in main.py); these verify the construction contract +
the env-var factory + callback-timeout parsing. Mirrors test_qgen_crew_loop.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyser_crew_loop import (
    DEFAULT_CALLBACK_TIMEOUT_S,
    DEFAULT_SUBSCRIPTION,
    ENV_CALLBACK_TIMEOUT,
    ENV_WEAKNESS_SUBSCRIPTION,
    WeaknessAnalyserCrewPubsubLoop,
)


@dataclass
class _FakeSubscriber:
    async def handle_message(self, msg: Any) -> None:  # pragma: no cover
        return None


class TestConstruction:
    def test_requires_project(self) -> None:
        with pytest.raises(ValueError, match="project"):
            WeaknessAnalyserCrewPubsubLoop(subscriber=_FakeSubscriber(), project="", subscription="x")

    def test_requires_subscription(self) -> None:
        with pytest.raises(ValueError, match="subscription"):
            WeaknessAnalyserCrewPubsubLoop(subscriber=_FakeSubscriber(), project="chora-489812", subscription="")


class TestFromEnv:
    def test_none_when_nats_url_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NATS_URL", raising=False)
        monkeypatch.delenv(ENV_WEAKNESS_SUBSCRIPTION, raising=False)
        assert WeaknessAnalyserCrewPubsubLoop.from_env(subscriber=_FakeSubscriber()) is None

    def test_defaults_to_canonical_subscription(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.delenv(ENV_WEAKNESS_SUBSCRIPTION, raising=False)
        loop = WeaknessAnalyserCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        # Pinned to the contract topic, not the legacy Pub/Sub subscription id.
        assert DEFAULT_SUBSCRIPTION == "chora.consumption.weakness_doc.uploaded.v1"
        assert loop._subscription == DEFAULT_SUBSCRIPTION  # noqa: SLF001
        assert loop._project == "chora-ai-kernel"  # noqa: SLF001

    def test_subscription_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv(ENV_WEAKNESS_SUBSCRIPTION, "chora-ai-kernel.custom-weakness-sub")
        loop = WeaknessAnalyserCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._subscription == "chora-ai-kernel.custom-weakness-sub"  # noqa: SLF001


class TestCallbackTimeout:
    def test_default_is_generous(self) -> None:
        loop = WeaknessAnalyserCrewPubsubLoop(subscriber=_FakeSubscriber(), project="chora-489812", subscription="x")
        assert loop._callback_timeout_s == DEFAULT_CALLBACK_TIMEOUT_S  # noqa: SLF001
        # The multimodal call over a large marked-test PDF can be slow.
        assert DEFAULT_CALLBACK_TIMEOUT_S >= 180

    def test_explicit_param(self) -> None:
        loop = WeaknessAnalyserCrewPubsubLoop(
            subscriber=_FakeSubscriber(),
            project="chora-489812",
            subscription="x",
            callback_timeout_s=200.0,
        )
        assert loop._callback_timeout_s == 200.0  # noqa: SLF001

    def test_from_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv(ENV_CALLBACK_TIMEOUT, "420")
        loop = WeaknessAnalyserCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._callback_timeout_s == 420.0  # noqa: SLF001

    def test_from_env_invalid_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv(ENV_CALLBACK_TIMEOUT, "not-a-number")
        loop = WeaknessAnalyserCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._callback_timeout_s == DEFAULT_CALLBACK_TIMEOUT_S  # noqa: SLF001

    def test_from_env_nonpositive_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv(ENV_CALLBACK_TIMEOUT, "0")
        loop = WeaknessAnalyserCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._callback_timeout_s == DEFAULT_CALLBACK_TIMEOUT_S  # noqa: SLF001
