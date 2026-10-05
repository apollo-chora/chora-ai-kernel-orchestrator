"""Tests for QGenCrewPubsubLoop construction + from_env factory.

The actual ``start`` / ``stop`` lifecycle is exercised in integration
(production wiring lands in main.py); these tests verify the construction
contract + the env-var factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_loop import (
    DEFAULT_CALLBACK_TIMEOUT_S,
    DEFAULT_SUBSCRIPTION,
    ENV_CALLBACK_TIMEOUT,
    QGenCrewPubsubLoop,
)


@dataclass
class _FakeSubscriber:
    async def handle_message(self, msg: Any) -> None:  # pragma: no cover
        return None


class TestConstruction:
    def test_requires_project(self) -> None:
        with pytest.raises(ValueError, match="project"):
            QGenCrewPubsubLoop(
                subscriber=_FakeSubscriber(),
                project="",
                subscription="x",
            )

    def test_requires_subscription(self) -> None:
        with pytest.raises(ValueError, match="subscription"):
            QGenCrewPubsubLoop(
                subscriber=_FakeSubscriber(),
                project="chora-489812",
                subscription="",
            )


class TestFromEnv:
    def test_none_when_nats_url_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NATS_URL", raising=False)
        monkeypatch.delenv("QGEN_CREW_SUBSCRIPTION", raising=False)
        assert QGenCrewPubsubLoop.from_env(subscriber=_FakeSubscriber()) is None

    def test_defaults_to_canonical_subscription(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.delenv("QGEN_CREW_SUBSCRIPTION", raising=False)
        loop = QGenCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._subscription == DEFAULT_SUBSCRIPTION  # noqa: SLF001
        assert loop._project == "chora-ai-kernel"  # noqa: SLF001
        assert loop._url == "nats://nats:4222"  # noqa: SLF001

    def test_subscription_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv("QGEN_CREW_SUBSCRIPTION", "chora-ai-kernel.custom-qgen-sub")
        loop = QGenCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._subscription == (  # noqa: SLF001
            "chora-ai-kernel.custom-qgen-sub"
        )


class TestCallbackTimeout:
    """The per-message callback wait must accommodate the multi-attempt
    high-tier actor↔critic loop (the prior hardcoded 120s timed out at
    ~3 attempts of gemini-3.1-pro-preview). Pub/Sub auto-extends the
    message lease while the callback runs, so this is the real bound."""

    def test_default_is_generous(self) -> None:
        loop = QGenCrewPubsubLoop(
            subscriber=_FakeSubscriber(),
            project="chora-489812",
            subscription="x",
        )
        assert loop._callback_timeout_s == DEFAULT_CALLBACK_TIMEOUT_S  # noqa: SLF001
        # Must clear the old 120s that was too tight for the loop.
        assert DEFAULT_CALLBACK_TIMEOUT_S >= 240

    def test_explicit_param(self) -> None:
        loop = QGenCrewPubsubLoop(
            subscriber=_FakeSubscriber(),
            project="chora-489812",
            subscription="x",
            callback_timeout_s=180.0,
        )
        assert loop._callback_timeout_s == 180.0  # noqa: SLF001

    def test_from_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv(ENV_CALLBACK_TIMEOUT, "420")
        loop = QGenCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._callback_timeout_s == 420.0  # noqa: SLF001

    def test_from_env_invalid_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv(ENV_CALLBACK_TIMEOUT, "not-a-number")
        loop = QGenCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._callback_timeout_s == DEFAULT_CALLBACK_TIMEOUT_S  # noqa: SLF001

    def test_from_env_nonpositive_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_URL", "nats://nats:4222")
        monkeypatch.setenv(ENV_CALLBACK_TIMEOUT, "0")
        loop = QGenCrewPubsubLoop.from_env(subscriber=_FakeSubscriber())
        assert loop is not None
        assert loop._callback_timeout_s == DEFAULT_CALLBACK_TIMEOUT_S  # noqa: SLF001
