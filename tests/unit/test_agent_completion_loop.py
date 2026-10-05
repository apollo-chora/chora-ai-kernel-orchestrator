"""RED — the StreamingPull wiring for the agent completion lanes (ADR-253 D2/D5).

Pull on BOTH sides, so no gateway push route and no Istio allowlist entry are
needed. The testable surface is env resolution and subscription naming; start/stop
is an SDK roundtrip covered in integration.
"""

from __future__ import annotations

import pytest


def _cls():
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_completion_loop import (
        AgentCompletionPubsubLoop,
    )

    return AgentCompletionPubsubLoop


def test_the_default_subscription_is_derived_per_role(monkeypatch) -> None:
    """One completion topic per agent role (D4), so one subscription per role."""
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_completion_loop import (
        completion_subscription_name,
    )

    assert completion_subscription_name("oe_evaluate") == (
        "chora-ai-kernel-orchestrator.agent-dispatch-oe-evaluate-completed"
    )
    assert completion_subscription_name("oe_moderate") == (
        "chora-ai-kernel-orchestrator.agent-dispatch-oe-moderate-completed"
    )


def test_from_env_returns_none_without_nats_url(monkeypatch) -> None:
    monkeypatch.delenv("NATS_URL", raising=False)
    assert _cls().from_env(subscriber=object(), agent_roles=("oe_evaluate",)) is None


def test_from_env_builds_one_subscription_per_role(monkeypatch) -> None:
    monkeypatch.setenv("NATS_URL", "nats://nats:4222")
    monkeypatch.delenv("AGENT_COMPLETION_SUBSCRIPTIONS", raising=False)

    loop = _cls().from_env(subscriber=object(), agent_roles=("oe_evaluate", "oe_moderate"))

    assert loop is not None
    assert loop.subscriptions == [
        "chora-ai-kernel-orchestrator.agent-dispatch-oe-evaluate-completed",
        "chora-ai-kernel-orchestrator.agent-dispatch-oe-moderate-completed",
    ]


def test_an_explicit_env_override_wins(monkeypatch) -> None:
    """Per [[secrets-and-env]] the deployment must be able to repoint a lane
    without a code change."""
    monkeypatch.setenv("NATS_URL", "nats://nats:4222")
    monkeypatch.setenv("AGENT_COMPLETION_SUBSCRIPTIONS", "sub-a, sub-b ,")

    loop = _cls().from_env(subscriber=object(), agent_roles=("oe_evaluate",))

    assert loop.subscriptions == ["sub-a", "sub-b"]


def test_no_roles_and_no_override_is_refused(monkeypatch) -> None:
    """A completion loop with nothing to listen on would start clean and park
    every run forever."""
    monkeypatch.setenv("NATS_URL", "nats://nats:4222")
    monkeypatch.delenv("AGENT_COMPLETION_SUBSCRIPTIONS", raising=False)

    with pytest.raises(ValueError, match="subscription"):
        _cls().from_env(subscriber=object(), agent_roles=())
