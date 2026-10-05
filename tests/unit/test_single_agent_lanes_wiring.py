"""RED: composition root for the generic single-agent lanes (the two fog folds
and the caller-facing companion_turn lane). Each enabled lane: its own DB connection, the runtime's D3a saver for
its crew, ONE executor pinned to its role, the outbox writer, graph, runner,
inbox, subscriber, a pull loop on its request subscription, and the lane
registered on the kennel runtime (crew + role) so the single role-routed
completion loop resumes it. No model-gateway client anywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

import chora_ai_kernel_orchestrator.adapter.pubsub.single_agent_lanes_wiring as wiring_mod
import chora_ai_kernel_orchestrator.adapter.secrets as secrets_mod
from chora_ai_kernel_orchestrator.adapter.pubsub.single_agent_lanes_wiring import (
    ROLES_WITH_SUBSCRIBER_AGENTS,
    SINGLE_AGENT_LANES,
    SingleAgentLaneComponents,
    _assemble_single_agent_lane,
    build_single_agent_lanes_from_env,
    single_agent_lane_specs_from_env,
)
from chora_ai_kernel_orchestrator.orchestrators.consumption_lanes import (
    companion_turn_contract,
    dose_recommendation_contract,
)
from chora_ai_kernel_orchestrator.orchestrators.fold_lanes import (
    companion_reflection_contract,
    kg_exploration_contract,
)
from chora_ai_kernel_orchestrator.orchestrators.single_agent_workflow import (
    SingleAgentRequestSubscriber,
    SingleAgentWorkflowRunner,
)


class _FakeConn:
    def cursor(self) -> Any:  # pragma: no cover - never exercised
        raise AssertionError("cursor must not be exercised in a unit test")


class _FakeRuntime:
    def __init__(self) -> None:
        self.lanes: list[dict[str, Any]] = []
        self.ledger = object()
        self.pubsub_project = "chora-489812"

    def register_lane(self, name: str, *, crew: str, roles: Any, runner: Any) -> None:
        self.lanes.append({"name": name, "crew": crew, "roles": tuple(roles), "runner": runner})


def test_the_fold_lane_table_names_both_folds_with_their_default_subscriptions() -> None:
    by_name = {spec.contract.name: spec for spec in SINGLE_AGENT_LANES}
    assert set(by_name) == {"kg_exploration", "companion_reflection", "companion_turn", "dose_recommendation"}
    kg, refl = by_name["kg_exploration"], by_name["companion_reflection"]
    assert kg.enabled_env == "KG_EXPLORATION_ENABLED"
    assert kg.subscription_env == "KG_EXPLORATION_SUBSCRIPTION"
    assert kg.default_subscription == "chora-ai-kernel-orchestrator.concept-suggestion-requested"
    assert refl.enabled_env == "COMPANION_REFLECTION_ENABLED"
    assert refl.subscription_env == "COMPANION_REFLECTION_SUBSCRIPTION"
    assert refl.default_subscription == "chora-ai-kernel-orchestrator.goal-knowledge-synthesis-requested"
    assert kg.contract.role == "kg_explore" and refl.contract.role == "companion_chat"
    dose = by_name["dose_recommendation"]
    assert dose.enabled_env == "DOSE_RECOMMENDATION_ENABLED"
    assert dose.subscription_env == "DOSE_RECOMMENDATION_SUBSCRIPTION"
    assert dose.default_subscription == ("chora-ai-kernel-orchestrator.consumption-dose-recommendation-requested")
    assert dose.contract.role == "recommend"
    assert dose.cap_env == "DOSE_RECOMMENDATION_TENANT_INFLIGHT_CAP"


def test_specs_from_env_select_only_enabled_lanes_and_honour_the_subscription_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KG_EXPLORATION_ENABLED", "true")
    monkeypatch.delenv("COMPANION_REFLECTION_ENABLED", raising=False)
    monkeypatch.setenv("KG_EXPLORATION_SUBSCRIPTION", "custom.sub")
    specs = single_agent_lane_specs_from_env()
    assert [s.contract.name for s in specs] == ["kg_exploration"]
    assert specs[0].subscription == "custom.sub"
    monkeypatch.setenv("COMPANION_REFLECTION_ENABLED", "true")
    monkeypatch.delenv("KG_EXPLORATION_SUBSCRIPTION", raising=False)
    specs = single_agent_lane_specs_from_env()
    assert [s.contract.name for s in specs] == ["kg_exploration", "companion_reflection"]
    assert specs[0].subscription == "chora-ai-kernel-orchestrator.concept-suggestion-requested"


async def test_build_returns_nothing_when_no_lane_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KG_EXPLORATION_ENABLED", raising=False)
    monkeypatch.delenv("COMPANION_REFLECTION_ENABLED", raising=False)
    assert await build_single_agent_lanes_from_env(runtime=_FakeRuntime()) == []


async def test_build_requires_the_runtime_and_the_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KG_EXPLORATION_ENABLED", "true")
    monkeypatch.setenv("CHORA_PUBSUB_PROJECT", "chora-489812")
    with pytest.raises(RuntimeError, match="runtime"):
        await build_single_agent_lanes_from_env(runtime=None)
    monkeypatch.delenv("CHORA_PUBSUB_PROJECT", raising=False)
    assert await build_single_agent_lanes_from_env(runtime=_FakeRuntime()) is None


async def test_build_returns_none_when_the_dsn_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPANION_REFLECTION_ENABLED", "true")
    monkeypatch.setenv("CHORA_PUBSUB_PROJECT", "chora-489812")
    monkeypatch.setattr(secrets_mod, "resolve_dsn", lambda: "")
    assert await build_single_agent_lanes_from_env(runtime=_FakeRuntime()) is None


def test_assemble_builds_one_lane_pinned_to_its_role_and_registers_it() -> None:
    runtime = _FakeRuntime()
    captured: dict[str, Any] = {}
    real_builder = wiring_mod.build_single_agent_graph

    def _spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_builder(**kwargs)

    wiring_mod.build_single_agent_graph = _spy  # type: ignore[assignment]
    try:
        components = _assemble_single_agent_lane(
            contract=kg_exploration_contract(),
            subscription="chora-ai-kernel-orchestrator.concept-suggestion-requested",
            pubsub_project="chora-489812",
            db_conn=_FakeConn(),
            checkpointer=None,
            runtime=runtime,
            callback_timeout_s=60.0,
            nats_url="nats://nats:4222",
        )
    finally:
        wiring_mod.build_single_agent_graph = real_builder  # type: ignore[assignment]

    assert isinstance(components, SingleAgentLaneComponents)
    assert components.name == "kg_exploration"
    assert captured["contract"].role == "kg_explore"
    assert captured["executor"]._allowed_roles == {"kg_explore"}
    assert captured["source_project"] == "chora-489812"
    assert isinstance(components.runner, SingleAgentWorkflowRunner)
    assert components.runner._graph is components.graph
    assert isinstance(components.subscriber, SingleAgentRequestSubscriber)
    assert components.subscriber._runner is components.runner
    assert components.pubsub_loop.subscriptions == ["chora-ai-kernel-orchestrator.concept-suggestion-requested"]
    assert components.pubsub_loop.project == "chora-489812"
    assert runtime.lanes == [
        {"name": "kg_exploration", "crew": "kg_exploration", "roles": ("kg_explore",), "runner": components.runner}
    ]


def test_assemble_registers_the_reflection_lane_on_the_shared_chat_role() -> None:
    runtime = _FakeRuntime()
    components = _assemble_single_agent_lane(
        contract=companion_reflection_contract(),
        subscription="chora-ai-kernel-orchestrator.goal-knowledge-synthesis-requested",
        pubsub_project="chora-489812",
        db_conn=_FakeConn(),
        checkpointer=None,
        runtime=runtime,
        callback_timeout_s=60.0,
        nats_url="nats://nats:4222",
    )
    assert runtime.lanes[0]["crew"] == "companion_reflection" and runtime.lanes[0]["roles"] == ("companion_chat",)
    assert components.pubsub_loop.subscriptions == ["chora-ai-kernel-orchestrator.goal-knowledge-synthesis-requested"]


def test_assemble_requires_the_runtime() -> None:
    with pytest.raises(RuntimeError, match="runtime"):
        _assemble_single_agent_lane(
            contract=kg_exploration_contract(),
            subscription="s",
            pubsub_project="p",
            db_conn=_FakeConn(),
            checkpointer=None,
            runtime=None,
            callback_timeout_s=60.0,
            nats_url="nats://nats:4222",
        )


def test_no_gateway_client_in_the_wiring() -> None:
    import inspect

    src = inspect.getsource(wiring_mod)
    for name in ("ModelGatewayMultimodalClient", "ModelGatewayTextClient", "ModelGatewayImageClient"):
        assert name not in src


# --------------------------------------------------------------------------- #
# companion_turn row (ADR-254 D1/D4/D8)
# --------------------------------------------------------------------------- #


def test_the_table_carries_the_companion_turn_lane_on_its_provisioned_subscription() -> None:
    spec = {s.contract.name: s for s in SINGLE_AGENT_LANES}["companion_turn"]
    assert spec.enabled_env == "COMPANION_TURN_ENABLED"
    assert spec.subscription_env == "COMPANION_TURN_SUBSCRIPTION"
    # the subscription the coordinator provisioned (120s ack, no push endpoint)
    assert spec.default_subscription == ("chora-ai-kernel-orchestrator.consumption-companion-turn-requested")
    assert spec.contract.role == "companion_chat"
    assert spec.cap_env == "COMPANION_TURN_TENANT_INFLIGHT_CAP"


def test_companion_turn_takes_no_cap_unless_one_is_configured(monkeypatch) -> None:
    monkeypatch.delenv("COMPANION_TURN_TENANT_INFLIGHT_CAP", raising=False)
    monkeypatch.setenv("COMPANION_TURN_ENABLED", "true")
    monkeypatch.delenv("KG_EXPLORATION_ENABLED", raising=False)
    monkeypatch.delenv("COMPANION_REFLECTION_ENABLED", raising=False)
    specs = single_agent_lane_specs_from_env()
    assert [s.contract.name for s in specs] == ["companion_turn"]
    assert specs[0].contract.tenant_inflight_cap is None


def test_a_configured_cap_reaches_the_contract_with_its_rejected_result(monkeypatch) -> None:
    monkeypatch.setenv("COMPANION_TURN_ENABLED", "true")
    monkeypatch.setenv("COMPANION_TURN_TENANT_INFLIGHT_CAP", "4")
    monkeypatch.delenv("KG_EXPLORATION_ENABLED", raising=False)
    monkeypatch.delenv("COMPANION_REFLECTION_ENABLED", raising=False)
    contract = single_agent_lane_specs_from_env()[0].contract
    assert contract.tenant_inflight_cap == 4
    # a cap without a rejected() cannot be constructed, so a capped request is
    # never dropped in silence
    assert contract.rejected is not None


@pytest.mark.parametrize("bad", ["nope", "0", "-3"])
def test_an_unreadable_cap_raises_rather_than_meaning_no_cap(monkeypatch, bad: str) -> None:
    monkeypatch.setenv("COMPANION_TURN_ENABLED", "true")
    monkeypatch.setenv("COMPANION_TURN_TENANT_INFLIGHT_CAP", bad)
    monkeypatch.delenv("KG_EXPLORATION_ENABLED", raising=False)
    monkeypatch.delenv("COMPANION_REFLECTION_ENABLED", raising=False)
    with pytest.raises(ValueError, match="COMPANION_TURN_TENANT_INFLIGHT_CAP"):
        single_agent_lane_specs_from_env()


def test_the_table_order_is_stable_so_lane_startup_is_deterministic() -> None:
    assert [s.contract.name for s in SINGLE_AGENT_LANES] == [
        "kg_exploration",
        "companion_reflection",
        "companion_turn",
        "dose_recommendation",
    ]


# ── the request-side callback bound is declared per lane, not inherited ───────
# Regression guard for Item C. AgentCompletionPubsubLoop.DEFAULT_CALLBACK_TIMEOUT_S
# is 60.0 and its comment justifies 60s for a COMPLETION resume ("no model call
# in it"). The three generic lanes reuse that loop on their REQUEST side, so
# before this guard they inherited that number silently, with no env escape
# hatch (the wiring does not go through AgentCompletionPubsubLoop.from_env, so
# AGENT_COMPLETION_CALLBACK_TIMEOUT_SECONDS never reached them). The value is
# correct for these lanes: the request callback is decode -> inbox -> graph to
# park -> ack, and PubSubAgentExecutor publishes rather than calling a model.
# What was wrong was that it was invisible. LaneSpec.callback_timeout_s has NO
# default on purpose: a new lane cannot be added without stating its number.


def test_every_lane_declares_its_request_callback_bound_explicitly() -> None:
    """Behaviour-identical to the inherited default: every lane is still 60.0."""
    assert [s.callback_timeout_s for s in SINGLE_AGENT_LANES] == [60.0, 60.0, 60.0, 60.0]


def test_resolving_a_lane_preserves_its_declared_callback_bound() -> None:
    for spec in SINGLE_AGENT_LANES:
        assert spec.resolved().callback_timeout_s == spec.callback_timeout_s


def test_the_declared_bound_reaches_the_loop_rather_than_the_loop_default() -> None:
    """Threads a NON-default value on purpose. Asserting 60.0 here would pass
    even if the wiring dropped the argument, because 60.0 IS the loop default,
    so only a distinct number proves the value is actually carried."""
    runtime = _FakeRuntime()
    components = _assemble_single_agent_lane(
        contract=kg_exploration_contract(),
        subscription="chora-ai-kernel-orchestrator.concept-suggestion-requested",
        pubsub_project="chora-489812",
        db_conn=_FakeConn(),
        checkpointer=None,
        runtime=runtime,
        callback_timeout_s=137.0,
        nats_url="nats://nats:4222",
    )
    assert components.pubsub_loop._callback_timeout_s == 137.0


def test_the_live_lanes_assemble_on_sixty_seconds() -> None:
    """The unchanged-value proof the legibility change is required to carry."""
    runtime = _FakeRuntime()
    components = _assemble_single_agent_lane(
        contract=kg_exploration_contract(),
        subscription="chora-ai-kernel-orchestrator.concept-suggestion-requested",
        pubsub_project="chora-489812",
        db_conn=_FakeConn(),
        checkpointer=None,
        runtime=runtime,
        callback_timeout_s=60.0,
        nats_url="nats://nats:4222",
    )
    assert components.pubsub_loop._callback_timeout_s == 60.0


# --------------------------------------------------------------------------- #
# the answerer guard: a lane whose ROLE has no subscriber-capable agent must
# refuse to start rather than park runs nothing can ever resume.
# --------------------------------------------------------------------------- #


def test_the_guard_fires_for_a_role_whose_agent_half_does_not_exist() -> None:
    """RED-FIRST. `recommend` has no subscriber: chora-recommender runs the ADK
    web launcher and carries no AGENT_DISPATCH_* env, so nothing consumes the
    recommend request topic and no completion can ever be published. Enabling
    the lane would bind, decode and PARK every request forever.

    The guard must REFUSE, so DOSE_RECOMMENDATION_ENABLED alone cannot arm it."""
    with pytest.raises(RuntimeError, match="no subscriber-capable agent"):
        _assemble_single_agent_lane(
            contract=dose_recommendation_contract(),
            subscription="chora-ai-kernel-orchestrator.consumption-dose-recommendation-requested",
            pubsub_project="chora-489812",
            db_conn=_FakeConn(),
            checkpointer=None,
            runtime=_FakeRuntime(),
            callback_timeout_s=60.0,
            nats_url="nats://nats:4222",
        )


def test_the_guard_does_not_fire_for_a_role_that_has_one() -> None:
    """The discrimination control. A guard that refused every lane would pass
    the test above while breaking the three live lanes, so prove it lets a role
    WITH a subscriber through. companion_chat is subscriber-only and live."""
    components = _assemble_single_agent_lane(
        contract=companion_turn_contract(),
        subscription="chora-ai-kernel-orchestrator.consumption-companion-turn-requested",
        pubsub_project="chora-489812",
        db_conn=_FakeConn(),
        checkpointer=None,
        runtime=_FakeRuntime(),
        callback_timeout_s=60.0,
        nats_url="nats://nats:4222",
    )
    assert components.name == "companion_turn"


def test_every_lane_in_the_live_table_passes_its_own_guard() -> None:
    """The table and the guard must not be able to disagree: a lane shipped in
    SINGLE_AGENT_LANES whose role has no answerer is the defect this guard
    exists for, and it should fail HERE rather than at a boot in prod."""
    unanswerable = [s.contract.name for s in SINGLE_AGENT_LANES if s.contract.role not in ROLES_WITH_SUBSCRIBER_AGENTS]
    assert unanswerable == ["dose_recommendation"], (
        "expected exactly the known-dormant dose lane here; anything else means a "
        "lane shipped that would park runs nothing can resume: " + repr(unanswerable)
    )
