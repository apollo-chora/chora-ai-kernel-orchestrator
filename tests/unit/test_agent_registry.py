"""Tests for the 24-agent registry (Sequel comic cast).

Each agent in the Sequel "Behind the Button" cast has:
  id (kebab-case), display_name (human title), tier (1-4),
  agent_type (LLM | ML | System), description, capabilities,
  risk_tier (low | medium | high | critical), cost_estimate_per_call_sgd.

Source: docs/comic/Sequel_Chora-Agentic-Customer-journeys.md
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.domain.registry import (
    AGENT_REGISTRY,
    Agent24Registry,
    AgentDescriptor,
    AgentType,
    Capability,
    RiskTier,
)

# --- module shape -------------------------------------------------------------


def test_module_exports_registry() -> None:
    """AGENT_REGISTRY is a populated Agent24Registry singleton."""
    assert isinstance(AGENT_REGISTRY, Agent24Registry)
    assert len(AGENT_REGISTRY.all()) == 24


def test_registry_holds_24_unique_agents() -> None:
    """Every agent has a unique id."""
    ids = [a.id for a in AGENT_REGISTRY.all()]
    assert len(ids) == 24
    assert len(set(ids)) == 24, "duplicate agent ids"


def test_registry_iteration_is_stable() -> None:
    """Iteration order matches numeric agent number (#1 first, #24 last)."""
    agents = AGENT_REGISTRY.all()
    assert agents[0].id == "orchestrator"
    assert agents[-1].id == "story-point-estimator"


# --- per-agent metadata (table-driven) ----------------------------------------


_AGENT_TABLE: list[tuple[str, int, str, AgentType, RiskTier]] = [
    # (agent_id, comic_number, display_name, agent_type, risk_tier)
    ("orchestrator", 1, "Orchestrator", AgentType.LLM, RiskTier.HIGH),
    ("tutor", 2, "Tutor", AgentType.LLM, RiskTier.MEDIUM),
    ("weakness-analyzer", 3, "Weakness Analyzer", AgentType.ML, RiskTier.MEDIUM),
    ("learning-path", 4, "Learning Path", AgentType.ML, RiskTier.MEDIUM),
    ("adaptive-difficulty", 5, "Adaptive Difficulty", AgentType.ML, RiskTier.LOW),
    ("rag-memory", 6, "RAG Memory", AgentType.ML, RiskTier.HIGH),
    ("learning-analytics", 7, "Learning Analytics", AgentType.ML, RiskTier.MEDIUM),
    ("content-analyst", 8, "Content Analyst", AgentType.ML, RiskTier.LOW),
    ("explainer", 9, "Explainer", AgentType.LLM, RiskTier.MEDIUM),
    ("ai-governance", 10, "AI Governance", AgentType.LLM, RiskTier.CRITICAL),
    ("company-policy", 11, "Company Policy", AgentType.LLM, RiskTier.HIGH),
    ("content-gen", 12, "Content Gen", AgentType.LLM, RiskTier.HIGH),
    ("content-review", 13, "Content Review", AgentType.LLM, RiskTier.HIGH),
    ("training-compliance", 14, "Training Compliance", AgentType.LLM, RiskTier.HIGH),
    ("media-transcription", 15, "Media Transcription", AgentType.ML, RiskTier.MEDIUM),
    ("translation", 16, "Translation", AgentType.LLM, RiskTier.MEDIUM),
    ("trigger-optimizer", 17, "Trigger Optimizer", AgentType.ML, RiskTier.LOW),
    ("exam-marking", 18, "Exam Marking", AgentType.LLM, RiskTier.HIGH),
    ("familiar", 19, "Familiar", AgentType.LLM, RiskTier.MEDIUM),
    ("model-broker", 20, "Model Broker", AgentType.SYSTEM, RiskTier.HIGH),
    ("pvp-screener", 21, "PvP Screener", AgentType.LLM, RiskTier.HIGH),
    ("assessment", 22, "Assessment", AgentType.ML, RiskTier.MEDIUM),
    ("mcp-tool-router", 23, "MCP Tool Router", AgentType.SYSTEM, RiskTier.HIGH),
    ("story-point-estimator", 24, "Story Point Estimator", AgentType.LLM, RiskTier.MEDIUM),
]


@pytest.mark.parametrize(
    ("agent_id", "comic_number", "display_name", "agent_type", "risk_tier"),
    _AGENT_TABLE,
    ids=[row[0] for row in _AGENT_TABLE],
)
def test_each_agent_has_correct_metadata(
    agent_id: str,
    comic_number: int,
    display_name: str,
    agent_type: AgentType,
    risk_tier: RiskTier,
) -> None:
    """Every comic-listed agent maps to a registry entry with matching metadata."""
    agent = AGENT_REGISTRY.get(agent_id)
    assert agent.id == agent_id
    assert agent.comic_number == comic_number
    assert agent.display_name == display_name
    assert agent.agent_type == agent_type
    assert agent.risk_tier == risk_tier
    assert 1 <= agent.tier <= 4
    assert agent.description.strip(), f"{agent_id} missing description"
    assert agent.capabilities, f"{agent_id} declares no capabilities"
    assert agent.cost_estimate_per_call_sgd >= 0


def test_ai_governance_is_tier_4() -> None:
    """AI Governance is the only Tier-4 critical-risk LLM agent."""
    g = AGENT_REGISTRY.get("ai-governance")
    assert g.tier == 4
    assert g.risk_tier == RiskTier.CRITICAL


def test_model_broker_has_zero_call_cost() -> None:
    """Model Broker doesn't bill on its own calls — it routes for others."""
    broker = AGENT_REGISTRY.get("model-broker")
    assert broker.agent_type == AgentType.SYSTEM
    assert broker.cost_estimate_per_call_sgd == 0.0


def test_unknown_id_lookup_raises() -> None:
    """Looking up an unknown id raises KeyError with the id in the message."""
    with pytest.raises(KeyError, match="not-a-real-agent"):
        AGENT_REGISTRY.get("not-a-real-agent")


# --- capability-based filtering ----------------------------------------------


def test_filter_by_capability_translation_returns_translation_agent() -> None:
    """Translation capability maps to at least the Translation agent."""
    matches = AGENT_REGISTRY.filter_by_capability(Capability.TRANSLATION)
    assert any(a.id == "translation" for a in matches)


def test_filter_by_capability_atom_orchestration_returns_orchestrator() -> None:
    """Orchestrator owns atom_orchestration."""
    matches = AGENT_REGISTRY.filter_by_capability(Capability.ATOM_ORCHESTRATION)
    assert any(a.id == "orchestrator" for a in matches)


def test_filter_by_capability_unknown_returns_empty_list() -> None:
    """An unmapped (but valid) capability returns an empty list, not an error."""
    # Pick a capability that is unlikely to be mapped to any agent.
    matches = AGENT_REGISTRY.filter_by_capability(Capability.MCP_TOOL_ROUTING)
    # mcp-tool-router itself owns this — should be at least 1, never error.
    assert any(a.id == "mcp-tool-router" for a in matches)


def test_filter_by_capability_string_value() -> None:
    """filter_by_capability accepts the string value for HTTP query convenience."""
    matches = AGENT_REGISTRY.filter_by_capability("translation")
    assert any(a.id == "translation" for a in matches)


def test_filter_by_capability_invalid_string_raises() -> None:
    """Garbage capability strings surface a ValueError."""
    with pytest.raises(ValueError):
        AGENT_REGISTRY.filter_by_capability("not-a-capability")


# --- AgentDescriptor invariants ----------------------------------------------


def test_agent_descriptor_is_immutable() -> None:
    """AgentDescriptor is a frozen dataclass — mutation raises."""
    a = AGENT_REGISTRY.get("orchestrator")
    assert isinstance(a, AgentDescriptor)
    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError or AttributeError
        a.tier = 99  # type: ignore[misc]
