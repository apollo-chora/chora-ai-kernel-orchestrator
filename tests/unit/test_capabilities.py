"""Tests for the agent capability enum.

Capabilities are tags that map to the things one or more of the 24 agents
in the Sequel comic can do (atom orchestration, content generation,
translation, etc.). The registry uses these for filterable lookup.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.domain.registry import Capability


def test_capability_is_string_enum() -> None:
    """Capability values are kebab-case strings (URL-safe filter values)."""
    assert isinstance(Capability.ATOM_ORCHESTRATION.value, str)
    assert Capability.ATOM_ORCHESTRATION.value == "atom-orchestration"


def test_capability_covers_24_agent_value_streams() -> None:
    """Capability set covers every value stream demonstrated by the 24 agents."""
    expected = {
        "atom-orchestration",
        "tutoring",
        "weakness-analysis",
        "learning-path",
        "adaptive-difficulty",
        "rag-memory",
        "learning-analytics",
        "content-analysis",
        "explanation",
        "ai-governance",
        "company-policy",
        "content-generation",
        "content-review",
        "training-compliance",
        "media-transcription",
        "translation",
        "trigger-optimization",
        "exam-marking",
        "familiar",
        "model-broker",
        "pvp-screening",
        "assessment",
        "mcp-tool-routing",
        "story-point-estimation",
    }
    actual = {c.value for c in Capability}
    assert expected.issubset(actual), expected - actual


def test_capability_lookup_from_string() -> None:
    """Capability(value) round-trips via the enum."""
    assert Capability("translation") is Capability.TRANSLATION


def test_capability_unknown_value_raises() -> None:
    """Unknown capability strings surface a ValueError."""
    with pytest.raises(ValueError):
        Capability("does-not-exist")
