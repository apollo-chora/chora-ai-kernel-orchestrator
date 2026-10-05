"""6-agent content gate registry.

Per audit `docs/m13/audit-aikernel-fillgaps.md` §2.3 + §8 Wave 2 the
existing 24-agent registry only covers 4 of the 6 Phyllis MVP §5.7
gate stages. This module registers the canonical six WITHOUT renaming
the legacy 24 entries — they coexist (the legacy registry stays the
public surface for non-AI-Assist crews).

Per ADR-141 the autonomy ladder is HOOTL / HOTL / HITL Level 0-2;
Level 3 is **prohibited**. Each agent declares an `imda_dimension`
(canonical labels: accountability / transparency / safety_and_robustness
/ fairness_and_human_oversight) and an `hitl_imda_dimension` for the
dimension applied when the HITL escalation actually fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AIAssistAgentRole(StrEnum):
    """Canonical role for each gate stage."""

    VALIDATOR = "validator"
    CLASSIFIER = "classifier"
    WEB_RESEARCHER = "web-researcher"
    QA_GENERATOR = "qa-generator"
    EVALUATOR = "evaluator"
    REPORTER = "reporter"


class AutonomyLevel(StrEnum):
    """ADR-141 autonomy ladder.

    HOOTL  = Human Out Of The Loop (full automation, audit trail only)
    HOTL   = Human On The Loop (interruptible monitoring)
    HITL_LEVEL_0 = Human In The Loop, single-pass approve
    HITL_LEVEL_1 = Human In The Loop, dual-control approve
    HITL_LEVEL_2 = Human In The Loop, multi-stakeholder review
    Level 3 is **prohibited** — agents that would require it must split
    into smaller capabilities.
    """

    HOOTL = "HOOTL"
    HOTL = "HOTL"
    HITL_LEVEL_0 = "HITL_LEVEL_0"
    HITL_LEVEL_1 = "HITL_LEVEL_1"
    HITL_LEVEL_2 = "HITL_LEVEL_2"


@dataclass(frozen=True)
class AIAssistAgentDescriptor:
    """Immutable metadata for one of the six gate agents."""

    role: AIAssistAgentRole
    agent_id: str
    display_name: str
    risk_tier: int  # 1 (cheapest) -> 4 (highest oversight)
    autonomy_level: AutonomyLevel
    guardrail_yaml_ref: str  # "chora-guardrail/configs/agents/{role}.yaml"
    imda_dimension: str  # canonical label per ADR-141
    hitl_imda_dimension: str = "fairness_and_human_oversight"
    owner_team: str = "Team 3"  # Platform


_AGENTS: tuple[AIAssistAgentDescriptor, ...] = (
    AIAssistAgentDescriptor(
        role=AIAssistAgentRole.VALIDATOR,
        agent_id="validator",
        display_name="AI Assist — Validator",
        risk_tier=2,
        autonomy_level=AutonomyLevel.HOOTL,
        guardrail_yaml_ref="chora-guardrail/configs/agents/validator.yaml",
        imda_dimension="safety_and_robustness",
    ),
    AIAssistAgentDescriptor(
        role=AIAssistAgentRole.CLASSIFIER,
        agent_id="classifier",
        display_name="AI Assist — Classifier",
        risk_tier=1,
        autonomy_level=AutonomyLevel.HOOTL,
        guardrail_yaml_ref="chora-guardrail/configs/agents/classifier.yaml",
        imda_dimension="transparency",
    ),
    AIAssistAgentDescriptor(
        role=AIAssistAgentRole.WEB_RESEARCHER,
        agent_id="web-researcher",
        display_name="AI Assist — Web Researcher",
        risk_tier=3,
        autonomy_level=AutonomyLevel.HOTL,
        guardrail_yaml_ref="chora-guardrail/configs/agents/web-researcher.yaml",
        imda_dimension="safety_and_robustness",
    ),
    AIAssistAgentDescriptor(
        role=AIAssistAgentRole.QA_GENERATOR,
        agent_id="qa-generator",
        display_name="AI Assist — Q&A Generator",
        risk_tier=3,
        autonomy_level=AutonomyLevel.HOTL,
        guardrail_yaml_ref="chora-guardrail/configs/agents/qa-generator.yaml",
        imda_dimension="accountability",
    ),
    AIAssistAgentDescriptor(
        role=AIAssistAgentRole.EVALUATOR,
        agent_id="evaluator",
        display_name="AI Assist — Evaluator",
        risk_tier=3,
        autonomy_level=AutonomyLevel.HOTL,
        guardrail_yaml_ref="chora-guardrail/configs/agents/evaluator.yaml",
        imda_dimension="safety_and_robustness",
    ),
    AIAssistAgentDescriptor(
        role=AIAssistAgentRole.REPORTER,
        agent_id="reporter",
        display_name="AI Assist — Reporter",
        risk_tier=3,
        autonomy_level=AutonomyLevel.HITL_LEVEL_0,
        guardrail_yaml_ref="chora-guardrail/configs/agents/reporter.yaml",
        imda_dimension="transparency",
    ),
)


@dataclass(frozen=True)
class AIAssistAgentRegistry:
    """Read-only registry over the six AI Assist gate agents."""

    _agents: tuple[AIAssistAgentDescriptor, ...] = field(default_factory=lambda: _AGENTS)

    def all(self) -> tuple[AIAssistAgentDescriptor, ...]:
        return self._agents

    def get(self, agent_id: str) -> AIAssistAgentDescriptor:
        for a in self._agents:
            if a.agent_id == agent_id:
                return a
        raise KeyError(f"unknown ai-assist agent_id: {agent_id!r}")

    def get_by_role(self, role: AIAssistAgentRole) -> AIAssistAgentDescriptor:
        for a in self._agents:
            if a.role == role:
                return a
        raise KeyError(f"unknown ai-assist role: {role!r}")


AI_ASSIST_AGENT_REGISTRY: AIAssistAgentRegistry = AIAssistAgentRegistry()
