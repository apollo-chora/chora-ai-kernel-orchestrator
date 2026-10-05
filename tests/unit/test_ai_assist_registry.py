"""Tests for the 6-agent content gate registry (S3.4 P0 RED).

Per audit `docs/m13/audit-aikernel-fillgaps.md` §2.3 + §8 the registry is
missing 3 agents (Validator / Web Researcher / Reporter) and the existing
4 (`content-gen`, `content-review`, `translation`, `assessment`) don't
match the Phyllis MVP §5.7 names. We register the canonical 6 alongside
the existing 24 — they are not renames.

ADR-141 mandates the canonical labels: HITL flag is the
`fairness_and_human_oversight` D4 dimension.
"""

from __future__ import annotations


def test_six_agent_gate_descriptors_exist() -> None:
    """The AI Assist crew registry exposes exactly six descriptors."""
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew.registry import (
        AI_ASSIST_AGENT_REGISTRY,
        AIAssistAgentRole,
    )

    expected_roles = (
        AIAssistAgentRole.VALIDATOR,
        AIAssistAgentRole.CLASSIFIER,
        AIAssistAgentRole.WEB_RESEARCHER,
        AIAssistAgentRole.QA_GENERATOR,
        AIAssistAgentRole.EVALUATOR,
        AIAssistAgentRole.REPORTER,
    )
    actual = tuple(d.role for d in AI_ASSIST_AGENT_REGISTRY.all())
    assert actual == expected_roles


def test_descriptors_carry_risk_tier_and_autonomy() -> None:
    """Each agent declares risk_tier (1-4) + autonomy_level (HOOTL/HOTL/HITL)."""
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew.registry import (
        AI_ASSIST_AGENT_REGISTRY,
        AutonomyLevel,
    )

    for d in AI_ASSIST_AGENT_REGISTRY.all():
        assert 1 <= d.risk_tier <= 4
        # ADR-141 prohibits Level 3 — registry must enforce.
        assert d.autonomy_level in {
            AutonomyLevel.HOOTL,
            AutonomyLevel.HOTL,
            AutonomyLevel.HITL_LEVEL_0,
            AutonomyLevel.HITL_LEVEL_1,
            AutonomyLevel.HITL_LEVEL_2,
        }
        # Each agent must declare a guardrail YAML path so Tier 4 composition
        # can be loaded by the Guardrail Service.
        assert d.guardrail_yaml_ref.endswith(".yaml")


def test_registry_lookup_by_role_and_agent_id() -> None:
    """Registry exposes both role-based and id-based lookup."""
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew.registry import (
        AI_ASSIST_AGENT_REGISTRY,
        AIAssistAgentRole,
    )

    validator = AI_ASSIST_AGENT_REGISTRY.get_by_role(AIAssistAgentRole.VALIDATOR)
    assert validator.agent_id == "validator"
    by_id = AI_ASSIST_AGENT_REGISTRY.get("validator")
    assert by_id is validator


def test_reporter_is_critical_risk_tier_with_imda_d2() -> None:
    """Reporter emits IMDA D2 transparency evidence — must be tier-3+."""
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew.registry import (
        AI_ASSIST_AGENT_REGISTRY,
        AIAssistAgentRole,
    )

    reporter = AI_ASSIST_AGENT_REGISTRY.get_by_role(AIAssistAgentRole.REPORTER)
    assert reporter.risk_tier >= 3
    assert reporter.imda_dimension == "transparency"


def test_reporter_records_human_oversight_dimension_when_hitl_engaged() -> None:
    """The Reporter descriptor declares the dimension applied when HITL fires."""
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew.registry import (
        AI_ASSIST_AGENT_REGISTRY,
        AIAssistAgentRole,
    )

    reporter = AI_ASSIST_AGENT_REGISTRY.get_by_role(AIAssistAgentRole.REPORTER)
    assert reporter.hitl_imda_dimension == "fairness_and_human_oversight"
