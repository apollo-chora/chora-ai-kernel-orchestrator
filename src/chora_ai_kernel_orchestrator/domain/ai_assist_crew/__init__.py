"""AI Assist 6-agent content gate domain module.

Per Phyllis MVP §5.7 + canonical Tier 5 D20 fixed-core crew per Content
{verb} domain, the Creation crew is a 6-agent gate:

    Validator -> Classifier -> Web Researcher -> Q&A Generator
              -> Evaluator -> Reporter

All artefacts in this package are pure domain (no infra imports). The
adapter layer wires gRPC + checkpointers + HTTP around it.
"""

from __future__ import annotations

from chora_ai_kernel_orchestrator.domain.ai_assist_crew.registry import (
    AI_ASSIST_AGENT_REGISTRY,
    AIAssistAgentDescriptor,
    AIAssistAgentRole,
    AutonomyLevel,
)
from chora_ai_kernel_orchestrator.domain.ai_assist_crew.state import (
    AIAssistState,
    Classification,
    Evaluation,
    QAPair,
    Report,
    ValidatorResult,
    WebResearch,
    new_ai_assist_state,
)

__all__ = [
    "AIAssistAgentDescriptor",
    "AIAssistAgentRole",
    "AIAssistState",
    "AI_ASSIST_AGENT_REGISTRY",
    "AutonomyLevel",
    "Classification",
    "Evaluation",
    "QAPair",
    "Report",
    "ValidatorResult",
    "WebResearch",
    "new_ai_assist_state",
]
