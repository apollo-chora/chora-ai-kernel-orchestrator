"""Agent #19 Familiar — pure business logic.

Per the 24-agent registry (comic #19): "In-game RPG companion's voice —
adjusts tone, remembers preferences within consent scope." LLM tier 2,
risk tier MEDIUM, capabilities = (FAMILIAR, EXPLANATION).

CRITICAL DISTINCTION (memory `feedback_familiar_vs_agent`):

    Familiar ENTITY = an in-game RPG companion aggregate inside Content
    Consumption (services/chora-consumption/internal/domain/familiar/).
    Pure game mechanics — XP curve, level traits, summoning ceremony.

    Familiar AGENT (this package) = the LLM-aware adapter concern that
    composes nudge text, daily-dose curation lines, and RPG dialogue.
    Lives in AI Kernel and is invoked via the orchestrator's
    /orchestrate endpoint by Content Consumption (over HTTP).

Three call kinds:

    PROACTIVE_NUDGE       — comic Ch6 P14 P1 invariant: text contains a
                            topic-aware suggestion (e.g. "I noticed
                            you're authoring on {topic}").
    DAILY_DOSE_CURATION   — comic Ch6 P14 P4: a 5-atom dose nudge line
                            referencing topics in the dose.
    RPG_DIALOGUE          — companion-voice reply to a learner message,
                            personalised by familiar name + topic.

This module is deliberately deterministic + LLM-free. The Model Broker
+ guardrail screen still run when the agent is dispatched in the
LangGraph pipeline; the agent's job is to assemble the prompt-shaped
text from typed inputs.
"""

from chora_ai_kernel_orchestrator.domain.agents.familiar.agent import (
    FamiliarKind,
    FamiliarRequest,
    FamiliarResponse,
    compose_daily_dose_curation,
    compose_proactive_nudge,
    compose_rpg_dialogue,
    dispatch,
)

__all__ = [
    "FamiliarKind",
    "FamiliarRequest",
    "FamiliarResponse",
    "compose_daily_dose_curation",
    "compose_proactive_nudge",
    "compose_rpg_dialogue",
    "dispatch",
]
