"""qgen 2-agent crew domain — pure types + value objects.

Public surface:

    QGenCrewState           — LangGraph TypedDict
    CandidatePayload        — one AiAssistCandidate (JSON-encoded payload)
    CritiqueResult          — qgen_critic verdict (qualitative; NOT scoring)
    GuardrailResult         — Cloud Model Armor pre/post-screen verdict
    DEFAULT_MAX_RETRIES     — quality-loop budget (3 = 4 total attempts)
    assert_valid_refusal_reason
"""

from chora_ai_kernel_orchestrator.domain.qgen_crew.state import (
    DEFAULT_MAX_RETRIES,
    CandidatePayload,
    CritiqueResult,
    GuardrailResult,
    QGenCrewState,
    assert_valid_refusal_reason,
)

__all__ = [
    "DEFAULT_MAX_RETRIES",
    "CandidatePayload",
    "CritiqueResult",
    "GuardrailResult",
    "QGenCrewState",
    "assert_valid_refusal_reason",
]
