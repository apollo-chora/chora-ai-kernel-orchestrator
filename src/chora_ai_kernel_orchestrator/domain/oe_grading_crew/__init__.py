"""OE grading crew domain (ADR-172).

Pure-domain state + value objects for the per-submission OE grading crew:
an evaluator→moderator reject/re-grade loop per OE question, followed by an
assess_summary pass that writes the whole-assessment overall comment.

No infra imports (no httpx / grpc / langgraph) — side effects live in the
adapter layer, mirroring domain/qgen_crew.

Public surface:

    OEGradingState         — LangGraph TypedDict
    OEQuestionInput        — one OE question to grade
    MCQResult              — one MCQ outcome (assess_summary context)
    EvaluationResult       — oe_evaluator output (evaluate mode)
    ModerationResult       — oe_moderator verdict (accept/reject + feedback)
    GuardrailResult        — Cloud Model Armor pre/post-screen verdict
    GradedOEQuestion       — terminal per-OE-question result
    DEFAULT_MAX_ITERATIONS — re-grade loop budget
    current_question       — cursor helper
"""

from chora_ai_kernel_orchestrator.domain.oe_grading_crew.state import (
    DEFAULT_MAX_ITERATIONS,
    EvaluationResult,
    GradedOEQuestion,
    GuardrailResult,
    MCQResult,
    ModerationResult,
    OEGradingState,
    OEQuestionInput,
    current_question,
)

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "EvaluationResult",
    "GradedOEQuestion",
    "GuardrailResult",
    "MCQResult",
    "ModerationResult",
    "OEGradingState",
    "OEQuestionInput",
    "current_question",
]
