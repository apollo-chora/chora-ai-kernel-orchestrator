"""OEGradingState TypedDict + per-stage value objects (ADR-172).

The OE grading crew is the agentic backing for chora-delivery's per-submission
OE grading. On chora.delivery.grading.submission_requested.v1 it runs, for EACH
OE question in the submission:

    guardrail_pre(answer) ─► oe_evaluator ─► guardrail_post ─► oe_moderator
                                  ▲                                  │
                                  └──────── retry (≤ max) ◄──────────┘ reject
                                                                     │ accept
                                                                     ▼
                                                        record per-question grade

Then a single assess_summary pass (oe_evaluator in "assess_summary" mode, no
moderator loop) writes the whole-assessment overall comment. Terminal node
publishes chora.delivery.grading.submission_completed.v1.

Mirrors domain/qgen_crew/state.py (evaluator↔moderator ≙ question↔critic). Pure
domain — no infra imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

# Default max re-grade iterations for the moderator-rejection branch. attempt is
# 1-based: attempt=1 is the first grade; max_iterations=2 → up to 3 total grades
# before the crew ships the last evaluator output flagged quality_flagged=True.
DEFAULT_MAX_ITERATIONS: int = 2


@dataclass(frozen=True)
class OEQuestionInput:
    """One OE question to grade, projected from submission_requested.v1."""

    test_set_question_id: str
    question_id: str
    prompt: str
    rubric_json: str  # mandatory weighted rubric (criteria + weights)
    model_answer: str
    learner_response: str
    points_possible: int
    subject: str = ""
    topic: str = ""


@dataclass(frozen=True)
class MCQResult:
    """One MCQ outcome carried for the assess_summary pass (read-only context)."""

    test_set_question_id: str
    correct: bool
    points_earned: float
    points_possible: int


@dataclass(frozen=True)
class EvaluationResult:
    """oe_evaluator output for one OE answer — drives the moderator gate.

    criterion_scores_json mirrors QuestionGradeBatchResult.criterion_scores_json
    (criterion_id / title / score / max_score / feedback). comment is ALWAYS
    present (ADR-172 §D4) — for a right answer or a wrong one.
    """

    points_earned: float
    points_possible: int
    criterion_scores_json: str
    comment: str
    grading_model_id: str = ""
    grading_response_id: str = ""


@dataclass(frozen=True)
class ModerationResult:
    """oe_moderator output — qualitative judge of the evaluator's grading.

    Like qgen_critic: accept/reject + feedback only; the moderator never writes
    a score (the evaluator re-grades with feedback on reject).
    """

    accepted: bool
    moderation_notes: str = ""
    suggested_revisions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GuardrailResult:
    """Cloud Model Armor pre-/post-screen verdict (see [[cloud-model-armor-guardrails]])."""

    allowed: bool
    armor_verdict: str = ""
    user_facing_message: str = ""


@dataclass(frozen=True)
class GradedOEQuestion:
    """A terminal per-OE-question result for submission_completed.v1.graded[]."""

    test_set_question_id: str
    question_id: str
    points_earned: float
    points_possible: int
    criterion_scores_json: str
    comment: str
    grading_model_id: str
    grading_response_id: str
    quality_flagged: bool
    attempt_count: int


class OEGradingState(TypedDict, total=False):
    """LangGraph state for the per-submission OE grading crew.

    ``total=False`` so partial-dict node returns merge via the default reducer.
    """

    # ---- Caller-supplied (from submission_requested.v1) -------------------
    grading_job_id: str
    submission_id: str
    assessment_id: str
    tenant_id: str
    gcid: str  # learner_gcid (for per-tenant cost attribution)
    passing_threshold_percent: int
    model_tier: str
    per_question_feedback_enabled: bool
    subject: str
    total_points_possible: int
    mcq_points_earned: float
    traceparent: str
    tracestate: str
    max_iterations: int

    oe_questions: list[OEQuestionInput]
    mcq_results: list[MCQResult]

    # ---- ADR-197 M-B.2 prompt-override resolution (per agent role) --------
    # The runner resolves the active prompt override for oe_evaluator +
    # oe_moderator at handle_requested (via the optional PromptResolver bound
    # to the shared psycopg conn) and stores the WINNING result here, but ONLY
    # when an override actually applied (segments non-empty). No resolver wired
    # OR an embedded default ⇒ NONE of these keys set ⇒ the grading nodes
    # thread nothing ⇒ byte-identical pre-registry behaviour.
    #   prompt_overrides_* : {segment_id -> override body} (role/task/examples)
    #   prompt_version_*   : the winning override plan's opaque id
    #   prompt_source_*    : "tenant_override" | "platform_override"
    prompt_overrides_evaluator: dict[str, str]
    prompt_version_evaluator: str
    prompt_source_evaluator: str
    prompt_overrides_moderator: dict[str, str]
    prompt_version_moderator: str
    prompt_source_moderator: str

    # ---- Per-question loop state -----------------------------------------
    # current_index points at the OE question being graded; attempt is 1-based.
    current_index: int
    attempt: int
    current_evaluation: EvaluationResult | None
    moderation_notes: str  # carried into the next re-grade attempt
    # Transient quality_gate→router decision ("retry" | "record"). Set by
    # quality_gate_node and read by route_after_quality_gate so the two cannot
    # disagree (the prior dual-threshold check under-ran the loop by one grade).
    loop_decision: str
    guardrail_pre_result: GuardrailResult | None
    guardrail_post_result: GuardrailResult | None
    moderation_result: ModerationResult | None

    # ---- Accumulated results ---------------------------------------------
    graded: list[GradedOEQuestion]
    overall_comment: str
    overall_comment_model_id: str
    overall_comment_response_id: str

    # ---- Trace (IMDA D2 transparency) ------------------------------------
    pipeline_trace: list[dict[str, Any]]

    # ---- Terminal --------------------------------------------------------
    outcome: str  # SUCCESS | PARTIAL | FAILED
    failure_message: str
    mana_debited: int

    # ---- Errors (orchestrator-internal) ----------------------------------
    errors: list[str]


def current_question(state: OEGradingState) -> OEQuestionInput | None:
    """Return the OE question at current_index, or None when the loop is done."""
    idx = state.get("current_index", 0)
    questions = state.get("oe_questions", [])
    if 0 <= idx < len(questions):
        return questions[idx]
    return None
