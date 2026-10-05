"""OE grading crew StateGraph (ADR-172).

Per-submission OE grading. For EACH OE question in the submission the graph runs
an evaluator→moderator reject/re-grade loop; once every question is graded it
runs a single assess_summary pass and publishes the result.

    validate_input
        │
        ▼  (per OE question, current_index pointer)
    guardrail_pre ──(block)──► record_blocked ──┐
        │ (allow)                               │
        ▼                                       │
    evaluate ◄────────┐                         │
        │             │                         │
        ▼             │ retry                    │
    guardrail_post    │                          │
        │             │                          │
        ▼             │                          │
    moderate          │                          │
        │             │                          │
        ▼             │                          │
    quality_gate ─────┘                          │
        │ (record)                               │
        ▼                                        │
    record_question ◄────────────────────────────┘
        │   ├─ next ──► guardrail_pre (next question)
        │   └─ summary
        ▼
    assess_summary
        │
        ▼
    publish_completed → END

Mirrors orchestrators/qgen_crew.py (evaluate↔moderate ≙ generate↔critique). The
agent dispatch rides the Pub/Sub lane (ADR-253/254): the crew parks and the
subscriber-only oe_evaluator + oe_moderator answer on the bus. Cloud Model
Armor screening (ADR-152) is done HERE in the orchestrator (guardrail_pre =
learner answer, guardrail_post = grader comment) via the same tier-mapped
ModelArmorGuardrailPort qgen uses. Screening is central at chora-model-gateway
plus here, never agent-side: the executor sends only a 'BEGIN' trigger as the user
message and the grading content rides in the agent's instruction, so the
orchestrator (which holds the untrusted answer) is the correct screening point.

Per ADR-250 D1 the guardrail port is MANDATORY: an absent port is a construction
failure, not a pass-through, matching qgen's posture (qgen_crew_wiring refuses to
run when the mapping is unreadable). Unit tests inject an explicit fake. Per D2
each crew member screens under its own agent id (the pre-screen as oe_evaluator,
the post-screen as oe_moderator), so per-agent telemetry separates the two.

The rubric-weighted composite (points_earned) is computed deterministically HERE
(domain/oe_grading_crew/scoring.py) from the evaluator's per-criterion sub-scores
+ the authored rubric weights — the LLM emits ONLY {criterion_scores, comment}
(ADR-172 §D4). chora-delivery's ApplyGrading then persists points_earned verbatim.
"""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    ROLE_OE_EVALUATE,
    ROLE_OE_MODERATE,
)
from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    GuardrailScreenInput,
    ScreenResult,
    Verdict,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    reraise_if_dispatch_park,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
    AgentDispatchError,
)
from chora_ai_kernel_orchestrator.domain.oe_grading_crew.scoring import (
    ScoringError,
    compute_composite,
)
from chora_ai_kernel_orchestrator.domain.oe_grading_crew.state import (
    DEFAULT_MAX_ITERATIONS,
    EvaluationResult,
    GradedOEQuestion,
    GuardrailResult,
    ModerationResult,
    OEGradingState,
    current_question,
)

logger = logging.getLogger(__name__)

# User-facing refusal surfaced on a Cloud Model Armor BLOCK verdict (parity with
# qgen_crew._GUARDRAIL_BLOCK_USER_MESSAGE).
_GUARDRAIL_BLOCK_USER_MESSAGE = (
    "This answer couldn't be graded automatically — it may conflict with our "
    "content guidelines. An instructor will review it."
)

# grading_model_id sentinel stamped on a guardrail-blocked question so the R+
# queue + downstream can tell "blocked by safety filter" from a real 0.
_GUARDRAIL_BLOCK_MODEL_ID = "guardrail_block"


class _ExecutorLike(Protocol):
    """Duck-typed agent executor (the same seam qgen_crew uses).

    Production: the PubSubAgentExecutor built in oe_grading_crew_wiring, which
    parks the run and publishes the dispatch request. Tests: an in-memory fake
    returning canned AgentExecutorResponse objects keyed by `agent_role`.
    """

    async def execute(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        agid: str,
        agent_role: str,
        input_payload: str,
        workflow_id: str = "",
        prompt_template_id: str = "",
        context_window: list[dict[str, str]] | None = None,
        available_tools: list[str] | None = None,
    ) -> AgentExecutorResponse: ...


class GuardrailPort(Protocol):
    """Duck-typed tier-mapped Cloud Model Armor port — the SAME contract qgen
    uses (adapter.modelarmor.ModelArmorGuardrailPort). Returns a 3-state
    ScreenResult; the nodes collapse it via _screen_result_to_guardrail_result.
    """

    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult: ...


def _screen_result_to_guardrail_result(result: ScreenResult) -> GuardrailResult:
    """Collapse the port's 3-state ScreenResult into the node's GuardrailResult.
    ALLOW/INSPECT_ONLY → allowed; BLOCK → blocked + canonical refusal. Mirrors
    qgen_crew._screen_result_to_guardrail_result (Security-wins parity)."""
    if result.verdict in (Verdict.ALLOW, Verdict.INSPECT_ONLY):
        return GuardrailResult(allowed=True)
    return GuardrailResult(
        allowed=False,
        armor_verdict=f"armor:{result.reason or 'block_unspecified'}",
        user_facing_message=_GUARDRAIL_BLOCK_USER_MESSAGE,
    )


def _trace(state: OEGradingState, **row: Any) -> list[dict[str, Any]]:
    """Append-only pipeline_trace row (IMDA D2)."""
    rows = list(state.get("pipeline_trace", []))
    rows.append(row)
    return rows


def _loads(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        return {}


def _loads_list(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return []


def _thread_prompt_overrides(target: dict[str, Any], state: OEGradingState, suffix: str) -> None:
    """ADR-197 M-B.2 — thread the per-agent resolved prompt override (stored by
    the runner under ``prompt_overrides_<suffix>`` / ``prompt_version_<suffix>``
    / ``prompt_source_<suffix>``) into the executor payload ``target`` using the
    PINNED contract keys the OE Go composer + the executor's
    ``_build_session_state`` read: ``prompt_overrides_json`` (a JSON-encoded
    ``{segment_id -> body}`` map), ``resolved_prompt_version``, ``prompt_source``.

    Only set when an override actually applied (the runner stores nothing
    otherwise), so the grading payload is byte-for-byte unchanged when no
    resolver is wired OR no active override exists. ``suffix`` is ``evaluator``
    for the evaluate + assess_summary hops / ``moderator`` for the moderate hop.
    """
    segments = state.get(f"prompt_overrides_{suffix}")
    if not segments:
        return
    target["prompt_overrides_json"] = json.dumps(segments)
    version = str(state.get(f"prompt_version_{suffix}") or "")
    if version:
        target["resolved_prompt_version"] = version
    source = str(state.get(f"prompt_source_{suffix}") or "")
    if source:
        target["prompt_source"] = source


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def validate_input_node(state: OEGradingState) -> dict[str, Any]:
    """Initialise the per-question loop. Fail-loud on missing submission_id."""
    if not state.get("submission_id"):
        return {"outcome": "FAILED", "failure_message": "missing submission_id", "errors": ["missing submission_id"]}
    return {
        "current_index": 0,
        "attempt": 1,
        "graded": [],
        "moderation_notes": "",
        "max_iterations": int(state.get("max_iterations") or DEFAULT_MAX_ITERATIONS),
        "pipeline_trace": _trace(state, name="validate_input", status="OK"),
    }


async def guardrail_pre_node(state: OEGradingState, *, guardrail: GuardrailPort) -> dict[str, Any]:
    """Screen the current learner answer (untrusted free-text) pre-evaluation via
    Cloud Model Armor (ADR-152). A BLOCK routes to record_blocked (no LLM call);
    the answer never reaches the evaluator.

    Screens as oe_evaluator (ADR-250 D2): this content is bound for the
    evaluator."""
    q = current_question(state)
    if q is None:
        return {"guardrail_pre_result": GuardrailResult(allowed=True)}
    screen = await guardrail.screen(
        GuardrailScreenInput(
            tenant_id=state.get("tenant_id", ""),
            gcid=state.get("gcid", ""),
            agent_id="oe_evaluator",
            content=q.learner_response,
            direction="input",
        )
    )
    res = _screen_result_to_guardrail_result(screen)
    return {
        "guardrail_pre_result": res,
        "pipeline_trace": _trace(
            state, name="guardrail_pre", status="ALLOW" if res.allowed else "BLOCK", index=state.get("current_index", 0)
        ),
    }


def route_after_guardrail_pre(state: OEGradingState) -> str:
    res = state.get("guardrail_pre_result")
    return "blocked" if (res is not None and not res.allowed) else "evaluate"


async def record_blocked_node(state: OEGradingState) -> dict[str, Any]:
    """Record a guardrail-blocked OE question as a flagged 0 (never sent to the
    LLM) + advance the pointer. The mandatory HITL gate (ADR-172 §D6) surfaces it
    for instructor review."""
    q = current_question(state)
    res = state.get("guardrail_pre_result")
    graded = list(state.get("graded", []))
    if q is not None:
        graded.append(
            GradedOEQuestion(
                test_set_question_id=q.test_set_question_id,
                question_id=q.question_id,
                points_earned=0.0,
                points_possible=q.points_possible,
                criterion_scores_json="[]",
                comment=(res.user_facing_message if res else "") or _GUARDRAIL_BLOCK_USER_MESSAGE,
                grading_model_id=_GUARDRAIL_BLOCK_MODEL_ID,
                grading_response_id="",
                quality_flagged=True,
                attempt_count=0,
            )
        )
    return {
        "graded": graded,
        "current_index": int(state.get("current_index", 0)) + 1,
        "attempt": 1,
        "current_evaluation": None,
        "moderation_notes": "",
        "moderation_result": None,
        "guardrail_pre_result": None,
    }


async def evaluate_node(state: OEGradingState, *, executor: _ExecutorLike) -> dict[str, Any]:
    """Dispatch oe_evaluate (evaluate mode) for the current OE question. The LLM
    emits {criterion_scores, comment}; the rubric-weighted composite is computed
    deterministically here (scoring.compute_composite), never by the LLM."""
    q = current_question(state)
    if q is None:
        return {"errors": ["evaluate: no current question"]}
    attempt = int(state.get("attempt", 1))
    payload = {
        "mode": "evaluate",
        "submission_id": state.get("submission_id", ""),
        "test_set_question_id": q.test_set_question_id,
        "question_id": q.question_id,
        "prompt": q.prompt,
        "rubric_json": q.rubric_json,
        "model_answer": q.model_answer,
        "learner_answer": q.learner_response,
        "points_possible": q.points_possible,
        "subject": q.subject or state.get("subject", ""),
        "topic": q.topic,
        "prior_moderator_feedback": state.get("moderation_notes", ""),
        "attempt_index": attempt - 1,
        "max_iterations": int(state.get("max_iterations") or DEFAULT_MAX_ITERATIONS),
        "gcid": state.get("gcid", ""),
        "traceparent": state.get("traceparent", ""),
        "tracestate": state.get("tracestate", ""),
    }
    # ADR-197 M-B.2 — thread the oe_evaluator prompt override (when resolved).
    _thread_prompt_overrides(payload, state, "evaluator")
    try:
        resp = await executor.execute(
            execution_id=f"{state.get('submission_id', '')}:{q.test_set_question_id}:{attempt}",
            tenant_id=state.get("tenant_id", ""),
            agid="",
            agent_role=ROLE_OE_EVALUATE,
            input_payload=json.dumps(payload),
        )
    except AgentDispatchError as exc:
        # ADR-254 D5: a FAILED completion (an agent failure, or the park reaper
        # settling a run the agent never answered) SETTLES the submission as
        # outcome=FAILED instead of escaping the graph with no terminal, so the
        # caller gets a status, never silence. Typed on purpose: a LangGraph
        # park is not an AgentDispatchError and still propagates.
        return _dispatch_failed(state, step="evaluate", exc=exc, attempt=attempt)
    raw = _loads(resp.output_payload)
    criterion_scores = raw.get("criterion_scores")
    if not isinstance(criterion_scores, list):
        criterion_scores = []
    try:
        points_earned = compute_composite(q.rubric_json, criterion_scores, q.points_possible)
        scoring_status = "OK"
    except ScoringError as exc:
        # A grade couldn't be grounded (malformed rubric / zero weight-sum). This
        # is gated upstream by qgen-critic; if it slips through, degrade to 0 and
        # let the mandatory HITL gate (§D6) catch it — every grade is reviewed.
        logger.warning("oe_grading.scoring_failed tsq=%s: %s", q.test_set_question_id, exc)
        points_earned = 0.0
        scoring_status = "SCORING_ERROR"
    evaluation = EvaluationResult(
        points_earned=points_earned,
        points_possible=q.points_possible,
        criterion_scores_json=json.dumps(criterion_scores),
        comment=str(raw.get("comment", "")),
        grading_model_id=str(raw.get("model_id") or raw.get("grading_model_id") or ""),
        grading_response_id=str(raw.get("response_id") or raw.get("grading_response_id") or ""),
    )
    return {
        "current_evaluation": evaluation,
        "pipeline_trace": _trace(
            state,
            name="evaluate",
            status=scoring_status,
            index=state.get("current_index", 0),
            attempt=attempt,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        ),
    }


async def guardrail_post_node(state: OEGradingState, *, guardrail: GuardrailPort) -> dict[str, Any]:
    """Screen the evaluator's comment (model output) post-evaluation via Cloud
    Model Armor (ADR-152). A block redacts the comment but keeps the score (the
    moderator still judges the grading).

    Screens as oe_moderator (ADR-250 D2): this node gates the comment on its way
    into the moderator (evaluate → guardrail_post → moderate), so attributing it
    to the consuming member keeps per-agent telemetry separable. Attribution
    only: both ids are declared and enforced balanced, so no tier moves."""
    ev = state.get("current_evaluation")
    if ev is None:
        return {"guardrail_post_result": GuardrailResult(allowed=True)}
    screen = await guardrail.screen(
        GuardrailScreenInput(
            tenant_id=state.get("tenant_id", ""),
            gcid=state.get("gcid", ""),
            agent_id="oe_moderator",
            content=ev.comment,
            direction="output",
        )
    )
    res = _screen_result_to_guardrail_result(screen)
    if not res.allowed:
        ev = EvaluationResult(
            points_earned=ev.points_earned,
            points_possible=ev.points_possible,
            criterion_scores_json=ev.criterion_scores_json,
            comment=res.user_facing_message or "[comment redacted by safety filter]",
            grading_model_id=ev.grading_model_id,
            grading_response_id=ev.grading_response_id,
        )
    return {
        "current_evaluation": ev,
        "guardrail_post_result": res,
        "pipeline_trace": _trace(
            state,
            name="guardrail_post",
            status="ALLOW" if res.allowed else "BLOCK",
            index=state.get("current_index", 0),
        ),
    }


async def moderate_node(state: OEGradingState, *, executor: _ExecutorLike) -> dict[str, Any]:
    """Dispatch oe_moderate — qualitative judge of the evaluator's grading. The
    LLM emits {accepted, feedback}; on reject the evaluator re-grades."""
    q = current_question(state)
    ev = state.get("current_evaluation")
    if q is None or ev is None:
        return {"moderation_result": ModerationResult(accepted=True)}
    attempt = int(state.get("attempt", 1))
    evaluation_json = json.dumps(
        {
            "points_earned": ev.points_earned,
            "points_possible": ev.points_possible,
            "criterion_scores": _loads_list(ev.criterion_scores_json),
            "comment": ev.comment,
        }
    )
    payload = {
        "submission_id": state.get("submission_id", ""),
        "test_set_question_id": q.test_set_question_id,
        "question_id": q.question_id,
        "prompt": q.prompt,
        "rubric_json": q.rubric_json,
        "model_answer": q.model_answer,
        "learner_answer": q.learner_response,
        "subject": q.subject or state.get("subject", ""),
        "topic": q.topic,
        "evaluation_json": evaluation_json,
        "attempt_index": attempt - 1,
        "max_iterations": int(state.get("max_iterations") or DEFAULT_MAX_ITERATIONS),
        "gcid": state.get("gcid", ""),
        "traceparent": state.get("traceparent", ""),
        "tracestate": state.get("tracestate", ""),
    }
    # ADR-197 M-B.2 — thread the oe_moderator prompt override (when resolved).
    _thread_prompt_overrides(payload, state, "moderator")
    try:
        resp = await executor.execute(
            execution_id=f"{state.get('submission_id', '')}:{q.test_set_question_id}:mod:{attempt}",
            tenant_id=state.get("tenant_id", ""),
            agid="",
            agent_role=ROLE_OE_MODERATE,
            input_payload=json.dumps(payload),
        )
    except AgentDispatchError as exc:
        # ADR-254 D5: see evaluate_node; a FAILED moderate settles the run.
        return _dispatch_failed(state, step="moderate", exc=exc, attempt=attempt)
    raw = _loads(resp.output_payload)
    moderation = ModerationResult(
        accepted=bool(raw.get("accepted", True)),
        # ADR-172 moderator output schema is {accepted, feedback}; keep a
        # moderation_notes fallback for older/alternate emissions.
        moderation_notes=str(raw.get("feedback") or raw.get("moderation_notes") or ""),
        suggested_revisions=list(raw.get("suggested_revisions", []) or []),
    )
    return {
        "moderation_result": moderation,
        "pipeline_trace": _trace(
            state,
            name="moderate",
            status="ACCEPT" if moderation.accepted else "REJECT",
            index=state.get("current_index", 0),
            attempt=attempt,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        ),
    }


async def quality_gate_node(state: OEGradingState) -> dict[str, Any]:
    """Decide retry vs record + bump attempt on a retry. Writes a single
    ``loop_decision`` that the router reads, so the gate and the router cannot
    disagree.

    attempt is 1-based (attempt=1 is the first grade). max_iterations=2 allows up
    to 3 total grades (1 initial + 2 re-grades): the loop retries while
    attempt < max_iters + 1, then records the last evaluator output (flagged on
    exhaustion). The increment happens ONLY on a retry, and the router keys off
    loop_decision rather than re-deriving from the (now-incremented) attempt —
    fixing the prior off-by-one where the router saw the bumped attempt and
    recorded a grade early."""
    mod = state.get("moderation_result")
    attempt = int(state.get("attempt", 1))
    max_iters = int(state.get("max_iterations") or DEFAULT_MAX_ITERATIONS)
    accepted = bool(mod and mod.accepted)
    if accepted or attempt >= max_iters + 1:
        return {"loop_decision": "record"}
    # retry — carry the moderator feedback into the next evaluation
    return {
        "attempt": attempt + 1,
        "moderation_notes": (mod.moderation_notes if mod else ""),
        "loop_decision": "retry",
    }


def route_after_quality_gate(state: OEGradingState) -> str:
    return "retry" if state.get("loop_decision") == "retry" else "record"


def route_after_evaluate(state: OEGradingState) -> str:
    """A FAILED dispatch goes straight to the terminal; no screening, no
    moderation, no further dispatch (ADR-254 D5)."""
    return "failed" if state.get("outcome") == "FAILED" else "screen"


def route_after_moderate(state: OEGradingState) -> str:
    return "failed" if state.get("outcome") == "FAILED" else "gate"


def _dispatch_failed(
    state: OEGradingState,
    *,
    step: str,
    exc: Exception,
    attempt: int,
) -> dict[str, Any]:
    """The settled shape of a FAILED agent dispatch: outcome + message + trace.

    The runner publishes ``submission_completed`` with ``outcome=FAILED`` and
    this ``failure_message`` from the terminal state, so chora-delivery learns
    the submission could not be graded instead of waiting forever.
    """
    message = f"{step}: {exc}"[:1000]
    logger.error(
        "oe_grading.dispatch_failed",
        extra={
            "step": step,
            "submission_id": state.get("submission_id", ""),
            "tenant_id": state.get("tenant_id", ""),
            "attempt": attempt,
            "err": message,
        },
    )
    return {
        "outcome": "FAILED",
        "failure_message": message,
        "errors": [message],
        "pipeline_trace": _trace(
            state,
            name=step,
            status="DISPATCH_FAILED",
            index=state.get("current_index", 0),
            attempt=attempt,
        ),
    }


async def record_question_node(state: OEGradingState) -> dict[str, Any]:
    """Record the current OE question's terminal grade + advance the pointer."""
    q = current_question(state)
    ev = state.get("current_evaluation")
    mod = state.get("moderation_result")
    graded = list(state.get("graded", []))
    if q is not None and ev is not None:
        graded.append(
            GradedOEQuestion(
                test_set_question_id=q.test_set_question_id,
                question_id=q.question_id,
                points_earned=ev.points_earned,
                points_possible=ev.points_possible,
                criterion_scores_json=ev.criterion_scores_json,
                comment=ev.comment,
                grading_model_id=ev.grading_model_id,
                grading_response_id=ev.grading_response_id,
                quality_flagged=not bool(mod and mod.accepted),
                attempt_count=int(state.get("attempt", 1)),
            )
        )
    return {
        "graded": graded,
        "current_index": int(state.get("current_index", 0)) + 1,
        "attempt": 1,
        "current_evaluation": None,
        "moderation_notes": "",
        "moderation_result": None,
        "guardrail_pre_result": None,
    }


def route_after_record(state: OEGradingState) -> str:
    return "next" if current_question(state) is not None else "summary"


def _build_results_digest(state: OEGradingState) -> tuple[str, float, int, float]:
    """Build the verbatim all-answers digest the assess_summary agent narrates
    from (every question's type / topic / outcome). Returns
    (digest, total_earned, total_possible, score_percent)."""
    graded = state.get("graded", [])
    oe_earned = sum(g.points_earned for g in graded)
    total_earned = oe_earned + float(state.get("mcq_points_earned", 0) or 0)
    total_possible = int(state.get("total_points_possible", 0) or 0)
    pct = round((total_earned / total_possible) * 100, 1) if total_possible else 0.0
    passing = int(state.get("passing_threshold_percent", 0) or 0)

    lines: list[str] = [
        f"Overall: {total_earned} of {total_possible} points ({pct}%); passing threshold {passing}%.",
    ]
    for m in state.get("mcq_results", []):
        lines.append(f"- [MCQ] {'correct' if m.correct else 'incorrect'} ({m.points_earned}/{m.points_possible} pts)")
    oe_by_id = {qq.test_set_question_id: qq for qq in state.get("oe_questions", [])}
    for g in graded:
        qq = oe_by_id.get(g.test_set_question_id)
        prompt = qq.prompt if qq else ""
        topic = qq.topic if qq else ""
        lines.append(
            f"- [OE] topic={topic or 'n/a'} | {g.points_earned}/{g.points_possible} pts "
            f"| Q: {prompt} | grader comment: {g.comment}"
        )
    return "\n".join(lines), total_earned, total_possible, pct


async def assess_summary_node(state: OEGradingState, *, executor: _ExecutorLike) -> dict[str, Any]:
    """Single assess_summary pass — oe_evaluate (assess_summary mode) writes the
    whole-assessment overall comment spanning MCQ + OE (no moderator loop)."""
    results_digest, total_earned, total_possible, pct = _build_results_digest(state)
    payload = {
        "mode": "assess_summary",
        "submission_id": state.get("submission_id", ""),
        "assessment_id": state.get("assessment_id", ""),
        "subject": state.get("subject", ""),
        "passing_threshold_percent": int(state.get("passing_threshold_percent", 0) or 0),
        "results_digest": results_digest,
        "gcid": state.get("gcid", ""),
        "traceparent": state.get("traceparent", ""),
        "tracestate": state.get("tracestate", ""),
    }
    # ADR-197 M-B.2 — assess_summary is the oe_evaluator in assess_summary mode,
    # so it carries the evaluator's prompt override (when resolved).
    _thread_prompt_overrides(payload, state, "evaluator")
    try:
        resp = await executor.execute(
            execution_id=f"{state.get('submission_id', '')}:summary",
            tenant_id=state.get("tenant_id", ""),
            agid="",
            agent_role=ROLE_OE_EVALUATE,
            input_payload=json.dumps(payload),
        )
        raw = _loads(resp.output_payload)
        overall = str(raw.get("overall_comment", "")) or str(raw.get("comment", ""))
        return {
            "overall_comment": overall,
            "overall_comment_model_id": str(raw.get("model_id") or raw.get("grading_model_id") or ""),
            "overall_comment_response_id": str(raw.get("response_id") or ""),
            "pipeline_trace": _trace(
                state,
                name="assess_summary",
                status="OK",
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            ),
        }
    except Exception as exc:  # noqa: BLE001 — summary is best-effort; never fail the whole submission
        # ADR-253: a park is control flow, not a failure. Without this the
        # Pub/Sub dispatch below is swallowed, the summary is never published,
        # and the run settles with an empty overall comment.
        reraise_if_dispatch_park(exc)
        logger.warning("oe_grading.assess_summary_failed: %s", exc)
        return {"overall_comment": "", "pipeline_trace": _trace(state, name="assess_summary", status="ERROR")}


async def publish_completed_node(state: OEGradingState) -> dict[str, Any]:
    """Terminal — mark outcome. The runner reads the terminal state and
    publishes chora.delivery.grading.submission_completed.v1."""
    graded = state.get("graded", [])
    if state.get("outcome") == "FAILED":
        # ADR-254 D5: a settled dispatch failure stays FAILED; the runner
        # publishes it with the failure_message already in state.
        return {
            "pipeline_trace": _trace(state, name="publish_completed", status="FAILED", graded=len(graded)),
        }
    any_flagged = any(g.quality_flagged for g in graded)
    return {
        "outcome": "PARTIAL" if any_flagged else "SUCCESS",
        "pipeline_trace": _trace(state, name="publish_completed", status="OK", graded=len(graded)),
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_oe_grading_crew_graph(
    *,
    executor: _ExecutorLike,
    guardrail: GuardrailPort,
    checkpointer: Any | None = None,
) -> Any:
    """Compile the OE grading crew StateGraph.

    checkpointer: PostgresSaver in prod (keyed on thread_id = submission_id for
    D6 pod-death resume); None for unit tests.
    guardrail: ModelArmorGuardrailPort in prod (real Cloud Model Armor
    screening). MANDATORY per ADR-250 D1: a workload that cannot resolve its
    guardrail configuration does not call a model, so an absent port raises here
    rather than letting an unscreened learner answer reach the evaluator. Tests
    inject an explicit fake; there is no pass-through branch to fall into.
    """
    if guardrail is None:
        raise ValueError(
            "build_oe_grading_crew_graph: guardrail port is mandatory (ADR-250 "
            "D1); refusing to compile a crew that would grade unscreened "
            "learner answers"
        )
    graph: StateGraph = StateGraph(OEGradingState)

    graph.add_node("validate_input", validate_input_node)
    graph.add_node("guardrail_pre", partial(guardrail_pre_node, guardrail=guardrail))
    graph.add_node("record_blocked", record_blocked_node)
    graph.add_node("evaluate", partial(evaluate_node, executor=executor))
    graph.add_node("guardrail_post", partial(guardrail_post_node, guardrail=guardrail))
    graph.add_node("moderate", partial(moderate_node, executor=executor))
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("record_question", record_question_node)
    graph.add_node("assess_summary", partial(assess_summary_node, executor=executor))
    graph.add_node("publish_completed", publish_completed_node)

    graph.add_edge(START, "validate_input")
    # validate → (has OE questions ? guardrail_pre : assess_summary)
    graph.add_conditional_edges(
        "validate_input",
        lambda s: "grade" if current_question(s) is not None and s.get("outcome") != "FAILED" else "summary",
        {"grade": "guardrail_pre", "summary": "assess_summary"},
    )
    # guardrail_pre → (block ? record_blocked : evaluate)
    graph.add_conditional_edges(
        "guardrail_pre",
        route_after_guardrail_pre,
        {"evaluate": "evaluate", "blocked": "record_blocked"},
    )
    graph.add_conditional_edges(
        "record_blocked",
        route_after_record,
        {"next": "guardrail_pre", "summary": "assess_summary"},
    )
    # evaluate → (dispatch FAILED ? terminal : guardrail_post) (ADR-254 D5)
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"screen": "guardrail_post", "failed": "publish_completed"},
    )
    graph.add_edge("guardrail_post", "moderate")
    # moderate → (dispatch FAILED ? terminal : quality_gate)
    graph.add_conditional_edges(
        "moderate",
        route_after_moderate,
        {"gate": "quality_gate", "failed": "publish_completed"},
    )
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {"retry": "evaluate", "record": "record_question"},
    )
    graph.add_conditional_edges(
        "record_question",
        route_after_record,
        {"next": "guardrail_pre", "summary": "assess_summary"},
    )
    graph.add_edge("assess_summary", "publish_completed")
    graph.add_edge("publish_completed", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
