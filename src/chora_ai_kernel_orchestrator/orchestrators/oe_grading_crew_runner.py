"""OEGradingCrewRunner — drives the OE grading StateGraph from Pub/Sub (ADR-172,
transport per ADR-253).

Two entry points, because under ADR-253 D2 a submission is graded across MANY
invocations rather than one:

``handle_requested(event)`` — one submission_requested.v1:
  1. parse chora.delivery.grading.submission_requested.v1 → OEGradingState
     (split questions[] into oe_questions + mcq_results)
  2. graph.ainvoke(state, thread_id=submission_id)
  3. settle: the graph normally comes back PARKED on its first agent dispatch,
     and a parked return publishes nothing.

``handle_completion(payload)`` — one agent completion event:
  1. graph.ainvoke(Command(resume=payload), thread_id=payload["thread_id"])
  2. settle: parked again on the next dispatch (publish nothing), or finished
     (publish chora.delivery.grading.submission_completed.v1 via the JSON outbox).

⚠ The interrupt check in ``_settle`` is the load-bearing line. An ``ainvoke``
return no longer means "graded" — it usually means "waiting on an agent". Both
mistakes are silent: publishing early grades a learner on an empty result set,
never publishing leaves the submission in PENDING_OE_GRADING forever.

Idempotency rides the LangGraph checkpointer (same thread_id → same terminal
state) + the outbox idempotency_key. It is NOT made redundant by the
single-transaction dispatch write (ADR-253 D3a): Pub/Sub is at-least-once
regardless of how cleanly the publish side commits.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from langgraph.types import Command

from chora_ai_kernel_orchestrator.adapter.pubsub.oe_grading_crew_publisher import (
    OEGradingCrewOutboxWriter,
)
from chora_ai_kernel_orchestrator.domain.oe_grading_crew.state import (
    DEFAULT_MAX_ITERATIONS,
    GradedOEQuestion,
    MCQResult,
    OEGradingState,
    OEQuestionInput,
)
from chora_ai_kernel_orchestrator.domain.prompt_registry import (
    EMBEDDED_PROMPT_VERSIONS,
    SOURCE_EMBEDDED,
)
from chora_ai_kernel_orchestrator.observability.agent_trace import (
    agent_span_traceparent,
)

logger = logging.getLogger(__name__)

# Stable snake_case crew identifier for the OE grading crew — feeds the O+
# /o/agents Crews + Agents hierarchy (sibling of MCQ_AI_ASSIST_CREW_NAME in
# qgen_crew_runner). The two agents map to the deployed GKE Deployments
# chora-oe-evaluator + chora-oe-moderator.
OE_GRADING_CREW_NAME = "oe_grading"

# Per-agent registry ids (agid). MUST match the deployed agent names + the
# /o/agents registry tiles. NEVER "qgen_crew".
AGID_OE_EVALUATOR = "oe_evaluator"
AGID_OE_MODERATOR = "oe_moderator"

# IMDA dimension label per ADR-141 — an agent grading decision is D1
# accountability evidence (same as the qgen path).
IMDA_DIMENSION_ACCOUNTABILITY = "accountability"

# OE grading questions are always open-ended; the attributes["question_type"]
# the O+ consumer reads is therefore "oe" for both OE agents.
_OE_QUESTION_TYPE = "oe"


def _str(d: dict[str, Any], k: str) -> str:
    v = d.get(k)
    return str(v) if v is not None else ""


def _num(d: dict[str, Any], k: str, default: float = 0) -> float:
    try:
        return float(d.get(k, default) or default)
    except (TypeError, ValueError):
        return default


def parse_requested(event: dict[str, Any]) -> OEGradingState:
    """Project submission_requested.v1 (body + envelope-merged attrs) into state."""
    questions = event.get("questions") or []
    oe_questions: list[OEQuestionInput] = []
    mcq_results: list[MCQResult] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        # chora-delivery emits question_type as UPPERCASE "MCQ"/"OE"
        # (assessment_handler.emitSubmissionRequestedEvent uses string(QuestionType));
        # case-fold so the projection matches regardless of producer casing.
        qtype = _str(q, "question_type").strip().lower()
        if qtype == "oe":
            oe_questions.append(
                OEQuestionInput(
                    test_set_question_id=_str(q, "test_set_question_id"),
                    question_id=_str(q, "question_id"),
                    prompt=_str(q, "prompt"),
                    rubric_json=_str(q, "rubric_json") or "[]",
                    model_answer=_str(q, "model_answer"),
                    learner_response=_str(q, "oe_response_text"),
                    points_possible=int(_num(q, "points_possible")),
                    subject=_str(q, "subject"),
                    topic=_str(q, "topic"),
                )
            )
        elif qtype == "mcq":
            mcq_results.append(
                MCQResult(
                    test_set_question_id=_str(q, "test_set_question_id"),
                    correct=bool(q.get("mcq_correct", False)),
                    points_earned=_num(q, "mcq_points_earned"),
                    points_possible=int(_num(q, "points_possible")),
                )
            )
    state: OEGradingState = {
        "grading_job_id": _str(event, "grading_job_id"),
        "submission_id": _str(event, "submission_id"),
        "assessment_id": _str(event, "assessment_id"),
        "tenant_id": _str(event, "tenant_id"),
        "gcid": _str(event, "learner_gcid"),
        "passing_threshold_percent": int(_num(event, "passing_threshold_percent")),
        "model_tier": _str(event, "model_tier") or "T1",
        "per_question_feedback_enabled": bool(event.get("per_question_feedback_enabled", True)),
        "subject": _str(event, "subject"),
        "total_points_possible": int(_num(event, "total_points_possible")),
        "mcq_points_earned": _num(event, "mcq_points_earned"),
        "traceparent": _str(event, "traceparent"),
        "tracestate": _str(event, "tracestate"),
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "oe_questions": oe_questions,
        "mcq_results": mcq_results,
        "graded": [],
        "pipeline_trace": [],
        "errors": [],
    }
    return state


def _graded_to_wire(g: GradedOEQuestion) -> dict[str, Any]:
    return {
        "test_set_question_id": g.test_set_question_id,
        "question_id": g.question_id,
        "points_earned": g.points_earned,
        "points_possible": g.points_possible,
        "criterion_scores_json": g.criterion_scores_json,
        "comment": g.comment,
        "grading_model_id": g.grading_model_id,
        "grading_response_id": g.grading_response_id,
        "quality_flagged": g.quality_flagged,
        # NOTE: attempt_count is internal crew state (moderation-iteration count)
        # — chora-delivery does not persist it, and quality_flagged already
        # carries the "needs review" signal. Deliberately NOT put on the wire
        # (emit-but-drop is silent data loss). Add end-to-end if R+ ever surfaces it.
    }


class OEGradingCrewRunner:
    """Wires the compiled graph + the outbox publisher."""

    def __init__(
        self,
        *,
        graph: Any,
        publisher: OEGradingCrewOutboxWriter,
        agent_decision_emitter: Any | None = None,
        thread_id_for: Any | None = None,
        prompt_resolver: Any | None = None,
    ) -> None:
        self._graph = graph
        self._publisher = publisher
        # Optional AgentDecisionLogOutboxWriter (Gate #8 — per-agent O+
        # /o/agents tile hydration). When None (unit tests / not wired) the
        # runner behaves exactly as the pre-emission baseline.
        self._agent_decision_emitter = agent_decision_emitter
        self._thread_id_for = thread_id_for or (lambda s: s.get("submission_id", ""))
        # ADR-197 M-B.2 — optional PromptResolver. None ⇒ no prompt-override
        # resolution (byte-identical pre-registry baseline). When wired,
        # handle_requested resolves the active override per agent role
        # (oe_evaluator + oe_moderator) and threads it into the grading payload
        # + durable record. Resolver errors are NOT swallowed (fail-loud).
        self._prompt_resolver = prompt_resolver

    async def handle_requested(self, event: dict[str, Any]) -> None:
        state = parse_requested(event)
        submission_id = state.get("submission_id", "")
        if not submission_id:
            logger.warning("oe_grading_crew_runner: missing submission_id; dropping")
            return
        thread_id = self._thread_id_for(state)
        # ADR-197 M-B.2 — resolve the active prompt override per agent role
        # BEFORE the graph runs so the segment map rides every grading executor
        # call + the durable AgentDecisionLog. No-op when no resolver is wired OR
        # no active override applies (byte-identical baseline).
        await self._apply_prompt_overrides(state)
        logger.info(
            "oe_grading_crew_runner.requested",
            extra={
                "submission_id": submission_id,
                "tenant_id": state.get("tenant_id"),
                "oe_questions": len(state.get("oe_questions", [])),
                "thread_id": thread_id,
            },
        )
        terminal: dict[str, Any] = await self._graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}})
        await self._settle(terminal, thread_id=thread_id, state=state)

    async def handle_completion(self, completion: dict[str, Any]) -> None:
        """Resume a parked run with one agent completion (ADR-253 D2).

        The completion carries the thread it belongs to, so this consumer keeps
        no state of its own: it resumes that thread and settles whatever comes
        back. Most completions land the run on its NEXT dispatch and publish
        nothing; the last one finishes the submission.

        Refuses a completion with no ``thread_id`` — resuming a guessed thread
        would inject one submission's grade into another's run.
        """
        thread_id = str(completion.get("thread_id") or "").strip()
        if not thread_id:
            raise ValueError(
                "oe_grading_crew_runner.handle_completion: completion carries no "
                "thread_id; refusing to resume a guessed thread"
            )
        logger.info(
            "oe_grading_crew_runner.completion",
            extra={
                "thread_id": thread_id,
                "status": completion.get("status"),
                "idempotency_key": completion.get("idempotency_key", ""),
            },
        )
        terminal: dict[str, Any] = await self._graph.ainvoke(
            Command(resume=dict(completion)),
            config={"configurable": {"thread_id": thread_id}},
        )
        await self._settle(terminal, thread_id=thread_id, state=terminal)

    async def _settle(
        self,
        terminal: dict[str, Any],
        *,
        thread_id: str,
        state: OEGradingState | dict[str, Any],
    ) -> None:
        """Publish the terminal — but ONLY when the run actually finished.

        ⚠ Under ADR-253 D2 an ``ainvoke`` return does not mean the submission is
        graded. The graph parks at every agent dispatch, and a parked return
        carries ``__interrupt__``. Publishing on that return would tell
        chora-delivery the submission is graded before any agent has answered,
        so the interrupt check is the load-bearing line in this method.
        """
        if terminal.get("__interrupt__"):
            logger.info(
                "oe_grading_crew_runner.parked",
                extra={"thread_id": thread_id, "submission_id": terminal.get("submission_id", "")},
            )
            return

        # AgentDecisionLog emit (Gate #8) — one oe_evaluator + one oe_moderator
        # decision per submission, IMDA D1 accountability. Write-to-outbox
        # BEFORE the terminal publish (same rationale as qgen): a mid-run crash
        # leaves the decision rows pending for the dispatcher rather than
        # dropped. Best-effort — a transient emit failure MUST NOT suppress the
        # submission_completed publish.
        await self._emit_agent_decisions(state, terminal)

        submission_id = str(terminal.get("submission_id") or state.get("submission_id") or thread_id)
        graded = [_graded_to_wire(g) for g in terminal.get("graded", [])]
        await self._publisher.publish_submission_completed(
            submission_id=submission_id,
            grading_job_id=str(terminal.get("grading_job_id") or state.get("grading_job_id") or ""),
            assessment_id=terminal.get("assessment_id", state.get("assessment_id", "")),
            tenant_id=terminal.get("tenant_id", state.get("tenant_id", "")),
            learner_gcid=terminal.get("gcid", state.get("gcid", "")),
            graded=graded,
            overall_comment=terminal.get("overall_comment", ""),
            overall_comment_model_id=terminal.get("overall_comment_model_id", ""),
            # IMDA D2 provenance — the LLM response id for the overall comment
            # (delivery's SubmissionGrading.OverallCommentResponseID). The graph
            # produces it in assess_summary_node; thread it through so the D2 audit
            # trail for the overall comment is complete (mirrors per-question
            # grading_response_id).
            overall_comment_response_id=terminal.get("overall_comment_response_id", ""),
            outcome=terminal.get("outcome", "SUCCESS"),
            failure_message=terminal.get("failure_message", ""),
            traceparent=str(terminal.get("traceparent") or state.get("traceparent") or ""),
            tracestate=str(terminal.get("tracestate") or state.get("tracestate") or ""),
        )

    async def _apply_prompt_overrides(self, state: OEGradingState) -> None:
        """ADR-197 M-B.2 — resolve + store the per-agent prompt override.

        For each OE agent role (oe_evaluator + oe_moderator) ask the resolver for
        the active override for ``(tenant_id, agid)`` and, ONLY when an override
        actually applied (``segments`` non-empty), store the winning segment map
        + version + source under per-agent state keys. The grading nodes thread
        these into the executor payload; ``_emit_agent_decisions`` stamps the
        version + source onto the durable record.

        No resolver wired ⇒ no-op. Embedded default ⇒ no keys set ⇒
        byte-identical pre-registry behaviour. Resolver errors are NOT swallowed
        (fail-loud — the caller NACKs).
        """
        if self._prompt_resolver is None:
            return
        tenant_id = state.get("tenant_id", "")
        for agid, suffix in (
            (AGID_OE_EVALUATOR, "evaluator"),
            (AGID_OE_MODERATOR, "moderator"),
        ):
            resolved = await self._prompt_resolver.resolve(tenant_id, agid)
            segments = getattr(resolved, "segments", None)
            if not segments:
                continue
            state[f"prompt_overrides_{suffix}"] = dict(segments)  # type: ignore[literal-required]
            state[f"prompt_version_{suffix}"] = str(getattr(resolved, "version", "") or "")  # type: ignore[literal-required]
            state[f"prompt_source_{suffix}"] = str(getattr(resolved, "source", "") or "")  # type: ignore[literal-required]

    async def _emit_agent_decisions(
        self,
        state: OEGradingState | dict[str, Any],
        terminal: dict[str, Any],
    ) -> None:
        """Emit ONE oe_evaluator + ONE oe_moderator AgentDecisionLog event per
        submission (Gate #8), mirroring the qgen outbox-writer pattern so O+
        /o/agents OE-grading tiles populate.

        The crew grades each OE question via an oe_evaluator → oe_moderator
        reject/re-grade loop; we attribute ONE run-level decision to each agent:

          * oe_evaluator (the grade) — FAILED outcome → "failed", else any
            quality_flagged question → "completed_with_warning", else "accepted".
          * oe_moderator (moderation) — any quality_flagged → the moderator
            rejected ≥1 grade until retries exhausted (or a guardrail-blocked
            answer was recorded flagged) → "completed_with_warning", else
            "accepted".

        assist_id / crew_id == submission_id (the OE runner is submission-scoped,
        so the submission IS the orchestration identifier). tenant_id is the REAL
        per-request tenant — NEVER normalised to platform. Tokens are split per
        agent (evaluator = evaluate + assess_summary hops; moderator = moderate
        hop) so the two tiles do not double-count.

        Best-effort + INDEPENDENTLY guarded per
        [[feedback-d6-resilience-first-class]] — a transient emit failure on one
        agent MUST NOT suppress the other (or the terminal publish).
        """
        if self._agent_decision_emitter is None:
            return

        submission_id = state.get("submission_id", "") or terminal.get("submission_id", "")
        tenant_id = terminal.get("tenant_id", "") or state.get("tenant_id", "")
        gcid = terminal.get("gcid", "") or state.get("gcid", "")
        graded = terminal.get("graded") or []
        pipeline_trace = terminal.get("pipeline_trace") or []
        outcome = str(terminal.get("outcome") or "")
        max_iterations = int(terminal.get("max_iterations") or state.get("max_iterations") or DEFAULT_MAX_ITERATIONS)

        flagged_count = sum(1 for g in graded if getattr(g, "quality_flagged", False))
        any_flagged = flagged_count > 0
        graded_count = len(graded)
        # Most re-grades on any single question — the representative attempt
        # count for the agents' decisions.
        max_attempt = max(
            (int(getattr(g, "attempt_count", 0) or 0) for g in graded),
            default=0,
        )

        occurred_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
        guardrail_outcome = _oe_guardrail_outcome(pipeline_trace)

        if outcome == "FAILED":
            evaluator_decision = "failed"
        elif any_flagged:
            evaluator_decision = "completed_with_warning"
        else:
            evaluator_decision = "accepted"
        moderator_decision = "completed_with_warning" if any_flagged else "accepted"

        # ADR-197 M-A.3 — prompt-shaping condition discriminants per agent,
        # mirroring the Go EvaluatorConditions/ModeratorConditions extractors so
        # the durable record + the live span agree. The evaluator mode is
        # ``evaluate`` when OE questions were graded, else ``assess_summary``
        # (MCQ-only submission). attempt_index is 0-based (max_attempt is 1-based
        # — the most re-grades on any single question). has_prior_moderator_
        # feedback is true when a re-grade ran (the moderator rejected ≥1 grade
        # and fed feedback back into the evaluator). subject is the run-wide
        # subject the orchestrator genuinely holds (omitted when blank).
        oe_subject = str(terminal.get("subject") or state.get("subject") or "")
        evaluator_mode = "evaluate" if graded else "assess_summary"
        evaluator_conditions = _oe_evaluator_conditions(
            mode=evaluator_mode,
            attempt_index=max_attempt - 1,
            subject=oe_subject,
            has_prior_moderator_feedback=max_attempt > 1,
        )
        moderator_conditions = _oe_moderator_conditions(
            attempt_index=max_attempt - 1,
            subject=oe_subject,
        )
        # ADR-197 M-B.2 + CHO-2364 - stamp the prompt version + source onto each
        # agent's condition map so the durable AgentDecisionLog records WHICH
        # prompt shaped the grading (rides the M-A.3 field-21 chain - no proto
        # change). Read from state (the runner stored the resolver values there
        # before ainvoke). EVERY decision stamps: resolver values on the override
        # path, else EMBEDDED_PROMPT_VERSIONS[agid] + "embedded".
        _stamp_prompt_override(evaluator_conditions, state, "evaluator")
        _stamp_prompt_override(moderator_conditions, state, "moderator")

        # (agid, decision, output_summary, token-hops) per agent.
        specs: list[tuple[str, str, str, tuple[str, ...]]] = [
            (
                AGID_OE_EVALUATOR,
                evaluator_decision,
                str(terminal.get("overall_comment") or ""),
                ("evaluate", "assess_summary"),
            ),
            (
                AGID_OE_MODERATOR,
                moderator_decision,
                f"{flagged_count} of {graded_count} graded question(s) flagged for review.",
                ("moderate",),
            ),
        ]

        for agid, decision, notes, hops in specs:
            prompt_tokens, completion_tokens, cached_tokens = _oe_aggregate_tokens(pipeline_trace, hops)
            # Select the agent's condition discriminants (ADR-197 M-A.3) so each
            # durable record carries the SAME map the live span stamped.
            prompt_conditions = evaluator_conditions if agid == AGID_OE_EVALUATOR else moderator_conditions
            # Per-agent marker span: give THIS agent's decision a DISTINCT span
            # id within the shared run trace so the O+ /o/agents 'View in Cloud
            # Trace' deep-link lands on oe_evaluator / oe_moderator individually
            # (previously both shared state's traceparent → identical span id).
            # The span ALSO carries the decision EVIDENCE — the verdict + a
            # bounded reasoning summary (§9) — so an auditor who deep-links from
            # O+ Decision-Traces reads WHAT each grading agent decided + WHY on
            # the span (IMDA D2). PII discipline: only the verdict + the agent's
            # own summary notes ride the span — never the raw learner answer.
            # Best-effort — returns the inbound traceparent unchanged on failure.
            agent_tp = agent_span_traceparent(
                agid,
                state.get("traceparent", ""),
                attributes={
                    "chora.question_type": _OE_QUESTION_TYPE,
                    "chora.guardrail_outcome": guardrail_outcome,
                    "chora.attempt_count": str(max_attempt),
                    "chora.max_retries": str(max_iterations),
                    "chora.quality_warning": str(any_flagged).lower(),
                },
                decision=decision,
                reasoning_summary=notes,
            )
            try:
                await self._agent_decision_emitter.emit(
                    assist_id=submission_id,
                    agid=agid,
                    question_type=_OE_QUESTION_TYPE,
                    tenant_id=tenant_id,
                    gcid=gcid,
                    decision=decision,
                    attempt_count=max_attempt,
                    max_retries=max_iterations,
                    critic_notes=notes,
                    quality_warning=any_flagged,
                    chora_imda_dimension=IMDA_DIMENSION_ACCOUNTABILITY,
                    occurred_at=occurred_at,
                    # agent_tp = this agent's own marker-span traceparent (NOT
                    # the shared run traceparent) → agent-specific deep-link.
                    traceparent=agent_tp,
                    tracestate=state.get("tracestate", ""),
                    crew_name=OE_GRADING_CREW_NAME,
                    crew_id=submission_id,
                    is_resume=False,
                    is_eval_run=False,
                    adapter_version="",
                    guardrail_outcome=guardrail_outcome,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    prompt_conditions=prompt_conditions,
                )
            except Exception:
                # Best-effort. Log and continue so the OTHER agent's decision
                # + the terminal publish still fire.
                logger.exception(
                    "oe_grading_crew_runner.agent_decision_emit_failed",
                    extra={
                        "submission_id": submission_id,
                        "agid": agid,
                        "decision": decision,
                    },
                )


def _coerce_token(v: Any) -> int:
    """Best-effort non-negative int from a trace token field (None/str → 0)."""
    try:
        return max(int(v), 0)
    except (TypeError, ValueError):
        return 0


def _oe_aggregate_tokens(
    pipeline_trace: list[dict[str, Any]],
    hop_names: tuple[str, ...],
) -> tuple[int, int, int]:
    """Sum input/output/cached tokens across the OE pipeline_trace rows whose
    ``name`` is one of ``hop_names`` so each agent carries its OWN token counts
    (evaluator = evaluate + assess_summary; moderator = moderate)."""
    prompt = completion = cached = 0
    for row in pipeline_trace:
        if not isinstance(row, dict):
            continue
        if str(row.get("name") or "") not in hop_names:
            continue
        prompt += _coerce_token(row.get("input_tokens"))
        completion += _coerce_token(row.get("output_tokens"))
        cached += _coerce_token(row.get("cached_tokens"))
    return prompt, completion, cached


def _oe_guardrail_outcome(pipeline_trace: list[dict[str, Any]]) -> str:
    """Derive the Cloud Model Armor verdict surface ∈ {"block", "pass", ""}
    from the OE trace's guardrail_pre / guardrail_post rows (status ALLOW /
    BLOCK). "block" when any guardrail row blocked, else "pass" when at least
    one ran, else "" (guardrails not invoked — e.g. unit-test traces)."""
    saw = False
    for row in pipeline_trace:
        if not isinstance(row, dict):
            continue
        if not str(row.get("name") or "").startswith("guardrail_"):
            continue
        saw = True
        if str(row.get("status") or "").upper() in (
            "BLOCK",
            "BLOCKED",
            "REFUSED",
            "REJECTED",
        ):
            return "block"
    return "pass" if saw else ""


# suffix used in the per-agent state keys -> the agent's registry id (agid),
# which keys EMBEDDED_PROMPT_VERSIONS for the always-stamp fallback below.
_PROMPT_SUFFIX_TO_AGID: dict[str, str] = {
    "evaluator": AGID_OE_EVALUATOR,
    "moderator": AGID_OE_MODERATOR,
}


def _stamp_prompt_override(conditions: dict[str, str], state: OEGradingState | dict[str, Any], suffix: str) -> None:
    """ADR-197 M-B.2 + CHO-2364 - stamp ``prompt_version`` + ``prompt_source``
    onto a prompt condition map IN PLACE for EVERY decision.

    Reads the per-agent keys the runner stored on ``state`` before the graph
    ran (``prompt_version_<suffix>`` / ``prompt_source_<suffix>``). When the
    resolver provided a version (override path) the resolver values are kept
    verbatim. When it did not (no resolver wired OR the embedded default won),
    the stamp falls back to ``EMBEDDED_PROMPT_VERSIONS[agid]`` + source
    ``"embedded"`` so the durable record always carries prompt provenance - a
    decision row is never version-blank.
    """
    version = str(state.get(f"prompt_version_{suffix}") or "")
    source = str(state.get(f"prompt_source_{suffix}") or "")
    if not version:
        version = EMBEDDED_PROMPT_VERSIONS[_PROMPT_SUFFIX_TO_AGID[suffix]]
        source = SOURCE_EMBEDDED
    conditions["prompt_version"] = version
    if source:
        conditions["prompt_source"] = source


def _oe_evaluator_conditions(
    *,
    mode: str,
    attempt_index: int,
    subject: str,
    has_prior_moderator_feedback: bool,
) -> dict[str, str]:
    """ADR-197 M-A.3 — the oe_evaluator prompt-shaping condition discriminants,
    mirroring the Go ``EvaluatorConditions`` extractor key-for-key so the durable
    record + the live agent span agree.

    Keys: ``mode`` (``evaluate`` when OE questions were graded, else
    ``assess_summary`` — the evaluator's two modes); ``attempt_index`` (0-based,
    the Go ``tc.AttemptIndex`` convention); ``subject`` (omitted when blank);
    ``has_prior_moderator_feedback`` (``"true"`` only when a re-grade ran, i.e.
    the moderator rejected a prior grade and fed feedback back). Blanks/false
    are OMITTED — no fabricated discriminants.
    """
    conditions: dict[str, str] = {
        "mode": mode,
        "attempt_index": str(max(attempt_index, 0)),
    }
    subject = (subject or "").strip()
    if subject:
        conditions["subject"] = subject
    if has_prior_moderator_feedback:
        conditions["has_prior_moderator_feedback"] = "true"
    return conditions


def _oe_moderator_conditions(
    *,
    attempt_index: int,
    subject: str,
) -> dict[str, str]:
    """ADR-197 M-A.3 — the oe_moderator prompt-shaping condition discriminants,
    mirroring the Go ``ModeratorConditions`` extractor: ``attempt_index``
    (0-based) + ``subject`` (omitted when blank).
    """
    conditions: dict[str, str] = {"attempt_index": str(max(attempt_index, 0))}
    subject = (subject or "").strip()
    if subject:
        conditions["subject"] = subject
    return conditions


__all__ = ["OEGradingCrewRunner", "parse_requested"]
