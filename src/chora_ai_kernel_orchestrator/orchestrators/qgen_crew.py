"""qgen 2-agent crew StateGraph — AI Assist quality-loop orchestrator.

Per docs/m13/ack-oe-ai-assist-plan-2026-05-17.md the qgen crew is a
DISTINCT 2-agent crew (DIFFERENT engine pool than the legacy 6-agent
`ai_assist_crew` content gate). Two engines:

    qgen_question (LIVE)  ─►  generator (existing 3-agent ADR-153 pipeline)
    qgen_critic   (LIVE)  ─►  critic   (single-agent ADK Go per Step 2)

Both agents run on GKE (ns ai-kernel) and are reached over the Pub/Sub
dispatch lanes (qgen_generate / qgen_critique / qgen_render, ADR-253/254): an
agent call PARKS the run and its completion resumes it. This orchestrator
wires them as a LangGraph state machine with:

    validate_input
       │
       ▼
    guardrail_pre  ─(blocked)─►  publish_refused (reason=GUARDRAIL_PRE)
       │ (allowed)
       ▼
    generate (qgen_question)  ◄──(loop on reject + retries left)──────────┐
       │ ─(dispatch FAILED)─►  publish_refused (reason=VALIDATION)         │
       ▼                                                                  │
    guardrail_post ─(blocked)──►  publish_refused (reason=GUARDRAIL_POST) │
       │                                                                  │
       │ (allowed + evaluator self-rejected: scored=null/below_threshold) │
       ├──────────────────────────────►  quality_gate (skip critique)     │
       │ (allowed + candidate ok)                                         │
       ▼                                                                  │
    critique (qgen_critic)                                                │
       │                                                                  │
       ▼                                                                  │
    quality_gate  ─(accepted)─►   publish_completed                       │
       │                                                                  │
       ├─(rejected + retries left)──────────────────────────────────────►─┘
       │
       └─(rejected + retries exhausted)─►  publish_completed_with_warning

Per user clarification 2026-05-17:
  - critic role = qualitative critique, NOT scoring
  - retries exhausted → completed.v1 with quality_warning + last
    critic_notes (NOT refused.v1; refused.v1 reserved for guardrail blocks)

Per [[langgraph-orchestrator-python]] + [[agentic-resilience-d6]]:
  - Checkpointer supplied by the caller (PostgresSaver in production,
    InMemorySaver in tests) keeps this module pure-orchestration.
  - thread_id per job_id at invocation time (each AI Assist job gets
    its own checkpoint thread for D6 P1 pod-death recovery).
  - Compile once per process; ainvoke is the entry point.

Per [[ai-observability-cloud-trace]]:
  - Every outbound executor call propagates W3C traceparent via the
    dispatch metadata adapter (Cloud Trace continues the
    same trace tree from the chora-creation handler's publish span).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    LANE_ROLE_RENDER,
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
from chora_ai_kernel_orchestrator.domain.qgen_crew import (
    DEFAULT_MAX_RETRIES,
    CandidatePayload,
    CritiqueResult,
    GuardrailResult,
    QGenCrewState,
    assert_valid_refusal_reason,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_grounding import (
    deterministic_draft_id,
    grounding_prompt_block,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_set_plan import (
    AgentContractViolation,
    chunk_type_plan,
    compute_generation_summary,
    dedup_context_block,
    effective_image_caps,
    enforce_image_caps,
    image_caps,
    parse_critique_set_response,
    parse_set_response,
    total_count,
    validate_type_plan,
)
from chora_ai_kernel_orchestrator.orchestrators.testset_composer import (
    fallback_proposal,
    normalise_proposal,
)

logger = logging.getLogger(__name__)

# Canonical agent role identifiers used by the dispatch adapter
# dispatch (one row per role in AGENT_EXECUTOR_TARGETS_JSON). These same
# identifiers index the per-agent guardrail tier mapping in
# config/agent-guardrail-mapping.yaml (vendored; both → balanced per
# ADR-169) — the pre-screen tags agent_id=qgen_question, the post-screen
# tags agent_id=qgen_critic.
ROLE_GENERATE = "qgen_question"
ROLE_CRITIQUE = "qgen_critic"
# ADR-254 D2/D12: the scene image is a dispatch to the qgen_renderer (role
# qgen_render, the third qgen crew member); its agid is the renderer itself.
ROLE_RENDER = LANE_ROLE_RENDER
AGID_RENDER = "qgen_renderer"

# ---- Single-pass SET generation (CHO-1819) --------------------------------
# Default bound on the regenerate-rejected loop. The first generate is
# regen_round=0; rejected candidates are regenerated for at most this many
# extra rounds, then any still-rejected ship WITH a per-candidate
# quality_warning (matching the locked single-path exhausted semantics). The
# runner may override via state["max_regen_rounds"]. Kept small (wall-clock ≈
# 1 generate + 1 batched critique + at most 1 regen + re-critique).
DEFAULT_MAX_REGEN_ROUNDS = 1

# ADR-254 D12: images render ONE AT A TIME from a loop node (one park per
# scene image on the qgen_render lane; Mermaid in-process on Kroki). The former
# in-node fan-out and its QGEN_IMAGE_SCENE_CONCURRENCY cap are gone: the
# renderer's own subscriber (HPA, MaxOutstandingMessages=1) is the platform-wide
# throttle on the shared-quota image model, and a job's images now arrive
# sequentially rather than racing it.

# ADR-251 D1 (CHO-2396) — questions per generate call in the set lane's chunk
# loop. Each chunk is ONE LLM call whose response must fit the token ceiling
# and the agents' 120s write window (CHO-2394); 10 keeps a call at ~20-40s with
# comfortable margin and pipelines critique + images sooner.
ENV_SET_MAX_PER_CALL = "QGEN_SET_MAX_PER_CALL"
DEFAULT_SET_CHUNK_SIZE = 10


def _chunk_size() -> int:
    """Set-lane chunk size; unset/garbage keeps the default, loudly."""
    raw = os.getenv(ENV_SET_MAX_PER_CALL, "").strip()
    if not raw:
        return DEFAULT_SET_CHUNK_SIZE
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "qgen_crew.bad_env",
            extra={"env": ENV_SET_MAX_PER_CALL, "value": raw[:32]},
        )
        return DEFAULT_SET_CHUNK_SIZE


# User-facing copy on a set-mode generator contract violation (malformed
# wrapper / off-plan type / over-count / illegal short set). FAIL-LOUD — the
# generator output is never padded, truncated, or silently coerced.
_SET_VALIDATION_USER_MESSAGE = "The generator returned an invalid question set; no questions were produced."

# User-facing message surfaced on a Cloud Model Armor BLOCK verdict —
# preserved verbatim from the retired _GuardrailAdapter so the FE refusal
# copy is unchanged across the ADR-169 cutover.
_GUARDRAIL_BLOCK_USER_MESSAGE = (
    "Your prompt couldn't be processed — it may conflict with our content guidelines. Please rephrase and try again."
)


# ---------------------------------------------------------------------------
# Adapter protocols — duck-typed for unit testability
# ---------------------------------------------------------------------------


class _ExecutorLike(Protocol):
    """Duck-typed AgentExecutor multi-client.

    Production: the QGenDispatchAdapter over the PubSubAgentExecutor.
    Tests: an in-memory fake that returns canned AgentExecutorResponse
    objects keyed by `agent_role`.
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


class _GuardrailLike(Protocol):
    """Duck-typed tier-mapped Cloud Model Armor guardrail port.

    Production: ``adapter.modelarmor.ModelArmorGuardrailPort`` — resolves the
    per-agent template tier from ``agent-guardrail-mapping.yaml`` and dispatches
    pre/post screening (ADR-169 retired the bare-env ``_GuardrailAdapter``).
    Tests: an in-memory fake honouring the same ``screen(payload)`` signature.

    The port returns a 3-state :class:`ScreenResult` (ALLOW / BLOCK /
    INSPECT_ONLY); the nodes collapse it into the downstream
    :class:`GuardrailResult` contract via :func:`_screen_result_to_guardrail_result`.
    """

    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult: ...


# --- W8 image-render ports (duck-typed; thin adapters injected at composition) ---


class _KrokiLike(Protocol):
    """Duck-typed KrokiClient — POST Mermaid source → diagram bytes."""

    async def render(self, *, source: str, output_format: str = ...) -> bytes: ...


class _GcsImageLike(Protocol):
    """Duck-typed GcsImageUploadAdapter: upload bytes -> (gs_uri, signed_url)
    for the in-process Mermaid path, and sign an EXISTING gs:// object for the
    scene image the qgen_render agent wrote (ADR-254 D12)."""

    async def upload_and_sign(
        self, *, tenant_id: str, job_id: str, data: bytes, content_type: str
    ) -> tuple[str, str]: ...

    async def sign_read_url(self, gs_uri: str) -> str: ...


def _screen_result_to_guardrail_result(result: ScreenResult) -> GuardrailResult:
    """Collapse the Port's 3-state :class:`ScreenResult` into the node's
    :class:`GuardrailResult` contract.

    Parity with the retired ``_GuardrailAdapter``:

    - ``ALLOW`` / ``INSPECT_ONLY`` → ``allowed=True`` (INSPECT_ONLY is
      audit-only; it does not gate the LLM call).
    - ``BLOCK`` → ``allowed=False`` with ``armor:<reason>`` verdict + the
      canonical user-facing refusal message.
    """
    allowed = result.verdict in (Verdict.ALLOW, Verdict.INSPECT_ONLY)
    if allowed:
        return GuardrailResult(allowed=True)
    return GuardrailResult(
        allowed=False,
        armor_verdict=f"armor:{result.reason or 'block_unspecified'}",
        user_facing_message=_GUARDRAIL_BLOCK_USER_MESSAGE,
    )


# ---------------------------------------------------------------------------
# Trace helpers (IMDA D2 transparency evidence per [[adr141-imda-dimension-
# labels-reconciliation]])
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _append_trace(
    state: QGenCrewState,
    name: str,
    *,
    status: str,
    attempt: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    engine_resource: str = "",
    model_id: str = "",
    notes: str = "",
    candidate_json: str | None = None,
    accepted: bool | None = None,
    suggested_revisions: list[str] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> list[dict[str, Any]]:
    """Append one PipelineTraceStep-shaped row (per chora-contracts OpenAPI
    AiAssistJob.pipeline_trace[]). Trace flows into the
    AiAssistCompleted.pipeline_trace_json field on publish.

    W6 per-attempt loop-state capture (CR qgen, 2026-06-01): the optional
    ``candidate_json`` / ``accepted`` / ``suggested_revisions`` keys enrich
    the ``generate`` + ``critique`` rows so a downstream managed-Eval harness
    can reconstruct the actor↔critic trajectory tuple
    ``(attempt_index, candidate_produced, critic_accepted, critique_notes,
    suggested_revisions)`` PER ATTEMPT by zipping the two rows on ``attempt``
    — WITHOUT a new event field or proto change. These keys are additive and
    opaque inside ``pipeline_trace_json``; the FE trace widget ignores unknown
    keys. ``critique_notes`` is already carried in ``notes`` (not duplicated).

    ``model_id`` — the concrete model the agent reported for the hop — rides
    the same additive-key convention so the AgentDecisionLog outbox writer can
    stamp proto field 6 (chora-observability prices the per-hop token counts
    per model).
    """
    existing = list(state.get("pipeline_trace") or [])
    existing.append(
        _trace_row(
            name,
            status=status,
            attempt=attempt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            engine_resource=engine_resource,
            model_id=model_id,
            notes=notes,
            candidate_json=candidate_json,
            accepted=accepted,
            suggested_revisions=suggested_revisions,
            started_at=started_at,
            completed_at=completed_at,
        )
    )
    return existing


def _trace_row(
    name: str,
    *,
    status: str,
    attempt: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    engine_resource: str = "",
    model_id: str = "",
    notes: str = "",
    candidate_json: str | None = None,
    accepted: bool | None = None,
    suggested_revisions: list[str] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Build ONE PipelineTraceStep-shaped row (the body of :func:`_append_trace`,
    extracted so the set-native nodes that emit MANY rows per node return
    (e.g. ``critique_set`` emits one tokened ``critique`` row per call plus a
    ``critique_verdict`` row per candidate) can build them without N reads of
    ``state["pipeline_trace"]``).

    C3 real-duration capture: the LLM-issuing nodes (generate / critique) pass
    started_at/completed_at captured around executor.execute(...). Non-LLM
    (instantaneous) nodes omit them and fall back to _now_iso() twice
    (zero-duration is correct for them). started_at <= completed_at.
    """
    entry: dict[str, Any] = {
        "name": name,
        "status": status,
        "started_at": started_at if started_at is not None else _now_iso(),
        "completed_at": completed_at if completed_at is not None else _now_iso(),
    }
    if attempt is not None:
        entry["attempt"] = attempt
    if input_tokens is not None:
        entry["input_tokens"] = input_tokens
    if output_tokens is not None:
        entry["output_tokens"] = output_tokens
    if engine_resource:
        entry["engine_resource"] = engine_resource
    if model_id:
        entry["model_id"] = model_id
    if notes:
        entry["notes"] = notes
    if candidate_json is not None:
        entry["candidate_json"] = candidate_json
    if accepted is not None:
        entry["accepted"] = accepted
    if suggested_revisions is not None:
        entry["suggested_revisions"] = suggested_revisions
    return entry


def _append_error(state: QGenCrewState, msg: str) -> list[str]:
    existing = list(state.get("errors") or [])
    existing.append(msg)
    return existing


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def validate_input_node(state: QGenCrewState) -> dict[str, Any]:
    """Fail-loud validation of caller-supplied inputs.

    Per [[feedback-no-stubs-real-wiring]] we route to publish_refused
    (reason=VALIDATION) on bad input rather than silently coercing.
    """
    missing: list[str] = []
    for k in ("job_id", "tenant_id", "gcid", "prompt", "question_type"):
        if not str(state.get(k) or "").strip():
            missing.append(k)

    qt = state.get("question_type", "")
    # Set-mode (mixed-type / batch, CHO-1819) carries question_type="mixed" —
    # the actual per-type quotas live in state["type_plan"] and are validated by
    # parse_set_response in generate_set_node. The single-type lanes stay
    # mcq | oe only, so this widening is additive + set_mode-gated.
    allowed_types = {"mcq", "oe", "mixed"} if state.get("set_mode") else {"mcq", "oe"}
    if qt and qt not in allowed_types:
        # Reserved sentinels in QuestionType enum aren't in Phyllis scope.
        return {
            "refusal_reason": "VALIDATION",
            "refusal_user_facing_message": (f"question_type {qt!r} is not supported (mcq | oe only)"),
            "pipeline_trace": _append_trace(
                state,
                "validate_input",
                status="REJECTED",
                notes=f"unsupported question_type: {qt}",
            ),
        }

    if missing:
        return {
            "refusal_reason": "VALIDATION",
            "refusal_user_facing_message": (f"missing required field(s): {', '.join(missing)}"),
            "pipeline_trace": _append_trace(
                state,
                "validate_input",
                status="REJECTED",
                notes=f"missing: {', '.join(missing)}",
            ),
        }

    # Initialise loop state on first entry.
    return {
        "attempt_count": 0,  # bumped to 1 on first generate node entry
        "max_retries": state.get("max_retries") or DEFAULT_MAX_RETRIES,
        "pipeline_trace": _append_trace(state, "validate_input", status="ACCEPTED"),
    }


async def guardrail_pre_node(state: QGenCrewState, *, guardrail: _GuardrailLike) -> dict[str, Any]:
    """Cloud Model Armor pre-screen on the author's prompt + metadata.

    Blocks PII-laden / jailbreak / out-of-scope inputs BEFORE the
    generator burns mana. Per [[cloud-model-armor-guardrails]] +
    ADR-169 this routes through the tier-mapped
    :class:`ModelArmorGuardrailPort` — the template tier is resolved from
    ``agent-guardrail-mapping.yaml`` by ``agent_id=qgen_question`` (balanced),
    NOT a bare env template name.
    """
    try:
        result = await guardrail.screen(
            GuardrailScreenInput(
                tenant_id=state.get("tenant_id", ""),
                gcid=_gcid(state),
                agent_id=ROLE_GENERATE,
                content=state.get("prompt", ""),
                direction="input",
            )
        )
        verdict = _screen_result_to_guardrail_result(result)
    except Exception as exc:  # transport error — fail loud
        msg = f"guardrail_pre transport error: {exc.__class__.__name__}: {exc}"
        logger.exception(
            "qgen_crew.guardrail_pre.transport_error",
            extra={
                "job_id": state.get("job_id", ""),
                "tenant_id": state.get("tenant_id", ""),
                "agent_id": ROLE_GENERATE,
            },
        )
        return {
            "errors": _append_error(state, msg),
            "refusal_reason": "GUARDRAIL_PRE",
            "refusal_user_facing_message": ("Pre-screening failed — please try again in a moment."),
            "pipeline_trace": _append_trace(
                state,
                "guardrail_pre",
                status="FAILED",
                notes=msg,
            ),
        }
    return {
        "guardrail_pre_result": verdict,
        "refusal_reason": "" if verdict.allowed else "GUARDRAIL_PRE",
        "refusal_armor_verdict": verdict.armor_verdict if not verdict.allowed else "",
        "refusal_user_facing_message": (verdict.user_facing_message if not verdict.allowed else ""),
        "pipeline_trace": _append_trace(
            state,
            "guardrail_pre",
            status="ACCEPTED" if verdict.allowed else "REFUSED",
            notes=verdict.armor_verdict,
        ),
    }


async def generate_node(state: QGenCrewState, *, executor: _ExecutorLike) -> dict[str, Any]:
    """Dispatch to qgen_question Vertex AI Agent Engine (ROLE_GENERATE)."""
    attempt = (state.get("attempt_count") or 0) + 1
    gen_input: dict[str, Any] = {
        "prompt": state.get("prompt", ""),
        "question_type": state.get("question_type", ""),
        "metadata": state.get("metadata") or {},
        "attempt_index": attempt - 1,
        "prior_critic_notes": state.get("critic_notes") or "",
        # W7 follow-up (2026-06-01) — forward the author gcid so the executor
        # (_build_session_state) + gateway TenantPropagationPlugin attribute
        # per-USER, not the synthetic ``qgen-anon:{tenant_id}`` fallback W7
        # observed. Empty gcid still degrades to the executor's anon fallback.
        "gcid": _gcid(state),
        # W8 (CR qgen 2026-06-01) — forward the author's per-image opt-in so
        # the generation agent knows whether to emit an image_spec for the
        # stem and/or the model answer. Author-SELECTED (not model-decided);
        # default False ⇒ no image (the live MCQ loop is unchanged). The
        # candidate.image_specs[] the agent returns is rendered downstream
        # by render_image.
        "image_for_stem": _bool(state.get("image_for_stem")),
        "image_for_answer": _bool(state.get("image_for_answer")),
        # CR qgen Phase B2 — propagate W3C trace context into the agent
        # call so qgen_question continues this workflow trace (one
        # distributed trace orchestrator → agent → gateway). The executor
        # stamps these into ADK session state; if absent here it falls
        # back to reading the current span itself.
        **_current_w3c_trace_context(),
    }
    # CHO-1658 — model_answer_fill: forward the compose intent + the author's
    # existing question so the executor (reasoning_engine_executor) surfaces the
    # fill keys author_stem / author_options / author_rubric / model_answer. Added
    # ONLY when set so the live new_question generate payload is byte-for-byte
    # unchanged (the executor defaults intent to new_question + reads no
    # existing_question when absent).
    intent = str(state.get("intent") or "")
    if intent:
        gen_input["intent"] = intent
    existing_question = state.get("existing_question")
    if existing_question:
        gen_input["existing_question"] = existing_question
    # EPIC-1a grounding passthrough (2026-06-09) — when batch source material was
    # uploaded, forward the gs:// blob + MIME + mode to the qgen_question agent so
    # it inlines the bytes as a Gemini multimodal part (contents_json inlineData).
    # Added ONLY when material/edges are present so the live single-candidate
    # generate payload is byte-for-byte unchanged.
    source_blob_uri = str(state.get("source_blob_uri") or "")
    if source_blob_uri:
        gen_input["source_blob_uri"] = source_blob_uri
        gen_input["source_mime_type"] = str(state.get("source_mime_type") or "")
        gen_input["grounding_mode"] = str(state.get("grounding_mode") or "")
    target_growth_edges = state.get("target_growth_edges") or []
    if target_growth_edges:
        gen_input["target_growth_edges"] = list(target_growth_edges)
    # Lane 1c multi-file + rubric grounding (CHO-1703) — grounded BATCH jobs
    # carry the EFFECTIVE role-tagged file list (resolved by QGenBatchRunner.
    # _build_state; pre-1c f17/18 events are synthesized into a single-entry
    # list). Forward it to the agent (the executor surfaces it as
    # source_files_json on session state — the forward seam for the
    # multi-FileData groundingplugin) and append the grounding + citations
    # prompt block so EVERY grounded-batch candidate model-reports
    # citations {source_file, page, excerpt} (D9) and aligns to the rubric
    # mark scheme when one rides the job (D5/D6). Only set on grounded
    # batch state — single + ungrounded payloads stay byte-for-byte
    # unchanged.
    source_files = state.get("source_files") or []
    if source_files:
        gen_input["source_files"] = list(source_files)
        gen_input["prompt"] = str(gen_input.get("prompt") or "") + grounding_prompt_block(
            files=list(source_files),
            grounding_mode=str(state.get("grounding_mode") or ""),
            question_type=str(state.get("question_type") or ""),
        )
    # ADR-197 M-B.2 — thread the qgen_question prompt override (when resolved).
    _thread_prompt_overrides(gen_input, state, "question")
    payload = json.dumps(gen_input)
    started_at = _now_iso()
    try:
        resp = await executor.execute(
            execution_id=f"{state.get('job_id', '')}:generate:{attempt}",
            tenant_id=state.get("tenant_id", ""),
            agid=ROLE_GENERATE,
            agent_role=ROLE_GENERATE,
            input_payload=payload,
            workflow_id=str(state.get("job_id") or ""),
            prompt_template_id="qgen_crew::generate",
        )
    except Exception as exc:
        # ADR-253: a park is control flow, not a failure. Without this the
        # Pub/Sub dispatch above is swallowed and the author is shown a
        # GUARDRAIL_PRE refusal on a run that merely parked.
        reraise_if_dispatch_park(exc)
        msg = f"generate executor error: {exc.__class__.__name__}: {exc}"
        # The refusal reaches the AUTHOR through state, but a systemic agent
        # failure looks like nothing at all on the operator side unless it is
        # logged here: every run just refuses, one at a time, and the lane
        # reports healthy throughout.
        logger.warning(
            "qgen_crew.generate.executor_failed",
            extra={"job_id": str(state.get("job_id") or ""), "err": msg},
            exc_info=True,
        )
        # ADR-254: a FAILED dispatch (the agent reported it could not do the
        # work, or the reaper synthesised the failure) ends the job as a
        # VALIDATION-class refusal via route_after_generate, the same class the
        # set lane uses. It is NOT retried: the agent already retried its
        # transient model errors, and a reaped lane would only park the author
        # for another deadline per attempt. (The wire vocabulary is fixed to
        # GUARDRAIL_PRE | GUARDRAIL_POST | VALIDATION by chora-creation.)
        return {
            "attempt_count": attempt,
            "errors": _append_error(state, msg),
            "refusal_reason": "VALIDATION",
            "pipeline_trace": _append_trace(
                state,
                "generate",
                status="FAILED",
                attempt=attempt,
                notes=msg,
                started_at=started_at,
                completed_at=_now_iso(),
            ),
        }
    completed_at = _now_iso()
    # C3 — real per-hop token split off the executor response. input_tokens
    # falls back to tokens_consumed_total for the M11 baseline shape (executors
    # that only emit the total); output_tokens is now the REAL value (was
    # hardcoded 0). Keeps the live MCQ loop's existing trace shape green.
    gen_input_tokens = _input_tokens_of(resp)
    gen_output_tokens = int(getattr(resp, "output_tokens", 0) or 0)

    raw = _safe_json_loads(resp.output_payload or "", {})

    # Detect the evaluator's self-rejection signal — qgen_question's
    # terminal evaluator emits {"scored": null, "reason": "below_threshold"}
    # when the composite quality score < 0.6 (per StepEvaluation3 in
    # agents/qgen_adk_go/internal/agent/composer_question.go). This is a
    # SUCCESS wire path; route to retry without invoking the critic.
    #
    # Surfaced by the MCQ smoke (deleted with the HTTP executor, 2026-08-23)
    # (commit 03c2ae2b) — see the pytest.skip block at L210-217 for
    # the live wire-shape evidence.
    if _is_evaluator_below_threshold(raw):
        reason = str(raw.get("reason") or "below_threshold")
        details = str(raw.get("details") or "")
        notes = f"evaluator below_threshold: {reason}"
        if details:
            notes = f"{notes}; {details}"
        below_threshold_payload = json.dumps(raw)
        return {
            "attempt_count": attempt,
            "evaluator_below_threshold": True,
            "evaluator_reason": reason,
            # Persist a placeholder candidate so downstream nodes that
            # peek at current_candidate (e.g., guardrail_post) don't
            # explode. payload_json carries the verbatim wire shape so
            # IMDA D2 audit can replay the evaluator's verdict.
            "current_candidate": CandidatePayload(
                stem="",
                question_type=state.get("question_type", ""),
                payload_json=below_threshold_payload,
            ),
            "critic_notes": "",
            "pipeline_trace": _append_trace(
                state,
                "generate",
                status="REJECTED",
                attempt=attempt,
                input_tokens=gen_input_tokens,
                output_tokens=gen_output_tokens,
                model_id=resp.model_id,
                notes=notes,
                started_at=started_at,
                completed_at=completed_at,
                # W6 loop-state capture: the below-threshold attempt still
                # "produced" a candidate (the evaluator's verbatim verdict
                # payload); stamp it so the harness sees candidate_N on every
                # generate row, including self-rejected attempts.
                candidate_json=below_threshold_payload,
            ),
        }

    candidate = CandidatePayload(
        stem=str(raw.get("stem", "")),
        question_type=state.get("question_type", ""),
        payload_json=json.dumps(raw),
    )
    return {
        "attempt_count": attempt,
        # Clear any prior below-threshold flag — a real candidate produced.
        "evaluator_below_threshold": False,
        "evaluator_reason": "",
        "current_candidate": candidate,
        # Clear prior critic_notes — they've been consumed by this attempt.
        "critic_notes": "",
        "pipeline_trace": _append_trace(
            state,
            "generate",
            status="COMPLETED",
            attempt=attempt,
            input_tokens=gen_input_tokens,
            output_tokens=gen_output_tokens,
            model_id=resp.model_id,
            notes="candidate produced",
            started_at=started_at,
            completed_at=completed_at,
            # W6 loop-state capture: stamp the exact serialized candidate
            # this attempt produced (same string carried downstream) so the
            # managed-Eval harness can read candidate_N off the `generate`
            # row for attempt N.
            candidate_json=candidate.payload_json,
        ),
    }


async def guardrail_post_node(state: QGenCrewState, *, guardrail: _GuardrailLike) -> dict[str, Any]:
    """Cloud Model Armor post-screen on the candidate payload.

    Catches generated content that slipped past the pre-screen — e.g.,
    distractors that fabricate PII, or model_answer that leaks tenant-
    private info via hallucination. Per ADR-169 this routes through the
    tier-mapped :class:`ModelArmorGuardrailPort` with
    ``agent_id=qgen_critic`` (balanced) + ``direction="output"``.
    """
    candidate = state.get("current_candidate")
    text = candidate.payload_json if candidate else ""
    try:
        result = await guardrail.screen(
            GuardrailScreenInput(
                tenant_id=state.get("tenant_id", ""),
                gcid=_gcid(state),
                agent_id=ROLE_CRITIQUE,
                content=text,
                direction="output",
            )
        )
        verdict = _screen_result_to_guardrail_result(result)
    except Exception as exc:
        msg = f"guardrail_post transport error: {exc.__class__.__name__}: {exc}"
        logger.exception(
            "qgen_crew.guardrail_post.transport_error",
            extra={
                "job_id": state.get("job_id", ""),
                "tenant_id": state.get("tenant_id", ""),
                "agent_id": ROLE_CRITIQUE,
                "attempt": state.get("attempt_count"),
            },
        )
        return {
            "errors": _append_error(state, msg),
            "refusal_reason": "GUARDRAIL_POST",
            "refusal_user_facing_message": ("Post-screening failed — please try again in a moment."),
            "pipeline_trace": _append_trace(
                state,
                "guardrail_post",
                status="FAILED",
                attempt=state.get("attempt_count"),
                notes=msg,
            ),
        }
    return {
        "guardrail_post_result": verdict,
        "refusal_reason": "" if verdict.allowed else "GUARDRAIL_POST",
        "refusal_armor_verdict": verdict.armor_verdict if not verdict.allowed else "",
        "refusal_user_facing_message": (verdict.user_facing_message if not verdict.allowed else ""),
        "last_candidate_on_refusal": (None if verdict.allowed else state.get("current_candidate")),
        "pipeline_trace": _append_trace(
            state,
            "guardrail_post",
            status="ACCEPTED" if verdict.allowed else "REFUSED",
            attempt=state.get("attempt_count"),
            notes=verdict.armor_verdict,
        ),
    }


async def critique_node(state: QGenCrewState, *, executor: _ExecutorLike) -> dict[str, Any]:
    """Dispatch to qgen_critic Vertex AI Agent Engine (ROLE_CRITIQUE).

    Returns CritiqueResult with accepted (bool) + critique_notes (str)
    + suggested_revisions (list[str]). Per user clarification 2026-05-17
    the critic is QUALITATIVE — no numeric scoring.
    """
    candidate = state.get("current_candidate")
    if candidate is None:
        msg = "critique called with no current_candidate — pipeline bug"
        return {
            "errors": _append_error(state, msg),
            "critic_result": CritiqueResult(
                accepted=False,
                critique_notes=msg,
                suggested_revisions=[],
            ),
            "pipeline_trace": _append_trace(
                state,
                "critique",
                status="FAILED",
                attempt=state.get("attempt_count"),
                notes=msg,
            ),
        }
    attempt = state.get("attempt_count") or 1
    # CR qgen Phase B2 — merge W3C trace context into the candidate payload so
    # qgen_critic continues this workflow trace. Decode → add traceparent /
    # tracestate → re-encode; the executor lifts these into ADK session state.
    # If the candidate JSON isn't a dict (defensive), fall back to the raw
    # string so we never drop the candidate.
    #
    # CHANGE 2 (debt-free critic context, 2026-06-01) — qgen_critic's
    # ``_build_session_state`` ALREADY reads ``author_prompt``,
    # ``attempt_index`` and ``prior_critic_notes`` off the input object, but
    # critique_node never put them there (the critic got empty defaults). Merge
    # them with the EXACT key names the executor expects so the critic critiques
    # WITH real context. Context-enrichment only: if the candidate payload won't
    # decode to a dict we keep the raw string (don't crash).
    trace_ctx = _current_w3c_trace_context()
    decoded = _safe_json_loads(candidate.payload_json, None)
    if isinstance(decoded, dict):
        decoded = {
            **decoded,
            "author_prompt": state.get("prompt", ""),
            # 0-based attempt index (attempt is 1-based; the generator uses the
            # same 0-based convention for its own attempt_index).
            "attempt_index": attempt - 1,
            "prior_critic_notes": state.get("critic_notes", "") or "",
            # Authoring-metadata wire-through (2026-06-03) — forward the author's
            # Subject / Cognitive Level / Difficulty so the executor stamps
            # subject_hint / cognitive_level_hint / difficulty_hint, which the
            # critic reads (critic.go:477-479 → metadataHints). generate_node
            # already forwards metadata; the critic side was the missing half.
            "metadata": state.get("metadata") or {},
            # W7 follow-up — forward author gcid for per-user attribution
            # (executor reads input_obj.gcid; absent today → synthetic anon).
            "gcid": _gcid(state),
            **trace_ctx,
        }
        # ADR-197 M-B.2 — thread the qgen_critic prompt override (when resolved).
        _thread_prompt_overrides(decoded, state, "critic")
        payload = json.dumps(decoded)
    else:
        payload = candidate.payload_json
    started_at = _now_iso()
    try:
        resp = await executor.execute(
            execution_id=f"{state.get('job_id', '')}:critique:{attempt}",
            tenant_id=state.get("tenant_id", ""),
            agid=ROLE_CRITIQUE,
            agent_role=ROLE_CRITIQUE,
            input_payload=payload,
            workflow_id=str(state.get("job_id") or ""),
            prompt_template_id="qgen_crew::critique",
        )
    except Exception as exc:
        # ADR-253: a park is control flow, not a failure. Without this the
        # Pub/Sub dispatch above is swallowed and the author is shown a
        # critique refusal on a run that merely parked.
        reraise_if_dispatch_park(exc)
        msg = f"critique executor error: {exc.__class__.__name__}: {exc}"
        logger.warning(
            "qgen_crew.critique.executor_failed",
            extra={"job_id": str(state.get("job_id") or ""), "err": msg},
            exc_info=True,
        )
        return {
            "errors": _append_error(state, msg),
            "critic_result": CritiqueResult(
                accepted=False,
                critique_notes=msg,
                suggested_revisions=[],
            ),
            "pipeline_trace": _append_trace(
                state,
                "critique",
                status="FAILED",
                attempt=attempt,
                notes=msg,
                started_at=started_at,
                completed_at=_now_iso(),
            ),
        }
    completed_at = _now_iso()
    raw = _safe_json_loads(resp.output_payload or "", {})
    result = CritiqueResult(
        accepted=bool(raw.get("accepted", False)),
        critique_notes=str(raw.get("critique_notes", "")),
        suggested_revisions=list(raw.get("suggested_revisions", []) or []),
    )
    return {
        "critic_result": result,
        "critic_notes": result.critique_notes,
        "pipeline_trace": _append_trace(
            state,
            "critique",
            status="ACCEPTED" if result.accepted else "REJECTED",
            attempt=attempt,
            # C3 — real per-hop split (input falls back to total for the M11
            # baseline; output is now the REAL value, was implicitly absent).
            input_tokens=_input_tokens_of(resp),
            output_tokens=int(getattr(resp, "output_tokens", 0) or 0),
            model_id=resp.model_id,
            notes=result.critique_notes[:200] if result.critique_notes else "",
            started_at=started_at,
            completed_at=completed_at,
            # W6 loop-state capture: the critic's verdict on candidate_N
            # (critique_notes is already in `notes`). The harness zips this
            # row to the same-attempt `generate` row to recover the full
            # (candidate, accepted, notes, suggested_revisions) tuple.
            accepted=result.accepted,
            suggested_revisions=result.suggested_revisions,
        ),
    }


async def quality_gate_node(state: QGenCrewState) -> dict[str, Any]:
    """Pure state-transition node. The conditional routing happens in the
    edge function `route_after_quality_gate` (LangGraph requires nodes to
    return state deltas; the routing function returns the next node name).

    We emit a trace row here so the per-attempt decision shows up in
    pipeline_trace_json for IMDA D2 transparency.

    Two rejection sources funnel through this gate:
      1. ``critic_result.accepted == False`` — qgen_critic verdict
      2. ``evaluator_below_threshold == True`` — qgen_question evaluator's
         self-rejection (composite < 0.6); the critic was SKIPPED for
         this attempt so critic_result is None / stale.
    Both retry until ``max_retries+1`` attempts are exhausted, then emit
    completed.v1 with ``quality_warning=True``.
    """
    critic = state.get("critic_result")
    attempt = state.get("attempt_count") or 0
    max_retries = state.get("max_retries") or DEFAULT_MAX_RETRIES
    eval_rejected = bool(state.get("evaluator_below_threshold") or False)
    eval_reason = str(state.get("evaluator_reason") or "")

    if eval_rejected and attempt < max_retries + 1:
        decision = "RETRY"
        notes = f"evaluator below_threshold (reason={eval_reason!r}) on attempt {attempt}; loop"
    elif eval_rejected:
        decision = "QUALITY_WARNING"
        notes = (
            f"evaluator below_threshold (reason={eval_reason!r}) on attempt "
            f"{attempt} (exhausted max_retries={max_retries}); emit "
            f"completed.v1 with quality_warning"
        )
    elif critic is None:
        decision = "FAILED"
        notes = "no critic_result"
    elif critic.accepted:
        decision = "ACCEPTED"
        # Prefer the critic's OWN accept rationale (IMDA D2 — surfaced to O+
        # Decision-Traces + Cloud Trace via reasoning_summary). Fall back to a
        # descriptive note when the critic returned none, so the audit record is
        # never the bare generic string.
        critic_rationale = (critic.critique_notes or "").strip()
        notes = (
            critic_rationale
            if critic_rationale
            else f"critic accepted on attempt {attempt} (no revisions requested); passed quality gate"
        )
    elif attempt < max_retries + 1:
        # attempt is 1-based; max_retries=3 means up to 4 attempts.
        decision = "RETRY"
        notes = f"critic rejected on attempt {attempt}; loop"
    else:
        decision = "QUALITY_WARNING"
        notes = (
            f"critic rejected on attempt {attempt} (exhausted max_retries="
            f"{max_retries}); emit completed.v1 with quality_warning"
        )

    return {
        "pipeline_trace": _append_trace(
            state,
            "quality_gate",
            status=decision,
            attempt=attempt,
            notes=notes,
        ),
    }


async def render_image_node(
    state: QGenCrewState,
    *,
    kroki: _KrokiLike | None,
    gcs: _GcsImageLike | None,
) -> dict[str, Any]:
    """W8 / ADR-254 D12: PLAN the author-opted image(s) on the ACCEPTED
    candidate. The renders themselves run one per superstep in
    :func:`render_next_node` (one park per scene image on the qgen_render lane,
    Mermaid in-process on Kroki) and :func:`render_finalize_node` stamps the
    results; this node only decides WHAT renders.

    Inserted on the ACCEPTED path: ``quality_gate -> render_image ->
    [render_next ...] -> render_finalize -> publish_completed``.

    DORMANT by default: when the accepted candidate has NO ``image_specs`` this
    is a PURE pass-through (an EMPTY delta, no trace row), so the live MCQ
    loop's pipeline_trace is byte-for-byte unchanged.

    ``image_specs`` is a LIST of 0-2 entries, each ``{"mode": "mermaid"|"scene",
    "source": "...", "placement": "stem"|"answer"}``:

      - ``mode=="mermaid"`` -> Kroki -> the kennel's own upload + V4 URL
      - ``mode=="scene"``   -> ONE qgen_render dispatch -> the agent's gs://
        object, signed by the kennel
      - ``placement=="stem"``   -> ``candidate.image_url``
      - ``placement=="answer"`` -> ``candidate.answer_image_url``
      - the published candidate has ``image_specs`` REMOVED

    Fail-LOUD only when an image_spec IS present but a required client is None
    (Kroki / GCS: a mis-config, not a transient error). An EXHAUSTED candidate
    (shipping with quality_warning) skips rendering entirely (CHO-2399,
    mirroring the set lane's CHO-2395 semantics): published, never rendered.
    """
    candidate = state.get("current_candidate")
    if candidate is None:
        return {}

    decoded = _safe_json_loads(candidate.payload_json, None)
    specs = decoded.get("image_specs") if isinstance(decoded, dict) else None
    if not isinstance(specs, list) or not specs:
        # No image work: pure pass-through (no trace row, candidate unchanged).
        return {}

    # CHO-2399 (the CHO-2395 sibling-screen fix): a candidate that ships under
    # the exhausted semantics (critic rejected with the retry budget spent, or
    # the evaluator self-rejected on the last attempt; reaching this node with
    # either signal means terminal, because the gate only routes here then) is
    # PUBLISHED with quality_warning but never rendered: a failed-QC question
    # must not spend gemini-3-pro-image shared quota. The skip precedes the
    # wiring check so a rejected-only spec never makes unconfigured image
    # clients a mis-config, and it is a deliberate THIRD tally, never a render
    # failure.
    critic = state.get("critic_result")
    exhausted = (critic is not None and not critic.accepted) or bool(state.get("evaluator_below_threshold") or False)
    if exhausted:
        skipped = sum(1 for s in specs if isinstance(s, dict))
        cleaned_payload = dict(decoded)
        cleaned_payload.pop("image_specs", None)
        return {
            "current_candidate": CandidatePayload(
                stem=candidate.stem,
                question_type=candidate.question_type,
                payload_json=json.dumps(cleaned_payload),
                critic_notes=candidate.critic_notes,
            ),
            "pipeline_trace": _append_trace(
                state,
                "render_image",
                status="COMPLETED",
                attempt=state.get("attempt_count"),
                notes=(f"rendered 0/0 image(s); {skipped} skipped (quality_warning)"),
            ),
        }

    _require_image_clients("render_image", kroki=kroki, gcs=gcs)

    attempt = int(state.get("attempt_count") or 0)
    job_id = str(state.get("job_id") or "")
    pending: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            continue
        placement, field_name, gcs_field = _spec_fields(spec)
        pending.append(
            {
                "idx": 0,
                "spec_index": spec_index,
                "placement": placement,
                "field_name": field_name,
                "gcs_field": gcs_field,
                "mode": str(spec.get("mode") or "").strip().lower(),
                "source": str(spec.get("source") or ""),
                "execution_id": f"{job_id}:render:a{attempt}:s{spec_index}:{placement}",
            }
        )
    return {
        "pending_renders": pending,
        "render_results": [],
        "render_plan": {
            "lane": "single",
            "emitted": len(specs),
            "dropped": 0,
            "skipped": 0,
            "cap_notes": [],
        },
    }


def _require_image_clients(node: str, *, kroki: _KrokiLike | None, gcs: _GcsImageLike | None) -> None:
    """An image is requested but the pipeline isn't wired: a mis-config, not a
    transient error. Fail loud per [[feedback-no-stubs-real-wiring]]."""
    if kroki is None or gcs is None:
        raise RuntimeError(
            f"{node}: image_specs present but an image client is unconfigured "
            "(kroki / gcs). Set KROKI_ENDPOINT + QGEN_IMAGE_GCS_BUCKET per "
            "[[secrets-and-env]]."
        )


def _spec_fields(spec: dict[str, Any]) -> tuple[str, str, str]:
    """(placement, candidate url field, candidate gs:// field) for one spec."""
    placement = str(spec.get("placement") or "stem").strip().lower()
    field_name = "answer_image_url" if placement == "answer" else "image_url"
    gcs_field = "answer_image_gcs_uri" if placement == "answer" else "image_gcs_uri"
    return placement, field_name, gcs_field


async def render_spec(
    *,
    spec: dict[str, Any],
    executor: _ExecutorLike,
    kroki: _KrokiLike,
    gcs: _GcsImageLike,
    tenant_id: str,
    job_id: str,
    gcid: str,
    trace_ctx: dict[str, str],
    execution_id: str,
    workflow_id: str,
    chunk_id: str = "",
    source_image_uri: str = "",
    source_image_mime: str = "",
) -> tuple[str, str]:
    """Render ONE image_spec and return ``(gs_uri, signed_url)``: the canonical
    durable object path (persisted for image-to-image regen, ADR-210) + its
    signed read URL (display).

    ``mode=="mermaid"``: deterministic, in-process: Kroki -> the kennel's own
    upload + V4 signing (ADR-254 D12 keeps Kroki and the signing in the kennel).

    ``mode=="scene"``: ONE ``qgen_render`` dispatch (a park on the Pub/Sub
    transport) carrying ``render_prompt`` / ``mode`` / ``job_id`` / ``chunk_id``
    and, for an EDIT (ADR-210 image-to-image), ``source_image_uri`` +
    ``source_image_mime`` read by the agent by reference; the completion's
    ``image_uri`` (the object the agent wrote) is signed here. The agent's
    model call is already screened at the gateway (ADR-152).

    Raises on any failure (unknown mode / empty source / agent FAILED / no
    gs:// in the answer / upload or signing error) so the caller can fail-soft
    per spec. A LangGraph park is NOT a failure: the caller's ``except`` must
    call ``reraise_if_dispatch_park`` first.
    """
    mode = str(spec.get("mode") or "").strip().lower()
    source = str(spec.get("source") or "")
    if not source:
        raise ValueError("image_spec has empty source")

    if mode == "mermaid":
        data = await kroki.render(source=source, output_format="png")
        if not data:
            raise RuntimeError("image render produced no bytes (mode=mermaid)")
        return await gcs.upload_and_sign(
            tenant_id=tenant_id,
            job_id=job_id,
            data=data,
            content_type="image/png",
        )
    if mode == "scene":
        payload: dict[str, Any] = {
            "render_prompt": source,
            "mode": "scene",
            "job_id": job_id,
            "gcid": gcid,
            **trace_ctx,
        }
        if chunk_id:
            payload["chunk_id"] = chunk_id
        if source_image_uri:
            payload["source_image_uri"] = source_image_uri
            if source_image_mime:
                payload["source_image_mime"] = source_image_mime
        resp = await executor.execute(
            execution_id=execution_id,
            tenant_id=tenant_id,
            agid=AGID_RENDER,
            agent_role=ROLE_RENDER,
            input_payload=json.dumps(payload),
            workflow_id=workflow_id,
            prompt_template_id="qgen_crew::render",
        )
        raw = _safe_json_loads(resp.output_payload or "", {})
        image_uri = str(raw.get("image_uri") or "") if isinstance(raw, dict) else ""
        if not image_uri.startswith("gs://"):
            raise RuntimeError(
                f"qgen_render returned no gs:// image_uri (got {str(resp.output_payload or '')[:160]!r})"
            )
        url = await gcs.sign_read_url(image_uri)
        return image_uri, url
    raise ValueError(f"unknown image_spec mode: {mode!r} (mermaid | scene)")


async def render_next_node(
    state: QGenCrewState,
    *,
    executor: _ExecutorLike,
    kroki: _KrokiLike | None,
    gcs: _GcsImageLike | None,
) -> dict[str, Any]:
    """Render the HEAD of ``pending_renders`` (one image per superstep). A
    scene image parks here on its qgen_render dispatch and resumes on the
    completion; the loop edge re-enters this node until the queue is empty.
    Everything before the dispatch is a pure read of the queue head, so the
    node's re-execution on resume rebuilds the same request (same execution
    id, same idempotency key). Fail-SOFT per spec (the question still ships,
    the failure is recorded for finalize to make LOUD), never on a park."""
    pending = list(state.get("pending_renders") or [])
    if not pending:
        return {}
    item = dict(pending[0])
    results = list(state.get("render_results") or [])
    tenant_id = str(state.get("tenant_id") or "")
    job_id = str(state.get("job_id") or "")
    chunk_id = f"c{int(state.get('chunk_index') or 0)}" if state.get("set_mode") else ""
    try:
        if kroki is None or gcs is None:
            _require_image_clients("render_next", kroki=kroki, gcs=gcs)
        gs_uri, url = await render_spec(
            spec={"mode": item.get("mode"), "source": item.get("source")},
            executor=executor,
            kroki=kroki,  # type: ignore[arg-type]
            gcs=gcs,  # type: ignore[arg-type]
            tenant_id=tenant_id,
            job_id=job_id,
            gcid=_gcid(state),
            trace_ctx=_current_w3c_trace_context(),
            execution_id=str(item.get("execution_id") or ""),
            workflow_id=job_id,
            chunk_id=chunk_id,
        )
        item.update({"url": url, "gs_uri": gs_uri, "error": ""})
    except Exception as exc:  # fail-soft per spec; a park is control flow
        reraise_if_dispatch_park(exc)
        msg = (
            f"render_image failed for candidate {item.get('idx')} "
            f"placement={item.get('placement')} mode={item.get('mode')!r}: "
            f"{exc.__class__.__name__}: {exc}"
        )
        logger.exception(
            "qgen_crew.render_image.failed",
            extra={
                "job_id": job_id,
                "tenant_id": tenant_id,
                "candidate_index": item.get("idx"),
                "placement": item.get("placement"),
                "mode": item.get("mode"),
            },
        )
        item.update({"url": None, "gs_uri": None, "error": msg})
    results.append(item)
    return {"pending_renders": pending[1:], "render_results": results}


async def render_finalize_node(state: QGenCrewState) -> dict[str, Any]:
    """Stamp the rendered URLs onto the candidate(s), strip the raw
    ``image_specs`` (never published), write the render trace row + tallies
    and raise ``quality_warning`` when an author-requested image is missing
    (a forced image is INTEGRAL to the item: the stem says "using the
    illustration below", so a missing image BREAKS it). Pure pass-through when
    no render was planned."""
    plan = dict(state.get("render_plan") or {})
    lane = str(plan.get("lane") or "")
    if not lane:
        return {}
    results = list(state.get("render_results") or [])
    warnings = [str(r.get("error")) for r in results if r.get("error")]
    rendered_per_candidate: dict[int, dict[str, str]] = {}
    rendered_count = 0
    for r in results:
        if r.get("error"):
            continue
        bucket = rendered_per_candidate.setdefault(int(r.get("idx") or 0), {})
        bucket[str(r["field_name"])] = str(r["url"])
        bucket[str(r["gcs_field"])] = str(r["gs_uri"])
        rendered_count += 1
    emitted = int(plan.get("emitted") or 0)
    reset: dict[str, Any] = {
        "pending_renders": [],
        "render_results": [],
        "render_plan": {},
    }

    if lane == "single":
        candidate = state.get("current_candidate")
        if candidate is None:
            return reset
        decoded = _safe_json_loads(candidate.payload_json, None)
        new_payload = dict(decoded) if isinstance(decoded, dict) else {}
        new_payload.pop("image_specs", None)
        new_payload.update(rendered_per_candidate.get(0, {}))
        notes = f"rendered {rendered_count}/{emitted} image(s)" + (
            f"; {len(warnings)} failed (fail-soft)" if warnings else ""
        )
        delta: dict[str, Any] = {
            **reset,
            "current_candidate": CandidatePayload(
                stem=candidate.stem,
                question_type=candidate.question_type,
                payload_json=json.dumps(new_payload),
                critic_notes=candidate.critic_notes,
            ),
            "pipeline_trace": _append_trace(
                state,
                "render_image",
                status="DEGRADED" if warnings else "COMPLETED",
                attempt=state.get("attempt_count"),
                notes=notes,
            ),
        }
        if warnings:
            # FAIL-LOUD. Per-spec fail-soft keeps the question, but the author
            # MUST be told: quality_warning is wired end-to-end (proto ->
            # chora-creation -> the A+ author banner + the O+ oversight gate).
            delta["errors"] = _append_error(state, "; ".join(warnings))
            delta["quality_warning"] = True
        return delta

    # ---- set lane ----
    capped = [dict(c) for c in (plan.get("capped") or [])]
    renderable_idx = [int(i) for i in (plan.get("renderable_idx") or [])]
    dropped = int(plan.get("dropped") or 0)
    skipped_specs = int(plan.get("skipped") or 0)
    cap_notes = [str(n) for n in (plan.get("cap_notes") or [])]
    accepted = list(state.get("accepted_set") or [])

    processed: list[dict[str, Any]] = []
    for idx, cand in enumerate(capped):
        nc = _strip_image_specs(cand)
        nc.update(rendered_per_candidate.get(idx, {}))
        processed.append(nc)
    by_original = dict(zip(renderable_idx, processed, strict=True))
    out = [by_original[i] if i in by_original else _strip_image_specs(c) for i, c in enumerate(accepted)]

    # Denominator is what the agent EMITTED, so a spec we trimmed and a spec the
    # renderer refused both show up as "an image this question does not have".
    missing = max(0, emitted - rendered_count)
    detail = ""
    if dropped:
        detail += f"; {dropped} dropped over budget"
    if warnings:
        detail += f"; {len(warnings)} failed (fail-soft)"
    if skipped_specs:
        detail += f"; {skipped_specs} skipped (quality_warning)"
    rows = list(state.get("pipeline_trace") or [])
    rows.append(
        _trace_row(
            "render_image",
            status="DEGRADED" if missing else "COMPLETED",
            notes=f"rendered {rendered_count}/{emitted} image(s)" + detail,
        )
    )
    delta = {
        **reset,
        "accepted_set": out,
        "pipeline_trace": rows,
        # Structured mirror of the notes tallies (ADR-251 D5) for the
        # chunk_completed publish.
        "chunk_image_tallies": {
            "rendered": rendered_count,
            "dropped": dropped,
            "failed": len(warnings),
            "skipped": skipped_specs,
        },
    }
    if warnings or cap_notes:
        delta["errors"] = _append_error(state, "; ".join(cap_notes + warnings))
    if missing:
        # FAIL-LOUD: a forced image is INTEGRAL to the item (composer_question.go
        # integralImageRule), so a question that shipped without one is broken,
        # not merely plainer. This fires for a cap DROP as well as a render
        # failure, because either way a delivered question references an
        # illustration it does not carry.
        delta["quality_warning"] = True
    return delta


def route_after_render_prepare(state: QGenCrewState) -> str:
    """render_image / render_image_set -> render_next (queue non-empty) OR
    render_finalize (nothing to render, incl. the pure pass-through)."""
    return "render_next" if state.get("pending_renders") else "render_finalize"


def route_after_render_next(state: QGenCrewState) -> str:
    """render_next -> render_next (more images) OR render_finalize."""
    return "render_next" if state.get("pending_renders") else "render_finalize"


def route_after_render_finalize(state: QGenCrewState) -> str:
    """render_finalize -> publish_chunk (set lane) OR publish_completed."""
    return "publish_chunk" if state.get("set_mode") else "publish_completed"


# ---------------------------------------------------------------------------
# Single-pass SET-native nodes (CHO-1819) — mixed-type batch + honest
# strict-shortfall + per-type image budget + bounded regenerate. N=1 is a
# one-element set, so single + batch collapse onto this path when set_mode is
# True. The legacy single-candidate lane above stays byte-for-byte unchanged
# (set_mode absent ⇒ route_after_guardrail_pre → "generate").
# ---------------------------------------------------------------------------


def _set_execution_id(state: QGenCrewState, step: str) -> str:
    """``{job}:{step}:c{chunk}:r{round}`` (ADR-254 D2): unique per chunk AND
    regenerate round, so the dispatch idempotency key never collides across
    the chunk loop."""
    return (
        f"{state.get('job_id', '')}:{step}:c{int(state.get('chunk_index') or 0)}:r{int(state.get('regen_round') or 0)}"
    )


def _round_type_plan(state: QGenCrewState) -> list[dict[str, Any]]:
    """The plan for the CURRENT generate round: the REDUCED round_type_plan
    when a bounded regenerate is in flight, else the CURRENT chunk's plan
    (ADR-251 D1), else the FULL type_plan (direct-node callers and pre-chunk
    fixtures)."""
    return list(state.get("round_type_plan") or state.get("chunk_plan") or state.get("type_plan") or [])


async def generate_set_node(state: QGenCrewState, *, executor: _ExecutorLike) -> dict[str, Any]:
    """Single generate-set dispatch to qgen_question (ROLE_GENERATE).

    Mirrors :func:`generate_node`'s executor-call + grounding passthrough + trace
    shape, but builds a SET payload (set_mode + the round's type_plan +
    avoid_concepts) and FAIL-LOUD parses the wrapper via
    :func:`parse_set_response`. The trace row name stays ``generate`` (the
    observability ABI). A contract violation → VALIDATION refusal (never pads).
    """
    type_plan = _round_type_plan(state)
    regen_round = int(state.get("regen_round") or 0)
    grounding_mode = str(state.get("grounding_mode") or "")
    question_type = str(state.get("question_type") or "mixed")

    gen_input: dict[str, Any] = {
        "prompt": state.get("prompt", ""),
        "set_mode": True,
        "type_plan": type_plan,
        "metadata": state.get("metadata") or {},
        "attempt_index": regen_round,
        "gcid": _gcid(state),
        **_current_w3c_trace_context(),
    }
    # Regenerate dedup — the executor stamps avoid_concepts_json; we ALSO append
    # the dedup_context_block to the prompt (mirrors how grounding_prompt_block
    # is appended) so even a not-yet-redeployed agent that ignores the state key
    # still sees "do not repeat these concepts" in its prompt. Round 0 has none.
    avoid_concepts = list(state.get("avoid_concepts") or [])
    if avoid_concepts:
        gen_input["avoid_concepts"] = avoid_concepts
        suffix = str(state.get("regen_prompt_suffix") or "") or dedup_context_block(
            avoid_concepts, list(state.get("rejected_notes") or [])
        )
        if suffix:
            gen_input["prompt"] = str(gen_input.get("prompt") or "") + "\n" + suffix

    # Grounding passthrough — identical to generate_node so a grounded set
    # inlines the uploaded blob + appends the citations/rubric prompt block.
    source_blob_uri = str(state.get("source_blob_uri") or "")
    if source_blob_uri:
        gen_input["source_blob_uri"] = source_blob_uri
        gen_input["source_mime_type"] = str(state.get("source_mime_type") or "")
        gen_input["grounding_mode"] = grounding_mode
    target_growth_edges = state.get("target_growth_edges") or []
    if target_growth_edges:
        gen_input["target_growth_edges"] = list(target_growth_edges)
    source_files = state.get("source_files") or []
    if source_files:
        gen_input["source_files"] = list(source_files)
        gen_input["prompt"] = str(gen_input.get("prompt") or "") + grounding_prompt_block(
            files=list(source_files),
            grounding_mode=grounding_mode,
            question_type=question_type,
        )

    # ADR-197 M-B.2 (CHO-2368 P2 fix) — the set lane threads the qgen_question
    # override exactly like generate_node. Without this the runner still stamps
    # the resolver's version/source on batch decisions, claiming an override
    # that never reached the agent (false provenance).
    _thread_prompt_overrides(gen_input, state, "question")
    payload = json.dumps(gen_input)
    started_at = _now_iso()
    try:
        # ADR-254 D2: the execution id (and so the dispatch idempotency key)
        # carries the CHUNK as well as the round. Keyed on the round alone, chunk
        # 2's first generate collided with chunk 1's and deduped to nothing.
        resp = await executor.execute(
            execution_id=_set_execution_id(state, "generate_set"),
            tenant_id=state.get("tenant_id", ""),
            agid=ROLE_GENERATE,
            agent_role=ROLE_GENERATE,
            input_payload=payload,
            workflow_id=str(state.get("job_id") or ""),
            prompt_template_id="qgen_crew::generate",
        )
    except Exception as exc:
        # ADR-253: a park is control flow, not a failure. Without this the
        # Pub/Sub dispatch above is swallowed and the author is shown a
        # VALIDATION refusal on a run that merely parked.
        reraise_if_dispatch_park(exc)
        msg = f"generate_set executor error: {exc.__class__.__name__}: {exc}"
        logger.warning(
            "qgen_crew.generate_set.executor_failed",
            extra={"job_id": str(state.get("job_id") or ""), "err": msg},
            exc_info=True,
        )
        return {
            "refusal_reason": "VALIDATION",
            "refusal_user_facing_message": _SET_VALIDATION_USER_MESSAGE,
            "errors": _append_error(state, msg),
            "pipeline_trace": _append_trace(
                state,
                "generate",
                status="FAILED",
                attempt=regen_round + 1,
                notes=msg,
                started_at=started_at,
                completed_at=_now_iso(),
            ),
        }
    completed_at = _now_iso()
    gen_input_tokens = _input_tokens_of(resp)
    gen_output_tokens = int(getattr(resp, "output_tokens", 0) or 0)

    raw = _safe_json_loads(resp.output_payload or "", {})
    try:
        candidates, model_summary = parse_set_response(raw, type_plan, grounding_mode=grounding_mode)
    except AgentContractViolation as exc:
        msg = f"generate_set contract violation: {exc}"
        return {
            "refusal_reason": "VALIDATION",
            "refusal_user_facing_message": _SET_VALIDATION_USER_MESSAGE,
            "errors": _append_error(state, msg),
            "pipeline_trace": _append_trace(
                state,
                "generate",
                status="FAILED",
                attempt=regen_round + 1,
                input_tokens=gen_input_tokens,
                output_tokens=gen_output_tokens,
                notes=msg,
                started_at=started_at,
                completed_at=completed_at,
            ),
        }

    return {
        "candidate_set": candidates,
        "model_shortfall_reason": str(model_summary.get("shortfall_reason") or ""),
        "pipeline_trace": _append_trace(
            state,
            "generate",
            status="COMPLETED",
            attempt=regen_round + 1,
            input_tokens=gen_input_tokens,
            output_tokens=gen_output_tokens,
            notes=f"generated {len(candidates)} candidate(s)",
            started_at=started_at,
            completed_at=completed_at,
        ),
    }


async def guardrail_post_set_node(state: QGenCrewState, *, guardrail: _GuardrailLike) -> dict[str, Any]:
    """Cloud Model Armor post-screen on the WHOLE candidate set (one
    concatenated screen, whole-batch refuse-on-block — no per-candidate
    granularity, preserving the no-partial-success safety posture). Pass-through
    when generate_set already refused (empty set)."""
    if state.get("refusal_reason"):
        return {}
    candidate_set = state.get("candidate_set") or []
    text = json.dumps(candidate_set)
    try:
        result = await guardrail.screen(
            GuardrailScreenInput(
                tenant_id=state.get("tenant_id", ""),
                gcid=_gcid(state),
                agent_id=ROLE_CRITIQUE,
                content=text,
                direction="output",
            )
        )
        verdict = _screen_result_to_guardrail_result(result)
    except Exception as exc:
        msg = f"guardrail_post_set transport error: {exc.__class__.__name__}: {exc}"
        logger.exception(
            "qgen_crew.guardrail_post_set.transport_error",
            extra={
                "job_id": state.get("job_id", ""),
                "tenant_id": state.get("tenant_id", ""),
            },
        )
        return {
            "errors": _append_error(state, msg),
            "refusal_reason": "GUARDRAIL_POST",
            "refusal_user_facing_message": ("Post-screening failed — please try again in a moment."),
            "pipeline_trace": _append_trace(
                state,
                "guardrail_post",
                status="FAILED",
                attempt=int(state.get("regen_round") or 0) + 1,
                notes=msg,
            ),
        }
    return {
        "guardrail_post_result": verdict,
        "refusal_reason": "" if verdict.allowed else "GUARDRAIL_POST",
        "refusal_armor_verdict": verdict.armor_verdict if not verdict.allowed else "",
        "refusal_user_facing_message": (verdict.user_facing_message if not verdict.allowed else ""),
        "pipeline_trace": _append_trace(
            state,
            "guardrail_post",
            status="ACCEPTED" if verdict.allowed else "REFUSED",
            attempt=int(state.get("regen_round") or 0) + 1,
            notes=verdict.armor_verdict,
        ),
    }


async def critique_set_node(state: QGenCrewState, *, executor: _ExecutorLike) -> dict[str, Any]:
    """ONE batched qgen_critic call for the CURRENT ``candidate_set`` (CHO-2397,
    ADR-251 D2/D3; replaces the per-candidate semaphore fan-out).

    The payload carries ``set_mode`` + a ``candidates`` array with
    orchestrator-assigned index-based ``candidate_id``s (c0..cN per chunk +
    round); :func:`parse_critique_set_response` enforces the echo contract
    STRICTLY. Fail-loud by design, owner-ruled: an executor error or a
    correlation violation RAISES out of the node (no per-candidate fallback,
    no fabricated rejection) so the subscriber NACKs and redelivery resumes
    the chunk at critique, bounded by the delivery attempts then DLQ.

    Accepted candidates accumulate into the CUMULATIVE ``accepted_set``;
    rejected ones (with their critic notes attached) become the latest-round
    ``rejected_set``. Trace ABI: ONE row named ``critique`` carries the call's
    real token counts (one TokenUsageLedger event per real call, by
    construction at the model gateway) + per-candidate ``critique_verdict``
    rows with the verdicts and NO token fields, so the runner's per-hop
    aggregation stays one-to-one with real calls."""
    candidate_set = list(state.get("candidate_set") or [])
    if not candidate_set:
        return {
            "accepted_set": list(state.get("accepted_set") or []),
            "rejected_set": [],
            "pipeline_trace": list(state.get("pipeline_trace") or []),
        }

    regen_round = int(state.get("regen_round") or 0)
    ids = [f"c{i}" for i in range(len(candidate_set))]
    wire_candidates = [
        {"candidate_id": cid, **(cand if isinstance(cand, dict) else {})}
        for cid, cand in zip(ids, candidate_set, strict=True)
    ]
    enriched = {
        "set_mode": True,
        "candidates": wire_candidates,
        "author_prompt": state.get("prompt", ""),
        "attempt_index": regen_round,
        "prior_critic_notes": "",
        "metadata": state.get("metadata") or {},
        "gcid": _gcid(state),
        **_current_w3c_trace_context(),
    }
    # ADR-197 M-B.2 (CHO-2368 P2 fix) — the set critique hop threads the
    # qgen_critic override exactly like critique_node (see generate_set_node's
    # matching note on the false-provenance stakes).
    _thread_prompt_overrides(enriched, state, "critic")
    payload = json.dumps(enriched)
    started_at = _now_iso()
    resp = await executor.execute(
        execution_id=_set_execution_id(state, "critique_set"),
        tenant_id=state.get("tenant_id", ""),
        agid=ROLE_CRITIQUE,
        agent_role=ROLE_CRITIQUE,
        input_payload=payload,
        workflow_id=str(state.get("job_id") or ""),
        prompt_template_id="qgen_crew::critique",
    )
    completed_at = _now_iso()

    raw = _safe_json_loads(resp.output_payload or "", {})
    verdicts = parse_critique_set_response(raw, ids)

    newly_accepted: list[dict[str, Any]] = []
    newly_rejected: list[dict[str, Any]] = []
    rows = list(state.get("pipeline_trace") or [])
    accepted_count = sum(1 for v in verdicts.values() if v["accepted"])
    rows.append(
        _trace_row(
            "critique",
            status="COMPLETED",
            attempt=regen_round + 1,
            input_tokens=_input_tokens_of(resp),
            output_tokens=int(getattr(resp, "output_tokens", 0) or 0),
            notes=(
                f"chunk {int(state.get('chunk_index') or 0) + 1}"
                f"/{int(state.get('chunk_count') or 1)}: 1 call, "
                f"{len(candidate_set)} candidate(s), {accepted_count} accepted, "
                f"{len(candidate_set) - accepted_count} rejected"
            ),
            started_at=started_at,
            completed_at=completed_at,
        )
    )
    for cid, cand in zip(ids, candidate_set, strict=True):
        verdict = verdicts[cid]
        if verdict["accepted"]:
            newly_accepted.append(cand)
        else:
            rejected = dict(cand)
            # Carry the critic note so regenerate can build the dedup block and
            # quality_gate can stamp the included-with-warning candidate.
            rejected["_critic_notes"] = verdict["critique_notes"]
            newly_rejected.append(rejected)
        rows.append(
            _trace_row(
                "critique_verdict",
                status="ACCEPTED" if verdict["accepted"] else "REJECTED",
                attempt=regen_round + 1,
                notes=verdict["critique_notes"][:200],
                accepted=verdict["accepted"],
                suggested_revisions=verdict["suggested_revisions"],
            )
        )

    accepted_set = list(state.get("accepted_set") or []) + newly_accepted
    return {
        "accepted_set": accepted_set,
        "rejected_set": newly_rejected,
        "pipeline_trace": rows,
    }


async def quality_gate_set_node(state: QGenCrewState) -> dict[str, Any]:
    """Pure decision node for the set lane's CURRENT chunk. RETRY
    (→ regenerate_rejected) while rejects remain AND the chunk's regen budget
    is unspent; otherwise CHUNK-TERMINAL — route to render_image_set. Any
    still-rejected candidates on a terminal pass are INCLUDED with a
    per-candidate ``quality_warning`` (matching the locked single-path
    exhausted semantics — completed.v1 + quality_warning, never refused.v1).
    The job-level GenerationSummary is computed once, in finalize_set_node,
    over ALL chunks (ADR-251 D1); the job-level warning is written only when
    True so a clean later chunk can never overwrite an earlier chunk's warning
    (the delta-merge would make False win)."""
    accepted = list(state.get("accepted_set") or [])
    rejected = list(state.get("rejected_set") or [])
    regen_round = int(state.get("regen_round") or 0)
    max_regen = state.get("max_regen_rounds")
    if max_regen is None:
        max_regen = DEFAULT_MAX_REGEN_ROUNDS

    if rejected and regen_round < int(max_regen):
        return {
            "pipeline_trace": _append_trace(
                state,
                "quality_gate",
                status="RETRY",
                attempt=regen_round + 1,
                notes=(
                    f"{len(rejected)} candidate(s) rejected on round "
                    f"{regen_round}; regenerate (budget {regen_round + 1}/"
                    f"{int(max_regen)})"
                ),
            ),
        }

    quality_warning = bool(rejected)
    delta: dict[str, Any]
    if quality_warning:
        included: list[dict[str, Any]] = []
        for cand in rejected:
            cc = {k: v for k, v in cand.items() if k != "_critic_notes"}
            cc["quality_warning"] = True
            notes = str(cand.get("_critic_notes") or "")
            if notes:
                cc["critic_notes"] = notes
            included.append(cc)
        final_set = accepted + included
        decision = "QUALITY_WARNING"
        notes = (
            f"{len(rejected)} candidate(s) included with quality_warning (regen exhausted after round {regen_round})"
        )
        delta = {"quality_warning": True}
    else:
        final_set = accepted
        decision = "ACCEPTED"
        notes = f"all {len(accepted)} candidate(s) accepted"
        delta = {}

    delta.update(
        {
            "accepted_set": final_set,
            "rejected_set": [],
            "pipeline_trace": _append_trace(
                state,
                "quality_gate",
                status=decision,
                attempt=regen_round + 1,
                notes=notes,
            ),
        }
    )
    return delta


async def regenerate_rejected_node(state: QGenCrewState) -> dict[str, Any]:
    """State-prep for ONE bounded regenerate round (no LLM here — generate_set
    does the dispatch). Builds the REDUCED round plan (only the rejected
    types/counts), the dedup context (already-covered stems + why prior
    attempts were rejected), and bumps regen_round. The FULL ``type_plan`` is
    left intact so the terminal summary reconciles against the original
    requested counts."""
    rejected = list(state.get("rejected_set") or [])
    accepted = list(state.get("accepted_set") or [])
    regen_round = int(state.get("regen_round") or 0) + 1

    # ADR-251 D1 — a regen round is scoped to the CURRENT chunk, so its image
    # budgets and forced-image flags come from the chunk's apportioned plan
    # (falls back to the full plan for direct-node callers).
    full_plan = list(state.get("chunk_plan") or state.get("type_plan") or [])
    caps = image_caps(full_plan)
    # CHO-1825 — the per-type deterministic image opt-ins must survive onto the
    # REDUCED round plan, or regenerated questions of a forced type ship
    # imageless. Index the original quotas by type to copy the flags forward.
    flags_by_type = {str(q.get("question_type") or ""): q for q in full_plan}
    counts: dict[str, int] = {}
    for cand in rejected:
        qt = str(cand.get("question_type") or "")
        counts[qt] = counts.get(qt, 0) + 1
    round_plan: list[dict[str, Any]] = []
    for qt, n in counts.items():
        if n <= 0:
            continue
        entry: dict[str, Any] = {
            "question_type": qt,
            "count": n,
            "max_images": min(caps.get(qt, 0), n),
        }
        src = flags_by_type.get(qt) or {}
        if src.get("image_for_stem"):
            entry["image_for_stem"] = True
        if src.get("image_for_answer"):
            entry["image_for_answer"] = True
        round_plan.append(entry)

    # Dedup context covers everything already delivered: prior chunks'
    # completed candidates (ADR-251 D1) plus this chunk's accepted so far.
    covered = list(state.get("completed_candidates") or []) + accepted
    accepted_stems = [str(c.get("stem") or "") for c in covered]
    rejected_notes = [str(c.get("_critic_notes") or "") for c in rejected]

    return {
        "regen_round": regen_round,
        "round_type_plan": round_plan,
        "avoid_concepts": accepted_stems,
        "rejected_notes": rejected_notes,
        "regen_prompt_suffix": dedup_context_block(accepted_stems, rejected_notes),
        # Consumed — the re-critique repopulates rejected_set with what's still
        # rejected after the regenerated candidates are judged.
        "rejected_set": [],
        "pipeline_trace": _append_trace(
            state,
            "regenerate_rejected",
            status="RETRY",
            attempt=regen_round,
            notes=f"regenerating {len(rejected)} rejected candidate(s)",
        ),
    }


def _strip_image_specs(candidate: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy with the raw ``image_specs`` stripped (never published)."""
    return {k: v for k, v in candidate.items() if k != "image_specs"}


async def render_image_set_node(
    state: QGenCrewState,
    *,
    kroki: _KrokiLike | None,
    gcs: _GcsImageLike | None,
) -> dict[str, Any]:
    """PLAN the per-candidate images across the accepted set, bounded by the
    per-type cap (ADR-254 D12: the renders run one per superstep in
    :func:`render_next_node`, the results land in :func:`render_finalize_node`).
    The :func:`enforce_image_caps` backstop deterministically drops image_specs
    an over-emitting agent returned beyond the cap (LOUD warning trace). Pure
    pass-through when no candidate carries image_specs; fail-LOUD when a
    renderable spec is present but an image client is unconfigured
    (mis-config). Candidates the critic rejected (candidate-level
    ``quality_warning``, stamped by quality_gate_set_node under the exhausted
    semantics) are published but never rendered (CHO-2395); their skip is a
    deliberate third tally, distinct from both the over-cap trim and a render
    failure."""
    accepted = list(state.get("accepted_set") or [])
    if not accepted:
        return {}

    # CHO-2395: a candidate the critic REJECTED is published (with
    # quality_warning, per the locked exhausted semantics) but must not spend
    # gemini-3-pro-image shared quota (~22s/image) on an illustration for a
    # question the author is being warned about anyway. Partition BEFORE the
    # caps and the wiring check, so skipped specs neither consume the per-type
    # budget nor turn an unconfigured image client into a mis-config when no
    # renderable spec exists.
    renderable_idx = [i for i, c in enumerate(accepted) if not c.get("quality_warning")]
    renderable = [accepted[i] for i in renderable_idx]
    skipped_specs = sum(len(c.get("image_specs") or []) for c in accepted if c.get("quality_warning"))

    # CHO-1825: the render backstop admits the discretionary AI-decide budget
    # PLUS the deterministic author-forced allowance (image_for_stem /
    # image_for_answer force ONE image per question), so an author-forced image
    # is NEVER dropped as "over budget" when max_images == 0. ADR-251 D1: the
    # budget is the CURRENT chunk's apportioned plan (chunk_type_plan preserves
    # the aggregate), falling back to the full plan for direct-node callers.
    caps = effective_image_caps(list(state.get("chunk_plan") or state.get("type_plan") or []))
    # Count what the agent EMITTED before the backstop trims, because the honest
    # denominator for "does this question have the image it references" is what
    # was asked for, not what survived our own cap. Deliberately skipped
    # (quality_warning) specs are NOT in this denominator.
    emitted_specs = sum(len(c.get("image_specs") or []) for c in renderable)
    capped, dropped = enforce_image_caps(renderable, caps)

    rows = list(state.get("pipeline_trace") or [])
    # Kept SEPARATE from render failures. A deterministic over-budget trim and a
    # throttled renderer are different events: the first is us working as
    # designed, the second is the vendor.
    cap_notes: list[str] = []
    if dropped:
        msg = f"agent emitted {dropped} image(s) over the per-type cap; dropped deterministically (backstop)"
        cap_notes.append(msg)
        logger.warning(
            "qgen_crew.render_image_set.over_cap",
            extra={"job_id": state.get("job_id", ""), "dropped": dropped},
        )
        rows.append(_trace_row("render_image", status="WARNING", notes=msg))

    if skipped_specs:
        logger.info(
            "qgen_crew.render_image_set.skipped_rejected",
            extra={
                "job_id": state.get("job_id", ""),
                "skipped_specs": skipped_specs,
            },
        )

    total_specs = sum(len(c.get("image_specs") or []) for c in capped)
    if total_specs == 0:
        # No renderable image work: strip any (now-empty or skipped)
        # image_specs + return.
        by_original = dict(zip(renderable_idx, [_strip_image_specs(c) for c in capped], strict=True))
        cleaned = [by_original[i] if i in by_original else _strip_image_specs(c) for i, c in enumerate(accepted)]
        delta: dict[str, Any] = {
            "accepted_set": cleaned,
            # Structured mirror of the notes tallies (ADR-251 D5): the
            # chunk_completed publish carries these as typed counts.
            "chunk_image_tallies": {
                "rendered": 0,
                "dropped": dropped,
                "failed": 0,
                "skipped": skipped_specs,
            },
        }
        if skipped_specs:
            # The deliberate skip must stay visible even when nothing renders,
            # or "all images belonged to rejected candidates" reads as a clean
            # imageless run.
            rows.append(
                _trace_row(
                    "render_image",
                    status="COMPLETED",
                    notes=(f"rendered 0/0 image(s); {skipped_specs} skipped (quality_warning)"),
                )
            )
            delta["pipeline_trace"] = rows
        if dropped:
            delta["pipeline_trace"] = rows
            delta["errors"] = _append_error(state, "; ".join(cap_notes))
        return delta

    _require_image_clients("render_image_set", kroki=kroki, gcs=gcs)

    job_id = str(state.get("job_id") or "")
    chunk_index = int(state.get("chunk_index") or 0)
    pending: list[dict[str, Any]] = []
    for idx, cand in enumerate(capped):
        for spec_index, spec in enumerate(cand.get("image_specs") or []):
            if not isinstance(spec, dict):
                continue
            placement, field_name, gcs_field = _spec_fields(spec)
            pending.append(
                {
                    "idx": idx,
                    "spec_index": spec_index,
                    "placement": placement,
                    "field_name": field_name,
                    "gcs_field": gcs_field,
                    "mode": str(spec.get("mode") or "").strip().lower(),
                    "source": str(spec.get("source") or ""),
                    "execution_id": (f"{job_id}:render:c{chunk_index}:i{idx}:s{spec_index}:{placement}"),
                }
            )
    delta = {
        "pending_renders": pending,
        "render_results": [],
        "render_plan": {
            "lane": "set",
            "capped": capped,
            "renderable_idx": renderable_idx,
            "dropped": dropped,
            "skipped": skipped_specs,
            "emitted": emitted_specs,
            "cap_notes": cap_notes,
        },
    }
    if dropped:
        delta["pipeline_trace"] = rows
    return delta


async def publish_completed_node(state: QGenCrewState) -> dict[str, Any]:
    """Terminal node — populates completed_candidate (and quality_warning
    if applicable). The PUBLISH side-effect (Pub/Sub event emission to
    chora.creation.ai_assist.completed.v1) is performed by the
    subscriber-side wrapper after ainvoke returns — keeps this node pure.

    quality_warning is True when EITHER:
      - critic rejected and we exhausted retries, OR
      - the evaluator self-rejected (below_threshold) on the last attempt
        (critic was never invoked for that attempt).
    """
    # Set lane (CHO-1819): the candidates live in accepted_set (read by the
    # runner) and quality_warning was already decided by quality_gate_set_node.
    # Preserve it rather than re-deriving from the single-path critic_result
    # (which is None in set mode) — otherwise the set's warning would be lost.
    if state.get("set_mode"):
        quality_warning = bool(state.get("quality_warning") or False)
        return {
            "quality_warning": quality_warning,
            "pipeline_trace": _append_trace(
                state,
                "publish_completed",
                status="COMPLETED",
                notes="set quality_warning" if quality_warning else "set accepted",
            ),
        }

    critic = state.get("critic_result")
    eval_rejected = bool(state.get("evaluator_below_threshold") or False)
    critic_rejected = critic is not None and not critic.accepted
    # OR in a warning already raised upstream - render_image_node sets it when
    # an author-requested image could not be rendered. Re-deriving purely from
    # the critic verdict (the previous shape) SILENTLY DROPPED that warning, so
    # a clean-critique candidate with a missing integral image shipped looking
    # perfect (job b4d12cd0, 2026-08-14).
    render_warning = bool(state.get("quality_warning") or False)
    quality_warning = critic_rejected or eval_rejected or render_warning
    if eval_rejected:
        notes = f"quality_warning (evaluator below_threshold: {state.get('evaluator_reason') or 'unspecified'})"
    elif critic_rejected:
        notes = "quality_warning"
    elif render_warning:
        notes = "quality_warning (image render degraded)"
    else:
        notes = "accepted"
    return {
        "completed_candidate": state.get("current_candidate"),
        "quality_warning": quality_warning,
        "pipeline_trace": _append_trace(
            state,
            "publish_completed",
            status="COMPLETED",
            attempt=state.get("attempt_count"),
            notes=notes,
        ),
    }


async def publish_refused_node(state: QGenCrewState) -> dict[str, Any]:
    """Terminal node — validates the refusal envelope. Pub/Sub publish to
    chora.creation.ai_assist.refused.v1 is the subscriber-side wrapper's
    job.

    Fails loud if refusal_reason is unset OR not in the allowed set
    per [[feedback-no-stubs-real-wiring]].
    """
    reason = state.get("refusal_reason", "")
    if not reason:
        # Pipeline bug — we routed to publish_refused without setting
        # refusal_reason. Fail loud rather than emit a malformed event.
        raise ValueError("publish_refused called with empty refusal_reason — pipeline bug")
    assert_valid_refusal_reason(reason)
    return {
        "pipeline_trace": _append_trace(
            state,
            "publish_refused",
            status="REFUSED",
            attempt=state.get("attempt_count"),
            notes=f"reason={reason} verdict={state.get('refusal_armor_verdict', '')}",
        ),
    }


# ---------------------------------------------------------------------------
# Conditional edge functions (LangGraph routers)
# ---------------------------------------------------------------------------


def route_after_validate(state: QGenCrewState) -> str:
    """validate_input → guardrail_pre OR publish_refused."""
    return "publish_refused" if state.get("refusal_reason") else "guardrail_pre"


def route_after_guardrail_pre(state: QGenCrewState) -> str:
    """guardrail_pre → generate_set (set lane) OR generate (legacy) OR
    publish_refused. ``set_mode`` selects the single-pass set lane (CHO-1819);
    its absence keeps the live single-candidate path byte-for-byte unchanged."""
    if state.get("refusal_reason"):
        return "publish_refused"
    if state.get("set_mode"):
        return "generate_set"
    return "generate"


def route_after_generate_set(state: QGenCrewState) -> str:
    """generate_set → guardrail_post_set OR publish_refused (on a VALIDATION
    contract violation — no Armor screen burned on an empty set)."""
    return "publish_refused" if state.get("refusal_reason") else "guardrail_post_set"


def route_after_guardrail_post_set(state: QGenCrewState) -> str:
    """guardrail_post_set → critique_set OR publish_refused (whole-batch block)."""
    return "publish_refused" if state.get("refusal_reason") else "critique_set"


def route_after_quality_gate_set(state: QGenCrewState) -> str:
    """quality_gate_set → regenerate_rejected (rejects remain ∧ regen budget
    unspent) OR render_image_set (terminal). Mirrors quality_gate_set_node's
    own RETRY-vs-terminal decision so the router + node never diverge."""
    rejected = state.get("rejected_set") or []
    regen_round = int(state.get("regen_round") or 0)
    max_regen = state.get("max_regen_rounds")
    if max_regen is None:
        max_regen = DEFAULT_MAX_REGEN_ROUNDS
    if rejected and regen_round < int(max_regen):
        return "regenerate_rejected"
    return "render_image_set"


def route_after_generate(state: QGenCrewState) -> str:
    """generate → guardrail_post OR publish_refused.

    generate_node records an executor failure (a FAILED dispatch completion, a
    pre-park refusal) as ``refusal_reason``; guardrail_post would RESET that
    field, so the refusal must leave the loop here. Without this edge a failed
    dispatch rode into guardrail_post -> critique -> quality_gate -> RETRY and
    dispatched up to max_retries more times before publishing an empty
    candidate with quality_warning (ADR-254 bus exposure).
    """
    return "publish_refused" if state.get("refusal_reason") else "guardrail_post"


def route_after_guardrail_post(state: QGenCrewState) -> str:
    """guardrail_post → critique OR quality_gate OR publish_refused.

    Three branches:
      1. ``refusal_reason`` set (post-screen blocked) → publish_refused
      2. ``evaluator_below_threshold`` True → quality_gate (skip critique;
         the evaluator already self-rejected, so handing the empty-stem
         candidate to the critic would just confuse it)
      3. Default → critique
    """
    if state.get("refusal_reason"):
        return "publish_refused"
    if state.get("evaluator_below_threshold"):
        return "quality_gate"
    return "critique"


def route_after_quality_gate(state: QGenCrewState) -> str:
    """quality_gate → render_image OR generate (retry).

    Decision tree (mirrors quality_gate_node's trace):
      - critic accepted    → render_image (then publish_completed)
      - critic rejected OR evaluator below_threshold
        + attempt_count <= max_retries  → generate (loop)
        + attempt_count >  max_retries  → render_image (then
                                          publish_completed with
                                          quality_warning per
                                          publish_completed_node)

    W8: both terminal-completed branches now route through ``render_image``
    first. render_image is a PURE no-op pass-through when the candidate has no
    ``image_specs`` (the default), so the live MCQ loop is unchanged; it only
    does work when the author opted into a stem / answer image. The
    ``render_image → publish_completed`` edge is unconditional.

    Per user clarification 2026-05-17 retries exhausted emits
    completed.v1 with quality_warning, NOT refused.v1.
    """
    critic = state.get("critic_result")
    attempt = state.get("attempt_count") or 0
    max_retries = state.get("max_retries") or DEFAULT_MAX_RETRIES
    eval_rejected = bool(state.get("evaluator_below_threshold") or False)
    # Critic acceptance only matters if the evaluator didn't self-reject
    # this attempt (the critic was skipped under that branch).
    if not eval_rejected and critic is not None and critic.accepted:
        return "render_image"
    if attempt < max_retries + 1:
        return "generate"
    return "render_image"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gcid(state: QGenCrewState) -> str:
    """Resolve the author GCID from state.

    Canonical key is ``gcid`` (per :class:`QGenCrewState`); we also honour
    ``user_gcid`` for forward-compat with callers that namespace the field.
    Returns ``""`` when neither is set — the Port stamps it into the
    ScreenRequest for audit; an empty value is acceptable (matches the
    legacy adapter which passed ``gcid=""``).
    """
    return str(state.get("gcid") or state.get("user_gcid") or "").strip()


def _thread_prompt_overrides(target: dict[str, Any], state: QGenCrewState, suffix: str) -> None:
    """ADR-197 M-B.2 — thread the per-agent resolved prompt override (stored by
    the runner under ``prompt_overrides_<suffix>`` / ``prompt_version_<suffix>``
    / ``prompt_source_<suffix>``) into the executor input dict ``target`` using
    the PINNED contract keys the Go composer + the executor's
    ``_build_session_state`` read: ``prompt_overrides_json`` (a JSON-encoded
    ``{segment_id -> body}`` map), ``resolved_prompt_version``, ``prompt_source``.

    Only set when an override actually applied (the runner stores nothing
    otherwise), so the live single-candidate payload is byte-for-byte unchanged
    when no resolver is wired OR no active override exists. ``suffix`` is
    ``question`` for the generator hop / ``critic`` for the critique hop.
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


def _safe_json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default


def _bool(value: Any) -> bool:
    """Coerce a state value to a plain bool. Tolerates None / missing keys
    (default False) so the author's per-image opt-in degrades safely when the
    chora-contracts field isn't yet present on the started.v1 envelope (W8 —
    the field is being added in a parallel contracts layer)."""
    return bool(value)


def _input_tokens_of(resp: AgentExecutorResponse) -> int:
    """Per-hop input-token count off the executor response, with the C3
    total fallback: prefer the explicit ``input_tokens`` split; when the
    executor only emitted ``tokens_consumed_total`` (the M11 baseline shape,
    e.g. the legacy gRPC fakes), fall back to the total so the trace row's
    input_tokens stays non-zero and the live MCQ loop's existing assertions
    keep passing.
    """
    split = int(getattr(resp, "input_tokens", 0) or 0)
    if split:
        return split
    return int(getattr(resp, "tokens_consumed_total", 0) or 0)


def _current_w3c_trace_context() -> dict[str, str]:
    """Capture the active OTel span as a W3C ``traceparent`` (+ ``tracestate``)
    dict so the agent call can CONTINUE this workflow trace.

    The qgen_crew graph runs inside the runner's ``qgen_crew.handle_started``
    span (parented to the FE/started.v1 traceparent — see
    orchestrators/qgen_crew_runner.py ~L480). Reading the current span here
    yields the trace_id + span_id the agent must inherit so the orchestrator →
    qgen_question / qgen_critic → gateway chain renders as ONE distributed
    trace per workflow (CR qgen Phase B2, 2026-06-01).

    Returns an empty dict (best-effort, fail-soft) when OTel is uninstalled or
    no valid span is active — the agent then starts its own local trace tree
    rather than crashing. Mirrors adapter/agent_io/agent_response.py
    ``_build_traceparent_metadata``.
    """
    try:  # OTel may be uninstalled in trimmed-down envs.
        from opentelemetry import trace as _trace
        from opentelemetry.trace import format_span_id, format_trace_id
    except ImportError:  # pragma: no cover — defensive
        return {}

    span = _trace.get_current_span()
    ctx = span.get_span_context() if span is not None else None
    if ctx is None or not ctx.is_valid:
        return {}

    flags = "01" if ctx.trace_flags.sampled else "00"
    out = {"traceparent": (f"00-{format_trace_id(ctx.trace_id)}-{format_span_id(ctx.span_id)}-{flags}")}
    if ctx.trace_state:
        out["tracestate"] = ",".join(f"{k}={v}" for k, v in ctx.trace_state.items())
    return out


def _is_evaluator_below_threshold(raw: Any) -> bool:
    """Detect the evaluator's below-threshold self-rejection signal.

    The qgen_question reasoning engine's terminal evaluator sub-agent
    legitimately emits ``{"scored": null, "reason": "below_threshold", ...}``
    when the candidate's composite quality score < 0.6 (per
    StepEvaluation3 in ``agents/qgen_adk_go/internal/agent/composer_question.
    go``). The executor's ``_map_response`` preserves the shape verbatim
    (the unwrap-`scored` branch is gated on ``isinstance(scored, dict)``
    which is False when scored is null) — so the orchestrator sees the
    raw dict here.

    This shape is a SUCCESS wire path, NOT an error: the generator
    finished and the evaluator declined to surface the candidate. The
    orchestrator must route back to ``generate`` (retry) WITHOUT
    invoking the (separate) critic, who would otherwise be handed an
    empty-stem candidate.

    Recognised shapes:
      {"scored": null, "reason": "below_threshold"}
      {"scored": null, "reason": "below_threshold", "details": "..."}
    """
    return isinstance(raw, dict) and "scored" in raw and raw.get("scored") is None and bool(raw.get("reason"))


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ADR-251 D1 (CHO-2396) — the set lane's GRAPH-structured chunk loop.
#
#   plan_chunks → generate_set → guardrail_post_set → critique_set
#       → quality_gate_set → {regenerate_rejected ↺ | render_image_set}
#       → publish_chunk → {generate_set (next chunk) | finalize_set}
#       → publish_completed
#
# The loop lives IN THE GRAPH, never inside a node: LangGraph persists state at
# node transitions, so an in-node loop's progress dies with the pod and makes
# chunk-granular resume (ADR-251 D4) impossible. That variant was considered
# and rejected as a shortcut (owner directive 2026-08-16).
# ---------------------------------------------------------------------------


async def plan_chunks_node(state: QGenCrewState) -> dict[str, Any]:
    """Set-lane entry: validate the full plan (defence-in-depth behind
    chora-creation's own aiassist.MaxBatchCount), then split it into chunk
    plans bounded by QGEN_SET_MAX_PER_CALL. A validation breach routes to
    publish_refused (reason=VALIDATION) exactly like validate_input_node."""
    type_plan = list(state.get("type_plan") or [])
    requested = total_count(type_plan)
    issues = validate_type_plan(type_plan, requested_count=requested)
    if issues:
        return {
            "refusal_reason": "VALIDATION",
            "refusal_user_facing_message": "; ".join(issues),
            "pipeline_trace": _append_trace(
                state,
                "plan_chunks",
                status="REJECTED",
                notes="; ".join(issues)[:300],
            ),
        }

    size = _chunk_size()
    chunks = chunk_type_plan(type_plan, size)
    return {
        "chunk_plans": chunks,
        "chunk_count": len(chunks),
        "chunk_index": 0,
        "chunk_plan": list(chunks[0]) if chunks else [],
        "completed_candidates": [],
        "accumulated_shortfall_reasons": [],
        "pipeline_trace": _append_trace(
            state,
            "plan_chunks",
            status="COMPLETED",
            notes=(f"{requested} question(s) in {len(chunks)} chunk(s) of <= {size}"),
        ),
    }


def route_after_plan_chunks(state: QGenCrewState) -> str:
    if state.get("refusal_reason"):
        return "publish_refused"
    return "generate_set"


async def publish_chunk_node(state: QGenCrewState, *, publisher: Any | None = None) -> dict[str, Any]:
    """Chunk boundary: publish the chunk_completed.v1 outbox row (ADR-251 D5,
    CHO-2398), fold the finished chunk's candidates into the cumulative job
    result, record the per-chunk tallies in the trace, clear the per-chunk
    working state, and advance the cursor.

    ``publisher`` is the QGenCrewTerminalOutboxWriter (duck-typed via
    ``publish_chunk_completed``); None keeps the pre-D5 fold byte-identical
    (legacy callers + tests). A publish failure RAISES out of the node: the
    drive fails loudly, the registry row survives, and the boot sweep resumes
    from the checkpoint; the deterministic per-chunk idempotency key makes
    the re-emit harmless."""
    chunk_index = int(state.get("chunk_index") or 0)
    chunk_count = int(state.get("chunk_count") or 0)
    chunk_result = list(state.get("accepted_set") or [])
    completed = list(state.get("completed_candidates") or []) + chunk_result

    warned = sum(1 for c in chunk_result if c.get("quality_warning"))
    notes = f"chunk {chunk_index + 1}/{chunk_count}: {len(chunk_result)} candidate(s)"
    if warned:
        notes += f"; {warned} warned"

    if publisher is not None:
        tallies = dict(state.get("chunk_image_tallies") or {})
        trace_ctx = _current_w3c_trace_context()
        await publisher.publish_chunk_completed(
            assist_id=str(state.get("job_id") or ""),
            tenant_id=str(state.get("tenant_id") or ""),
            author_gcid=_gcid(state),
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            candidates_payload_json=json.dumps(chunk_result),
            candidate_count=len(chunk_result),
            warned_count=warned,
            images_rendered=int(tallies.get("rendered") or 0),
            images_dropped=int(tallies.get("dropped") or 0),
            images_failed=int(tallies.get("failed") or 0),
            images_skipped=int(tallies.get("skipped") or 0),
            traceparent=str(trace_ctx.get("traceparent") or ""),
            tracestate=str(trace_ctx.get("tracestate") or ""),
        )

    reasons = list(state.get("accumulated_shortfall_reasons") or [])
    chunk_reason = str(state.get("model_shortfall_reason") or "").strip()
    if chunk_reason:
        reasons.append(chunk_reason)

    # Cross-chunk dedup: the next chunk's generate prompt carries every stem
    # already delivered, via the same avoid_concepts channel the regenerate
    # round uses.
    avoid = list(state.get("avoid_concepts") or [])
    avoid.extend(str(c.get("stem") or "") for c in chunk_result if c.get("stem"))

    next_index = chunk_index + 1
    chunk_plans = list(state.get("chunk_plans") or [])
    next_plan = list(chunk_plans[next_index]) if next_index < len(chunk_plans) else []
    return {
        "completed_candidates": completed,
        "accumulated_shortfall_reasons": reasons,
        "chunk_index": next_index,
        "chunk_plan": next_plan,
        "avoid_concepts": avoid,
        # Per-chunk working state resets for the next cycle.
        "accepted_set": [],
        "rejected_set": [],
        "candidate_set": [],
        "regen_round": 0,
        "round_type_plan": [],
        "rejected_notes": [],
        "regen_prompt_suffix": "",
        "model_shortfall_reason": "",
        "chunk_image_tallies": {},
        "pending_renders": [],
        "render_results": [],
        "render_plan": {},
        "pipeline_trace": _append_trace(
            state,
            "publish_chunk",
            status="COMPLETED",
            notes=notes,
        ),
    }


def route_after_publish_chunk(state: QGenCrewState) -> str:
    if int(state.get("chunk_index") or 0) < int(state.get("chunk_count") or 0):
        return "generate_set"
    return "finalize_set"


async def finalize_set_node(state: QGenCrewState) -> dict[str, Any]:
    """Job terminal for the chunk loop: reassemble the full set, compute the
    honest GenerationSummary against the ORIGINAL type_plan, and settle the
    job-level quality_warning (sticky: a render-raised or chunk-raised warning
    survives; explicit False only when nothing warned)."""
    final_set = list(state.get("completed_candidates") or [])
    type_plan = list(state.get("type_plan") or [])
    chunk_count = int(state.get("chunk_count") or 0)
    reasons = [r for r in (state.get("accumulated_shortfall_reasons") or []) if r]
    summary = compute_generation_summary(
        type_plan,
        final_set,
        grounding_mode=str(state.get("grounding_mode") or ""),
        model_shortfall_reason="; ".join(reasons),
    )
    any_warn = bool(state.get("quality_warning")) or any(c.get("quality_warning") for c in final_set)
    return {
        "accepted_set": final_set,
        "generation_summary": summary,
        "quality_warning": any_warn,
        "pipeline_trace": _append_trace(
            state,
            "finalize_set",
            status="COMPLETED",
            notes=(f"{len(final_set)} candidate(s) across {chunk_count} chunk(s)"),
        ),
    }


async def compose_test_set_node(state: QGenCrewState, *, executor: _ExecutorLike) -> dict[str, Any]:
    """Lane 1c test-set composition on the bus (ADR-180 D4/D5, ADR-254 D2):
    ONE ``mode=compose`` dispatch on the qgen_generate lane over the FINAL
    accepted set, after finalize_set and before publish_completed.

    The agent (chora-qgen-question) reads the candidates (each carrying the
    SAME deterministic draft_id the terminal stamps, so order/points line up
    with what chora-creation stores) and the source files BY REFERENCE, and
    answers ``{"proposed_test_set": {title, description, order, points}}``;
    the kennel repairs it into the contract shape (:func:`normalise_proposal`).

    DEFENSIVE CONTRACT (D4): a FAILED dispatch, an unusable answer or any
    other error degrades to the DETERMINISTIC :func:`fallback_proposal`
    (trace row FALLBACK); the batch never fails because composing failed. A
    park is control flow and is never swallowed. Pass-through when the
    feature is off (``compose_enabled`` unset: no dispatch, no trace row).
    """
    if not state.get("compose_enabled"):
        return {}
    job_id = str(state.get("job_id") or "")
    accepted = [c for c in (state.get("accepted_set") or []) if isinstance(c, dict)]
    candidates = [{**c, "draft_id": deterministic_draft_id(job_id, i)} for i, c in enumerate(accepted)]
    files = [f for f in (state.get("source_files") or []) if isinstance(f, dict)]
    prompt = str(state.get("prompt") or "")
    fb = fallback_proposal(
        payload=SimpleNamespace(prompt=prompt, assist_id=job_id),
        candidates=candidates,
        files=files,
    )
    started_at = _now_iso()
    payload: dict[str, Any] = {
        "mode": "compose",
        "candidates": candidates,
        "author_prompt": prompt,
        "metadata": dict(state.get("metadata") or {}),
        "grounding_mode": str(state.get("grounding_mode") or ""),
        "gcid": _gcid(state),
        **_current_w3c_trace_context(),
    }
    if files:
        payload["source_files"] = [
            {
                "gs_uri": str(f.get("gs_uri") or f.get("blob_uri") or ""),
                "mime_type": str(f.get("mime_type") or ""),
                "role": str(f.get("role") or "source"),
            }
            for f in files
        ]
    try:
        resp = await executor.execute(
            execution_id=f"{job_id}:compose",
            tenant_id=str(state.get("tenant_id") or ""),
            agid=ROLE_GENERATE,
            agent_role=ROLE_GENERATE,
            input_payload=json.dumps(payload),
            workflow_id=job_id,
            prompt_template_id="qgen_crew::compose",
        )
    except Exception as exc:
        reraise_if_dispatch_park(exc)
        msg = f"compose dispatch failed: {exc.__class__.__name__}: {exc}"
        logger.exception("qgen_crew.compose_test_set.fallback", extra={"job_id": job_id})
        return {
            "proposed_test_set": fb,
            "compose_used_fallback": True,
            "pipeline_trace": _append_trace(
                state,
                "compose_test_set",
                status="FALLBACK",
                notes=f"deterministic fallback proposal ({msg})",
                started_at=started_at,
                completed_at=_now_iso(),
            ),
        }
    completed_at = _now_iso()
    raw = _safe_json_loads(resp.output_payload or "", {})
    proposal_raw = raw.get("proposed_test_set") if isinstance(raw, dict) else None
    in_tok = _input_tokens_of(resp)
    out_tok = int(getattr(resp, "output_tokens", 0) or 0)
    if not isinstance(proposal_raw, dict):
        logger.warning(
            "qgen_crew.compose_test_set.unusable_answer",
            extra={"job_id": job_id, "head": str(resp.output_payload or "")[:160]},
        )
        return {
            "proposed_test_set": fb,
            "compose_used_fallback": True,
            "pipeline_trace": _append_trace(
                state,
                "compose_test_set",
                status="FALLBACK",
                input_tokens=in_tok,
                output_tokens=out_tok,
                notes="deterministic fallback proposal (compose answer carried no proposed_test_set)",
                started_at=started_at,
                completed_at=completed_at,
            ),
        }
    proposal = normalise_proposal(proposal_raw, candidates=candidates, fallback=fb)
    return {
        "proposed_test_set": proposal,
        "compose_used_fallback": False,
        "pipeline_trace": _append_trace(
            state,
            "compose_test_set",
            status="COMPLETED",
            input_tokens=in_tok,
            output_tokens=out_tok,
            notes=f"proposed test set: {len(proposal.get('order') or [])} question(s)",
            started_at=started_at,
            completed_at=completed_at,
        ),
    }


def route_after_finalize_set(state: QGenCrewState) -> str:
    """finalize_set -> compose_test_set (feature on) OR publish_completed."""
    return "compose_test_set" if state.get("compose_enabled") else "publish_completed"


def build_qgen_crew_graph(
    *,
    executor: _ExecutorLike,
    guardrail: _GuardrailLike,
    checkpointer: Any | None = None,
    kroki: _KrokiLike | None = None,
    gcs: _GcsImageLike | None = None,
    publisher: Any | None = None,
) -> Any:
    """Compile the qgen 2-agent crew StateGraph.

    ``publisher`` (ADR-251 D5, CHO-2398): the QGenCrewTerminalOutboxWriter,
    duck-typed via ``publish_chunk_completed``; when wired, publish_chunk
    emits one chunk_completed.v1 outbox row per finished chunk. None keeps
    the chunk boundary a pure fold (legacy callers + tests byte-identical).

    Args:
        executor: the QGenDispatchAdapter (production) or duck-typed
            fake (tests). Dispatches to qgen_question + qgen_critic
            Vertex AI Agent Engine reasoning engines via gRPC.
        guardrail: tier-mapped :class:`ModelArmorGuardrailPort` (production)
            or a duck-typed fake (tests). Per ADR-169 the template tier is
            resolved per agent_id from ``agent-guardrail-mapping.yaml`` —
            the pre-screen tags ``qgen_question`` and the post-screen tags
            ``qgen_critic`` (both balanced). No bare env template name is
            threaded through anymore.
        checkpointer: LangGraph checkpointer (PostgresSaver in prod,
            InMemorySaver in tests). When None the compiled graph
            runs stateless (development convenience; production MUST
            supply PostgresSaver per [[agentic-resilience-d6]] P1).
        kroki / gcs: the image clients that stay in the kennel (KrokiClient
            for Mermaid, GcsImageUploadAdapter for the Mermaid upload + the
            signing of every image); injected at composition root. BOTH
            default None: the render nodes are a pure no-op pass-through
            unless a candidate actually carries ``image_specs`` (so the live
            MCQ loop works with no image clients wired). When image_specs ARE
            present but a client is None the plan node fails loud (mis-config
            per [[secrets-and-env]]). The scene image itself is a qgen_render
            DISPATCH through ``executor`` (ADR-254 D12), not a client here.

    Returns:
        Compiled LangGraph StateGraph ready for `await graph.ainvoke({...},
        config={"configurable": {"thread_id": job_id}})`.
    """
    graph: StateGraph = StateGraph(QGenCrewState)

    # Async bound-callable wrappers — LangGraph node fns take only `state`,
    # so we close over the executor + guardrail via lambdas. Type ignore
    # because LangGraph 0.6+ accepts async callables but mypy's stubs are
    # conservative.
    async def _validate(state: QGenCrewState) -> dict[str, Any]:
        return await validate_input_node(state)

    async def _gpre(state: QGenCrewState) -> dict[str, Any]:
        return await guardrail_pre_node(state, guardrail=guardrail)

    async def _generate(state: QGenCrewState) -> dict[str, Any]:
        return await generate_node(state, executor=executor)

    async def _gpost(state: QGenCrewState) -> dict[str, Any]:
        return await guardrail_post_node(state, guardrail=guardrail)

    async def _critique(state: QGenCrewState) -> dict[str, Any]:
        return await critique_node(state, executor=executor)

    async def _qgate(state: QGenCrewState) -> dict[str, Any]:
        return await quality_gate_node(state)

    async def _render_image(state: QGenCrewState) -> dict[str, Any]:
        return await render_image_node(state, kroki=kroki, gcs=gcs)

    async def _render_next(state: QGenCrewState) -> dict[str, Any]:
        return await render_next_node(state, executor=executor, kroki=kroki, gcs=gcs)

    async def _render_finalize(state: QGenCrewState) -> dict[str, Any]:
        return await render_finalize_node(state)

    async def _pub_completed(state: QGenCrewState) -> dict[str, Any]:
        return await publish_completed_node(state)

    async def _pub_refused(state: QGenCrewState) -> dict[str, Any]:
        return await publish_refused_node(state)

    # Set-native lane (CHO-1819) — single-pass mixed-type generation.
    async def _generate_set(state: QGenCrewState) -> dict[str, Any]:
        return await generate_set_node(state, executor=executor)

    async def _gpost_set(state: QGenCrewState) -> dict[str, Any]:
        return await guardrail_post_set_node(state, guardrail=guardrail)

    async def _critique_set(state: QGenCrewState) -> dict[str, Any]:
        return await critique_set_node(state, executor=executor)

    async def _qgate_set(state: QGenCrewState) -> dict[str, Any]:
        return await quality_gate_set_node(state)

    async def _regenerate_rejected(state: QGenCrewState) -> dict[str, Any]:
        return await regenerate_rejected_node(state)

    async def _render_image_set(state: QGenCrewState) -> dict[str, Any]:
        return await render_image_set_node(state, kroki=kroki, gcs=gcs)

    # ADR-251 D1 chunk-loop nodes (CHO-2396).
    async def _plan_chunks(state: QGenCrewState) -> dict[str, Any]:
        return await plan_chunks_node(state)

    async def _publish_chunk(state: QGenCrewState) -> dict[str, Any]:
        return await publish_chunk_node(state, publisher=publisher)

    async def _finalize_set(state: QGenCrewState) -> dict[str, Any]:
        return await finalize_set_node(state)

    graph.add_node("validate_input", _validate)
    graph.add_node("guardrail_pre", _gpre)
    graph.add_node("generate", _generate)
    graph.add_node("guardrail_post", _gpost)
    graph.add_node("critique", _critique)
    graph.add_node("quality_gate", _qgate)
    graph.add_node("render_image", _render_image)
    # ADR-254 D12: the render loop (one image per superstep, shared by both lanes).
    graph.add_node("render_next", _render_next)
    graph.add_node("render_finalize", _render_finalize)
    graph.add_node("publish_completed", _pub_completed)
    graph.add_node("publish_refused", _pub_refused)
    # Set lane nodes.
    graph.add_node("plan_chunks", _plan_chunks)
    graph.add_node("generate_set", _generate_set)
    graph.add_node("guardrail_post_set", _gpost_set)
    graph.add_node("critique_set", _critique_set)
    graph.add_node("quality_gate_set", _qgate_set)
    graph.add_node("regenerate_rejected", _regenerate_rejected)
    graph.add_node("render_image_set", _render_image_set)
    graph.add_node("publish_chunk", _publish_chunk)
    graph.add_node("finalize_set", _finalize_set)

    async def _compose(state: QGenCrewState) -> dict[str, Any]:
        return await compose_test_set_node(state, executor=executor)

    graph.add_node("compose_test_set", _compose)

    graph.add_edge(START, "validate_input")

    graph.add_conditional_edges(
        "validate_input",
        route_after_validate,
        {"guardrail_pre": "guardrail_pre", "publish_refused": "publish_refused"},
    )
    graph.add_conditional_edges(
        "guardrail_pre",
        route_after_guardrail_pre,
        {
            "generate": "generate",
            # ADR-251 D1 — the set lane enters through the chunk planner; the
            # route fn still names the lane ("generate_set"), the mapping
            # lands it on plan_chunks.
            "generate_set": "plan_chunks",
            "publish_refused": "publish_refused",
        },
    )
    graph.add_conditional_edges(
        "plan_chunks",
        route_after_plan_chunks,
        {"generate_set": "generate_set", "publish_refused": "publish_refused"},
    )
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {"guardrail_post": "guardrail_post", "publish_refused": "publish_refused"},
    )
    graph.add_conditional_edges(
        "guardrail_post",
        route_after_guardrail_post,
        {
            "critique": "critique",
            "quality_gate": "quality_gate",
            "publish_refused": "publish_refused",
        },
    )
    graph.add_edge("critique", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {"generate": "generate", "render_image": "render_image"},
    )
    # W8 / ADR-254 D12: render_image PLANS the images on the terminal-completed
    # path (a no-op pass-through unless the accepted candidate carries
    # image_specs), render_next renders ONE per superstep (a scene image parks
    # on its qgen_render dispatch), render_finalize stamps the results and
    # routes to the lane's publish node.
    graph.add_conditional_edges(
        "render_image",
        route_after_render_prepare,
        {"render_next": "render_next", "render_finalize": "render_finalize"},
    )
    graph.add_conditional_edges(
        "render_next",
        route_after_render_next,
        {"render_next": "render_next", "render_finalize": "render_finalize"},
    )
    graph.add_conditional_edges(
        "render_finalize",
        route_after_render_finalize,
        {"publish_completed": "publish_completed", "publish_chunk": "publish_chunk"},
    )

    # Set-native lane edges (CHO-1819 + ADR-251 D1 chunk loop): plan_chunks →
    # generate_set → guardrail_post_set → critique_set → quality_gate_set →
    # {regenerate_rejected ↺ | render_image_set} → publish_chunk →
    # {generate_set (next chunk) | finalize_set} → publish_completed.
    # regenerate_rejected loops back to generate_set (the bounded per-chunk
    # dedup regenerate); both lanes converge at publish_completed /
    # publish_refused.
    graph.add_conditional_edges(
        "generate_set",
        route_after_generate_set,
        {
            "guardrail_post_set": "guardrail_post_set",
            "publish_refused": "publish_refused",
        },
    )
    graph.add_conditional_edges(
        "guardrail_post_set",
        route_after_guardrail_post_set,
        {"critique_set": "critique_set", "publish_refused": "publish_refused"},
    )
    graph.add_edge("critique_set", "quality_gate_set")
    graph.add_conditional_edges(
        "quality_gate_set",
        route_after_quality_gate_set,
        {
            "regenerate_rejected": "regenerate_rejected",
            "render_image_set": "render_image_set",
        },
    )
    graph.add_edge("regenerate_rejected", "generate_set")
    # ADR-251 D1 / ADR-254 D12: render_image_set plans the chunk's images, the
    # shared render loop renders them, render_finalize routes to publish_chunk
    # (set_mode); then next chunk or finalize_set -> publish_completed.
    graph.add_conditional_edges(
        "render_image_set",
        route_after_render_prepare,
        {"render_next": "render_next", "render_finalize": "render_finalize"},
    )
    graph.add_conditional_edges(
        "publish_chunk",
        route_after_publish_chunk,
        {"generate_set": "generate_set", "finalize_set": "finalize_set"},
    )
    # ADR-254 D2: the test-set composer is a dispatch on the qgen_generate lane
    # (one more park, only when QGEN_TESTSET_COMPOSE_ENABLED threads
    # compose_enabled into the set state); off => straight to the terminal.
    graph.add_conditional_edges(
        "finalize_set",
        route_after_finalize_set,
        {"compose_test_set": "compose_test_set", "publish_completed": "publish_completed"},
    )
    graph.add_edge("compose_test_set", "publish_completed")

    graph.add_edge("publish_completed", END)
    graph.add_edge("publish_refused", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
