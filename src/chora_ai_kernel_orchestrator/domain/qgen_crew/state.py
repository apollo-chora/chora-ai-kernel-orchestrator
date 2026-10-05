"""QGenCrewState TypedDict + per-stage value objects for the qgen 2-agent crew.

Per docs/m13/ack-oe-ai-assist-plan-2026-05-17.md the qgen 2-agent crew
is the agentic backing for chora-creation's POST /api/atoms/ai-assist
async surface. It runs:

    qgen_question (generator)  ─►  qgen_critic (critic)  ─►  quality_gate
                                                                │
                              ┌─────────────────────────────────┘
                              ▼
                      ┌───────────────┐
                      │ accepted=true │ → publish completed.v1
                      └───────────────┘
                              │
                       accepted=false
                              │
                      ┌───────────────┐
                      │ retries < max │ → loop back to generator
                      └───────────────┘
                              │
                       retries == max
                              │
                      ┌───────────────────────────┐
                      │ publish completed.v1 with │
                      │ quality_warning + last    │
                      │ critic_notes              │
                      └───────────────────────────┘

Pre/post guardrail (Cloud Model Armor) nodes flank generation; either-side
block routes to publish_refused with refusal_reason set.

Pure-domain module: no infra imports (no httpx / grpc / langgraph). All
side effects live in the adapter layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

# Default max retries for the critic-rejection branch. The first attempt
# is attempt_count=1; max_retries=3 → up to 4 total attempts before the
# orchestrator emits completed.v1 with quality_warning=True.
DEFAULT_MAX_RETRIES: int = 3

# Allowed refusal_reason values for the AiAssistRefused publish path —
# kept in sync with chora-contracts/openapi/creation-questions.yaml
# `AiAssistRefusal.reason` enum.
_ALLOWED_REFUSAL_REASONS: frozenset[str] = frozenset({"GUARDRAIL_PRE", "GUARDRAIL_POST", "VALIDATION"})


@dataclass(frozen=True)
class CandidatePayload:
    """One generated AiAssistCandidate (per OpenAPI AiAssistCandidate).

    Stored verbatim from the qgen_question agent output and from the
    quality-loop's regenerator iterations. The orchestrator carries the
    full payload as a JSON-encoded string in
    AiAssistCompleted.candidate_payload_json on publish — same as the
    Pub/Sub event schema in chora-contracts/proto/events/creation/
    ai_assist.proto §AiAssistCompleted.candidate_payload_json.
    """

    stem: str
    question_type: str  # "mcq" | "oe"
    payload_json: str  # JSON-encoded mcq_payload OR oe_payload per question_type
    critic_notes: str = ""


@dataclass(frozen=True)
class CritiqueResult:
    """qgen_critic agent output — drives the quality_gate conditional edge.

    Mirrors CritiqueOutput in services/chora-ai-kernel-orchestrator/
    agents/qgen_adk_go/internal/agent/critic.go §outputCritiqueMCQ +
    §outputCritiqueOE.

    Per user clarification 2026-05-17 — qualitative critique ONLY (NOT
    numeric scoring; "scoring is a different agent / different concern").
    """

    accepted: bool
    critique_notes: str = ""
    suggested_revisions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GuardrailResult:
    """Cloud Model Armor pre- or post-screen verdict.

    Per [[cloud-model-armor-guardrails]] the SDK call returns a verdict
    that maps to either ALLOW (proceed) or BLOCK (route to refused.v1).
    """

    allowed: bool
    armor_verdict: str = ""  # e.g., "armor:pii_high_risk_block"
    user_facing_message: str = ""


class QGenCrewState(TypedDict, total=False):
    """LangGraph state for the qgen 2-agent crew.

    `total=False` so partial-dict node returns merge cleanly via the
    LangGraph default reducer.
    """

    # ---- Caller-supplied (from AiAssistStarted event) ---------------------
    job_id: str
    tenant_id: str
    gcid: str
    prompt: str
    question_type: str  # "mcq" | "oe"
    metadata: dict[str, str]  # {category, subject, cognitive_level, ...}
    max_retries: int

    # ---- W8 author image opt-in (per-image, author-SELECTED not model-decided)
    # Threaded into the generate executor payload so the generation agent knows
    # whether to emit an image_spec for the stem and/or the model answer. Both
    # default False (absent) ⇒ no image — the live MCQ loop is unchanged. These
    # land on the started.v1 envelope in a parallel chora-contracts layer; the
    # runner reads them defensively (getattr / dict .get with default False).
    image_for_stem: bool
    image_for_answer: bool

    # ---- CHO-1658 model_answer_fill -------------------------------------------
    # The compose intent ("" for the live new_question loop; "model_answer_fill"
    # for the fill path) + the author's EXISTING question content (stem +
    # mcq_options / oe_rubric + model_answer). generate_node forwards both to the
    # qgen_question executor ONLY when set (intent non-empty / existing_question
    # non-empty), so the live loop's generate payload is byte-stable. The executor
    # reads input_obj["intent"] + input_obj["existing_question"] into the fill
    # session keys (author_stem / author_options / author_rubric / model_answer).
    intent: str
    existing_question: dict[str, Any]

    # ---- EPIC-1a grounding (batch source-material) ------------------------
    # Threaded into generate_node's agent payload so the qgen_question agent can
    # inline the uploaded blob as a Gemini multimodal part (contents_json
    # inlineData — the renderer is agent-side). Empty source_blob_uri ⇒ no
    # grounding, so the live single-candidate path is unchanged.
    #   grounding_mode: "" | "starting_point" (seed) | "strict" (closed-book)
    source_blob_uri: str
    source_mime_type: str
    grounding_mode: str
    target_growth_edges: list[str]  # concept_keys to bias generation (1b seam)

    # ---- Lane 1c multi-file + rubric grounding (CHO-1703, additive) -------
    # The EFFECTIVE role-tagged grounding-file list resolved by the batch
    # runner (source_files f20 canonical; f17/18 single-file fallback).
    # Entries are {"blob_uri", "mime_type", "role"} dicts; role ∈
    # source|rubric. Only grounded BATCH jobs set the key — generate_node
    # forwards it to the agent + appends the grounding/citations prompt
    # block, so single + ungrounded paths stay byte-for-byte unchanged.
    source_files: list[dict[str, str]]

    # ---- ADR-197 M-B.2 prompt-override resolution (per agent role) --------
    # The runner resolves the active prompt override for qgen_question +
    # qgen_critic at handle_started (via the optional PromptResolver bound to
    # the shared psycopg conn) and stores the WINNING result here, but ONLY
    # when an override actually applied (segments non-empty). When no resolver
    # is wired OR the resolver returns the embedded default, NONE of these keys
    # are set → the graph nodes thread nothing → byte-identical pre-registry
    # behaviour. Declared as channels so LangGraph's default reducer persists
    # the input values through to the terminal state (undeclared keys are
    # dropped on merge — see the round_type_plan note above).
    #   prompt_overrides_*  : {segment_id -> override body} (role/task/examples)
    #   prompt_version_*    : the winning override plan's opaque id
    #   prompt_source_*     : "tenant_override" | "platform_override"
    prompt_overrides_question: dict[str, str]
    prompt_version_question: str
    prompt_source_question: str
    prompt_overrides_critic: dict[str, str]
    prompt_version_critic: str
    prompt_source_critic: str

    # ---- Loop state -------------------------------------------------------
    # attempt_count is 1-based: 1 = first attempt; max value = max_retries+1.
    attempt_count: int
    current_candidate: CandidatePayload | None
    critic_notes: str  # carried into the next regenerator attempt

    # ---- Single-pass SET generation (CHO-1819) ----------------------------
    # Present ONLY for set-mode (mixed-type / batch) jobs. Absent ⇒ the legacy
    # single-candidate lane above is byte-for-byte unchanged. type_plan is the
    # resolved [{question_type, count, max_images}] quotas; set_mode gates the
    # set nodes; avoid_concepts seeds the regenerate dedup block. candidate_set
    # is the raw parsed wrapper; accepted_set / rejected_set partition it after
    # critique; regen_round bounds the regenerate-rejected loop; model_shortfall
    # _reason is the agent's strict-shortfall note; generation_summary is the
    # authoritative honest summary published on completed.v1.
    set_mode: bool
    type_plan: list[dict[str, Any]]
    avoid_concepts: list[str]
    candidate_set: list[dict[str, Any]]
    accepted_set: list[dict[str, Any]]
    rejected_set: list[dict[str, Any]]
    regen_round: int
    max_regen_rounds: int
    model_shortfall_reason: str
    generation_summary: dict[str, Any]
    # Regenerate-round bookkeeping (set ONLY during a bounded regenerate loop):
    # round_type_plan is the REDUCED plan (rejected types/counts) the current
    # generate round must satisfy — the FULL type_plan above is preserved so the
    # terminal summary reconciles against the original requested counts;
    # rejected_notes carries the prior round's critic notes + regen_prompt_suffix
    # the dedup_context_block appended to the regenerate prompt. These MUST be
    # declared so LangGraph's default reducer merges them across the loop edge
    # (undeclared keys are dropped on merge).
    round_type_plan: list[dict[str, Any]]
    rejected_notes: list[str]
    regen_prompt_suffix: str

    # ---- ADR-251 D1 chunk loop (CHO-2396) ---------------------------------
    # plan_chunks splits type_plan into chunk_plans (chunk_type_plan, bounded
    # by QGEN_SET_MAX_PER_CALL); the per-chunk cycle generate → critique →
    # quality_gate → render → publish_chunk runs against chunk_plan (the
    # CURRENT chunk's quotas, image budgets and forced-image flags), and
    # publish_chunk moves the chunk's results into completed_candidates before
    # clearing the per-chunk working keys and advancing chunk_index.
    # finalize_set computes the job-level GenerationSummary over
    # completed_candidates against the FULL original type_plan. All declared
    # so LangGraph's default reducer persists them across the loop edges
    # (undeclared keys are dropped on merge — see the round_type_plan note).
    chunk_plans: list[list[dict[str, Any]]]
    chunk_count: int
    chunk_index: int
    chunk_plan: list[dict[str, Any]]
    completed_candidates: list[dict[str, Any]]
    # render_image_set's structured tally mirror (ADR-251 D5): the
    # chunk_completed.v1 publish carries these as typed counts; publish_chunk
    # resets it with the rest of the per-chunk working state.
    chunk_image_tallies: dict[str, int]
    # ADR-254 D12: the render loop's queue (one image per superstep; a scene
    # image parks on its qgen_render dispatch), its results and the plan the
    # finalize node applies (lane, capped set, tallies). Cleared per chunk.
    pending_renders: list[dict[str, Any]]
    render_results: list[dict[str, Any]]
    render_plan: dict[str, Any]
    # ADR-254 D2 (mode=compose on the qgen_generate lane): the set lane's
    # test-set composition. compose_enabled is threaded from the wiring
    # (QGEN_TESTSET_COMPOSE_ENABLED); proposed_test_set is the contract-valid
    # ProposedTestSet the terminal publishes beside the candidates;
    # compose_used_fallback marks the deterministic degraded proposal (D4).
    compose_enabled: bool
    proposed_test_set: dict[str, Any]
    compose_used_fallback: bool
    # generate_set overwrites model_shortfall_reason on EVERY call, so a clean
    # later chunk would erase an earlier chunk's strict-shortfall reason;
    # publish_chunk folds each non-empty per-chunk reason in here and
    # finalize_set joins them for the job-level summary.
    accumulated_shortfall_reasons: list[str]

    # ---- Evaluator self-rejection signal (bug #5 from 2026-05-17 smoke) ---
    # qgen_question's terminal evaluator sub-agent legitimately emits
    # ``{"scored": null, "reason": "below_threshold", ...}`` when the
    # composite score < 0.6 (per ``agents/qgen_adk_go/internal/agent/
    # composer_question.go`` §StepEvaluation3). This is a SUCCESS wire
    # path — NOT an error — and tells the orchestrator to regenerate
    # WITHOUT invoking the (separate) critic, who would otherwise see
    # an empty-stem candidate and get confused.
    #
    # ``generate_node`` populates these keys when it detects the shape.
    # ``quality_gate_node`` + ``route_after_guardrail_post`` consume them:
    #
    #   - True + retries-left  → retry generator
    #   - True + retries-out   → publish_completed w/ quality_warning
    #
    # The critique node is SKIPPED in both branches (the critic never
    # sees the below-threshold candidate).
    evaluator_below_threshold: bool
    evaluator_reason: str  # e.g., "below_threshold"; FE-renderable

    # ---- Per-stage verdicts -----------------------------------------------
    guardrail_pre_result: GuardrailResult | None
    guardrail_post_result: GuardrailResult | None
    critic_result: CritiqueResult | None

    # ---- Trace (IMDA D2 transparency) -------------------------------------
    # Append-only list of per-step rows. Each row mirrors PipelineTraceStep
    # in chora-contracts/openapi/creation-questions.yaml §PipelineTraceStep.
    pipeline_trace: list[dict[str, Any]]

    # ---- Terminal state ---------------------------------------------------
    # Mutually-exclusive: at most one populated.
    completed_candidate: CandidatePayload | None
    quality_warning: bool
    refusal_reason: str  # one of _ALLOWED_REFUSAL_REASONS when set
    refusal_armor_verdict: str
    refusal_user_facing_message: str
    last_candidate_on_refusal: CandidatePayload | None

    # ---- Errors (orchestrator-internal) -----------------------------------
    errors: list[str]


def assert_valid_refusal_reason(reason: str) -> None:
    """Guard used by the publish_refused node — fails loud per
    [[feedback-no-stubs-real-wiring]] so we never publish a malformed
    AiAssistRefused.refusal_reason that the FE can't render."""
    if reason not in _ALLOWED_REFUSAL_REASONS:
        raise ValueError(f"refusal_reason must be one of {sorted(_ALLOWED_REFUSAL_REASONS)}; got {reason!r}")
