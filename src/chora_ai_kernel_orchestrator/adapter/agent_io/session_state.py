"""Agent session-state builder, shared by every transport (ADR-253 D6 / ADR-254 D2).

Moved out of the HTTP executor when the qgen crew went onto the Pub/Sub lanes,
and the only builder left once that executor was deleted (RULING A,
2026-08-23): the subscriber-only Go binaries merge the dispatch
``input_payload`` into ADK session state VERBATIM
(``chora-adk-common/agentdispatch`` ``Request.SessionState``), so the kennel
has to ship the exact session-state shape the composers read (``input_payload``
= the prompt text, ``subject_hint``, ``type_plan_json``, ``avoid_concepts_json``,
``author_stem`` ...). One builder, one shape, one transport.
"""

from __future__ import annotations

import json
from typing import Any

# Agent ids (the HTTP-era executor keys, also the Armor tier + decision-log
# identities) and the ADR-254 lane roles the Pub/Sub topics are keyed on. The
# two deliberately differ for qgen: the lane role names the WORK (generate /
# critique / render), the agent id names WHO does it.
ROLE_QGEN_QUESTION = "qgen_question"
ROLE_QGEN_CRITIC = "qgen_critic"
ROLE_OE_EVALUATE = "oe_evaluate"
ROLE_OE_MODERATE = "oe_moderate"

LANE_ROLE_GENERATE = "qgen_generate"
LANE_ROLE_CRITIQUE = "qgen_critique"
LANE_ROLE_RENDER = "qgen_render"

QUESTION_ROLES: frozenset[str] = frozenset({ROLE_QGEN_QUESTION, LANE_ROLE_GENERATE})
CRITIC_ROLES: frozenset[str] = frozenset({ROLE_QGEN_CRITIC, LANE_ROLE_CRITIQUE})
# Roles whose crew node controls the session state entirely: every payload key
# is merged as is (the OE grading keys; the qgen_render payload keys).
MERGE_ROLES: frozenset[str] = frozenset({ROLE_OE_EVALUATE, ROLE_OE_MODERATE, LANE_ROLE_RENDER})


def decode_input(payload: str) -> Any:
    """Decode the caller-supplied JSON. On parse failure forward as
    {raw: <string>} so the downstream agent sees the input verbatim and
    can fail loud: never silently drop input per
    [[feedback-no-stubs-real-wiring]].
    """
    if not payload:
        return {}
    try:
        decoded = json.loads(payload)
        if isinstance(decoded, dict):
            return decoded
        return {"raw": payload}
    except (ValueError, TypeError):
        return {"raw": payload}


def current_w3c_trace_context() -> dict[str, str]:
    """Capture the active OTel span as a W3C ``traceparent`` (+ ``tracestate``)
    dict. Best-effort fallback for ``_build_session_state`` when the caller
    didn't thread the trace context via ``input_obj`` (CR qgen Phase B2).

    Returns an empty dict when OTel is uninstalled or no valid span is active.
    Mirrors orchestrators/qgen_crew.py ``_current_w3c_trace_context`` +
    the gRPC client's ``_build_traceparent_metadata`` (deleted with it).
    """
    try:  # OTel may be uninstalled in trimmed-down envs.
        from opentelemetry import trace as _trace
        from opentelemetry.trace import format_span_id, format_trace_id
    except ImportError:  # pragma: no cover (defensive)
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


def stamp_metadata_hints(base_state: dict[str, Any], input_obj: dict[str, Any]) -> None:
    """Stamp the author's Subject / Cognitive Level / Difficulty selections
    as the ``*_hint`` session-state keys the ADK Go composer + critic read
    (``agents/qgen_adk_go/internal/agent/composer_question.go`` +
    ``critic.go:477-479``).

    The FE forwards these under ``input_obj['metadata']`` (a ``str->str`` map);
    generate_node / critique_node thread that map into the executor input.
    Only NON-EMPTY values are stamped so the composer/critic ``metadataHints``
    line stays clean (an empty hint must remain absent, not blank). This closes
    the producer gap the 2026-06-03 authoring-metadata audit surfaced: the
    reader side was wired all along but the executor never set the keys.
    """
    md = input_obj.get("metadata")
    if not isinstance(md, dict):
        return
    for src_key, dst_key in (
        ("subject", "subject_hint"),
        ("cognitive_level", "cognitive_level_hint"),
        ("difficulty", "difficulty_hint"),
    ):
        val = str(md.get(src_key) or "").strip()
        if val:
            base_state[dst_key] = val


def stamp_set_plan(base_state: dict[str, Any], input_obj: dict[str, Any]) -> None:
    """Stamp the single-pass SET-generation keys (CHO-1819) the ADK Go composer
    reads in set mode: ``set_mode`` (bool), ``type_plan_json`` (JSON array of
    ``{question_type, count, max_images}`` quotas), and ``avoid_concepts_json``
    (JSON array of already-covered stems the regenerate-rejected pass must not
    repeat). Stamped ONLY when a NON-EMPTY type_plan is present, so the live
    single-candidate / legacy-batch path stays byte-identical (absent ⇒ no set
    keys ⇒ composer ``SetMode==false``). The set-graph runner threads these via
    input_obj; they are absent for every single / legacy-batch invocation.
    """
    type_plan = input_obj.get("type_plan")
    if not (isinstance(type_plan, list) and type_plan):
        return
    base_state["type_plan_json"] = json.dumps(type_plan)
    base_state["set_mode"] = True
    avoid = input_obj.get("avoid_concepts")
    if isinstance(avoid, list) and avoid:
        base_state["avoid_concepts_json"] = json.dumps(avoid)


def build_session_state(
    *,
    agent_role: str,
    execution_id: str,
    tenant_id: str,
    input_obj: dict[str, Any],
) -> dict[str, Any]:
    """Build the agent session state for ``agent_role`` from the crew node's
    input object. The qgen_question / qgen_critic contracts are hard-coded and
    served for BOTH the agent ids (the HTTP executor's keys) and the ADR-254
    lane roles (``qgen_generate`` / ``qgen_critique``): the subscriber-only Go
    binaries merge the dispatch payload into session state verbatim, so the
    kennel ships this shape on the wire. The OE roles and ``qgen_render`` are
    a verbatim merge (the node controls the state). Any other role falls
    through to the minimal {tenant_id, user_gcid, trace} state.
    """
    gcid = str(input_obj.get("gcid") or input_obj.get("author_gcid") or "")
    if not gcid:
        # The orchestrator typically threads the author's gcid via
        # input_obj. If absent, use the tenant_id-derived sentinel so
        # the mana plugin's "session state must include tenant_id +
        # user_gcid" guard passes.
        gcid = f"qgen-anon:{tenant_id}"
    base_state: dict[str, Any] = {
        "tenant_id": tenant_id,
        "user_gcid": gcid,
        "author_gcid": gcid,
    }

    # ADR-197 M-B.2: prompt-override injection. When the orchestrator resolved
    # an active override for this (tenant, agent), the crew node threads the
    # segment map (JSON) + version + source via input_obj. Surface them on
    # session.state so the ADK Go composer's overrideOr(segment_id, embedded)
    # merge picks them up. Set BEFORE the role branches so it applies uniformly
    # to qgen_question / qgen_critic (explicit base_state.update) AND the OE
    # roles (generic input_obj merge would copy them too: same value). Absent /
    # empty ⇒ NOT set ⇒ the composer falls back to the embedded default →
    # byte-identical pre-registry behaviour.
    for _po_key in (
        "prompt_overrides_json",
        "resolved_prompt_version",
        "prompt_source",
    ):
        _po_val = input_obj.get(_po_key)
        if _po_val:
            base_state[_po_key] = _po_val

    # CR qgen Phase B2: W3C trace-context continuation. The qgen_crew nodes
    # thread the orchestrator's traceparent (+ optional tracestate) via
    # input_obj; if absent (e.g. a direct executor caller), fall back to the
    # active OTel span so the agent still inherits the workflow trace. The
    # qgen agents read state["traceparent"] in a BeforeAgentCallback
    # (chora-adk-common/tracing.AddInboundLink) and LINK their root span to
    # the orchestrator span. Empty when no span is active: safe no-op
    # agent-side.
    traceparent = str(input_obj.get("traceparent") or "")
    tracestate = str(input_obj.get("tracestate") or "")
    if not traceparent:
        captured = current_w3c_trace_context()
        traceparent = captured.get("traceparent", "")
        tracestate = tracestate or captured.get("tracestate", "")
    if traceparent:
        base_state["traceparent"] = traceparent
        if tracestate:
            base_state["tracestate"] = tracestate
    if agent_role in MERGE_ROLES:
        # ADR-172 OE grading crew: the crew node controls session.state by
        # threading every grading key (mode / prompt / rubric_json /
        # model_answer / learner_response / points_possible /
        # prior_moderation_notes / attempt_index / evaluation_json / summary
        # context) via input_obj. Merge them all so the agent's [TASK]
        # template renders, then return (no qgen-specific keys apply).
        for _k, _v in input_obj.items():
            if _k not in ("gcid", "author_gcid", "traceparent", "tracestate"):
                base_state[_k] = _v
        return base_state

    if agent_role in QUESTION_ROLES and str(input_obj.get("mode") or "").strip().lower() == "compose":
        # ADR-254 D2 (mode=compose, the Lane 1c test-set composer on the
        # qgen_generate lane): the chora-qgen-question port reads these keys
        # verbatim (mode, candidates as a list or JSON string, author_prompt,
        # metadata, source_files [{gs_uri, mime_type, role}] with the first
        # rubric winning, grounding_mode) and answers {"proposed_test_set":
        # {title, description, order, points}}; an unknown mode is a permanent
        # unknown_mode, missing/invalid candidates or source_files are
        # permanent faults by name. None of the generation keys apply.
        base_state["mode"] = "compose"
        base_state["candidates"] = list(input_obj.get("candidates") or [])
        base_state["author_prompt"] = str(input_obj.get("author_prompt") or "")
        base_state["metadata"] = dict(input_obj.get("metadata") or {})
        files = [f for f in (input_obj.get("source_files") or []) if isinstance(f, dict)]
        if files:
            base_state["source_files"] = files
        base_state["grounding_mode"] = str(input_obj.get("grounding_mode") or "")
        return base_state

    if agent_role in QUESTION_ROLES:
        intent = str(input_obj.get("intent") or "new_question")
        question_type = str(input_obj.get("question_type") or "mcq")
        base_state.update(
            {
                "intent": intent,
                "question_type": question_type,
                "input_payload": str(input_obj.get("prompt") or ""),
                # W8 author-opt-in image flags. generate_node forwards these in
                # input_payload; surface them on session.state so the ADK Go
                # composer's BuildTaskContextFromState (readStateBool) emits the
                # [IMAGE] block + image_specs output-schema fragment for the
                # opted-in part(s). Absent ⇒ False (dormant; byte-identical
                # pre-W8 prompt). Omitting these was the 2026-06-02 e2e gap:
                # the composer always read false → no image_specs → render_image
                # stayed dormant despite both drawer toggles being on.
                "image_for_stem": bool(input_obj.get("image_for_stem") or False),
                "image_for_answer": bool(input_obj.get("image_for_answer") or False),
            }
        )
        # CHO-1822: image_regen: surface the author's CURRENT (edited, unsaved)
        # context so the qgen agent's BuildTaskContextFromState composes the
        # imageRegenBlock (mermaid/scene decision + integral image from the
        # current stem/answer). Absent ⇒ the agent's regen block is empty.
        if intent == "image_regen":
            base_state["placement"] = str(input_obj.get("placement") or "stem")
            base_state["refinement_prompt"] = str(input_obj.get("refinement_prompt") or "")
            base_state["current_stem"] = str(input_obj.get("current_stem") or "")
            base_state["current_model_answer"] = str(input_obj.get("current_model_answer") or "")
            base_state["original_mode"] = str(input_obj.get("original_mode") or "")
            base_state["original_source"] = str(input_obj.get("original_source") or "")
        # Authoring-metadata wire-through (2026-06-03): surface the author's
        # Subject / Cognitive Level / Difficulty selections so the ADK Go
        # composer conditions generation on them (was a dead control pre-fix).
        stamp_metadata_hints(base_state, input_obj)
        # Single-pass SET generation (CHO-1819): surface the typed type_plan +
        # set_mode so the composer's set-mode branch emits the candidates +
        # generation_summary wrapper. Absent/empty ⇒ legacy single-candidate
        # prompt (byte-identical).
        stamp_set_plan(base_state, input_obj)
        # EPIC-1a batch grounding (2026-06-09): surface the uploaded source
        # material so the qgen_question agent's groundingplugin
        # (BeforeModelCallback) injects a gs:// FileData part into the LLM
        # request and Vertex Gemini reads it directly. generate_node forwards
        # these in input_payload ONLY when material was uploaded; absent ⇒ no
        # grounding (the live single-candidate + non-grounded batch paths are
        # byte-for-byte unchanged).
        source_blob_uri = str(input_obj.get("source_blob_uri") or "")
        source_mime_type = str(input_obj.get("source_mime_type") or "")
        # Lane 1c multi-file + rubric grounding (CHO-1703): surface the full
        # role-tagged file list as source_files_json. The DEPLOYED
        # groundingplugin reads only the scalar pair below (one FileData
        # part); source_files_json is the ADDITIVE forward seam a future
        # agent build consumes to append one FileData part per file (sources
        # + the role-distinct rubric). When the caller threads only
        # source_files (no scalars), DERIVE the scalar mirror from the first
        # role="source" entry so today's agent still grounds by-reference.
        source_files = input_obj.get("source_files")
        if isinstance(source_files, list) and source_files:
            base_state["source_files_json"] = json.dumps(source_files)
            if not source_blob_uri:
                for f in source_files:
                    if not isinstance(f, dict):
                        continue
                    uri = str(f.get("blob_uri") or "").strip()
                    role = str(f.get("role") or "").strip().lower()
                    if uri and role != "rubric":
                        source_blob_uri = uri
                        source_mime_type = str(f.get("mime_type") or "")
                        break
        if source_blob_uri:
            base_state["source_blob_uri"] = source_blob_uri
            base_state["source_mime_type"] = source_mime_type
            grounding_mode = str(input_obj.get("grounding_mode") or "")
            if grounding_mode:
                base_state["grounding_mode"] = grounding_mode
        # For ai_model_answer (Intent=model_answer_fill) surface the
        # author's existing stem / options / rubric as TOP-LEVEL state
        # keys so the qgen_question agent's fill_mcq / fill_oe templates
        # can inject them via session.state placeholders
        # ({{author_stem}}, {{author_options}}, {{author_rubric}},
        # {{model_answer}}). Without this the agent has no author content
        # to fill against and falls through to a free-generation pass
        # (surfaced 2026-05-17 MCQ+OE smoke wave). Companion fix at
        # agents/qgen_adk_go (sibling agent, bug #4).
        if intent == "model_answer_fill":
            existing = input_obj.get("existing_question") or {}
            if isinstance(existing, dict):
                base_state["author_stem"] = str(existing.get("stem") or "")
                if question_type == "mcq":
                    options = existing.get("mcq_options") or []
                    if isinstance(options, list):
                        base_state["author_options"] = options
                elif question_type == "oe":
                    rubric = existing.get("oe_rubric") or []
                    if isinstance(rubric, list):
                        base_state["author_rubric"] = rubric
                    # Author MAY pre-supply a model answer; thread it
                    # when present so the agent can refine rather than
                    # re-draft.
                    model_answer = existing.get("model_answer")
                    if isinstance(model_answer, str) and model_answer:
                        base_state["model_answer"] = model_answer
    elif agent_role in CRITIC_ROLES:
        base_state.update(
            {
                "job_id": execution_id,
                "question_type": str(input_obj.get("question_type") or "mcq"),
                "attempt_index": int(input_obj.get("attempt_index") or 0),
                "max_attempts": int(input_obj.get("max_attempts") or 4),
                "prior_critic_notes": str(input_obj.get("prior_critic_notes") or ""),
                # The critique node threads the author's prompt as author_prompt
                # (the HTTP-era builder read "prompt", which the candidate payload
                # never carried, so the critic's author_prompt state key was
                # always blank and it relied on the raw JSON); accept both.
                "author_prompt": str(input_obj.get("author_prompt") or input_obj.get("prompt") or ""),
                "input_payload": json.dumps(input_obj),
            }
        )
        # Batch critique (CHO-2397, ADR-251 D2): surface set_mode so the ADK
        # Go critic composes the candidates-array prompt with the verdicts
        # echo contract. Stamped ONLY when the crew node sent it, so every
        # single-candidate critique stays byte-identical (absent => the
        # composer's SetMode==false path).
        if input_obj.get("set_mode"):
            base_state["set_mode"] = True
        # Authoring-metadata wire-through (2026-06-03): the critic already reads
        # subject_hint / cognitive_level_hint / difficulty_hint (critic.go:477-479
        # → metadataHints); stamp them from input_obj['metadata'] so the critic
        # judges WITH the author's intended axes. critique_node forwards metadata.
        stamp_metadata_hints(base_state, input_obj)
    return base_state


__all__ = [
    "CRITIC_ROLES",
    "LANE_ROLE_CRITIQUE",
    "LANE_ROLE_GENERATE",
    "LANE_ROLE_RENDER",
    "MERGE_ROLES",
    "QUESTION_ROLES",
    "ROLE_OE_EVALUATE",
    "ROLE_OE_MODERATE",
    "ROLE_QGEN_CRITIC",
    "ROLE_QGEN_QUESTION",
    "build_session_state",
    "current_w3c_trace_context",
    "decode_input",
    "stamp_metadata_hints",
    "stamp_set_plan",
]
