"""The two fog folds as ``LaneContract``s for the generic single-agent workflow
(ADR-254 D2 / D4 / D13; retire ``chora-fog-orchestrator``).

kg_exploration_workflow
    ``chora.consumption.concept_suggestion.requested.v1`` -> dispatch ``kg_explore``
    with ``exploration_json`` (the request object the Go ``kg_explorer`` agent
    decodes, ``internal/agent/prompt.go`` Request) + ``request_source`` (absent
    means ``learner_request``, the metered learner door; ``campaign_free_reveal``
    / ``atom_refresh`` are exempt at the gateway, D7) -> the agent answers
    ``{concepts, edges, model_id, prompt_version, prompt_source}``, already
    filtered fail-closed (ADR-244 D3) and soft-capped -> publish
    ``chora.consumption.concept_suggestion.emitted.v1`` with the FROZEN body,
    idempotency derived from the REQUEST key. A FAILED dispatch or a non-JSON
    answer still closes the loop with an EMPTY proposal (the old lane's deliberate
    behaviour: consumption treats empty as "nothing to suggest"), logged loudly.

companion_reflection_workflow
    ``chora.consumption.goal_knowledge.synthesis_requested.v1`` -> dispatch
    ``companion_chat`` ``turn_kind=reflect`` with ``familiar_id`` + ``reflection_json``
    ``{companion_name, goal_title, concepts_total, concepts_mastered,
    shaky_concepts, memories}`` (WP-A's binary contract, no conversation: reflect
    is stateless) -> one envelope ``{turn_kind, reply_text, ..., nothing_to_say?,
    model_id, prompt_version, prompt_source}`` -> publish
    ``chora.consumption.goal_knowledge.synthesized.v1``: ``content_hash`` and
    ``root_concept_id`` echoed VERBATIM (consumption's drift guard), a decline
    PUBLISHES with ``nothing_to_say=true`` and an empty text (CHO-2180),
    ``generated_by_model_id`` mandatory, ``MAX_SYNTHESIS_CHARS=600`` in lockstep
    with consumption + the agent. A request with no memories and no shaky
    concepts is SKIPPED (ACKed, nothing published: consumption's claim TTL
    expires; retrying emptiness re-reads the same emptiness). Anything that
    cannot be cached honestly (no model id, oversize, empty, a refusal, a FAILED
    dispatch, a non-JSON envelope) publishes NOTHING, loudly: a fabricated or
    blank reflection cached as the Companion's memory would be worse.

Both result events ride the kennel's outbox with ``source_service =
chora-ai-kernel-orchestrator`` (consumption requires the field non-empty and
never compares it, verified at G0).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Any

from chora_ai_kernel_orchestrator.orchestrators.single_agent_workflow import (
    STATUS_OK,
    DispatchSpec,
    LaneContract,
    ResultEvent,
    Skipped,
    envelope_scope,
    loads_json_object,
)

logger = logging.getLogger(__name__)

# ---- kg_exploration --------------------------------------------------------
TOPIC_CONCEPT_SUGGESTION_EMITTED = "chora.consumption.concept_suggestion.emitted.v1"
EVENT_TYPE_CONCEPT_SUGGESTION_EMITTED = "consumption.concept_suggestion.emitted"
ROLE_KG_EXPLORE = "kg_explore"
SOURCE_LEARNER_REQUEST = "learner_request"
MAX_CONCEPTS = 6
MAX_EDGES = 6
_VALID_EDGE_CLASSES = ("hierarchy", "lateral")
# The Go ``kg_explorer`` Request struct (internal/agent/prompt.go), verbatim.
_EXPLORATION_KEYS = (
    "focal_title",
    "focal_atom_refs",
    "existing_concepts",
    "map_theme",
    "atom_catalogue",
    "request_source",
    "sub_goal",
    "goal_title",
    "ancestors",
    "weakness",
)

# ---- companion_reflection --------------------------------------------------
TOPIC_GOAL_KNOWLEDGE_SYNTHESIZED = "chora.consumption.goal_knowledge.synthesized.v1"
EVENT_TYPE_GOAL_KNOWLEDGE_SYNTHESIZED = "consumption.goal_knowledge.synthesized"
ROLE_COMPANION_CHAT = "companion_chat"
TURN_KIND_REFLECT = "reflect"
# Mirror of chora-consumption familiar_goal_knowledge.MaxSynthesisChars AND the
# chat binary's MaxSynthesisChars (600). Keep the three in lockstep.
MAX_SYNTHESIS_CHARS = 600
_REFLECTION_REQUIRED = ("tenant_id", "gcid", "familiar_id", "goal_id", "content_hash")


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


# The envelope reader and the answer parser live on the engine (one parser for
# every lane); these aliases keep this module's call sites unchanged.
_envelope_scope = envelope_scope
_loads_object = loads_json_object


def _result_key(state: dict[str, Any]) -> str:
    """Idempotency derived from the REQUEST: a redelivery dedupes (ADR-254 D4)."""
    return state.get("idempotency_key") or state["event_id"]


# --------------------------------------------------------------------------- #
# kg_exploration
# --------------------------------------------------------------------------- #


def _kg_decode(body: dict[str, Any], attrs: dict[str, str]) -> dict[str, Any] | Skipped:
    scope = _envelope_scope(body, attrs, gcid_body_key="learner_gcid")
    if not scope["tenant_id"] or not scope["gcid"]:
        # An unscoped request is a producer bug: do not dispatch.
        raise ValueError("concept_suggestion: request missing tenant_id/gcid")
    if not scope["event_id"]:
        raise ValueError("concept_suggestion: request missing event_id (the run identity)")
    return {**scope, "request": dict(body)}


def _kg_dispatch(state: dict[str, Any]) -> DispatchSpec:
    request = state["request"]
    source = str(request.get("request_source") or "").strip().lower() or SOURCE_LEARNER_REQUEST
    exploration = {k: request.get(k) for k in _EXPLORATION_KEYS if k in request}
    exploration["request_source"] = source
    return DispatchSpec(
        execution_id=f"{state['event_id']}:explore",
        input_payload={
            "exploration_json": json.dumps(exploration, separators=(",", ":"), ensure_ascii=False),
            "request_source": source,
            # read back out by the executor for the ENVELOPE
            "gcid": state["gcid"],
            "traceparent": state.get("traceparent", ""),
            "tracestate": state.get("tracestate", ""),
        },
        workflow_id=state["event_id"],
    )


def _normalise_concepts(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        refs = [str(a).strip() for a in (entry.get("atom_refs") or []) if str(a).strip()]
        out.append(
            {"title": title[:200], "rationale": str(entry.get("rationale") or "").strip()[:200], "atom_refs": refs}
        )
        if len(out) >= MAX_CONCEPTS:
            break
    return out


def _normalise_edges(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        src = str(entry.get("source_concept_id") or "").strip()
        tgt = str(entry.get("target_concept_id") or "").strip()
        cls = str(entry.get("edge_class") or "").strip()
        if not src or not tgt or src == tgt or cls not in _VALID_EDGE_CLASSES:
            continue
        out.append(
            {
                "source_concept_id": src,
                "target_concept_id": tgt,
                "edge_class": cls,
                "rationale": str(entry.get("rationale") or "").strip()[:200],
            }
        )
        if len(out) >= MAX_EDGES:
            break
    return out


def _kg_result(state: dict[str, Any], completion: dict[str, Any]) -> ResultEvent | None:
    request = state["request"]
    concepts: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    model_used = ""
    if completion.get("status") != STATUS_OK:
        logger.error(
            "kg_exploration.dispatch_failed_empty_proposal",
            extra={
                "event_id": state["event_id"],
                "tenant_id": state["tenant_id"],
                "error": str(completion.get("error_message", "") or "")[:200],
            },
        )
    else:
        envelope = _loads_object(str(completion.get("output_payload", "") or ""))
        if envelope is None:
            logger.warning(
                "kg_exploration.non_json_answer_empty_proposal",
                extra={"event_id": state["event_id"], "tenant_id": state["tenant_id"]},
            )
        else:
            concepts = _normalise_concepts(envelope.get("concepts"))
            edges = _normalise_edges(envelope.get("edges"))
            model_used = str(envelope.get("model_id") or "")
    body = {
        "tenant_id": state["tenant_id"],
        "learner_gcid": state["gcid"],
        "familiar_id": str(request.get("familiar_id", "") or ""),
        "map_theme": str(request.get("map_theme", "") or ""),
        "focal_concept_id": str(request.get("focal_concept_id", "") or ""),
        "model_used": model_used,
        # run correlation = the originating requested event id (ADR-197 stamp).
        "run_id": state["event_id"],
        "concepts": concepts,
        "edges": edges,
    }
    return ResultEvent(
        topic=TOPIC_CONCEPT_SUGGESTION_EMITTED,
        event_type=EVENT_TYPE_CONCEPT_SUGGESTION_EMITTED,
        idempotency_key=_result_key(state),
        body=body,
        tenant_id=state["tenant_id"],
        gcid=state["gcid"],
        workflow_id=state["event_id"],
        traceparent=state.get("traceparent", ""),
        tracestate=state.get("tracestate", ""),
    )


def kg_exploration_contract() -> LaneContract:
    return LaneContract(
        name="kg_exploration",
        role=ROLE_KG_EXPLORE,
        request_kind="concept_suggestion.requested",
        decode=_kg_decode,
        dispatch=_kg_dispatch,
        result=_kg_result,
    )


# --------------------------------------------------------------------------- #
# companion_reflection
# --------------------------------------------------------------------------- #


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _reflection_decode(body: dict[str, Any], attrs: dict[str, str]) -> dict[str, Any] | Skipped:
    scope = _envelope_scope(body, attrs, gcid_body_key="learner_gcid")
    request = {
        "familiar_id": str(body.get("familiar_id", "") or ""),
        "familiar_name": str(body.get("familiar_name", "") or ""),
        "goal_id": str(body.get("goal_id", "") or ""),
        "goal_title": str(body.get("goal_title", "") or ""),
        # nullable in the domain: "" is a legitimate value, echoed back as-is
        "root_concept_id": str(body.get("root_concept_id", "") or ""),
        "concepts_total": _as_int(body.get("concepts_total")),
        "concepts_mastered": _as_int(body.get("concepts_mastered")),
        "shaky_concepts": [c for c in (body.get("shaky_concepts") or []) if isinstance(c, dict)],
        "memories": [m for m in (body.get("memories") or []) if isinstance(m, dict)],
        "content_hash": str(body.get("content_hash", "") or ""),
        "prompt_version": str(body.get("prompt_version", "") or ""),
        "requested_at": str(body.get("requested_at", "") or ""),
        "request_source": str(body.get("request_source", "") or ""),
    }
    merged = {**scope, **{k: request[k] for k in ("familiar_id", "goal_id", "content_hash")}}
    missing = [f for f in _REFLECTION_REQUIRED if not str(merged.get(f, "") or "").strip()]
    if missing:
        # Every one of these is needed to write the cache row: a completion
        # consumption refuses is not worth a model call.
        raise ValueError(
            f"goal_knowledge: request missing required field(s) {', '.join(missing)}; "
            "refusing to synthesise a reflection consumption cannot write"
        )
    if not scope["event_id"]:
        raise ValueError("goal_knowledge: request missing event_id (the run identity)")
    if not (request["memories"] or request["shaky_concepts"]):
        # Nothing to ground a reflection in: the Companion has nothing honest to
        # say and a model call could only invent one. Not an error: retrying
        # emptiness re-reads the same emptiness; consumption's claim expires.
        return Skipped("no_signal")
    return {**scope, "request": request}


def _reflection_dispatch(state: dict[str, Any]) -> DispatchSpec:
    request = state["request"]
    reflection = {
        "companion_name": request["familiar_name"],
        "goal_title": request["goal_title"],
        "concepts_total": request["concepts_total"],
        "concepts_mastered": request["concepts_mastered"],
        "shaky_concepts": request["shaky_concepts"],
        "memories": request["memories"],
    }
    return DispatchSpec(
        execution_id=f"{state['event_id']}:reflect",
        input_payload={
            "turn_kind": TURN_KIND_REFLECT,
            "familiar_id": request["familiar_id"],
            "reflection_json": json.dumps(reflection, separators=(",", ":"), ensure_ascii=False),
            "gcid": state["gcid"],
            "traceparent": state.get("traceparent", ""),
            "tracestate": state.get("tracestate", ""),
        },
        workflow_id=state["event_id"],
    )


def _strip_wrapping(raw: str) -> str:
    """Strip a markdown fence and/or wrapping quotes the model added anyway;
    deterministic unwrapping only (no heuristic that could cut into the voice)."""
    s = (raw or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1 :] if nl != -1 else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


def _reflection_result(state: dict[str, Any], completion: dict[str, Any]) -> ResultEvent | None:
    request = state["request"]
    log_scope = {
        "event_id": state["event_id"],
        "tenant_id": state["tenant_id"],
        "goal_id": request["goal_id"],
        "familiar_id": request["familiar_id"],
    }
    if completion.get("status") != STATUS_OK:
        logger.error(
            "companion_reflection.dispatch_failed_nothing_published",
            extra={**log_scope, "error": str(completion.get("error_message", "") or "")[:200]},
        )
        return None
    envelope = _loads_object(str(completion.get("output_payload", "") or ""))
    if envelope is None:
        logger.error("companion_reflection.non_json_envelope_nothing_published", extra=log_scope)
        return None
    model_id = str(envelope.get("model_id") or "").strip()
    if not model_id:
        # consumption's RecordSynthesis / RecordNoReflection refuse an
        # unattributable decision stamp: publishing would only be refused.
        logger.error("companion_reflection.no_model_id_nothing_published", extra=log_scope)
        return None
    if str(envelope.get("refusal_reason") or "").strip():
        logger.warning(
            "companion_reflection.refused_nothing_published",
            extra={**log_scope, "refusal_reason": str(envelope.get("refusal_reason"))[:120]},
        )
        return None
    prompt_version = str(envelope.get("prompt_version") or "")
    if bool(envelope.get("nothing_to_say")):
        text, nothing_to_say = "", True
    else:
        text = _strip_wrapping(str(envelope.get("reply_text") or ""))
        if not text:
            logger.error("companion_reflection.empty_reflection_nothing_published", extra=log_scope)
            return None
        if len(text) > MAX_SYNTHESIS_CHARS:
            logger.error(
                "companion_reflection.oversize_reflection_nothing_published",
                extra={**log_scope, "chars": len(text), "contract": MAX_SYNTHESIS_CHARS},
            )
            return None
        nothing_to_say = False
    body = {
        "tenant_id": state["tenant_id"],
        "learner_gcid": state["gcid"],
        "familiar_id": request["familiar_id"],
        "goal_id": request["goal_id"],
        "synthesis_text": text,
        "generated_by_run_id": state["event_id"],
        "generated_by_model_id": model_id,
        "prompt_version": prompt_version,
        "nothing_to_say": nothing_to_say,
        # ECHOED VERBATIM: consumption re-checks both to refuse a reflection
        # whose world moved (ADR-214 re-root). Never recomputed here.
        "content_hash": request["content_hash"],
        "root_concept_id": request["root_concept_id"],
        "generated_at": _utc_now_iso(),
    }
    return ResultEvent(
        topic=TOPIC_GOAL_KNOWLEDGE_SYNTHESIZED,
        event_type=EVENT_TYPE_GOAL_KNOWLEDGE_SYNTHESIZED,
        idempotency_key=_result_key(state),
        body=body,
        tenant_id=state["tenant_id"],
        gcid=state["gcid"],
        workflow_id=state["event_id"],
        traceparent=state.get("traceparent", ""),
        tracestate=state.get("tracestate", ""),
    )


def companion_reflection_contract() -> LaneContract:
    return LaneContract(
        name="companion_reflection",
        role=ROLE_COMPANION_CHAT,
        request_kind="goal_knowledge.synthesis_requested",
        decode=_reflection_decode,
        dispatch=_reflection_dispatch,
        result=_reflection_result,
    )


__all__ = [
    "EVENT_TYPE_CONCEPT_SUGGESTION_EMITTED",
    "EVENT_TYPE_GOAL_KNOWLEDGE_SYNTHESIZED",
    "MAX_CONCEPTS",
    "MAX_EDGES",
    "MAX_SYNTHESIS_CHARS",
    "ROLE_COMPANION_CHAT",
    "ROLE_KG_EXPLORE",
    "TOPIC_CONCEPT_SUGGESTION_EMITTED",
    "TOPIC_GOAL_KNOWLEDGE_SYNTHESIZED",
    "TURN_KIND_REFLECT",
    "companion_reflection_contract",
    "kg_exploration_contract",
]
