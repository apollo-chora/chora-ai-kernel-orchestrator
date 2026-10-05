"""The caller-facing consumption lanes as ``LaneContract``s (ADR-254 D1/D4/D8).

companion_turn_workflow
    ``chora.consumption.companion_turn.requested.v1`` -> dispatch
    ``companion_chat`` with the request body passed through **VERBATIM** ->
    ``chora.consumption.companion_turn.completed.v1`` keyed on ``turn_id``.

    Consumption keeps the SSE stream to the browser, writes the turn row and
    this request into its outbox in ONE transaction, answers the browser
    ``accepted`` the moment the request is durable, and then WAITS for the
    completion. That waiting learner is what makes this lane different from the
    two fog folds in three ways worth stating, because each one is a silent
    failure if it regresses:

    1. **VERBATIM pass-through.** ``agentdispatch.Request.SessionState()``
       merges every ``input_payload`` key by design, and
       ``instancedispatch.stampIdentityFromState`` reads ``prompt_overrides_json``
       and ``familiar_config`` straight out of that state. Both fold decodes
       hand-pick their fields; a decode written that way here drops the prompt
       overrides consumption resolved and the learner gets a flatter voice with
       no error anywhere. It also stringifies ``growth_stage``, which the
       growth-stage plugin refuses PERMANENTLY. So the body is copied, not
       rebuilt, and the only keys added are ``turn_kind`` plus the envelope
       fields ``PubSubAgentExecutor`` reads back out to stamp the dispatch.
    2. **Always publish.** A reflection that cannot be cached honestly publishes
       nothing and the caller's claim TTL closes the loop. Here nothing is the
       one answer that cannot be given: it leaves a learner on a spinner until
       consumption's turn deadline. Every terminal path emits a result, FAILED
       with a machine ``error_code`` when it must.
    3. **turn_id round-trips UNCHANGED and is the workflow id.** It is a UUID by
       ``turn_id UUID PRIMARY KEY`` (migration 0111) and, since dbd8082d0, by
       the consumption edge refusing a malformed client-minted id. This lane
       refuses anything else rather than tolerating it: the engine's
       ``workflow_uuid`` would otherwise derive a stable UUIDv5 and the O+ trail
       would key on an id consumption never minted. A tolerant reader is a
       reader that cannot detect the thing it tolerates.

    Wire keys are deliberately PRE-RENAME (``familiar_id``, ``familiar_config``,
    ``familiar_memory``): ADR-254 D6 cuts them in a coordinated change, and a
    test on the consumption side pins the ``companion_*`` twins absent so nobody
    finishes the rename here by accident. The one place the rename HAS happened
    is the result body, whose contract field is ``companion_id``.

    The topic carries no schema binding (JSON-wire, ``schema: null``), so
    nothing on the bus rejects a shape this lane cannot read. The decode is the
    only gate, which is why it fails loud rather than filling in blanks.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from typing import Any

from chora_ai_kernel_orchestrator.orchestrators.prose_narrative import prose_narrative
from chora_ai_kernel_orchestrator.orchestrators.single_agent_workflow import (
    STATUS_OK,
    DispatchSpec,
    LaneContract,
    ResultEvent,
    envelope_scope,
    loads_json_object,
)

logger = logging.getLogger(__name__)

TOPIC_COMPANION_TURN_COMPLETED = "chora.consumption.companion_turn.completed.v1"
EVENT_TYPE_COMPANION_TURN_COMPLETED = "consumption.companion_turn.completed"
ROLE_COMPANION_CHAT = "companion_chat"
LANE_COMPANION_TURN = "companion_turn"
REQUEST_KIND_COMPANION_TURN = "companion_turn.requested"

# The wire status vocabulary of ADR-254 D4 (CompanionTurnCompleted.status).
WIRE_OK = "OK"
WIRE_FAILED = "FAILED"
WIRE_REJECTED = "REJECTED"

# Machine tokens for CompanionTurnCompleted.error_code. Never empty when the
# status is not OK: consumption surfaces the code, so "it failed somehow" would
# reach a learner as an unexplained error.
ERROR_CODE_AGENT_FAILED = "agent_failed"
ERROR_CODE_AGENT_CONTRACT = "agent_contract_violation"
ERROR_CODE_NO_MODEL_ID = "agent_no_model_id"
ERROR_CODE_EMPTY_REPLY = "agent_empty_reply"
ERROR_CODE_TENANT_CAP = "tenant_in_flight_cap"

# "typed" for this lane: consumption stamps it on every kind that rides here
# (typed / skill / ceremony / ritual / greeting all publish turn_kind=typed on
# the wire), and the kennel stamps it too so the dispatch is self-describing.
TURN_KIND_TYPED = "typed"

# The proto marks these REQUIRED and the agent genuinely cannot run without
# them: conversation_id keys the Postgres-backed ADK session, familiar_id names
# the instance whose config composes the prompt. `message` is deliberately NOT
# here: the producer sets it unconditionally but a greeting turn can carry an
# empty one, and NACKing that to the DLQ would break greetings.
_REQUIRED_BODY_FIELDS = ("conversation_id", "familiar_id")


def _utc_now_iso() -> str:
    """RFC3339 with a Z, matching what the producer emits on the same wire."""
    return _dt.datetime.now(tz=_dt.UTC).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #


def _turn_id(state: dict[str, Any]) -> str:
    """The turn id, read back out of the request rather than off a top-level
    state key.

    ``SingleAgentState`` is the LangGraph state SCHEMA, and LangGraph filters
    state to the keys that schema declares. A lane-specific key parked at the
    top level is silently dropped between ``decode`` and ``dispatch``, which a
    unit test calling ``contract.dispatch(...)`` directly cannot see because it
    never crosses the engine. ``request`` is a declared key and the body is
    carried verbatim, so the id validated in decode is the id read here.
    """
    return str(state["request"]["turn_id"])


def _turn_decode(body: dict[str, Any], attrs: dict[str, str]) -> dict[str, Any]:
    scope = envelope_scope(body, attrs, gcid_body_key="gcid")
    if not scope["tenant_id"] or not scope["gcid"]:
        # An unscoped turn cannot be metered or attributed: do not dispatch it.
        raise ValueError("companion_turn: request missing tenant_id/gcid")
    if not scope["event_id"]:
        raise ValueError("companion_turn: request missing event_id (the run identity)")

    turn_id = str(body.get("turn_id", "") or "").strip()
    if not turn_id:
        raise ValueError(
            "companion_turn: request carries no turn_id; consumption's waiter polls "
            "its turn store by that id, so a turn without one can never be answered"
        )
    try:
        uuid.UUID(turn_id)
    except ValueError as exc:
        raise ValueError(
            f"companion_turn: turn_id {turn_id!r} is not a UUID. Refusing rather than "
            "deriving one: the engine would key the outbox on a stable UUIDv5 that "
            "consumption never minted, and the learner's turn would never be matched"
        ) from exc

    key = scope["idempotency_key"]
    if key and key != turn_id:
        raise ValueError(
            f"companion_turn: envelope idempotency_key {key!r} disagrees with body "
            f"turn_id {turn_id!r}; the contract says they are the same value, so a "
            "disagreement is a producer bug rather than something to pick a winner for"
        )

    missing = [f for f in _REQUIRED_BODY_FIELDS if not str(body.get(f, "") or "").strip()]
    if missing:
        raise ValueError(f"companion_turn: request missing required field(s) {', '.join(missing)}")

    # The body is COPIED, never rebuilt. See point 1 in the module docstring.
    # turn_id is NOT carried as a top-level key: the engine's state schema
    # would drop it. It rides inside the verbatim request and _turn_id reads it.
    return {**scope, "request": dict(body)}


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def _turn_dispatch(state: dict[str, Any]) -> DispatchSpec:
    turn_id = _turn_id(state)
    payload = dict(state["request"])
    payload["turn_kind"] = TURN_KIND_TYPED
    # The envelope is authoritative for identity and trace; PubSubAgentExecutor
    # reads these three back out of input_payload to stamp the dispatch, and the
    # Go SessionState() then drops them in favour of the envelope it carried.
    payload["gcid"] = state["gcid"]
    payload["traceparent"] = state.get("traceparent", "")
    payload["tracestate"] = state.get("tracestate", "")
    return DispatchSpec(
        # Stable per turn: a redelivery rebuilds the SAME request and collides
        # with the queued outbox row instead of raising a second dispatch
        # against a conversation-scoped ADK session that would replay the turn.
        execution_id=f"{turn_id}:turn",
        input_payload=payload,
        workflow_id=turn_id,
    )


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #


def _entries(
    raw: Any, *, required: str, optional: tuple[str, ...], event: str, legacy_event: str, log_scope: dict[str, Any]
) -> list[dict[str, str]]:
    """Normalise one agent-envelope list into the objects consumption decodes.

    Consumption unmarshals ``[]TurnGrounding`` / ``[]TurnToolCall``; passing a
    bare string through makes it fail to decode the whole result, so the turn
    never completes. TWO producer shapes are accepted deliberately, because the
    kennel and the companion_chat agent are SEPARATE deployments and a rollout
    puts one ahead of the other:

    * ``str``  -- the historical shape, lifted into ``{required: value}``.
    * ``dict`` -- the richer shape, VALIDATED then passed through, so provenance
      a string cannot express (``revision_id``) survives to the learner.

    A dict is never trusted: one missing the required key is a malformed entry,
    not a citation, and is dropped LOUDLY rather than forwarded half-formed.

    ⚠ RETIREMENT CONDITION, because this WILL be proposed as dead code once the
    agent stops emitting strings. **The condition is "no reachable rollback
    target emits strings", NOT "no strings observed".** Those differ: the string
    branch is a claim about REACHABILITY, and reachability is a fact about the
    deploy policy and the registry rather than about what spoke recently. Under
    R26 the sanctioned recovery path is a rollback to an already attested
    digest, so every pre-2026-08-23 companion_chat image is a live
    string-emitting producer for as long as it can be rolled back to. **This
    branch is therefore the rollback safety net for the kennel/agent pair**, and
    deleting it converts an agent rollback into SILENT loss of grounding and
    tool_calls. Recorded as a deliberate permanent compat shim (G25).
    """
    out: list[dict[str, str]] = []
    legacy = 0
    for entry in raw or []:
        if isinstance(entry, str):
            # ⚠ THE LEGACY PRODUCER SHAPE. Counted and reported below, never
            # silent: a fail-quiet compatibility path cannot be reasoned about,
            # because "no strings arrived" and "strings arrived and were quietly
            # lifted" look identical from outside.
            value = entry.strip()
            if value:
                out.append({required: value})
                legacy += 1
            continue
        if isinstance(entry, dict):
            raw_value = entry.get(required)
            if not isinstance(raw_value, str) or not raw_value.strip():
                logger.error(f"{event}_missing_{required}", extra={**log_scope, "keys": sorted(map(str, entry))})
                continue
            ref = {required: raw_value.strip()}
            for key in optional:
                extra_value = entry.get(key)
                if isinstance(extra_value, str) and extra_value.strip():
                    ref[key] = extra_value.strip()
            out.append(ref)
            continue
        logger.error(event, extra={**log_scope, "got": type(entry).__name__})
    if legacy:
        # INFO, not a warning: lifting is CORRECT behaviour, and this is the
        # only evidence that the legacy branch is still carrying traffic.
        # Verified 2026-08-23 that this surfaces: this logger inherits the root
        # stderr handler (effective level INFO, propagate True, no filters), and
        # sibling modules' INFO lines appear in the pod log.
        logger.info(legacy_event, extra={**log_scope, "count": legacy})
    return out


def _grounding(raw: Any, *, log_scope: dict[str, Any]) -> list[dict[str, str]]:
    """Agent ``grounding`` -> consumption ``[]TurnGrounding``."""
    return _entries(
        raw,
        required="atom_id",
        optional=("revision_id", "title", "snippet"),
        event="companion_turn.grounding_entry_not_a_string",
        legacy_event="companion_turn.grounding_legacy_string_lifted",
        log_scope=log_scope,
    )


def _tool_calls(raw: Any, *, log_scope: dict[str, Any]) -> list[dict[str, str]]:
    """Agent ``tool_calls`` -> consumption ``[]TurnToolCall``. Arguments never
    ride the caller-facing wire, so only the name and a summary cross."""
    return _entries(
        raw,
        required="name",
        optional=("summary",),
        event="companion_turn.tool_call_entry_not_a_string",
        legacy_event="companion_turn.tool_call_legacy_string_lifted",
        log_scope=log_scope,
    )


def _completed_event(
    state: dict[str, Any],
    *,
    status: str,
    error_code: str = "",
    model_id: str = "",
    reply_text: str = "",
    grounding: list[dict[str, str]] | None = None,
    tool_calls: list[dict[str, str]] | None = None,
    refusal_reason: str = "",
    prompt_version: str = "",
    prompt_source: str = "",
    prompt_hash: str = "",
) -> ResultEvent:
    """One CompanionTurnCompleted, whatever happened. Keys are the proto field
    names; ``companion_id`` is the post-rename spelling of the request's
    ``familiar_id``."""
    request = state["request"]
    turn_id = _turn_id(state)
    body = {
        "turn_id": turn_id,
        "conversation_id": str(request.get("conversation_id", "") or ""),
        "companion_id": str(request.get("familiar_id", "") or ""),
        "status": status,
        "error_code": error_code,
        # The kennel workflow that served the turn, for the O+ trail. It is the
        # same value the dispatch and the outbox row key on.
        "workflow_id": turn_id,
        "generated_by_model_id": model_id,
        "reply_text": reply_text,
        "grounding": grounding or [],
        "tool_calls": tool_calls or [],
        "refusal_reason": refusal_reason,
        # N9: the ADR-197 prompt provenance the AGENT stamps on its envelope.
        # This lane used to drop both, so consumption's parser (waiting since
        # CHO-2368) was never fed and every ritual step stamped an empty
        # version. Forwarding two strings the agent already produced keeps the
        # kennel deterministic (ADR-254 D5): no model call, no branch on
        # content.
        #
        # ALWAYS PRESENT, even on a FAILED terminal and even when the agent sent
        # nothing, so a consumer never has to tell "this turn had no version"
        # apart from "this event shape is older". Empty is the honest value for
        # an agent image that predates the stamp; an invented version would be a
        # false provenance claim indistinguishable from a real one.
        "prompt_version": prompt_version,
        "prompt_source": prompt_source,
        # The SHA-256 of the instruction the agent actually composed and ran.
        # FORWARDED, never computed here: this lane does not hold the prompt,
        # and a hash taken anywhere but the compose site is a true hash of the
        # wrong string, which is indistinguishable from a real one.
        "prompt_hash": prompt_hash,
        "completed_at": _utc_now_iso(),
    }
    return ResultEvent(
        topic=TOPIC_COMPANION_TURN_COMPLETED,
        event_type=EVENT_TYPE_COMPANION_TURN_COMPLETED,
        # turn_id is the caller's business key: a redelivered completion dedupes
        # and consumption replays its stored terminal result.
        idempotency_key=turn_id,
        body=body,
        tenant_id=state["tenant_id"],
        gcid=state["gcid"],
        workflow_id=turn_id,
        traceparent=state.get("traceparent", ""),
        tracestate=state.get("tracestate", ""),
    )


def _turn_result(state: dict[str, Any], completion: dict[str, Any]) -> ResultEvent:
    log_scope = {"event_id": state["event_id"], "tenant_id": state["tenant_id"], "turn_id": _turn_id(state)}

    if completion.get("status") != STATUS_OK:
        logger.error(
            "companion_turn.dispatch_failed",
            extra={**log_scope, "error": str(completion.get("error_message", "") or "")[:200]},
        )
        return _completed_event(state, status=WIRE_FAILED, error_code=ERROR_CODE_AGENT_FAILED)

    envelope = loads_json_object(str(completion.get("output_payload", "") or ""))
    if envelope is None:
        logger.error("companion_turn.non_json_envelope", extra=log_scope)
        return _completed_event(state, status=WIRE_FAILED, error_code=ERROR_CODE_AGENT_CONTRACT)

    model_id = str(envelope.get("model_id") or "").strip()
    if not model_id:
        # generated_by_model_id is mandatory on OK. An unattributable reply is a
        # governance hole (which model said this to a learner?), so it degrades
        # to FAILED rather than reaching the learner with a blank attribution.
        logger.error("companion_turn.no_model_id", extra=log_scope)
        return _completed_event(state, status=WIRE_FAILED, error_code=ERROR_CODE_NO_MODEL_ID)

    reply_text = str(envelope.get("reply_text") or "")
    if not reply_text.strip():
        logger.error("companion_turn.empty_reply", extra=log_scope)
        return _completed_event(state, status=WIRE_FAILED, error_code=ERROR_CODE_EMPTY_REPLY, model_id=model_id)

    return _completed_event(
        state,
        status=WIRE_OK,
        model_id=model_id,
        reply_text=reply_text,
        grounding=_grounding(envelope.get("grounding"), log_scope=log_scope),
        tool_calls=_tool_calls(envelope.get("tool_calls"), log_scope=log_scope),
        # An honest decline is a SUCCESSFUL turn: the copy is in reply_text and
        # the reason names the fence that produced it (ADR-249).
        refusal_reason=str(envelope.get("refusal_reason") or "").strip(),
        # Absent stays empty. Unlike model_id above, a missing prompt version
        # does NOT degrade the turn to FAILED: an unattributable MODEL is a
        # governance hole, while a missing prompt version is an older agent
        # image, and refusing the learner's answer over it would be a far worse
        # outcome than an unstamped one.
        prompt_version=str(envelope.get("prompt_version") or "").strip(),
        prompt_source=str(envelope.get("prompt_source") or "").strip(),
        prompt_hash=str(envelope.get("prompt_hash") or "").strip(),
    )


def _turn_rejected(state: dict[str, Any]) -> ResultEvent:
    """The per-tenant in-flight cap answer (ADR-254 D5). Always present, so
    turning the cap on is a config change and never a silent drop."""
    logger.warning(
        "companion_turn.rejected_tenant_in_flight_cap",
        extra={"tenant_id": state["tenant_id"], "turn_id": _turn_id(state)},
    )
    return _completed_event(state, status=WIRE_REJECTED, error_code=ERROR_CODE_TENANT_CAP)


def companion_turn_contract(*, tenant_inflight_cap: int | None = None) -> LaneContract:
    """The companion_turn lane. ``tenant_inflight_cap`` is off by default and
    resolved from the environment by the wiring, never inline here."""
    return LaneContract(
        name=LANE_COMPANION_TURN,
        role=ROLE_COMPANION_CHAT,
        request_kind=REQUEST_KIND_COMPANION_TURN,
        decode=_turn_decode,
        dispatch=_turn_dispatch,
        result=_turn_result,
        rejected=_turn_rejected,
        tenant_inflight_cap=tenant_inflight_cap,
    )


# --------------------------------------------------------------------------- #
# dose_recommendation (ADR-254 D1/D4, gate G-b)
# --------------------------------------------------------------------------- #
#
# ``chora.consumption.dose_recommendation.requested.v1`` -> dispatch ``recommend``
# -> ``chora.consumption.dose_recommendation.completed.v1``, keyed on
# ``dose_request_id``.
#
# Consumption's BusDoseRecommender writes a turn row (kind ``dose``, lane
# ``dose_recommendation``) and this request into its outbox in ONE transaction,
# then BLOCKS its handler on the turn store until the completion lands or its
# 30s deadline expires. An error there degrades to templated copy rather than a
# 5xx, so a lane that publishes nothing costs the learner the AI dose silently.
#
# ⚠ THIS LANE DOES NOT PASS THE BODY VERBATIM, AND THAT IS THE ONE PLACE IT MUST
# DIFFER FROM companion_turn. Copying the verbatim pass-through here is a SILENT
# break, and it is the green-on-both-sides shape: consumption publishes
# ``context_json`` as ONE JSON STRING holding
# ``{mana_tier, learner_persona, user_prompt, topic_hint, candidates[]}``, while
# the recommender reads DISCRETE session-state keys (``tenant_id``,
# ``user_gcid`` falling back to ``learner_gcid``, ``topic_hint``,
# ``learner_persona``, ``atom_candidates``). Passed verbatim, ``context_json``
# lands in state as one opaque string the agent never opens:
# ``candidatesFromState`` returns (nil,false), the tool falls back to its own
# searcher, THE DOSE STILL RETURNS ATOMS, nothing errors anywhere, and
# consumption's composed candidates plus the learner's persona are discarded.
# So the lane UNPACKS, and ``atom_candidates`` is written as a JSON STRING
# because the tool json-unmarshals a string read out of state, not an array.
#
# ⚠ ``gcid`` AND ``user_gcid`` are BOTH stamped, at two different layers. The
# envelope keeps ``gcid`` because that is what PubSubAgentExecutor reads back to
# stamp the dispatch; state additionally carries ``user_gcid`` because that is
# the key the tool reads. Stamping only ``gcid`` makes the tool fall through to
# the MODEL's ``req.LearnerGCID``: a silent mis-attribution rather than a crash.

TOPIC_DOSE_RECOMMENDATION_COMPLETED = "chora.consumption.dose_recommendation.completed.v1"
EVENT_TYPE_DOSE_RECOMMENDATION_COMPLETED = "consumption.dose_recommendation.completed"
ROLE_RECOMMEND = "recommend"
LANE_DOSE_RECOMMENDATION = "dose_recommendation"
REQUEST_KIND_DOSE_RECOMMENDATION = "dose_recommendation.requested"

#: No atoms AND no usable rationale: the agent answered with nothing the learner
#: can see. Distinct from agent_empty_reply so the two lanes stay legible in O+.
ERROR_CODE_EMPTY_RECOMMENDATION = "agent_empty_recommendation"

#: Upper bound on picks carried to the caller. The dose composes a handful; an
#: unbounded list would let one bad generation write an arbitrarily large row.
MAX_DOSE_ATOMS = 20

#: ⚠ SWITCHABLE ARM (ruled by the owner via the coordinator, 2026-08-23).
#: True  = zero atoms with usable PROSE rationale publishes OK (an honest "nothing
#:         new today" is a legitimate answer and laundering it as FAILED destroys
#:         the one useful thing the agent produced).
#: False = zero atoms is always FAILED.
#: Held behind ONE name because the ruling is conditional on consumption's OK
#: path actually rendering an empty id list as words; WP-D owns that answer and
#: flipping this is a one-line change, not a rewrite.
OK_ON_EMPTY_ATOMS_WITH_PROSE = True

#: The keys the recommender actually reads out of session state, and the
#: context_json key each is unpacked from. Confirmed by grep against
#: recommender_adk_go, not assumed from the producer's doc comment.
_DOSE_CONTEXT_KEYS = ("learner_persona", "topic_hint", "mana_tier", "user_prompt")


def _dose_request_id(state: dict[str, Any]) -> str:
    """Read back out of the verbatim request, never off a top-level state key.

    Same reasoning as ``_turn_id``: ``SingleAgentState`` is the LangGraph state
    SCHEMA and undeclared top-level keys are filtered out between decode and
    dispatch, which a test calling ``contract.dispatch(...)`` by hand cannot see.
    """
    return str(state["request"]["dose_request_id"])


def _dose_decode(body: dict[str, Any], attrs: dict[str, str]) -> dict[str, Any]:
    scope = envelope_scope(body, attrs, gcid_body_key="gcid")
    if not scope["tenant_id"] or not scope["gcid"]:
        # Unscoped: cannot be metered, attributed or RLS-scoped. Do not dispatch.
        raise ValueError("dose_recommendation: request missing tenant_id/gcid")
    if not scope["event_id"]:
        raise ValueError("dose_recommendation: request missing event_id (the run identity)")

    dose_request_id = str(body.get("dose_request_id", "") or "").strip()
    if not dose_request_id:
        raise ValueError(
            "dose_recommendation: request carries no dose_request_id; consumption's "
            "waiter polls its turn store by that id, so a request without one can "
            "never be answered and the learner waits out the deadline"
        )
    try:
        uuid.UUID(dose_request_id)
    except ValueError as exc:
        raise ValueError(
            f"dose_recommendation: dose_request_id {dose_request_id!r} is not a UUID. "
            "Refusing rather than deriving one: the engine would key the outbox on a "
            "stable UUIDv5 consumption never minted, and the dose would never match"
        ) from exc

    key = scope["idempotency_key"]
    if key and key != dose_request_id:
        raise ValueError(
            f"dose_recommendation: envelope idempotency_key {key!r} disagrees with body "
            f"dose_request_id {dose_request_id!r}; the contract says they are the same "
            "value, so a disagreement is a producer bug rather than a tie to break"
        )

    return {**scope, "request": dict(body)}


def _dose_dispatch(state: dict[str, Any]) -> DispatchSpec:
    request = state["request"]
    dose_request_id = _dose_request_id(state)

    # UNPACK context_json. A malformed or absent context is NOT fatal: the agent
    # falls back to its searcher and still composes a dose. It is logged because
    # a silently empty context is the exact degradation this lane exists to stop.
    context = loads_json_object(str(request.get("context_json", "") or "")) or {}
    if not context:
        logger.warning(
            "dose_recommendation.context_json_unusable",
            extra={"event_id": state["event_id"], "tenant_id": state["tenant_id"], "dose_request_id": dose_request_id},
        )

    payload: dict[str, Any] = {
        "dose_request_id": dose_request_id,
        "goal_id": str(request.get("goal_id", "") or ""),
        "trigger": str(request.get("trigger", "") or ""),
        "companion_id": str(request.get("companion_id", "") or ""),
    }
    for key in _DOSE_CONTEXT_KEYS:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            payload[key] = value

    # ⚠ A JSON STRING, not the list: candidatesFromState unmarshals a string it
    # reads out of state. Handing it a list makes the read fail and the agent
    # silently falls back to its searcher, discarding consumption's ranking.
    candidates = context.get("candidates")
    if isinstance(candidates, list) and candidates:
        payload["atom_candidates"] = json.dumps(candidates, separators=(",", ":"), ensure_ascii=False)

    payload["gcid"] = state["gcid"]
    payload["user_gcid"] = state["gcid"]  # the key the TOOL reads; see the note above
    payload["tenant_id"] = state["tenant_id"]
    payload["traceparent"] = state.get("traceparent", "")
    payload["tracestate"] = state.get("tracestate", "")

    return DispatchSpec(
        # Stable per request: a redelivery rebuilds the SAME dispatch and
        # collides with the queued outbox row instead of raising a second run.
        execution_id=f"{dose_request_id}:dose",
        input_payload=payload,
        workflow_id=dose_request_id,
    )


def _recommended_atom_ids(raw: Any, *, log_scope: dict[str, Any]) -> list[str]:
    """Project the agent's rich per-atom shape down to the ordered id list the
    caller consumes.

    ⚠ ARRAY ORDER IS THE RECOMMENDATION ORDER AND IT IS CARRIED, NEVER
    RECOMPUTED. ``rank_score`` is disclosure only and is legitimately ZERO for
    every atom on the candidate path, because consumption already ranked the
    candidates by order. Sorting on it here would flatten the dose to an
    arbitrary order while looking like an improvement.
    """
    out: list[str] = []
    seen: set[str] = set()
    for entry in raw or []:
        if not isinstance(entry, dict):
            logger.error(
                "dose_recommendation.atom_entry_not_an_object", extra={**log_scope, "got": type(entry).__name__}
            )
            continue
        atom_id = str(entry.get("atom_id") or "").strip()
        if not atom_id or atom_id in seen:
            continue
        seen.add(atom_id)
        out.append(atom_id)
    if len(out) > MAX_DOSE_ATOMS:
        # NO SILENT CAPS. A dose truncated from 25 to 20 must not read as one
        # that genuinely had 20; the non-dict branch above already logs, so a
        # quiet truncation is the only bounded outcome that looks complete.
        logger.warning(
            "dose_recommendation.atom_cap_truncated",
            extra={
                **log_scope,
                "cap": MAX_DOSE_ATOMS,
                "returned": MAX_DOSE_ATOMS,
                "dropped": len(out) - MAX_DOSE_ATOMS,
            },
        )
        out = out[:MAX_DOSE_ATOMS]
    return out


def _dose_completed_event(
    state: dict[str, Any],
    *,
    status: str,
    error_code: str = "",
    model_id: str = "",
    atom_ids: list[str] | None = None,
    rationale: str = "",
) -> ResultEvent:
    """One DoseRecommendationCompleted, whatever happened. Keys are the proto
    field names (proto/events/consumption/dose_recommendation.proto)."""
    request = state["request"]
    dose_request_id = _dose_request_id(state)
    body = {
        "dose_request_id": dose_request_id,
        "goal_id": str(request.get("goal_id", "") or ""),
        "status": status,
        "error_code": error_code,
        "workflow_id": dose_request_id,
        "generated_by_model_id": model_id,
        "recommended_atom_ids": atom_ids or [],
        "rationale": rationale,
        "completed_at": _utc_now_iso(),
    }
    return ResultEvent(
        topic=TOPIC_DOSE_RECOMMENDATION_COMPLETED,
        event_type=EVENT_TYPE_DOSE_RECOMMENDATION_COMPLETED,
        # dose_request_id is the caller's business key: a redelivered completion
        # dedupes and consumption replays its stored terminal result.
        idempotency_key=dose_request_id,
        body=body,
        tenant_id=state["tenant_id"],
        gcid=state["gcid"],
        workflow_id=dose_request_id,
        traceparent=state.get("traceparent", ""),
        tracestate=state.get("tracestate", ""),
    )


def _dose_result(state: dict[str, Any], completion: dict[str, Any]) -> ResultEvent:
    log_scope = {
        "event_id": state["event_id"],
        "tenant_id": state["tenant_id"],
        "dose_request_id": _dose_request_id(state),
    }

    if completion.get("status") != STATUS_OK:
        logger.error(
            "dose_recommendation.dispatch_failed",
            extra={**log_scope, "error": str(completion.get("error_message", "") or "")[:200]},
        )
        return _dose_completed_event(state, status=WIRE_FAILED, error_code=ERROR_CODE_AGENT_FAILED)

    envelope = loads_json_object(str(completion.get("output_payload", "") or ""))
    if envelope is None:
        logger.error("dose_recommendation.non_json_envelope", extra=log_scope)
        return _dose_completed_event(state, status=WIRE_FAILED, error_code=ERROR_CODE_AGENT_CONTRACT)

    model_id = str(envelope.get("model_id") or "").strip()
    if not model_id:
        # generated_by_model_id is mandatory on OK: an unattributable
        # recommendation is a governance hole (which model chose these atoms?).
        logger.error("dose_recommendation.no_model_id", extra=log_scope)
        return _dose_completed_event(state, status=WIRE_FAILED, error_code=ERROR_CODE_NO_MODEL_ID)

    atom_ids = _recommended_atom_ids(envelope.get("atoms"), log_scope=log_scope)
    # ⚠ PROSE, not merely non-blank. Consumption's proseNarrative drops blank, a
    # leading brace or bracket, and anything json.Valid accepts (bare scalars
    # included), so a non-blank but JSON-shaped rationale would publish OK and
    # render the learner NOTHING. Ported, with shared vectors, in
    # orchestrators/prose_narrative.py.
    rationale = prose_narrative(str(envelope.get("rationale") or ""))

    if not atom_ids and not (OK_ON_EMPTY_ATOMS_WITH_PROSE and rationale):
        logger.error("dose_recommendation.empty_recommendation", extra={**log_scope, "had_rationale": bool(rationale)})
        return _dose_completed_event(
            state, status=WIRE_FAILED, error_code=ERROR_CODE_EMPTY_RECOMMENDATION, model_id=model_id
        )

    return _dose_completed_event(state, status=WIRE_OK, model_id=model_id, atom_ids=atom_ids, rationale=rationale)


def _dose_rejected(state: dict[str, Any]) -> ResultEvent:
    """The per-tenant in-flight cap answer (ADR-254 D5). Always present, so
    turning the cap on is a config change and never a silent drop."""
    logger.warning(
        "dose_recommendation.rejected_tenant_in_flight_cap",
        extra={"tenant_id": state["tenant_id"], "dose_request_id": _dose_request_id(state)},
    )
    return _dose_completed_event(state, status=WIRE_REJECTED, error_code=ERROR_CODE_TENANT_CAP)


def dose_recommendation_contract(*, tenant_inflight_cap: int | None = None) -> LaneContract:
    """The dose_recommendation lane. ``tenant_inflight_cap`` is off by default
    and resolved from the environment by the wiring, never inline here."""
    return LaneContract(
        name=LANE_DOSE_RECOMMENDATION,
        role=ROLE_RECOMMEND,
        request_kind=REQUEST_KIND_DOSE_RECOMMENDATION,
        decode=_dose_decode,
        dispatch=_dose_dispatch,
        result=_dose_result,
        rejected=_dose_rejected,
        tenant_inflight_cap=tenant_inflight_cap,
    )


__all__ = [
    "ERROR_CODE_AGENT_CONTRACT",
    "ERROR_CODE_AGENT_FAILED",
    "ERROR_CODE_EMPTY_RECOMMENDATION",
    "ERROR_CODE_EMPTY_REPLY",
    "ERROR_CODE_NO_MODEL_ID",
    "ERROR_CODE_TENANT_CAP",
    "EVENT_TYPE_COMPANION_TURN_COMPLETED",
    "EVENT_TYPE_DOSE_RECOMMENDATION_COMPLETED",
    "LANE_COMPANION_TURN",
    "LANE_DOSE_RECOMMENDATION",
    "MAX_DOSE_ATOMS",
    "OK_ON_EMPTY_ATOMS_WITH_PROSE",
    "REQUEST_KIND_COMPANION_TURN",
    "REQUEST_KIND_DOSE_RECOMMENDATION",
    "ROLE_COMPANION_CHAT",
    "ROLE_RECOMMEND",
    "TOPIC_COMPANION_TURN_COMPLETED",
    "TOPIC_DOSE_RECOMMENDATION_COMPLETED",
    "TURN_KIND_TYPED",
    "WIRE_FAILED",
    "WIRE_OK",
    "WIRE_REJECTED",
    "companion_turn_contract",
    "dose_recommendation_contract",
]
