"""RED: the companion_turn lane as a LaneContract (ADR-254 D1/D4/D8, R20).

chora.consumption.companion_turn.requested.v1 -> dispatch companion_chat with
the request body passed through VERBATIM -> chora.consumption.companion_turn.
completed.v1 keyed on turn_id.

Three properties this lane does NOT share with either fold lane, each with a
test below that fails if it regresses:

* VERBATIM pass-through. Both fold decodes hand-pick their fields
  (``_reflection_decode`` builds an explicit dict), and a companion_turn lane
  written that way silently drops ``prompt_overrides_json`` +
  ``resolved_prompt_version``, which consumption resolves and the chat binary
  reads out of session state. The failure is a working chat with a flatter
  voice and no error anywhere, so it is pinned here against the payload
  WP-D's own producer emits.
* ALWAYS PUBLISH. A reflection that cannot be cached honestly publishes
  nothing and the caller's claim TTL closes the loop. A chat turn has a
  learner waiting on an SSE stream, so every terminal path publishes a result,
  FAILED with an ``error_code`` when it must.
* turn_id ROUND-TRIPS UNCHANGED and is the workflow id. Enforced upstream by
  ``turn_id UUID PRIMARY KEY`` (migration 0111) and by the consumption edge
  since dbd8082d0; enforced here by refusing anything else rather than by
  tolerating it, because a tolerant reader cannot detect what it tolerates.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.orchestrators.consumption_lanes import (
    ERROR_CODE_AGENT_FAILED,
    ERROR_CODE_TENANT_CAP,
    EVENT_TYPE_COMPANION_TURN_COMPLETED,
    TOPIC_COMPANION_TURN_COMPLETED,
    TURN_KIND_TYPED,
    companion_turn_contract,
)
from chora_ai_kernel_orchestrator.orchestrators.single_agent_workflow import (
    STATUS_FAILED,
    STATUS_OK,
    SingleAgentWorkflowRunner,
    build_single_agent_graph,
)

# WP-D's captured producer wire (2026-08-23): NOT hand-written here. Produced by
# BuildCompanionTurnRequestPayload + events.NewEnvelope + outbox.BuildRow, the
# three calls the live path makes. The golden files are vendored into this repo
# at testdata/companion_turn_goldens/ (captured from the consumption producer).
_GOLDEN_DIR = pathlib.Path(__file__).resolve().parents[2] / "testdata" / "companion_turn_goldens"


def _golden(name: str) -> dict:
    path = _GOLDEN_DIR / name
    # Fail loud rather than skip: a missing golden means the producer contract
    # moved or the tree is not the one we think it is, and a skipped drift pin
    # is indistinguishable from a passing one.
    assert path.is_file(), f"producer golden missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def golden_body() -> dict:
    return _golden("companion_turn_requested.wire.json")


def golden_attrs() -> dict[str, str]:
    # On the wire tracestate is DROPPED when empty (envelopeAttributes), so the
    # attributes a subscriber actually sees omit it. Model that, not the row.
    attrs = {k: str(v) for k, v in _golden("companion_turn_requested.envelope.json").items()}
    return {k: v for k, v in attrs.items() if v != ""}


TURN_ID = "01957c8c-2222-7000-aaaa-222222222222"


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #


def test_decode_accepts_the_producers_captured_wire() -> None:
    """The whole point: decode the bytes WP-D's producer emits, not ours."""
    state = companion_turn_contract().decode(golden_body(), golden_attrs())
    # turn_id rides INSIDE the verbatim request, not as a top-level state key:
    # SingleAgentState is the LangGraph state schema and LangGraph filters state
    # to the keys it declares, so a lane key at the top level is dropped between
    # decode and dispatch.
    assert state["request"]["turn_id"] == TURN_ID
    assert "turn_id" not in state, "a top-level turn_id would be silently dropped by the engine"
    assert state["tenant_id"] == "11111111-1111-7111-8111-111111111111"
    assert state["gcid"] == "00000000-0000-7000-8000-000000001999"
    assert state["event_id"] == "01a02d66-3917-741f-999a-f9c71736ba62"
    # tracestate is absent on the wire and must decode as empty, not blow up.
    assert state["tracestate"] == ""


def test_decode_keeps_the_request_body_verbatim() -> None:
    """A hand-picked decode drops these two and nothing anywhere errors."""
    body = golden_body()
    state = companion_turn_contract().decode(body, golden_attrs())
    assert state["request"] == body, "the request must survive decode unchanged"
    for key in (
        "prompt_overrides_json",
        "resolved_prompt_version",
        "familiar_memory",
        "learner_weakness",
        "locale",
        "mana_action_code",
    ):
        assert state["request"][key] == body[key], f"{key} was dropped by decode"


def test_decode_refuses_a_missing_turn_id() -> None:
    body = golden_body()
    del body["turn_id"]
    with pytest.raises(ValueError, match="turn_id"):
        companion_turn_contract().decode(body, golden_attrs())


def test_decode_refuses_a_non_uuid_turn_id_rather_than_deriving_one() -> None:
    """No tolerant parse: the engine would otherwise derive a UUIDv5 workflow
    id, and consumption's waiter polls by the turn_id it minted."""
    body = golden_body()
    body["turn_id"] = "not-a-uuid"
    attrs = golden_attrs()
    attrs["idempotency_key"] = "not-a-uuid"
    with pytest.raises(ValueError, match="turn_id"):
        companion_turn_contract().decode(body, attrs)


def test_decode_refuses_when_the_envelope_key_disagrees_with_the_body() -> None:
    """Same discipline as the envelope-vs-body tenant check (fcd73d754): the
    contract says envelope.idempotency_key IS turn_id, so a disagreement is a
    producer bug, not something to pick a winner for."""
    attrs = golden_attrs()
    attrs["idempotency_key"] = "01957c8c-9999-7000-aaaa-999999999999"
    with pytest.raises(ValueError, match="idempotency_key"):
        companion_turn_contract().decode(golden_body(), attrs)


@pytest.mark.parametrize("field", ["conversation_id", "familiar_id"])
def test_decode_refuses_a_request_missing_a_required_field(field: str) -> None:
    body = golden_body()
    del body[field]
    with pytest.raises(ValueError, match=field):
        companion_turn_contract().decode(body, golden_attrs())


def test_decode_allows_an_empty_message() -> None:
    """The producer sets `message` unconditionally and a greeting turn can
    carry an empty one; NACKing that to the DLQ would break greetings."""
    body = golden_body()
    body["message"] = ""
    state = companion_turn_contract().decode(body, golden_attrs())
    assert state["request"]["message"] == ""


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def _state() -> dict:
    return companion_turn_contract().decode(golden_body(), golden_attrs())


def test_dispatch_carries_the_body_verbatim_into_the_input_payload() -> None:
    spec = companion_turn_contract().dispatch(_state())
    body = golden_body()
    for key, value in body.items():
        if key == "gcid":  # overridden with the authoritative envelope value
            continue
        assert spec.input_payload[key] == value, f"{key} did not reach the agent"


def test_dispatch_uses_turn_id_as_the_workflow_id_unchanged() -> None:
    spec = companion_turn_contract().dispatch(_state())
    assert spec.workflow_id == TURN_ID


def test_dispatch_stamps_turn_kind_and_the_envelope_fields() -> None:
    spec = companion_turn_contract().dispatch(_state())
    assert spec.input_payload["turn_kind"] == TURN_KIND_TYPED
    # read back out by PubSubAgentExecutor to build the dispatch envelope
    assert spec.input_payload["gcid"] == "00000000-0000-7000-8000-000000001999"
    assert spec.input_payload["traceparent"].startswith("00-")
    assert "tracestate" in spec.input_payload


def test_dispatch_keeps_growth_stage_a_json_number() -> None:
    """The growth-stage plugin refuses a string PERMANENTLY, so a decode that
    stringifies the body (as the reflection decode does) breaks every turn of
    a hatched Companion."""
    body = golden_body()
    body["familiar_config"] = '{"growth_stage":3}'
    body["growth_stage"] = 3
    state = companion_turn_contract().decode(body, golden_attrs())
    spec = companion_turn_contract().dispatch(state)
    assert spec.input_payload["growth_stage"] == 3
    assert isinstance(spec.input_payload["growth_stage"], int)
    assert not isinstance(spec.input_payload["growth_stage"], bool)


def test_dispatch_execution_id_is_stable_for_the_same_turn() -> None:
    """A redelivery must collide with the queued dispatch row, not raise a
    second one against a persistent ADK session."""
    a = companion_turn_contract().dispatch(_state())
    b = companion_turn_contract().dispatch(_state())
    assert a.execution_id == b.execution_id
    assert TURN_ID in a.execution_id


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #


def _agent_ok(**over) -> dict:
    envelope = {
        "turn_kind": "typed",
        "reply_text": "Because dividing by a fraction is multiplying by its reciprocal.",
        "grounding": ["01957c8c-a10a-7000-aaaa-a10aa10aa10a"],
        "tool_calls": ["cite_atom"],
        "model_id": "gemini-2.5-flash",
        "prompt_version": "v1",
        "prompt_source": "registry",
        "prompt_hash": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    }
    envelope.update(over)
    return {"status": STATUS_OK, "output_payload": json.dumps(envelope), "error_message": ""}


def test_result_maps_an_ok_turn_onto_the_caller_topic() -> None:
    event = companion_turn_contract().result(_state(), _agent_ok())
    assert event is not None
    assert event.topic == TOPIC_COMPANION_TURN_COMPLETED
    assert event.event_type == EVENT_TYPE_COMPANION_TURN_COMPLETED
    assert event.idempotency_key == TURN_ID
    assert event.workflow_id == TURN_ID
    body = event.body
    assert body["turn_id"] == TURN_ID
    assert body["conversation_id"] == "01957c8c-4444-7000-aaaa-444444444444"
    assert body["status"] == "OK"
    assert body["error_code"] == ""
    assert body["generated_by_model_id"] == "gemini-2.5-flash"
    assert body["reply_text"].startswith("Because dividing")
    assert body["completed_at"]


def test_result_lifts_grounding_and_tool_calls_into_the_objects_consumption_decodes() -> None:
    """The agent answers with lists of STRINGS (agents.go collects atom ids and
    FunctionCall names); consumption unmarshals into []TurnGrounding /
    []TurnToolCall. Passing the strings through makes consumption NACK the
    whole result and the learner's turn never completes."""
    event = companion_turn_contract().result(_state(), _agent_ok())
    assert event is not None
    assert event.body["grounding"] == [{"atom_id": "01957c8c-a10a-7000-aaaa-a10aa10aa10a"}]
    assert event.body["tool_calls"] == [{"name": "cite_atom"}]


def test_result_forwards_the_prompt_version_and_source_the_agent_emitted() -> None:
    """N9: the agent stamps prompt_version and prompt_source on its envelope and
    this lane DROPPED both, so consumption's parser (waiting since CHO-2368) was
    never fed and every ritual step stamped an empty version.

    The kennel stays deterministic (ADR-254 D5): this forwards two strings the
    agent already produced. It calls no model and branches on no content."""
    event = companion_turn_contract().result(_state(), _agent_ok())
    assert event is not None
    assert event.body["prompt_version"] == "v1"
    assert event.body["prompt_source"] == "registry"


def test_result_never_fabricates_a_prompt_version_an_older_agent_did_not_send() -> None:
    """Both-shape tolerance, the older-agent direction. An agent image that
    predates the stamp emits no prompt fields; the keys must still be PRESENT
    and EMPTY, never absent and never invented. An invented version is a false
    provenance claim, and it is indistinguishable from a real one."""
    old_agent = _agent_ok()
    envelope = json.loads(old_agent["output_payload"])
    del envelope["prompt_version"]
    del envelope["prompt_source"]
    old_agent["output_payload"] = json.dumps(envelope)

    event = companion_turn_contract().result(_state(), old_agent)
    assert event is not None
    assert event.body["prompt_version"] == ""
    assert event.body["prompt_source"] == ""


def test_result_carries_the_prompt_fields_on_every_terminal_not_just_ok() -> None:
    """A FAILED turn still carries the keys, so a consumer never has to tell
    'this turn had no version' apart from 'this event shape is older'."""
    failed = {"status": STATUS_FAILED, "output_payload": "", "error_message": "boom"}
    event = companion_turn_contract().result(_state(), failed)
    assert event is not None
    assert event.body["prompt_version"] == ""
    assert event.body["prompt_source"] == ""


def test_result_forwards_the_prompt_hash_the_agent_computed() -> None:
    """N9 step 3. The agent hashes the prompt it ACTUALLY composed and ran, at
    the only point that holds it. This lane forwards the string; it never
    computes one, because a hash taken anywhere else is a true hash of the wrong
    prompt and is indistinguishable from a real one."""
    event = companion_turn_contract().result(_state(), _agent_ok())
    assert event is not None
    assert event.body["prompt_hash"] == ("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")


def test_result_never_invents_a_prompt_hash_an_older_agent_did_not_send() -> None:
    """An agent image predating the stamp sends no hash. The key stays present
    and EMPTY. Anything else here would be a 64-character lie."""
    old_agent = _agent_ok()
    envelope = json.loads(old_agent["output_payload"])
    del envelope["prompt_hash"]
    old_agent["output_payload"] = json.dumps(envelope)

    event = companion_turn_contract().result(_state(), old_agent)
    assert event is not None
    assert event.body["prompt_hash"] == ""


def test_result_publishes_failed_when_the_dispatch_failed() -> None:
    """A learner is on the other end of an SSE stream: publishing nothing
    leaves them on a spinner until consumption's turn deadline."""
    completion = {"status": STATUS_FAILED, "output_payload": "", "error_message": "agent exploded"}
    event = companion_turn_contract().result(_state(), completion)
    assert event is not None, "a failed turn must still close the loop"
    assert event.body["status"] == "FAILED"
    assert event.body["error_code"] == ERROR_CODE_AGENT_FAILED
    assert event.body["reply_text"] == ""
    assert event.body["generated_by_model_id"] == ""
    assert event.idempotency_key == TURN_ID


def test_result_publishes_failed_when_the_agent_envelope_is_not_json() -> None:
    completion = {"status": STATUS_OK, "output_payload": "not json at all", "error_message": ""}
    event = companion_turn_contract().result(_state(), completion)
    assert event is not None
    assert event.body["status"] == "FAILED"
    assert event.body["error_code"]


def test_result_refuses_to_publish_ok_without_a_model_id() -> None:
    """generated_by_model_id is mandatory on OK: an unattributable reply would
    be a governance hole, so it degrades to FAILED rather than to a blank."""
    event = companion_turn_contract().result(_state(), _agent_ok(model_id=""))
    assert event is not None
    assert event.body["status"] == "FAILED"
    assert event.body["generated_by_model_id"] == ""


def test_result_carries_an_honest_refusal_through_as_ok() -> None:
    event = companion_turn_contract().result(
        _state(), _agent_ok(refusal_reason="off_topic", reply_text="Let us stay with maths.")
    )
    assert event is not None
    assert event.body["status"] == "OK"
    assert event.body["refusal_reason"] == "off_topic"
    assert event.body["reply_text"] == "Let us stay with maths."


def test_result_publishes_failed_on_an_empty_reply() -> None:
    event = companion_turn_contract().result(_state(), _agent_ok(reply_text=""))
    assert event is not None
    assert event.body["status"] == "FAILED"
    assert event.body["error_code"]


# --------------------------------------------------------------------------- #
# backpressure
# --------------------------------------------------------------------------- #


def test_a_capped_contract_answers_a_rejected_turn_rather_than_dropping_it() -> None:
    contract = companion_turn_contract(tenant_inflight_cap=2)
    assert contract.tenant_inflight_cap == 2
    assert contract.rejected is not None
    event = contract.rejected(_state())
    assert event is not None
    assert event.body["status"] == "REJECTED"
    assert event.body["error_code"] == ERROR_CODE_TENANT_CAP
    assert event.idempotency_key == TURN_ID


def test_the_uncapped_contract_is_the_default() -> None:
    assert companion_turn_contract().tenant_inflight_cap is None


# --------------------------------------------------------------------------- #
# THROUGH THE ENGINE, not around it
# --------------------------------------------------------------------------- #
#
# Every test above calls the contract's functions directly, which is how a real
# defect shipped: `SingleAgentState` is the LangGraph state SCHEMA and LangGraph
# FILTERS state to the keys it declares, so a lane-specific key returned by
# decode at the top level is dropped before dispatch ever sees it. Calling
# dispatch(decode(...)) by hand cannot observe that, because it never crosses
# the engine. These drive the real compiled graph and the real runner.


class _StubExecutor:
    """Answers in-line instead of parking, so the graph runs to its emit node."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return type("R", (), {"output_payload": self.payload, "input_tokens": 0, "output_tokens": 0})()


class _StubOutbox:
    def __init__(self) -> None:
        self.queued: list[dict[str, Any]] = []

    async def queue_request(self, request: dict[str, Any]) -> str:
        self.queued.append(request)
        return f"row-{len(self.queued)}"


def _run_through_engine(agent_payload: str) -> tuple[_StubExecutor, _StubOutbox]:
    from langgraph.checkpoint.memory import InMemorySaver

    contract = companion_turn_contract()
    executor, outbox = _StubExecutor(agent_payload), _StubOutbox()
    graph = build_single_agent_graph(
        contract=contract,
        executor=executor,
        outbox_writer=outbox,
        source_project="chora-489812",
        checkpointer=InMemorySaver(),
    )
    runner = SingleAgentWorkflowRunner(
        graph=graph, contract=contract, outbox_writer=outbox, source_project="chora-489812"
    )
    import asyncio

    outcome = asyncio.run(runner.handle_request(golden_body(), golden_attrs()))
    assert outcome.outcome == "completed", outcome
    return executor, outbox


def test_the_turn_survives_the_engines_state_schema_end_to_end() -> None:
    """REGRESSION: turn_id reached dispatch as a KeyError in production because
    it was returned as a top-level state key and LangGraph filtered it out.
    Anything the lane needs must ride a key the schema declares."""
    executor, outbox = _run_through_engine(
        json.dumps(
            {
                "turn_kind": "typed",
                "reply_text": "Reciprocals.",
                "grounding": [],
                "tool_calls": [],
                "model_id": "gemini-2.5-flash",
                "prompt_version": "v1",
                "prompt_source": "registry",
            }
        )
    )

    assert len(executor.calls) == 1, "the lane must dispatch exactly once"
    call = executor.calls[0]
    assert call["agent_role"] == "companion_chat"
    assert call["workflow_id"] == TURN_ID, "turn_id must reach the dispatch as the workflow id"
    payload = json.loads(call["input_payload"])
    # the two keys a hand-picked decode drops, checked THROUGH the engine
    assert payload["prompt_overrides_json"] == golden_body()["prompt_overrides_json"]
    assert payload["resolved_prompt_version"] == golden_body()["resolved_prompt_version"]
    assert payload["turn_kind"] == TURN_KIND_TYPED

    assert len(outbox.queued) == 1, "exactly one completion must be queued"
    queued = outbox.queued[0]
    assert queued["topic"] == TOPIC_COMPANION_TURN_COMPLETED
    assert queued["idempotency_key"] == TURN_ID
    assert queued["body"]["turn_id"] == TURN_ID
    assert queued["body"]["status"] == "OK"
    assert queued["body"]["reply_text"] == "Reciprocals."


def test_a_failed_turn_still_reaches_the_outbox_through_the_engine() -> None:
    """The always-publish invariant, asserted where it actually matters."""
    _, outbox = _run_through_engine("this is not json")
    assert len(outbox.queued) == 1
    assert outbox.queued[0]["body"]["status"] == "FAILED"
    assert outbox.queued[0]["body"]["error_code"]


def test_lift_accepts_objects_so_a_richer_agent_envelope_is_not_silently_dropped() -> None:
    """The lift exists because the agent answered with STRINGS. If the agent is
    ever taught to answer with the objects consumption already declares (so the
    citation can carry its revision), the OLD lift drops every entry on the
    isinstance(str) check and the learner silently loses grounding AND
    tool_calls. Kennel and agent are separate deployments, so the lift must
    accept both shapes or the pair has a deploy-ORDER hazard.
    """
    from chora_ai_kernel_orchestrator.orchestrators.consumption_lanes import (
        _grounding,
        _tool_calls,
    )

    scope: dict[str, object] = {}

    # strings still lift (the shape production emits today)
    assert _grounding(["a1"], log_scope=scope) == [{"atom_id": "a1"}]
    assert _tool_calls(["cite_atom"], log_scope=scope) == [{"name": "cite_atom"}]

    # objects pass through, carrying the provenance a string cannot express
    assert _grounding([{"atom_id": "a1", "revision_id": "r1"}], log_scope=scope) == [
        {"atom_id": "a1", "revision_id": "r1"}
    ]
    assert _tool_calls([{"name": "cite_atom"}], log_scope=scope) == [{"name": "cite_atom"}]

    # an object without the required key is still a defect, not a pass-through
    assert _grounding([{"revision_id": "r1"}], log_scope=scope) == []
    assert _tool_calls([{"summary": "no name"}], log_scope=scope) == []

    # and a shape that is neither is still refused
    assert _grounding([42], log_scope=scope) == []
    assert _tool_calls([42], log_scope=scope) == []


def _old_kennel_lift(raw: object) -> list[dict[str, str]]:
    """The lift EXACTLY as it shipped before the both-shapes change: a
    strings-only branch. Reproduced here so the deploy-order constraint is
    pinned by a test rather than by anyone's memory of a decision.
    """
    out: list[dict[str, str]] = []
    for entry in raw or []:  # type: ignore[union-attr]
        if not isinstance(entry, str):
            continue
        value = entry.strip()
        if value:
            out.append({"atom_id": value})
    return out


def test_cross_version_matrix_pins_why_the_kennel_must_deploy_first() -> None:
    """kennel and companion_chat are SEPARATE deployments, so a rollout puts one
    ahead of the other. Three of the four combinations are safe; exactly one is
    not, and it is the reason the kennel lands first.
    """
    from chora_ai_kernel_orchestrator.orchestrators.consumption_lanes import _grounding

    scope: dict[str, object] = {}
    old_agent = ["a1"]  # []string
    new_agent = [{"atom_id": "a1", "revision_id": "r1"}]  # []object

    # 1. old kennel + old agent: the shape the lift was written for.
    assert _old_kennel_lift(old_agent) == [{"atom_id": "a1"}]

    # 2. new kennel + old agent: unchanged, so rolling the kennel alone is safe.
    assert _grounding(old_agent, log_scope=scope) == [{"atom_id": "a1"}]

    # 3. new kennel + new agent: the target state, and revision_id SURVIVES,
    #    which is the whole reason for the change.
    assert _grounding(new_agent, log_scope=scope) == [{"atom_id": "a1", "revision_id": "r1"}]

    # 4. old kennel + new agent: THE KNOWN-BAD CELL. Every entry is dropped and
    #    the learner silently loses grounding. Asserted, not avoided, so that
    #    creating this combination fails a test instead of a turn.
    assert _old_kennel_lift(new_agent) == [], (
        "if this ever passes, the ordering constraint has changed: verify "
        "before allowing the agent to roll ahead of the kennel"
    )


def test_legacy_string_lift_is_reported_and_the_object_path_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The compat branch must not be fail-quiet.

    Without a signal, "no strings arrived" and "strings arrived and were quietly
    lifted" are indistinguishable, so nobody can ever answer whether the branch
    still carries traffic. The retirement condition is about REACHABILITY, not
    observed traffic, but the observation is the prerequisite for asking at all.
    """
    import logging

    from chora_ai_kernel_orchestrator.orchestrators.consumption_lanes import (
        _grounding,
        _tool_calls,
    )

    scope: dict[str, object] = {}

    # LEGACY shape -> reported, with the count.
    with caplog.at_level(logging.INFO):
        caplog.clear()
        assert _grounding(["a1", "a2"], log_scope=scope) == [
            {"atom_id": "a1"},
            {"atom_id": "a2"},
        ]
        lifted = [r for r in caplog.records if r.getMessage() == "companion_turn.grounding_legacy_string_lifted"]
        assert len(lifted) == 1, "one line per call, not per entry"
        assert lifted[0].count == 2, f"count must name how many were lifted: {lifted[0].count}"

        caplog.clear()
        _tool_calls(["cite_atom"], log_scope=scope)
        assert [r for r in caplog.records if r.getMessage() == "companion_turn.tool_call_legacy_string_lifted"]

    # DISCRIMINATOR: the object path must stay silent, or the signal means nothing.
    with caplog.at_level(logging.INFO):
        caplog.clear()
        assert _grounding([{"atom_id": "a1", "revision_id": "r1"}], log_scope=scope) == [
            {"atom_id": "a1", "revision_id": "r1"}
        ]
        _tool_calls([{"name": "cite_atom"}], log_scope=scope)
        assert not [r for r in caplog.records if "legacy_string_lifted" in r.getMessage()], (
            "the object path must emit NO legacy signal, else the signal cannot "
            "discriminate and would read as permanent legacy traffic"
        )

    # An empty list is not legacy traffic either.
    with caplog.at_level(logging.INFO):
        caplog.clear()
        assert _grounding([], log_scope=scope) == []
        assert not [r for r in caplog.records if "legacy_string_lifted" in r.getMessage()]
