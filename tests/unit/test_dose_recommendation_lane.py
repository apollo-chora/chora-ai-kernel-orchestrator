"""The dose_recommendation lane (ADR-254 D1/D4, gate G-b).

⚠ FIXTURE PROVENANCE, STATED BECAUSE IT IS WEAKER THAN THE SIBLING LANE'S.
companion_turn pins against a PRODUCER-CAPTURED golden in the consumption
repo's client testdata. There is no such capture for this lane, so the body
below is HAND-BUILT from reading the producer's Go source. That pins MY READING
of the producer, not the producer, and a hand-built fixture cannot detect the
producer drifting away from it. WP-D should capture a golden the way they did
for companion_turn; until then treat every "matches the producer" claim here as
one step weaker than it looks.

The tests that matter most are at the bottom: they drive the REAL compiled graph
through the REAL runner. Calling ``contract.dispatch(contract.decode(...))`` by
hand cannot see LangGraph filtering an undeclared top-level key out of the state,
which is exactly how a defect shipped on the companion_turn lane (d7aef5735).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.orchestrators.consumption_lanes import (
    ERROR_CODE_AGENT_CONTRACT,
    ERROR_CODE_AGENT_FAILED,
    ERROR_CODE_EMPTY_RECOMMENDATION,
    ERROR_CODE_NO_MODEL_ID,
    ERROR_CODE_TENANT_CAP,
    MAX_DOSE_ATOMS,
    TOPIC_DOSE_RECOMMENDATION_COMPLETED,
    WIRE_FAILED,
    WIRE_OK,
    WIRE_REJECTED,
    dose_recommendation_contract,
)
from chora_ai_kernel_orchestrator.orchestrators.single_agent_workflow import (
    SingleAgentWorkflowRunner,
    build_single_agent_graph,
)

DOSE_ID = "01957c8c-3333-7000-aaaa-333333333333"
TENANT = "11111111-1111-7111-8111-111111111111"
GCID = "00000000-0000-7000-8000-000000001999"


def dose_context() -> dict[str, Any]:
    """The `doseContext` struct bus_dose_recommender.go marshals into
    context_json: mana_tier, learner_persona, user_prompt, topic_hint,
    candidates[] (AtomCandidate = atom_id/title/snippet)."""
    return {
        "mana_tier": "standard",
        "learner_persona": "curious-explorer",
        "user_prompt": "something about fractions",
        "topic_hint": "fractions",
        "candidates": [
            {"atom_id": "aaaaaaaa-0000-7000-8000-000000000001", "title": "Halves", "snippet": "..."},
            {"atom_id": "aaaaaaaa-0000-7000-8000-000000000002", "title": "Thirds", "snippet": "..."},
        ],
    }


def golden_body(**over: Any) -> dict[str, Any]:
    body = {
        "dose_request_id": DOSE_ID,
        "tenant_id": TENANT,
        "gcid": GCID,
        "trigger": "daily_dose_ai",
        "context_json": json.dumps(dose_context()),
        "requested_at": "2026-08-23T09:00:00Z",
    }
    body.update(over)
    return body


def golden_attrs(**over: str) -> dict[str, str]:
    attrs = {
        "event_id": "01957c8c-9999-7000-aaaa-999999999999",
        "idempotency_key": DOSE_ID,
        "tenant_id": TENANT,
        "gcid": GCID,
        "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    }
    attrs.update(over)
    return attrs


def contract():
    return dose_recommendation_contract()


def decoded(**over: Any) -> dict[str, Any]:
    return contract().decode(golden_body(**over), golden_attrs())


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #


def test_decode_carries_the_request_body_and_the_envelope_scope() -> None:
    state = decoded()
    assert state["tenant_id"] == TENANT
    assert state["gcid"] == GCID
    assert state["request"]["dose_request_id"] == DOSE_ID
    assert state["request"]["context_json"], "context_json must survive decode"


def test_decode_refuses_a_missing_dose_request_id() -> None:
    with pytest.raises(ValueError, match="no dose_request_id"):
        contract().decode(
            {k: v for k, v in golden_body().items() if k != "dose_request_id"},
            golden_attrs(idempotency_key=""),
        )


def test_decode_refuses_a_non_uuid_dose_request_id_rather_than_deriving_one() -> None:
    with pytest.raises(ValueError, match="not a UUID"):
        contract().decode(golden_body(dose_request_id="dose-42"), golden_attrs(idempotency_key="dose-42"))


def test_decode_refuses_when_the_envelope_key_disagrees_with_the_body() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        contract().decode(golden_body(), golden_attrs(idempotency_key="01957c8c-4444-7000-aaaa-444444444444"))


@pytest.mark.parametrize("attr", ["tenant_id", "gcid"])
def test_decode_refuses_an_unscoped_request(attr: str) -> None:
    body = {k: v for k, v in golden_body().items() if k != attr}
    attrs = {k: v for k, v in golden_attrs().items() if k != attr}
    with pytest.raises(ValueError, match="tenant_id/gcid"):
        contract().decode(body, attrs)


# --------------------------------------------------------------------------- #
# dispatch: the UNPACK, which is the one place this lane must differ
# --------------------------------------------------------------------------- #


def test_dispatch_unpacks_context_json_into_the_keys_the_agent_actually_reads() -> None:
    """REGRESSION GUARD. A verbatim pass-through (correct on companion_turn)
    leaves context_json as one opaque string; the recommender reads DISCRETE
    keys, so it would silently ignore consumption's whole context and fall back
    to its own searcher while the dose still returned atoms and nothing errored."""
    payload = contract().dispatch(decoded()).input_payload
    assert payload["learner_persona"] == "curious-explorer"
    assert payload["topic_hint"] == "fractions"
    assert payload["mana_tier"] == "standard"
    assert payload["user_prompt"] == "something about fractions"


def test_dispatch_writes_atom_candidates_as_a_json_string_not_a_list() -> None:
    """candidatesFromState json-unmarshals a STRING read out of session state.
    Handing it a list makes the read fail and the ranking is silently discarded."""
    payload = contract().dispatch(decoded()).input_payload
    raw = payload["atom_candidates"]
    assert isinstance(raw, str), "a list here is silently dropped by the agent"
    assert [a["atom_id"] for a in json.loads(raw)] == [
        "aaaaaaaa-0000-7000-8000-000000000001",
        "aaaaaaaa-0000-7000-8000-000000000002",
    ], "candidate ORDER is consumption's ranking and must survive"


def test_dispatch_stamps_both_gcid_and_user_gcid() -> None:
    """They live at DIFFERENT layers: the executor reads gcid to stamp the
    dispatch envelope; the TOOL reads user_gcid. Stamping only gcid makes the
    tool fall through to the MODEL's value, a silent mis-attribution."""
    payload = contract().dispatch(decoded()).input_payload
    assert payload["gcid"] == GCID
    assert payload["user_gcid"] == GCID


def test_dispatch_uses_dose_request_id_as_the_workflow_id_unchanged() -> None:
    spec = contract().dispatch(decoded())
    assert spec.workflow_id == DOSE_ID
    assert spec.execution_id == f"{DOSE_ID}:dose"


def test_dispatch_survives_an_unusable_context_json_without_raising() -> None:
    """An absent or malformed context is a degradation, not a producer bug: the
    agent still composes a dose from its searcher. It must not NACK to the DLQ."""
    payload = contract().dispatch(decoded(context_json="not json")).input_payload
    assert "atom_candidates" not in payload
    assert payload["user_gcid"] == GCID


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #


def envelope_json(**over: Any) -> str:
    env = {
        "atoms": [
            {"atom_id": "bbbb0001-0000-7000-8000-000000000001", "title": "A", "rank_score": 0.1},
            {"atom_id": "bbbb0002-0000-7000-8000-000000000002", "title": "B", "rank_score": 0.9},
        ],
        "rationale": "These follow on from what you did yesterday.",
        "model_id": "gemini-3.1-pro-preview",
        "prompt_version": "v1",
        "prompt_source": "registry",
    }
    env.update(over)
    return json.dumps(env)


def result_body(payload: str, status: str = "OK") -> dict[str, Any]:
    ev = contract().result(decoded(), {"status": status, "output_payload": payload})
    assert ev is not None, "this lane must publish on EVERY terminal path"
    assert ev.topic == TOPIC_DOSE_RECOMMENDATION_COMPLETED
    return ev.body


def test_result_projects_atoms_to_ids_in_array_order_not_by_rank_score() -> None:
    """Array order IS the recommendation order and is carried, never recomputed.
    rank_score is disclosure only and is legitimately ZERO on the candidate path,
    so sorting by it would flatten the dose to an arbitrary order."""
    body = result_body(envelope_json())
    assert body["recommended_atom_ids"] == [
        "bbbb0001-0000-7000-8000-000000000001",
        "bbbb0002-0000-7000-8000-000000000002",
    ], "descending rank_score must NOT reorder the list"
    assert body["status"] == WIRE_OK
    assert body["generated_by_model_id"] == "gemini-3.1-pro-preview"
    assert body["workflow_id"] == DOSE_ID


def test_result_dedupes_preserving_first_occurrence() -> None:
    dup = "bbbb0001-0000-7000-8000-000000000001"
    body = result_body(
        envelope_json(atoms=[{"atom_id": dup}, {"atom_id": "bbbb0003-0000-7000-8000-000000000003"}, {"atom_id": dup}])
    )
    assert body["recommended_atom_ids"] == [dup, "bbbb0003-0000-7000-8000-000000000003"]


def test_result_publishes_failed_when_the_dispatch_failed() -> None:
    body = result_body("", status="FAILED")
    assert body["status"] == WIRE_FAILED
    assert body["error_code"] == ERROR_CODE_AGENT_FAILED


def test_result_publishes_failed_when_the_envelope_is_not_json() -> None:
    body = result_body("I could not do that.")
    assert body["status"] == WIRE_FAILED
    assert body["error_code"] == ERROR_CODE_AGENT_CONTRACT


def test_result_refuses_to_publish_ok_without_a_model_id() -> None:
    body = result_body(envelope_json(model_id=""))
    assert body["status"] == WIRE_FAILED
    assert body["error_code"] == ERROR_CODE_NO_MODEL_ID


def test_empty_atoms_with_prose_rationale_is_an_honest_ok() -> None:
    body = result_body(envelope_json(atoms=[], rationale="Nothing new for you today."))
    assert body["status"] == WIRE_OK
    assert body["recommended_atom_ids"] == []
    assert body["rationale"] == "Nothing new for you today."


@pytest.mark.parametrize("rationale", ["", "   ", '{"a":1}', "42", "true", "null", '"q"'])
def test_empty_atoms_without_prose_is_failed_not_a_silent_empty_dose(rationale: str) -> None:
    """A merely non-blank rationale was the WRONG predicate: consumption's proseNarrative also
    drops a leading brace/bracket and anything json.Valid accepts, so a
    JSON-shaped rationale would publish OK and show the learner NOTHING."""
    body = result_body(envelope_json(atoms=[], rationale=rationale))
    assert body["status"] == WIRE_FAILED
    assert body["error_code"] == ERROR_CODE_EMPTY_RECOMMENDATION


def test_a_json_shaped_rationale_is_stripped_even_when_atoms_are_present() -> None:
    body = result_body(envelope_json(rationale='{"blurb":"hi"}'))
    assert body["status"] == WIRE_OK, "atoms alone still make a usable dose"
    assert body["rationale"] == "", "a JSON blob must never reach the learner as copy"


def test_the_rejected_arm_answers_rather_than_dropping() -> None:
    ev = dose_recommendation_contract(tenant_inflight_cap=1).rejected(decoded())
    assert ev is not None
    assert ev.body["status"] == WIRE_REJECTED
    assert ev.body["error_code"] == ERROR_CODE_TENANT_CAP


# --------------------------------------------------------------------------- #
# THROUGH THE REAL ENGINE (d7aef5735: the pairwise tests above are blind here)
# --------------------------------------------------------------------------- #


class _StubExecutor:
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
    import asyncio

    from langgraph.checkpoint.memory import InMemorySaver

    c = contract()
    executor, outbox = _StubExecutor(agent_payload), _StubOutbox()
    graph = build_single_agent_graph(
        contract=c, executor=executor, outbox_writer=outbox, source_project="chora-489812", checkpointer=InMemorySaver()
    )
    runner = SingleAgentWorkflowRunner(graph=graph, contract=c, outbox_writer=outbox, source_project="chora-489812")
    outcome = asyncio.run(runner.handle_request(golden_body(), golden_attrs()))
    assert outcome.outcome == "completed", outcome
    return executor, outbox


def test_the_dose_survives_the_engines_state_schema_end_to_end() -> None:
    """dose_request_id must reach dispatch as the workflow id THROUGH the engine,
    and the unpacked context keys must survive the state filter with it."""
    executor, outbox = _run_through_engine(envelope_json())

    assert len(executor.calls) == 1, "the lane must dispatch exactly once"
    call = executor.calls[0]
    assert call["agent_role"] == "recommend"
    assert call["workflow_id"] == DOSE_ID, "dose_request_id must be the workflow id"

    payload = json.loads(call["input_payload"])
    assert payload["learner_persona"] == "curious-explorer"
    assert isinstance(payload["atom_candidates"], str)
    assert payload["user_gcid"] == GCID
    assert len(outbox.queued) == 1, "exactly one completion must be queued"


def test_the_atom_cap_says_so_when_it_truncates(caplog: Any) -> None:
    """No silent caps. A dose truncated from 25 to MAX_DOSE_ATOMS must not be
    indistinguishable from one that genuinely had MAX_DOSE_ATOMS: the
    non-dict-entry branch two lines up already logs, so a bounded result that
    stays quiet reads as complete when it is not."""
    over = [{"atom_id": f"bbbb{i:04d}-0000-7000-8000-{i:012d}"} for i in range(MAX_DOSE_ATOMS + 5)]
    with caplog.at_level(logging.WARNING):
        body = result_body(envelope_json(atoms=over))

    assert len(body["recommended_atom_ids"]) == MAX_DOSE_ATOMS, "the cap must still bind"
    assert body["status"] == WIRE_OK
    records = [r for r in caplog.records if "atom_cap" in r.getMessage() or "truncat" in r.getMessage()]
    assert records, "the cap truncated silently: 25 atoms in, 20 out, and nothing said so"
    rec = records[0]
    assert getattr(rec, "returned", None) == MAX_DOSE_ATOMS
    assert getattr(rec, "dropped", None) == 5, "the log must say HOW MANY were dropped"


def test_the_cap_stays_quiet_when_it_does_not_bite(caplog: Any) -> None:
    """Negative control: a log on every dose would be noise, not evidence."""
    exact = [{"atom_id": f"bbbb{i:04d}-0000-7000-8000-{i:012d}"} for i in range(MAX_DOSE_ATOMS)]
    with caplog.at_level(logging.WARNING):
        body = result_body(envelope_json(atoms=exact))
    assert len(body["recommended_atom_ids"]) == MAX_DOSE_ATOMS
    assert not [r for r in caplog.records if "atom_cap" in r.getMessage()], "exactly at the cap is not truncation"
