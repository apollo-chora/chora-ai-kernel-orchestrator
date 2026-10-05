"""RED: the two fog folds as LaneContracts (ADR-254 D2/D4/D13).

kg_exploration_workflow: concept_suggestion.requested.v1 -> kg_explore
(exploration_json + request_source) -> concept_suggestion.emitted.v1, fields
unchanged, idempotency derived from the request key, a FAILED dispatch still
closes the loop with an empty proposal.

companion_reflection_workflow: goal_knowledge.synthesis_requested.v1 ->
companion_chat turn_kind=reflect (familiar_id + reflection_json) ->
goal_knowledge.synthesized.v1: content_hash + root_concept_id echoed VERBATIM,
a decline PUBLISHES with nothing_to_say=true, generated_by_model_id mandatory,
MAX_SYNTHESIS_CHARS=600 in lockstep, no-signal requests are skipped silently,
source_service = chora-ai-kernel-orchestrator.
"""

from __future__ import annotations

import json

import pytest

from chora_ai_kernel_orchestrator.orchestrators.fold_lanes import (
    MAX_SYNTHESIS_CHARS,
    TOPIC_CONCEPT_SUGGESTION_EMITTED,
    TOPIC_GOAL_KNOWLEDGE_SYNTHESIZED,
    companion_reflection_contract,
    kg_exploration_contract,
)
from chora_ai_kernel_orchestrator.orchestrators.single_agent_workflow import Skipped

TENANT = "11111111-1111-7111-8111-111111111111"
GCID = "00000000-0000-7000-8000-000000001999"
EVENT = "01a02a3b-c5b2-7c13-bf7d-a9ecdb8ccfb1"


def _attrs(**over: str) -> dict[str, str]:
    a = {
        "event_id": EVENT,
        "idempotency_key": f"req.{EVENT}",
        "tenant_id": TENANT,
        "gcid": GCID,
        "traceparent": "00-aa-bb-01",
        "tracestate": "chora=1",
    }
    a.update(over)
    return a


# --------------------------------------------------------------------------- #
# kg_exploration
# --------------------------------------------------------------------------- #

KG_BODY = {
    "familiar_id": "fam-1",
    "map_theme": "cells",
    "focal_concept_id": "c-osmosis",
    "focal_title": "Osmosis",
    "focal_atom_refs": ["atom-1"],
    "existing_concepts": [
        {"concept_id": "c-osmosis", "title": "Osmosis"},
        {"concept_id": "c-diff", "title": "Diffusion"},
    ],
    "atom_catalogue": [
        {
            "atom_id": "atom-1",
            "title": "Osmosis basics",
            "atom_type": "MCQ",
            "topic_tags": ["cells"],
            "relevance": "high",
        }
    ],
    "request_source": "campaign_free_reveal",
    "sub_goal": "explain water movement",
    "goal_title": "Cell transport",
    "ancestors": ["Cells"],
    "weakness": {"descriptor": "confuses solute and solvent", "misconceptions": [], "evidence": []},
}


def test_kg_contract_identity() -> None:
    c = kg_exploration_contract()
    assert c.name == "kg_exploration" and c.role == "kg_explore"
    assert c.request_kind == "concept_suggestion.requested"
    assert c.tenant_inflight_cap is None


def test_kg_decode_requires_tenant_gcid_and_event_id() -> None:
    c = kg_exploration_contract()
    with pytest.raises(ValueError, match="tenant_id"):
        c.decode(KG_BODY, _attrs(tenant_id=""))
    with pytest.raises(ValueError, match="event_id"):
        c.decode(KG_BODY, _attrs(event_id=""))
    state = c.decode(KG_BODY, _attrs())
    assert not isinstance(state, Skipped)
    assert state["tenant_id"] == TENANT and state["gcid"] == GCID and state["event_id"] == EVENT
    assert state["request"]["focal_title"] == "Osmosis"
    # body fallbacks when the attributes are thin (consumption sets both; be tolerant)
    state2 = c.decode({**KG_BODY, "tenant_id": TENANT, "learner_gcid": GCID}, {"event_id": EVENT})
    assert state2["tenant_id"] == TENANT and state2["gcid"] == GCID


def test_kg_dispatch_forwards_the_request_as_exploration_json_plus_request_source() -> None:
    c = kg_exploration_contract()
    spec = c.dispatch(c.decode(KG_BODY, _attrs()))
    assert spec.execution_id == f"{EVENT}:explore" and spec.workflow_id == EVENT
    payload = spec.input_payload
    assert payload["request_source"] == "campaign_free_reveal"
    exploration = json.loads(payload["exploration_json"])
    # the Go Request struct's fields, verbatim
    for k in (
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
    ):
        assert k in exploration, k
    assert exploration["atom_catalogue"][0]["atom_id"] == "atom-1"
    assert exploration["weakness"]["descriptor"] == "confuses solute and solvent"
    # envelope keys the executor reads back
    assert payload["gcid"] == GCID and payload["traceparent"] == "00-aa-bb-01"


def test_kg_dispatch_defaults_an_absent_request_source_to_learner_request() -> None:
    c = kg_exploration_contract()
    spec = c.dispatch(c.decode({**KG_BODY, "request_source": ""}, _attrs()))
    assert spec.input_payload["request_source"] == "learner_request"
    assert json.loads(spec.input_payload["exploration_json"])["request_source"] == "learner_request"


def test_kg_result_maps_the_agents_envelope_onto_the_frozen_emitted_body() -> None:
    c = kg_exploration_contract()
    state = c.decode(KG_BODY, _attrs())
    envelope = {
        "concepts": [{"title": "Tonicity", "rationale": "next step", "atom_refs": ["atom-1"]}],
        "edges": [
            {
                "source_concept_id": "c-osmosis",
                "target_concept_id": "c-diff",
                "edge_class": "lateral",
                "rationale": "related",
            }
        ],
        "model_id": "gemini-2.5-flash",
        "prompt_version": "v3",
        "prompt_source": "embedded_fallback",
    }
    ev = c.result(state, {"status": "OK", "output_payload": json.dumps(envelope), "error_message": ""})
    assert ev is not None
    assert ev.topic == TOPIC_CONCEPT_SUGGESTION_EMITTED == "chora.consumption.concept_suggestion.emitted.v1"
    assert ev.event_type == "consumption.concept_suggestion.emitted"
    assert ev.idempotency_key == f"req.{EVENT}", "derived from the REQUEST key (ADR-254 D4)"
    assert ev.tenant_id == TENANT and ev.gcid == GCID and ev.workflow_id == EVENT
    assert ev.traceparent == "00-aa-bb-01"
    body = ev.body
    assert body["tenant_id"] == TENANT and body["learner_gcid"] == GCID
    assert body["familiar_id"] == "fam-1" and body["map_theme"] == "cells"
    assert body["focal_concept_id"] == "c-osmosis" and body["run_id"] == EVENT
    assert body["model_used"] == "gemini-2.5-flash"
    assert body["concepts"] == [{"title": "Tonicity", "rationale": "next step", "atom_refs": ["atom-1"]}]
    assert body["edges"] == [
        {
            "source_concept_id": "c-osmosis",
            "target_concept_id": "c-diff",
            "edge_class": "lateral",
            "rationale": "related",
        }
    ]
    assert set(body) == {
        "tenant_id",
        "learner_gcid",
        "familiar_id",
        "map_theme",
        "focal_concept_id",
        "model_used",
        "run_id",
        "concepts",
        "edges",
    }


def test_kg_result_tolerates_fenced_json_and_drops_malformed_entries() -> None:
    c = kg_exploration_contract()
    state = c.decode(KG_BODY, _attrs())
    answer = {
        "concepts": [{"title": " Tonicity "}, "junk", {"title": ""}],
        "edges": [
            {"source_concept_id": "a", "target_concept_id": "a", "edge_class": "lateral"},
            {"source_concept_id": "a", "target_concept_id": "b", "edge_class": "weird"},
            {"source_concept_id": "c-osmosis", "target_concept_id": "c-diff", "edge_class": "hierarchy"},
        ],
        "model_id": "m",
    }
    raw = "```json\n" + json.dumps(answer) + "\n```"
    ev = c.result(state, {"status": "OK", "output_payload": raw, "error_message": ""})
    assert ev is not None
    assert ev.body["concepts"] == [{"title": "Tonicity", "rationale": "", "atom_refs": []}]
    assert [e["edge_class"] for e in ev.body["edges"]] == ["hierarchy"]


def test_kg_failed_dispatch_still_closes_the_loop_with_an_empty_proposal() -> None:
    c = kg_exploration_contract()
    state = c.decode(KG_BODY, _attrs())
    ev = c.result(state, {"status": "FAILED", "output_payload": "", "error_message": "FAILED: unknown_request_source"})
    assert ev is not None
    assert ev.body["concepts"] == [] and ev.body["edges"] == [] and ev.body["model_used"] == ""
    assert ev.idempotency_key == f"req.{EVENT}"


def test_kg_non_json_answer_is_an_empty_proposal_not_an_error() -> None:
    c = kg_exploration_contract()
    ev = c.result(c.decode(KG_BODY, _attrs()), {"status": "OK", "output_payload": "sorry, no", "error_message": ""})
    assert ev is not None and ev.body["concepts"] == [] and ev.body["edges"] == []


# --------------------------------------------------------------------------- #
# companion_reflection
# --------------------------------------------------------------------------- #

GK_BODY = {
    "familiar_id": "fam-1",
    "familiar_name": "Ember",
    "goal_id": "goal-1",
    "goal_title": "Cell transport",
    "root_concept_id": "c-root",
    "concepts_total": 9,
    "concepts_mastered": 4,
    "shaky_concepts": [{"concept_label": "Osmosis", "strength": 0.3}],
    "memories": [{"content": "Learner: what is tonicity?\nFamiliar: ...", "created_at": "2026-08-20T10:00:00Z"}],
    "content_hash": "h-123",
    "prompt_version": "v1",
    "requested_at": "2026-08-22T17:00:00Z",
    "request_source": "goal_knowledge_lazy_regen",
}


def test_reflection_contract_identity() -> None:
    c = companion_reflection_contract()
    assert c.name == "companion_reflection" and c.role == "companion_chat"
    assert c.request_kind == "goal_knowledge.synthesis_requested"


def test_reflection_decode_requires_the_fields_consumption_needs_and_skips_no_signal() -> None:
    c = companion_reflection_contract()
    for missing in ("familiar_id", "goal_id", "content_hash"):
        with pytest.raises(ValueError, match=missing):
            c.decode({**GK_BODY, missing: ""}, _attrs())
    with pytest.raises(ValueError, match="tenant_id"):
        c.decode(GK_BODY, _attrs(tenant_id=""))
    skipped = c.decode({**GK_BODY, "shaky_concepts": [], "memories": []}, _attrs())
    assert isinstance(skipped, Skipped) and skipped.reason == "no_signal"
    state = c.decode(GK_BODY, _attrs())
    assert state["request"]["familiar_name"] == "Ember" and state["request"]["concepts_total"] == 9


def test_reflection_dispatch_is_a_reflect_turn_with_familiar_id_and_reflection_json() -> None:
    c = companion_reflection_contract()
    spec = c.dispatch(c.decode(GK_BODY, _attrs()))
    assert spec.execution_id == f"{EVENT}:reflect" and spec.workflow_id == EVENT
    payload = spec.input_payload
    assert payload["turn_kind"] == "reflect" and payload["familiar_id"] == "fam-1"
    assert "conversation_id" not in payload, "reflect is stateless (WP-A: per-dispatch session)"
    reflection = json.loads(payload["reflection_json"])
    assert reflection == {
        "companion_name": "Ember",
        "goal_title": "Cell transport",
        "concepts_total": 9,
        "concepts_mastered": 4,
        "shaky_concepts": [{"concept_label": "Osmosis", "strength": 0.3}],
        "memories": [{"content": "Learner: what is tonicity?\nFamiliar: ...", "created_at": "2026-08-20T10:00:00Z"}],
    }
    assert payload["gcid"] == GCID and payload["traceparent"] == "00-aa-bb-01"


def _env(**over: object) -> str:
    base = {
        "turn_kind": "reflect",
        "reply_text": "You kept circling back to osmosis; let us pin tonicity next.",
        "grounding": [],
        "tool_calls": [],
        "model_id": "gemini-2.5-flash",
        "prompt_version": "reflect-v1",
        "prompt_source": "embedded_fallback",
    }
    base.update(over)
    return json.dumps(base)


def test_reflection_result_publishes_the_synthesized_body_with_verbatim_echoes() -> None:
    c = companion_reflection_contract()
    state = c.decode(GK_BODY, _attrs())
    ev = c.result(state, {"status": "OK", "output_payload": _env(), "error_message": ""})
    assert ev is not None
    assert ev.topic == TOPIC_GOAL_KNOWLEDGE_SYNTHESIZED == "chora.consumption.goal_knowledge.synthesized.v1"
    assert ev.event_type == "consumption.goal_knowledge.synthesized"
    assert ev.idempotency_key == f"req.{EVENT}", "derived from the request key: a redelivery dedupes"
    body = ev.body
    assert body["synthesis_text"].startswith("You kept circling")
    assert body["nothing_to_say"] is False
    assert body["generated_by_run_id"] == EVENT and body["generated_by_model_id"] == "gemini-2.5-flash"
    assert body["prompt_version"] == "reflect-v1"
    assert body["content_hash"] == "h-123" and body["root_concept_id"] == "c-root"
    assert body["tenant_id"] == TENANT and body["learner_gcid"] == GCID
    assert body["familiar_id"] == "fam-1" and body["goal_id"] == "goal-1" and body["generated_at"]
    assert set(body) == {
        "tenant_id",
        "learner_gcid",
        "familiar_id",
        "goal_id",
        "synthesis_text",
        "generated_by_run_id",
        "generated_by_model_id",
        "prompt_version",
        "nothing_to_say",
        "content_hash",
        "root_concept_id",
        "generated_at",
    }


def test_reflection_decline_publishes_nothing_to_say() -> None:
    c = companion_reflection_contract()
    ev = c.result(
        c.decode(GK_BODY, _attrs()),
        {"status": "OK", "output_payload": _env(reply_text="", nothing_to_say=True), "error_message": ""},
    )
    assert ev is not None
    assert ev.body["nothing_to_say"] is True and ev.body["synthesis_text"] == ""
    assert ev.body["generated_by_model_id"] == "gemini-2.5-flash"


@pytest.mark.parametrize("case", ["no_model", "too_long", "empty", "refusal", "failed", "not_json"])
def test_reflection_publishes_nothing_when_the_answer_cannot_be_cached_honestly(case: str) -> None:
    c = companion_reflection_contract()
    state = c.decode(GK_BODY, _attrs())
    completion = {"status": "OK", "output_payload": "", "error_message": ""}
    if case == "no_model":
        completion["output_payload"] = _env(model_id="")
    elif case == "too_long":
        completion["output_payload"] = _env(reply_text="x" * (MAX_SYNTHESIS_CHARS + 1))
    elif case == "empty":
        completion["output_payload"] = _env(reply_text="   ")
    elif case == "refusal":
        completion["output_payload"] = _env(reply_text="", refusal_reason="armor_block")
    elif case == "failed":
        completion = {"status": "FAILED", "output_payload": "", "error_message": "FAILED: invalid_reflection_json"}
    elif case == "not_json":
        completion["output_payload"] = "plain prose"
    assert c.result(state, completion) is None


def test_reflection_unwraps_a_fenced_or_quoted_reply() -> None:
    c = companion_reflection_contract()
    ev = c.result(
        c.decode(GK_BODY, _attrs()),
        {"status": "OK", "output_payload": _env(reply_text='"Quoted reflection."'), "error_message": ""},
    )
    assert ev is not None and ev.body["synthesis_text"] == "Quoted reflection."
