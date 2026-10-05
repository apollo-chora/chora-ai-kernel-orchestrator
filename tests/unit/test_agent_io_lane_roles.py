"""RED: the transport-neutral agent I/O helpers serve the ADR-254 lane roles.

The Go qgen binaries merge the dispatch ``input_payload`` into ADK session
state VERBATIM (``agentdispatch.Request.SessionState``), so the kennel must
ship the session-state shape the composer reads (``input_payload`` = the prompt
text, ``subject_hint``, ``type_plan_json``, ...). That transform used to live
inside the HTTP executor (``_build_session_state``); it moves to a
transport-neutral module and must treat the lane roles (``qgen_generate`` /
``qgen_critique``) exactly as it treats the agent ids (``qgen_question`` /
``qgen_critic``). The response mapper likewise unwraps the generator's
``candidate`` envelope for the lane role, or the FE would see the candidate
double-nested again.
"""

from __future__ import annotations

import json

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    LANE_ROLE_CRITIQUE,
    LANE_ROLE_GENERATE,
    LANE_ROLE_RENDER,
    build_session_state,
    map_agent_response,
)


def test_lane_role_constants_are_the_go_dispatch_roles() -> None:
    assert (LANE_ROLE_GENERATE, LANE_ROLE_CRITIQUE, LANE_ROLE_RENDER) == (
        "qgen_generate",
        "qgen_critique",
        "qgen_render",
    )


def test_generate_lane_role_builds_the_question_shape() -> None:
    plan = [{"question_type": "mcq", "count": 2, "max_images": 0}]
    state = build_session_state(
        agent_role=LANE_ROLE_GENERATE,
        execution_id="job-1:generate_set:c0:r0",
        tenant_id="11111111-1111-7111-8111-111111111111",
        input_obj={
            "prompt": "Photosynthesis, P5",
            "question_type": "mcq",
            "metadata": {"subject": "Science", "difficulty": "medium"},
            "type_plan": plan,
            "avoid_concepts": ["stomata"],
            "gcid": "00000000-0000-7000-8000-000000001999",
            "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01",
            "image_for_stem": True,
        },
    )
    # The composer reads the prompt under input_payload and the hints as *_hint.
    assert state["input_payload"] == "Photosynthesis, P5"
    assert state["subject_hint"] == "Science"
    assert state["difficulty_hint"] == "medium"
    assert "cognitive_level_hint" not in state
    assert state["set_mode"] is True
    assert json.loads(state["type_plan_json"]) == plan
    assert json.loads(state["avoid_concepts_json"]) == ["stomata"]
    assert state["image_for_stem"] is True and state["image_for_answer"] is False
    assert state["intent"] == "new_question" and state["question_type"] == "mcq"
    # Envelope-bound keys stay (the Pub/Sub executor lifts gcid + trace context
    # from the payload into the envelope; the Go merge skips them).
    assert state["user_gcid"] == "00000000-0000-7000-8000-000000001999"
    assert state["author_gcid"] == state["user_gcid"]
    assert state["traceparent"].startswith("00-")
    assert state["tenant_id"] == "11111111-1111-7111-8111-111111111111"


def test_critique_lane_role_builds_the_critic_shape() -> None:
    candidate = {
        "stem": "Which gas?",
        "question_type": "mcq",
        "author_prompt": "P",
        "attempt_index": 1,
        "prior_critic_notes": "weak distractors",
        "metadata": {"cognitive_level": "remember"},
        "set_mode": True,
        "candidates": [{"candidate_id": "c0", "stem": "Which gas?"}],
    }
    state = build_session_state(
        agent_role=LANE_ROLE_CRITIQUE,
        execution_id="job-1:critique_set:c0:r0",
        tenant_id="11111111-1111-7111-8111-111111111111",
        input_obj=candidate,
    )
    assert state["job_id"] == "job-1:critique_set:c0:r0"
    assert state["attempt_index"] == 1
    assert state["prior_critic_notes"] == "weak distractors"
    assert state["cognitive_level_hint"] == "remember"
    assert state["set_mode"] is True
    assert json.loads(state["input_payload"])["candidates"][0]["candidate_id"] == "c0"


def test_legacy_agent_id_and_lane_role_build_identical_state() -> None:
    payload = {"prompt": "P", "question_type": "oe", "gcid": "g"}
    legacy = build_session_state(agent_role="qgen_question", execution_id="e", tenant_id="t", input_obj=payload)
    lane = build_session_state(agent_role=LANE_ROLE_GENERATE, execution_id="e", tenant_id="t", input_obj=payload)
    assert legacy == lane
    legacy_c = build_session_state(agent_role="qgen_critic", execution_id="e", tenant_id="t", input_obj=payload)
    lane_c = build_session_state(agent_role=LANE_ROLE_CRITIQUE, execution_id="e", tenant_id="t", input_obj=payload)
    assert legacy_c == lane_c


def test_render_lane_role_is_a_verbatim_merge_plus_identity() -> None:
    """qgen_render has no HTTP-era shape: every key the node threads rides as
    is (render_prompt / mode / source_image_uri / job_id / chunk_id)."""
    state = build_session_state(
        agent_role=LANE_ROLE_RENDER,
        execution_id="job-1:render:c0:i1:stem",
        tenant_id="t",
        input_obj={"render_prompt": "a red apple", "mode": "scene", "job_id": "job-1", "chunk_id": "c0", "gcid": "g"},
    )
    assert state["render_prompt"] == "a red apple"
    assert state["mode"] == "scene" and state["job_id"] == "job-1" and state["chunk_id"] == "c0"
    assert state["tenant_id"] == "t" and state["user_gcid"] == "g"


def test_map_agent_response_unwraps_candidate_for_the_generate_lane_role() -> None:
    text = json.dumps({"candidate": {"stem": "S", "question_type": "mcq"}, "input_tokens": 3, "output_tokens": 4})
    resp = map_agent_response(execution_id="e", terminal_text=text, agent_role=LANE_ROLE_GENERATE)
    out = json.loads(resp.output_payload)
    assert out["stem"] == "S", "the candidate must be unwrapped for the lane role exactly as for qgen_question"
    assert resp.input_tokens == 3 and resp.output_tokens == 4 and resp.tokens_consumed_total == 7


def test_map_agent_response_passes_set_wrapper_through_for_the_generate_lane_role() -> None:
    text = json.dumps({"candidates": [{"stem": "a"}], "generation_summary": {"requested_total": 1}})
    resp = map_agent_response(execution_id="e", terminal_text=text, agent_role=LANE_ROLE_GENERATE)
    out = json.loads(resp.output_payload)
    assert out["candidates"] == [{"stem": "a"}] and "generation_summary" in out


def test_compose_mode_on_the_generate_lane_ships_the_port_keys_verbatim() -> None:
    """ADR-254 D2: mode=compose rides the qgen_generate lane; the Go port reads
    mode / candidates / author_prompt / metadata / source_files / grounding_mode
    from session state, so the kennel ships exactly those (no generation keys)."""
    state = build_session_state(
        agent_role=LANE_ROLE_GENERATE,
        execution_id="job-1:compose",
        tenant_id="t1",
        input_obj={
            "mode": "compose",
            "candidates": [{"draft_id": "d0", "stem": "A"}, {"draft_id": "d1", "stem": "B"}],
            "author_prompt": "Create a test set like the uploaded paper.",
            "metadata": {"subject": "Biology"},
            "source_files": [
                {"gs_uri": "gs://b/exam.pdf", "mime_type": "application/pdf", "role": "source"},
                {"gs_uri": "gs://b/marks.pdf", "mime_type": "application/pdf", "role": "rubric"},
            ],
            "grounding_mode": "strict",
            "gcid": "g1",
            "traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01",
        },
    )
    assert state["mode"] == "compose"
    assert [c["draft_id"] for c in state["candidates"]] == ["d0", "d1"]
    assert state["author_prompt"] == "Create a test set like the uploaded paper."
    assert state["metadata"] == {"subject": "Biology"}
    assert [f["role"] for f in state["source_files"]] == ["source", "rubric"]
    assert state["grounding_mode"] == "strict"
    assert state["tenant_id"] == "t1" and state["user_gcid"] == "g1"
    for absent in ("input_payload", "type_plan_json", "set_mode", "intent", "image_for_stem"):
        assert absent not in state, absent


def test_compose_mode_without_source_files_omits_the_key() -> None:
    state = build_session_state(
        agent_role=LANE_ROLE_GENERATE,
        execution_id="j:compose",
        tenant_id="t",
        input_obj={"mode": "compose", "candidates": [], "author_prompt": "p"},
    )
    assert state["mode"] == "compose" and "source_files" not in state
