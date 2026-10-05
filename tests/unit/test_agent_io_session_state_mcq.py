"""MCQ session-state + response-unwrap contract for adapter/agent_io.

Ported from the deleted `test_executor_state_mapping_mcq.py` when the HTTP
`ReasoningEngineExecutor` was removed (RULING A, 2026-08-23). The assertions
are unchanged; what moved is the seam they are taken at. The old tests drove
`ReasoningEngineExecutor.execute()` through a fake httpx client and read the
session state back out of the captured `:query` POST body. The lanes no longer
speak HTTP, so the same two pure functions are exercised directly:

* `build_session_state` builds the state the ADK Go composer branches on
  (the qgen_question `fill_mcq` / `new_mcq` template branches, the W8 image
  opt-in flags, the critic's attempt counters).
* `map_agent_response` unwraps the evaluator's `scored.candidate` wrapper to
  the flat candidate the qgen crew nodes expect.

No live LLM calls, no transport: both functions are pure.
"""

from __future__ import annotations

import json

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    ROLE_QGEN_CRITIC,
    ROLE_QGEN_QUESTION,
    build_session_state,
    map_agent_response,
)

# -----------------------------------------------------------------------------
# build_session_state: MCQ-specific keys for qgen_question
# -----------------------------------------------------------------------------


class TestSessionStateForMCQ:
    def test_mcq_model_answer_fill_state_keys(self) -> None:
        """qgen_question + question_type=mcq + intent=model_answer_fill
        produces the right state keys for the fill_mcq template branch.

        Per composer_question.go resolveTaskAndOutput the fill_mcq path is
        triggered when (Intent=model_answer_fill, QuestionType=mcq). The
        kennel exposes both keys via session state so the ADK Go agent's
        instruction provider can branch correctly.
        """
        author_stem = "Author stem (preserved verbatim)"
        state = build_session_state(
            agent_role=ROLE_QGEN_QUESTION,
            execution_id="job-mcq-fill:generate:1",
            tenant_id="00000000-0000-7000-8000-000000000001",
            input_obj={
                "prompt": author_stem,
                "intent": "model_answer_fill",
                "question_type": "mcq",
                "gcid": "gcid-mcq-author-1",
            },
        )

        # Core mana-plugin keys.
        assert state["tenant_id"] == "00000000-0000-7000-8000-000000000001"
        assert state["user_gcid"] == "gcid-mcq-author-1"
        assert state["author_gcid"] == "gcid-mcq-author-1"

        # MCQ-specific branching keys.
        assert state["intent"] == "model_answer_fill"
        assert state["question_type"] == "mcq"
        assert state["input_payload"] == author_stem

    def test_mcq_new_question_state_keys(self) -> None:
        """qgen_question + question_type=mcq + intent=new_question produces
        the right state keys for the new_mcq template branch."""
        state = build_session_state(
            agent_role=ROLE_QGEN_QUESTION,
            execution_id="job-mcq-new:generate:1",
            tenant_id="tenant-mcq-1",
            input_obj={
                "prompt": "Generate a MCQ on photosynthesis",
                "intent": "new_question",
                "question_type": "mcq",
            },
        )

        # new_mcq branch keys.
        assert state["intent"] == "new_question"
        assert state["question_type"] == "mcq"
        # Mana plugin gcid synthesised when not supplied.
        assert state["user_gcid"].startswith("qgen-anon:")
        assert state["author_gcid"] == state["user_gcid"]

    def test_mcq_forwards_image_optin_flags_to_session_state(self) -> None:
        """W8: the author's per-image opt-in flags (image_for_stem /
        image_for_answer) MUST land on session state so the ADK Go composer's
        BuildTaskContextFromState (readStateBool) emits the [IMAGE] block and
        the image_specs output-schema fragment.

        Regression guard for the 2026-06-02 e2e gap: generate_node forwarded
        the flags in input_payload, but the state builder dropped them, so the
        composer always read false, no image_specs, render stayed dormant and
        no images were rendered despite both drawer toggles being on.
        """
        state = build_session_state(
            agent_role=ROLE_QGEN_QUESTION,
            execution_id="job-mcq-img:generate:1",
            tenant_id="tenant-img-1",
            input_obj={
                "prompt": "Generate a beginner MCQ about the water cycle",
                "intent": "new_question",
                "question_type": "mcq",
                "image_for_stem": True,
                "image_for_answer": True,
            },
        )

        # The two W8 opt-in flags must reach the Go composer via session state.
        assert state["image_for_stem"] is True
        assert state["image_for_answer"] is True

    def test_mcq_image_optin_flags_default_false_when_absent(self) -> None:
        """W8: when the author opted into NEITHER part (flags absent from the
        input object, the pre-W8 default), session state carries both as an
        explicit False so the composer's readStateBool path stays dormant
        (byte-identical pre-W8 prompt; render is a pure pass-through)."""
        state = build_session_state(
            agent_role=ROLE_QGEN_QUESTION,
            execution_id="job-mcq-noimg:generate:1",
            tenant_id="tenant-noimg-1",
            input_obj={
                "prompt": "Generate a beginner MCQ about the water cycle",
                "intent": "new_question",
                "question_type": "mcq",
            },
        )

        assert state["image_for_stem"] is False
        assert state["image_for_answer"] is False

    def test_mcq_critic_state_keys(self) -> None:
        """qgen_critic + question_type=mcq carries job_id + attempt_index +
        max_attempts on session state per the critic's instruction template.
        The full candidate JSON travels under input_payload."""
        candidate_payload = {
            "stem": "What is mitochondria?",
            "question_type": "mcq",
            "options": [
                {"option_id": "a", "label": "Powerhouse", "is_correct": True},
                {"option_id": "b", "label": "Storehouse", "is_correct": False},
            ],
        }
        state = build_session_state(
            agent_role=ROLE_QGEN_CRITIC,
            execution_id="job-mcq-crit:critique:2",
            tenant_id="tenant-crit-1",
            input_obj=candidate_payload,
        )

        # Critic-specific keys.
        assert state["job_id"] == "job-mcq-crit:critique:2"
        assert state["question_type"] == "mcq"
        assert state["attempt_index"] == 0
        assert state["max_attempts"] == 4
        # The full candidate JSON travels via input_payload so the critic's
        # instruction template can inspect the options and stem.
        decoded_input = json.loads(state["input_payload"])
        assert decoded_input["stem"] == "What is mitochondria?"
        assert decoded_input["question_type"] == "mcq"
        assert len(decoded_input["options"]) == 2


# -----------------------------------------------------------------------------
# map_agent_response: MCQ candidate unwrap preserves load-bearing fields
# -----------------------------------------------------------------------------


class TestResponseUnwrapForMCQ:
    def test_unwraps_mcq_candidate_and_preserves_options(self) -> None:
        """When the qgen_question evaluator wraps the MCQ candidate under
        `scored.candidate`, the unwrap puts the candidate fields at the top
        level of `output_payload`. The score fields move under the `_scored`
        sibling key."""
        mcq_options = [
            {
                "option_id": "a",
                "label": "Cell membrane",
                "is_correct": False,
                "explainer": "Membrane controls transport, not energy.",
            },
            {
                "option_id": "b",
                "label": "Mitochondria",
                "is_correct": True,
                "explainer": "Mitochondria are the site of ATP production.",
            },
            {
                "option_id": "c",
                "label": "Ribosomes",
                "is_correct": False,
                "explainer": "Ribosomes synthesise proteins, not ATP.",
            },
            {
                "option_id": "d",
                "label": "Lysosomes",
                "is_correct": False,
                "explainer": "Lysosomes digest waste; not ATP production.",
            },
        ]
        evaluator_text = json.dumps(
            {
                "scored": {
                    "candidate": {
                        "stem": "Where is ATP produced?",
                        "options": mcq_options,
                        "intent": "new_question",
                        "question_type": "mcq",
                    },
                    "factuality": 0.95,
                    "clarity": 0.9,
                    "difficulty": 0.7,
                    "composite": 0.85,
                }
            }
        )

        resp = map_agent_response(
            execution_id="unwrap-mcq-1",
            terminal_text=evaluator_text,
            agent_role=ROLE_QGEN_QUESTION,
        )

        decoded = json.loads(resp.output_payload)
        # Flat candidate shape: options at the top level, not nested under the
        # `_scored` sibling key.
        assert decoded["stem"] == "Where is ATP produced?"
        assert decoded["question_type"] == "mcq"
        assert decoded["intent"] == "new_question"

        assert isinstance(decoded["options"], list)
        assert len(decoded["options"]) == 4
        for i, opt in enumerate(decoded["options"]):
            assert opt["option_id"] == mcq_options[i]["option_id"]
            assert opt["label"] == mcq_options[i]["label"]
            assert opt["is_correct"] == mcq_options[i]["is_correct"]
            assert opt["explainer"] == mcq_options[i]["explainer"]

        correct = [o for o in decoded["options"] if o["is_correct"]]
        assert len(correct) == 1
        assert correct[0]["option_id"] == "b"

        assert "_scored" in decoded
        scored = decoded["_scored"]
        assert scored["factuality"] == 0.95
        assert scored["clarity"] == 0.9
        assert scored["difficulty"] == 0.7
        assert scored["composite"] == 0.85
        # The candidate itself is not duplicated inside the score sibling.
        assert "candidate" not in scored

    def test_unwraps_mcq_fill_candidate_with_filled_explainers(self) -> None:
        """The fill_mcq template's evaluator output carries the verbatim author
        stem plus options with FILLED explainers; the unwrap MUST preserve
        those filled fields at the top level."""
        filled_options = [
            {
                "option_id": "a",
                "label": "Author label A",
                "is_correct": False,
                "explainer": "Filled by agent: explains why A is wrong.",
            },
            {
                "option_id": "b",
                "label": "Author label B",
                "is_correct": True,
                "explainer": "Filled by agent: affirms B is correct.",
            },
        ]
        evaluator_text = json.dumps(
            {
                "scored": {
                    "candidate": {
                        "stem": "Author stem verbatim",
                        "options": filled_options,
                        "intent": "model_answer_fill",
                        "question_type": "mcq",
                    },
                    "composite": 0.82,
                }
            }
        )

        resp = map_agent_response(
            execution_id="unwrap-mcq-fill-1",
            terminal_text=evaluator_text,
            agent_role=ROLE_QGEN_QUESTION,
        )

        decoded = json.loads(resp.output_payload)
        assert decoded["intent"] == "model_answer_fill"
        assert decoded["question_type"] == "mcq"
        assert decoded["stem"] == "Author stem verbatim"
        # Each option carries its filled explainer.
        for i, opt in enumerate(decoded["options"]):
            assert opt["explainer"] == filled_options[i]["explainer"]
            assert opt["explainer"].strip(), f"option[{i}] explainer is empty after unwrap: {opt!r}"
        # _scored carries the composite from the evaluator.
        assert decoded["_scored"]["composite"] == 0.82
