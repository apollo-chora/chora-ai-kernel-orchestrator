"""Open-ended (OE) session-state contract for adapter/agent_io.

Ported from the deleted `test_executor_state_mapping_oe.py` when the HTTP
`ReasoningEngineExecutor` was removed (RULING A, 2026-08-23). The assertions
are unchanged; the seam moved. The old tests drove the executor through a fake
httpx client and read session state back out of the captured `:query` POST
body. The lanes ship that same state as the dispatch payload now, so the pure
builder is exercised directly.

Two load-bearing shapes, and they deliberately differ:

* qgen_question: `input_payload` is the author's PROMPT STRING verbatim, so
  the ADK Go composer's `taskNewOE` substitutes natural language into the
  [INPUT] block.
* qgen_critic: `input_payload` is the JSON-encoded WHOLE candidate, so the
  critic's `taskCritiqueOE` can parse stem, model answer and rubric.

No live LLM calls, no transport: `build_session_state` is pure.
"""

from __future__ import annotations

import json

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    ROLE_QGEN_CRITIC,
    ROLE_QGEN_QUESTION,
    build_session_state,
)

# ---------------------------------------------------------------------------
# qgen_question with question_type=oe
# ---------------------------------------------------------------------------


class TestOEGenerateSessionState:
    def test_input_payload_is_plain_prompt_string_not_json(self) -> None:
        """`state["input_payload"]` must be the AUTHOR'S PROMPT STRING
        verbatim, not a JSON-encoded blob of the whole payload.

        This is the load-bearing invariant for the qgen_question ADK Go
        composer: `taskNewOE` substitutes `{{input_payload}}` into the [INPUT]
        block and the LLM reads it as natural language. If the orchestrator
        re-JSON-encoded the payload here, the LLM would see escape sequences
        instead of the author's question. The qgen_question branch of
        `build_session_state` writes
        `state["input_payload"] = str(input_obj.get("prompt") or "")`.
        """
        prompt_text = (
            "Generate one open-ended question on cellular respiration. "
            'It should mention "mitochondria" and ask the learner to '
            "compare aerobic vs anaerobic pathways."
        )
        state = build_session_state(
            agent_role=ROLE_QGEN_QUESTION,
            execution_id="oe-new-1",
            tenant_id="00000000-0000-7000-8000-000000000099",
            input_obj={
                "intent": "new_question",
                "question_type": "oe",
                "prompt": prompt_text,
                "metadata": {"subject": "biology"},
                "attempt_index": 0,
                "prior_critic_notes": "",
            },
        )

        # CRITICAL: input_payload must be the prompt as a plain string, NOT a
        # JSON-encoded object.
        assert state["input_payload"] == prompt_text, (
            f"input_payload must be the raw prompt string, got {state['input_payload']!r}"
        )
        # And specifically NOT a JSON object (no leading '{', no escaping of
        # the embedded quotes).
        assert not state["input_payload"].startswith("{")
        assert '\\"mitochondria\\"' not in state["input_payload"]

    def test_question_type_oe_and_intent_threaded(self) -> None:
        """`state["question_type"] == "oe"` and `state["intent"] ==
        "new_question"` flow into the agent session state so the agent picks
        the new_oe template branch."""
        state = build_session_state(
            agent_role=ROLE_QGEN_QUESTION,
            execution_id="oe-new-2",
            tenant_id="t",
            input_obj={
                "intent": "new_question",
                "question_type": "oe",
                "prompt": "p",
            },
        )
        assert state["question_type"] == "oe"
        assert state["intent"] == "new_question"

    def test_intent_model_answer_fill_threaded_for_oe(self) -> None:
        """`intent=model_answer_fill` for OE must thread through so the agent
        picks the fill_oe template branch (composer_question.go 307-308)."""
        state = build_session_state(
            agent_role=ROLE_QGEN_QUESTION,
            execution_id="oe-fill-1",
            tenant_id="t",
            input_obj={
                "intent": "model_answer_fill",
                "question_type": "oe",
                "prompt": "fill prompt",
            },
        )
        assert state["intent"] == "model_answer_fill"
        assert state["question_type"] == "oe"


# ---------------------------------------------------------------------------
# qgen_critic with question_type=oe
# ---------------------------------------------------------------------------


class TestOECritiqueSessionState:
    def test_input_payload_carries_full_candidate_json(self) -> None:
        """For the critic, `state["input_payload"]` is the JSON-encoded
        candidate (the WHOLE OE payload, not just a stem).

        The qgen_critic branch writes `"input_payload": json.dumps(input_obj)`
        so the critic's `taskCritiqueOE` template can parse the full candidate
        per its prompt instructions (critic.go 274-278).
        """
        candidate_in = {
            "stem": "Explain photosynthesis.",
            "question_type": "oe",
            "oe_payload": {
                "model_answer": ("photosynthesis converts light to chemical energy via chlorophyll."),
                "rubric": [
                    {"criterion_id": "c1", "title": "x", "weight": 0.5},
                    {"criterion_id": "c2", "title": "y", "weight": 0.5},
                ],
                "grader_tier": "T1",
            },
        }
        state = build_session_state(
            agent_role=ROLE_QGEN_CRITIC,
            execution_id="oe-job-1:critique:1",
            tenant_id="t-1",
            input_obj=candidate_in,
        )

        # The critic state's input_payload must round-trip the full candidate
        # so the critic prompt can read it.
        roundtrip = json.loads(state["input_payload"])
        assert roundtrip["stem"] == candidate_in["stem"]
        assert roundtrip["question_type"] == "oe"
        assert roundtrip["oe_payload"]["model_answer"] == (candidate_in["oe_payload"]["model_answer"])
        assert roundtrip["oe_payload"]["rubric"] == candidate_in["oe_payload"]["rubric"]
        assert roundtrip["oe_payload"]["grader_tier"] == "T1"

    def test_oe_quality_loop_state_keys_present(self) -> None:
        """The critic state MUST carry the quality-loop context keys: job_id,
        attempt_index, max_attempts and prior_critic_notes.

        These come from the qgen_crew LangGraph state per `critique_node` and
        are rendered into the critic's prompt by the agent's instruction
        template. Without them the critic cannot tell whether this is a first
        attempt or a retry.
        """
        state = build_session_state(
            agent_role=ROLE_QGEN_CRITIC,
            execution_id="oe-job-42:critique:2",
            tenant_id="tenant-x",
            input_obj={
                "stem": "stem",
                "question_type": "oe",
                "oe_payload": {"model_answer": "...", "rubric": []},
                "attempt_index": 1,
                "max_attempts": 4,
                "prior_critic_notes": ("first-attempt critique was: rubric too sparse"),
                "prompt": "original author prompt",
            },
        )

        # Quality-loop keys.
        assert state["job_id"] == "oe-job-42:critique:2"
        assert state["question_type"] == "oe"
        assert state["attempt_index"] == 1
        assert state["max_attempts"] == 4
        assert state["prior_critic_notes"] == "first-attempt critique was: rubric too sparse"
        # author_prompt carries the original prompt for context.
        assert state["author_prompt"] == "original author prompt"

    def test_oe_critic_state_defaults_when_loop_keys_missing(self) -> None:
        """When `attempt_index` / `max_attempts` are absent from the input
        payload (orchestrator early-fail-loud scenario), the state derivation
        must still emit safe defaults (attempt_index=0, max_attempts=4,
        prior_critic_notes='') so the critic does not blow up on missing
        template variables.
        """
        state = build_session_state(
            agent_role=ROLE_QGEN_CRITIC,
            execution_id="oe-default-1",
            tenant_id="t",
            input_obj={
                "stem": "s",
                "question_type": "oe",
                "oe_payload": {"model_answer": "a", "rubric": []},
            },
        )
        assert state["attempt_index"] == 0
        assert state["max_attempts"] == 4
        assert state["prior_critic_notes"] == ""
