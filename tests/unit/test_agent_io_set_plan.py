"""Unit tests — P1a executor dual-shape (CHO-1819 single-pass set generation).

Two pure pieces (no live LLM):

1. ``stamp_set_plan`` (via ``build_session_state``, qgen_question role) stamps
   the single-pass SET keys the ADK Go composer reads in set mode — ``set_mode``
   + ``type_plan_json`` (+ ``avoid_concepts_json``). An ABSENT/empty type_plan
   stamps NO set keys, so the live single-candidate / legacy-batch path stays
   byte-identical.

2. ``map_agent_response`` accepts BOTH the new object-wrapper
   ``{"candidates": [...], "generation_summary": {...}}`` (passed through
   unchanged so the set graph's generate_set node receives the array) AND the
   legacy ``{"candidate": {...}}`` / ``{"scored": {"candidate": {...}}}`` single
   shapes (still unwrapped) — the rollout-safety contract while ONE deployed
   agent serves both the live single path and the new set path.
"""

from __future__ import annotations

import json

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    ROLE_QGEN_QUESTION,
    build_session_state,
    map_agent_response,
)

_TENANT = "11111111-1111-7111-8111-111111111111"


def _state(extra: dict) -> dict:
    base = {
        "prompt": "Generate questions.",
        "question_type": "mcq",
        "intent": "new_question",
        "gcid": "00000000-0000-7000-8000-000000001999",
    }
    base.update(extra)
    return build_session_state(
        agent_role=ROLE_QGEN_QUESTION,
        execution_id="job-set",
        tenant_id=_TENANT,
        input_obj=base,
    )


class TestStampSetPlan:
    def test_type_plan_stamps_set_mode_and_json(self) -> None:
        plan = [
            {"question_type": "mcq", "count": 8, "max_images": 3},
            {"question_type": "oe", "count": 2, "max_images": 1},
        ]
        s = _state({"type_plan": plan, "set_mode": True})
        assert s["set_mode"] is True
        assert json.loads(s["type_plan_json"]) == plan

    def test_absent_type_plan_stamps_no_set_keys(self) -> None:
        s = _state({})
        assert "type_plan_json" not in s
        assert "set_mode" not in s
        assert "avoid_concepts_json" not in s

    def test_avoid_concepts_stamped_as_json(self) -> None:
        s = _state(
            {
                "type_plan": [{"question_type": "mcq", "count": 1, "max_images": 0}],
                "avoid_concepts": ["Newton's first law", "inertia"],
            }
        )
        assert json.loads(s["avoid_concepts_json"]) == [
            "Newton's first law",
            "inertia",
        ]

    def test_empty_type_plan_list_is_legacy(self) -> None:
        # An explicitly-empty plan ⇒ legacy single-type path (no set keys), so
        # the deployed single/legacy-batch traffic is byte-identical.
        s = _state({"type_plan": []})
        assert "type_plan_json" not in s
        assert "set_mode" not in s

    def test_per_type_image_flags_survive_into_type_plan_json(self) -> None:
        # CHO-1825 — the deterministic per-type image opt-ins ride through the
        # executor stamp (json.dumps passthrough) so the Go set-mode composer
        # reads them off type_plan_json and FORCES the image per question.
        plan = [
            {"question_type": "mcq", "count": 2, "max_images": 0, "image_for_stem": True},
            {"question_type": "oe", "count": 1, "max_images": 0, "image_for_answer": True},
        ]
        s = _state({"type_plan": plan, "set_mode": True})
        assert json.loads(s["type_plan_json"]) == plan


class TestMapResponseDualShape:
    def _resp(self, payload: dict):  # type: ignore[no-untyped-def]
        return map_agent_response(
            execution_id="j",
            terminal_text=json.dumps(payload),
            agent_role=ROLE_QGEN_QUESTION,
        )

    def test_set_wrapper_passes_through_unchanged(self) -> None:
        wrapper = {
            "candidates": [
                {
                    "stem": "Q1",
                    "question_type": "mcq",
                    "options": [{"option_id": "a", "label": "x", "is_correct": True}],
                },
                {
                    "stem": "Q2",
                    "question_type": "oe",
                    "oe_payload": {"model_answer": "..."},
                },
            ],
            "generation_summary": {
                "requested_total": 2,
                "generated_total": 2,
                "per_type": {
                    "mcq": {"requested": 1, "generated": 1},
                    "oe": {"requested": 1, "generated": 1},
                },
                "shortfall_reason": "",
            },
            "input_tokens": 100,
            "output_tokens": 200,
        }
        r = self._resp(wrapper)
        out = json.loads(r.output_payload)
        assert out["candidates"] == wrapper["candidates"]
        assert out["generation_summary"]["generated_total"] == 2
        # Token split captured from the top level (not lost to an unwrap).
        assert r.input_tokens == 100
        assert r.output_tokens == 200

    def test_legacy_bare_candidate_still_unwraps(self) -> None:
        legacy = {
            "candidate": {
                "stem": "Q",
                "question_type": "mcq",
                "options": [{"option_id": "a", "label": "x", "is_correct": True}],
            }
        }
        out = json.loads(self._resp(legacy).output_payload)
        assert out["stem"] == "Q"  # flattened
        assert "candidate" not in out

    def test_legacy_scored_candidate_still_unwraps(self) -> None:
        scored = {
            "scored": {
                "candidate": {"stem": "S", "question_type": "mcq"},
                "composite": 0.9,
            }
        }
        out = json.loads(self._resp(scored).output_payload)
        assert out["stem"] == "S"
        assert out["_scored"]["composite"] == 0.9
