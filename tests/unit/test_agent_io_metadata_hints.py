"""Unit test: `build_session_state` stamps the author's Subject /
Cognitive Level / Difficulty selections as session-state HINT keys for the
qgen_question + qgen_critic roles, so the ADK Go composer + critic actually
condition generation on them (authoring-metadata wire-through, 2026-06-03).

Background: the FE forwards `metadata.{subject, cognitive_level, difficulty}`
end-to-end into `input_obj['metadata']` (generate_node already; critique_node
after the sibling fix). The ADK Go critic
(agents/qgen_adk_go/internal/agent/critic.go:477-479) ALREADY reads
`state['subject_hint']` / `state['cognitive_level_hint']` /
`state['difficulty_hint']`, and the Go composer reads the same keys after the
sibling change — but the Python executor never *stamped* them, so they were
always empty (the dead-wire the 2026-06-03 audit surfaced). This test closes
that producer gap.

No live LLM calls: `build_session_state` is a pure function.
"""

from __future__ import annotations

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    ROLE_QGEN_CRITIC,
    ROLE_QGEN_QUESTION,
    build_session_state,
)

_TENANT = "11111111-1111-7111-8111-111111111111"


def _state(role: str, metadata: dict) -> dict:
    return build_session_state(
        agent_role=role,
        execution_id="job-1",
        tenant_id=_TENANT,
        input_obj={
            "prompt": "Generate one MCQ on Newton's first law.",
            "question_type": "mcq",
            "intent": "new_question",
            "metadata": metadata,
            "gcid": "00000000-0000-7000-8000-000000001999",
        },
    )


class TestQuestionRoleStampsHints:
    def test_all_three_hints_stamped(self) -> None:
        s = _state(
            ROLE_QGEN_QUESTION,
            {
                "subject": "Physics — Newtonian mechanics",
                "cognitive_level": "synthesis",
                "difficulty": "advanced",
            },
        )
        assert s["subject_hint"] == "Physics — Newtonian mechanics"
        assert s["cognitive_level_hint"] == "synthesis"
        assert s["difficulty_hint"] == "advanced"

    def test_absent_metadata_stamps_no_hint_keys(self) -> None:
        s = _state(ROLE_QGEN_QUESTION, {})
        assert "subject_hint" not in s
        assert "cognitive_level_hint" not in s
        assert "difficulty_hint" not in s

    def test_partial_metadata_only_present_keys(self) -> None:
        s = _state(ROLE_QGEN_QUESTION, {"cognitive_level": "synthesis"})
        assert s["cognitive_level_hint"] == "synthesis"
        assert "subject_hint" not in s
        assert "difficulty_hint" not in s


class TestCriticRoleStampsHints:
    def test_all_three_hints_stamped(self) -> None:
        s = _state(
            ROLE_QGEN_CRITIC,
            {
                "subject": "Physics — Newtonian mechanics",
                "cognitive_level": "synthesis",
                "difficulty": "advanced",
            },
        )
        assert s["subject_hint"] == "Physics — Newtonian mechanics"
        assert s["cognitive_level_hint"] == "synthesis"
        assert s["difficulty_hint"] == "advanced"

    def test_absent_metadata_stamps_no_hint_keys(self) -> None:
        s = _state(ROLE_QGEN_CRITIC, {})
        assert "subject_hint" not in s
        assert "cognitive_level_hint" not in s
        assert "difficulty_hint" not in s
