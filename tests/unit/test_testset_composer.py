"""Test-set composition helpers (CHO-1703 W2; ADR-254 D2).

The LLM call moved onto the qgen_generate lane (``compose_test_set_node``,
covered by test_qgen_batch_payload_shape.py + test_qgen_runner_park_resume.py).
What stays in testset_composer.py is pure and deterministic: the D4 fallback
proposal, the contract repair of the agent's proposal, and the tolerant JSON
recovery for raw completions.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.orchestrators.testset_composer import (
    DEFAULT_POINTS,
    DESCRIPTION_MAX_CHARS,
    TITLE_MAX_CHARS,
    fallback_proposal,
    normalise_proposal,
    parse_completion_json,
)


class _Payload:
    def __init__(self, *, prompt: str = "", assist_id: str = "job-1") -> None:
        self.prompt = prompt
        self.assist_id = assist_id


def _cands(n: int) -> list[dict[str, str]]:
    return [{"draft_id": f"d-{i}", "stem": f"Q{i}", "question_type": "mcq"} for i in range(n)]


_FILES = [
    {"blob_uri": "gs://b/exam-paper.pdf", "mime_type": "application/pdf", "role": "source"},
    {"blob_uri": "gs://b/marks.pdf", "mime_type": "application/pdf", "role": "rubric"},
]


# -----------------------------------------------------------------------------
# fallback_proposal (D4, deterministic, total)
# -----------------------------------------------------------------------------


def test_fallback_proposal_title_from_source_basename() -> None:
    fb = fallback_proposal(payload=_Payload(prompt="ignored"), candidates=_cands(2), files=_FILES)
    assert fb["title"].endswith("exam-paper")
    assert "grounded on exam-paper.pdf" in fb["description"]
    assert fb["order"] == ["d-0", "d-1"]
    assert fb["points"] == {"d-0": DEFAULT_POINTS, "d-1": DEFAULT_POINTS}


def test_fallback_proposal_title_from_prompt_when_no_files() -> None:
    fb = fallback_proposal(payload=_Payload(prompt="Five MCQs about photosynthesis"), candidates=_cands(1), files=[])
    assert "Five MCQs about photosynthesis" in fb["title"]
    assert fb["description"] == "Auto-proposed from 1 generated question(s)."


def test_fallback_proposal_generic_title_when_nothing_to_derive() -> None:
    fb = fallback_proposal(payload=_Payload(prompt=""), candidates=[], files=[])
    assert fb["title"] == "Generated test set"
    assert fb["order"] == [] and fb["points"] == {}


def test_fallback_proposal_skips_candidates_without_draft_id() -> None:
    cands = [{"draft_id": "d-0"}, {"stem": "no id"}, "garbage", {"draft_id": ""}]
    fb = fallback_proposal(payload=_Payload(), candidates=cands, files=[])
    assert fb["order"] == ["d-0"] and fb["points"] == {"d-0": DEFAULT_POINTS}


# -----------------------------------------------------------------------------
# normalise_proposal: the agent's answer repaired into the contract shape
# -----------------------------------------------------------------------------


def test_normalise_repairs_partial_order_and_points() -> None:
    cands = _cands(3)
    fb = fallback_proposal(payload=_Payload(prompt="p"), candidates=cands, files=[])
    out = normalise_proposal(
        {"title": "Paper", "description": "Desc", "order": ["d-2", "zzz", "d-2"], "points": {"d-0": 5, "d-2": "12"}},
        candidates=cands,
        fallback=fb,
    )
    # unknown + duplicate dropped, missing appended in submission order
    assert out["order"] == ["d-2", "d-0", "d-1"]
    assert out["points"] == {"d-0": 5, "d-1": DEFAULT_POINTS, "d-2": 12}
    assert out["title"] == "Paper" and out["description"] == "Desc"


def test_normalise_clamps_title_description_and_points() -> None:
    cands = _cands(2)
    fb = fallback_proposal(payload=_Payload(prompt="p"), candidates=cands, files=[])
    out = normalise_proposal(
        {"title": "T" * 1000, "description": "D" * 5000, "order": ["d-0", "d-1"], "points": {"d-0": 0, "d-1": 1000}},
        candidates=cands,
        fallback=fb,
    )
    assert len(out["title"]) == TITLE_MAX_CHARS and len(out["description"]) == DESCRIPTION_MAX_CHARS
    assert out["points"] == {"d-0": 1, "d-1": 100}


def test_normalise_blank_title_uses_the_fallback_title_and_boolean_points_default() -> None:
    cands = _cands(1)
    fb = fallback_proposal(payload=_Payload(prompt="Photosynthesis"), candidates=cands, files=[])
    out = normalise_proposal({"title": "   ", "order": ["d-0"], "points": {"d-0": True}}, candidates=cands, fallback=fb)
    assert out["title"] == fb["title"] and out["description"] == fb["description"]
    assert out["points"] == {"d-0": DEFAULT_POINTS}


# -----------------------------------------------------------------------------
# parse_completion_json: tolerant recovery for raw text
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"title": "x"}',
        '```json\n{"title": "x"}\n```',
        'Sure! Here it is: {"title": "x"} hope that helps',
    ],
)
def test_parse_completion_json_recovers_the_object(text: str) -> None:
    assert parse_completion_json(text) == {"title": "x"}


@pytest.mark.parametrize("text", ["", "   ", "no json here", "[1, 2, 3]"])
def test_parse_completion_json_refuses_without_an_object(text: str) -> None:
    with pytest.raises(ValueError):
        parse_completion_json(text)
