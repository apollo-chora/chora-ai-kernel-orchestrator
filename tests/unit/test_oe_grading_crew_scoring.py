"""scoring.py unit tests — rubric-weighted composite (ADR-172 §D4).

The composite math encodes the grade a learner gets, so it is exercised
independently of any LLM call (Go-parity with the canonical scoring.go contract).
"""

from __future__ import annotations

import json

import pytest

from chora_ai_kernel_orchestrator.domain.oe_grading_crew.scoring import (
    ScoringError,
    compute_composite,
    parse_rubric,
)

_FRACTIONAL = json.dumps(
    [{"criterion_id": "c1", "title": "A", "weight": 0.5}, {"criterion_id": "c2", "title": "B", "weight": 0.5}]
)
_PERCENT = json.dumps([{"criterion_id": "c1", "weight": 50}, {"criterion_id": "c2", "weight": 50}])
_FULL = [{"criterion_id": "c1", "score": 1, "max_score": 1}, {"criterion_id": "c2", "score": 0.5, "max_score": 1}]


def test_fractional_weights() -> None:
    # 0.5*(1/1) + 0.5*(0.5/1) = 0.75 → 7.5 of 10
    assert compute_composite(_FRACTIONAL, _FULL, 10) == pytest.approx(7.5)


def test_percentage_weights_normalise_same() -> None:
    assert compute_composite(_PERCENT, _FULL, 10) == pytest.approx(7.5)


def test_unmatched_criterion_contributes_zero() -> None:
    # Only c1 scored → 0.5*(1) + 0.5*(0) = 0.5 → 5.0
    assert compute_composite(_FRACTIONAL, [{"criterion_id": "c1", "score": 1, "max_score": 1}], 10) == pytest.approx(
        5.0
    )


def test_over_max_score_clamped() -> None:
    # Hallucinated 2/1 on both → clamp to 1 each → 10.0 (never exceeds points_possible)
    over = [{"criterion_id": "c1", "score": 2, "max_score": 1}, {"criterion_id": "c2", "score": 2, "max_score": 1}]
    assert compute_composite(_FRACTIONAL, over, 10) == pytest.approx(10.0)


def test_degenerate_max_score_treated_as_one() -> None:
    bad = [{"criterion_id": "c1", "score": 1, "max_score": 0}, {"criterion_id": "c2", "score": 1, "max_score": 0}]
    # max<=0 → treated as /1; score 1/1 clamped → 1.0 each → 10.0
    assert compute_composite(_FRACTIONAL, bad, 10) == pytest.approx(10.0)


def test_object_wrapper_rubric_shape() -> None:
    wrapped = json.dumps({"rubric": json.loads(_FRACTIONAL)})
    assert compute_composite(wrapped, _FULL, 10) == pytest.approx(7.5)


def test_rounds_to_two_dp() -> None:
    r = json.dumps([{"criterion_id": "c1", "weight": 1.0}])
    s = [{"criterion_id": "c1", "score": 1, "max_score": 3}]  # 1/3 * 10 = 3.333...
    assert compute_composite(r, s, 10) == pytest.approx(3.33)


def test_none_scores_grades_zero() -> None:
    assert compute_composite(_FRACTIONAL, None, 10) == pytest.approx(0.0)


def test_empty_rubric_fails_loud() -> None:
    with pytest.raises(ScoringError):
        compute_composite("[]", _FULL, 10)
    with pytest.raises(ScoringError):
        compute_composite("", _FULL, 10)


def test_zero_weight_sum_fails_loud() -> None:
    with pytest.raises(ScoringError):
        compute_composite(json.dumps([{"criterion_id": "c1", "weight": 0}]), _FULL, 10)


def test_malformed_json_fails_loud() -> None:
    with pytest.raises(ScoringError):
        compute_composite("{not json", _FULL, 10)


def test_parse_rubric_string_weight_coerced() -> None:
    crit = parse_rubric(json.dumps([{"criterion_id": "c1", "weight": "1.0"}]))
    assert crit[0].weight == pytest.approx(1.0)
    assert crit[0].criterion_id == "c1"
