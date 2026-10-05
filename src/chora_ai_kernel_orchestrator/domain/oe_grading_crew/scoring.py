"""Rubric parsing + weighted-composite scoring math (ADR-172 §D4).

The oe_evaluator agent scores EACH rubric criterion on a 0..max_score scale and
emits ONLY ``{criterion_scores, comment}`` — it never computes the composite
(an LLM is unreliable at weighted arithmetic, and a hallucinated composite could
inflate the grade). The deterministic composite is derived HERE, in the graph's
``evaluate_node``, from the LLM sub-scores + the authored rubric weights:

    normalised    = Σ_i ( weight_i / Σweight × clamp01(score_i / max_score_i) )
    points_earned = round(normalised × points_possible, 2)

This is the Python port of the canonical Go reference (``agents/oe_grading_adk_go``'s
former ``scoring.go``, which is intentionally NOT shipped — the composite lives
on the orchestrator path where the rubric + sub-scores meet; chora-delivery's
``ApplyGrading`` then persists ``points_earned`` verbatim).

Weights may be authored fractional (~1.0) OR percentage (~100); both are accepted
and normalised by their own sum so the composite is convention-independent. A
zero/blank weight-sum or an empty rubric is a fail-loud error (a grade cannot be
grounded) per [[feedback-no-stubs-real-wiring]] — the caller surfaces it rather
than silently emitting points_earned=0.

Pure + deterministic (ADR-141 D2 transparency) and the unit-test core of the
crew — these functions encode the grade a learner gets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RubricCriterion:
    """One weighted rubric criterion. Matches the OpenAPI OEPayload.rubric shape
    (criterion_id / title / description / weight) snapshot into rubric_json."""

    criterion_id: str
    title: str
    weight: float


class ScoringError(ValueError):
    """A grade could not be grounded (empty/malformed rubric or zero weight-sum).

    Subclasses ValueError so existing ``except (ValueError, TypeError)`` callers
    keep working; raised loud so the orchestrator surfaces the authoring/data bug
    instead of silently grading 0.
    """


def parse_rubric(rubric_json: str) -> list[RubricCriterion]:
    """Decode a rubric_json snapshot into criteria.

    Tolerates either a bare JSON array ``[ {...}, ... ]`` OR an object wrapper
    ``{ "rubric": [ ... ] }`` (both shapes flow from creation's OEPayload).
    Fail-loud on malformed JSON or an empty criterion list — a grade cannot be
    grounded without a rubric (ADR-172 §D4 makes the rubric a precondition).
    """
    trimmed = (rubric_json or "").strip()
    if not trimmed:
        raise ScoringError("scoring: rubric_json is empty (ADR-172 §D4 makes the rubric a grading precondition)")
    try:
        decoded = json.loads(trimmed)
    except (ValueError, TypeError) as exc:
        raise ScoringError(f"scoring: rubric_json decode failed: {exc}") from exc

    if isinstance(decoded, dict):
        decoded = decoded.get("rubric")
    if not isinstance(decoded, list):
        raise ScoringError("scoring: rubric_json is not a list (nor {rubric: [...]})")

    criteria: list[RubricCriterion] = []
    for raw in decoded:
        if not isinstance(raw, dict):
            continue
        criteria.append(
            RubricCriterion(
                criterion_id=str(raw.get("criterion_id", "")),
                title=str(raw.get("title", "")),
                weight=_as_float(raw.get("weight")),
            )
        )
    if not criteria:
        raise ScoringError("scoring: rubric has no criteria")
    return criteria


def compute_composite(
    rubric_json: str,
    criterion_scores: list[dict[str, Any]] | None,
    points_possible: int,
) -> float:
    """Map the evaluator's per-criterion sub-scores to a weighted composite
    points_earned ∈ [0, points_possible].

    Contract (mirrors the Go ComputeComposite, asserted in scoring_test.py):
      - empty/malformed rubric → ScoringError (fail-loud).
      - weight-sum ≤ 0 → ScoringError (can't normalise).
      - a criterion with no matching sub-score contributes 0 (unanswered).
      - a sub-score whose max_score ≤ 0 is treated as 0/1 (degenerate max).
      - per-criterion ratio clamped to [0,1] so a hallucinated over-max score
        cannot push the composite above points_possible.
      - result rounded to 2 dp (cents-of-a-point; float points field).
    """
    criteria = parse_rubric(rubric_json)

    weight_sum = sum(c.weight for c in criteria)
    if weight_sum <= 0:
        raise ScoringError(f"scoring: rubric weight-sum is {weight_sum} (≤0); cannot normalise")

    score_by_id: dict[str, dict[str, Any]] = {}
    for s in criterion_scores or []:
        if isinstance(s, dict):
            score_by_id[str(s.get("criterion_id", ""))] = s

    normalised = 0.0
    for c in criteria:
        ratio = 0.0  # unmatched criterion → 0 (the learner did not address it)
        score_entry = score_by_id.get(c.criterion_id)
        if score_entry is not None:
            max_score = _as_float(score_entry.get("max_score"))
            if max_score <= 0:
                max_score = 1.0
            ratio = _clamp01(_as_float(score_entry.get("score")) / max_score)
        normalised += (c.weight / weight_sum) * ratio

    return round(normalised * float(points_possible), 2)


def _clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def _as_float(v: Any) -> float:
    """Coerce a numeric ``any`` (LLM JSON hands us int/float/str) to float; 0 on miss."""
    if isinstance(v, bool):  # bool is an int subclass — exclude it explicitly
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return 0.0
    return 0.0


__all__ = ["RubricCriterion", "ScoringError", "compute_composite", "parse_rubric"]
