"""CHO-1973 Wave A — pure Growth-Edge review-panel builder.

The panel is the CANONICAL UX shape the FE renders at the bounded HITL review,
and the exact body the ``review_pending`` event carries. It is built from the
checkpoint ``candidate_edges`` so it is deterministic + round-trips: the
``proposed_edge_id`` minted here MUST resolve back to the same candidate edge on
resume (the inverse mapping lives in the graph node). NO I/O here — pure
transform, trivially unit-tested.
"""

from __future__ import annotations

import json
from typing import Any

from chora_ai_kernel_orchestrator.domain.weakness_analyser_crew.panel import (
    FREE_OUTPUT_KINDS,
    MAX_CANDIDATE_STRUGGLES,
    METERED_OUTPUT_KINDS,
    OUTPUT_KINDS,
    build_review_panel,
    index_from_proposed_edge_id,
    proposed_edge_id_for_index,
)


def _edge(
    concept_key: str,
    *,
    strength: float = 0.8,
    summary: str = "",
    angles: list[str] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """One candidate edge in the body-dict shape the graph stores in state."""
    descriptor: dict[str, Any] = {}
    if summary:
        descriptor["summary"] = summary
    if angles:
        descriptor["suggested_angles"] = angles
    return {
        "concept_label": label or concept_key.replace("-", " "),
        "concept_key": concept_key,
        "category": "arithmetic",
        "tags": ["fractions"],
        "confidence": 0.9,
        "strength": strength,
        "descriptor_json": json.dumps(descriptor),
    }


def _panel(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "candidate_edges": [
            _edge(
                "adding-fractions",
                strength=0.85,
                summary="mixes denominators",
                angles=["visual area model", "common denominators"],
            ),
            _edge("long-division", strength=0.3, summary="loses place value"),
        ],
        "upload_id": "upload-1",
        "tenant_id": "tenant-1",
        "learner_gcid": "gcid-1",
        "requested_outputs": {"focused_dose": True, "study_aids": True},
        "output_prices": {"practice_test": 120, "study_aids": 60},
    }
    kwargs.update(overrides)
    return build_review_panel(**kwargs)


# --------------------------------------------------------------------------- #
# proposed_edge_id scheme — stable + invertible
# --------------------------------------------------------------------------- #


def test_proposed_edge_id_round_trips_index() -> None:
    for i in (0, 1, 2, 7, 24):
        assert index_from_proposed_edge_id(proposed_edge_id_for_index(i)) == i


def test_index_from_unparseable_id_is_none() -> None:
    assert index_from_proposed_edge_id("") is None
    assert index_from_proposed_edge_id("nope") is None
    assert index_from_proposed_edge_id("pe-") is None
    assert index_from_proposed_edge_id("pe-x") is None
    assert index_from_proposed_edge_id("pe--1") is None


# --------------------------------------------------------------------------- #
# panel shape
# --------------------------------------------------------------------------- #


def test_panel_carries_identity_fields() -> None:
    p = _panel()
    assert p["upload_id"] == "upload-1"
    assert p["tenant_id"] == "tenant-1"
    assert p["learner_gcid"] == "gcid-1"


def test_proposed_edges_mapped_from_candidates_in_order() -> None:
    p = _panel()
    pe = p["proposed_edges"]
    assert [e["proposed_edge_id"] for e in pe] == ["pe-0", "pe-1"]
    assert pe[0]["concept_label"] == "adding fractions"
    assert pe[0]["summary"] == "mixes denominators"
    assert pe[0]["suggested_angles"] == ["visual area model", "common denominators"]
    assert pe[0]["strength"] == 0.85


def test_proposed_edge_id_resolves_back_to_same_candidate() -> None:
    candidates = [
        _edge("adding-fractions"),
        _edge("long-division"),
        _edge("place-value"),
    ]
    p = _panel(candidate_edges=candidates)
    for proposed in p["proposed_edges"]:
        idx = index_from_proposed_edge_id(proposed["proposed_edge_id"])
        assert idx is not None
        assert candidates[idx]["concept_label"] == proposed["concept_label"]


def test_suggested_difficulty_derives_from_strength() -> None:
    p = _panel(
        candidate_edges=[
            _edge("a", strength=0.9),  # very weak -> practice harder
            _edge("b", strength=0.6),  # mid -> standard
            _edge("c", strength=0.25),  # near mastered -> easier
        ]
    )
    diffs = [e["suggested_difficulty"] for e in p["proposed_edges"]]
    assert diffs == ["harder", "standard", "easier"]
    assert all(d in {"easier", "standard", "harder"} for d in diffs)


# --------------------------------------------------------------------------- #
# available_outputs — free vs metered pricing + default selection
# --------------------------------------------------------------------------- #


def test_available_outputs_cover_all_kinds_in_canonical_order() -> None:
    p = _panel()
    kinds = [o["kind"] for o in p["available_outputs"]]
    assert kinds == list(OUTPUT_KINDS)
    assert set(FREE_OUTPUT_KINDS) == {"focused_dose", "familiar_coaching"}
    assert set(METERED_OUTPUT_KINDS) == {"practice_test", "study_aids"}


def test_free_outputs_priced_zero_metered_from_price_table() -> None:
    p = _panel()
    by_kind = {o["kind"]: o for o in p["available_outputs"]}
    assert by_kind["focused_dose"]["mana_price"] == 0
    assert by_kind["familiar_coaching"]["mana_price"] == 0
    assert by_kind["practice_test"]["mana_price"] == 120
    assert by_kind["study_aids"]["mana_price"] == 60


def test_default_selected_reflects_requested_outputs() -> None:
    p = _panel()
    by_kind = {o["kind"]: o for o in p["available_outputs"]}
    assert by_kind["focused_dose"]["default_selected"] is True
    assert by_kind["study_aids"]["default_selected"] is True
    assert by_kind["practice_test"]["default_selected"] is False


def test_focused_dose_defaults_selected_when_no_requested_outputs() -> None:
    # The uploaded event's nil requested_outputs means "core edges + dose only".
    p = _panel(requested_outputs={})
    by_kind = {o["kind"]: o for o in p["available_outputs"]}
    assert by_kind["focused_dose"]["default_selected"] is True
    assert by_kind["familiar_coaching"]["default_selected"] is False


def test_unconfigured_metered_price_is_zero() -> None:
    # No price table -> metered outputs price 0 (the loud-log/flag is the caller's;
    # the pure builder never fabricates a price).
    p = _panel(output_prices={})
    by_kind = {o["kind"]: o for o in p["available_outputs"]}
    assert by_kind["practice_test"]["mana_price"] == 0
    assert by_kind["study_aids"]["mana_price"] == 0


# --------------------------------------------------------------------------- #
# familiar + candidate_struggles
# --------------------------------------------------------------------------- #


def test_familiar_defaults_to_empty_when_unresolved() -> None:
    p = _panel(familiar=None)
    assert p["familiar"] == {}


def test_familiar_passthrough_when_resolved() -> None:
    fam = {"familiar_id": "fam-7", "name": "Ember", "species": "dragon"}
    p = _panel(familiar=fam)
    assert p["familiar"] == fam


def _struggle(key: str, *, label: str | None = None, confidence: float = 0.3) -> dict[str, Any]:
    """One retained below-threshold concept in the raw shape the graph stores."""
    return {"concept_key": key, "concept_label": label or key.replace("-", " "), "confidence": confidence}


def test_candidate_struggles_default_empty() -> None:
    p = _panel(candidate_struggles=None)
    assert p["candidate_struggles"] == []


def test_candidate_struggles_built_from_retained_below_threshold() -> None:
    # CHO-1973 Q2: the retained below-threshold concepts become the bounded picker.
    p = _panel(candidate_struggles=[_struggle("borrowing", label="Borrowing", confidence=0.4)])
    cs = p["candidate_struggles"]
    # the panel entry is the proto CandidateStruggle shape — {key, label} ONLY
    # (the ranking confidence is stripped; it never reaches the wire).
    assert cs == [{"concept_key": "borrowing", "concept_label": "Borrowing"}]
    assert "confidence" not in cs[0]


def test_candidate_struggles_dedup_against_proposed_edges() -> None:
    # default candidate_edges keys: adding-fractions, long-division. A retained
    # struggle that matches a proposed edge must be dropped (never re-offer it).
    p = _panel(
        candidate_struggles=[
            _struggle("long-division", confidence=0.4),  # already a proposed edge -> drop
            _struggle("borrowing", confidence=0.3),  # new concept -> keep
        ]
    )
    assert [s["concept_key"] for s in p["candidate_struggles"]] == ["borrowing"]


def test_candidate_struggles_dedup_within_itself() -> None:
    p = _panel(
        candidate_struggles=[
            _struggle("borrowing", confidence=0.4),
            _struggle("borrowing", confidence=0.2),  # duplicate key -> collapse to one
        ]
    )
    assert [s["concept_key"] for s in p["candidate_struggles"]] == ["borrowing"]


def test_candidate_struggles_sorted_by_confidence_desc() -> None:
    p = _panel(
        candidate_struggles=[
            _struggle("low", confidence=0.1),
            _struggle("high", confidence=0.45),
            _struggle("mid", confidence=0.3),
        ]
    )
    assert [s["concept_key"] for s in p["candidate_struggles"]] == ["high", "mid", "low"]


def test_candidate_struggles_capped_to_max() -> None:
    many = [_struggle(f"c{i}", confidence=0.1 + i / 100) for i in range(MAX_CANDIDATE_STRUGGLES + 5)]
    p = _panel(candidate_struggles=many)
    cs = p["candidate_struggles"]
    assert len(cs) == MAX_CANDIDATE_STRUGGLES
    # the highest-confidence concepts survive the cap (descending order).
    assert cs[0]["concept_key"] == f"c{MAX_CANDIDATE_STRUGGLES + 4}"


def test_candidate_struggle_missing_label_or_key_is_skipped() -> None:
    # fail-loud: a retained concept lacking a usable key/label is skipped, never
    # fabricated. The remaining good concept still surfaces.
    p = _panel(
        candidate_struggles=[
            {"concept_key": "", "concept_label": "no key", "confidence": 0.4},
            {"concept_key": "no-label", "concept_label": "", "confidence": 0.4},
            _struggle("good", confidence=0.3),
        ]
    )
    assert [s["concept_key"] for s in p["candidate_struggles"]] == ["good"]
