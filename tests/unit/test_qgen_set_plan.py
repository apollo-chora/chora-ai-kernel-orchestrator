"""Unit tests — qgen_set_plan pure helpers (CHO-1819 single-pass set generation).

These are the LLM-free seams the set-native graph + runner compose: type-plan
construction/validation, per-type image-cap backstop, honest generation-summary
reconciliation, the regenerate dedup-context block, token-limit chunking, and
the FAIL-LOUD agent-output parser. No graph, no LLM — pure functions.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.orchestrators.qgen_set_plan import (
    AgentContractViolation,
    build_type_plan,
    chunk_type_plan,
    compute_generation_summary,
    dedup_context_block,
    effective_image_caps,
    enforce_image_caps,
    image_caps,
    parse_set_response,
    total_count,
    validate_type_plan,
)

_MIXED = [
    {"question_type": "mcq", "count": 8, "max_images": 3},
    {"question_type": "oe", "count": 2, "max_images": 1},
]


# --- build_type_plan ---------------------------------------------------------


def test_build_type_plan_passthrough_typed_quotas() -> None:
    plan = build_type_plan(_MIXED, question_type="mcq", requested_count=10)
    assert plan == _MIXED


def test_build_type_plan_empty_falls_back_to_single_quota() -> None:
    # Legacy single-type path: empty typed plan ⇒ ONE quota from content_type +
    # requested_count, images off (the W8 booleans govern the legacy path).
    plan = build_type_plan([], question_type="oe", requested_count=5)
    assert plan == [{"question_type": "oe", "count": 5, "max_images": 0}]


def test_build_type_plan_coerces_string_ints() -> None:
    plan = build_type_plan(
        [{"question_type": "mcq", "count": "4", "max_images": "2"}],
        question_type="mcq",
        requested_count=4,
    )
    assert plan == [{"question_type": "mcq", "count": 4, "max_images": 2}]


def test_total_count_sums_quotas() -> None:
    assert total_count(_MIXED) == 10


def test_image_caps_maps_type_to_max() -> None:
    assert image_caps(_MIXED) == {"mcq": 3, "oe": 1}


# --- build_type_plan: per-type author image opt-ins (CHO-1825) ---------------


def test_build_type_plan_preserves_image_flags_when_set() -> None:
    # A deterministic per-type toggle (image_for_stem / image_for_answer) MUST
    # survive build_type_plan so it reaches the Go agent via type_plan_json.
    plan = build_type_plan(
        [
            {
                "question_type": "mcq",
                "count": 2,
                "max_images": 0,
                "image_for_stem": True,
            },
            {
                "question_type": "oe",
                "count": 1,
                "max_images": 0,
                "image_for_answer": True,
            },
        ],
        question_type="mixed",
        requested_count=3,
    )
    assert plan == [
        {
            "question_type": "mcq",
            "count": 2,
            "max_images": 0,
            "image_for_stem": True,
        },
        {
            "question_type": "oe",
            "count": 1,
            "max_images": 0,
            "image_for_answer": True,
        },
    ]


def test_build_type_plan_omits_image_flags_when_off() -> None:
    # proto3-omit-false parity: an OFF quota keeps the legacy 3-key shape, so the
    # existing mixed-batch behaviour (+ every consumer that does exact-equality)
    # is byte-for-byte unchanged. False flags are NOT written.
    plan = build_type_plan(
        [{"question_type": "mcq", "count": 2, "max_images": 1, "image_for_stem": False, "image_for_answer": False}],
        question_type="mcq",
        requested_count=2,
    )
    assert plan == [{"question_type": "mcq", "count": 2, "max_images": 1}]


def test_build_type_plan_reads_image_flags_from_protobuf_object() -> None:
    # build_type_plan also tolerates a generated-protobuf message object (getattr
    # branch); the per-type flags read off it the same way.
    class _Q:
        question_type = "mcq"
        count = 3
        max_images = 0
        image_for_stem = True
        image_for_answer = False

    plan = build_type_plan([_Q()], question_type="mcq", requested_count=3)
    assert plan == [{"question_type": "mcq", "count": 3, "max_images": 0, "image_for_stem": True}]


# --- effective_image_caps (render backstop = budget + forced allowance) ------


def test_effective_image_caps_matches_image_caps_without_flags() -> None:
    # With no deterministic toggles, the render backstop cap is just the
    # discretionary AI-decide budget (== image_caps).
    assert effective_image_caps(_MIXED) == image_caps(_MIXED) == {"mcq": 3, "oe": 1}


def test_effective_image_caps_admits_forced_images_over_zero_budget() -> None:
    # The crux of CHO-1825: image_for_stem forces ONE stem image per question,
    # so the backstop must admit `count` images even when max_images == 0 —
    # otherwise enforce_image_caps would drop the very images the author forced.
    plan = [{"question_type": "mcq", "count": 2, "max_images": 0, "image_for_stem": True}]
    assert effective_image_caps(plan) == {"mcq": 2}


def test_effective_image_caps_both_parts_forced_doubles_allowance() -> None:
    plan = [{"question_type": "oe", "count": 3, "max_images": 0, "image_for_stem": True, "image_for_answer": True}]
    assert effective_image_caps(plan) == {"oe": 6}


def test_effective_image_caps_sums_budget_and_forced_allowance() -> None:
    # Discretionary budget (max_images) stacks ON TOP of the forced allowance.
    plan = [{"question_type": "mcq", "count": 4, "max_images": 2, "image_for_stem": True}]
    assert effective_image_caps(plan) == {"mcq": 6}


def test_effective_image_caps_forced_images_survive_enforce_backstop() -> None:
    # End-to-end of the pure pieces: 2 MCQ each forced a stem image, zero budget.
    # effective_image_caps + enforce_image_caps keep BOTH (dropped == 0).
    plan = [{"question_type": "mcq", "count": 2, "max_images": 0, "image_for_stem": True}]
    cands = [
        {"question_type": "mcq", "image_specs": [{"mode": "scene", "source": "s", "placement": "stem"}]}
        for _ in range(2)
    ]
    kept, dropped = enforce_image_caps(cands, effective_image_caps(plan))
    assert dropped == 0
    assert all(c["image_specs"] for c in kept)


# --- validate_type_plan (fail-loud, defence in depth) ------------------------


def test_validate_accepts_well_formed_plan() -> None:
    assert validate_type_plan(_MIXED, requested_count=10, max_batch=50) == []


def test_validate_rejects_sum_mismatch() -> None:
    issues = validate_type_plan(_MIXED, requested_count=9, max_batch=50)
    assert any("requested_count" in i for i in issues)


def test_validate_rejects_count_below_one() -> None:
    bad = [{"question_type": "mcq", "count": 0, "max_images": 0}]
    assert validate_type_plan(bad, requested_count=0, max_batch=50)


def test_validate_rejects_images_over_count() -> None:
    bad = [{"question_type": "mcq", "count": 2, "max_images": 5}]
    assert any("max_images" in i for i in validate_type_plan(bad, 2, 50))


def test_validate_rejects_unknown_type() -> None:
    bad = [{"question_type": "essay", "count": 1, "max_images": 0}]
    assert any("question_type" in i for i in validate_type_plan(bad, 1, 50))


def test_validate_rejects_over_max_batch() -> None:
    bad = [{"question_type": "mcq", "count": 60, "max_images": 0}]
    assert any("max_batch" in i or "50" in i for i in validate_type_plan(bad, 60, 50))


def test_validate_rejects_empty_plan() -> None:
    assert validate_type_plan([], requested_count=0, max_batch=50)


# --- chunk_type_plan (token-limit splitting) ---------------------------------


def test_chunk_small_plan_single_chunk() -> None:
    assert chunk_type_plan(_MIXED, max_per_call=20) == [_MIXED]


def test_chunk_large_plan_splits_preserving_type_and_total() -> None:
    plan = [{"question_type": "mcq", "count": 20, "max_images": 4}]
    chunks = chunk_type_plan(plan, max_per_call=8)
    # 20 split into 8 + 8 + 4, all mcq, images apportioned not exceeding the cap.
    assert sum(q["count"] for ch in chunks for q in ch) == 20
    assert all(sum(q["count"] for q in ch) <= 8 for ch in chunks)
    assert all(q["question_type"] == "mcq" for ch in chunks for q in ch)
    assert sum(q["max_images"] for ch in chunks for q in ch) == 4


# --- enforce_image_caps (backstop) -------------------------------------------


def _cand(qt: str, specs: list[dict]) -> dict:
    return {"question_type": qt, "image_specs": specs}


def test_enforce_caps_under_budget_unchanged() -> None:
    cands = [
        _cand("mcq", [{"mode": "scene", "source": "s", "placement": "stem"}]),
        _cand("mcq", []),
    ]
    out, dropped = enforce_image_caps(cands, {"mcq": 2, "oe": 0})
    assert dropped == 0
    assert out[0]["image_specs"]


def test_enforce_caps_drops_over_budget_deterministically() -> None:
    cands = [
        _cand("mcq", [{"mode": "scene", "source": "1", "placement": "stem"}]),
        _cand("mcq", [{"mode": "scene", "source": "2", "placement": "stem"}]),
        _cand("mcq", [{"mode": "scene", "source": "3", "placement": "stem"}]),
    ]
    out, dropped = enforce_image_caps(cands, {"mcq": 2})
    assert dropped == 1
    kept = sum(1 for c in out if c["image_specs"])
    assert kept == 2  # first two by array order survive


def test_enforce_caps_independent_per_type() -> None:
    cands = [
        _cand("mcq", [{"mode": "scene", "source": "m", "placement": "stem"}]),
        _cand("oe", [{"mode": "scene", "source": "o", "placement": "answer"}]),
    ]
    out, dropped = enforce_image_caps(cands, {"mcq": 0, "oe": 1})
    assert dropped == 1
    assert not out[0]["image_specs"]  # mcq cap 0 → dropped
    assert out[1]["image_specs"]  # oe cap 1 → kept


# --- compute_generation_summary ----------------------------------------------


def test_summary_full_count_no_shortfall() -> None:
    accepted = [{"question_type": "mcq"}] * 8 + [{"question_type": "oe"}] * 2
    s = compute_generation_summary(_MIXED, accepted, grounding_mode="strict", model_shortfall_reason="")
    assert s["requested_total"] == 10
    assert s["generated_total"] == 10
    # FLAT map matching the proto GenerationSummary.generated_per_type.
    assert s["generated_per_type"] == {"mcq": 8, "oe": 2}
    assert s["shortfall_reason"] == ""


def test_summary_strict_shortfall_carries_reason() -> None:
    accepted = [{"question_type": "mcq"}] * 6 + [{"question_type": "oe"}] * 1
    s = compute_generation_summary(
        _MIXED, accepted, grounding_mode="strict", model_shortfall_reason="Source supported 6 MCQ + 1 OE."
    )
    assert s["generated_total"] == 7
    assert s["generated_per_type"] == {"mcq": 6, "oe": 1}
    assert "Source supported" in s["shortfall_reason"]


# --- dedup_context_block -----------------------------------------------------


def test_dedup_block_lists_accepted_and_notes() -> None:
    block = dedup_context_block(
        accepted_stems=["What is inertia?", "Define momentum."],
        rejected_notes=["too vague"],
    )
    assert "inertia" in block and "momentum" in block
    assert "too vague" in block


def test_dedup_block_empty_when_nothing() -> None:
    assert dedup_context_block([], []) == ""


# --- parse_set_response (FAIL-LOUD) ------------------------------------------


def _wrapper(cands: list[dict], summary: dict | None = None) -> dict:
    out: dict = {"candidates": cands}
    if summary is not None:
        out["generation_summary"] = summary
    return out


def test_parse_valid_wrapper_returns_candidates_and_summary() -> None:
    cands = [{"stem": "a", "question_type": "mcq"}, {"stem": "b", "question_type": "oe"}]
    raw = _wrapper(cands, {"shortfall_reason": ""})
    plan = [{"question_type": "mcq", "count": 1, "max_images": 0}, {"question_type": "oe", "count": 1, "max_images": 0}]
    got_cands, got_summary = parse_set_response(raw, plan, grounding_mode="")
    assert got_cands == cands
    assert got_summary == {"shortfall_reason": ""}


def test_parse_legacy_bare_candidate_wrapped_as_single() -> None:
    # Rollout tolerance: a not-yet-redeployed agent emits {"candidate": {...}}.
    raw = {"candidate": {"stem": "x", "question_type": "mcq"}}
    plan = [{"question_type": "mcq", "count": 1, "max_images": 0}]
    cands, _ = parse_set_response(raw, plan, grounding_mode="")
    assert cands == [{"stem": "x", "question_type": "mcq"}]


def test_parse_legacy_flattened_candidate_wrapped_as_single() -> None:
    raw = {"stem": "y", "question_type": "oe", "oe_payload": {}}
    plan = [{"question_type": "oe", "count": 1, "max_images": 0}]
    cands, _ = parse_set_response(raw, plan, grounding_mode="")
    assert cands == [raw]


def test_parse_raises_on_no_candidates_non_legacy() -> None:
    plan = [{"question_type": "mcq", "count": 2, "max_images": 0}]
    with pytest.raises(AgentContractViolation):
        parse_set_response({"junk": 1}, plan, grounding_mode="")


def test_parse_raises_on_out_of_plan_type() -> None:
    raw = _wrapper([{"stem": "a", "question_type": "essay"}])
    plan = [{"question_type": "mcq", "count": 1, "max_images": 0}]
    with pytest.raises(AgentContractViolation):
        parse_set_response(raw, plan, grounding_mode="")


def test_parse_raises_on_over_count() -> None:
    raw = _wrapper([{"stem": "a", "question_type": "mcq"}, {"stem": "b", "question_type": "mcq"}])
    plan = [{"question_type": "mcq", "count": 1, "max_images": 0}]
    with pytest.raises(AgentContractViolation):
        parse_set_response(raw, plan, grounding_mode="")


def test_parse_raises_on_short_set_when_not_strict() -> None:
    # generated < requested is illegal unless strict + shortfall_reason.
    raw = _wrapper([{"stem": "a", "question_type": "mcq"}], {"shortfall_reason": ""})
    plan = [{"question_type": "mcq", "count": 3, "max_images": 0}]
    with pytest.raises(AgentContractViolation):
        parse_set_response(raw, plan, grounding_mode="starting_point")


def test_parse_allows_short_set_when_strict_with_reason() -> None:
    raw = _wrapper([{"stem": "a", "question_type": "mcq"}], {"shortfall_reason": "Source supported 1 MCQ."})
    plan = [{"question_type": "mcq", "count": 3, "max_images": 0}]
    cands, summary = parse_set_response(raw, plan, grounding_mode="strict")
    assert len(cands) == 1
    assert "Source supported" in summary["shortfall_reason"]


# ---------------------------------------------------------------------------
# ADR-251 D1 (CHO-2396) — chunking carries the forced-image flags; cap = 200
# ---------------------------------------------------------------------------


def test_chunk_type_plan_carries_forced_image_flags() -> None:
    """CHO-1825's lesson recurs at the chunk seam: image_for_stem /
    image_for_answer force ONE image per question of the type, so every CHUNK
    plan must carry the flags or effective_image_caps computes a zero forced
    allowance for later chunks and enforce_image_caps drops the very images
    the author forced."""
    plan = [
        {
            "question_type": "mcq",
            "count": 25,
            "max_images": 0,
            "image_for_stem": True,
            "image_for_answer": True,
        }
    ]
    chunks = chunk_type_plan(plan, 10)
    assert [sum(q["count"] for q in c) for c in chunks] == [10, 10, 5]
    for chunk in chunks:
        for entry in chunk:
            assert entry.get("image_for_stem") is True, entry
            assert entry.get("image_for_answer") is True, entry
    # The forced allowance therefore survives per chunk.
    assert effective_image_caps(chunks[2]) == {"mcq": 10}


def test_chunk_type_plan_without_flags_adds_no_flag_keys() -> None:
    plan = [{"question_type": "mcq", "count": 12, "max_images": 2}]
    for chunk in chunk_type_plan(plan, 10):
        for entry in chunk:
            assert "image_for_stem" not in entry
            assert "image_for_answer" not in entry


def test_validate_default_cap_is_200() -> None:
    """ADR-251 D1 (owner-ruled): max_batch default rises 50 -> 200, matching
    chora-creation's aiassist.MaxBatchCount. 200 passes; 201 refuses with both
    numbers named."""
    ok = [{"question_type": "mcq", "count": 200, "max_images": 0}]
    assert validate_type_plan(ok, requested_count=200) == []

    over = [{"question_type": "mcq", "count": 201, "max_images": 0}]
    issues = validate_type_plan(over, requested_count=201)
    assert issues, "201 must be refused by the default cap"
    joined = " ".join(issues)
    assert "201" in joined and "200" in joined, joined
