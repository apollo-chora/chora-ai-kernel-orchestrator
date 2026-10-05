"""Tests for agent #19 Familiar (LLM tier 2).

Per `feedback_familiar_vs_agent`: Familiar entity (Content Consumption RPG
companion) is distinct from the AI agent powering it. This module covers
the AI agent — proactive nudge composition, daily-dose curation, RPG
dialogue.

Comic Ch6 P14 P1 invariant under test:
    Familiar Proactive Nudge generated text contains topic-aware suggestion
    e.g., "I noticed you're authoring on {topic}".

Hexagonal: agent code is pure (no LLM SDK imports, no HTTP). It composes
prompt strings from the FamiliarRequest and returns a FamiliarResponse.
The orchestrator pipeline injects model + guardrail adapters around it.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.domain.agents.familiar import (
    FamiliarKind,
    FamiliarRequest,
    compose_daily_dose_curation,
    compose_proactive_nudge,
    compose_rpg_dialogue,
    dispatch,
)

# --- Proactive nudge ---------------------------------------------------------


def test_proactive_nudge_mentions_topic_for_authoring_signal() -> None:
    """Comic Ch6 P14 P1 invariant: nudge text contains topic-aware suggestion."""
    req = FamiliarRequest(
        kind=FamiliarKind.PROACTIVE_NUDGE,
        owner_gcid="gcid-1",
        familiar_name="Spark",
        topic="quadratic equations",
        signal="authoring",
    )
    resp = compose_proactive_nudge(req)
    assert "quadratic equations" in resp.message
    # Comic anchor wording: "I noticed you're authoring on {topic}"
    assert "I noticed you're authoring on quadratic equations" in resp.message


def test_proactive_nudge_uses_familiar_name_for_personalisation() -> None:
    req = FamiliarRequest(
        kind=FamiliarKind.PROACTIVE_NUDGE,
        owner_gcid="gcid-2",
        familiar_name="Aether",
        topic="photosynthesis",
        signal="authoring",
    )
    resp = compose_proactive_nudge(req)
    assert resp.familiar_name == "Aether"
    assert "photosynthesis" in resp.message


def test_proactive_nudge_signals_review_branch() -> None:
    req = FamiliarRequest(
        kind=FamiliarKind.PROACTIVE_NUDGE,
        owner_gcid="gcid-3",
        familiar_name="Spark",
        topic="cell division",
        signal="review-due",
    )
    resp = compose_proactive_nudge(req)
    assert "cell division" in resp.message
    assert "review" in resp.message.lower()


def test_proactive_nudge_signal_curiosity_branch() -> None:
    req = FamiliarRequest(
        kind=FamiliarKind.PROACTIVE_NUDGE,
        owner_gcid="gcid-4",
        familiar_name="Spark",
        topic="wave functions",
        signal="curiosity",
    )
    resp = compose_proactive_nudge(req)
    assert "wave functions" in resp.message


def test_proactive_nudge_blank_topic_rejected() -> None:
    with pytest.raises(ValueError):
        compose_proactive_nudge(
            FamiliarRequest(
                kind=FamiliarKind.PROACTIVE_NUDGE,
                owner_gcid="gcid-5",
                familiar_name="Spark",
                topic="   ",
                signal="authoring",
            )
        )


def test_proactive_nudge_blank_owner_gcid_rejected() -> None:
    with pytest.raises(ValueError):
        compose_proactive_nudge(
            FamiliarRequest(
                kind=FamiliarKind.PROACTIVE_NUDGE,
                owner_gcid="",
                familiar_name="Spark",
                topic="algebra",
                signal="authoring",
            )
        )


def test_proactive_nudge_default_signal_falls_back_to_authoring_form() -> None:
    """Unknown / empty signal still includes topic + uses default phrasing."""
    req = FamiliarRequest(
        kind=FamiliarKind.PROACTIVE_NUDGE,
        owner_gcid="gcid-6",
        familiar_name="Spark",
        topic="meiosis",
        signal="",
    )
    resp = compose_proactive_nudge(req)
    assert "meiosis" in resp.message


# --- Daily Dose curation -----------------------------------------------------


def test_daily_dose_curation_describes_dose_with_topics() -> None:
    """Daily dose curation summary lists topics from supplied seed atoms."""
    req = FamiliarRequest(
        kind=FamiliarKind.DAILY_DOSE_CURATION,
        owner_gcid="gcid-10",
        familiar_name="Spark",
        topic="algebra",
        dose_topics=["algebra", "geometry", "trigonometry"],
    )
    resp = compose_daily_dose_curation(req)
    # Comic Ch6 P14 P4 anchor: "Your Daily Dose is ready! 5 atoms..."
    assert "Daily Dose" in resp.message
    assert "algebra" in resp.message
    assert "geometry" in resp.message


def test_daily_dose_curation_handles_empty_topic_list() -> None:
    req = FamiliarRequest(
        kind=FamiliarKind.DAILY_DOSE_CURATION,
        owner_gcid="gcid-11",
        familiar_name="Spark",
        topic="anything",
        dose_topics=[],
    )
    resp = compose_daily_dose_curation(req)
    assert "Daily Dose" in resp.message


def test_daily_dose_curation_dedupes_topics_case_insensitively() -> None:
    req = FamiliarRequest(
        kind=FamiliarKind.DAILY_DOSE_CURATION,
        owner_gcid="gcid-12",
        familiar_name="",
        topic="anything",
        dose_topics=["Algebra", "algebra", "geometry"],
    )
    resp = compose_daily_dose_curation(req)
    # Only one "Algebra" — dedup should drop the second occurrence.
    assert resp.message.count("lgebra") == 1
    assert "geometry" in resp.message
    # Empty familiar_name falls back to "Familiar" voice.
    assert resp.familiar_name == "Familiar"


# --- RPG Dialogue -------------------------------------------------------------


def test_rpg_dialogue_includes_familiar_name_and_topic() -> None:
    req = FamiliarRequest(
        kind=FamiliarKind.RPG_DIALOGUE,
        owner_gcid="gcid-20",
        familiar_name="Aether",
        topic="black holes",
        learner_message="What's gravity?",
    )
    resp = compose_rpg_dialogue(req)
    assert "Aether" in resp.message
    assert "black holes" in resp.message
    assert resp.familiar_name == "Aether"


def test_rpg_dialogue_blank_learner_message_rejected() -> None:
    with pytest.raises(ValueError):
        compose_rpg_dialogue(
            FamiliarRequest(
                kind=FamiliarKind.RPG_DIALOGUE,
                owner_gcid="gcid-21",
                familiar_name="Aether",
                topic="atoms",
                learner_message="",
            )
        )


# --- dispatch() — kind selector ----------------------------------------------


def test_dispatch_routes_to_proactive_nudge_for_proactive_nudge_kind() -> None:
    req = FamiliarRequest(
        kind=FamiliarKind.PROACTIVE_NUDGE,
        owner_gcid="gcid-30",
        familiar_name="Spark",
        topic="logarithms",
        signal="authoring",
    )
    resp = dispatch(req)
    assert "logarithms" in resp.message
    assert "I noticed you're authoring on logarithms" in resp.message


def test_dispatch_routes_to_daily_dose_curation_for_daily_dose_kind() -> None:
    req = FamiliarRequest(
        kind=FamiliarKind.DAILY_DOSE_CURATION,
        owner_gcid="gcid-31",
        familiar_name="Spark",
        topic="anything",
        dose_topics=["calculus"],
    )
    resp = dispatch(req)
    assert "Daily Dose" in resp.message
    assert "calculus" in resp.message


def test_dispatch_routes_to_rpg_dialogue_for_rpg_dialogue_kind() -> None:
    req = FamiliarRequest(
        kind=FamiliarKind.RPG_DIALOGUE,
        owner_gcid="gcid-32",
        familiar_name="Aether",
        topic="orbital mechanics",
        learner_message="Tell me more",
    )
    resp = dispatch(req)
    assert "Aether" in resp.message
    assert "orbital mechanics" in resp.message


def test_dispatch_unknown_kind_defensive_guard() -> None:
    """Defensive guard: calling dispatch with a non-StrEnum value raises."""
    # Build a frozen FamiliarRequest then bypass the StrEnum check via
    # object.__setattr__ to simulate a kind that no compose_* handles.
    req = FamiliarRequest(
        kind=FamiliarKind.PROACTIVE_NUDGE,  # placeholder; mutated below
        owner_gcid="gcid-99",
        familiar_name="Spark",
        topic="x",
        signal="authoring",
    )
    object.__setattr__(req, "kind", "not-a-real-kind")
    with pytest.raises(ValueError):
        dispatch(req)
