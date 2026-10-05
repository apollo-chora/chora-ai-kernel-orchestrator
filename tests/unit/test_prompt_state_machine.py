"""Unit tests for the prompt-override promotion state machine (ADR-197 M-C.1).

Mirrors the closure-saga state machine
(services/chora-closure-orchestrator/.../domain/closure/state.py) — a pure,
infrastructure-free transition table with a fail-loud assert.

Lifecycle:

    draft -> pending_eval -> pending_hitl -> active -> archived
                 |                  |
                 +--> rejected <----+ --> archived

archived is terminal.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.domain.prompt_registry.state_machine import (
    ALL_STATES,
    IllegalPromptTransitionError,
    PromptPlanState,
    assert_transition,
    can_transition,
    is_terminal_state,
)

# -----------------------------------------------------------------------------
# Enum
# -----------------------------------------------------------------------------


class TestPromptPlanState:
    def test_all_states_returns_six_values(self) -> None:
        assert len(ALL_STATES) == 6
        for s in (
            PromptPlanState.DRAFT,
            PromptPlanState.PENDING_EVAL,
            PromptPlanState.PENDING_HITL,
            PromptPlanState.ACTIVE,
            PromptPlanState.REJECTED,
            PromptPlanState.ARCHIVED,
        ):
            assert s in ALL_STATES

    def test_values_are_lowercase_strings(self) -> None:
        # Wire values match the migration 0008 status CHECK literals exactly.
        assert PromptPlanState.DRAFT.value == "draft"
        assert PromptPlanState.PENDING_EVAL.value == "pending_eval"
        assert PromptPlanState.PENDING_HITL.value == "pending_hitl"
        assert PromptPlanState.ACTIVE.value == "active"
        assert PromptPlanState.REJECTED.value == "rejected"
        assert PromptPlanState.ARCHIVED.value == "archived"

    def test_is_strenum_so_value_equals_str(self) -> None:
        # StrEnum members compare equal to their string value (used by the
        # adapter which binds the .value to the SQL guard).
        assert PromptPlanState.ACTIVE == "active"


# -----------------------------------------------------------------------------
# can_transition() — table-driven
# -----------------------------------------------------------------------------


class TestCanTransition:
    @pytest.mark.parametrize(
        ("from_state", "to_state", "want"),
        [
            # legal forward
            (PromptPlanState.DRAFT, PromptPlanState.PENDING_EVAL, True),
            (PromptPlanState.PENDING_EVAL, PromptPlanState.PENDING_HITL, True),
            (PromptPlanState.PENDING_EVAL, PromptPlanState.REJECTED, True),
            (PromptPlanState.PENDING_HITL, PromptPlanState.ACTIVE, True),
            (PromptPlanState.PENDING_HITL, PromptPlanState.REJECTED, True),
            (PromptPlanState.ACTIVE, PromptPlanState.ARCHIVED, True),
            (PromptPlanState.REJECTED, PromptPlanState.ARCHIVED, True),
            # illegal — skipping a stage
            (PromptPlanState.DRAFT, PromptPlanState.ACTIVE, False),
            (PromptPlanState.DRAFT, PromptPlanState.PENDING_HITL, False),
            (PromptPlanState.PENDING_EVAL, PromptPlanState.ACTIVE, False),
            # illegal — backward
            (PromptPlanState.PENDING_HITL, PromptPlanState.PENDING_EVAL, False),
            (PromptPlanState.ACTIVE, PromptPlanState.PENDING_HITL, False),
            (PromptPlanState.ACTIVE, PromptPlanState.DRAFT, False),
            # illegal — terminal has no successor
            (PromptPlanState.ARCHIVED, PromptPlanState.ACTIVE, False),
            (PromptPlanState.ARCHIVED, PromptPlanState.DRAFT, False),
            # illegal — rejected cannot reactivate
            (PromptPlanState.REJECTED, PromptPlanState.ACTIVE, False),
            (PromptPlanState.REJECTED, PromptPlanState.PENDING_EVAL, False),
            # same-state is never a transition
            (PromptPlanState.DRAFT, PromptPlanState.DRAFT, False),
            (PromptPlanState.ACTIVE, PromptPlanState.ACTIVE, False),
        ],
    )
    def test_table(
        self,
        from_state: PromptPlanState,
        to_state: PromptPlanState,
        want: bool,
    ) -> None:
        assert can_transition(from_state, to_state) is want


# -----------------------------------------------------------------------------
# assert_transition() — fail-loud
# -----------------------------------------------------------------------------


class TestAssertTransition:
    def test_legal_transition_does_not_raise(self) -> None:
        # Should be a no-op (returns None) for every legal edge.
        assert_transition(PromptPlanState.DRAFT, PromptPlanState.PENDING_EVAL)
        assert_transition(PromptPlanState.PENDING_HITL, PromptPlanState.ACTIVE)

    def test_illegal_transition_raises(self) -> None:
        with pytest.raises(IllegalPromptTransitionError):
            assert_transition(PromptPlanState.DRAFT, PromptPlanState.ACTIVE)

    def test_same_state_raises(self) -> None:
        with pytest.raises(IllegalPromptTransitionError):
            assert_transition(PromptPlanState.ACTIVE, PromptPlanState.ACTIVE)

    def test_error_is_value_error_subclass(self) -> None:
        # Callers that catch the broad ValueError still see the failure.
        assert issubclass(IllegalPromptTransitionError, ValueError)

    def test_error_message_names_both_states(self) -> None:
        with pytest.raises(IllegalPromptTransitionError) as exc:
            assert_transition(PromptPlanState.ARCHIVED, PromptPlanState.ACTIVE)
        msg = str(exc.value)
        assert "archived" in msg
        assert "active" in msg


# -----------------------------------------------------------------------------
# is_terminal_state()
# -----------------------------------------------------------------------------


class TestIsTerminalState:
    def test_archived_is_terminal(self) -> None:
        assert is_terminal_state(PromptPlanState.ARCHIVED) is True

    def test_active_is_not_terminal(self) -> None:
        assert is_terminal_state(PromptPlanState.ACTIVE) is False

    def test_rejected_is_not_terminal(self) -> None:
        # rejected still progresses to archived.
        assert is_terminal_state(PromptPlanState.REJECTED) is False
