"""Prompt-override promotion state machine — pure functions on the lifecycle.

ADR-197 M-C.1. Mirrors the closure-saga state machine
(``services/chora-closure-orchestrator/.../domain/closure/state.py``): a pure,
infrastructure-free transition table plus a fail-loud assert. The DB enforces
the same lifecycle through the 0008 status CHECK + the guarded UPDATE
(``WHERE status = expected_from``) in the pg adapter — this module is the
single source of truth for *which* edges are legal.

Lifecycle::

    draft -> pending_eval -> pending_hitl -> active -> archived
                 |                  |
                 +--> rejected <----+ --> archived

* ``draft``        — authored, not yet submitted.
* ``pending_eval`` — submitted; automated eval gate running.
* ``pending_hitl`` — eval passed; awaiting human (HITL) sign-off.
* ``active``       — promoted; the live override for its scope/tenant (the 0007
                     partial unique index guarantees one active per scope/tenant).
* ``rejected``     — failed eval OR HITL; not promotable.
* ``archived``     — terminal; a superseded active plan or a retired rejected one.

Hexagonal: dependency-free w.r.t. infrastructure.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class PromptPlanState(StrEnum):
    """Promotion lifecycle state. Values match the migration 0008
    ``prompt_override_plan_status_chk`` literals exactly (less no prefix)."""

    DRAFT = "draft"
    PENDING_EVAL = "pending_eval"
    PENDING_HITL = "pending_hitl"
    ACTIVE = "active"
    REJECTED = "rejected"
    ARCHIVED = "archived"


# Canonical ordered list (table-driven tests, admin dropdowns, dashboards).
ALL_STATES: Final[tuple[PromptPlanState, ...]] = (
    PromptPlanState.DRAFT,
    PromptPlanState.PENDING_EVAL,
    PromptPlanState.PENDING_HITL,
    PromptPlanState.ACTIVE,
    PromptPlanState.REJECTED,
    PromptPlanState.ARCHIVED,
)


# Forward-only progressions (no backward edges). A plan fails forward into
# ``rejected`` from either gate, and both ``active`` + ``rejected`` retire into
# the terminal ``archived``.
_ALLOWED: Final[dict[PromptPlanState, frozenset[PromptPlanState]]] = {
    PromptPlanState.DRAFT: frozenset({PromptPlanState.PENDING_EVAL}),
    PromptPlanState.PENDING_EVAL: frozenset({PromptPlanState.PENDING_HITL, PromptPlanState.REJECTED}),
    PromptPlanState.PENDING_HITL: frozenset({PromptPlanState.ACTIVE, PromptPlanState.REJECTED}),
    PromptPlanState.ACTIVE: frozenset({PromptPlanState.ARCHIVED}),
    PromptPlanState.REJECTED: frozenset({PromptPlanState.ARCHIVED}),
    PromptPlanState.ARCHIVED: frozenset(),  # terminal
}


class IllegalPromptTransitionError(ValueError):
    """Raised by :func:`assert_transition` on a disallowed state change.

    Subclasses ``ValueError`` so callers that catch the broad error (the
    repository / transition service fail-loud paths) still observe it.
    """


class StalePromptTransitionError(RuntimeError):
    """Raised by the repository's GUARDED ``UPDATE`` when no row matched the
    expected ``from`` state (0 rows updated) — a stale or already-applied
    transition.

    Subclasses ``RuntimeError`` so existing callers that catch the broad error
    (and existing tests asserting ``pytest.raises(RuntimeError)``) still observe
    it. The M-C.2 HITL-approval handler catches THIS specifically to treat a
    Pub/Sub re-delivery as an idempotent no-op (a genuine infra error — e.g. a
    dropped connection — surfaces as a different exception and still propagates,
    so the consumer NACKs + Pub/Sub redelivers: fail-loud, never swallow).
    """


def can_transition(from_state: PromptPlanState, to_state: PromptPlanState) -> bool:
    """Report whether ``from_state -> to_state`` is permitted.

    Same-state transitions (``from == to``) always return ``False``.
    """
    if from_state == to_state:
        return False
    successors = _ALLOWED.get(from_state)
    if successors is None:
        return False
    return to_state in successors


def assert_transition(from_state: PromptPlanState, to_state: PromptPlanState) -> None:
    """Fail loud unless ``from_state -> to_state`` is a legal edge.

    No-op on a legal transition; raises :class:`IllegalPromptTransitionError`
    otherwise. The message names both states for diagnosability.
    """
    if not can_transition(from_state, to_state):
        raise IllegalPromptTransitionError(f"illegal prompt-plan transition {from_state} -> {to_state}")


def is_terminal_state(s: PromptPlanState) -> bool:
    """Report whether ``s`` is terminal (no successor)."""
    successors = _ALLOWED.get(s)
    if successors is None:
        return True
    return len(successors) == 0


__all__ = [
    "ALL_STATES",
    "IllegalPromptTransitionError",
    "PromptPlanState",
    "StalePromptTransitionError",
    "assert_transition",
    "can_transition",
    "is_terminal_state",
]
