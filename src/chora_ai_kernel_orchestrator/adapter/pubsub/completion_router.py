"""CompletionRouter: ONE role -> runner binding for every dispatch completion.

ADR-254 D5. Before this, each lane built its own ``AgentCompletionPubsubLoop``
bound to one runner and a hard-coded role bundle (``OE_DISPATCH_ROLES``); the
growth-edge conversion shipped with nothing consuming its completions because
the loop knew only OE's roles (docs/TODO-DEVELOPMENT.md, 2026-08-21). The
router is the single place a role is bound to the runner that resumes it. The
completion loop, the park reaper and the generic single-agent workflow all
route through it, and a role dispatched without a registered runner is a
STARTUP error (``require_roles_registered``), never a silent sink.

It also settles the park ledger (migration 0056):

  * a real completion resumes the runner, then flips the row to 'completed';
  * a completion for a row the reaper already settled is RECORDED
    (``late_completion_at``) and never resumed; the run already carries its
    FAILED status and resuming it twice would double-settle it;
  * a completion for a thread with NO ledger row (parked before 0056 landed)
    still resumes: refusing it would strand the OE runs parked during the
    2026-08-21 outage. The backfill at startup closes most of that gap; this
    branch covers whatever it could not see.

A runner failure propagates: the subscriber NACKs, Pub/Sub redelivers, the
row stays 'parked' for the retry or the reaper. The ledger is flipped only
after the resume returned.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.domain.agent_dispatch.park import ParkRecord

logger = logging.getLogger(__name__)

OUTCOME_RESUMED = "resumed"
OUTCOME_RESUMED_UNLEDGERED = "resumed_unledgered"
OUTCOME_LATE = "late"


class _RunnerLike(Protocol):
    async def handle_completion(self, completion: dict[str, Any]) -> Any: ...


class _LedgerLike(Protocol):
    async def get(self, idempotency_key: str) -> ParkRecord | None: ...
    async def mark_completed(self, idempotency_key: str) -> bool: ...
    async def record_late_completion(self, idempotency_key: str) -> None: ...


class CompletionRouter:
    """Routes a completion to the runner registered for its ``agent_role`` and
    the CREW the park ledger row names (ADR-254 D5: one role can be dispatched
    by several workflows, e.g. ``companion_chat`` by the reflection workflow,
    the diagnosis voice step and the typed-chat lane)."""

    def __init__(self, *, ledger: _LedgerLike) -> None:
        if ledger is None:
            raise ValueError("CompletionRouter: a park ledger store is required")
        self._ledger = ledger
        # (role, crew) -> runner. crew "" = a legacy registration without one.
        self._runners: dict[tuple[str, str], _RunnerLike] = {}

    # ---- registry -------------------------------------------------------

    def register(self, agent_role: str, runner: _RunnerLike, *, crew: str = "") -> None:
        role = _require(agent_role, "agent_role")
        crew_name = (crew or "").strip()
        if runner is None or not hasattr(runner, "handle_completion"):
            raise ValueError(f"CompletionRouter: runner for {role!r} must expose handle_completion")
        if (role, crew_name) in self._runners:
            raise ValueError(
                f"CompletionRouter: agent_role {role!r} is already bound for crew "
                f"{crew_name or '(none)'!r} to {type(self._runners[(role, crew_name)]).__name__}; "
                "a (role, crew) resumes through exactly one runner"
            )
        self._runners[(role, crew_name)] = runner

    @property
    def roles(self) -> list[str]:
        return sorted({role for role, _ in self._runners})

    def crews_for(self, agent_role: str) -> list[str]:
        role = _require(agent_role, "agent_role")
        return sorted(crew for r, crew in self._runners if r == role)

    def runner_for(self, agent_role: str, *, crew: str | None = None) -> _RunnerLike:
        """The runner for (role, crew). With no crew hint (an unledgered
        completion) or an unknown crew (a renamed one) the role's ONLY runner
        is returned; a shared role without a usable hint is refused, never
        guessed: resuming the wrong workflow would inject one run's answer
        into another."""
        role = _require(agent_role, "agent_role")
        bound = {c: r for (rl, c), r in self._runners.items() if rl == role}
        if not bound:
            raise ValueError(
                f"CompletionRouter: no runner registered for agent_role {role!r}; registered roles: {self.roles}"
            )
        hint = (crew or "").strip()
        if hint and hint in bound:
            return bound[hint]
        if len(bound) == 1:
            only_crew, runner = next(iter(bound.items()))
            if hint:
                logger.warning(
                    "completion_router.crew_hint_unknown_single_runner",
                    extra={"agent_role": role, "crew_hint": hint, "bound_crew": only_crew},
                )
            return runner
        raise ValueError(
            f"CompletionRouter: agent_role {role!r} is bound to several crews {sorted(bound)} "
            f"and the completion names {hint or 'no crew'!r}: ambiguous, refusing to guess"
        )

    def require_roles_registered(self, agent_roles: Iterable[str]) -> None:
        """Refuse to start a lane that dispatches a role nobody resumes."""
        missing = sorted({r for r in agent_roles if r not in self.roles})
        if missing:
            raise ValueError(
                "CompletionRouter: dispatch roles without a registered completion "
                f"runner: {missing}. A run parked on one of these could never be "
                "resumed; refusing to start."
            )

    # ---- routing --------------------------------------------------------

    async def handle_completion(self, completion: dict[str, Any]) -> str:
        """The ``_RunnerLike`` surface ``AgentCompletionSubscriber`` drives.

        Returns the outcome (``resumed`` | ``resumed_unledgered`` | ``late``)
        for callers and tests; the subscriber ignores the value.
        """
        role, key = _role_and_key(completion)
        park = await self._ledger.get(key)
        if park is None:
            runner = self.runner_for(role)
            logger.warning(
                "completion_router.unledgered_completion",
                extra={"agent_role": role, "idempotency_key": key, "thread_id": completion.get("thread_id", "")},
            )
            await runner.handle_completion(completion)
            return OUTCOME_RESUMED_UNLEDGERED
        if not park.is_parked:
            await self._ledger.record_late_completion(key)
            logger.warning(
                "completion_router.late_completion",
                extra={
                    "agent_role": role,
                    "idempotency_key": key,
                    "thread_id": park.thread_id,
                    "state": park.state.value,
                    "settled_by": park.settled_by,
                },
            )
            return OUTCOME_LATE
        runner = self.runner_for(role, crew=park.crew)
        await runner.handle_completion(completion)
        await self._ledger.mark_completed(key)
        return OUTCOME_RESUMED

    async def resume_reaped(self, completion: dict[str, Any]) -> None:
        """Resume a parked thread with a reaper-synthesized FAILED completion.

        Does NOT touch the ledger: the reaper flips the row to 'reaped' itself,
        after this returns, so a failed resume leaves the row parked. The reaper
        stamps the park's ``crew`` on the completion so a shared role routes.
        """
        role, _ = _role_and_key(completion)
        runner = self.runner_for(role, crew=str(completion.get("crew") or ""))
        await runner.handle_completion(completion)


def _role_and_key(completion: dict[str, Any]) -> tuple[str, str]:
    role = str(completion.get("agent_role") or "").strip()
    key = str(completion.get("idempotency_key") or "").strip()
    if not role:
        raise ValueError("CompletionRouter: completion carries no agent_role")
    if not key:
        raise ValueError("CompletionRouter: completion carries no idempotency_key")
    return role, key


def _require(value: str, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"CompletionRouter: {field} is required")
    return text


__all__ = [
    "OUTCOME_LATE",
    "OUTCOME_RESUMED",
    "OUTCOME_RESUMED_UNLEDGERED",
    "CompletionRouter",
]
