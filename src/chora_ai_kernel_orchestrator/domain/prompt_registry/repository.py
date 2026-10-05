"""Repository port for the prompt-override registry (ADR-197 M-B.1).

The domain depends only on this Protocol — never on psycopg. The pg adapter
(``adapter/pg/prompt_override_repository.py``) implements it; unit tests inject
an in-memory fake. This keeps the precedence/merge logic in the pure
``PromptResolver`` and the SQL in the adapter (hexagonal: adapters depend on the
domain port, never the reverse).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import PromptPlanRecord, ScopeOverride
from .state_machine import PromptPlanState


@runtime_checkable
class PromptOverrideRepository(Protocol):
    """Port over the prompt-override registry — the M-B.1 read path plus the
    M-C.1 promotion write paths.

    Read contract (:meth:`fetch_active_overrides`):

    * Returns a list with AT MOST one entry per scope — at most one tenant-scope
      ``ScopeOverride`` and one platform-scope ``ScopeOverride`` (each backed by
      that scope's single active plan; enforced by the 0007 partial unique
      indexes).
    * Only segments overriding the requested ``agent_id`` are included. A scope
      whose active plan does not override this agent contributes no entry.
    * Empty list when no active override applies → resolver returns the embedded
      default.
    * ``tenant_id`` may be empty/blank for a tenant-agnostic resolution; in that
      case only platform-scope overrides can be returned.

    Precedence between the returned scopes is decided by ``PromptResolver`` (the
    pure domain), NOT here.

    Write contract (M-C.1 — behaviour-neutral, no caller wires these yet):

    * :meth:`create_draft_plan` mints a UUIDv7 plan in ``draft``.
    * :meth:`add_segment` appends an override block to a plan.
    * :meth:`update_plan_status` is a GUARDED transition (``WHERE
      status = expected_from``) — fail-loud on 0 rows (illegal/stale change).
    * :meth:`activate_plan` ATOMICALLY archives the prior active plan for the
      scope/tenant then promotes this plan to ``active`` (one transaction; the
      0007 partial unique index keeps "one active per scope/tenant").

    The legal edges are owned by the pure
    :mod:`...domain.prompt_registry.state_machine`; the guarded UPDATE is the DB
    enforcement of the same lifecycle.
    """

    async def fetch_active_overrides(self, *, tenant_id: str, agent_id: str) -> list[ScopeOverride]: ...

    async def get_plan(self, plan_id: str) -> PromptPlanRecord | None:
        """Read a plan's scope/tenant/status (M-C.2). Returns ``None`` when the
        plan does not exist.

        The HITL-approval handler reads the scope + tenant here to drive
        ``activate`` (which archives the prior active plan for the same
        scope/tenant) and the status to make a re-delivered approval an
        idempotent no-op.
        """
        ...

    async def create_draft_plan(
        self,
        *,
        plan_code: str,
        scope: str,
        tenant_id: str | None,
        created_by: str,
        agent_id: str | None = None,
        version_label: str | None = None,
    ) -> str:
        """Insert a new ``draft`` plan and return its UUIDv7 ``plan_id``.

        ``agent_id`` scopes the plan to one agent's active-override slot
        (0009); ``version_label`` is the display version ('1.1.0') stamped
        into prompt_conditions and shown on O+.
        """
        ...

    async def add_segment(
        self,
        *,
        plan_id: str,
        agent_id: str,
        segment_id: str,
        body: str,
        content_hash: str | None = None,
        note: str = "",
    ) -> str:
        """Append an override segment to ``plan_id``; return the segment row id.

        ``content_hash`` defaults to ``sha256(body)`` hex when not supplied.
        """
        ...

    async def update_plan_status(
        self,
        *,
        plan_id: str,
        expected_from: PromptPlanState,
        to: PromptPlanState,
        approved_by: str | None = None,
        eval_run_id: str | None = None,
    ) -> None:
        """Guarded status transition. Raises if no row is in ``expected_from``."""
        ...

    async def activate_plan(
        self,
        *,
        plan_id: str,
        scope: str,
        tenant_id: str | None,
        approved_by: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Atomically archive the prior active OVERRIDE plan for the
        scope/tenant AND agent, then promote ``plan_id`` to ``active``. Raises
        if ``plan_id`` is not in ``pending_hitl`` (stale activation). Baseline
        catalogue plans are never archived (0009 carve-out)."""
        ...


__all__ = ["PromptOverrideRepository"]
