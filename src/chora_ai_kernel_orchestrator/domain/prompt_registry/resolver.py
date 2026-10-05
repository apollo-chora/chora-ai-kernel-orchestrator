"""``PromptResolver`` — the pure precedence/merge logic (ADR-197 M-B.1).

Modelled on ADR-178's ``PricePlanResolver`` (chora-identity
``internal/domain/user_mana/price_plan_resolver.go``). Holds NO SQL — it
delegates the lookup to a ``PromptOverrideRepository`` port and applies the
documented precedence ladder.

Precedence (ADR-197 §Addendum "M-B design grounded in deployed reality"):

    resolve(tenant_id, agent_id) :=
        active TENANT override plan      -> source="tenant_override"
        active PLATFORM override plan    -> source="platform_override"
        embedded default                 -> source="embedded" (segments={}, version=None)

WHOLE-ACTIVE-PLAN-WINS PER SCOPE (documented choice): if the active tenant plan
overrides ANY segment for the agent, its segment map is returned AS A WHOLE — the
platform plan is NOT merged in for the missing segments. Segments the winning
plan does not override fall back to the embedded default in the Go composer
(``overrideOr(segment_id, embedded)``). This avoids silent cross-scope blending
and keeps a tenant override a coherent, reviewable unit.

Fail-loud: an empty ``agent_id``, an unknown scope, or two active overrides at
the same scope are programming/data-integrity errors and raise ``ValueError``
rather than silently picking one.
"""

from __future__ import annotations

from .models import (
    SCOPE_PLATFORM,
    SCOPE_TENANT,
    SOURCE_PLATFORM_OVERRIDE,
    SOURCE_TENANT_OVERRIDE,
    VALID_SCOPES,
    Resolved,
    ScopeOverride,
)
from .repository import PromptOverrideRepository

# Precedence order (highest first) paired with the source label each scope wins
# under. Tenant override beats platform override beats embedded.
_PRECEDENCE: tuple[tuple[str, str], ...] = (
    (SCOPE_TENANT, SOURCE_TENANT_OVERRIDE),
    (SCOPE_PLATFORM, SOURCE_PLATFORM_OVERRIDE),
)


class PromptResolver:
    """Resolve the active prompt override for ``(tenant_id, agent_id)``.

    Pure domain service over a :class:`PromptOverrideRepository`. Unit-testable
    with an in-memory fake repo.
    """

    def __init__(self, repository: PromptOverrideRepository) -> None:
        if repository is None:
            raise ValueError("PromptResolver requires a PromptOverrideRepository")
        self._repo = repository

    async def resolve(self, tenant_id: str, agent_id: str) -> Resolved:
        """Return the winning :class:`Resolved` per the precedence ladder.

        No active override for ``(tenant, agent)`` → the embedded default
        (``Resolved(segments={}, version=None, source="embedded")``).
        """
        agent = (agent_id or "").strip()
        if not agent:
            raise ValueError("PromptResolver.resolve requires a non-empty agent_id")
        tenant = (tenant_id or "").strip()

        overrides = await self._repo.fetch_active_overrides(tenant_id=tenant, agent_id=agent)

        by_scope: dict[str, ScopeOverride] = {}
        for ov in overrides:
            if ov.scope not in VALID_SCOPES:
                raise ValueError(f"PromptResolver: unknown override scope {ov.scope!r} (agent_id={agent!r})")
            if ov.scope in by_scope:
                raise ValueError(
                    f"PromptResolver: more than one active {ov.scope!r} override "
                    f"for agent_id={agent!r} — the 0007 partial unique index "
                    f"should guarantee at most one active plan per scope"
                )
            by_scope[ov.scope] = ov

        # Whole-active-plan-wins per scope: tenant beats platform beats embedded.
        # A scope with an empty segment map does not override this agent → skip.
        for scope, source in _PRECEDENCE:
            chosen = by_scope.get(scope)
            if chosen is not None and chosen.segments:
                return Resolved(
                    segments=dict(chosen.segments),
                    version=chosen.version,
                    source=source,
                )

        # No active override → embedded default (byte-identical to today).
        return Resolved(segments={}, version=None, source="embedded")


__all__ = ["PromptResolver"]
