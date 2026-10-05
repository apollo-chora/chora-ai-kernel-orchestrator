"""Pure domain dataclasses for the prompt-override registry (ADR-197 M-B.1).

These are infrastructure-free value objects — no psycopg, no I/O, no env. The
adapter (``adapter/pg/prompt_override_repository.py``) maps DB rows onto these;
the resolver (``resolver.py``) operates over them.

Model (ADR-197 §Decision.2 + §Addendum "M-B design grounded in deployed
reality"): the registry is OVERRIDES-ONLY. The embedded default prompt stays
the engineer-deployed source of truth in the Go binary. When no active override
exists for ``(tenant, agent)`` the resolver returns the canonical
``Resolved(segments={}, version=None, source="embedded")`` and the Go composer
falls back to the embedded default → byte-identical behaviour.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Final

# Canonical override scopes. "platform" plans are world-readable (the
# platform-override precedence rung); "tenant" plans are RLS-scoped.
SCOPE_PLATFORM: Final[str] = "platform"
SCOPE_TENANT: Final[str] = "tenant"
VALID_SCOPES: Final[frozenset[str]] = frozenset({SCOPE_PLATFORM, SCOPE_TENANT})

# Canonical ``Resolved.source`` values (the precedence rung that won). These are
# the PINNED contract M-B.2 + the Go composers depend on.
SOURCE_TENANT_OVERRIDE: Final[str] = "tenant_override"
SOURCE_PLATFORM_OVERRIDE: Final[str] = "platform_override"
SOURCE_EMBEDDED: Final[str] = "embedded"


@dataclass(frozen=True)
class PromptOverridePlan:
    """A row of ``prompt_override_plan`` — a named, versioned, scoped plan.

    Precedence is encoded by ``scope`` + ``tenant_id``. At most one ``active``
    plan exists per scope (platform) / per tenant (enforced by partial unique
    indexes in migration 0007).
    """

    plan_id: str
    plan_code: str
    scope: str  # SCOPE_PLATFORM | SCOPE_TENANT
    tenant_id: str | None = None
    status: str = "draft"  # draft | active | archived
    created_by: str | None = None
    effective_from: _dt.datetime | None = None


@dataclass(frozen=True)
class PromptPlanRecord:
    """A minimal read of a ``prompt_override_plan`` row (ADR-197 M-C.2).

    The HITL-approval handler needs the plan's ``scope`` + ``tenant_id`` to
    drive activation (the activate path archives the prior active plan for the
    same scope/tenant) and the ``status`` to make a re-delivered approval an
    idempotent no-op (only a ``pending_hitl`` plan is promotable). Distinct from
    the richer :class:`PromptOverridePlan` — this is the narrow read contract the
    approval handler's :class:`PlanScopeReader` port returns.
    """

    plan_id: str
    scope: str  # SCOPE_PLATFORM | SCOPE_TENANT
    tenant_id: str | None
    status: str  # one of state_machine.PromptPlanState values
    # 0009 (CHO-2368): the per-agent plan axis + display version. None on
    # legacy agent-less plans; the approval handler threads agent_id into
    # activation so archiving stays scoped to the same agent's overrides.
    agent_id: str | None = None
    version_label: str | None = None


@dataclass(frozen=True)
class PromptOverrideSegment:
    """A row of ``prompt_override_segment`` — one overridden behavioural block.

    Only behavioural blocks ([ROLE]/[TASK]/[EXAMPLES]/tone) are ever stored;
    the structural [EXPECTED OUTPUT] contract + safety preamble are never
    overridable (ADR-197 §Decision.3) and always come from the embedded default.
    """

    segment_row_id: str
    plan_id: str
    agent_id: str
    segment_id: str
    body: str
    content_hash: str | None = None
    version: int = 1
    note: str = ""


@dataclass(frozen=True)
class ScopeOverride:
    """The active override for one agent at ONE scope (tenant OR platform).

    This is the unit the repository returns and the resolver chooses between.
    ``segments`` maps ``segment_id -> override body``. ``version`` is the active
    plan's opaque identifier (the plan_id string) — the version/cache token the
    caller stamps for explainability (ADR-197 §Decision.5). It is non-empty when
    a ScopeOverride exists.
    """

    scope: str  # SCOPE_PLATFORM | SCOPE_TENANT
    version: str
    segments: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Resolved:
    """PINNED CONTRACT — what ``PromptResolver.resolve`` returns.

    M-B.2 + the Go composers depend on this exact shape:

    * ``segments`` — ``segment_id -> override body`` (empty when source is
      embedded).
    * ``version`` — the winning override plan's opaque id, or ``None`` for the
      embedded default.
    * ``source`` — one of ``tenant_override`` / ``platform_override`` /
      ``embedded``.

    No active override → ``Resolved(segments={}, version=None,
    source="embedded")`` and the caller uses the embedded default
    (byte-identical to pre-registry behaviour).
    """

    segments: dict[str, str] = field(default_factory=dict)
    version: str | None = None
    source: str = SOURCE_EMBEDDED


# The canonical embedded fallback — returned whenever no active override wins.
EMBEDDED: Final[Resolved] = Resolved(segments={}, version=None, source=SOURCE_EMBEDDED)


@dataclass(frozen=True)
class CataloguePlanVersion:
    """One catalogue row of the per-agent version list (CHO-2368 read API).

    Covers both kinds: the immutable ``baseline`` transcription rows seeded by
    migration 0009 and the ``override`` plans that move through the six-state
    promotion gate. ``version_label`` is the display version ('v1', '1.1.0');
    ``plan_id`` stays the opaque internal token. Timestamps are ISO-8601
    strings (rendered by the SQL) so the wire shape is JSON-ready.
    """

    plan_id: str
    plan_code: str
    agent_id: str
    version_label: str
    kind: str  # 'baseline' | 'override'
    status: str  # one of state_machine.PromptPlanState values
    created_at: str
    activated_at: str | None
    approved_by: str | None
    eval_run_id: str | None


@dataclass(frozen=True)
class CatalogueSegmentView:
    """One segment of a catalogue version, with its display metadata.

    ``locked`` marks display-only segments (output contracts, safety preambles,
    fence frames) the override lane must never touch; ``position`` preserves
    the composition order for the O+ modal.
    """

    segment_id: str
    body: str
    locked: bool
    position: int
    content_hash: str | None
    note: str
    version: int


__all__ = [
    "EMBEDDED",
    "PromptOverridePlan",
    "PromptOverrideSegment",
    "PromptPlanRecord",
    "Resolved",
    "ScopeOverride",
    "SCOPE_PLATFORM",
    "SCOPE_TENANT",
    "SOURCE_EMBEDDED",
    "SOURCE_PLATFORM_OVERRIDE",
    "SOURCE_TENANT_OVERRIDE",
    "VALID_SCOPES",
]
