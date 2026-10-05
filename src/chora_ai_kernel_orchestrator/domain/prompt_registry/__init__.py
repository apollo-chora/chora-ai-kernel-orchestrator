"""Prompt-override registry bounded context (ADR-197 M-B.1).

Pure domain for the optional prompt-override layer that sits on top of every
agent's embedded default prompt. Overrides-only: zero rows → embedded default →
byte-identical behaviour. Behaviour-neutral in M-B.1 (NOT wired into the request
path; the resolver is wired into the composers in M-B.2).

Public surface:

* :class:`PromptResolver` — pure precedence/merge service.
* :class:`PromptOverrideRepository` — the repository port (pg adapter implements).
* :class:`Resolved` — the PINNED return contract (segments / version / source).
* :class:`ScopeOverride` / :class:`PromptOverridePlan` / :class:`PromptOverrideSegment`
  — value objects.
"""

from __future__ import annotations

from .embedded_versions import EMBEDDED_PROMPT_VERSIONS
from .models import (
    EMBEDDED,
    SCOPE_PLATFORM,
    SCOPE_TENANT,
    SOURCE_EMBEDDED,
    SOURCE_PLATFORM_OVERRIDE,
    SOURCE_TENANT_OVERRIDE,
    VALID_SCOPES,
    CataloguePlanVersion,
    CatalogueSegmentView,
    PromptOverridePlan,
    PromptOverrideSegment,
    PromptPlanRecord,
    Resolved,
    ScopeOverride,
)
from .promotion_approval_handler import (
    PlanScopeReader,
    PromptPromotionApprovalHandler,
)
from .repository import PromptOverrideRepository
from .resolver import PromptResolver
from .state_machine import (
    ALL_STATES,
    IllegalPromptTransitionError,
    PromptPlanState,
    StalePromptTransitionError,
    assert_transition,
    can_transition,
    is_terminal_state,
)
from .transition_service import (
    ActivationAuditEmitter,
    HITLRequestEmitter,
    PromptPromotionService,
)

__all__ = [
    "ALL_STATES",
    "EMBEDDED",
    "EMBEDDED_PROMPT_VERSIONS",
    "ActivationAuditEmitter",
    "CataloguePlanVersion",
    "CatalogueSegmentView",
    "HITLRequestEmitter",
    "IllegalPromptTransitionError",
    "PlanScopeReader",
    "PromptOverridePlan",
    "PromptOverrideRepository",
    "PromptOverrideSegment",
    "PromptPlanRecord",
    "PromptPlanState",
    "PromptPromotionApprovalHandler",
    "PromptPromotionService",
    "PromptResolver",
    "Resolved",
    "SCOPE_PLATFORM",
    "SCOPE_TENANT",
    "SOURCE_EMBEDDED",
    "SOURCE_PLATFORM_OVERRIDE",
    "SOURCE_TENANT_OVERRIDE",
    "ScopeOverride",
    "StalePromptTransitionError",
    "VALID_SCOPES",
    "assert_transition",
    "can_transition",
    "is_terminal_state",
]
