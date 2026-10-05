"""``PromptPromotionApprovalHandler`` — the consumer half of the ADR-197 M-C.2
HITL round-trip.

When a human approves/rejects a prompt-override plan in the O+ Human-Oversight
queue, the governance domain records a ``chora.governance.audit.recorded.v1``
entry (``AuditEntryRecorded``). The orchestrator subscribes to that topic; this
PURE domain handler decides — over ports only, no infrastructure imports —
whether a given decoded audit event is a prompt-override HITL decision and, if
so, drives the promotion transition:

* ``decision_approve`` -> :meth:`PromptPromotionService.activate` (with the
  plan's stored scope + tenant, read via the :class:`PlanScopeReader` port).
* ``decision_reject``  -> :meth:`PromptPromotionService.mark_rejected`.

STRICT filtering (load-bearing): the handler acts ONLY when
``target_resource_uri`` starts with ``hitl_decision:prompt-override-plan:``.
Every other audit event on the high-volume topic — including the orchestrator's
OWN ``ACTIVATE`` audit (prefix ``chora.ai_kernel/prompt_plan:``, action
``ACTIVATE``) — is left untouched. This prevents an activation→audit→activation
feedback loop.

Idempotent re-delivery: a replayed approval for an already-activated plan is a
safe no-op — the status pre-check skips it, and the guarded-transition
:class:`StalePromptTransitionError` is the TOCTOU backstop (caught + logged). A
genuine error (e.g. a dropped DB connection) is NOT a StaleError, so it
propagates → the consumer NACKs → Pub/Sub redelivers (fail-loud).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from .models import PromptPlanRecord
from .state_machine import (
    IllegalPromptTransitionError,
    PromptPlanState,
    StalePromptTransitionError,
)

logger = logging.getLogger(__name__)

# The HITL-decision resource URI prefix for a prompt-override plan. The O+
# approver records target_resource_uri = "hitl_decision:{decision_id}" and the
# prompt-registry decision_id is "prompt-override-plan:{plan_id}", so the full
# prefix the handler strips to recover the plan_id is:
RESOURCE_PREFIX = "hitl_decision:prompt-override-plan:"

# Action discriminants. Substring-matched against the lower-cased ``action`` so
# both ``decision_approve`` and an UPPER_SNAKE ``HITL_DECISION_APPROVE`` route.
ACTION_APPROVE = "decision_approve"
ACTION_REJECT = "decision_reject"


@runtime_checkable
class PlanScopeReader(Protocol):
    """Narrow read port the handler needs — the plan's scope/tenant/status.

    The pg ``PostgresPromptOverrideRepository`` satisfies it structurally
    (it exposes ``get_plan``); unit tests inject a fake.
    """

    async def get_plan(self, plan_id: str) -> PromptPlanRecord | None: ...


@runtime_checkable
class _PromotionService(Protocol):
    """Duck-typed :class:`PromptPromotionService` surface the handler drives."""

    async def activate(
        self,
        plan_id: str,
        approved_by: str,
        *,
        scope: str,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        annotation: str = "",
        traceparent: str = "",
        tracestate: str = "",
    ) -> None: ...

    async def mark_rejected(self, plan_id: str, reason: str) -> None: ...


class PromptPromotionApprovalHandler:
    """Drives the promotion transition from a decoded audit-recorded event."""

    def __init__(
        self,
        *,
        transition_service: _PromotionService,
        plan_reader: PlanScopeReader,
    ) -> None:
        if transition_service is None:
            raise ValueError("PromptPromotionApprovalHandler requires a transition_service")
        if plan_reader is None:
            raise ValueError("PromptPromotionApprovalHandler requires a plan_reader")
        self._svc = transition_service
        self._reader = plan_reader

    def matches(self, decoded: dict[str, Any]) -> bool:
        """Cheap STRICT gate — is this decoded audit a prompt-override HITL
        decision? Used by the consumer to skip the high-volume non-matching
        traffic without an inbox write."""
        uri = str((decoded or {}).get("target_resource_uri", ""))
        return uri.startswith(RESOURCE_PREFIX)

    async def handle(self, decoded: dict[str, Any]) -> bool:
        """Process one decoded ``AuditEntryRecorded``.

        Returns ``True`` when the event was a prompt-override HITL decision the
        handler consumed (whether or not it produced a transition — e.g. a
        missing plan or an idempotent replay still counts as consumed), and
        ``False`` when the event is out of scope (ignored). The caller acks in
        both cases; the boolean is for observability.
        """
        if not self.matches(decoded):
            # Not a prompt-promotion HITL decision — NEVER act on it.
            return False

        uri = str(decoded.get("target_resource_uri", ""))
        plan_id = uri[len(RESOURCE_PREFIX) :].strip()
        if not plan_id:
            logger.warning(
                "prompt_promotion_approval.malformed_resource_uri",
                extra={"target_resource_uri": uri},
            )
            return False

        action = str(decoded.get("action", "")).lower()
        actor_gcid = str(decoded.get("actor_gcid", "")).strip()

        if ACTION_APPROVE in action:
            await self._approve(plan_id, actor_gcid, decoded)
            return True
        if ACTION_REJECT in action:
            await self._reject(plan_id, decoded)
            return True

        logger.warning(
            "prompt_promotion_approval.unknown_action",
            extra={"plan_id": plan_id, "action": action},
        )
        return False

    # ---- internals --------------------------------------------------------

    async def _approve(self, plan_id: str, actor_gcid: str, decoded: dict[str, Any]) -> None:
        plan = await self._reader.get_plan(plan_id)
        if plan is None:
            logger.warning(
                "prompt_promotion_approval.plan_not_found",
                extra={"plan_id": plan_id},
            )
            return
        if plan.status != PromptPlanState.PENDING_HITL.value:
            # A replayed approve sees status='active' (or rejected/archived):
            # skip activate so we do NOT emit a spurious activation audit for a
            # non-pending plan. The guarded activate below is the TOCTOU backstop.
            logger.info(
                "prompt_promotion_approval.activate_skipped_status",
                extra={"plan_id": plan_id, "status": plan.status},
            )
            return
        try:
            await self._svc.activate(
                plan_id,
                actor_gcid,
                scope=plan.scope,
                tenant_id=plan.tenant_id,
                agent_id=plan.agent_id,
                annotation=str(decoded.get("annotation", "")),
                traceparent=str(decoded.get("traceparent", "")),
                tracestate=str(decoded.get("tracestate", "")),
            )
        except (IllegalPromptTransitionError, StalePromptTransitionError) as exc:
            # Race: the plan moved out of pending_hitl between the read and the
            # guarded UPDATE — idempotent no-op (a real infra error is neither
            # of these and propagates so the consumer NACKs).
            logger.info(
                "prompt_promotion_approval.activate_noop",
                extra={"plan_id": plan_id, "reason": str(exc)},
            )

    async def _reject(self, plan_id: str, decoded: dict[str, Any]) -> None:
        plan = await self._reader.get_plan(plan_id)
        if plan is None:
            logger.warning(
                "prompt_promotion_approval.plan_not_found",
                extra={"plan_id": plan_id},
            )
            return
        if plan.status != PromptPlanState.PENDING_HITL.value:
            logger.info(
                "prompt_promotion_approval.reject_skipped_status",
                extra={"plan_id": plan_id, "status": plan.status},
            )
            return
        reason = str(decoded.get("annotation", "")) or "rejected at HITL gate"
        try:
            await self._svc.mark_rejected(plan_id, reason=reason)
        except (IllegalPromptTransitionError, StalePromptTransitionError) as exc:
            logger.info(
                "prompt_promotion_approval.reject_noop",
                extra={"plan_id": plan_id, "reason": str(exc)},
            )


__all__ = [
    "ACTION_APPROVE",
    "ACTION_REJECT",
    "RESOURCE_PREFIX",
    "PlanScopeReader",
    "PromptPromotionApprovalHandler",
]
