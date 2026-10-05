"""``PromptPromotionService`` — promotion orchestration (ADR-197 M-C.1).

Pure-ish domain service over two ports — a :class:`PromptOverrideRepository`
(the guarded write methods) and an :class:`ActivationAuditEmitter`. Holds NO
infrastructure (no psycopg, no Pub/Sub, no env): it validates every transition
against the pure :mod:`...state_machine` and delegates the writes.

Lifecycle methods::

    submit_for_eval   draft        -> pending_eval
    mark_eval_passed  pending_eval -> pending_hitl   (threads eval_run_id)
    mark_rejected     pending_eval | pending_hitl -> rejected
    activate          pending_hitl -> active         (audit-first, then atomic)

ACTIVATION ORDERING (load-bearing, ADR-197 §Decision IMDA D1): ``activate``
emits the activation audit BEFORE the status write. If the audit emit raises,
the plan is NOT activated — an un-audited activation must never happen
(fail-loud accountability gate). The residual inverse risk (audit lands, then
``activate_plan`` fails) is bounded: the audit row is in the outbox as a
pending event and the activation simply did not occur; a retry re-emits the
idempotent audit (same ``prompt_activation.{plan_id}.{actor}`` key) and
re-attempts activation.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from .repository import PromptOverrideRepository
from .state_machine import PromptPlanState, assert_transition

logger = logging.getLogger(__name__)


@runtime_checkable
class ActivationAuditEmitter(Protocol):
    """Port that records the IMDA-D1 activation audit (the pubsub adapter
    implements it; unit tests inject a recording fake)."""

    async def emit_activation(
        self,
        *,
        plan_id: str,
        actor_gcid: str,
        tenant_id: str,
        scope: str,
        annotation: str = "",
        traceparent: str = "",
        tracestate: str = "",
    ) -> str: ...


@runtime_checkable
class HITLRequestEmitter(Protocol):
    """Port that emits the HITL gate-requested event (ADR-197 M-C.2 — feeds the
    O+ Human-Oversight queue so a human can approve/reject the plan).

    The pubsub adapter (:class:`PromptHITLRequestEmitter`) implements it over the
    generic HITL outbox writer; unit tests inject a recording fake.
    """

    async def emit_hitl_request(
        self,
        *,
        plan_id: str,
        requester_gcid: str,
        tenant_id: str,
        scope: str,
        eval_run_id: str,
        agent_ids: list[str],
        segment_ids: list[str],
        summary: str = "",
        traceparent: str = "",
        tracestate: str = "",
    ) -> str: ...


class PromptPromotionService:
    """Drive the prompt-override promotion lifecycle over the repository +
    activation-audit ports."""

    def __init__(
        self,
        *,
        repo: PromptOverrideRepository,
        audit_emitter: ActivationAuditEmitter,
        hitl_emitter: HITLRequestEmitter | None = None,
    ) -> None:
        if repo is None:
            raise ValueError("PromptPromotionService requires a repo")
        if audit_emitter is None:
            raise ValueError("PromptPromotionService requires an audit_emitter")
        self._repo = repo
        self._audit = audit_emitter
        # Optional so the M-C.1 construction (repo + audit only) stays valid; the
        # M-C.2 HITL round-trip injects it. ``request_hitl_approval`` fails loud
        # when it is called without one.
        self._hitl = hitl_emitter

    async def submit_for_eval(self, plan_id: str) -> None:
        """draft -> pending_eval (queue the automated eval gate)."""
        assert_transition(PromptPlanState.DRAFT, PromptPlanState.PENDING_EVAL)
        await self._repo.update_plan_status(
            plan_id=plan_id,
            expected_from=PromptPlanState.DRAFT,
            to=PromptPlanState.PENDING_EVAL,
        )

    async def mark_eval_passed(self, plan_id: str, eval_run_id: str) -> None:
        """pending_eval -> pending_hitl (eval passed; await human sign-off)."""
        assert_transition(PromptPlanState.PENDING_EVAL, PromptPlanState.PENDING_HITL)
        await self._repo.update_plan_status(
            plan_id=plan_id,
            expected_from=PromptPlanState.PENDING_EVAL,
            to=PromptPlanState.PENDING_HITL,
            eval_run_id=eval_run_id,
        )

    async def request_hitl_approval(
        self,
        plan_id: str,
        *,
        requester_gcid: str,
        tenant_id: str,
        scope: str,
        eval_run_id: str,
        agent_ids: list[str] | None = None,
        segment_ids: list[str] | None = None,
        summary: str = "",
        traceparent: str = "",
        tracestate: str = "",
    ) -> None:
        """pending_eval -> pending_hitl, emitting the HITL gate request FIRST.

        Mirrors :meth:`activate`'s audit-first gate: the HITL gate event is
        emitted to the outbox BEFORE the status flip (emit-before-commit). If the
        emit raises, the plan is NOT moved to ``pending_hitl`` — a plan must
        never sit in ``pending_hitl`` without a gate in the O+ Human-Oversight
        queue (fail-loud). The residual inverse risk (gate lands, status flip
        fails) is bounded by the idempotent outbox key + the guarded UPDATE on
        retry, identical to ``activate``.

        Distinct from :meth:`mark_eval_passed` (which flips the same edge
        silently): use THIS method on the eval-passed path so the human gate is
        actually enqueued.
        """
        assert_transition(PromptPlanState.PENDING_EVAL, PromptPlanState.PENDING_HITL)
        if self._hitl is None:
            raise ValueError("request_hitl_approval requires a hitl_emitter (none injected at construction)")

        # (a) Human-oversight gate — emit the HITL request BEFORE the status
        #     write. A raise here aborts the transition (no un-gated pending_hitl).
        await self._hitl.emit_hitl_request(
            plan_id=plan_id,
            requester_gcid=requester_gcid,
            tenant_id=tenant_id,
            scope=scope,
            eval_run_id=eval_run_id,
            agent_ids=list(agent_ids or []),
            segment_ids=list(segment_ids or []),
            summary=summary,
            traceparent=traceparent,
            tracestate=tracestate,
        )

        # (b) Guarded transition (threads eval_run_id, same as mark_eval_passed).
        await self._repo.update_plan_status(
            plan_id=plan_id,
            expected_from=PromptPlanState.PENDING_EVAL,
            to=PromptPlanState.PENDING_HITL,
            eval_run_id=eval_run_id,
        )

    async def mark_rejected(
        self,
        plan_id: str,
        reason: str,
        *,
        from_state: PromptPlanState = PromptPlanState.PENDING_HITL,
    ) -> None:
        """{pending_eval | pending_hitl} -> rejected.

        ``from_state`` defaults to ``pending_hitl`` (a human reviewer rejecting
        at the HITL gate); the eval-failure path passes ``pending_eval``. The
        ``reason`` is logged (M-C.1 has no rejection-reason column; persisting it
        as a rejection annotation/event is M-C.2's concern — it is surfaced
        loudly here, never silently dropped).
        """
        assert_transition(from_state, PromptPlanState.REJECTED)
        logger.info(
            "prompt_override_plan.rejected",
            extra={
                "plan_id": plan_id,
                "from_state": str(from_state),
                "reason": reason,
            },
        )
        await self._repo.update_plan_status(
            plan_id=plan_id,
            expected_from=from_state,
            to=PromptPlanState.REJECTED,
        )

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
    ) -> None:
        """pending_hitl -> active. Audit FIRST, then activate atomically.

        ``approved_by`` is the admin GCID promoting the plan — it is both the
        audit actor and the activation row's ``approved_by``.
        """
        assert_transition(PromptPlanState.PENDING_HITL, PromptPlanState.ACTIVE)

        audit_tenant = tenant_id or ""
        note = annotation or f"activated prompt override plan {plan_id}"

        # (a) Accountability gate — emit the activation audit BEFORE the status
        #     write. A raise here aborts activation (no un-audited activation).
        await self._audit.emit_activation(
            plan_id=plan_id,
            actor_gcid=approved_by,
            tenant_id=audit_tenant,
            scope=scope,
            annotation=note,
            traceparent=traceparent,
            tracestate=tracestate,
        )

        # (b) Atomic archive-prior-then-activate. agent_id keeps the archive
        #     scoped to the same agent's override slot (0009 carve-out).
        await self._repo.activate_plan(
            plan_id=plan_id,
            scope=scope,
            tenant_id=tenant_id,
            approved_by=approved_by,
            agent_id=agent_id,
        )


__all__ = [
    "ActivationAuditEmitter",
    "HITLRequestEmitter",
    "PromptPromotionService",
]
