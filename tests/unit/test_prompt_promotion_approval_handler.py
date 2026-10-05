"""Unit tests for ``PromptPromotionApprovalHandler`` (ADR-197 M-C.2).

The handler is the consumer half of the HITL round-trip: it receives a decoded
``chora.governance.audit.recorded.v1`` event and, ONLY when it is a
prompt-override HITL decision (target_resource_uri starts with
``hitl_decision:prompt-override-plan:``), drives the promotion transition:

* ``decision_approve`` -> ``transition_service.activate(plan_id, ...)`` with the
  plan's stored scope + tenant.
* ``decision_reject``  -> ``transition_service.mark_rejected(plan_id, reason)``.

Critical invariants:
  * STRICT filtering — a non-prompt-promotion audit event (incl. the
    orchestrator's OWN ``ACTIVATE`` audit) is NEVER acted on (no transition).
  * Idempotent replay — a re-delivered approve for an already-activated plan is
    a safe no-op (status pre-check + the guarded-transition StaleError backstop).
  * Domain-only — the handler calls ports, no infra imports.
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.domain.prompt_registry.models import (
    PromptPlanRecord,
)
from chora_ai_kernel_orchestrator.domain.prompt_registry.promotion_approval_handler import (
    PromptPromotionApprovalHandler,
)
from chora_ai_kernel_orchestrator.domain.prompt_registry.state_machine import (
    StalePromptTransitionError,
)

_PLAN = "01970000-0000-7000-8000-0000000000aa"
_URI = f"hitl_decision:prompt-override-plan:{_PLAN}"


# -----------------------------------------------------------------------------
# Recording fakes
# -----------------------------------------------------------------------------


class _FakeService:
    def __init__(
        self,
        *,
        activate_exc: Exception | None = None,
        reject_exc: Exception | None = None,
    ) -> None:
        self.activate_calls: list[dict[str, Any]] = []
        self.reject_calls: list[dict[str, Any]] = []
        self._activate_exc = activate_exc
        self._reject_exc = reject_exc

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
        self.activate_calls.append(
            {
                "plan_id": plan_id,
                "approved_by": approved_by,
                "scope": scope,
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "annotation": annotation,
            }
        )
        if self._activate_exc is not None:
            raise self._activate_exc

    async def mark_rejected(self, plan_id: str, reason: str, **_: Any) -> None:
        self.reject_calls.append({"plan_id": plan_id, "reason": reason})
        if self._reject_exc is not None:
            raise self._reject_exc


class _FakeReader:
    def __init__(self, record: PromptPlanRecord | None) -> None:
        self._record = record
        self.calls: list[str] = []

    async def get_plan(self, plan_id: str) -> PromptPlanRecord | None:
        self.calls.append(plan_id)
        return self._record


def _handler(
    record: PromptPlanRecord | None,
    *,
    activate_exc: Exception | None = None,
    reject_exc: Exception | None = None,
) -> tuple[PromptPromotionApprovalHandler, _FakeService, _FakeReader]:
    svc = _FakeService(activate_exc=activate_exc, reject_exc=reject_exc)
    reader = _FakeReader(record)
    handler = PromptPromotionApprovalHandler(transition_service=svc, plan_reader=reader)
    return handler, svc, reader


def _pending() -> PromptPlanRecord:
    return PromptPlanRecord(
        plan_id=_PLAN,
        scope="tenant",
        tenant_id="t-1",
        status="pending_hitl",
        agent_id="qgen_question",
        version_label="1.1.0",
    )


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_requires_transition_service(self) -> None:
        with pytest.raises(ValueError, match="transition_service"):
            PromptPromotionApprovalHandler(transition_service=None, plan_reader=_FakeReader(None))

    def test_requires_plan_reader(self) -> None:
        with pytest.raises(ValueError, match="plan_reader"):
            PromptPromotionApprovalHandler(transition_service=_FakeService(), plan_reader=None)


# -----------------------------------------------------------------------------
# matches() — strict filtering
# -----------------------------------------------------------------------------


class TestMatches:
    def test_matches_prompt_promotion_hitl_decision(self) -> None:
        handler, _, _ = _handler(_pending())
        assert handler.matches({"target_resource_uri": _URI}) is True

    def test_does_not_match_other_audit(self) -> None:
        handler, _, _ = _handler(_pending())
        assert handler.matches({"target_resource_uri": "chora.creation/atom:1"}) is False

    def test_does_not_match_own_activate_audit(self) -> None:
        # The orchestrator's OWN activation audit uses a DIFFERENT prefix —
        # it must never loop back into another activation.
        handler, _, _ = _handler(_pending())
        own = {"target_resource_uri": f"chora.ai_kernel/prompt_plan:{_PLAN}"}
        assert handler.matches(own) is False


# -----------------------------------------------------------------------------
# approve
# -----------------------------------------------------------------------------


class TestApprove:
    @pytest.mark.asyncio
    async def test_approve_activates_with_plan_scope_and_tenant(self) -> None:
        handler, svc, reader = _handler(_pending())
        acted = await handler.handle(
            {
                "target_resource_uri": _URI,
                "action": "decision_approve",
                "actor_gcid": "approver-1",
                "annotation": "looks good",
            }
        )
        assert acted is True
        assert reader.calls == [_PLAN]
        assert len(svc.activate_calls) == 1
        call = svc.activate_calls[0]
        assert call["plan_id"] == _PLAN
        assert call["approved_by"] == "approver-1"
        assert call["scope"] == "tenant"
        assert call["tenant_id"] == "t-1"
        assert call["agent_id"] == "qgen_question"
        assert call["annotation"] == "looks good"
        assert svc.reject_calls == []

    @pytest.mark.asyncio
    async def test_approve_uppercase_action_substring(self) -> None:
        handler, svc, _ = _handler(_pending())
        await handler.handle(
            {
                "target_resource_uri": _URI,
                "action": "HITL_DECISION_APPROVE",
                "actor_gcid": "approver-1",
            }
        )
        assert len(svc.activate_calls) == 1

    @pytest.mark.asyncio
    async def test_platform_scope_activation(self) -> None:
        record = PromptPlanRecord(plan_id=_PLAN, scope="platform", tenant_id=None, status="pending_hitl")
        handler, svc, _ = _handler(record)
        await handler.handle(
            {
                "target_resource_uri": _URI,
                "action": "decision_approve",
                "actor_gcid": "approver-1",
            }
        )
        call = svc.activate_calls[0]
        assert call["scope"] == "platform"
        assert call["tenant_id"] is None

    @pytest.mark.asyncio
    async def test_approve_missing_plan_acks_without_activate(self) -> None:
        handler, svc, _ = _handler(None)  # get_plan returns None
        acted = await handler.handle(
            {
                "target_resource_uri": _URI,
                "action": "decision_approve",
                "actor_gcid": "approver-1",
            }
        )
        assert acted is True  # consumed (ack), but no transition
        assert svc.activate_calls == []


# -----------------------------------------------------------------------------
# reject
# -----------------------------------------------------------------------------


class TestReject:
    @pytest.mark.asyncio
    async def test_reject_marks_rejected_with_annotation_reason(self) -> None:
        handler, svc, _ = _handler(_pending())
        acted = await handler.handle(
            {
                "target_resource_uri": _URI,
                "action": "decision_reject",
                "actor_gcid": "approver-1",
                "annotation": "tone too casual",
            }
        )
        assert acted is True
        assert svc.activate_calls == []
        assert svc.reject_calls == [{"plan_id": _PLAN, "reason": "tone too casual"}]

    @pytest.mark.asyncio
    async def test_reject_defaults_reason_when_no_annotation(self) -> None:
        handler, svc, _ = _handler(_pending())
        await handler.handle(
            {
                "target_resource_uri": _URI,
                "action": "decision_reject",
                "actor_gcid": "approver-1",
            }
        )
        assert svc.reject_calls[0]["reason"]  # non-empty default


# -----------------------------------------------------------------------------
# strict filtering — non-prompt-promotion audit events are ignored
# -----------------------------------------------------------------------------


class TestIgnoresNonMatching:
    @pytest.mark.asyncio
    async def test_other_audit_event_not_acted_on(self) -> None:
        handler, svc, reader = _handler(_pending())
        acted = await handler.handle(
            {
                "target_resource_uri": "chora.creation/atom:1",
                "action": "decision_approve",  # action matches but resource doesn't
                "actor_gcid": "x",
            }
        )
        assert acted is False
        assert svc.activate_calls == []
        assert svc.reject_calls == []
        assert reader.calls == []  # never even read the plan

    @pytest.mark.asyncio
    async def test_own_activate_audit_not_reactivated(self) -> None:
        handler, svc, _ = _handler(_pending())
        acted = await handler.handle(
            {
                "target_resource_uri": f"chora.ai_kernel/prompt_plan:{_PLAN}",
                "action": "ACTIVATE",
                "actor_gcid": "x",
            }
        )
        assert acted is False
        assert svc.activate_calls == []

    @pytest.mark.asyncio
    async def test_matching_resource_unknown_action_not_acted(self) -> None:
        handler, svc, _ = _handler(_pending())
        acted = await handler.handle(
            {
                "target_resource_uri": _URI,
                "action": "decision_escalate",
                "actor_gcid": "x",
            }
        )
        assert acted is False
        assert svc.activate_calls == []
        assert svc.reject_calls == []


# -----------------------------------------------------------------------------
# idempotent replay
# -----------------------------------------------------------------------------


class TestIdempotentReplay:
    @pytest.mark.asyncio
    async def test_replay_on_already_active_plan_skips_activate(self) -> None:
        # status pre-check: a replayed approve sees status='active' -> no-op.
        record = PromptPlanRecord(plan_id=_PLAN, scope="tenant", tenant_id="t-1", status="active")
        handler, svc, _ = _handler(record)
        acted = await handler.handle(
            {
                "target_resource_uri": _URI,
                "action": "decision_approve",
                "actor_gcid": "approver-1",
            }
        )
        assert acted is True  # consumed/ack, no crash
        assert svc.activate_calls == []

    @pytest.mark.asyncio
    async def test_toctou_stale_activate_is_caught(self) -> None:
        # Race: status reads pending_hitl but the guarded activate matched 0
        # rows (StalePromptTransitionError) — caught, treated as idempotent.
        handler, svc, _ = _handler(
            _pending(),
            activate_exc=StalePromptTransitionError("activate_plan: 0 rows ... not in pending_hitl"),
        )
        acted = await handler.handle(
            {
                "target_resource_uri": _URI,
                "action": "decision_approve",
                "actor_gcid": "approver-1",
            }
        )
        assert acted is True  # did not raise
        assert len(svc.activate_calls) == 1  # attempted once

    @pytest.mark.asyncio
    async def test_toctou_stale_reject_is_caught(self) -> None:
        handler, svc, _ = _handler(
            _pending(),
            reject_exc=StalePromptTransitionError("update_plan_status: 0 rows ..."),
        )
        acted = await handler.handle(
            {
                "target_resource_uri": _URI,
                "action": "decision_reject",
                "actor_gcid": "approver-1",
            }
        )
        assert acted is True

    @pytest.mark.asyncio
    async def test_unexpected_activate_error_propagates(self) -> None:
        # A non-stale error (e.g. DB down) MUST propagate so the consumer NACKs
        # and Pub/Sub redelivers (fail-loud — never swallow a real failure).
        handler, _, _ = _handler(_pending(), activate_exc=RuntimeError("connection reset"))
        with pytest.raises(RuntimeError, match="connection reset"):
            await handler.handle(
                {
                    "target_resource_uri": _URI,
                    "action": "decision_approve",
                    "actor_gcid": "approver-1",
                }
            )
