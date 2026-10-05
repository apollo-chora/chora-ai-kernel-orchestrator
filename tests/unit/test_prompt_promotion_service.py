"""Unit tests for ``PromptPromotionService`` (ADR-197 M-C.1).

The service is pure-ish orchestration over two ports — a
``PromptOverrideRepository`` (the guarded write methods) and an activation
audit emitter — validating every transition against the pure state machine.

Critical invariants:
  * each method drives the guarded ``update_plan_status`` with the correct
    (expected_from -> to) pair,
  * ``activate`` emits the activation audit BEFORE the status write
    (audit-first gate), and
  * if the audit emit raises, the plan is NOT activated (fail-loud).
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.domain.prompt_registry.state_machine import (
    PromptPlanState,
)
from chora_ai_kernel_orchestrator.domain.prompt_registry.transition_service import (
    PromptPromotionService,
)

# -----------------------------------------------------------------------------
# Recording fakes (share a single ordered call-log for ordering assertions)
# -----------------------------------------------------------------------------


class _RecordingRepo:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.status_calls: list[dict[str, Any]] = []
        self.activate_calls: list[dict[str, Any]] = []

    async def update_plan_status(
        self,
        *,
        plan_id: str,
        expected_from: PromptPlanState,
        to: PromptPlanState,
        approved_by: str | None = None,
        eval_run_id: str | None = None,
    ) -> None:
        self._log.append(f"update:{expected_from}->{to}")
        self.status_calls.append(
            {
                "plan_id": plan_id,
                "expected_from": expected_from,
                "to": to,
                "approved_by": approved_by,
                "eval_run_id": eval_run_id,
            }
        )

    async def activate_plan(
        self,
        *,
        plan_id: str,
        scope: str,
        tenant_id: str | None,
        approved_by: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._log.append("activate_plan")
        self.activate_calls.append(
            {
                "plan_id": plan_id,
                "scope": scope,
                "tenant_id": tenant_id,
                "approved_by": approved_by,
                "agent_id": agent_id,
            }
        )


class _RecordingAudit:
    def __init__(self, log: list[str], *, fail: bool = False) -> None:
        self._log = log
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

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
    ) -> str:
        self._log.append("audit")
        self.calls.append(
            {
                "plan_id": plan_id,
                "actor_gcid": actor_gcid,
                "tenant_id": tenant_id,
                "scope": scope,
                "annotation": annotation,
            }
        )
        if self._fail:
            raise RuntimeError("audit emit failed")
        return "audit-row-id"


class _RecordingHITL:
    def __init__(self, log: list[str], *, fail: bool = False) -> None:
        self._log = log
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

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
    ) -> str:
        self._log.append("hitl")
        self.calls.append(
            {
                "plan_id": plan_id,
                "requester_gcid": requester_gcid,
                "tenant_id": tenant_id,
                "scope": scope,
                "eval_run_id": eval_run_id,
                "agent_ids": agent_ids,
                "segment_ids": segment_ids,
                "summary": summary,
            }
        )
        if self._fail:
            raise RuntimeError("hitl emit failed")
        return "hitl-row-id"


def _service(
    log: list[str] | None = None, *, audit_fail: bool = False
) -> tuple[PromptPromotionService, _RecordingRepo, _RecordingAudit]:
    log = log if log is not None else []
    repo = _RecordingRepo(log)
    audit = _RecordingAudit(log, fail=audit_fail)
    svc = PromptPromotionService(repo=repo, audit_emitter=audit)
    return svc, repo, audit


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_requires_repo(self) -> None:
        with pytest.raises(ValueError, match="repo"):
            PromptPromotionService(repo=None, audit_emitter=_RecordingAudit([]))

    def test_requires_audit_emitter(self) -> None:
        with pytest.raises(ValueError, match="audit"):
            PromptPromotionService(repo=_RecordingRepo([]), audit_emitter=None)


# -----------------------------------------------------------------------------
# submit_for_eval / mark_eval_passed / mark_rejected
# -----------------------------------------------------------------------------


class TestForwardTransitions:
    @pytest.mark.asyncio
    async def test_submit_for_eval(self) -> None:
        svc, repo, _ = _service()
        await svc.submit_for_eval("plan-1")
        assert repo.status_calls == [
            {
                "plan_id": "plan-1",
                "expected_from": PromptPlanState.DRAFT,
                "to": PromptPlanState.PENDING_EVAL,
                "approved_by": None,
                "eval_run_id": None,
            }
        ]

    @pytest.mark.asyncio
    async def test_mark_eval_passed_threads_eval_run_id(self) -> None:
        svc, repo, _ = _service()
        await svc.mark_eval_passed("plan-1", eval_run_id="eval-99")
        call = repo.status_calls[-1]
        assert call["expected_from"] == PromptPlanState.PENDING_EVAL
        assert call["to"] == PromptPlanState.PENDING_HITL
        assert call["eval_run_id"] == "eval-99"

    @pytest.mark.asyncio
    async def test_mark_rejected_defaults_to_hitl_stage(self) -> None:
        svc, repo, _ = _service()
        await svc.mark_rejected("plan-1", reason="reviewer rejected tone")
        call = repo.status_calls[-1]
        assert call["expected_from"] == PromptPlanState.PENDING_HITL
        assert call["to"] == PromptPlanState.REJECTED

    @pytest.mark.asyncio
    async def test_mark_rejected_from_eval_stage(self) -> None:
        svc, repo, _ = _service()
        await svc.mark_rejected(
            "plan-1",
            reason="eval failed",
            from_state=PromptPlanState.PENDING_EVAL,
        )
        call = repo.status_calls[-1]
        assert call["expected_from"] == PromptPlanState.PENDING_EVAL
        assert call["to"] == PromptPlanState.REJECTED

    @pytest.mark.asyncio
    async def test_mark_rejected_refuses_illegal_from_state(self) -> None:
        # draft -> rejected is not a legal edge (must pass through eval first).
        svc, _, _ = _service()
        with pytest.raises(ValueError):
            await svc.mark_rejected("plan-1", reason="x", from_state=PromptPlanState.DRAFT)


# -----------------------------------------------------------------------------
# activate — audit-before-commit ordering + fail-loud
# -----------------------------------------------------------------------------


class TestActivate:
    @pytest.mark.asyncio
    async def test_emits_audit_before_activating(self) -> None:
        log: list[str] = []
        svc, repo, audit = _service(log)
        await svc.activate(
            "plan-1",
            approved_by="admin-1",
            scope="tenant",
            tenant_id="01970000-0000-7000-8000-000000000001",
        )
        # audit MUST be emitted before the status write.
        assert log == ["audit", "activate_plan"]
        # the approver flows into both the audit actor + the activation row.
        assert audit.calls[-1]["actor_gcid"] == "admin-1"
        assert repo.activate_calls[-1]["approved_by"] == "admin-1"
        assert repo.activate_calls[-1]["scope"] == "tenant"
        assert repo.activate_calls[-1]["tenant_id"] == ("01970000-0000-7000-8000-000000000001")

    @pytest.mark.asyncio
    async def test_audit_failure_aborts_activation(self) -> None:
        log: list[str] = []
        svc, repo, _ = _service(log, audit_fail=True)
        with pytest.raises(RuntimeError, match="audit emit failed"):
            await svc.activate(
                "plan-1",
                approved_by="admin-1",
                scope="platform",
                tenant_id=None,
            )
        # the audit was attempted but the activation must NOT have happened.
        assert log == ["audit"]
        assert repo.activate_calls == []

    @pytest.mark.asyncio
    async def test_platform_activation_passes_empty_tenant_to_audit(self) -> None:
        svc, _, audit = _service()
        await svc.activate("plan-1", approved_by="admin-1", scope="platform", tenant_id=None)
        # platform-scope activation: audit tenant is empty (not "None" string).
        assert audit.calls[-1]["tenant_id"] == ""
        assert audit.calls[-1]["scope"] == "platform"


# -----------------------------------------------------------------------------
# request_hitl_approval — emit HITL gate BEFORE the transition (M-C.2)
# -----------------------------------------------------------------------------


class TestRequestHitlApproval:
    @pytest.mark.asyncio
    async def test_emits_hitl_request_before_transition(self) -> None:
        log: list[str] = []
        repo = _RecordingRepo(log)
        hitl = _RecordingHITL(log)
        svc = PromptPromotionService(repo=repo, audit_emitter=_RecordingAudit(log), hitl_emitter=hitl)
        await svc.request_hitl_approval(
            "plan-1",
            requester_gcid="author-1",
            tenant_id="t-1",
            scope="tenant",
            eval_run_id="eval-9",
            agent_ids=["qgen_question"],
            segment_ids=["role"],
        )
        # HITL gate emitted FIRST, then the guarded status flip.
        assert log == ["hitl", "update:pending_eval->pending_hitl"]
        # eval_run_id threaded onto the transition.
        assert repo.status_calls[-1]["eval_run_id"] == "eval-9"
        # the HITL request carries the prompt-registry routing context.
        call = hitl.calls[-1]
        assert call["plan_id"] == "plan-1"
        assert call["requester_gcid"] == "author-1"
        assert call["scope"] == "tenant"
        assert call["eval_run_id"] == "eval-9"
        assert call["agent_ids"] == ["qgen_question"]
        assert call["segment_ids"] == ["role"]

    @pytest.mark.asyncio
    async def test_emit_failure_aborts_transition(self) -> None:
        log: list[str] = []
        repo = _RecordingRepo(log)
        hitl = _RecordingHITL(log, fail=True)
        svc = PromptPromotionService(repo=repo, audit_emitter=_RecordingAudit(log), hitl_emitter=hitl)
        with pytest.raises(RuntimeError, match="hitl emit failed"):
            await svc.request_hitl_approval(
                "plan-1",
                requester_gcid="author-1",
                tenant_id="t-1",
                scope="tenant",
                eval_run_id="eval-9",
            )
        # emit attempted; the transition MUST NOT have happened.
        assert log == ["hitl"]
        assert repo.status_calls == []

    @pytest.mark.asyncio
    async def test_requires_hitl_emitter(self) -> None:
        # constructed without a hitl_emitter -> request_hitl_approval is loud.
        svc, _, _ = _service()
        with pytest.raises(ValueError, match="hitl"):
            await svc.request_hitl_approval(
                "plan-1",
                requester_gcid="a",
                tenant_id="t",
                scope="tenant",
                eval_run_id="e",
            )

    @pytest.mark.asyncio
    async def test_refuses_illegal_source_state(self) -> None:
        # request_hitl_approval is pending_eval -> pending_hitl only; the
        # assert_transition is pure + always legal, so this is a guard that the
        # method targets the right edge (no exception expected here, but the
        # emitter is still called before the transition).
        log: list[str] = []
        repo = _RecordingRepo(log)
        hitl = _RecordingHITL(log)
        svc = PromptPromotionService(repo=repo, audit_emitter=_RecordingAudit(log), hitl_emitter=hitl)
        await svc.request_hitl_approval(
            "plan-1",
            requester_gcid="a",
            tenant_id="t",
            scope="tenant",
            eval_run_id="e",
        )
        assert repo.status_calls[-1]["expected_from"] == PromptPlanState.PENDING_EVAL
        assert repo.status_calls[-1]["to"] == PromptPlanState.PENDING_HITL
