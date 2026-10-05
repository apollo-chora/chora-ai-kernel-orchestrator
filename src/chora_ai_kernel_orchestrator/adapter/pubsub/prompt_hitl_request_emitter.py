"""PromptHITLRequestEmitter — adapter implementing the domain
``HITLRequestEmitter`` port (ADR-197 M-C.2) for the prompt-override promotion
gate.

The producer half of the HITL round-trip: when a prompt-override plan passes the
automated eval gate, the ``PromptPromotionService.request_hitl_approval`` path
calls this emitter to enqueue a human-oversight gate on
``chora.governance.hitl.requested.v1`` so the O+ Human-Oversight queue carries
the plan for a reviewer to approve / reject.

It does NOT re-implement the outbox machinery — it delegates to the generic
:class:`HITLDecisionOutboxWriter` (same topic, same transactional outbox, same
deterministic idempotency key), mapping the prompt-registry-specific fields onto
the generic gate shape:

    decision_id    = "prompt-override-plan:{plan_id}"   (the approval comes back
                     as target_resource_uri "hitl_decision:{decision_id}")
    run_id         = plan_id   (the gate's aggregate root; it lands in the
                     UUID-typed outbox workflow_id, so it cannot carry the
                     human-readable eval_run_id — that rides edit_payload)
    gcid           = requester_gcid
    agent_id       = "prompt-registry"
    autonomy_level = "hitl_l0"   (Level-0 — a human MUST approve before activate)
    edit_payload   = {plan_id, scope, agent_ids, segment_ids, eval_run_id}

Connection / commit semantics are the writer's (caller-owned conn, no commit
here) — see :class:`HITLDecisionOutboxWriter`.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

logger = logging.getLogger(__name__)

# The aggregate that authored the gate — surfaces in the O+ queue + groups the
# gate under the Crews+Agents hierarchy.
PROMPT_REGISTRY_AGENT_ID = "prompt-registry"

# HITL Level-0 — a human approval is REQUIRED before the plan activates (no
# auto-promotion). Mirrors the autonomy-level vocabulary the O+ queue renders.
PROMPT_HITL_AUTONOMY_LEVEL = "hitl_l0"

# decision_id encoding — the O+ approver records back
# target_resource_uri = "hitl_decision:{decision_id}", so the M-C.2 consumer
# strips "hitl_decision:prompt-override-plan:" to recover the plan_id.
DECISION_ID_PREFIX = "prompt-override-plan:"

# Crew label for /o/agents grouping (mirrors the agent_id; the gate is not part
# of a multi-agent crew, so the single label is sufficient).
_CREW_NAME = "prompt-registry"


class PromptHITLRequestEmitter:
    """Maps a prompt-override promotion gate onto the generic HITL outbox writer.

    Implements the ``HITLRequestEmitter`` Protocol the
    ``PromptPromotionService`` depends on.
    """

    def __init__(self, *, writer: Any, crew_name: str = _CREW_NAME) -> None:
        if writer is None:
            raise ValueError("PromptHITLRequestEmitter requires a writer")
        self._writer = writer
        self._crew_name = crew_name

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
        """Emit ``chora.governance.hitl.requested.v1`` for ``plan_id``.

        Returns the outbox row id the writer minted. Raises ``ValueError`` on a
        blank plan_id (fail-loud — a gate with no plan is unactionable).
        """
        plan = (plan_id or "").strip()
        if not plan:
            raise ValueError("emit_hitl_request requires a plan_id")

        decision_id = f"{DECISION_ID_PREFIX}{plan}"
        occurred_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
        edit_payload: dict[str, Any] = {
            "plan_id": plan,
            "scope": scope,
            "agent_ids": list(agent_ids or []),
            "segment_ids": list(segment_ids or []),
            "eval_run_id": eval_run_id,
        }
        gate_summary = summary or (f"prompt override plan {plan} passed eval — awaiting human sign-off")

        row_id = str(
            await self._writer.emit(
                decision_id=decision_id,
                # The PLAN id, not the eval run id (CHO-2368 P2): this lands in
                # ai_kernel_outbox_events.workflow_id, which is UUID-typed,
                # while an ADR-174 eval run id is a human-readable candidate
                # label. The plan is this gate's aggregate root anyway; the
                # eval run rides edit_payload + the operator-facing summary.
                run_id=plan,
                tenant_id=tenant_id,
                gcid=requester_gcid,
                agent_id=PROMPT_REGISTRY_AGENT_ID,
                autonomy_level=PROMPT_HITL_AUTONOMY_LEVEL,
                summary=gate_summary,
                occurred_at=occurred_at,
                crew_name=self._crew_name,
                edit_payload=edit_payload,
                traceparent=traceparent,
                tracestate=tracestate,
            )
        )
        logger.info(
            "prompt_hitl_request.queued",
            extra={
                "plan_id": plan,
                "decision_id": decision_id,
                "eval_run_id": eval_run_id,
                "scope": scope,
                "row_id": row_id,
            },
        )
        return row_id


__all__ = [
    "DECISION_ID_PREFIX",
    "PROMPT_HITL_AUTONOMY_LEVEL",
    "PROMPT_REGISTRY_AGENT_ID",
    "PromptHITLRequestEmitter",
]
