"""Unit tests for ``PromptHITLRequestEmitter`` (ADR-197 M-C.2).

The emitter is the producer half of the HITL round-trip: when a prompt-override
plan passes the automated eval gate, it emits ``chora.governance.hitl.requested.
v1`` so the O+ Human-Oversight queue carries the gate for a human to approve /
reject. It reuses the generic ``HITLDecisionOutboxWriter`` (same outbox topic +
idempotency machinery) and maps the prompt-registry fields onto it.

Asserts: the canonical topic, the decision_id encoding (prompt-override-plan:
{plan_id}), the prompt-registry agent_id + hitl_l0 autonomy, and the
edit_payload carrying the plan's scope/agents/segments/eval_run_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.hitl_decision_outbox_writer import (
    TOPIC_HITL_REQUESTED,
    HITLDecisionOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.prompt_hitl_request_emitter import (
    PROMPT_HITL_AUTONOMY_LEVEL,
    PROMPT_REGISTRY_AGENT_ID,
    PromptHITLRequestEmitter,
)

_PLAN = "01970000-0000-7000-8000-0000000000aa"
_REQ = "01970000-0000-7000-9000-000000000001"
_TENANT = "01970000-0000-7000-8000-000000000001"


# -----------------------------------------------------------------------------
# Fake async connection / cursor (mirrors the audit-emitter test)
# -----------------------------------------------------------------------------


@dataclass
class _FakeCursor:
    executed: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def execute(self, sql: str, params: dict[str, Any]) -> None:
        self.executed.append((sql, params))

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


@dataclass
class _FakeConn:
    cur: _FakeCursor = field(default_factory=_FakeCursor)
    commits: list[int] = field(default_factory=list)

    def cursor(self) -> _FakeCursor:
        return self.cur

    async def commit(self) -> None:
        # The writer commits its OWN INSERT - the shared connection is drained
        # by a concurrent dispatcher whose transaction release would otherwise
        # discard an uncommitted row (live loss 2026-08-14, job fd80f4e9).
        self.commits.append(len(self.cur.executed))

    async def rollback(self) -> None:  # pragma: no cover - never taken here
        raise AssertionError("outbox writer must never roll back its own row")


def _row(conn: _FakeConn) -> dict[str, Any]:
    sql, params = conn.cur.executed[0]
    assert "INSERT INTO ai_kernel_outbox_events" in sql
    return params


async def _emit(conn: _FakeConn, **over: Any) -> str:
    writer = HITLDecisionOutboxWriter(conn=conn, source_project="chora-489812")
    emitter = PromptHITLRequestEmitter(writer=writer)
    kwargs: dict[str, Any] = {
        "plan_id": _PLAN,
        "requester_gcid": _REQ,
        "tenant_id": _TENANT,
        "scope": "tenant",
        "eval_run_id": "eval-run-9",
        "agent_ids": ["qgen_question"],
        "segment_ids": ["role", "task"],
    }
    kwargs.update(over)
    return await emitter.emit_hitl_request(**kwargs)


class TestConstruction:
    def test_requires_writer(self) -> None:
        with pytest.raises(ValueError, match="writer"):
            PromptHITLRequestEmitter(writer=None)


class TestEmit:
    @pytest.mark.asyncio
    async def test_writes_hitl_requested_outbox_row(self) -> None:
        conn = _FakeConn()
        row_id = await _emit(conn)
        assert UUID(row_id)  # writer returns a real uuid row id
        params = _row(conn)
        assert params["topic"] == TOPIC_HITL_REQUESTED
        # the requester is the row gcid; the tenant flows through
        assert params["gcid"] == _REQ
        assert params["tenant_id"] == _TENANT

    @pytest.mark.asyncio
    async def test_body_carries_prompt_registry_routing_fields(self) -> None:
        conn = _FakeConn()
        await _emit(conn)
        body = json.loads(_row(conn)["payload"])
        assert body["decision_id"] == f"prompt-override-plan:{_PLAN}"
        # CHO-2368 P2 (first live walk): run_id must be the PLAN id, not the
        # eval_run_id. It lands in ai_kernel_outbox_events.workflow_id, which
        # is UUID-typed, and an ADR-174 eval run id is a human-readable
        # candidate label (e.g. "cho2368-qgen-question-prompt-1-1-0-r1") that
        # raises 22P02 on INSERT. The plan IS this gate's aggregate root; the
        # eval run stays in edit_payload + the summary.
        assert body["run_id"] == _PLAN
        assert body["agent_id"] == PROMPT_REGISTRY_AGENT_ID
        assert body["autonomy_level"] == PROMPT_HITL_AUTONOMY_LEVEL

    @pytest.mark.asyncio
    async def test_edit_payload_carries_plan_context(self) -> None:
        conn = _FakeConn()
        await _emit(conn)
        body = json.loads(_row(conn)["payload"])
        edit = body["edit_payload"]
        assert edit["plan_id"] == _PLAN
        assert edit["scope"] == "tenant"
        assert edit["agent_ids"] == ["qgen_question"]
        assert edit["segment_ids"] == ["role", "task"]
        assert edit["eval_run_id"] == "eval-run-9"

    @pytest.mark.asyncio
    async def test_summary_defaulted_when_blank(self) -> None:
        conn = _FakeConn()
        await _emit(conn, summary="")
        body = json.loads(_row(conn)["payload"])
        assert body["summary"]  # non-empty default surfaced

    @pytest.mark.asyncio
    async def test_decision_id_prefix_constant(self) -> None:
        conn = _FakeConn()
        await _emit(conn, plan_id="plan-xyz")
        body = json.loads(_row(conn)["payload"])
        assert body["decision_id"] == "prompt-override-plan:plan-xyz"

    @pytest.mark.asyncio
    async def test_blank_plan_id_raises(self) -> None:
        conn = _FakeConn()
        with pytest.raises(ValueError, match="plan_id"):
            await _emit(conn, plan_id="  ")
