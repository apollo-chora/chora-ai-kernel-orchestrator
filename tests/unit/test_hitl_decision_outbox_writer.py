"""RED→GREEN tests for HITLDecisionOutboxWriter (Human-Oversight gate emit).

The writer satisfies the ``_HITLDecisionEmitter`` Protocol from
``orchestrators/qgen_crew_runner.py`` by INSERTing rows into
``ai_kernel_outbox_events`` (migration 0003_outbox.sql) with topic
``chora.governance.hitl.requested.v1``. The actual Pub/Sub publish hop is
handled by the existing ``OutboxDispatcher`` +
``GoogleCloudPubSubPublisher`` (separate adapter; this writer does NOT
publish directly).

Unlike the AgentDecisionLog writer (whose topic is Schema-Registry-bound
BINARY proto per ADR-167), the HITL gate topic is NOT yet
Schema-Registry-bound, so the body is JSON shaped EXACTLY as the
chora-governance projector's ``projector.IncomingEvent`` D4 routing
contract:

    imda_dimension = "fairness_and_human_oversight"
    event_type contains "hitl"
    decision_id / run_id / agent_id / operator_gcid / hitl_verdict /
    autonomy_level / edit_payload + envelope event_id/tenant_id/traceparent

so the governance ``routeD4`` → ``AppendHITLDecision`` + gateway
``mapHITLItem`` can render a pending gate once the C1 RLS-context fix +
the governance HITL-event subscriber land (FLAGGED — out of scope here).

D6 4-pillar contract (mirrors test_agent_decision_outbox_writer.py):
  * Pillar 1 (pod-death survival) — atomic with terminal transaction
    (shares the same psycopg AsyncConnection as the qgen crew writer).
  * Pillar 2 (delivery resilience) — OutboxDispatcher drains; DLQ on
    persistent failure.
  * Pillar 3 (multi-tenant isolation) — tenant_id stamped in envelope
    + payload.
  * Pillar 4 (Cloud Trace attribution) — traceparent + tracestate in
    envelope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.hitl_decision_outbox_writer import (
    HITL_VERDICT_PENDING,
    IMDA_DIM_FAIRNESS_HUMAN_OVERSIGHT,
    SCHEMA_VERSION,
    TOPIC_HITL_REQUESTED,
    HITLDecisionOutboxWriter,
)

# -----------------------------------------------------------------------------
# Fake AsyncConnection / Cursor — mirrors psycopg's AsyncConnection surface
# -----------------------------------------------------------------------------


@dataclass
class _FakeCursor:
    executed: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def execute(self, sql: str, params: dict[str, Any]) -> None:
        self.executed.append((sql, params))

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


@dataclass
class _FakeAsyncConnection:
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


def _decode_row(executed: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    sql, params = executed
    assert "INSERT INTO ai_kernel_outbox_events" in sql
    assert "ON CONFLICT (idempotency_key) DO NOTHING" in sql
    return params


async def _emit_one(
    writer: HITLDecisionOutboxWriter,
    **overrides: Any,
) -> dict[str, Any]:
    """Emit one HITL gate event + return the decoded outbox-row params."""
    kwargs: dict[str, Any] = {
        "decision_id": "01975c83-0000-7000-8000-0000000000aa",
        "run_id": "01975c83-0000-7000-8000-000000000001",
        "tenant_id": "00000000-0000-7000-8000-000000000001",
        "gcid": "00000000-0000-7000-8000-000000000002",
        "agent_id": "qgen_crew",
        "crew_name": "mcq_ai_assist",
        "autonomy_level": "hitl_l0",
        "summary": "critic rejected on attempt 4 (exhausted max_retries=3)",
        "occurred_at": "2026-06-02T10:00:00+00:00",
        "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "tracestate": "chora=on",
    }
    kwargs.update(overrides)
    await writer.emit(**kwargs)
    return _decode_row(writer._conn.cur.executed[-1])  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_requires_source_project(self) -> None:
        with pytest.raises(ValueError, match="source_project"):
            HITLDecisionOutboxWriter(conn=_FakeAsyncConnection(), source_project="")

    def test_requires_source_service(self) -> None:
        with pytest.raises(ValueError, match="source_service"):
            HITLDecisionOutboxWriter(
                conn=_FakeAsyncConnection(),
                source_project="chora-489812",
                source_service="",
            )

    def test_captures_project_and_service(self) -> None:
        w = HITLDecisionOutboxWriter(
            conn=_FakeAsyncConnection(),
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        assert w.source_project == "chora-489812"
        assert w.source_service == "chora-ai-kernel-orchestrator"


# -----------------------------------------------------------------------------
# Topic + dimension constants — locked to the D4 routing contract
# -----------------------------------------------------------------------------


def test_topic_constant_is_governance_hitl_requested() -> None:
    assert TOPIC_HITL_REQUESTED == "chora.governance.hitl.requested.v1"


def test_dimension_constant_is_d4_fairness_human_oversight() -> None:
    assert IMDA_DIM_FAIRNESS_HUMAN_OVERSIGHT == "fairness_and_human_oversight"


# -----------------------------------------------------------------------------
# Emit — outbox row shape
# -----------------------------------------------------------------------------


class TestEmit:
    @pytest.mark.asyncio
    async def test_emit_writes_pending_row_on_hitl_topic(self) -> None:
        w = HITLDecisionOutboxWriter(conn=_FakeAsyncConnection(), source_project="chora-489812")
        params = await _emit_one(w)
        sql = w._conn.cur.executed[-1][0]  # type: ignore[attr-defined]
        # status='pending' is a SQL literal in the INSERT (mirrors the
        # agent_decision writer) — assert it lands as pending.
        assert "'pending'" in sql
        assert params["topic"] == TOPIC_HITL_REQUESTED
        assert params["tenant_id"] == "00000000-0000-7000-8000-000000000001"
        assert params["workflow_id"] == "01975c83-0000-7000-8000-000000000001"

    @pytest.mark.asyncio
    async def test_emit_envelope_carries_d4_dimension_and_hitl_event_type(self) -> None:
        w = HITLDecisionOutboxWriter(conn=_FakeAsyncConnection(), source_project="chora-489812")
        params = await _emit_one(w)
        envelope = json.loads(params["envelope"])
        assert envelope["chora_imda_dimension"] == "fairness_and_human_oversight"
        # event_type MUST contain "hitl" so the governance projector routeD4
        # switch routes to AppendHITLDecision.
        assert "hitl" in envelope["event_type"]
        # 11 mandatory CLAUDE.md §6 envelope fields present.
        for key in (
            "event_id",
            "idempotency_key",
            "tenant_id",
            "gcid",
            "occurred_at",
            "published_at",
            "traceparent",
            "tracestate",
            "source_project",
            "source_service",
            "schema_version",
        ):
            assert key in envelope, f"missing envelope key {key}"
        assert envelope["schema_version"] == SCHEMA_VERSION
        assert envelope["traceparent"].startswith("00-")

    @pytest.mark.asyncio
    async def test_emit_body_matches_projector_incoming_event_d4_shape(self) -> None:
        """The JSON body MUST be the governance projector.IncomingEvent D4
        routing shape so routeD4 → AppendHITLDecision + gateway mapHITLItem
        can render a pending gate verbatim."""
        w = HITLDecisionOutboxWriter(conn=_FakeAsyncConnection(), source_project="chora-489812")
        params = await _emit_one(w)
        body = json.loads(params["payload"])
        assert body["imda_dimension"] == "fairness_and_human_oversight"
        assert "hitl" in body["event_type"]
        assert body["decision_id"] == "01975c83-0000-7000-8000-0000000000aa"
        assert body["run_id"] == "01975c83-0000-7000-8000-000000000001"
        assert body["agent_id"] == "qgen_crew"
        assert body["autonomy_level"] == "hitl_l0"
        # A *pending* gate carries the pending verdict sentinel + no operator.
        assert body["hitl_verdict"] == HITL_VERDICT_PENDING
        assert body["operator_gcid"] == ""
        # Summary/reason — mapHITLItem prefers summary, falls back to reason.
        assert "max_retries" in body["summary"]
        # event_id present so the synchronous evidenceProject guard passes.
        assert body["event_id"]
        # crew_name forwarded so /o/agents can group the gate.
        assert body.get("crew_name") == "mcq_ai_assist"

    @pytest.mark.asyncio
    async def test_emit_is_idempotent_per_decision(self) -> None:
        """Two emits for the same decision_id produce the same
        idempotency_key so a graph re-run from checkpoint does not create a
        duplicate gate (ON CONFLICT DO NOTHING swallows it)."""
        w = HITLDecisionOutboxWriter(conn=_FakeAsyncConnection(), source_project="chora-489812")
        p1 = await _emit_one(w)
        p2 = await _emit_one(w)
        assert p1["idempotency_key"] == p2["idempotency_key"]
        assert "hitl" in p1["idempotency_key"]

    @pytest.mark.asyncio
    async def test_emit_returns_row_id(self) -> None:
        w = HITLDecisionOutboxWriter(conn=_FakeAsyncConnection(), source_project="chora-489812")
        row_id = await w.emit(
            decision_id="d1",
            run_id="r1",
            tenant_id="t1",
            gcid="g1",
            agent_id="qgen_crew",
            autonomy_level="hitl_l0",
            summary="needs review",
            occurred_at="2026-06-02T10:00:00+00:00",
        )
        assert isinstance(row_id, str)
        assert row_id

    @pytest.mark.asyncio
    async def test_emit_with_malformed_occurred_at_still_lands_row(self) -> None:
        """D6 resilience: a malformed occurred_at falls back to publish time
        for the SQL column so the gate row still lands (the body keeps the
        original value for forensic continuity)."""
        w = HITLDecisionOutboxWriter(conn=_FakeAsyncConnection(), source_project="chora-489812")
        params = await _emit_one(w, occurred_at="not-a-timestamp")
        # The TIMESTAMPTZ column got a real datetime (fallback), so the INSERT
        # is well-formed.
        import datetime as _dt

        assert isinstance(params["occurred_at"], _dt.datetime)
        # The body preserves the original (malformed) created_at value.
        body = json.loads(params["payload"])
        assert body["created_at"] == "not-a-timestamp"
