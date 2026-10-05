"""Unit tests for ``PromptActivationAuditEmitter`` (ADR-197 M-C.1).

Mirrors ``AgentDecisionLogOutboxWriter`` — INSERTs a transactional-outbox row
into ``ai_kernel_outbox_events`` on topic ``chora.governance.audit.recorded.v1``
with a binary-proto ``AuditEntryRecorded`` payload. The OutboxDispatcher drains
+ publishes (the emitter does NOT publish directly).

Asserts: the canonical topic, the AuditEntryRecorded field mapping (ACTIVATE /
ALLOWED / the prompt_plan resource URI), the deterministic idempotency key, the
11 mandatory envelope fields + the D1 dimension, and loud error propagation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.prompt_activation_audit_emitter import (
    IMDA_DIM_ACCOUNTABILITY,
    PLATFORM_TENANT_ID,
    TOPIC_AUDIT_RECORDED,
    PromptActivationAuditEmitter,
)

# -----------------------------------------------------------------------------
# Fake async connection / cursor
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

    def cursor(self) -> _FakeCursor:
        return self.cur


def _row(conn: _FakeConn) -> dict[str, Any]:
    sql, params = conn.cur.executed[0]
    assert "INSERT INTO ai_kernel_outbox_events" in sql
    assert "ON CONFLICT (idempotency_key) DO NOTHING" in sql
    return params


# -----------------------------------------------------------------------------
# Proto walker (read-only)
# -----------------------------------------------------------------------------


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    result, shift = 0, 0
    while offset < len(data):
        b = data[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7
    raise ValueError("truncated varint")


def _walk(data: bytes) -> dict[int, list[Any]]:
    out: dict[int, list[Any]] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        f, wire = tag >> 3, tag & 0x07
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        else:
            raise ValueError(f"unsupported wire {wire}")
        out.setdefault(f, []).append(value)
    return out


def _s(w: dict[int, list[Any]], f: int) -> str:
    return w[f][0].decode("utf-8") if f in w else ""


def _i(w: dict[int, list[Any]], f: int) -> int:
    return int(w[f][0]) if f in w else 0


async def _emit(conn: _FakeConn, **over: Any) -> str:
    emitter = PromptActivationAuditEmitter(conn=conn, source_project="chora-489812")
    kwargs: dict[str, Any] = {
        "plan_id": "01970000-0000-7000-8000-0000000000aa",
        "actor_gcid": "01970000-0000-7000-9000-000000000001",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "scope": "tenant",
        "annotation": "activated plan",
    }
    kwargs.update(over)
    return await emitter.emit_activation(**kwargs)


# -----------------------------------------------------------------------------
# Construction + topic constant
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_requires_source_project(self) -> None:
        with pytest.raises(ValueError, match="source_project"):
            PromptActivationAuditEmitter(conn=_FakeConn(), source_project="")

    def test_topic_is_governance_audit_recorded(self) -> None:
        assert TOPIC_AUDIT_RECORDED == "chora.governance.audit.recorded.v1"


# -----------------------------------------------------------------------------
# emit_activation — row + payload + envelope
# -----------------------------------------------------------------------------


class TestEmitActivation:
    @pytest.mark.asyncio
    async def test_writes_outbox_row(self) -> None:
        conn = _FakeConn()
        row_id = await _emit(conn)
        assert UUID(row_id)  # row id is a real uuid
        params = _row(conn)
        assert params["topic"] == TOPIC_AUDIT_RECORDED
        assert params["event_type"] == "governance.audit.recorded"
        assert params["workflow_id"] == "01970000-0000-7000-8000-0000000000aa"
        assert params["tenant_id"] == "01970000-0000-7000-8000-000000000001"
        # the acting admin GCID is stamped as the row gcid.
        assert params["gcid"] == "01970000-0000-7000-9000-000000000001"

    @pytest.mark.asyncio
    async def test_payload_is_audit_entry_recorded_proto(self) -> None:
        conn = _FakeConn()
        await _emit(conn)
        w = _walk(_row(conn)["payload"])
        # audit_id (field 2) is a UUIDv7.
        assert UUID(_s(w, 2)).version == 7
        assert _s(w, 3) == "01970000-0000-7000-9000-000000000001"  # actor_gcid
        assert _s(w, 4) == "chora.ai_kernel/prompt_plan:01970000-0000-7000-8000-0000000000aa"
        assert _s(w, 5) == "ACTIVATE"  # action
        assert _i(w, 6) == 1  # result = AUDIT_RESULT_ALLOWED
        assert _s(w, 7) == "activated plan"  # annotation
        assert 10 in w  # occurred_at Timestamp present

    @pytest.mark.asyncio
    async def test_proto_envelope_carries_tenant_and_actor(self) -> None:
        conn = _FakeConn()
        await _emit(conn)
        w = _walk(_row(conn)["payload"])
        env = _walk(w[1][0])  # embedded EventEnvelope at field 1
        assert _s(env, 3) == "01970000-0000-7000-8000-000000000001"  # tenant_id
        assert _s(env, 4) == "01970000-0000-7000-9000-000000000001"  # gcid

    @pytest.mark.asyncio
    async def test_idempotency_key_is_deterministic(self) -> None:
        conn = _FakeConn()
        await _emit(conn)
        params = _row(conn)
        assert params["idempotency_key"] == (
            "prompt_activation.01970000-0000-7000-8000-0000000000aa.01970000-0000-7000-9000-000000000001"
        )

    @pytest.mark.asyncio
    async def test_envelope_has_mandatory_fields_and_dimension(self) -> None:
        conn = _FakeConn()
        await _emit(
            conn,
            traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
            tracestate="vendor=chora",
        )
        env = json.loads(_row(conn)["envelope"])
        for required in (
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
            "chora_imda_dimension",
        ):
            assert required in env, f"envelope missing {required}"
        assert env["chora_imda_dimension"] == IMDA_DIM_ACCOUNTABILITY
        assert env["source_project"] == "chora-489812"
        assert env["traceparent"] == ("00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01")

    @pytest.mark.asyncio
    async def test_platform_scope_falls_back_to_the_platform_tenant(self) -> None:
        # CHO-2368 P2 (first live walk): a platform-scope plan carries NO
        # tenant, but ai_kernel_outbox_events.tenant_id is UUID NOT NULL, so
        # emitting "" raised 22P02 and poisoned the consumer's transaction —
        # every approval NACKed and the activation never happened. This test
        # previously asserted the empty string: a fake connection has no column
        # typing, so the unit test was green while the real INSERT could not
        # work. The emitter now falls back to the platform tenant.
        conn = _FakeConn()
        await _emit(conn, scope="platform", tenant_id="")
        params = _row(conn)
        assert params["tenant_id"] == PLATFORM_TENANT_ID
        assert params["idempotency_key"].startswith("prompt_activation.")

    @pytest.mark.asyncio
    async def test_platform_tenant_fallback_is_overridable(self) -> None:
        conn = _FakeConn()
        emitter = PromptActivationAuditEmitter(
            conn=conn,
            source_project="chora-489812",
            platform_tenant_id="01970000-0000-7000-8000-0000000000ff",
        )
        await emitter.emit_activation(
            plan_id="019fa2d7-5d8a-7e00-8651-37d54873c40a",
            actor_gcid="00000000-0000-7000-8000-000000001999",
            tenant_id="",
            scope="platform",
        )
        assert _row(conn)["tenant_id"] == "01970000-0000-7000-8000-0000000000ff"

    @pytest.mark.asyncio
    async def test_blank_actor_raises(self) -> None:
        conn = _FakeConn()
        with pytest.raises(ValueError, match="actor_gcid"):
            await _emit(conn, actor_gcid="")

    @pytest.mark.asyncio
    async def test_blank_plan_id_raises(self) -> None:
        conn = _FakeConn()
        with pytest.raises(ValueError, match="plan_id"):
            await _emit(conn, plan_id="  ")


# -----------------------------------------------------------------------------
# Error propagation
# -----------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.asyncio
    async def test_db_error_propagates(self) -> None:
        @dataclass
        class _FailingCursor:
            async def execute(self, sql: str, params: dict[str, Any]) -> None:
                raise RuntimeError("DB down")

            async def __aenter__(self) -> _FailingCursor:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

        @dataclass
        class _FailingConn:
            def cursor(self) -> _FailingCursor:
                return _FailingCursor()

        emitter = PromptActivationAuditEmitter(conn=_FailingConn(), source_project="chora-489812")
        with pytest.raises(RuntimeError, match="DB down"):
            await emitter.emit_activation(
                plan_id="p",
                actor_gcid="g",
                tenant_id="t",
                scope="tenant",
            )
