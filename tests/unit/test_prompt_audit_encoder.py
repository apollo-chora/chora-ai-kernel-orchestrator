"""Unit tests for ``encode_audit_entry_recorded`` (ADR-197 M-C.1).

The activation audit emitter publishes ``chora.governance.audit.recorded.v1``
with a binary-proto ``chora.governance.v1.AuditEntryRecorded`` payload (the
topic is Schema-Registry-bound, encoding=BINARY — JSON is rejected at the
publish hop, same as the agent_decision topic).

These tests decode the wire bytes with a small read-only walker (mirrors the
test_agent_decision_outbox_writer walker) to assert the canonical field mapping
per chora-contracts/proto/events/governance/audit.proto.
"""

from __future__ import annotations

from typing import Any

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    AUDIT_RESULT_ALLOWED,
    encode_audit_entry_recorded,
)

# -----------------------------------------------------------------------------
# Proto wire-format walker (read-only) — varint + length-delimited only.
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
        field, wire = tag >> 3, tag & 0x07
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        else:
            raise ValueError(f"unsupported wire type {wire} at field {field}")
        out.setdefault(field, []).append(value)
    return out


def _s(w: dict[int, list[Any]], field: int) -> str:
    return w[field][0].decode("utf-8") if field in w else ""


def _i(w: dict[int, list[Any]], field: int) -> int:
    return int(w[field][0]) if field in w else 0


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _envelope() -> dict[str, Any]:
    return {
        "event_id": "01970000-0000-7000-8000-0000000000ee",
        "idempotency_key": "prompt_activation.plan-1.admin-1",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "gcid": "01970000-0000-7000-9000-000000000001",
        "occurred_at": "2026-06-28T12:00:00+00:00",
        "published_at": "2026-06-28T12:00:01+00:00",
        "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        "tracestate": "vendor=chora",
        "source_project": "chora-489812",
        "source_service": "chora-ai-kernel-orchestrator",
        "schema_version": "1",
        "chora_imda_dimension": "accountability",
    }


def _body() -> dict[str, Any]:
    return {
        "audit_id": "01970000-0000-7000-a000-000000000abc",
        "actor_gcid": "01970000-0000-7000-9000-000000000001",
        "target_resource_uri": "chora.ai_kernel/prompt_plan:plan-1",
        "action": "ACTIVATE",
        "result": AUDIT_RESULT_ALLOWED,
        "annotation": "activated prompt override plan plan-1",
        "occurred_at": "2026-06-28T12:00:00+00:00",
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestEncodeAuditEntryRecorded:
    def test_allowed_result_constant_is_one(self) -> None:
        # AUDIT_RESULT_ALLOWED == 1 per audit.proto enum.
        assert AUDIT_RESULT_ALLOWED == 1

    def test_top_level_scalar_fields(self) -> None:
        payload = encode_audit_entry_recorded(_envelope(), _body())
        w = _walk(payload)
        assert _s(w, 2) == "01970000-0000-7000-a000-000000000abc"  # audit_id
        assert _s(w, 3) == "01970000-0000-7000-9000-000000000001"  # actor_gcid
        assert _s(w, 4) == "chora.ai_kernel/prompt_plan:plan-1"  # target_resource_uri
        assert _s(w, 5) == "ACTIVATE"  # action
        assert _i(w, 6) == 1  # result = AUDIT_RESULT_ALLOWED
        assert _s(w, 7) == "activated prompt override plan plan-1"  # annotation

    def test_embedded_envelope_at_field_1(self) -> None:
        payload = encode_audit_entry_recorded(_envelope(), _body())
        w = _walk(payload)
        assert 1 in w
        env = _walk(w[1][0])
        # EventEnvelope: tenant_id @ 3, gcid @ 4, source_project @ 9.
        assert _s(env, 3) == "01970000-0000-7000-8000-000000000001"
        assert _s(env, 4) == "01970000-0000-7000-9000-000000000001"
        assert _s(env, 9) == "chora-489812"
        assert _s(env, 10) == "chora-ai-kernel-orchestrator"
        # occurred_at (Timestamp submessage) present @ envelope field 5.
        assert 5 in env

    def test_occurred_at_timestamp_at_field_10(self) -> None:
        payload = encode_audit_entry_recorded(_envelope(), _body())
        w = _walk(payload)
        # field 10 is a length-delimited Timestamp submessage.
        assert 10 in w
        ts = _walk(w[10][0])
        # seconds @ field 1 — non-zero for a 2026 timestamp.
        assert _i(ts, 1) > 0

    def test_result_defaults_to_allowed_when_omitted(self) -> None:
        # If the caller omits result, the encoder MUST still emit ALLOWED (1)
        # — a blank/zero result would be AUDIT_RESULT_UNSPECIFIED, which the
        # governance projector treats as a data-integrity error.
        body = _body()
        del body["result"]
        payload = encode_audit_entry_recorded(_envelope(), body)
        w = _walk(payload)
        assert _i(w, 6) == 1

    def test_empty_optional_strings_omitted(self) -> None:
        # source_ip (8) + user_agent (9) are not set for a system-emitted
        # activation → proto3 default-omit.
        payload = encode_audit_entry_recorded(_envelope(), _body())
        w = _walk(payload)
        assert 8 not in w
        assert 9 not in w
