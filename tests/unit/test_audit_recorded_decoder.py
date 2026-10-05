"""Unit tests for ``decode_audit_entry_recorded`` (ADR-197 M-C.2).

The HITL round-trip closes when the orchestrator's audit-recorded consumer can
decode the binary-proto ``chora.governance.v1.AuditEntryRecorded`` the O+
approver records back (the topic is Schema-Registry-bound, encoding=BINARY).

These tests round-trip the M-C.1 encoder against the new M-C.2 decoder so the
two stay byte-compatible — the decoder MUST read back every field the encoder
writes (the field numbers are pinned to chora-contracts/proto/events/governance/
audit.proto).
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    AUDIT_RESULT_ALLOWED,
    ProtoDecodeError,
    decode_audit_entry_recorded,
    encode_audit_entry_recorded,
)


def _envelope() -> dict[str, Any]:
    return {
        "event_id": "01970000-0000-7000-8000-0000000000ee",
        "idempotency_key": "audit.t-1.plan-1",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "gcid": "01970000-0000-7000-9000-000000000001",
        "occurred_at": "2026-06-28T12:00:00+00:00",
        "published_at": "2026-06-28T12:00:01+00:00",
        "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        "tracestate": "vendor=chora",
        "source_project": "chora-489812",
        "source_service": "chora-governance",
        "schema_version": "1",
        "chora_imda_dimension": "accountability",
    }


def _body(**over: Any) -> dict[str, Any]:
    body = {
        "audit_id": "01970000-0000-7000-a000-000000000abc",
        "actor_gcid": "01970000-0000-7000-9000-000000000001",
        "target_resource_uri": ("hitl_decision:prompt-override-plan:01970000-0000-7000-8000-0000000000aa"),
        "action": "decision_approve",
        "result": AUDIT_RESULT_ALLOWED,
        "annotation": "approved by reviewer",
        "occurred_at": "2026-06-28T12:00:00+00:00",
    }
    body.update(over)
    return body


class TestRoundTrip:
    def test_round_trips_scalar_fields(self) -> None:
        body = _body()
        out = decode_audit_entry_recorded(encode_audit_entry_recorded(_envelope(), body))
        assert out["audit_id"] == body["audit_id"]
        assert out["actor_gcid"] == body["actor_gcid"]
        assert out["target_resource_uri"] == body["target_resource_uri"]
        assert out["action"] == body["action"]
        assert out["result"] == AUDIT_RESULT_ALLOWED
        assert out["annotation"] == body["annotation"]

    def test_decodes_hitl_decision_uri_and_action(self) -> None:
        out = decode_audit_entry_recorded(encode_audit_entry_recorded(_envelope(), _body()))
        assert out["target_resource_uri"].startswith("hitl_decision:prompt-override-plan:")
        assert out["action"] == "decision_approve"

    def test_decodes_reject_action(self) -> None:
        out = decode_audit_entry_recorded(encode_audit_entry_recorded(_envelope(), _body(action="decision_reject")))
        assert out["action"] == "decision_reject"

    def test_decodes_envelope_tenant_gcid_and_trace(self) -> None:
        env = _envelope()
        out = decode_audit_entry_recorded(encode_audit_entry_recorded(env, _body()))
        # surfaced top-level for the consumer's dedup + handler scope
        assert out["tenant_id"] == env["tenant_id"]
        assert out["traceparent"] == env["traceparent"]
        assert out["tracestate"] == env["tracestate"]
        # nested envelope carries the dedup keys the consumer falls back to
        assert out["envelope"]["idempotency_key"] == env["idempotency_key"]
        assert out["envelope"]["event_id"] == env["event_id"]
        assert out["envelope"]["gcid"] == env["gcid"]

    def test_occurred_at_round_trips(self) -> None:
        out = decode_audit_entry_recorded(encode_audit_entry_recorded(_envelope(), _body()))
        assert out["occurred_at"].startswith("2026-06-28T12:00:00")

    def test_result_defaults_to_allowed(self) -> None:
        body = _body()
        del body["result"]
        out = decode_audit_entry_recorded(encode_audit_entry_recorded(_envelope(), body))
        # encoder defaults a missing result to ALLOWED(1)
        assert out["result"] == AUDIT_RESULT_ALLOWED

    def test_ignores_unknown_trailing_field(self) -> None:
        # Append an unknown varint field (field 15, wire 0) — forward-compat.
        # Tag = (15 << 3) | 0 = 120 (one byte); value 7.
        payload = bytearray(encode_audit_entry_recorded(_envelope(), _body()))
        payload += bytes([(15 << 3) | 0, 7])
        out = decode_audit_entry_recorded(bytes(payload))
        assert out["action"] == "decision_approve"

    def test_truncated_payload_raises(self) -> None:
        # A length-delimited field claiming more bytes than present.
        bad = bytes([(4 << 3) | 2]) + bytes([20]) + b"short"
        with pytest.raises(ProtoDecodeError):
            decode_audit_entry_recorded(bad)
