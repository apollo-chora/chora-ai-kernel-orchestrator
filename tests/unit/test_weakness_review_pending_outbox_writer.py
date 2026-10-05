"""RED→GREEN tests for WeaknessReviewPendingOutboxWriter (CHO-1973 Wave A).

The BINARY transactional-outbox emitter for
``chora.consumption.weakness.review_pending.v1`` — INSERTs one row into
``ai_kernel_outbox_events`` whose ``payload`` is canonical proto3 wire bytes from
``review_pending_proto_encoder`` (the topic carries a Schema-Registry BINARY
schema; a JSON publish dead-letters with INVALID_BINARY_PROTO_MESSAGE). The
existing ``OutboxDispatcher`` drains it.

Mirrors ``test_weakness_analyzed_outbox_writer.py`` — a local proto3 wire walker
+ fake AsyncConnection/Cursor; NO live DB, NO Pub/Sub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_review_pending_outbox_writer import (
    IMDA_DIM_OVERSIGHT,
    SCHEMA_VERSION,
    TOPIC_WEAKNESS_REVIEW_PENDING,
    WeaknessReviewPendingOutboxWriter,
)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while offset < len(data):
        b = data[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7
    raise ValueError("truncated varint")


def _walk_proto(data: bytes) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field_no = tag >> 3
        wire = tag & 0x07
        if wire == 0:
            value, offset = _read_varint(data, offset)
            out.setdefault(field_no, []).append({"wire": 0, "value": value})
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            out.setdefault(field_no, []).append({"wire": 2, "value": data[offset : offset + length]})
            offset += length
        elif wire == 5:
            out.setdefault(field_no, []).append({"wire": 5, "value": data[offset : offset + 4]})
            offset += 4
        else:
            raise ValueError(f"unsupported wire {wire} at field {field_no}")
    return out


def _first_str(fields: dict[int, list[dict[str, Any]]], field_no: int) -> str:
    return fields[field_no][0]["value"].decode("utf-8")


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

    def cursor(self) -> _FakeCursor:
        return self.cur


def _panel(**overrides: Any) -> dict[str, Any]:
    panel: dict[str, Any] = {
        "upload_id": "0190aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee",
        "tenant_id": "tenant-9",
        "learner_gcid": "learner-1",
        "familiar": {"familiar_id": "fam-7", "name": "Ember", "species": "dragon"},
        "proposed_edges": [
            {
                "proposed_edge_id": "pe-0",
                "concept_label": "adding fractions",
                "summary": "mixes denominators",
                "suggested_angles": ["area model"],
                "strength": 0.85,
                "suggested_difficulty": "harder",
            }
        ],
        "candidate_struggles": [],
        "available_outputs": [
            {"kind": "focused_dose", "mana_price": 0, "default_selected": True},
            {"kind": "practice_test", "mana_price": 120, "default_selected": False},
        ],
    }
    panel.update(overrides)
    return panel


class TestConstruction:
    def test_requires_source_project(self) -> None:
        with pytest.raises(ValueError, match="source_project"):
            WeaknessReviewPendingOutboxWriter(conn=_FakeAsyncConnection(), source_project="")

    def test_requires_source_service(self) -> None:
        with pytest.raises(ValueError, match="source_service"):
            WeaknessReviewPendingOutboxWriter(conn=_FakeAsyncConnection(), source_project="p", source_service="")


class TestPublish:
    async def test_inserts_single_pending_row_to_binary_topic(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessReviewPendingOutboxWriter(conn=conn, source_project="chora-489812")

        row_id = await writer.publish_review_pending(panel=_panel())

        assert isinstance(row_id, str) and row_id
        assert len(conn.cur.executed) == 1
        _sql, params = conn.cur.executed[0]
        assert params["topic"] == TOPIC_WEAKNESS_REVIEW_PENDING
        assert params["tenant_id"] == "tenant-9"
        assert params["gcid"] == "learner-1"
        assert params["workflow_id"] == _panel()["upload_id"]
        assert params["event_type"] == "consumption.weakness.review_pending"
        assert isinstance(params["payload"], (bytes, bytearray))

    async def test_idempotency_key_uses_upload_id(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessReviewPendingOutboxWriter(conn=conn, source_project="chora-489812")

        await writer.publish_review_pending(panel=_panel())

        _sql, params = conn.cur.executed[0]
        assert params["idempotency_key"] == f"weakness.review_pending.{_panel()['upload_id']}"

    async def test_envelope_carries_event_topic_and_oversight_dimension(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessReviewPendingOutboxWriter(conn=conn, source_project="chora-489812")

        await writer.publish_review_pending(
            panel=_panel(),
            traceparent="00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            tracestate="rojo=1",
        )

        _sql, params = conn.cur.executed[0]
        env = json.loads(params["envelope"])
        assert env["event_topic"] == TOPIC_WEAKNESS_REVIEW_PENDING
        assert "topic" not in env  # reserved name must NOT be an envelope attribute
        # the HITL review IS human oversight -> IMDA D4 (ADR-141 / ADR-205 D4)
        assert env["chora_imda_dimension"] == IMDA_DIM_OVERSIGHT
        assert env["source_service"] == "chora-ai-kernel-orchestrator"
        assert env["source_project"] == "chora-489812"
        assert env["schema_version"] == SCHEMA_VERSION
        assert env["traceparent"].endswith("-01")
        assert env["tracestate"] == "rojo=1"
        assert env["tenant_id"] == "tenant-9"
        assert env["gcid"] == "learner-1"
        assert env["event_id"] and env["event_id"] != env["idempotency_key"]

    async def test_payload_decodes_to_identity_and_proposed_edges_and_pending_at(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessReviewPendingOutboxWriter(conn=conn, source_project="chora-489812")

        await writer.publish_review_pending(panel=_panel())

        _sql, params = conn.cur.executed[0]
        fields = _walk_proto(bytes(params["payload"]))
        assert 1 in fields and fields[1][0]["wire"] == 2  # envelope submessage
        assert _first_str(fields, 2) == _panel()["upload_id"]
        assert _first_str(fields, 3) == "tenant-9"
        assert _first_str(fields, 4) == "learner-1"
        assert len(fields[6]) == 1  # one proposed edge
        assert len(fields[8]) == 2  # two available outputs
        assert 9 in fields  # pending_at Timestamp stamped at emission

    async def test_missing_upload_id_raises(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessReviewPendingOutboxWriter(conn=conn, source_project="chora-489812")
        with pytest.raises(ValueError, match="upload_id"):
            await writer.publish_review_pending(panel=_panel(upload_id=""))
        assert conn.cur.executed == []

    async def test_empty_proposed_edges_still_publishes(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessReviewPendingOutboxWriter(conn=conn, source_project="chora-489812")

        row_id = await writer.publish_review_pending(panel=_panel(proposed_edges=[]))

        assert row_id
        _sql, params = conn.cur.executed[0]
        fields = _walk_proto(bytes(params["payload"]))
        assert 6 not in fields  # no proposed edges
        assert _first_str(fields, 2) == _panel()["upload_id"]
