"""RED→GREEN tests for WeaknessAnalyzedOutboxWriter (W2-glue).

The writer is the BINARY transactional-outbox emitter for
``chora.consumption.weakness.analyzed.v1`` — it INSERTs one row into
``ai_kernel_outbox_events`` (migration 0003_outbox.sql) whose ``payload`` is the
canonical proto3 wire bytes produced by ``weakness_proto_encoder`` (the topic
carries a Pub/Sub Schema Registry BINARY schema; a JSON publish dead-letters with
INVALID_BINARY_PROTO_MESSAGE — the binary-proto trap). The existing
``OutboxDispatcher`` + ``GoogleCloudPubSubPublisher`` drain the row.

Mirrors ``test_qgen_crew_publisher.py`` (the qgen binary writer): a local proto3
wire walker + fake AsyncConnection/Cursor; NO live DB, NO Pub/Sub.

Routing note (asserted below): the Python ``GoogleCloudPubSubPublisher`` STRIPS
the reserved ``topic`` attribute, and chora-consumption's push handler routes by
the message ``topic`` attribute. So the writer puts the routing topic under the
non-reserved ``event_topic`` envelope key (same workaround as the OE grading
publisher) — without it the analyzed event reaches consumption with no routable
topic and is dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyzed_outbox_writer import (
    IMDA_DIM_TRANSPARENCY,
    SCHEMA_VERSION,
    TOPIC_WEAKNESS_ANALYZED,
    WeaknessAnalyzedOutboxWriter,
)

# -----------------------------------------------------------------------------
# Tiny proto3 wire-format walker (payload is BINARY, not JSON).
# -----------------------------------------------------------------------------


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
    """Walk top-level proto3 fields → {field_number: [{"wire", "value"}, ...]}.

    Repeated fields accumulate (so we can assert on multiple ``edges``).
    """
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


# -----------------------------------------------------------------------------
# Fake AsyncConnection / Cursor — mirrors psycopg's AsyncConnection surface.
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

    def cursor(self) -> _FakeCursor:
        return self.cur


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "upload_id": "0190aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee",
        "tenant_id": "tenant-9",
        "learner_gcid": "learner-1",
        "model_used": "gemini-2.5-pro",
        "input_token_count": 1200,
        "output_token_count": 340,
        "analyzed_at": "2026-06-09T10:00:00+00:00",
        "edges": [
            {
                "concept_label": "causes of riverine flooding",
                "concept_key": "riverine-flood-causes",
                "category": "physical-geography",
                "tags": ["flooding", "causation"],
                "confidence": 0.82,
                "strength": 0.7,
                "descriptor_json": json.dumps({"summary": "shaky on causation"}),
            }
        ],
    }
    body.update(overrides)
    return body


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_requires_source_project(self) -> None:
        with pytest.raises(ValueError, match="source_project"):
            WeaknessAnalyzedOutboxWriter(conn=_FakeAsyncConnection(), source_project="")

    def test_requires_source_service(self) -> None:
        with pytest.raises(ValueError, match="source_service"):
            WeaknessAnalyzedOutboxWriter(conn=_FakeAsyncConnection(), source_project="p", source_service="")

    def test_source_project_property(self) -> None:
        w = WeaknessAnalyzedOutboxWriter(conn=_FakeAsyncConnection(), source_project="chora-489812")
        assert w.source_project == "chora-489812"


# -----------------------------------------------------------------------------
# Outbox INSERT
# -----------------------------------------------------------------------------


class TestPublish:
    async def test_inserts_single_pending_row_to_binary_topic(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessAnalyzedOutboxWriter(conn=conn, source_project="chora-489812")

        row_id = await writer.publish_weakness_analyzed(body=_body())

        assert isinstance(row_id, str) and row_id
        assert len(conn.cur.executed) == 1
        _sql, params = conn.cur.executed[0]
        assert params["topic"] == TOPIC_WEAKNESS_ANALYZED
        assert params["tenant_id"] == "tenant-9"
        assert params["gcid"] == "learner-1"
        assert params["workflow_id"] == _body()["upload_id"]
        assert params["event_type"] == "consumption.weakness.analyzed"
        # Payload is BINARY proto bytes, not JSON.
        assert isinstance(params["payload"], (bytes, bytearray))

    async def test_idempotency_key_uses_upload_id(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessAnalyzedOutboxWriter(conn=conn, source_project="chora-489812")

        await writer.publish_weakness_analyzed(body=_body())

        _sql, params = conn.cur.executed[0]
        assert params["idempotency_key"] == f"weakness.analyzed.{_body()['upload_id']}"

    async def test_envelope_carries_event_topic_and_transparency(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessAnalyzedOutboxWriter(conn=conn, source_project="chora-489812")

        await writer.publish_weakness_analyzed(
            body=_body(),
            traceparent="00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            tracestate="rojo=00f067aa0ba902b7",
        )

        _sql, params = conn.cur.executed[0]
        env = json.loads(params["envelope"])
        # event_topic survives the publisher's reserved-key strip → consumption routes on it.
        assert env["event_topic"] == TOPIC_WEAKNESS_ANALYZED
        assert "topic" not in env  # reserved name must NOT be an envelope attribute
        assert env["chora_imda_dimension"] == IMDA_DIM_TRANSPARENCY
        assert env["source_service"] == "chora-ai-kernel-orchestrator"
        assert env["source_project"] == "chora-489812"
        assert env["schema_version"] == SCHEMA_VERSION
        assert env["traceparent"].endswith("-01")
        assert env["tracestate"] == "rojo=00f067aa0ba902b7"
        assert env["idempotency_key"] == f"weakness.analyzed.{_body()['upload_id']}"
        # event_id is a fresh UUID (NOT the idempotency_key).
        assert env["event_id"] and env["event_id"] != env["idempotency_key"]

    async def test_payload_decodes_to_upload_tenant_gcid_and_edges(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessAnalyzedOutboxWriter(conn=conn, source_project="chora-489812")

        await writer.publish_weakness_analyzed(body=_body())

        _sql, params = conn.cur.executed[0]
        fields = _walk_proto(bytes(params["payload"]))
        # field 1 = nested envelope (len), 2 = upload_id, 3 = tenant_id, 4 = learner_gcid
        assert 1 in fields and fields[1][0]["wire"] == 2
        assert _first_str(fields, 2) == _body()["upload_id"]
        assert _first_str(fields, 3) == "tenant-9"
        assert _first_str(fields, 4) == "learner-1"
        # field 5 = repeated ExtractedGrowthEdge (one here), field 6 = model_used.
        assert len(fields[5]) == 1
        assert _first_str(fields, 6) == "gemini-2.5-pro"
        # nested edge: field 1 = concept_label.
        edge_fields = _walk_proto(fields[5][0]["value"])
        assert _first_str(edge_fields, 1) == "causes of riverine flooding"

    async def test_missing_upload_id_raises(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessAnalyzedOutboxWriter(conn=conn, source_project="chora-489812")
        with pytest.raises(ValueError, match="upload_id"):
            await writer.publish_weakness_analyzed(body=_body(upload_id=""))
        assert conn.cur.executed == []  # nothing inserted

    async def test_missing_analyzed_at_falls_back_to_now(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessAnalyzedOutboxWriter(conn=conn, source_project="chora-489812")

        body = _body()
        body.pop("analyzed_at")
        await writer.publish_weakness_analyzed(body=body)

        _sql, params = conn.cur.executed[0]
        env = json.loads(params["envelope"])
        # occurred_at still populated (emission-time fallback) → encoder/Go side
        # never sees an empty timestamp.
        assert env["occurred_at"]

    async def test_empty_edges_still_publishes(self) -> None:
        conn = _FakeAsyncConnection()
        writer = WeaknessAnalyzedOutboxWriter(conn=conn, source_project="chora-489812")

        row_id = await writer.publish_weakness_analyzed(body=_body(edges=[]))

        assert row_id
        assert len(conn.cur.executed) == 1
        _sql, params = conn.cur.executed[0]
        fields = _walk_proto(bytes(params["payload"]))
        assert 5 not in fields  # no edges encoded
        assert _first_str(fields, 2) == _body()["upload_id"]  # but upload_id present
