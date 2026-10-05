"""RED->GREEN tests for WeaknessOutputsGeneratedOutboxWriter (WS-7 delivery).

The writer is the BINARY transactional-outbox emitter for
``chora.consumption.weakness.outputs_generated.v1``. It INSERTs one row into
``ai_kernel_outbox_events`` whose ``payload`` is the canonical proto3 wire bytes
from ``weakness_proto_encoder.encode_weakness_outputs_generated``; the existing
``OutboxDispatcher`` drains it.

WS-7 gap it closes: the crew generated study_aids / practice_test, metered them
against the learner's mana, and then dropped them on the floor because no
delivery path existed (see the comment at
``orchestrators/weakness_analyser_crew.py::_run_detached_generation``).

Two traps asserted here because both are silent in production:

  1. The binary-proto trap. JSON on a Schema-Registry BINARY topic dead-letters
     with INVALID_BINARY_PROTO_MESSAGE, which surfaces nowhere the learner or the
     operator looks.
  2. The ``event_topic`` routing workaround. The Python
     ``GoogleCloudPubSubPublisher`` STRIPS the reserved ``topic`` attribute, and
     chora-consumption's push handler routes by topic. Without ``event_topic`` in
     the envelope the event arrives with no routable topic and is dropped.

Mirrors ``test_weakness_analyzed_outbox_writer.py``: local wire walker + fake
AsyncConnection; NO live DB, NO Pub/Sub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_outputs_outbox_writer import (
    IMDA_DIM_TRANSPARENCY,
    SCHEMA_VERSION,
    TOPIC_WEAKNESS_OUTPUTS_GENERATED,
    WeaknessOutputsGeneratedOutboxWriter,
)

pytestmark = pytest.mark.unit


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


def _walk(data: bytes) -> dict[int, list]:
    out: dict[int, list] = {}
    i = 0
    while i < len(data):
        tag, i = _read_varint(data, i)
        field_no, wt = tag >> 3, tag & 0x7
        if wt == 0:
            v, i = _read_varint(data, i)
        elif wt == 5:
            v = data[i : i + 4]
            i += 4
        elif wt == 2:
            ln, i = _read_varint(data, i)
            v = data[i : i + ln]
            i += ln
        else:  # pragma: no cover
            raise AssertionError(f"unexpected wire type {wt}")
        out.setdefault(field_no, []).append(v)
    return out


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


UPLOAD_ID = "01970000-0000-7000-a000-000000000001"
TENANT_ID = "01970000-0000-7000-8000-000000000001"
GCID = "01970000-0000-7000-9000-000000000001"


def _writer(conn: _FakeAsyncConnection) -> WeaknessOutputsGeneratedOutboxWriter:
    return WeaknessOutputsGeneratedOutboxWriter(conn=conn, source_project="chora-489812")


def _body(outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "upload_id": UPLOAD_ID,
        "tenant_id": TENANT_ID,
        "learner_gcid": GCID,
        "generated_at": "2026-07-23T12:00:05Z",
        "outputs": outputs
        if outputs is not None
        else [{"type": "study_aids", "content": "Revise fluvial vs pluvial.", "metered": True}],
    }


class TestOutboxRow:
    @pytest.mark.asyncio
    async def test_inserts_exactly_one_row(self) -> None:
        conn = _FakeAsyncConnection()
        await _writer(conn).publish_outputs_generated(body=_body())
        assert len(conn.cur.executed) == 1

    @pytest.mark.asyncio
    async def test_row_targets_the_outputs_generated_topic(self) -> None:
        conn = _FakeAsyncConnection()
        await _writer(conn).publish_outputs_generated(body=_body())
        _sql, params = conn.cur.executed[0]
        assert params["topic"] == TOPIC_WEAKNESS_OUTPUTS_GENERATED
        assert params["topic"] == "chora.consumption.weakness.outputs_generated.v1"
        assert params["event_type"] == "consumption.weakness.outputs_generated"

    @pytest.mark.asyncio
    async def test_row_carries_tenant_and_learner_for_rls_and_sharding(self) -> None:
        conn = _FakeAsyncConnection()
        await _writer(conn).publish_outputs_generated(body=_body())
        _sql, params = conn.cur.executed[0]
        assert params["tenant_id"] == TENANT_ID
        assert params["gcid"] == GCID
        assert params["workflow_id"] == UPLOAD_ID

    @pytest.mark.asyncio
    async def test_insert_is_idempotent_on_the_key(self) -> None:
        """Redelivery must be a no-op at the outbox layer (D6 P2)."""
        conn = _FakeAsyncConnection()
        await _writer(conn).publish_outputs_generated(body=_body())
        sql, params = conn.cur.executed[0]
        assert "ON CONFLICT (idempotency_key) DO NOTHING" in sql
        assert params["idempotency_key"] == f"weakness.outputs_generated.{UPLOAD_ID}"


class TestEnvelope:
    @pytest.mark.asyncio
    async def test_carries_event_topic_routing_workaround(self) -> None:
        """Without this the publisher strips `topic` and consumption drops it."""
        conn = _FakeAsyncConnection()
        await _writer(conn).publish_outputs_generated(body=_body())
        _sql, params = conn.cur.executed[0]
        env = json.loads(params["envelope"])
        assert env["event_topic"] == TOPIC_WEAKNESS_OUTPUTS_GENERATED

    @pytest.mark.asyncio
    async def test_carries_the_mandatory_envelope_fields(self) -> None:
        conn = _FakeAsyncConnection()
        await _writer(conn).publish_outputs_generated(body=_body(), traceparent="00-abc-def-01", tracestate="chora=1")
        _sql, params = conn.cur.executed[0]
        env = json.loads(params["envelope"])
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
            assert env.get(key) not in (None, ""), f"envelope missing {key}"
        assert env["traceparent"] == "00-abc-def-01"
        assert env["schema_version"] == SCHEMA_VERSION
        assert env["chora_imda_dimension"] == IMDA_DIM_TRANSPARENCY


class TestPayloadIsBinaryProto:
    @pytest.mark.asyncio
    async def test_payload_decodes_as_the_flat_schema(self) -> None:
        conn = _FakeAsyncConnection()
        await _writer(conn).publish_outputs_generated(
            body=_body(
                [
                    {"type": "study_aids", "content": "prose", "metered": True},
                    {"type": "practice_test", "content": {"q": 1}, "metered": True},
                ]
            )
        )
        _sql, params = conn.cur.executed[0]
        payload = params["payload"]
        assert isinstance(payload, bytes), "must be binary, not a JSON str"
        got = _walk(payload)
        assert got[2][0].decode() == UPLOAD_ID
        assert len(got[5]) == 2
        assert _walk(got[5][0])[1][0].decode() == "study_aids"
        assert json.loads(_walk(got[5][1])[2][0].decode()) == {"q": 1}


class TestDegenerate:
    @pytest.mark.asyncio
    async def test_empty_outputs_still_emits(self) -> None:
        """'Generated nothing' is a real outcome and must be distinguishable
        from 'not generated yet' on the consumer side."""
        conn = _FakeAsyncConnection()
        row_id = await _writer(conn).publish_outputs_generated(body=_body([]))
        assert row_id
        assert len(conn.cur.executed) == 1

    @pytest.mark.asyncio
    async def test_missing_upload_id_raises_and_writes_nothing(self) -> None:
        conn = _FakeAsyncConnection()
        body = _body()
        body["upload_id"] = ""
        with pytest.raises(ValueError):
            await _writer(conn).publish_outputs_generated(body=body)
        assert conn.cur.executed == []

    def test_source_project_is_required(self) -> None:
        with pytest.raises(ValueError):
            WeaknessOutputsGeneratedOutboxWriter(conn=_FakeAsyncConnection(), source_project=" ")
