"""Wire-shape tests for the hand-rolled WeaknessOutputsGenerated encoder (WS-7).

Same named risk as the analyzed encoder: the binary-proto-encode trap. JSON
published to a BINARY Schema-Registry topic silently dead-letters with
INVALID_BINARY_PROTO_MESSAGE, and the generated proto is not importable in this
venv (gencode skew; chora_contracts_gen ships only in the runtime image). So we
decode the emitted bytes with a self-contained proto3 wire-walker and assert
field numbers, wire types, nesting and repeated fields match the FROZEN flat
schema chora-contracts/proto/events-flat/consumption/weakness/outputs_generated.proto.

WS-7 context: before this event the generated study_aids / practice_test
artifacts were produced, metered against the learner's mana, and then reached no
learner surface at all.
"""

from __future__ import annotations

import json
import struct

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_proto_encoder import (
    encode_weakness_outputs_generated,
)

pytestmark = pytest.mark.unit

WIRE_VARINT = 0
WIRE_FIXED32 = 5
WIRE_LEN = 2


def _read_varint(data: bytes, i: int) -> tuple[int, int]:
    shift = 0
    result = 0
    while True:
        b = data[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _walk(data: bytes) -> dict[int, list]:
    """Decode proto3 bytes into {field_no: [values]}."""
    out: dict[int, list] = {}
    i = 0
    while i < len(data):
        tag, i = _read_varint(data, i)
        field, wt = tag >> 3, tag & 0x7
        if wt == WIRE_VARINT:
            v, i = _read_varint(data, i)
        elif wt == WIRE_FIXED32:
            v = struct.unpack("<f", data[i : i + 4])[0]
            i += 4
        elif wt == WIRE_LEN:
            ln, i = _read_varint(data, i)
            v = data[i : i + ln]
            i += ln
        else:  # pragma: no cover - encoder never emits other wire types
            raise AssertionError(f"unexpected wire type {wt}")
        out.setdefault(field, []).append(v)
    return out


def _envelope() -> dict:
    return {
        "event_id": "01970000-0000-7000-e000-000000000009",
        "idempotency_key": "weakness.outputs_generated.upl-1",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "gcid": "01970000-0000-7000-9000-000000000001",
        "occurred_at": "2026-07-23T12:00:00Z",
        "published_at": "2026-07-23T12:00:01Z",
        "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "source_project": "chora-489812",
        "source_service": "chora-ai-kernel-orchestrator",
        "schema_version": "1",
        "chora_imda_dimension": "transparency",
    }


PRACTICE = {"questions": [{"stem": "What is 2+2?", "answer": "4"}]}


def _body() -> dict:
    return {
        "upload_id": "01970000-0000-7000-a000-000000000001",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "learner_gcid": "01970000-0000-7000-9000-000000000001",
        "generated_at": "2026-07-23T12:00:05Z",
        # Exactly the shape CrewOutputGenerator.generate returns: study_aids
        # carries a str, practice_test carries a dict.
        "outputs": [
            {"type": "study_aids", "content": "Focus on fluvial vs pluvial.", "metered": True},
            {"type": "practice_test", "content": PRACTICE, "metered": True},
        ],
    }


class TestTopLevelShape:
    def test_field_numbers_and_nesting(self) -> None:
        got = _walk(encode_weakness_outputs_generated(_envelope(), _body()))
        assert set(got) == {1, 2, 3, 4, 5, 6}
        assert got[2][0].decode() == "01970000-0000-7000-a000-000000000001"
        assert got[3][0].decode() == "01970000-0000-7000-8000-000000000001"
        assert got[4][0].decode() == "01970000-0000-7000-9000-000000000001"
        assert len(got[5]) == 2, "outputs is repeated field 5, one entry per artifact"

    def test_envelope_rides_field_1(self) -> None:
        got = _walk(encode_weakness_outputs_generated(_envelope(), _body()))
        env = _walk(got[1][0])
        assert env[1][0].decode() == "01970000-0000-7000-e000-000000000009"
        assert env[2][0].decode() == "weakness.outputs_generated.upl-1"
        assert env[14][0].decode() == "transparency"

    def test_generated_at_is_a_timestamp_submessage(self) -> None:
        got = _walk(encode_weakness_outputs_generated(_envelope(), _body()))
        ts = _walk(got[6][0])
        assert ts[1][0] > 0, "seconds must be set"


class TestOutputEncoding:
    def test_kind_comes_from_the_generator_type_key(self) -> None:
        """CrewOutputGenerator emits {"type": ...}; the wire field is `kind`."""
        got = _walk(encode_weakness_outputs_generated(_envelope(), _body()))
        kinds = [_walk(o)[1][0].decode() for o in got[5]]
        assert kinds == ["study_aids", "practice_test"]

    def test_str_content_is_json_encoded(self) -> None:
        """study_aids content is prose; it must arrive as a JSON string so the
        consumer can store one uniform JSONB column for both kinds."""
        got = _walk(encode_weakness_outputs_generated(_envelope(), _body()))
        aid = _walk(got[5][0])
        assert json.loads(aid[2][0].decode()) == "Focus on fluvial vs pluvial."

    def test_dict_content_is_json_encoded(self) -> None:
        got = _walk(encode_weakness_outputs_generated(_envelope(), _body()))
        test = _walk(got[5][1])
        assert json.loads(test[2][0].decode()) == PRACTICE

    def test_metered_true_rides_as_varint(self) -> None:
        got = _walk(encode_weakness_outputs_generated(_envelope(), _body()))
        assert _walk(got[5][0])[3][0] == 1

    def test_metered_false_is_omitted_proto3_default(self) -> None:
        body = _body()
        body["outputs"] = [{"type": "study_aids", "content": "x", "metered": False}]
        got = _walk(encode_weakness_outputs_generated(_envelope(), body))
        assert 3 not in _walk(got[5][0]), "proto3 omits a false bool"


class TestEmptyAndDegenerate:
    def test_no_outputs_still_encodes_a_valid_message(self) -> None:
        """Every kind screened out or soft-failed is a REAL outcome: the event
        still fires so consumption can tell 'generated nothing' apart from
        'not generated yet'."""
        body = _body()
        body["outputs"] = []
        got = _walk(encode_weakness_outputs_generated(_envelope(), body))
        assert 5 not in got
        assert got[2][0].decode() == "01970000-0000-7000-a000-000000000001"

    def test_missing_upload_id_raises(self) -> None:
        """Fail loud: an event that cannot be correlated to its upload is
        unroutable on the consumer side."""
        body = _body()
        body["upload_id"] = ""
        with pytest.raises(ValueError):
            encode_weakness_outputs_generated(_envelope(), body)
