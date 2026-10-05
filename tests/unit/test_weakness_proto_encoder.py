"""Round-trip + wire-shape tests for the hand-rolled WeaknessAnalyzed binary
proto encoder (W2 of the 1b Growth-Edge track).

The "binary-proto-encode trap" (JSON published to a BINARY Schema-Registry topic
silently dead-letters) is the named risk here. The generated proto is not
importable in this venv (gencode skew + chora_contracts_gen ships only in the
runtime image), so we verify against a SELF-CONTAINED proto3 wire-walker that
decodes the emitted bytes back to fields — proving field numbers, wire types,
nesting, repeated fields, and float32 packing match the FROZEN flat schema
chora-contracts/proto/events-flat/consumption/weakness/analyzed.proto.
"""

from __future__ import annotations

import struct

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_proto_encoder import (
    WeaknessProtoEncodeError,
    _encode_varint,
    encode_weakness_analyzed,
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
    """Decode proto3 bytes into {field_no: [values]} (values: int | float | bytes)."""
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
        "event_id": "01970000-0000-7000-e000-000000000001",
        "idempotency_key": "weakness.analyzed.upl-1",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "gcid": "01970000-0000-7000-9000-000000000001",
        "occurred_at": "2026-06-09T12:00:00Z",
        "published_at": "2026-06-09T12:00:01Z",
        "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "source_project": "chora-489812",
        "source_service": "chora-ai-kernel-orchestrator",
        "schema_version": "1",
        "chora_imda_dimension": "transparency",
    }


def _body() -> dict:
    return {
        "upload_id": "01970000-0000-7000-a000-000000000001",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "learner_gcid": "01970000-0000-7000-9000-000000000001",
        "model_used": "gemini-3-pro-preview",
        "input_token_count": 1200,
        "output_token_count": 340,
        "analyzed_at": "2026-06-09T12:00:05Z",
        "edges": [
            {
                "concept_label": "causes of riverine flooding",
                "concept_key": "causes-of-riverine-flooding",
                "category": "physical-geography",
                "tags": ["flooding", "causation"],
                "confidence": 0.9,
                "strength": 0.8,
                "descriptor_json": '{"summary":"confuses fluvial vs pluvial"}',
            },
            {
                "concept_label": "tectonic plate boundaries",
                "concept_key": "tectonic-plate-boundaries",
                "category": "physical-geography",
                "tags": [],
                "confidence": 0.7,
                "strength": 0.6,
                "descriptor_json": "{}",
            },
        ],
    }


def test_encode_weakness_analyzed_round_trips_top_level() -> None:
    raw = encode_weakness_analyzed(_envelope(), _body())
    assert isinstance(raw, bytes) and len(raw) > 0
    top = _walk(raw)

    # field 1 = Envelope (length-delimited submessage)
    assert 1 in top
    env = _walk(top[1][0])
    assert env[3][0].decode() == "01970000-0000-7000-8000-000000000001"  # tenant_id @3
    assert env[1][0].decode() == "01970000-0000-7000-e000-000000000001"  # event_id @1
    assert 5 in env  # occurred_at Timestamp submessage present
    assert env[11][0] == 1  # schema_version @11 coerced "1" -> int 1

    assert top[2][0].decode() == "01970000-0000-7000-a000-000000000001"  # upload_id @2
    assert top[3][0].decode() == "01970000-0000-7000-8000-000000000001"  # tenant_id @3
    assert top[4][0].decode() == "01970000-0000-7000-9000-000000000001"  # learner_gcid @4
    assert top[6][0].decode() == "gemini-3-pro-preview"  # model_used @6
    assert top[7][0] == 1200  # input_token_count @7
    assert top[8][0] == 340  # output_token_count @8
    assert 9 in top  # analyzed_at Timestamp @9


def test_encode_weakness_analyzed_edges_repeated_with_floats() -> None:
    raw = encode_weakness_analyzed(_envelope(), _body())
    top = _walk(raw)

    # field 5 repeated -> two edge submessages
    assert len(top[5]) == 2
    e0 = _walk(top[5][0])
    assert e0[1][0].decode() == "causes of riverine flooding"  # concept_label @1
    assert e0[2][0].decode() == "causes-of-riverine-flooding"  # concept_key @2
    assert e0[3][0].decode() == "physical-geography"  # category @3
    # tags @4 repeated string
    assert [t.decode() for t in e0[4]] == ["flooding", "causation"]
    # confidence @5 + strength @6 are float32 (fixed32)
    assert e0[5][0] == pytest.approx(0.9, abs=1e-6)
    assert e0[6][0] == pytest.approx(0.8, abs=1e-6)
    assert e0[7][0].decode() == '{"summary":"confuses fluvial vs pluvial"}'

    e1 = _walk(top[5][1])
    assert e1[1][0].decode() == "tectonic plate boundaries"
    assert 4 not in e1  # empty tags -> no field 4 emitted


def test_encode_empty_edges_is_valid() -> None:
    body = _body()
    body["edges"] = []
    raw = encode_weakness_analyzed(_envelope(), body)
    top = _walk(raw)
    assert 5 not in top  # no edges
    assert top[2][0].decode() == body["upload_id"]  # still a well-formed message


def test_encode_weakness_analyzed_output_selection() -> None:
    # Integration seam (ADR-205 D5 / CHO-1966): the learner's post-HITL output
    # selection rides field 10 (an inlined OutputSelection submessage) so the
    # consumption Familiar-RAG subscriber can gate familiar_coaching. Bools encode
    # as varints; proto3 omits the false default.
    body = _body()
    body["output_selection"] = {
        "focused_dose": True,
        "familiar_coaching": True,
        "practice_test": False,  # false -> omitted
        "study_aids": True,
    }
    top = _walk(encode_weakness_analyzed(_envelope(), body))
    assert 10 in top  # output_selection submessage present
    sel = _walk(top[10][0])
    assert sel[1][0] == 1  # focused_dose @1
    assert sel[2][0] == 1  # familiar_coaching @2
    assert 3 not in sel  # practice_test false omitted
    assert sel[4][0] == 1  # study_aids @4


def test_encode_omits_empty_output_selection() -> None:
    # No selection (the live single-shot path never sets it) or an all-false
    # selection -> field 10 omitted, so default-off familiar_coaching is
    # unambiguous on the wire (absent == all-false == no opt-in).
    body = _body()
    top = _walk(encode_weakness_analyzed(_envelope(), body))
    assert 10 not in top  # absent when no output_selection key

    body["output_selection"] = {
        "focused_dose": False,
        "familiar_coaching": False,
        "practice_test": False,
        "study_aids": False,
    }
    top2 = _walk(encode_weakness_analyzed(_envelope(), body))
    assert 10 not in top2  # all-false collapses to no submessage (canonical proto3)


def test_encode_omits_zero_floats_and_default_scalars() -> None:
    body = _body()
    body["edges"] = [
        {
            "concept_label": "x",
            "concept_key": "x",
            "confidence": 0.0,  # proto3 default -> omitted
            "strength": 0.0,
            "tags": [],
            "descriptor_json": "",
        }
    ]
    body["input_token_count"] = 0  # default -> omitted
    raw = encode_weakness_analyzed(_envelope(), body)
    top = _walk(raw)
    assert 7 not in top  # input_token_count 0 omitted
    edge = _walk(top[5][0])
    assert 5 not in edge and 6 not in edge  # zero floats omitted
    assert 7 not in edge  # empty descriptor_json omitted


def test_encode_accepts_string_token_counts() -> None:
    body = _body()
    body["input_token_count"] = "1200"  # publisher may pass strings
    raw = encode_weakness_analyzed(_envelope(), body)
    top = _walk(raw)
    assert top[7][0] == 1200


def test_encode_coerces_float_token_count() -> None:
    body = _body()
    body["output_token_count"] = 12.9  # float -> int
    top = _walk(encode_weakness_analyzed(_envelope(), body))
    assert top[8][0] == 12


def test_encode_minimal_envelope_and_no_analyzed_at() -> None:
    # Envelope without occurred_at/published_at + body without analyzed_at:
    # the optional Timestamp submessages are simply omitted.
    raw = encode_weakness_analyzed(
        {"event_id": "e", "tenant_id": "t"},
        {"upload_id": "u", "edges": []},
    )
    top = _walk(raw)
    env = _walk(top[1][0])
    assert 5 not in env and 6 not in env  # occurred_at/published_at omitted
    assert 9 not in top  # analyzed_at omitted


def test_encode_rejects_bool_token_count() -> None:
    body = _body()
    body["input_token_count"] = True  # bool must not silently map to int 1
    with pytest.raises(WeaknessProtoEncodeError):
        encode_weakness_analyzed(_envelope(), body)


def test_encode_rejects_unparseable_token_count() -> None:
    body = _body()
    body["input_token_count"] = "not-a-number"
    with pytest.raises(WeaknessProtoEncodeError):
        encode_weakness_analyzed(_envelope(), body)


def test_encode_rejects_unsupported_token_type() -> None:
    body = _body()
    body["input_token_count"] = [1, 2]
    with pytest.raises(WeaknessProtoEncodeError):
        encode_weakness_analyzed(_envelope(), body)


def test_encode_rejects_bad_timestamp() -> None:
    body = _body()
    body["analyzed_at"] = "definitely not rfc3339"
    with pytest.raises(WeaknessProtoEncodeError):
        encode_weakness_analyzed(_envelope(), body)


def test_encode_rejects_bad_confidence() -> None:
    body = _body()
    body["edges"][0]["confidence"] = "high"  # unparseable float
    with pytest.raises(WeaknessProtoEncodeError):
        encode_weakness_analyzed(_envelope(), body)


def test_encode_varint_rejects_negative() -> None:
    with pytest.raises(WeaknessProtoEncodeError):
        _encode_varint(-1)
