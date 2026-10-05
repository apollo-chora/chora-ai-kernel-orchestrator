"""Round-trip + wire-shape tests for the hand-rolled WeaknessReviewPending binary
proto encoder (CHO-1973 Wave A).

Same "binary-proto-encode trap" rationale as the analyzed encoder: a JSON publish
to a BINARY Schema-Registry topic dead-letters; the generated _pb2 is not
importable in this venv. So we emit canonical proto3 wire bytes by hand against
the FROZEN flat schema and verify with a SELF-CONTAINED wire-walker, proving
field numbers / wire types / nesting / repeated fields / float32 + int64 packing
match chora-contracts/proto/events-flat/consumption/weakness/review_pending.proto.
"""

from __future__ import annotations

import struct

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.review_pending_proto_encoder import (
    ReviewPendingProtoEncodeError,
    encode_weakness_review_pending,
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
        else:  # pragma: no cover
            raise AssertionError(f"unexpected wire type {wt}")
        out.setdefault(field, []).append(v)
    return out


def _envelope() -> dict:
    return {
        "event_id": "01970000-0000-7000-e000-000000000001",
        "idempotency_key": "weakness.review_pending.upl-1",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "gcid": "01970000-0000-7000-9000-000000000001",
        "occurred_at": "2026-07-01T12:00:00Z",
        "published_at": "2026-07-01T12:00:01Z",
        "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "source_project": "chora-489812",
        "source_service": "chora-ai-kernel-orchestrator",
        "schema_version": "1",
        "chora_imda_dimension": "fairness_and_human_oversight",
    }


def _panel() -> dict:
    return {
        "upload_id": "01970000-0000-7000-a000-000000000001",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "learner_gcid": "01970000-0000-7000-9000-000000000001",
        "familiar": {"familiar_id": "fam-7", "name": "Ember", "species": "dragon"},
        "proposed_edges": [
            {
                "proposed_edge_id": "pe-0",
                "concept_label": "adding fractions",
                "summary": "mixes denominators",
                "suggested_angles": ["area model", "common denominators"],
                "strength": 0.85,
                "suggested_difficulty": "harder",
            },
            {
                "proposed_edge_id": "pe-1",
                "concept_label": "long division",
                "summary": "",
                "suggested_angles": [],
                "strength": 0.0,
                "suggested_difficulty": "easier",
            },
        ],
        "candidate_struggles": [
            {"concept_key": "borrowing", "concept_label": "borrowing"},
        ],
        "available_outputs": [
            {"kind": "focused_dose", "mana_price": 0, "default_selected": True},
            {"kind": "practice_test", "mana_price": 120, "default_selected": False},
        ],
        "pending_at": "2026-07-01T12:00:00Z",
    }


def test_round_trips_top_level() -> None:
    raw = encode_weakness_review_pending(_envelope(), _panel())
    assert isinstance(raw, bytes) and len(raw) > 0
    top = _walk(raw)

    env = _walk(top[1][0])  # envelope @1
    assert env[1][0].decode() == "01970000-0000-7000-e000-000000000001"  # event_id
    assert env[3][0].decode() == "01970000-0000-7000-8000-000000000001"  # tenant_id
    assert env[11][0] == 1  # schema_version coerced

    assert top[2][0].decode() == "01970000-0000-7000-a000-000000000001"  # upload_id @2
    assert top[3][0].decode() == "01970000-0000-7000-8000-000000000001"  # tenant_id @3
    assert top[4][0].decode() == "01970000-0000-7000-9000-000000000001"  # learner_gcid @4
    assert 9 in top  # pending_at Timestamp @9


def test_familiar_submessage_round_trips() -> None:
    top = _walk(encode_weakness_review_pending(_envelope(), _panel()))
    fam = _walk(top[5][0])  # familiar @5
    assert fam[1][0].decode() == "fam-7"  # familiar_id @1
    assert fam[2][0].decode() == "Ember"  # name @2
    assert fam[3][0].decode() == "dragon"  # species @3


def test_empty_familiar_is_omitted() -> None:
    panel = _panel()
    panel["familiar"] = {}
    top = _walk(encode_weakness_review_pending(_envelope(), panel))
    assert 5 not in top  # absent familiar -> no submessage


def test_proposed_edges_repeated_with_float_strength() -> None:
    top = _walk(encode_weakness_review_pending(_envelope(), _panel()))
    assert len(top[6]) == 2  # two ProposedGrowthEdge submessages @6
    e0 = _walk(top[6][0])
    assert e0[1][0].decode() == "pe-0"  # proposed_edge_id @1
    assert e0[2][0].decode() == "adding fractions"  # concept_label @2
    assert e0[3][0].decode() == "mixes denominators"  # summary @3
    assert [a.decode() for a in e0[4]] == ["area model", "common denominators"]  # @4 repeated
    assert e0[5][0] == pytest.approx(0.85, abs=1e-6)  # strength @5 float32
    assert e0[6][0].decode() == "harder"  # suggested_difficulty @6

    e1 = _walk(top[6][1])
    assert e1[1][0].decode() == "pe-1"
    assert 3 not in e1  # empty summary omitted
    assert 4 not in e1  # empty suggested_angles omitted
    assert 5 not in e1  # 0.0 strength omitted (proto3 default)


def test_candidate_struggles_repeated() -> None:
    top = _walk(encode_weakness_review_pending(_envelope(), _panel()))
    assert len(top[7]) == 1  # CandidateStruggle @7
    s0 = _walk(top[7][0])
    assert s0[1][0].decode() == "borrowing"  # concept_key @1
    assert s0[2][0].decode() == "borrowing"  # concept_label @2


def test_available_outputs_repeated_with_int64_price_and_bool() -> None:
    top = _walk(encode_weakness_review_pending(_envelope(), _panel()))
    assert len(top[8]) == 2  # AvailableOutput @8
    o0 = _walk(top[8][0])
    assert o0[1][0].decode() == "focused_dose"  # kind @1
    assert 2 not in o0  # mana_price 0 omitted (proto3 default)
    assert o0[3][0] == 1  # default_selected true @3

    o1 = _walk(top[8][1])
    assert o1[1][0].decode() == "practice_test"
    assert o1[2][0] == 120  # mana_price @2 int64 varint
    assert 3 not in o1  # default_selected false omitted


def test_large_int64_mana_price() -> None:
    panel = _panel()
    panel["available_outputs"] = [
        {"kind": "study_aids", "mana_price": 5_000_000_000, "default_selected": True},
    ]
    top = _walk(encode_weakness_review_pending(_envelope(), panel))
    o = _walk(top[8][0])
    assert o[2][0] == 5_000_000_000  # exceeds int32 -> proves int64 varint


def test_empty_collections_are_valid_message() -> None:
    panel = _panel()
    panel["proposed_edges"] = []
    panel["candidate_struggles"] = []
    panel["available_outputs"] = []
    panel["familiar"] = {}
    top = _walk(encode_weakness_review_pending(_envelope(), panel))
    assert 6 not in top and 7 not in top and 8 not in top and 5 not in top
    assert top[2][0].decode() == panel["upload_id"]  # still well-formed


def test_missing_pending_at_omits_timestamp() -> None:
    panel = _panel()
    panel.pop("pending_at")
    top = _walk(encode_weakness_review_pending(_envelope(), panel))
    assert 9 not in top


def test_rejects_bad_pending_at() -> None:
    panel = _panel()
    panel["pending_at"] = "not-a-timestamp"
    with pytest.raises(ReviewPendingProtoEncodeError):
        encode_weakness_review_pending(_envelope(), panel)


def test_rejects_non_numeric_mana_price() -> None:
    panel = _panel()
    panel["available_outputs"] = [{"kind": "study_aids", "mana_price": "lots"}]
    with pytest.raises(ReviewPendingProtoEncodeError):
        encode_weakness_review_pending(_envelope(), panel)
