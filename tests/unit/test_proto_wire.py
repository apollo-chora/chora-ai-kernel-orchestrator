"""Tests for the minimal AiAssistStarted proto-wire decoder.

Covers:
* roundtrip from hand-built fixture bytes
* envelope field extraction (event_id + idempotency_key + traceparent)
* metadata map decoding (multi-entry)
* W8 author-opt-in image flags (field 13 image_for_stem / field 14
  image_for_answer — both bool/varint) decode true when present and
  default false when absent (byte-compatible with pre-W8 messages)
* JSON-vs-proto sniff heuristic
* malformed input fails loud

Wire-format note (W8 image flags)
----------------------------------
``image_for_stem`` (field 13) and ``image_for_answer`` (field 14) are
proto ``bool`` scalars → varint (wire type 0) on the wire:

    field 13 tag = (13 << 3) | 0 = 0x68 ; value byte 0x01 (true) / 0x00 (false)
    field 14 tag = (14 << 3) | 0 = 0x70 ; value byte 0x01 (true) / 0x00 (false)

The hand-built byte sequences below were cross-checked against the
protobuf library's own encoder for a TYPE_BOOL field at numbers 13/14:
``AiAssistStarted(image_for_stem=True, image_for_answer=True)`` serialises
to exactly ``68 01 70 01``; proto3 omits false scalars entirely (the
"absent → default false" case is zero bytes), which is why a pre-W8
message decodes to both flags False with no extra bytes.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire import (
    ProtoDecodeError,
    decode_ai_assist_started,
    looks_like_binary_proto,
)


def _build_envelope() -> bytes:
    # field 1 event_id="EV" → tag 0x0A, len 2, "EV"
    # field 2 idempotency_key="IDK" → tag 0x12, len 3, "IDK"
    # field 7 traceparent="00-...-01" → tag 0x3A, len 1, "T"
    return b"\x0a\x02EV\x12\x03IDK\x3a\x01T"


def _build_map_entry(key: str, value: str) -> bytes:
    kb = key.encode("utf-8")
    vb = value.encode("utf-8")
    return b"\x0a" + bytes([len(kb)]) + kb + b"\x12" + bytes([len(vb)]) + vb


def _build_ai_assist_started_fixture() -> bytes:
    env = _build_envelope()
    me1 = _build_map_entry("subject", "Biology")
    me2 = _build_map_entry("cognitive_level", "knowledge")
    return (
        # field 1 envelope (length-delimited)
        b"\x0a"
        + bytes([len(env)])
        + env
        +
        # field 2 assist_id="AS"
        b"\x12\x02AS"
        +
        # field 3 tenant_id="TEN"
        b"\x1a\x03TEN"
        +
        # field 4 author_gcid="GCID"
        b"\x22\x04GCID"
        +
        # field 6 content_type="mcq"
        b"\x32\x03mcq"
        +
        # field 7 prompt="hi"
        b"\x3a\x02hi"
        +
        # field 11 max_retries=3 (varint, wire_type 0). tag (11<<3)|0 = 88 = 0x58
        b"\x58\x03"
        +
        # field 12 metadata MapEntry #1 (length-delimited, tag 0x62)
        b"\x62"
        + bytes([len(me1)])
        + me1
        +
        # field 12 metadata MapEntry #2
        b"\x62"
        + bytes([len(me2)])
        + me2
    )


def test_decode_ai_assist_started_fixture_round_trips() -> None:
    bz = _build_ai_assist_started_fixture()
    out = decode_ai_assist_started(bz)

    assert out["assist_id"] == "AS"
    assert out["tenant_id"] == "TEN"
    assert out["author_gcid"] == "GCID"
    assert out["content_type"] == "mcq"
    assert out["question_type"] == "mcq"  # mirrored from content_type
    assert out["prompt"] == "hi"
    assert out["max_retries"] == 3
    assert out["metadata"] == {
        "subject": "Biology",
        "cognitive_level": "knowledge",
    }
    # Envelope nested + traceparent surfaced top-level
    assert out["envelope"]["event_id"] == "EV"
    assert out["envelope"]["idempotency_key"] == "IDK"
    assert out["envelope"]["traceparent"] == "T"
    assert out["traceparent"] == "T"


def test_decode_ai_assist_started_skips_unknown_fields() -> None:
    # Add an unknown field 99 (varint=66), should be skipped without raising.
    # Tag (99<<3)|0 = 792 → varint encoding 0x98 0x06.
    bz = (
        b"\x0a\x00"  # field 1 envelope (empty)
        b"\x12\x02AS"  # field 2 assist_id
        b"\x98\x06\x42"  # field 99 varint=66 (skipped)
    )
    out = decode_ai_assist_started(bz)
    assert out["assist_id"] == "AS"


def test_decode_surfaces_w8_image_flags_when_present() -> None:
    """W8 author-opt-in: a message carrying image_for_stem (field 13) +
    image_for_answer (field 14) set true surfaces both flags as True on the
    decoded payload so QGenCrewRunner.AiAssistStartedPayload.from_event
    picks them up (it reads ``image_for_stem`` / ``image_for_answer`` with a
    default-false fallback).
    """
    env = _build_envelope()
    bz = (
        # field 1 envelope (length-delimited)
        b"\x0a"
        + bytes([len(env)])
        + env
        +
        # field 2 assist_id="AS"
        b"\x12\x02AS"
        +
        # field 6 content_type="mcq"
        b"\x32\x03mcq"
        +
        # field 13 image_for_stem=true  (tag 0x68, varint 0x01)
        b"\x68\x01"
        +
        # field 14 image_for_answer=true (tag 0x70, varint 0x01)
        b"\x70\x01"
    )
    out = decode_ai_assist_started(bz)

    assert out["assist_id"] == "AS"
    assert out["content_type"] == "mcq"
    assert out["image_for_stem"] is True
    assert out["image_for_answer"] is True


def test_decode_w8_image_flags_independent() -> None:
    """The two flags decode independently — stem true, answer false (false
    encoded explicitly as varint 0x00, as a non-proto3 / explicit producer
    might emit it)."""
    bz = (
        b"\x12\x02AS"  # field 2 assist_id
        + b"\x68\x01"  # field 13 image_for_stem=true
        + b"\x70\x00"  # field 14 image_for_answer=false (explicit)
    )
    out = decode_ai_assist_started(bz)
    assert out["image_for_stem"] is True
    assert out["image_for_answer"] is False


def test_decode_w8_image_flags_default_false_when_absent() -> None:
    """Pre-W8 message (no field 13/14, exactly the existing fixture) decodes
    to both flags False — byte/behaviour-compatible with today. proto3 omits
    false scalars entirely so an opt-in-off producer emits zero extra bytes.
    """
    bz = _build_ai_assist_started_fixture()
    out = decode_ai_assist_started(bz)
    assert out["image_for_stem"] is False
    assert out["image_for_answer"] is False
    # The pre-W8 fixture must still decode all its original fields unchanged.
    assert out["assist_id"] == "AS"
    assert out["question_type"] == "mcq"
    assert out["metadata"] == {
        "subject": "Biology",
        "cognitive_level": "knowledge",
    }


# -----------------------------------------------------------------------------
# EPIC-1a — batch + grounding fields (AiAssistStarted f15-19), added by the 1a
# contract (c6a9e38b). All five are length-delimited (wire type 2):
#   job_kind=15 / grounding_mode=16 / source_blob_uri=17 /
#   source_mime_type=18 / target_growth_edges=19 (repeated).
# The decoder previously stopped at f14 ("other length-delimited fields
# ignored") so these were silently dropped — which routed every batch job to
# the single runner (count=N collapsed to 1) and starved the grounding plugin's
# session state. These tests pin the decode + the downstream from_event
# contract the live write-walk exposed.
# -----------------------------------------------------------------------------


def _encode_varint(value: int) -> bytes:
    """Encode an unsigned int as a base-128 varint (for multi-byte field tags)."""
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld_field(field: int, value: str) -> bytes:
    """Encode one length-delimited (wire type 2) string field.

    Tag = (field << 3) | 2 — cross-check of the bytes this produces:
        f15 → 0x7A ; f16 → 0x82 0x01 ; f17 → 0x8A 0x01 ;
        f18 → 0x92 0x01 ; f19 → 0x9A 0x01
    """
    tag = (field << 3) | 0x02
    vb = value.encode("utf-8")
    return _encode_varint(tag) + _encode_varint(len(vb)) + vb


def _build_epic1a_batch_fixture(*, edges: tuple[str, ...] = ("edge-photosynthesis",)) -> bytes:
    """An AiAssistStarted carrying job_kind=batch + requested_count=3 +
    grounding=strict + a gs:// source blob + N target growth edges."""
    body = (
        b"\x12\x02AS"  # f2 assist_id="AS"
        b"\x40\x03"  # f8 requested_count=3 (varint, tag 0x40)
        + _ld_field(15, "batch")  # f15 job_kind
        + _ld_field(16, "strict")  # f16 grounding_mode
        + _ld_field(17, "gs://chora-creation-batch-uploads-dev/x.pdf")  # f17
        + _ld_field(18, "application/pdf")  # f18 source_mime_type
    )
    for e in edges:
        body += _ld_field(19, e)  # f19 target_growth_edges (repeated)
    return body


def test_decode_surfaces_epic1a_batch_grounding_fields() -> None:
    out = decode_ai_assist_started(_build_epic1a_batch_fixture())
    assert out["assist_id"] == "AS"
    assert out["requested_count"] == 3
    assert out["job_kind"] == "batch"
    assert out["grounding_mode"] == "strict"
    assert out["source_blob_uri"] == "gs://chora-creation-batch-uploads-dev/x.pdf"
    assert out["source_mime_type"] == "application/pdf"
    assert out["target_growth_edges"] == ["edge-photosynthesis"]


def test_decode_epic1a_repeated_target_growth_edges_accumulate() -> None:
    out = decode_ai_assist_started(_build_epic1a_batch_fixture(edges=("edge-a", "edge-b", "edge-c")))
    assert out["target_growth_edges"] == ["edge-a", "edge-b", "edge-c"]


def test_decode_epic1a_fields_absent_on_pre_1a_message() -> None:
    """A pre-1a message (the existing fixture, no f15-19) decodes with NONE of
    the batch/grounding keys present — byte/behaviour-compatible with the live
    single-candidate path (from_event defaults them)."""
    out = decode_ai_assist_started(_build_ai_assist_started_fixture())
    assert "job_kind" not in out
    assert "grounding_mode" not in out
    assert "source_blob_uri" not in out
    assert "source_mime_type" not in out
    assert "target_growth_edges" not in out


def test_decode_then_from_event_routes_batch_with_grounding() -> None:
    """End-to-end decoder→consumer contract: the decoded dict must satisfy
    AiAssistStartedPayload.from_event so the router sends it to the batch runner
    (job_kind=batch) and the grounding plugin gets its session keys. This is the
    gap the live write-walk exposed — the runner unit tests pass a flat dict
    directly and never exercised the proto decoder."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
        AiAssistStartedPayload,
    )

    payload = AiAssistStartedPayload.from_event(
        decode_ai_assist_started(_build_epic1a_batch_fixture(edges=("edge-x", "edge-y")))
    )
    assert payload.job_kind == "batch"  # → QGenRunnerRouter routes to batch
    assert payload.requested_count == 3  # → batch loop emits N candidates
    assert payload.grounding_mode == "strict"
    assert payload.source_blob_uri.startswith("gs://")
    assert payload.source_mime_type == "application/pdf"
    assert payload.target_growth_edges == ("edge-x", "edge-y")


def test_looks_like_binary_proto_detects_proto_first_byte() -> None:
    bz = b"\x0asomething"
    assert looks_like_binary_proto(bz) is True


def test_looks_like_binary_proto_treats_json_as_not_proto() -> None:
    assert looks_like_binary_proto(b'{"foo":"bar"}') is False
    assert looks_like_binary_proto(b"") is False


def test_decode_raises_on_truncated_varint() -> None:
    # 0x80 = MSB set, signaling continuation, but buffer ends → truncated
    with pytest.raises(ProtoDecodeError):
        decode_ai_assist_started(b"\x80")


def test_decode_raises_on_overrunning_length() -> None:
    # field 1 length-delimited, length 99 but only 2 bytes follow
    with pytest.raises(ProtoDecodeError):
        decode_ai_assist_started(b"\x0a\x63XY")


# -----------------------------------------------------------------------------
# ADR-195 WS7 (D7) — AiAssistStartedV2 compose model decode.
#
# The .v2 message DROPS job_kind (tag 15, now reserved) and ADDS the explicit
# compose model: operation (tag 23) / intent (tag 24) / input_kind (tag 25), all
# length-delimited strings. Tags 23-25 exceed 15 so their field-tag varint spans
# TWO bytes — the decoder's varint reader handles that, but the hand-built
# fixtures below must emit the multi-byte tag, so we encode tags/lengths
# programmatically rather than by magic byte. ONE decoder serves both versions:
# a v1 message carries no tags 23-25 (keys stay absent) and a v2 message carries
# no tag 15 (job_kind stays absent), with no field-number collision between them.
# -----------------------------------------------------------------------------


def _proto_varint(n: int) -> bytes:
    """Encode ``n`` as a base-128 varint (proto wire-format)."""
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld_bytes(field: int, body: bytes) -> bytes:
    """Encode a length-delimited (wire type 2) field carrying ``body``."""
    return _proto_varint((field << 3) | 2) + _proto_varint(len(body)) + body


def _ld_str(field: int, s: str) -> bytes:
    """Encode a length-delimited string field."""
    return _ld_bytes(field, s.encode("utf-8"))


def _build_ai_assist_started_v2_fixture(*, intent: str = "new_question", input_kind: str = "prompt") -> bytes:
    """A minimal binary AiAssistStartedV2 — the compose model on tags 23-25, NO
    job_kind (tag 15). Mirrors what chora-creation's v2 encoder emits."""
    return (
        _ld_bytes(1, _build_envelope())  # field 1 envelope
        + _ld_str(2, "AS")  # assist_id
        + _ld_str(3, "TEN")  # tenant_id
        + _ld_str(6, "mcq")  # content_type
        + _ld_str(7, "Generate questions")  # prompt
        + _ld_str(23, "compose")  # operation
        + _ld_str(24, intent)  # intent
        + _ld_str(25, input_kind)  # input_kind
    )


def test_decode_v2_surfaces_compose_model() -> None:
    """A .v2 message surfaces operation/intent/input_kind top-level (so the
    router can dispatch on them) and carries NO job_kind."""
    bz = _build_ai_assist_started_v2_fixture(intent="new_question", input_kind="source_files")
    out = decode_ai_assist_started(bz)

    assert out["assist_id"] == "AS"
    assert out["operation"] == "compose"
    assert out["intent"] == "new_question"
    assert out["input_kind"] == "source_files"
    # v2 drops the legacy discriminant entirely.
    assert "job_kind" not in out


def test_decode_v2_image_regen_intent() -> None:
    """The image_regen intent rides the same tag 24 — surfaced for the router."""
    bz = _build_ai_assist_started_v2_fixture(intent="image_regen", input_kind="prompt")
    out = decode_ai_assist_started(bz)
    assert out["intent"] == "image_regen"
    assert out["operation"] == "compose"


def test_decode_v1_carries_no_compose_model() -> None:
    """A v1 message (no tags 23-25) leaves the compose-model keys ABSENT so the
    router falls back to the legacy job_kind discriminant — byte/behaviour
    compatible during parallel-publish."""
    out = decode_ai_assist_started(_build_ai_assist_started_fixture())
    assert "operation" not in out
    assert "intent" not in out
    assert "input_kind" not in out


# -----------------------------------------------------------------------------
# CHO-1658 — existing_question_json (tag 26): the author's existing question
# content for intent=model_answer_fill, JSON-encoded. decode surfaces it as the
# parsed dict under input_obj["existing_question"] — what the
# reasoning_engine_executor reads into the qgen agent's author_* session keys.
# -----------------------------------------------------------------------------


def test_decode_v2_surfaces_existing_question_from_json() -> None:
    eq = (
        '{"stem":"Explain photosynthesis.",'
        '"oe_rubric":[{"criterion":"accuracy","weight":1.0}],'
        '"model_answer":"placeholder"}'
    )
    bz = _build_ai_assist_started_v2_fixture(intent="model_answer_fill") + _ld_str(26, eq)
    out = decode_ai_assist_started(bz)
    assert out["intent"] == "model_answer_fill"
    assert out["existing_question"] == {
        "stem": "Explain photosynthesis.",
        "oe_rubric": [{"criterion": "accuracy", "weight": 1.0}],
        "model_answer": "placeholder",
    }


def test_decode_v2_existing_question_absent_when_no_field26() -> None:
    """No tag 26 ⇒ the key stays ABSENT (byte/behaviour-compatible for every
    non-fill event — the executor's `or {}` then degrades safely)."""
    out = decode_ai_assist_started(_build_ai_assist_started_v2_fixture())
    assert "existing_question" not in out


def test_decode_v2_existing_question_malformed_json_ignored() -> None:
    """A malformed JSON string is DROPPED (defensive) rather than raising — a
    broken fill degrades to a free-generation pass, never a NACK / poison loop."""
    bz = _build_ai_assist_started_v2_fixture(intent="model_answer_fill") + _ld_str(26, "{not json")
    out = decode_ai_assist_started(bz)
    assert "existing_question" not in out
