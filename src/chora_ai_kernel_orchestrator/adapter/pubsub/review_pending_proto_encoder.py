"""Hand-rolled proto3 wire encoder for ``chora.consumption.weakness.review_pending.v1``.

CHO-1973 Wave A. Same rationale as ``weakness_proto_encoder`` (the analyzed.v1
sibling): the topic carries a Pub/Sub Schema Registry ``encoding=BINARY`` schema —
a JSON publish dead-letters with ``INVALID_BINARY_PROTO_MESSAGE`` (the
binary-proto trap) — and the orchestrator venv has a protobuf gencode/runtime
skew that makes importing the generated ``_pb2`` raise. So we emit canonical wire
bytes by hand against the FLAT schema (one top-level message per Schema Registry
constraint):

* ``chora-contracts/proto/events-flat/consumption/weakness/review_pending.proto``

Kept SEPARATE from ``weakness_proto_encoder.py`` (the analyzed writer) +
``proto_wire_encoder.py`` (qgen) so concurrent worktree tracks never edit the same
file. The wire primitives are intentionally duplicated (the established
convention here) for self-containment.

Wire-format spec — https://protobuf.dev/programming-guides/encoding/. proto3 omits
default scalars (empty string, 0, 0.0, false); the Schema Registry accepts
canonical encodings only.
"""

from __future__ import annotations

import datetime as _dt
import struct
from typing import Any

_WIRE_VARINT = 0
_WIRE_FIXED32 = 5
_WIRE_LEN = 2


class ReviewPendingProtoEncodeError(ValueError):
    """Raised when the envelope or panel cannot be encoded."""


# --------------------------------------------------------------------------- #
# wire primitives
# --------------------------------------------------------------------------- #


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ReviewPendingProtoEncodeError(f"negative varint not supported: {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_tag(field: int, wire_type: int) -> bytes:
    return _encode_varint((field << 3) | wire_type)


def _append_string(out: bytearray, field: int, value: str) -> None:
    if not value:
        return
    encoded = value.encode("utf-8")
    out += _encode_tag(field, _WIRE_LEN)
    out += _encode_varint(len(encoded))
    out += encoded


def _append_varint_field(out: bytearray, field: int, value: int) -> None:
    if value == 0:
        return
    out += _encode_tag(field, _WIRE_VARINT)
    out += _encode_varint(value)


def _append_float_field(out: bytearray, field: int, value: float) -> None:
    """Append a ``float`` (proto wire-type 5, IEEE-754 little-endian fixed32);
    proto3 omits the 0.0 default."""
    if value == 0.0:
        return
    out += _encode_tag(field, _WIRE_FIXED32)
    out += struct.pack("<f", value)


def _append_len(out: bytearray, field: int, payload: bytes) -> None:
    if not payload:
        return
    out += _encode_tag(field, _WIRE_LEN)
    out += _encode_varint(len(payload))
    out += payload


# --------------------------------------------------------------------------- #
# coercion helpers
# --------------------------------------------------------------------------- #


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ReviewPendingProtoEncodeError("expected int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        if not value:
            return 0
        try:
            return int(value)
        except ValueError as exc:
            raise ReviewPendingProtoEncodeError(f"cannot parse {value!r} as int") from exc
    raise ReviewPendingProtoEncodeError(f"unsupported int type {type(value).__name__}")


def _coerce_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ReviewPendingProtoEncodeError(f"cannot parse {value!r} as float") from exc


def _parse_rfc3339(value: str) -> _dt.datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReviewPendingProtoEncodeError(f"cannot parse {value!r} as RFC3339") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.UTC)
    return dt.astimezone(_dt.UTC)


def _encode_timestamp(dt: _dt.datetime) -> bytes:
    out = bytearray()
    _append_varint_field(out, 1, int(dt.timestamp()))
    _append_varint_field(out, 2, dt.microsecond * 1000)
    return bytes(out)


# --------------------------------------------------------------------------- #
# nested-message encoders
# --------------------------------------------------------------------------- #


def _encode_envelope(envelope: dict[str, Any]) -> bytes:
    """Encode the nested ``Envelope`` submessage (chora.common.v1.EventEnvelope
    flattened layout — field numbers pinned to the live Schema Registry schema)."""
    out = bytearray()
    _append_string(out, 1, str(envelope.get("event_id", "")))
    _append_string(out, 2, str(envelope.get("idempotency_key", "")))
    _append_string(out, 3, str(envelope.get("tenant_id", "")))
    _append_string(out, 4, str(envelope.get("gcid", "")))
    occurred_at = _parse_rfc3339(str(envelope.get("occurred_at", "")))
    if occurred_at is not None:
        _append_len(out, 5, _encode_timestamp(occurred_at))
    published_at = _parse_rfc3339(str(envelope.get("published_at", "")))
    if published_at is not None:
        _append_len(out, 6, _encode_timestamp(published_at))
    _append_string(out, 7, str(envelope.get("traceparent", "")))
    _append_string(out, 8, str(envelope.get("tracestate", "")))
    _append_string(out, 9, str(envelope.get("source_project", "")))
    _append_string(out, 10, str(envelope.get("source_service", "")))
    _append_varint_field(out, 11, _coerce_int(envelope.get("schema_version")))
    _append_string(out, 12, str(envelope.get("correlation_id", "")))
    _append_string(out, 13, str(envelope.get("causation_id", "")))
    _append_string(out, 14, str(envelope.get("chora_imda_dimension", "")))
    _append_string(out, 15, str(envelope.get("imda_lifecycle_stage", "")))
    return bytes(out)


def _encode_familiar(familiar: dict[str, Any]) -> bytes:
    """Encode the nested ``ReviewFamiliar`` submessage (fields 1-3). An empty dict
    yields empty bytes → ``_append_len`` drops field 5 (no Familiar resolved)."""
    out = bytearray()
    _append_string(out, 1, str(familiar.get("familiar_id", "")))
    _append_string(out, 2, str(familiar.get("name", "")))
    _append_string(out, 3, str(familiar.get("species", "")))
    return bytes(out)


def _encode_proposed_edge(edge: dict[str, Any]) -> bytes:
    """Encode one ``ProposedGrowthEdge`` submessage (flat schema fields 1-6)."""
    out = bytearray()
    _append_string(out, 1, str(edge.get("proposed_edge_id", "")))
    _append_string(out, 2, str(edge.get("concept_label", "")))
    _append_string(out, 3, str(edge.get("summary", "")))
    for angle in edge.get("suggested_angles") or []:
        _append_string(out, 4, str(angle))
    _append_float_field(out, 5, _coerce_float(edge.get("strength")))
    _append_string(out, 6, str(edge.get("suggested_difficulty", "")))
    return bytes(out)


def _encode_candidate_struggle(struggle: dict[str, Any]) -> bytes:
    """Encode one ``CandidateStruggle`` submessage (flat schema fields 1-2)."""
    out = bytearray()
    _append_string(out, 1, str(struggle.get("concept_key", "")))
    _append_string(out, 2, str(struggle.get("concept_label", "")))
    return bytes(out)


def _encode_available_output(output: dict[str, Any]) -> bytes:
    """Encode one ``AvailableOutput`` submessage (kind=1 string, mana_price=2
    int64 varint, default_selected=3 bool). proto3 omits 0 price + false."""
    out = bytearray()
    _append_string(out, 1, str(output.get("kind", "")))
    _append_varint_field(out, 2, _coerce_int(output.get("mana_price")))
    _append_varint_field(out, 3, 1 if output.get("default_selected") else 0)
    return bytes(out)


# --------------------------------------------------------------------------- #
# public encoder
# --------------------------------------------------------------------------- #


def encode_weakness_review_pending(envelope: dict[str, Any], panel: dict[str, Any]) -> bytes:
    """Encode ``chora.consumption.v1.WeaknessReviewPending`` (flat schema) to
    canonical proto3 wire bytes for a BINARY Pub/Sub topic.

    ``panel`` keys: upload_id, tenant_id, learner_gcid, ``familiar`` (a dict of
    familiar_id / name / species — empty drops field 5), ``proposed_edges`` (list
    of dicts: proposed_edge_id / concept_label / summary / suggested_angles /
    strength / suggested_difficulty), ``candidate_struggles`` (list of dicts:
    concept_key / concept_label), ``available_outputs`` (list of dicts: kind /
    mana_price / default_selected), and an optional ``pending_at`` (RFC3339) on
    field 9.
    """
    out = bytearray()
    _append_len(out, 1, _encode_envelope(envelope))
    _append_string(out, 2, str(panel.get("upload_id", "")))
    _append_string(out, 3, str(panel.get("tenant_id", "")))
    _append_string(out, 4, str(panel.get("learner_gcid", "")))
    _append_len(out, 5, _encode_familiar(panel.get("familiar") or {}))
    for edge in panel.get("proposed_edges") or []:
        _append_len(out, 6, _encode_proposed_edge(edge))
    for struggle in panel.get("candidate_struggles") or []:
        _append_len(out, 7, _encode_candidate_struggle(struggle))
    for output in panel.get("available_outputs") or []:
        _append_len(out, 8, _encode_available_output(output))
    pending_at = _parse_rfc3339(str(panel.get("pending_at", "")))
    if pending_at is not None:
        _append_len(out, 9, _encode_timestamp(pending_at))
    return bytes(out)


__all__ = ["ReviewPendingProtoEncodeError", "encode_weakness_review_pending"]
