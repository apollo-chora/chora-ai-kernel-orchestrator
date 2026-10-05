"""Hand-rolled proto3 wire encoder for ``chora.consumption.weakness.analyzed.v1``.

Why hand-rolled (same rationale as the qgen ``proto_wire_encoder``): the topic is
attached to a Pub/Sub Schema Registry schema with ``encoding=BINARY`` — a JSON
publish dead-letters with ``INVALID_BINARY_PROTO_MESSAGE`` (the binary-proto
trap). The orchestrator venv also has a protobuf gencode/runtime skew that makes
importing the generated ``_pb2`` at runtime raise, so we emit canonical wire
bytes by hand against the FLAT schema (single top-level message per Schema
Registry constraint):

* ``chora-contracts/proto/events-flat/consumption/weakness/analyzed.proto``

Kept SEPARATE from ``proto_wire_encoder.py`` (qgen-owned) so the concurrent
batch-qgen track and this one never edit the same file in the shared worktree.

Wire-format spec — https://protobuf.dev/programming-guides/encoding/. proto3
omits default scalars (empty string, 0, 0.0, false) — the Schema Registry
accepts canonical encodings only.
"""

from __future__ import annotations

import datetime as _dt
import json
import struct
from typing import Any

_WIRE_VARINT = 0
_WIRE_FIXED32 = 5
_WIRE_LEN = 2


class WeaknessProtoEncodeError(ValueError):
    """Raised when the envelope or body cannot be encoded."""


# --------------------------------------------------------------------------- #
# wire primitives
# --------------------------------------------------------------------------- #


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise WeaknessProtoEncodeError(f"negative varint not supported: {value}")
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
    """Append a ``float`` (proto wire-type 5, IEEE-754 little-endian fixed32).
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
        raise WeaknessProtoEncodeError("expected int, got bool")
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
            raise WeaknessProtoEncodeError(f"cannot parse {value!r} as int") from exc
    raise WeaknessProtoEncodeError(f"unsupported int type {type(value).__name__}")


def _coerce_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise WeaknessProtoEncodeError(f"cannot parse {value!r} as float") from exc


def _parse_rfc3339(value: str) -> _dt.datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise WeaknessProtoEncodeError(f"cannot parse {value!r} as RFC3339") from exc
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


def _encode_edge(edge: dict[str, Any]) -> bytes:
    """Encode one ``ExtractedGrowthEdge`` submessage (flat schema fields 1-7)."""
    out = bytearray()
    _append_string(out, 1, str(edge.get("concept_label", "")))
    _append_string(out, 2, str(edge.get("concept_key", "")))
    _append_string(out, 3, str(edge.get("category", "")))
    for tag in edge.get("tags") or []:
        _append_string(out, 4, str(tag))
    _append_float_field(out, 5, _coerce_float(edge.get("confidence")))
    _append_float_field(out, 6, _coerce_float(edge.get("strength")))
    _append_string(out, 7, str(edge.get("descriptor_json", "")))
    return bytes(out)


def _encode_output_selection(selection: dict[str, Any]) -> bytes:
    """Encode the nested ``OutputSelection`` submessage (flat schema fields 1-4;
    ADR-205 D5 / CHO-1966). Bool fields encode as varints and proto3 omits the
    false default, so an all-false selection yields empty bytes — which
    ``_append_len`` then drops, collapsing all-false to absent (the canonical,
    unambiguous "no opt-in" on the wire)."""
    out = bytearray()
    _append_varint_field(out, 1, 1 if selection.get("focused_dose") else 0)
    _append_varint_field(out, 2, 1 if selection.get("familiar_coaching") else 0)
    _append_varint_field(out, 3, 1 if selection.get("practice_test") else 0)
    _append_varint_field(out, 4, 1 if selection.get("study_aids") else 0)
    return bytes(out)


# --------------------------------------------------------------------------- #
# public encoder
# --------------------------------------------------------------------------- #


def encode_weakness_analyzed(envelope: dict[str, Any], body: dict[str, Any]) -> bytes:
    """Encode ``chora.consumption.v1.WeaknessAnalyzed`` (flat schema) to canonical
    proto3 wire bytes for a BINARY Pub/Sub topic.

    ``body`` keys: upload_id, tenant_id, learner_gcid, model_used,
    input_token_count, output_token_count, analyzed_at (RFC3339), ``edges``
    (a list of dicts with concept_label / concept_key / category / tags /
    confidence / strength / descriptor_json), and an optional ``output_selection``
    (a dict of focused_dose / familiar_coaching / practice_test / study_aids
    bools) which rides field 10 (ADR-205 D5 / CHO-1966)."""
    out = bytearray()
    _append_len(out, 1, _encode_envelope(envelope))
    _append_string(out, 2, str(body.get("upload_id", "")))
    _append_string(out, 3, str(body.get("tenant_id", "")))
    _append_string(out, 4, str(body.get("learner_gcid", "")))
    for edge in body.get("edges") or []:
        _append_len(out, 5, _encode_edge(edge))
    _append_string(out, 6, str(body.get("model_used", "")))
    _append_varint_field(out, 7, _coerce_int(body.get("input_token_count")))
    _append_varint_field(out, 8, _coerce_int(body.get("output_token_count")))
    analyzed_at = _parse_rfc3339(str(body.get("analyzed_at", "")))
    if analyzed_at is not None:
        _append_len(out, 9, _encode_timestamp(analyzed_at))
    # field 10 = OutputSelection submessage (ADR-205 D5 / CHO-1966). Empty or
    # all-false → _append_len drops it (default-off familiar_coaching), so the
    # live single-shot path (which never sets it) emits an identical message.
    _append_len(out, 10, _encode_output_selection(body.get("output_selection") or {}))
    return bytes(out)


def _encode_generated_output(output: dict[str, Any]) -> bytes:
    """Encode one ``GeneratedLearnerOutput`` submessage (flat schema fields 1-3).

    ``CrewOutputGenerator.generate`` returns ``{"type", "content", "metered"}``
    where ``content`` is a str for study_aids and a dict for practice_test. Both
    are JSON-encoded onto the single ``content_json`` string field so the
    consumer stores ONE uniform JSONB column and a future artifact shape does not
    force a Schema Registry revision on this bound topic.
    """
    out = bytearray()
    _append_string(out, 1, str(output.get("type", "")))
    _append_string(out, 2, json.dumps(output.get("content")))
    # proto3 omits a false bool, so an unmetered artifact collapses to absent.
    _append_varint_field(out, 3, 1 if output.get("metered") else 0)
    return bytes(out)


def encode_weakness_outputs_generated(envelope: dict[str, Any], body: dict[str, Any]) -> bytes:
    """Encode ``chora.consumption.v1.WeaknessOutputsGenerated`` (flat schema) to
    canonical proto3 wire bytes for a BINARY Pub/Sub topic (WS-7).

    ``body`` keys: upload_id, tenant_id, learner_gcid, generated_at (RFC3339),
    and ``outputs`` (the list ``CrewOutputGenerator.generate`` returns).

    An EMPTY ``outputs`` list is valid and still encodes: every kind screened out
    or soft-failed is a real outcome, and the event firing is what lets
    consumption distinguish "generated nothing" from "not generated yet".

    Raises ``ValueError`` when ``upload_id`` is missing: the consumer correlates
    the artifacts to their upload by that id, so an event without one is
    unroutable and must fail loud rather than land undeliverable.
    """
    upload_id = str(body.get("upload_id", "")).strip()
    if not upload_id:
        raise ValueError("encode_weakness_outputs_generated: body missing upload_id")
    out = bytearray()
    _append_len(out, 1, _encode_envelope(envelope))
    _append_string(out, 2, upload_id)
    _append_string(out, 3, str(body.get("tenant_id", "")))
    _append_string(out, 4, str(body.get("learner_gcid", "")))
    for output in body.get("outputs") or []:
        _append_len(out, 5, _encode_generated_output(output))
    generated_at = _parse_rfc3339(str(body.get("generated_at", "")))
    if generated_at is not None:
        _append_len(out, 6, _encode_timestamp(generated_at))
    return bytes(out)


__all__ = [
    "WeaknessProtoEncodeError",
    "encode_weakness_analyzed",
    "encode_weakness_outputs_generated",
]
