"""Minimal proto3 wire-format encoder for the qgen 2-agent crew's
terminal events (``chora.creation.ai_assist.completed.v1`` +
``chora.creation.ai_assist.refused.v1``).

Why hand-rolled
---------------
Both topics are attached to a Pub/Sub Schema Registry schema with
``encoding=BINARY``. JSON publishes fail with
``INVALID_BINARY_PROTO_MESSAGE`` at the Pub/Sub publish hop (5/5
outbox rows status=failed prior to this fix).

Rather than add the ``chora-contracts/gen/python`` package as a runtime
dependency (which forces a multi-file Dockerfile COPY + invalidates the
build cache), we mirror the ``proto_wire.py`` decoder by hand-rolling
the small subset of the proto3 wire-format the orchestrator needs to
emit canonical ``AiAssistCompleted`` + ``AiAssistRefused`` wire bytes.

Wire-format spec — https://protobuf.dev/programming-guides/encoding/

Schema references (the flat protos ARE the canonical schemas attached
to the live Pub/Sub topics — single top-level message per Schema
Registry constraint):

* ``chora-contracts/proto/events-flat/creation/ai_assist/completed.proto``
* ``chora-contracts/proto/events-flat/creation/ai_assist/refused.proto``

Both messages have a nested ``Envelope`` submessage at field 1 (same
shape as ``chora.common.v1.EventEnvelope``) + a nested ``Timestamp``
submessage shape (int64 seconds @ field 1, int32 nanos @ field 2).

Mirrors the Go implementation at
``services/chora-creation/internal/adapter/events/protomarshal/protomarshal.go``
which encodes ``AiAssistStarted`` for the inbound side of the same
crew's pipeline. Field numbers + wire types are pinned to the deployed
Schema Registry schemas.

The envelope dict shape consumed here matches what
``qgen_crew_publisher.py:_write`` builds prior to INSERT. The body dict
contains only the top-level payload scalar fields.

Per the proto3 spec, default scalar values (empty string, 0, false) are
omitted from the wire bytes (this is what protoc-generated code does
too). The Pub/Sub Schema Registry accepts canonical encodings only.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any


class ProtoEncodeError(ValueError):
    """Raised when the envelope or body cannot be encoded."""


class ProtoDecodeError(ValueError):
    """Raised when binary input fails to parse (truncated / overrun / bad wire
    type). Subclasses ``ValueError`` so subscribers catching the broad error
    (then NACKing the malformed message) still observe it."""


# -----------------------------------------------------------------------------
# Wire-format primitives.
# -----------------------------------------------------------------------------


def _encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a proto3 base-128 varint."""
    if value < 0:
        # proto3 int32 negative values use 10-byte two's-complement encoding.
        # The AiAssist*.v1 schemas never emit negative ints (mana_charged +
        # attempt_count are non-negative), so we raise loud rather than
        # silently mis-encode.
        raise ProtoEncodeError(
            f"negative varint not supported in qgen wire encoder: {value}",
        )
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
    """Encode a (field_number, wire_type) tag as a varint."""
    return _encode_varint((field << 3) | wire_type)


def _append_string(out: bytearray, field: int, value: str) -> None:
    """Append a string field (wire-type 2 length-delimited)."""
    if not value:
        return
    encoded = value.encode("utf-8")
    out += _encode_tag(field, 2)
    out += _encode_varint(len(encoded))
    out += encoded


def _append_varint_field(out: bytearray, field: int, value: int) -> None:
    """Append a varint field (wire-type 0). Skipped when value == 0 per
    proto3 default-value omission."""
    if value == 0:
        return
    out += _encode_tag(field, 0)
    out += _encode_varint(value)


def _append_bool_field(out: bytearray, field: int, value: bool) -> None:
    """Append a bool field as a varint (0 = false, 1 = true).

    proto3 omits ``false`` (the default) — only ``true`` is emitted.
    """
    if not value:
        return
    out += _encode_tag(field, 0)
    out += _encode_varint(1)


def _append_length_delimited(out: bytearray, field: int, payload: bytes) -> None:
    """Append a length-delimited field (wire-type 2). Skipped if payload
    is empty — proto3 default of an empty submessage is also omitted."""
    if not payload:
        return
    out += _encode_tag(field, 2)
    out += _encode_varint(len(payload))
    out += payload


# -----------------------------------------------------------------------------
# Coercion helpers.
# -----------------------------------------------------------------------------


def _coerce_int(value: Any, field_label: str) -> int:
    """Coerce ``value`` to a non-negative int. Raises on type mismatch."""
    if value is None:
        return 0
    if isinstance(value, bool):
        # bool is a subclass of int; reject silently mis-mapping bool→int.
        raise ProtoEncodeError(
            f"{field_label}: expected int, got bool",
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        # The publisher may pass schema_version as a string ("1") since
        # the existing envelope dict carries it that way. Accept + parse.
        if not value:
            return 0
        try:
            return int(value)
        except ValueError as exc:
            raise ProtoEncodeError(
                f"{field_label}: cannot parse {value!r} as int",
            ) from exc
    raise ProtoEncodeError(
        f"{field_label}: unsupported type {type(value).__name__}",
    )


def _coerce_bool(value: Any, field_label: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    raise ProtoEncodeError(
        f"{field_label}: unsupported type {type(value).__name__}",
    )


def _parse_rfc3339(value: str) -> _dt.datetime | None:
    """Parse an RFC3339 / RFC3339Nano timestamp string into an aware UTC
    datetime. Returns ``None`` for empty input.

    Accepts both ``2026-05-17T12:00:00+00:00`` (Python isoformat) and
    ``2026-05-17T12:00:00Z`` (RFC3339 spelling).
    """
    if not value:
        return None
    # ``fromisoformat`` accepts ``+00:00`` natively; ``Z`` requires fixup.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ProtoEncodeError(
            f"timestamp: cannot parse {value!r} as RFC3339",
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.UTC)
    return dt.astimezone(_dt.UTC)


def _encode_timestamp(dt: _dt.datetime) -> bytes:
    """Encode a ``google.protobuf.Timestamp`` submessage:

    * field 1 = int64 seconds (varint, wire-type 0)
    * field 2 = int32 nanos   (varint, wire-type 0)

    proto3 default-omits seconds==0 + nanos==0.
    """
    out = bytearray()
    seconds = int(dt.timestamp())
    # ``microsecond`` is 0..999_999; convert to nanos.
    nanos = dt.microsecond * 1000
    _append_varint_field(out, 1, seconds)
    _append_varint_field(out, 2, nanos)
    return bytes(out)


# -----------------------------------------------------------------------------
# Envelope encoder — shared between AiAssistCompleted + AiAssistRefused.
# -----------------------------------------------------------------------------


def _encode_envelope(envelope: dict[str, Any]) -> bytes:
    """Encode the nested ``Envelope`` submessage per the
    ``chora.common.v1.EventEnvelope`` flattened layout:

    1  string   event_id
    2  string   idempotency_key
    3  string   tenant_id
    4  string   gcid
    5  Timestamp occurred_at
    6  Timestamp published_at
    7  string   traceparent
    8  string   tracestate
    9  string   source_project
    10 string   source_service
    11 int32    schema_version
    12 string   correlation_id
    13 string   causation_id
    14 string   chora_imda_dimension
    15 string   imda_lifecycle_stage
    """
    out = bytearray()
    _append_string(out, 1, str(envelope.get("event_id", "")))
    _append_string(out, 2, str(envelope.get("idempotency_key", "")))
    _append_string(out, 3, str(envelope.get("tenant_id", "")))
    _append_string(out, 4, str(envelope.get("gcid", "")))

    occurred_at = _parse_rfc3339(str(envelope.get("occurred_at", "")))
    if occurred_at is not None:
        _append_length_delimited(out, 5, _encode_timestamp(occurred_at))
    published_at = _parse_rfc3339(str(envelope.get("published_at", "")))
    if published_at is not None:
        _append_length_delimited(out, 6, _encode_timestamp(published_at))

    _append_string(out, 7, str(envelope.get("traceparent", "")))
    _append_string(out, 8, str(envelope.get("tracestate", "")))
    _append_string(out, 9, str(envelope.get("source_project", "")))
    _append_string(out, 10, str(envelope.get("source_service", "")))
    _append_varint_field(
        out,
        11,
        _coerce_int(envelope.get("schema_version"), "schema_version"),
    )
    _append_string(out, 12, str(envelope.get("correlation_id", "")))
    _append_string(out, 13, str(envelope.get("causation_id", "")))
    _append_string(out, 14, str(envelope.get("chora_imda_dimension", "")))
    _append_string(out, 15, str(envelope.get("imda_lifecycle_stage", "")))
    return bytes(out)


def _encode_generation_summary(summary: dict[str, Any]) -> bytes:
    """Encode the nested ``GenerationSummary`` submessage (CHO-1819 — honest
    mixed-type / strict-shortfall accounting):

    1  int32             requested_total
    2  int32             generated_total
    3  map<string,int32> generated_per_type   (repeated MapEntry submessage)
    4  string            shortfall_reason

    ``generated_count`` (AiAssistCompleted field 3) MUST equal
    ``generated_total``; the publisher passes both. ``shortfall_reason`` is
    non-empty ONLY on a legal strict shortfall.
    """
    out = bytearray()
    _append_varint_field(
        out,
        1,
        _coerce_int(summary.get("requested_total"), "requested_total"),
    )
    _append_varint_field(
        out,
        2,
        _coerce_int(summary.get("generated_total"), "generated_total"),
    )
    per_type = summary.get("generated_per_type") or {}
    for key, value in per_type.items():
        entry = bytearray()
        _append_string(entry, 1, str(key))
        _append_varint_field(
            entry,
            2,
            _coerce_int(value, "generated_per_type value"),
        )
        _append_length_delimited(out, 3, bytes(entry))
    _append_string(out, 4, str(summary.get("shortfall_reason", "")))
    return bytes(out)


# -----------------------------------------------------------------------------
# Public encoders.
# -----------------------------------------------------------------------------


def encode_ai_assist_completed(
    envelope: dict[str, Any],
    body: dict[str, Any],
) -> bytes:
    """Encode a binary-proto ``chora.creation.v1.AiAssistCompleted``.

    Schema: ``chora-contracts/proto/events-flat/creation/ai_assist/completed.proto``:

    1  Envelope envelope
    2  string   assist_id
    3  int32    generated_count
    4  string   screening_decision
    5  string   screening_explanation
    6  string   model_used
    7  int32    input_token_count
    8  int32    output_token_count
    9  int32    mana_charged
    10 Timestamp completed_at
    11 string   candidate_payload_json
    12 string   pipeline_trace_json
    13 bool     quality_warning
    14 int32    attempt_count
    15 string   critic_notes

    The publisher's ``body`` dict only carries a subset of these fields
    (assist_id, candidate_payload_json, pipeline_trace_json,
    quality_warning, attempt_count, critic_notes, mana_charged). The
    others (generated_count, screening_decision, screening_explanation,
    model_used, input_token_count, output_token_count) default to 0/""
    and proto3-default-omit per spec.

    ``completed_at`` (field 10) is synthesised from
    ``envelope["occurred_at"]`` since the publisher does not pass it
    explicitly.
    """
    out = bytearray()

    env_bytes = _encode_envelope(envelope)
    _append_length_delimited(out, 1, env_bytes)

    _append_string(out, 2, str(body.get("assist_id", "")))
    _append_varint_field(
        out,
        3,
        _coerce_int(body.get("generated_count"), "generated_count"),
    )
    _append_string(out, 4, str(body.get("screening_decision", "")))
    _append_string(out, 5, str(body.get("screening_explanation", "")))
    _append_string(out, 6, str(body.get("model_used", "")))
    _append_varint_field(
        out,
        7,
        _coerce_int(body.get("input_token_count"), "input_token_count"),
    )
    _append_varint_field(
        out,
        8,
        _coerce_int(body.get("output_token_count"), "output_token_count"),
    )
    _append_varint_field(
        out,
        9,
        _coerce_int(body.get("mana_charged"), "mana_charged"),
    )

    completed_at = _parse_rfc3339(str(envelope.get("occurred_at", "")))
    if completed_at is not None:
        _append_length_delimited(out, 10, _encode_timestamp(completed_at))

    _append_string(out, 11, str(body.get("candidate_payload_json", "")))
    _append_string(out, 12, str(body.get("pipeline_trace_json", "")))
    _append_bool_field(
        out,
        13,
        _coerce_bool(body.get("quality_warning"), "quality_warning"),
    )
    _append_varint_field(
        out,
        14,
        _coerce_int(body.get("attempt_count"), "attempt_count"),
    )
    _append_string(out, 15, str(body.get("critic_notes", "")))

    # Honest generation summary (field 16, CHO-1819) — proto3-omitted when the
    # publisher passes no summary (legacy single-candidate path).
    generation_summary = body.get("generation_summary")
    if generation_summary:
        _append_length_delimited(
            out,
            16,
            _encode_generation_summary(generation_summary),
        )

    return bytes(out)


def encode_ai_assist_progress(
    envelope: dict[str, Any],
    body: dict[str, Any],
) -> bytes:
    """Encode a binary-proto ``chora.creation.v1.AiAssistProgress`` (the
    mid-run live-trace event; emitted once per qgen graph node).

    Schema: ``chora-contracts/proto/events-flat/creation/ai_assist/progress.proto``:

    1  Envelope  envelope
    2  string    assist_id
    3  string    tenant_id
    4  string    author_gcid
    5  string    pipeline_trace_json   (cumulative-so-far)
    6  int32     step_index            (monotonic == len(trace))
    7  string    step_name
    8  string    step_status
    9  Timestamp progress_at

    ``progress_at`` (field 9) is synthesised from ``envelope["occurred_at"]``,
    mirroring ``completed_at`` on ``AiAssistCompleted``.
    """
    out = bytearray()

    env_bytes = _encode_envelope(envelope)
    _append_length_delimited(out, 1, env_bytes)

    _append_string(out, 2, str(body.get("assist_id", "")))
    _append_string(out, 3, str(body.get("tenant_id", "")))
    _append_string(out, 4, str(body.get("author_gcid", "")))
    _append_string(out, 5, str(body.get("pipeline_trace_json", "")))
    _append_varint_field(
        out,
        6,
        _coerce_int(body.get("step_index"), "step_index"),
    )
    _append_string(out, 7, str(body.get("step_name", "")))
    _append_string(out, 8, str(body.get("step_status", "")))

    progress_at = _parse_rfc3339(str(envelope.get("occurred_at", "")))
    if progress_at is not None:
        _append_length_delimited(out, 9, _encode_timestamp(progress_at))

    return bytes(out)


def encode_ai_assist_chunk_completed(
    envelope: dict[str, Any],
    body: dict[str, Any],
) -> bytes:
    """Encode a binary-proto ``chora.creation.v1.AiAssistChunkCompleted``
    (ADR-251 D5, CHO-2398: one per finished chunk of a multi-chunk set job).

    Schema: ``chora-contracts/proto/events-flat/creation/ai_assist/
    chunk_completed.proto``:

    1  Envelope  envelope
    2  string    assist_id
    3  string    tenant_id
    4  string    author_gcid
    5  int32     chunk_index              (0-based; the consumer's monotonic cursor)
    6  int32     chunk_count
    7  string    candidates_payload_json  (published shape; specs stripped)
    8  int32     candidate_count
    9  Timestamp chunk_completed_at
    10 int32     warned_count
    11 int32     images_rendered
    12 int32     images_dropped
    13 int32     images_failed
    14 int32     images_skipped

    ``chunk_completed_at`` is synthesised from ``envelope["occurred_at"]``,
    mirroring ``progress_at`` on ``AiAssistProgress``.
    """
    out = bytearray()

    env_bytes = _encode_envelope(envelope)
    _append_length_delimited(out, 1, env_bytes)

    _append_string(out, 2, str(body.get("assist_id", "")))
    _append_string(out, 3, str(body.get("tenant_id", "")))
    _append_string(out, 4, str(body.get("author_gcid", "")))
    _append_varint_field(
        out,
        5,
        _coerce_int(body.get("chunk_index"), "chunk_index"),
    )
    _append_varint_field(
        out,
        6,
        _coerce_int(body.get("chunk_count"), "chunk_count"),
    )
    _append_string(out, 7, str(body.get("candidates_payload_json", "")))
    _append_varint_field(
        out,
        8,
        _coerce_int(body.get("candidate_count"), "candidate_count"),
    )

    chunk_completed_at = _parse_rfc3339(str(envelope.get("occurred_at", "")))
    if chunk_completed_at is not None:
        _append_length_delimited(out, 9, _encode_timestamp(chunk_completed_at))

    _append_varint_field(
        out,
        10,
        _coerce_int(body.get("warned_count"), "warned_count"),
    )
    _append_varint_field(
        out,
        11,
        _coerce_int(body.get("images_rendered"), "images_rendered"),
    )
    _append_varint_field(
        out,
        12,
        _coerce_int(body.get("images_dropped"), "images_dropped"),
    )
    _append_varint_field(
        out,
        13,
        _coerce_int(body.get("images_failed"), "images_failed"),
    )
    _append_varint_field(
        out,
        14,
        _coerce_int(body.get("images_skipped"), "images_skipped"),
    )

    return bytes(out)


def encode_ai_assist_refused(
    envelope: dict[str, Any],
    body: dict[str, Any],
) -> bytes:
    """Encode a binary-proto ``chora.creation.v1.AiAssistRefused``.

    Schema: ``chora-contracts/proto/events-flat/creation/ai_assist/refused.proto``:

    1  Envelope envelope
    2  string   assist_id
    3  string   refusal_reason
    4  string   model_armor_verdict
    5  string   refusing_agent_id
    6  string   user_facing_message
    7  int32    mana_charged
    8  Timestamp refused_at
    9  string   last_candidate_payload_json
    10 int32    attempt_count

    ``refused_at`` (field 8) is synthesised from
    ``envelope["occurred_at"]`` since the publisher does not pass it
    explicitly. ``refusing_agent_id`` (field 5) is not currently passed
    by the publisher → default-omit.
    """
    out = bytearray()

    env_bytes = _encode_envelope(envelope)
    _append_length_delimited(out, 1, env_bytes)

    _append_string(out, 2, str(body.get("assist_id", "")))
    _append_string(out, 3, str(body.get("refusal_reason", "")))
    _append_string(out, 4, str(body.get("model_armor_verdict", "")))
    _append_string(out, 5, str(body.get("refusing_agent_id", "")))
    _append_string(out, 6, str(body.get("user_facing_message", "")))
    _append_varint_field(
        out,
        7,
        _coerce_int(body.get("mana_charged"), "mana_charged"),
    )

    refused_at = _parse_rfc3339(str(envelope.get("occurred_at", "")))
    if refused_at is not None:
        _append_length_delimited(out, 8, _encode_timestamp(refused_at))

    _append_string(out, 9, str(body.get("last_candidate_payload_json", "")))
    _append_varint_field(
        out,
        10,
        _coerce_int(body.get("attempt_count"), "attempt_count"),
    )

    return bytes(out)


# -----------------------------------------------------------------------------
# Observability encoders (ADR-167 — JSON→Protobuf wire-format migration).
#
# Both topics are Pub/Sub Schema-Registry-bound with encoding=BINARY, so the
# JSON the legacy outbox writers produced was rejected at the publish hop.
# These encoders emit canonical proto3 wire bytes per the generated schemas:
#
#   chora-contracts/proto/events/observability/agent_decision.proto
#   chora-contracts/proto/events/observability/token_usage.proto
#
# Field numbers + wire types are pinned to those schemas; the Go consumers
# (chora-governance + chora-observability, ADR-167 Phase 2) decode these
# exact bytes with the generated Go bindings.
# -----------------------------------------------------------------------------


# DECISION_KIND_CRITIQUE enum value per agent_decision.proto — the qgen
# quality gate is a critique decision (ADR-167 canonical mapping).
DECISION_KIND_CRITIQUE = 4


def _append_map_entry(out: bytearray, field: int, key: str, value: str) -> None:
    """Append one proto3 map<string,string> entry at ``field``.

    A map field is encoded as a repeated message; each entry is a
    submessage with ``key`` at field 1 + ``value`` at field 2. proto3 omits
    an entry whose key is empty (the encoder never emits a keyless pair).
    Empty values ARE emitted (an explicit "" value is meaningful for some
    keys) but here every key we write has a non-empty stringified value.
    """
    if not key:
        return
    entry = bytearray()
    _append_string(entry, 1, key)
    # Value field 2 — emit even when the string is empty so the key is not
    # silently dropped (a present key with "" value round-trips correctly).
    entry += _encode_tag(2, 2)
    encoded = value.encode("utf-8")
    entry += _encode_varint(len(encoded))
    entry += encoded
    _append_length_delimited(out, field, bytes(entry))


def encode_agent_decision_logged(
    envelope: dict[str, Any],
    body: dict[str, Any],
) -> bytes:
    """Encode a binary-proto ``chora.observability.v1.AgentDecisionLogged``.

    Schema — ``chora-contracts/proto/events/observability/agent_decision.proto``:

    1  Envelope envelope
    2  string   decision_id          (== assist_id)
    3  string   invocation_id        (== assist_id)
    4  string   agid                 (= body.agid — one of qgen_question /
                                       qgen_critic / oe_evaluator / oe_moderator;
                                       NEVER the legacy "qgen_crew", which matched
                                       no /o/agents registry tile)
    5  string   prompt_template_id   (omitted — empty)
    6  string   model_id             (body.model_id if present)
    7  DecisionKind decision_kind    (= DECISION_KIND_CRITIQUE)
    8  string   input_summary        (omitted — empty)
    9  string   output_summary       (= critic_notes)
    10 float    confidence           (omitted — wire-type 5; not used)
    11 Timestamp decided_at          (= occurred_at)
    12 string   crew_name
    13 string   crew_id
    14 bool     is_resume
    15 bool     is_eval_run
    16 string   adapter_version
    17 string   guardrail_outcome
    18 int64    prompt_tokens
    19 int64    completion_tokens
    20 int64    cached_tokens
    21 map<string,string> attributes  (qgen-specific detail as string pairs)

    Per the ADR-167 canonical mapping the GENERIC proto core stays
    cross-agent; the per-agent verdict + counts + citation hashes ride the
    attributes map: ``{"decision", "attempt_count", "max_retries",
    "quality_warning", "question_type", "input_hash", "output_hash"}`` (the
    hashes are sha256 hex over the agent's input/output — PII-safe citation,
    raw content never carried). Per ADR-197 M-A.3 the prompt-shaping condition
    discriminants ALSO ride this map under prefixed keys
    ``prompt_conditions.<key>`` (e.g. ``prompt_conditions.intent`` =
    ``new_question``) so the durable record carries the same discriminants the
    live agent span stamps — no proto change, additive within v1. ``agid`` (field 4) is read from ``body.agid`` so the
    producer attributes each decision to the concrete agent that made it
    (qgen_question / qgen_critic / oe_evaluator / oe_moderator) — the legacy
    hardcoded ``"qgen_crew"`` matched no /o/agents tile and left every tile
    reading 0.
    """
    out = bytearray()

    env_bytes = _encode_envelope(envelope)
    _append_length_delimited(out, 1, env_bytes)

    assist_id = str(body.get("assist_id", ""))
    _append_string(out, 2, assist_id)  # decision_id
    _append_string(out, 3, assist_id)  # invocation_id
    # agid (field 4) — the per-agent registry id, read from the body. The
    # producer (AgentDecisionLogOutboxWriter) requires a non-empty agid so a
    # blank never reaches here; the fallback is defensive only and is NEVER
    # "qgen_crew" (the legacy hardcode that matched no /o/agents tile).
    _append_string(out, 4, str(body.get("agid", "")))
    _append_string(out, 6, str(body.get("model_id", "")))
    _append_varint_field(out, 7, DECISION_KIND_CRITIQUE)
    _append_string(out, 9, str(body.get("critic_notes", "")))

    # decided_at (field 11) == occurred_at — the per-decision clock.
    decided_at = _parse_rfc3339(str(envelope.get("occurred_at", "")))
    if decided_at is not None:
        _append_length_delimited(out, 11, _encode_timestamp(decided_at))

    # Extension fields 12-20 (additive within v1).
    _append_string(out, 12, str(body.get("crew_name", "")))
    _append_string(out, 13, str(body.get("crew_id", "")))
    _append_bool_field(out, 14, _coerce_bool(body.get("is_resume"), "is_resume"))
    _append_bool_field(out, 15, _coerce_bool(body.get("is_eval_run"), "is_eval_run"))
    _append_string(out, 16, str(body.get("adapter_version", "")))
    _append_string(out, 17, str(body.get("guardrail_outcome", "")))
    _append_varint_field(out, 18, _coerce_int(body.get("prompt_tokens"), "prompt_tokens"))
    _append_varint_field(out, 19, _coerce_int(body.get("completion_tokens"), "completion_tokens"))
    _append_varint_field(out, 20, _coerce_int(body.get("cached_tokens"), "cached_tokens"))

    # attributes map (field 21) — qgen-specific detail as string pairs.
    # Stable insertion order so the wire bytes are deterministic per input.
    # A blank decision is omitted (proto3 map semantics); the consumer's
    # validateAgentDecision rejects a missing attributes[decision] loudly.
    decision = str(body.get("decision", ""))
    if decision:
        _append_map_entry(out, 21, "decision", decision)
    if "attempt_count" in body:
        _append_map_entry(
            out,
            21,
            "attempt_count",
            str(_coerce_int(body.get("attempt_count"), "attempt_count")),
        )
    if "max_retries" in body:
        _append_map_entry(
            out,
            21,
            "max_retries",
            str(_coerce_int(body.get("max_retries"), "max_retries")),
        )
    if "quality_warning" in body:
        qw = _coerce_bool(body.get("quality_warning"), "quality_warning")
        _append_map_entry(out, 21, "quality_warning", "true" if qw else "false")
    # question_type ("mcq" | "oe") — the O+ consumer reads it from the
    # attributes map to split qgen tiles by content type. proto3 omits an
    # empty entry, so a blank question_type is simply not emitted.
    question_type = str(body.get("question_type", ""))
    if question_type:
        _append_map_entry(out, 21, "question_type", question_type)

    # Citation hashes (sha256 hex over the agent's input/output) — the O+
    # reasoning-panel PII-safe citation (IMDA D2). Raw content is NEVER carried;
    # only the one-way hashes ride here. The chora-observability consumer reads
    # them into ReasoningSummary.{InputHash,OutputHash} so the panel renders a
    # real citation instead of the all-zeros sentinel. proto3 omits empty
    # entries, so a blank hash is simply not emitted (consumer keeps the zero
    # sentinel, same as the pre-citation behaviour).
    input_hash = str(body.get("input_hash", ""))
    if input_hash:
        _append_map_entry(out, 21, "input_hash", input_hash)
    output_hash = str(body.get("output_hash", ""))
    if output_hash:
        _append_map_entry(out, 21, "output_hash", output_hash)

    # ADR-197 M-A.3 — prompt-shaping condition discriminants ride the SAME
    # field-21 attributes map under prefixed keys ``prompt_conditions.<key>`` so
    # the durable record carries the exact discriminants the live agent span
    # already stamps (no proto change — additive within v1). The runner builds
    # the map (mirroring the Go QuestionConditions/CriticConditions/Evaluator
    # Conditions/ModeratorConditions extractors) and omits blanks/false there,
    # so each entry here is a non-empty stringified value. Iterated in the dict's
    # insertion order → deterministic wire bytes per input. A non-dict / absent
    # value emits nothing (proto3 map default; pre-M-A.3 callers unaffected).
    prompt_conditions = body.get("prompt_conditions")
    if isinstance(prompt_conditions, dict):
        for cond_key, cond_value in prompt_conditions.items():
            _append_map_entry(
                out,
                21,
                f"prompt_conditions.{cond_key}",
                str(cond_value),
            )

    return bytes(out)


# -----------------------------------------------------------------------------
# Governance audit encoder (ADR-197 M-C.1 — prompt-override activation audit).
#
# The chora.governance.audit.recorded.v1 topic is Schema-Registry-bound with
# encoding=BINARY, so the JSON a naive outbox writer would produce is rejected
# at the publish hop (same as the agent_decision topic). This encoder emits
# canonical proto3 wire bytes per the generated schema:
#
#   chora-contracts/proto/events/governance/audit.proto  message AuditEntryRecorded
#   chora-contracts/proto/events-flat/governance/audit/recorded.proto
#
# The Go consumer (chora-governance ingest) decodes these exact bytes with the
# generated bindings.
# -----------------------------------------------------------------------------


# AuditResult enum values per audit.proto. ALLOWED is the canonical result for a
# successful activation; UNSPECIFIED (0) is treated as a data-integrity error by
# the governance projector, so the encoder never emits a bare 0.
AUDIT_RESULT_UNSPECIFIED = 0
AUDIT_RESULT_ALLOWED = 1
AUDIT_RESULT_DENIED = 2
AUDIT_RESULT_ANOMALY = 3


def encode_audit_entry_recorded(
    envelope: dict[str, Any],
    body: dict[str, Any],
) -> bytes:
    """Encode a binary-proto ``chora.governance.v1.AuditEntryRecorded``.

    Schema — ``chora-contracts/proto/events/governance/audit.proto``:

    1  Envelope     envelope
    2  string       audit_id              (UUIDv7 — append-only entry id)
    3  string       actor_gcid            (GCID/AGID of the actor)
    4  string       target_resource_uri   (e.g. chora.ai_kernel/prompt_plan:<id>)
    5  string       action                (UPPER_SNAKE_CASE verb, e.g. ACTIVATE)
    6  AuditResult  result                (ALLOWED / DENIED / ANOMALY)
    7  string       annotation            (optional free text)
    8  string       source_ip             (empty for system-emitted entries)
    9  string       user_agent            (empty for system / agent entries)
    10 Timestamp    occurred_at

    ``result`` defaults to ``AUDIT_RESULT_ALLOWED`` when the body omits it — a
    bare 0 (UNSPECIFIED) would be a data-integrity signal to the projector, so
    we never emit it implicitly. ``occurred_at`` (field 10) prefers
    ``body['occurred_at']`` and falls back to ``envelope['occurred_at']``.
    """
    out = bytearray()

    env_bytes = _encode_envelope(envelope)
    _append_length_delimited(out, 1, env_bytes)

    _append_string(out, 2, str(body.get("audit_id", "")))
    _append_string(out, 3, str(body.get("actor_gcid", "")))
    _append_string(out, 4, str(body.get("target_resource_uri", "")))
    _append_string(out, 5, str(body.get("action", "")))

    # result — default to ALLOWED so the success path always carries a non-zero
    # enum (proto3 omits 0; the consumer rejects a missing/UNSPECIFIED result).
    result = body.get("result")
    if result is None:
        result = AUDIT_RESULT_ALLOWED
    _append_varint_field(out, 6, _coerce_int(result, "result"))

    _append_string(out, 7, str(body.get("annotation", "")))
    _append_string(out, 8, str(body.get("source_ip", "")))
    _append_string(out, 9, str(body.get("user_agent", "")))

    occurred_at = _parse_rfc3339(str(body.get("occurred_at", envelope.get("occurred_at", ""))))
    if occurred_at is not None:
        _append_length_delimited(out, 10, _encode_timestamp(occurred_at))

    return bytes(out)


# -----------------------------------------------------------------------------
# Governance audit DECODER (ADR-197 M-C.2 — the HITL approval round-trip).
#
# The O+ approver's decision lands back as a binary-proto AuditEntryRecorded on
# chora.governance.audit.recorded.v1. The orchestrator's audit-recorded consumer
# decodes it with this function (the inverse of ``encode_audit_entry_recorded``).
# Hand-rolled for the same reason as the encoder + proto_wire.py decoder: avoid
# pulling chora-contracts/gen/python as a runtime dep. Permissive — unknown
# fields are skipped (forward-compat); a malformed buffer raises ProtoDecodeError
# so the consumer NACKs.
# -----------------------------------------------------------------------------


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Read one base-128 varint at ``offset``. Returns (value, new_offset)."""
    result = 0
    shift = 0
    while offset < len(data):
        b = data[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7
        if shift > 63:
            raise ProtoDecodeError("varint exceeds 64 bits")
    raise ProtoDecodeError("varint truncated")


def _read_length_delimited(data: bytes, offset: int) -> tuple[bytes, int]:
    length, offset = _read_varint(data, offset)
    if offset + length > len(data):
        raise ProtoDecodeError("length-delimited field overruns buffer")
    return data[offset : offset + length], offset + length


def _skip_field(data: bytes, offset: int, wire_type: int) -> int:
    if wire_type == 0:  # varint
        _, offset = _read_varint(data, offset)
    elif wire_type == 1:  # 64-bit fixed
        offset += 8
    elif wire_type == 2:  # length-delimited
        _, offset = _read_length_delimited(data, offset)
    elif wire_type == 5:  # 32-bit fixed
        offset += 4
    else:
        raise ProtoDecodeError(f"unsupported wire type {wire_type}")
    return offset


def _decode_timestamp_iso(data: bytes) -> str:
    """Decode a ``google.protobuf.Timestamp`` submessage (seconds @ 1, nanos @ 2)
    back to an RFC3339 UTC string. Empty (0/0) -> ``""``."""
    seconds = 0
    nanos = 0
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0 and field == 1:
            seconds, offset = _read_varint(data, offset)
        elif wire_type == 0 and field == 2:
            nanos, offset = _read_varint(data, offset)
        else:
            offset = _skip_field(data, offset, wire_type)
    if seconds == 0 and nanos == 0:
        return ""
    dt = _dt.datetime.fromtimestamp(seconds, tz=_dt.UTC).replace(microsecond=nanos // 1000)
    return dt.isoformat()


def _decode_envelope_fields(data: bytes) -> dict[str, str]:
    """Decode the EventEnvelope submessage to the string fields the consumer's
    dedup + trace plumbing needs (mirrors ``proto_wire._decode_envelope`` plus
    source_project/source_service for completeness)."""
    keys = {
        1: "event_id",
        2: "idempotency_key",
        3: "tenant_id",
        4: "gcid",
        7: "traceparent",
        8: "tracestate",
        9: "source_project",
        10: "source_service",
    }
    env: dict[str, str] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 2 and field in keys:
            buf, offset = _read_length_delimited(data, offset)
            env[keys[field]] = buf.decode("utf-8")
        else:
            offset = _skip_field(data, offset, wire_type)
    return env


def decode_audit_entry_recorded(payload_bytes: bytes) -> dict[str, Any]:
    """Decode a binary-proto ``chora.governance.v1.AuditEntryRecorded`` into a
    flat dict (the inverse of :func:`encode_audit_entry_recorded`).

    Output keys: ``audit_id`` / ``actor_gcid`` / ``target_resource_uri`` /
    ``action`` / ``result`` (int enum) / ``annotation`` / ``source_ip`` /
    ``user_agent`` / ``occurred_at`` (RFC3339 or "") plus the nested
    ``envelope`` dict and the envelope's ``tenant_id`` / ``gcid`` /
    ``traceparent`` / ``tracestate`` surfaced top-level for the consumer.
    """
    data = bytes(payload_bytes)
    out: dict[str, Any] = {
        "audit_id": "",
        "actor_gcid": "",
        "target_resource_uri": "",
        "action": "",
        "result": 0,
        "annotation": "",
        "source_ip": "",
        "user_agent": "",
        "occurred_at": "",
        "envelope": {},
        "tenant_id": "",
        "gcid": "",
        "traceparent": "",
        "tracestate": "",
    }
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 2:
            buf, offset = _read_length_delimited(data, offset)
            if field == 1:
                env = _decode_envelope_fields(buf)
                out["envelope"] = env
                out["tenant_id"] = env.get("tenant_id", "")
                out["gcid"] = env.get("gcid", "")
                out["traceparent"] = env.get("traceparent", "")
                out["tracestate"] = env.get("tracestate", "")
            elif field == 2:
                out["audit_id"] = buf.decode("utf-8")
            elif field == 3:
                out["actor_gcid"] = buf.decode("utf-8")
            elif field == 4:
                out["target_resource_uri"] = buf.decode("utf-8")
            elif field == 5:
                out["action"] = buf.decode("utf-8")
            elif field == 7:
                out["annotation"] = buf.decode("utf-8")
            elif field == 8:
                out["source_ip"] = buf.decode("utf-8")
            elif field == 9:
                out["user_agent"] = buf.decode("utf-8")
            elif field == 10:
                out["occurred_at"] = _decode_timestamp_iso(buf)
            # other length-delimited fields skipped (forward-compat)
        elif wire_type == 0:
            value, offset = _read_varint(data, offset)
            if field == 6:
                out["result"] = value
        else:
            offset = _skip_field(data, offset, wire_type)
    return out


__all__ = [
    "AUDIT_RESULT_ALLOWED",
    "AUDIT_RESULT_ANOMALY",
    "AUDIT_RESULT_DENIED",
    "AUDIT_RESULT_UNSPECIFIED",
    "DECISION_KIND_CRITIQUE",
    "ProtoDecodeError",
    "ProtoEncodeError",
    "decode_audit_entry_recorded",
    "encode_agent_decision_logged",
    "encode_ai_assist_completed",
    "encode_ai_assist_progress",
    "encode_ai_assist_refused",
    "encode_audit_entry_recorded",
]
