"""Unit tests for ``proto_wire_encoder`` — the binary-proto encoder for
``chora.creation.ai_assist.completed.v1`` + ``chora.creation.ai_assist.refused.v1``.

Verification strategy
---------------------
Rather than depend on ``google.protobuf`` runtime + the generated Python
bindings (which would force a Dockerfile path-COPY refactor), we mirror
the Go ``walkTopLevelTags`` validation pattern: parse the encoder's
output bytes back with a small wire-format walker that records each
top-level (field_number, wire_type) it sees + the decoded scalar /
length-delimited contents. The tests then assert the expected fields
appear with the expected wire types + values.

This is sufficient to prove the encoder produces canonical proto3 wire
bytes that the Pub/Sub Schema Registry will accept — Schema Registry
validates per-field-number, per-wire-type, per-length-delimited
sub-message shape, which is exactly what the walker checks.

Reference patterns:

* Wire format spec — https://protobuf.dev/programming-guides/encoding/
* Symmetric decoder — ``proto_wire.py`` (decodes AiAssistStarted, a
  sibling message with the same envelope shape)
* Symmetric Go encoder — ``services/chora-creation/internal/adapter/
  events/protomarshal/protomarshal.go``
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    ProtoEncodeError,
    encode_ai_assist_completed,
    encode_ai_assist_refused,
)

# -----------------------------------------------------------------------------
# Tiny wire-format walker (read-only, mirrors ``walkTopLevelTags`` Go test util).
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


def _read_length_delimited(data: bytes, offset: int) -> tuple[bytes, int]:
    length, offset = _read_varint(data, offset)
    return data[offset : offset + length], offset + length


def _walk(data: bytes) -> list[dict[str, Any]]:
    """Parse ``data`` as a sequence of top-level proto3 fields. Returns a
    list of ``{"field": int, "wire": int, "value": ...}`` entries in the
    order they appear on the wire.

    Value decoding:
      * wire 0 (varint)            → int
      * wire 2 (length-delimited)  → bytes
      * other wire types raise (not used in AiAssist*.v1 messages)
    """
    out: list[dict[str, Any]] = []
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire = tag & 0x07
        if wire == 0:
            value, offset = _read_varint(data, offset)
            out.append({"field": field, "wire": 0, "value": value})
        elif wire == 2:
            buf, offset = _read_length_delimited(data, offset)
            out.append({"field": field, "wire": 2, "value": buf})
        else:
            raise ValueError(f"unsupported wire type {wire} at field {field}")
    return out


def _by_field(walked: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Flatten walker output to {field_number: entry}. Last-write-wins,
    which is the proto3 semantics for scalar fields anyway."""
    return {entry["field"]: entry for entry in walked}


# -----------------------------------------------------------------------------
# Envelope fixture — mirrors what ``qgen_crew_publisher._write`` builds.
# -----------------------------------------------------------------------------


def _make_envelope(**overrides: Any) -> dict[str, Any]:
    now_iso = _dt.datetime(2026, 5, 17, 12, 0, 0, tzinfo=_dt.UTC).isoformat()
    env: dict[str, Any] = {
        "event_id": "01970000-aaaa-7000-8000-000000000001",
        "idempotency_key": "ai_assist.completed.assist-1",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "gcid": "01970000-0000-7000-9000-000000000001",
        "occurred_at": now_iso,
        "published_at": now_iso,
        "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        "tracestate": "vendor=chora",
        "source_project": "chora-489812",
        "source_service": "chora-ai-kernel-orchestrator",
        "schema_version": "1",
        "chora_imda_dimension": "accountability",
    }
    env.update(overrides)
    return env


# -----------------------------------------------------------------------------
# AiAssistCompleted — field-by-field assertions.
# -----------------------------------------------------------------------------


class TestEncodeAiAssistCompleted:
    def test_envelope_only_when_body_empty(self) -> None:
        """An empty body still produces the envelope at field 1 + the
        synthesised completed_at Timestamp at field 10 (the outbox row
        always has a tenant_id / gcid / event_id + an occurred_at, so
        the envelope is never structurally empty)."""
        env = _make_envelope()
        out = encode_ai_assist_completed(env, {})
        walked = _walk(out)
        # Field 1 = envelope (length-delimited).
        assert walked[0]["field"] == 1
        assert walked[0]["wire"] == 2
        # Only the envelope + the synthesised completed_at submessage
        # appear — all scalar body fields default to "" / 0 / false and
        # proto3-default-omit.
        assert sorted({w["field"] for w in walked}) == [1, 10]

    def test_all_fields_populated(self) -> None:
        env = _make_envelope()
        body: dict[str, Any] = {
            "assist_id": "assist-1",
            "generated_count": 3,
            "screening_decision": "ALLOW",
            "screening_explanation": "clean",
            "model_used": "gemini-2.5-flash",
            "input_token_count": 120,
            "output_token_count": 340,
            "mana_charged": 5,
            "candidate_payload_json": '{"stem":"x"}',
            "pipeline_trace_json": "[]",
            "quality_warning": True,
            "attempt_count": 2,
            "critic_notes": "minor",
        }
        out = encode_ai_assist_completed(env, body)
        walked = _walk(out)
        by_field = _by_field(walked)

        # Field 1 — envelope submessage (wire-type 2).
        assert by_field[1]["wire"] == 2
        # Field 2 — assist_id (string).
        assert by_field[2]["wire"] == 2
        assert by_field[2]["value"].decode("utf-8") == "assist-1"
        # Field 3 — generated_count (varint).
        assert by_field[3]["wire"] == 0
        assert by_field[3]["value"] == 3
        # Field 4 — screening_decision.
        assert by_field[4]["value"].decode("utf-8") == "ALLOW"
        # Field 5 — screening_explanation.
        assert by_field[5]["value"].decode("utf-8") == "clean"
        # Field 6 — model_used.
        assert by_field[6]["value"].decode("utf-8") == "gemini-2.5-flash"
        # Field 7 — input_token_count (varint).
        assert by_field[7]["wire"] == 0
        assert by_field[7]["value"] == 120
        # Field 8 — output_token_count (varint).
        assert by_field[8]["value"] == 340
        # Field 9 — mana_charged (varint).
        assert by_field[9]["value"] == 5
        # Field 10 — completed_at (Timestamp submessage, wire 2).
        assert by_field[10]["wire"] == 2
        # Field 11 — candidate_payload_json.
        assert by_field[11]["value"].decode("utf-8") == '{"stem":"x"}'
        # Field 12 — pipeline_trace_json.
        assert by_field[12]["value"].decode("utf-8") == "[]"
        # Field 13 — quality_warning (bool as varint, true → 1).
        assert by_field[13]["wire"] == 0
        assert by_field[13]["value"] == 1
        # Field 14 — attempt_count.
        assert by_field[14]["value"] == 2
        # Field 15 — critic_notes.
        assert by_field[15]["value"].decode("utf-8") == "minor"

    def test_quality_warning_false_is_omitted(self) -> None:
        """proto3 omits ``false`` default — field 13 must NOT appear when
        quality_warning is False."""
        env = _make_envelope()
        body = {"assist_id": "a", "quality_warning": False}
        out = encode_ai_assist_completed(env, body)
        by_field = _by_field(_walk(out))
        assert 13 not in by_field

    def test_completed_at_synthesised_from_envelope_occurred_at(self) -> None:
        """The publisher does not pass completed_at explicitly — the
        encoder MUST synthesise it from envelope.occurred_at so the
        Timestamp field 10 is populated for Schema Registry validation."""
        env = _make_envelope()
        out = encode_ai_assist_completed(env, {"assist_id": "a"})
        by_field = _by_field(_walk(out))
        assert 10 in by_field, "completed_at must be present"
        # Sub-walk the Timestamp submessage; expect field 1 = seconds.
        ts_walked = _walk(by_field[10]["value"])
        ts_by_field = _by_field(ts_walked)
        assert 1 in ts_by_field
        # 2026-05-17T12:00:00Z → 1779105600 (sanity check via datetime).
        expected_secs = int(
            _dt.datetime(2026, 5, 17, 12, 0, 0, tzinfo=_dt.UTC).timestamp(),
        )
        assert ts_by_field[1]["value"] == expected_secs

    def test_zero_ints_are_omitted(self) -> None:
        """proto3 omits 0 defaults — mana_charged=0 must NOT appear."""
        env = _make_envelope()
        body = {"assist_id": "a", "mana_charged": 0, "attempt_count": 0}
        out = encode_ai_assist_completed(env, body)
        by_field = _by_field(_walk(out))
        assert 9 not in by_field
        assert 14 not in by_field

    def test_envelope_carries_event_id_and_idempotency_key(self) -> None:
        env = _make_envelope(event_id="evt-abc", idempotency_key="idem-xyz")
        out = encode_ai_assist_completed(env, {})
        envelope_buf = _by_field(_walk(out))[1]["value"]
        env_by_field = _by_field(_walk(envelope_buf))
        assert env_by_field[1]["value"].decode("utf-8") == "evt-abc"
        assert env_by_field[2]["value"].decode("utf-8") == "idem-xyz"

    def test_envelope_carries_imda_dimension(self) -> None:
        env = _make_envelope(chora_imda_dimension="accountability")
        out = encode_ai_assist_completed(env, {})
        envelope_buf = _by_field(_walk(out))[1]["value"]
        env_by_field = _by_field(_walk(envelope_buf))
        # Field 14 = chora_imda_dimension per envelope layout.
        assert env_by_field[14]["value"].decode("utf-8") == "accountability"

    def test_rfc3339_z_spelling_accepted(self) -> None:
        """The publisher emits Python isoformat with ``+00:00``; some
        callers (e.g. Go publishers) emit ``Z``. Both must work."""
        env = _make_envelope(occurred_at="2026-05-17T12:00:00Z")
        out = encode_ai_assist_completed(env, {"assist_id": "a"})
        by_field = _by_field(_walk(out))
        assert 10 in by_field  # completed_at synthesised
        envelope_buf = by_field[1]["value"]
        env_by_field = _by_field(_walk(envelope_buf))
        assert 5 in env_by_field  # occurred_at on envelope

    def test_invalid_timestamp_raises(self) -> None:
        env = _make_envelope(occurred_at="not-a-timestamp")
        with pytest.raises(ProtoEncodeError, match="RFC3339"):
            encode_ai_assist_completed(env, {})


# -----------------------------------------------------------------------------
# AiAssistRefused — field-by-field assertions.
# -----------------------------------------------------------------------------


class TestEncodeAiAssistRefused:
    def test_envelope_only_when_body_empty(self) -> None:
        env = _make_envelope(
            idempotency_key="ai_assist.refused.assist-1",
            chora_imda_dimension="safety_and_robustness",
        )
        out = encode_ai_assist_refused(env, {})
        walked = _walk(out)
        assert walked[0]["field"] == 1
        assert walked[0]["wire"] == 2
        # Field 8 = refused_at synthesised from envelope.occurred_at;
        # all scalar body fields default-omit per proto3.
        assert sorted({w["field"] for w in walked}) == [1, 8]

    def test_all_fields_populated(self) -> None:
        env = _make_envelope(
            idempotency_key="ai_assist.refused.assist-1",
            chora_imda_dimension="safety_and_robustness",
        )
        body: dict[str, Any] = {
            "assist_id": "assist-1",
            "refusal_reason": "GUARDRAIL_POST",
            "model_armor_verdict": "armor:jailbreak_attempt",
            "refusing_agent_id": "agent-x",
            "user_facing_message": "Your prompt couldn't be processed.",
            "mana_charged": 0,
            "last_candidate_payload_json": '{"stem":"y"}',
            "attempt_count": 1,
        }
        out = encode_ai_assist_refused(env, body)
        by_field = _by_field(_walk(out))

        # Field 1 — envelope.
        assert by_field[1]["wire"] == 2
        # Field 2 — assist_id.
        assert by_field[2]["value"].decode("utf-8") == "assist-1"
        # Field 3 — refusal_reason.
        assert by_field[3]["value"].decode("utf-8") == "GUARDRAIL_POST"
        # Field 4 — model_armor_verdict.
        assert by_field[4]["value"].decode("utf-8") == "armor:jailbreak_attempt"
        # Field 5 — refusing_agent_id.
        assert by_field[5]["value"].decode("utf-8") == "agent-x"
        # Field 6 — user_facing_message.
        assert by_field[6]["value"].decode("utf-8") == "Your prompt couldn't be processed."
        # Field 7 — mana_charged (0 → omitted).
        assert 7 not in by_field
        # Field 8 — refused_at (Timestamp submessage, wire 2).
        assert by_field[8]["wire"] == 2
        # Field 9 — last_candidate_payload_json.
        assert by_field[9]["value"].decode("utf-8") == '{"stem":"y"}'
        # Field 10 — attempt_count.
        assert by_field[10]["wire"] == 0
        assert by_field[10]["value"] == 1

    def test_refused_at_synthesised_from_envelope_occurred_at(self) -> None:
        env = _make_envelope(
            idempotency_key="ai_assist.refused.assist-1",
            chora_imda_dimension="safety_and_robustness",
        )
        body = {
            "assist_id": "assist-1",
            "refusal_reason": "GUARDRAIL_PRE",
            "model_armor_verdict": "armor:pii_high_risk_block",
        }
        out = encode_ai_assist_refused(env, body)
        by_field = _by_field(_walk(out))
        assert 8 in by_field
        ts_by_field = _by_field(_walk(by_field[8]["value"]))
        expected_secs = int(
            _dt.datetime(2026, 5, 17, 12, 0, 0, tzinfo=_dt.UTC).timestamp(),
        )
        assert ts_by_field[1]["value"] == expected_secs

    def test_envelope_imda_dimension_d3(self) -> None:
        env = _make_envelope(
            idempotency_key="ai_assist.refused.assist-1",
            chora_imda_dimension="safety_and_robustness",
        )
        out = encode_ai_assist_refused(env, {})
        envelope_buf = _by_field(_walk(out))[1]["value"]
        env_by_field = _by_field(_walk(envelope_buf))
        assert env_by_field[14]["value"].decode("utf-8") == "safety_and_robustness"

    def test_empty_body_fields_omitted(self) -> None:
        env = _make_envelope(
            idempotency_key="ai_assist.refused.assist-1",
            chora_imda_dimension="safety_and_robustness",
        )
        body = {
            "assist_id": "assist-1",
            "refusal_reason": "VALIDATION",
            "model_armor_verdict": "armor:unspecified",
            "user_facing_message": "bad input",
            "last_candidate_payload_json": "",
            "attempt_count": 0,
            "mana_charged": 0,
        }
        out = encode_ai_assist_refused(env, body)
        by_field = _by_field(_walk(out))
        # last_candidate_payload_json="" → field 9 omitted.
        assert 9 not in by_field
        # attempt_count=0 → field 10 omitted.
        assert 10 not in by_field
        # mana_charged=0 → field 7 omitted.
        assert 7 not in by_field
        # refusing_agent_id absent → field 5 omitted.
        assert 5 not in by_field


# -----------------------------------------------------------------------------
# Envelope encoding — shared between both messages.
# -----------------------------------------------------------------------------


class TestEnvelopeEncoding:
    def test_envelope_omits_unset_optional_fields(self) -> None:
        env: dict[str, Any] = {
            "event_id": "e-1",
            "tenant_id": "t-1",
        }
        out = encode_ai_assist_completed(env, {})
        envelope_buf = _by_field(_walk(out))[1]["value"]
        env_by_field = _by_field(_walk(envelope_buf))
        assert env_by_field[1]["value"].decode("utf-8") == "e-1"
        assert env_by_field[3]["value"].decode("utf-8") == "t-1"
        # Unset optionals: idempotency_key (2), gcid (4), traceparent (7), ...
        for f in (2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15):
            assert f not in env_by_field, f"field {f} should be omitted when unset"

    def test_schema_version_string_parsed_as_int(self) -> None:
        env = _make_envelope(schema_version="1")
        out = encode_ai_assist_completed(env, {})
        envelope_buf = _by_field(_walk(out))[1]["value"]
        env_by_field = _by_field(_walk(envelope_buf))
        # Field 11 = schema_version (varint).
        assert env_by_field[11]["wire"] == 0
        assert env_by_field[11]["value"] == 1

    def test_schema_version_int_accepted(self) -> None:
        env = _make_envelope(schema_version=2)
        out = encode_ai_assist_completed(env, {})
        envelope_buf = _by_field(_walk(out))[1]["value"]
        env_by_field = _by_field(_walk(envelope_buf))
        assert env_by_field[11]["value"] == 2


# -----------------------------------------------------------------------------
# Sanity — both encoders return bytes that start with field-1 envelope tag.
# -----------------------------------------------------------------------------


class TestWireSanity:
    def test_completed_starts_with_envelope_tag(self) -> None:
        """First byte 0x0A = tag(1, wire-type 2) — same as AiAssistStarted."""
        env = _make_envelope()
        out = encode_ai_assist_completed(env, {"assist_id": "a"})
        assert out[0] == 0x0A

    def test_refused_starts_with_envelope_tag(self) -> None:
        env = _make_envelope(
            idempotency_key="ai_assist.refused.a",
            chora_imda_dimension="safety_and_robustness",
        )
        out = encode_ai_assist_refused(env, {"assist_id": "a"})
        assert out[0] == 0x0A
