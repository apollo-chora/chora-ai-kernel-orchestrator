"""Unit tests for the ADR-167 observability encoders in ``proto_wire_encoder``:

* ``encode_agent_decision_logged`` →
  ``chora.observability.v1.AgentDecisionLogged``
  (topic ``chora.observability.agent_decision.logged.v1``)
(``encode_token_usage_recorded`` was REMOVED 2026-07-23 with the
orchestrator's token-usage lane -- see
tests/unit/test_token_usage_emitter_retired.py.)

Verification strategy
---------------------
Same as ``test_proto_wire_encoder.py``: rather than depend on the
generated Python bindings + a matching ``google.protobuf`` runtime (which
the orchestrator deliberately keeps OUT of its runtime image — see
``proto_wire_encoder`` module docstring), we parse the encoder's output
bytes back with a wire-format walker + assert field numbers / wire types /
values match the deployed Schema Registry schemas in
``chora-contracts/proto/events/observability/{agent_decision,token_usage}.proto``.

The Go consumer side (chora-governance + chora-observability) decodes these
exact bytes with the GENERATED Go bindings + its own proto-roundtrip
fixtures (ADR-167 Phase 2), so the producer↔consumer wire contract is
pinned from both ends.

Canonical mapping under test (qgen domain ⇄ AgentDecisionLogged) per ADR-167:

  assist_id            → decision_id (2) AND invocation_id (3)
  "qgen_crew"          → agid (4)
  DECISION_KIND_CRITIQUE (=4) → decision_kind (7)
  critic_notes         → output_summary (9)
  decided_at           → decided_at (11) Timestamp (== occurred_at)
  crew_name..cached_tokens → fields 12-20
  {decision, attempt_count, max_retries, quality_warning} → attributes (21) map
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    ProtoEncodeError,
    encode_agent_decision_logged,
)

# DECISION_KIND_CRITIQUE enum value per agent_decision.proto.
DECISION_KIND_CRITIQUE = 4


# -----------------------------------------------------------------------------
# Wire-format walker (mirrors test_proto_wire_encoder.py).
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
    """Last-write-wins flatten for scalar fields."""
    return {entry["field"]: entry for entry in walked}


def _collect_field(walked: list[dict[str, Any]], field: int) -> list[dict[str, Any]]:
    """All entries for a repeated field, in wire order."""
    return [e for e in walked if e["field"] == field]


def _decode_map_entry(buf: bytes) -> tuple[str, str]:
    """Decode a proto3 map<string,string> entry submessage: key=1, value=2."""
    walked = _walk(buf)
    by = _by_field(walked)
    key = by[1]["value"].decode("utf-8") if 1 in by else ""
    val = by[2]["value"].decode("utf-8") if 2 in by else ""
    return key, val


def _decode_map(walked: list[dict[str, Any]], field: int) -> dict[str, str]:
    """Decode all map entries at ``field`` into a Python dict."""
    out: dict[str, str] = {}
    for e in _collect_field(walked, field):
        k, v = _decode_map_entry(e["value"])
        out[k] = v
    return out


def _make_envelope(**overrides: Any) -> dict[str, Any]:
    now_iso = _dt.datetime(2026, 5, 17, 12, 0, 0, tzinfo=_dt.UTC).isoformat()
    env: dict[str, Any] = {
        "event_id": "01970000-aaaa-7000-8000-000000000001",
        "idempotency_key": "agent_decision.tenant-a.assist-1",
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
# AgentDecisionLogged
# -----------------------------------------------------------------------------


class TestEncodeAgentDecisionLogged:
    def test_canonical_mapping(self) -> None:
        env = _make_envelope()
        body: dict[str, Any] = {
            "assist_id": "assist-1",
            # agid now rides the body (per-agent attribution) — NEVER the
            # legacy hardcoded "qgen_crew". O+ /o/agents tiles key off this
            # string, so it MUST be the registry agid.
            "agid": "qgen_critic",
            "question_type": "mcq",
            "decision": "accepted",
            "attempt_count": 2,
            "max_retries": 3,
            "critic_notes": "stem ambiguous",
            "quality_warning": False,
            "crew_name": "mcq_ai_assist",
            "crew_id": "assist-1",
            "is_resume": False,
            "is_eval_run": False,
            "adapter_version": "",
            "guardrail_outcome": "pass",
            "prompt_tokens": 180,
            "completion_tokens": 100,
            "cached_tokens": 15,
        }
        out = encode_agent_decision_logged(env, body)
        walked = _walk(out)
        by = _by_field(walked)

        # Field 1 — envelope submessage.
        assert by[1]["wire"] == 2
        # Field 2 — decision_id == assist_id.
        assert by[2]["value"].decode("utf-8") == "assist-1"
        # Field 3 — invocation_id == assist_id.
        assert by[3]["value"].decode("utf-8") == "assist-1"
        # Field 4 — agid = body.agid (per-agent), NEVER "qgen_crew".
        assert by[4]["value"].decode("utf-8") == "qgen_critic"
        assert by[4]["value"].decode("utf-8") != "qgen_crew"
        # Field 7 — decision_kind = DECISION_KIND_CRITIQUE (varint).
        assert by[7]["wire"] == 0
        assert by[7]["value"] == DECISION_KIND_CRITIQUE
        # Field 9 — output_summary = critic_notes.
        assert by[9]["value"].decode("utf-8") == "stem ambiguous"
        # Field 11 — decided_at Timestamp submessage (wire 2).
        assert by[11]["wire"] == 2
        # Field 12 — crew_name.
        assert by[12]["value"].decode("utf-8") == "mcq_ai_assist"
        # Field 13 — crew_id.
        assert by[13]["value"].decode("utf-8") == "assist-1"
        # Field 17 — guardrail_outcome.
        assert by[17]["value"].decode("utf-8") == "pass"
        # Fields 18-20 — token counts (varint).
        assert by[18]["value"] == 180
        assert by[19]["value"] == 100
        assert by[20]["value"] == 15

        # Field 21 — attributes map: qgen-specific detail as string pairs.
        attrs = _decode_map(walked, 21)
        assert attrs["decision"] == "accepted"
        assert attrs["attempt_count"] == "2"
        assert attrs["max_retries"] == "3"
        assert attrs["quality_warning"] == "false"
        # question_type rides the attributes map (contract: present for qgen).
        assert attrs["question_type"] == "mcq"

    def test_agid_from_body_each_agent(self) -> None:
        """Field 4 (agid) is read from body.agid — the producer emits one
        event per agent (qgen_question / qgen_critic / oe_evaluator /
        oe_moderator). The legacy hardcoded "qgen_crew" is NEVER emitted."""
        for agid in ("qgen_question", "qgen_critic", "oe_evaluator", "oe_moderator"):
            out = encode_agent_decision_logged(
                _make_envelope(), {"assist_id": "a", "agid": agid, "decision": "accepted"}
            )
            by = _by_field(_walk(out))
            assert by[4]["value"].decode("utf-8") == agid
            assert by[4]["value"].decode("utf-8") != "qgen_crew"

    def test_question_type_in_attributes(self) -> None:
        """question_type ("mcq"|"oe") lands in the field-21 attributes map."""
        for qt in ("mcq", "oe"):
            out = encode_agent_decision_logged(
                _make_envelope(),
                {"assist_id": "a", "agid": "qgen_question", "decision": "accepted", "question_type": qt},
            )
            attrs = _decode_map(_walk(out), 21)
            assert attrs["question_type"] == qt

    def test_question_type_omitted_when_blank(self) -> None:
        """proto3 omits an empty map entry — a blank question_type means no
        attributes["question_type"] key is emitted."""
        out = encode_agent_decision_logged(
            _make_envelope(),
            {"assist_id": "a", "agid": "qgen_question", "decision": "accepted", "question_type": ""},
        )
        attrs = _decode_map(_walk(out), 21)
        assert "question_type" not in attrs

    def test_envelope_event_id_present(self) -> None:
        env = _make_envelope()
        out = encode_agent_decision_logged(env, {"assist_id": "a", "decision": "retry"})
        walked = _walk(out)
        by = _by_field(walked)
        # Decode the envelope submessage (field 1) + confirm event_id @ 1.
        env_bytes = by[1]["value"]
        env_by = _by_field(_walk(env_bytes))
        assert env_by[1]["value"].decode("utf-8") == env["event_id"]
        assert env_by[3]["value"].decode("utf-8") == env["tenant_id"]
        # chora_imda_dimension @ envelope field 14.
        assert env_by[14]["value"].decode("utf-8") == "accountability"

    def test_quality_warning_true_serialises_true(self) -> None:
        env = _make_envelope()
        out = encode_agent_decision_logged(
            env, {"assist_id": "a", "decision": "completed_with_warning", "quality_warning": True}
        )
        attrs = _decode_map(_walk(out), 21)
        assert attrs["quality_warning"] == "true"

    def test_extension_bools_emitted(self) -> None:
        env = _make_envelope()
        out = encode_agent_decision_logged(
            env,
            {
                "assist_id": "a",
                "decision": "accepted",
                "is_resume": True,
                "is_eval_run": True,
                "adapter_version": "lora-tenant-acme-v3",
            },
        )
        by = _by_field(_walk(out))
        # Field 14 — is_resume = true.
        assert by[14]["value"] == 1
        # Field 15 — is_eval_run = true.
        assert by[15]["value"] == 1
        # Field 16 — adapter_version.
        assert by[16]["value"].decode("utf-8") == "lora-tenant-acme-v3"

    def test_decision_omitted_when_blank(self) -> None:
        # proto3 omits empty map entries — a blank decision means no
        # attributes["decision"] key is emitted.
        env = _make_envelope()
        out = encode_agent_decision_logged(env, {"assist_id": "a", "decision": ""})
        attrs = _decode_map(_walk(out), 21)
        assert "decision" not in attrs

    def test_citation_hashes_in_attributes(self) -> None:
        """input_hash + output_hash (sha256 hex of the agent's input/output)
        ride the field-21 attributes map so the O+ reasoning panel renders a
        real PII-safe citation instead of the all-zeros sentinel. Raw content
        never leaves the producer — only the one-way hashes."""
        env = _make_envelope()
        out = encode_agent_decision_logged(
            env,
            {
                "assist_id": "a",
                "agid": "qgen_question",
                "decision": "accepted",
                "input_hash": "a" * 64,
                "output_hash": "b" * 64,
            },
        )
        attrs = _decode_map(_walk(out), 21)
        assert attrs["input_hash"] == "a" * 64
        assert attrs["output_hash"] == "b" * 64

    def test_citation_hashes_omitted_when_blank(self) -> None:
        """proto3 omits empty map entries — blank hashes mean no
        attributes["input_hash"]/["output_hash"] keys are emitted (the
        consumer then leaves the zero sentinel, same as today)."""
        env = _make_envelope()
        out = encode_agent_decision_logged(env, {"assist_id": "a", "decision": "accepted"})
        attrs = _decode_map(_walk(out), 21)
        assert "input_hash" not in attrs
        assert "output_hash" not in attrs

    def test_prompt_conditions_in_attributes(self) -> None:
        """ADR-197 M-A.3 — the prompt-shaping condition discriminants ride the
        field-21 attributes map under prefixed keys ``prompt_conditions.<key>``
        so the durable record carries the SAME discriminants the live agent span
        already does (no proto change — rides the existing attributes map)."""
        env = _make_envelope()
        out = encode_agent_decision_logged(
            env,
            {
                "assist_id": "a",
                "agid": "qgen_question",
                "decision": "accepted",
                "prompt_conditions": {
                    "intent": "new_question",
                    "question_type": "mcq",
                    "subject_hint": "Biology",
                    "image_for_stem": "true",
                },
            },
        )
        attrs = _decode_map(_walk(out), 21)
        assert attrs["prompt_conditions.intent"] == "new_question"
        assert attrs["prompt_conditions.question_type"] == "mcq"
        assert attrs["prompt_conditions.subject_hint"] == "Biology"
        assert attrs["prompt_conditions.image_for_stem"] == "true"

    def test_prompt_conditions_omitted_when_absent(self) -> None:
        """No prompt_conditions in the body → no prompt_conditions.* attribute
        keys (proto3 map default; backwards-compatible with pre-M-A.3 callers)."""
        env = _make_envelope()
        out = encode_agent_decision_logged(env, {"assist_id": "a", "agid": "qgen_question", "decision": "accepted"})
        attrs = _decode_map(_walk(out), 21)
        assert not any(k.startswith("prompt_conditions.") for k in attrs)

    def test_negative_token_count_raises(self) -> None:
        env = _make_envelope()
        with pytest.raises(ProtoEncodeError):
            encode_agent_decision_logged(env, {"assist_id": "a", "decision": "x", "prompt_tokens": -1})


# -----------------------------------------------------------------------------
# TokenUsageRecorded
# -----------------------------------------------------------------------------
