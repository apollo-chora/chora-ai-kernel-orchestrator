"""RED→GREEN tests for QGenCrewTerminalOutboxWriter.

The writer satisfies the ``_TerminalPublisher`` Protocol from
``orchestrators/qgen_crew_runner.py`` by INSERTing rows into
``ai_kernel_outbox_events`` (migration 0003_outbox.sql). The actual
Pub/Sub publish hop is handled by the existing ``OutboxDispatcher`` +
``GoogleCloudPubSubPublisher`` (separate adapter; this writer does NOT
publish directly).

This is the canonical transactional-outbox pattern per
[[feedback-d6-resilience-first-class]] + B.6.2.a; mirrors
``services/chora-closure-orchestrator/tests/unit/test_publisher_outbox.py``.

Pod-death survival invariant:
  * LangGraph PostgresSaver checkpoint + outbox row write hit the same DB.
  * On retry from checkpoint, ON CONFLICT (idempotency_key) DO NOTHING
    swallows duplicate INSERTs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_publisher import (
    IMDA_DIM_ACCOUNTABILITY,
    IMDA_DIM_SAFETY_AND_ROBUSTNESS,
    SCHEMA_VERSION,
    TOPIC_AI_ASSIST_COMPLETED,
    TOPIC_AI_ASSIST_PROGRESS,
    TOPIC_AI_ASSIST_REFUSED,
    QGenCrewTerminalOutboxWriter,
)

# -----------------------------------------------------------------------------
# Tiny proto3 wire-format walker — payload is now BINARY proto per Schema
# Registry constraint (not JSON). Mirrors the helpers in
# ``test_proto_wire_encoder.py``; kept local so the two test files stay
# independent.
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


def _walk_proto(data: bytes) -> dict[int, dict[str, Any]]:
    """Walk top-level proto3 fields → {field_number: {"wire": int, "value": ...}}."""
    out: dict[int, dict[str, Any]] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire = tag & 0x07
        if wire == 0:
            value, offset = _read_varint(data, offset)
            out[field] = {"wire": 0, "value": value}
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            out[field] = {"wire": 2, "value": data[offset : offset + length]}
            offset += length
        else:
            raise ValueError(f"unsupported wire {wire} at field {field}")
    return out


# -----------------------------------------------------------------------------
# Fake AsyncConnection / Cursor — mirrors psycopg's AsyncConnection surface
# -----------------------------------------------------------------------------


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
    commits: list[int] = field(default_factory=list)

    def cursor(self) -> _FakeCursor:
        return self.cur

    async def commit(self) -> None:
        # The writer commits its OWN INSERT - the shared connection is drained
        # by a concurrent dispatcher whose transaction release would otherwise
        # discard an uncommitted row (live loss 2026-08-14, job fd80f4e9).
        # Durability is asserted end-to-end in
        # tests/unit/test_outbox_shared_conn_durability.py.
        self.commits.append(len(self.cur.executed))

    async def rollback(self) -> None:  # pragma: no cover - never taken here
        raise AssertionError("outbox writer must never roll back its own row")


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_requires_source_project(self) -> None:
        with pytest.raises(ValueError, match="source_project"):
            QGenCrewTerminalOutboxWriter(
                conn=_FakeAsyncConnection(),
                source_project="",
            )

    def test_requires_source_service(self) -> None:
        with pytest.raises(ValueError, match="source_service"):
            QGenCrewTerminalOutboxWriter(
                conn=_FakeAsyncConnection(),
                source_project="chora-489812",
                source_service="",
            )

    def test_captures_project_and_service(self) -> None:
        w = QGenCrewTerminalOutboxWriter(
            conn=_FakeAsyncConnection(),
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        assert w.source_project == "chora-489812"
        assert w.source_service == "chora-ai-kernel-orchestrator"


# -----------------------------------------------------------------------------
# publish_completed
# -----------------------------------------------------------------------------


def _decode_row(executed: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    sql, params = executed
    assert "INSERT INTO ai_kernel_outbox_events" in sql
    assert "ON CONFLICT (idempotency_key) DO NOTHING" in sql
    return params


class TestPublishCompleted:
    @pytest.mark.asyncio
    async def test_writes_row_to_outbox(self) -> None:
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        row_id = await w.publish_completed(
            assist_id="01970000-7777-7000-a000-000000000001",
            tenant_id="01970000-0000-7000-8000-000000000001",
            author_gcid="01970000-0000-7000-9000-000000000001",
            candidate_payload_json='{"stem":"x"}',
            pipeline_trace_json="[]",
            quality_warning=False,
            attempt_count=1,
            critic_notes="",
            mana_charged=0,
            traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        )
        assert row_id != ""
        assert len(conn.cur.executed) == 1
        params = _decode_row(conn.cur.executed[0])
        assert params["topic"] == TOPIC_AI_ASSIST_COMPLETED
        assert params["event_type"] == "creation.ai_assist.completed"
        assert params["workflow_id"] == "01970000-7777-7000-a000-000000000001"
        assert params["tenant_id"] == "01970000-0000-7000-8000-000000000001"
        assert params["gcid"] == "01970000-0000-7000-9000-000000000001"

    @pytest.mark.asyncio
    async def test_payload_body_matches_subscriber_dto(self) -> None:
        """payload BYTEA must decode to the AiAssistCompleted binary-proto
        shape in chora-contracts/proto/events-flat/creation/ai_assist/
        completed.proto — the canonical schema attached to the live
        Pub/Sub topic via Schema Registry."""
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.publish_completed(
            assist_id="a-1",
            tenant_id="t-1",
            author_gcid="g-1",
            candidate_payload_json='{"stem":"x"}',
            pipeline_trace_json="[]",
            quality_warning=True,
            attempt_count=4,
            critic_notes="qa concerns",
            mana_charged=5,
        )
        params = _decode_row(conn.cur.executed[0])
        # payload is now binary proto (NOT JSON) per Schema Registry.
        fields = _walk_proto(params["payload"])
        # Field 1 = envelope (length-delimited submessage).
        assert fields[1]["wire"] == 2
        # Field 2 = assist_id.
        assert fields[2]["value"].decode("utf-8") == "a-1"
        # Field 9 = mana_charged (varint).
        assert fields[9]["value"] == 5
        # Field 10 = completed_at (Timestamp submessage, synthesised
        # from envelope.occurred_at).
        assert fields[10]["wire"] == 2
        # Field 11 = candidate_payload_json.
        assert fields[11]["value"].decode("utf-8") == '{"stem":"x"}'
        # Field 12 = pipeline_trace_json.
        assert fields[12]["value"].decode("utf-8") == "[]"
        # Field 13 = quality_warning (bool → varint 1 since True).
        assert fields[13]["value"] == 1
        # Field 14 = attempt_count.
        assert fields[14]["value"] == 4
        # Field 15 = critic_notes.
        assert fields[15]["value"].decode("utf-8") == "qa concerns"

    @pytest.mark.asyncio
    async def test_envelope_carries_all_mandatory_fields(self) -> None:
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(
            conn=conn,
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        await w.publish_completed(
            assist_id="a-1",
            tenant_id="t-1",
            author_gcid="g-1",
            candidate_payload_json="{}",
            pipeline_trace_json="[]",
            quality_warning=False,
            attempt_count=1,
            critic_notes="",
            mana_charged=0,
            traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
            tracestate="vendor=chora",
        )
        params = _decode_row(conn.cur.executed[0])
        env = json.loads(params["envelope"])
        for required in (
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
            "chora_imda_dimension",
        ):
            assert required in env, f"envelope missing: {required}"
        assert env["source_project"] == "chora-489812"
        assert env["source_service"] == "chora-ai-kernel-orchestrator"
        assert env["schema_version"] == SCHEMA_VERSION
        assert env["chora_imda_dimension"] == IMDA_DIM_ACCOUNTABILITY
        assert env["traceparent"] == ("00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01")

    @pytest.mark.asyncio
    async def test_idempotency_key_deterministic_per_assist_id(self) -> None:
        """Same assist_id + topic → same idempotency_key → ON CONFLICT DO
        NOTHING swallows the duplicate INSERT.

        This is the D6 P2 invariant: a graph re-run from checkpoint
        re-executes the terminal node + re-INSERTs; the outbox stays
        exactly-once.
        """
        conn1 = _FakeAsyncConnection()
        w1 = QGenCrewTerminalOutboxWriter(
            conn=conn1,
            source_project="chora-489812",
        )
        await w1.publish_completed(
            assist_id="same-id",
            tenant_id="t",
            author_gcid="g",
            candidate_payload_json="{}",
            pipeline_trace_json="[]",
            quality_warning=False,
            attempt_count=1,
            critic_notes="",
            mana_charged=0,
        )
        conn2 = _FakeAsyncConnection()
        w2 = QGenCrewTerminalOutboxWriter(
            conn=conn2,
            source_project="chora-489812",
        )
        await w2.publish_completed(
            assist_id="same-id",
            tenant_id="t",
            author_gcid="g",
            candidate_payload_json="{}",
            pipeline_trace_json="[]",
            quality_warning=False,
            attempt_count=1,
            critic_notes="",
            mana_charged=0,
        )
        params1 = _decode_row(conn1.cur.executed[0])
        params2 = _decode_row(conn2.cur.executed[0])
        assert params1["idempotency_key"] == params2["idempotency_key"]
        assert params1["idempotency_key"] == "ai_assist.completed.same-id"

    @pytest.mark.asyncio
    async def test_completed_and_refused_have_distinct_keys(self) -> None:
        """Same assist_id but different terminal types must produce distinct
        keys so both events land independently in the outbox."""
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.publish_completed(
            assist_id="a-1",
            tenant_id="t",
            author_gcid="g",
            candidate_payload_json="{}",
            pipeline_trace_json="[]",
            quality_warning=False,
            attempt_count=1,
            critic_notes="",
            mana_charged=0,
        )
        await w.publish_refused(
            assist_id="a-1",
            tenant_id="t",
            author_gcid="g",
            refusal_reason="GUARDRAIL_PRE",
            model_armor_verdict="armor:pii_high_risk_block",
            user_facing_message="x",
            last_candidate_payload_json="",
            pipeline_trace_json="[]",
            attempt_count=0,
            mana_charged=0,
        )
        params_done = _decode_row(conn.cur.executed[0])
        params_refused = _decode_row(conn.cur.executed[1])
        assert params_done["idempotency_key"] == "ai_assist.completed.a-1"
        assert params_refused["idempotency_key"] == "ai_assist.refused.a-1"


# -----------------------------------------------------------------------------
# publish_refused
# -----------------------------------------------------------------------------


class TestPublishProgress:
    @pytest.mark.asyncio
    async def test_writes_binary_progress_row_with_per_step_key(self) -> None:
        """publish_progress writes a binary-proto AiAssistProgress row to the
        progress topic with a PER-STEP idempotency key (so each node's emit
        lands independently, unlike the per-job completed/refused key)."""
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(conn=conn, source_project="chora-489812")
        row_id = await w.publish_progress(
            assist_id="a-1",
            tenant_id="t-1",
            author_gcid="g-1",
            pipeline_trace_json='[{"name":"generate","status":"COMPLETED"}]',
            step_index=3,
            step_name="generate",
            step_status="COMPLETED",
            traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        )
        assert row_id != ""
        params = _decode_row(conn.cur.executed[0])
        assert params["topic"] == TOPIC_AI_ASSIST_PROGRESS
        assert params["event_type"] == "creation.ai_assist.progress"
        # Per-step idempotency key (NOT the per-job completed/refused key).
        assert params["idempotency_key"] == "ai_assist.progress.3.a-1"
        # payload is binary proto (Schema-Registry attached), NOT JSON.
        fields = _walk_proto(params["payload"])
        assert fields[1]["wire"] == 2  # envelope submessage
        assert fields[2]["value"].decode("utf-8") == "a-1"  # assist_id
        assert fields[3]["value"].decode("utf-8") == "t-1"  # tenant_id
        assert fields[4]["value"].decode("utf-8") == "g-1"  # author_gcid
        # field 5 = pipeline_trace_json
        assert fields[5]["value"].decode("utf-8") == ('[{"name":"generate","status":"COMPLETED"}]')
        assert fields[6]["value"] == 3  # step_index (varint)
        assert fields[7]["value"].decode("utf-8") == "generate"  # step_name
        assert fields[8]["value"].decode("utf-8") == "COMPLETED"  # step_status

    @pytest.mark.asyncio
    async def test_distinct_steps_get_distinct_keys(self) -> None:
        """Two different step_index values for the same assist_id produce
        distinct idempotency keys → both land in the outbox (the growing trace
        is preserved, not collapsed by ON CONFLICT DO NOTHING)."""
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(conn=conn, source_project="chora-489812")
        await w.publish_progress(
            assist_id="same",
            tenant_id="t",
            author_gcid="g",
            pipeline_trace_json="[]",
            step_index=1,
        )
        await w.publish_progress(
            assist_id="same",
            tenant_id="t",
            author_gcid="g",
            pipeline_trace_json="[]",
            step_index=2,
        )
        k1 = _decode_row(conn.cur.executed[0])["idempotency_key"]
        k2 = _decode_row(conn.cur.executed[1])["idempotency_key"]
        assert k1 == "ai_assist.progress.1.same"
        assert k2 == "ai_assist.progress.2.same"
        assert k1 != k2


class TestPublishRefused:
    @pytest.mark.asyncio
    async def test_refused_writes_to_outbox(self) -> None:
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        row_id = await w.publish_refused(
            assist_id="a-1",
            tenant_id="t-1",
            author_gcid="g-1",
            refusal_reason="GUARDRAIL_PRE",
            model_armor_verdict="armor:pii_high_risk_block",
            user_facing_message="Your prompt couldn't be processed.",
            last_candidate_payload_json="",
            pipeline_trace_json="[]",
            attempt_count=0,
            mana_charged=0,
            traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        )
        assert row_id != ""
        params = _decode_row(conn.cur.executed[0])
        assert params["topic"] == TOPIC_AI_ASSIST_REFUSED
        assert params["event_type"] == "creation.ai_assist.refused"
        # payload is now binary proto per Schema Registry.
        fields = _walk_proto(params["payload"])
        # Field 3 = refusal_reason.
        assert fields[3]["value"].decode("utf-8") == "GUARDRAIL_PRE"
        # Field 4 = model_armor_verdict.
        assert fields[4]["value"].decode("utf-8") == "armor:pii_high_risk_block"
        # Field 6 = user_facing_message.
        assert fields[6]["value"].decode("utf-8") == "Your prompt couldn't be processed."
        # Field 9 = last_candidate_payload_json. Empty string → omitted
        # per proto3 default-value rule.
        assert 9 not in fields

    @pytest.mark.asyncio
    async def test_envelope_carries_event_topic_routing_hint(self) -> None:
        """The outbox publisher strips the reserved "topic" attribute, so
        the envelope MUST carry the non-reserved event_topic alias (the
        weakness/oe-grading writer precedent). Without it, consumption's
        proofing-test-terminal push dispatch matches neither `topic` nor
        `event_topic` and ACK-DROPS every qgen terminal — campaign question
        sets stay `requested` forever (walk-found, CHO-2087 day 2)."""
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(
            conn=conn,
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        await w.publish_completed(
            assist_id="a-1",
            tenant_id="t-1",
            author_gcid="g-1",
            candidate_payload_json="{}",
            pipeline_trace_json="[]",
            quality_warning=False,
            attempt_count=1,
            critic_notes="",
            mana_charged=0,
            traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
            tracestate="vendor=chora",
        )
        env = json.loads(_decode_row(conn.cur.executed[0])["envelope"])
        assert env["event_topic"] == "chora.creation.ai_assist.completed.v1"

        await w.publish_refused(
            assist_id="a-2",
            tenant_id="t-1",
            author_gcid="g-1",
            refusal_reason="GUARDRAIL_POST",
            model_armor_verdict="armor:jailbreak_attempt",
            user_facing_message="x",
            last_candidate_payload_json="{}",
            pipeline_trace_json="[]",
            attempt_count=1,
            mana_charged=0,
        )
        env2 = json.loads(_decode_row(conn.cur.executed[1])["envelope"])
        assert env2["event_topic"] == "chora.creation.ai_assist.refused.v1"

    async def test_refused_envelope_d3_dimension_plus_extras(self) -> None:
        """refused.v1 → D3 IMDA Safety & Robustness. The refusal_reason +
        model_armor_verdict ALSO land in the envelope so the dispatcher
        emits them as Pub/Sub message attributes — D3 governance
        subscribers index without payload decode."""
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.publish_refused(
            assist_id="a-1",
            tenant_id="t",
            author_gcid="g",
            refusal_reason="GUARDRAIL_POST",
            model_armor_verdict="armor:jailbreak_attempt",
            user_facing_message="x",
            last_candidate_payload_json='{"stem":"y"}',
            pipeline_trace_json="[]",
            attempt_count=1,
            mana_charged=0,
        )
        params = _decode_row(conn.cur.executed[0])
        env = json.loads(params["envelope"])
        assert env["chora_imda_dimension"] == IMDA_DIM_SAFETY_AND_ROBUSTNESS
        assert env["refusal_reason"] == "GUARDRAIL_POST"
        assert env["model_armor_verdict"] == "armor:jailbreak_attempt"

    @pytest.mark.asyncio
    async def test_empty_armor_verdict_coerced_to_unspecified(self) -> None:
        """ai_assist.proto §refused mandates a non-empty verdict tag —
        coerce empty to 'armor:unspecified' so D3 evidence is never blank.
        """
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.publish_refused(
            assist_id="a-1",
            tenant_id="t",
            author_gcid="g",
            refusal_reason="VALIDATION",
            model_armor_verdict="",
            user_facing_message="bad input",
            last_candidate_payload_json="",
            pipeline_trace_json="[]",
            attempt_count=0,
            mana_charged=0,
        )
        params = _decode_row(conn.cur.executed[0])
        # payload is binary proto; field 4 = model_armor_verdict.
        fields = _walk_proto(params["payload"])
        assert fields[4]["value"].decode("utf-8") == "armor:unspecified"
        env = json.loads(params["envelope"])
        assert env["model_armor_verdict"] == "armor:unspecified"


# -----------------------------------------------------------------------------
# Error propagation
# -----------------------------------------------------------------------------


class TestGenerationSummaryFields:
    """CHO-1819 — typed generated_count (field 3) + generation_summary
    (field 16) on completed.v1. Legacy single-path callers omit both → field 3
    is 0 (proto3-omitted) + field 16 absent (unchanged wire)."""

    @pytest.mark.asyncio
    async def test_generated_count_and_summary_encoded(self) -> None:
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(conn=conn, source_project="chora-489812")
        await w.publish_completed(
            assist_id="a-1",
            tenant_id="t-1",
            author_gcid="g-1",
            candidate_payload_json='{"candidates":[]}',
            pipeline_trace_json="[]",
            quality_warning=False,
            attempt_count=10,
            critic_notes="",
            mana_charged=0,
            generated_count=7,
            generation_summary={
                "requested_total": 10,
                "generated_total": 7,
                "generated_per_type": {"mcq": 6, "oe": 1},
                "shortfall_reason": "Source supported 7.",
            },
        )
        params = _decode_row(conn.cur.executed[0])
        fields = _walk_proto(params["payload"])
        # Field 3 = generated_count (varint).
        assert fields[3]["value"] == 7
        # Field 16 = GenerationSummary submessage (length-delimited).
        assert fields[16]["wire"] == 2
        sub = _walk_proto(fields[16]["value"])
        # 1=requested_total, 2=generated_total, 4=shortfall_reason.
        assert sub[1]["value"] == 10
        assert sub[2]["value"] == 7
        assert sub[4]["value"].decode("utf-8") == "Source supported 7."
        # Field 3 of the submessage = repeated map<string,int32> entries.
        assert 3 in sub  # at least one generated_per_type entry encoded

    @pytest.mark.asyncio
    async def test_legacy_completed_omits_summary_and_zero_count(self) -> None:
        conn = _FakeAsyncConnection()
        w = QGenCrewTerminalOutboxWriter(conn=conn, source_project="chora-489812")
        await w.publish_completed(
            assist_id="a-1",
            tenant_id="t-1",
            author_gcid="g-1",
            candidate_payload_json='{"stem":"x"}',
            pipeline_trace_json="[]",
            quality_warning=False,
            attempt_count=1,
            critic_notes="",
            mana_charged=0,
        )
        params = _decode_row(conn.cur.executed[0])
        fields = _walk_proto(params["payload"])
        # generated_count 0 ⇒ proto3-omitted (field 3 absent).
        assert 3 not in fields
        # generation_summary absent ⇒ field 16 absent.
        assert 16 not in fields


class TestErrors:
    @pytest.mark.asyncio
    async def test_db_error_propagates(self) -> None:
        """LangGraph runner has to see the failure to NACK the inbound
        started.v1 + let LangGraph re-run the terminal node from checkpoint."""

        @dataclass
        class _FailingCursor:
            async def execute(self, sql: str, params: dict[str, Any]) -> None:
                raise RuntimeError("DB connection refused")

            async def __aenter__(self) -> _FailingCursor:
                return self

            async def __aexit__(self, *exc_info: Any) -> None:
                return None

        @dataclass
        class _FailingConn:
            def cursor(self) -> _FailingCursor:
                return _FailingCursor()

        w = QGenCrewTerminalOutboxWriter(
            conn=_FailingConn(),
            source_project="chora-489812",
        )
        with pytest.raises(RuntimeError, match="DB connection refused"):
            await w.publish_completed(
                assist_id="a-1",
                tenant_id="t",
                author_gcid="g",
                candidate_payload_json="{}",
                pipeline_trace_json="[]",
                quality_warning=False,
                attempt_count=1,
                critic_notes="",
                mana_charged=0,
            )
