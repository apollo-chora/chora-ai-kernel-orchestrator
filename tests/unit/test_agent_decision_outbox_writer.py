"""RED→GREEN tests for AgentDecisionLogOutboxWriter (Gate #8).

The writer satisfies the ``_AgentDecisionLogEmitter`` Protocol from
``orchestrators/qgen_crew_runner.py`` by INSERTing rows into
``ai_kernel_outbox_events`` (migration 0003_outbox.sql) with topic
``chora.observability.agent_decision.logged.v1``. The actual Pub/Sub
publish hop is handled by the existing ``OutboxDispatcher`` +
``GoogleCloudPubSubPublisher`` (separate adapter; this writer does NOT
publish directly).

Canonical transactional-outbox pattern per [[feedback-d6-resilience-
first-class]] + B.6.2.a. Mirrors
``services/chora-ai-kernel-orchestrator/tests/unit/test_qgen_crew_publisher.py``.

D6 4-pillar contract:
  * Pillar 1 (pod-death survival) — atomic with terminal transaction
    (shares the same psycopg AsyncConnection as the qgen crew writer).
  * Pillar 2 (delivery resilience) — OutboxDispatcher drains; DLQ on
    persistent failure.
  * Pillar 3 (multi-tenant isolation) — tenant_id stamped in envelope
    + payload.
  * Pillar 4 (Cloud Trace attribution) — traceparent + tracestate in
    envelope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_decision_outbox_writer import (
    IMDA_DIM_ACCOUNTABILITY,
    SCHEMA_VERSION,
    TOPIC_AGENT_DECISION_LOGGED,
    AgentDecisionLogOutboxWriter,
)

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
        self.commits.append(len(self.cur.executed))

    async def rollback(self) -> None:  # pragma: no cover - never taken here
        raise AssertionError("outbox writer must never roll back its own row")


def _decode_row(executed: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    sql, params = executed
    assert "INSERT INTO ai_kernel_outbox_events" in sql
    assert "ON CONFLICT (idempotency_key) DO NOTHING" in sql
    return params


# -----------------------------------------------------------------------------
# Proto wire-format walker (ADR-167) — the payload BYTEA is now canonical
# binary protobuf chora.observability.v1.AgentDecisionLogged, not JSON. We
# parse it back with a small read-only walker (mirrors the Go consumer's
# proto.Unmarshal roundtrip) to assert the canonical qgen⇄proto mapping.
# -----------------------------------------------------------------------------


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    result, shift = 0, 0
    while offset < len(data):
        b = data[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7
    raise ValueError("truncated varint")


def _walk(data: bytes) -> dict[int, list[Any]]:
    """Return {field_number: [values...]} — varint→int, length-delimited→bytes."""
    out: dict[int, list[Any]] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field, wire = tag >> 3, tag & 0x07
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        else:
            raise ValueError(f"unsupported wire type {wire} at field {field}")
        out.setdefault(field, []).append(value)
    return out


def _decode_attributes(walked: dict[int, list[Any]]) -> dict[str, str]:
    """Decode the field-21 map<string,string> entries into a Python dict."""
    out: dict[str, str] = {}
    for entry in walked.get(21, []):
        e = _walk(entry)
        key = e[1][0].decode("utf-8") if 1 in e else ""
        val = e[2][0].decode("utf-8") if 2 in e else ""
        out[key] = val
    return out


def _decode_proto_payload(payload: bytes) -> dict[str, Any]:
    """Decode the AgentDecisionLogged proto payload into the
    projector-relevant view per the ADR-167 canonical mapping."""
    w = _walk(payload)

    def s(field: int) -> str:
        return w[field][0].decode("utf-8") if field in w else ""

    def i(field: int) -> int:
        return int(w[field][0]) if field in w else 0

    return {
        "decision_id": s(2),
        "invocation_id": s(3),
        "agid": s(4),
        "model_id": s(6),
        "decision_kind": i(7),
        "output_summary": s(9),
        "crew_name": s(12),
        "crew_id": s(13),
        "is_resume": bool(i(14)),
        "is_eval_run": bool(i(15)),
        "adapter_version": s(16),
        "guardrail_outcome": s(17),
        "prompt_tokens": i(18),
        "completion_tokens": i(19),
        "cached_tokens": i(20),
        "attributes": _decode_attributes(w),
        "has_decided_at": 11 in w,
    }


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_requires_source_project(self) -> None:
        with pytest.raises(ValueError, match="source_project"):
            AgentDecisionLogOutboxWriter(
                conn=_FakeAsyncConnection(),
                source_project="",
            )

    def test_requires_source_service(self) -> None:
        with pytest.raises(ValueError, match="source_service"):
            AgentDecisionLogOutboxWriter(
                conn=_FakeAsyncConnection(),
                source_project="chora-489812",
                source_service="",
            )

    def test_captures_project_and_service(self) -> None:
        w = AgentDecisionLogOutboxWriter(
            conn=_FakeAsyncConnection(),
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        assert w.source_project == "chora-489812"
        assert w.source_service == "chora-ai-kernel-orchestrator"


# -----------------------------------------------------------------------------
# Topic constant — locked to the canonical provisioned name
# -----------------------------------------------------------------------------


class TestTopicConstants:
    def test_topic_is_canonical_observability_form(self) -> None:
        """Per [[feedback-arch-ground-in-deployed-reality]] the canonical
        provisioned topic is observability.agent_decision.logged.v1 — NOT
        the deprecated ai_kernel.agent_decided.v1 form named in the plan
        doc. Tested as a constant so a copy-paste swap is caught at unit
        time, not at deploy time."""
        assert TOPIC_AGENT_DECISION_LOGGED == "chora.observability.agent_decision.logged.v1"


# -----------------------------------------------------------------------------
# emit() — happy path
# -----------------------------------------------------------------------------


class TestEmit:
    @pytest.mark.asyncio
    async def test_writes_row_to_outbox(self) -> None:
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        row_id = await w.emit(
            assist_id="01970000-7777-7000-a000-000000000001",
            agid="qgen_question",
            tenant_id="01970000-0000-7000-8000-000000000001",
            gcid="01970000-0000-7000-9000-000000000001",
            decision="accepted",
            attempt_count=2,
            max_retries=3,
            critic_notes="reasonable",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
            traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
            tracestate="vendor=chora",
        )
        assert row_id != ""
        assert len(conn.cur.executed) == 1
        params = _decode_row(conn.cur.executed[0])
        assert params["topic"] == TOPIC_AGENT_DECISION_LOGGED
        assert params["event_type"] == "observability.agent_decision.logged"
        assert params["workflow_id"] == "01970000-7777-7000-a000-000000000001"
        assert params["tenant_id"] == "01970000-0000-7000-8000-000000000001"
        assert params["gcid"] == "01970000-0000-7000-9000-000000000001"

    @pytest.mark.asyncio
    async def test_payload_proto_matches_canonical_mapping(self) -> None:
        """payload BYTEA must decode to the canonical
        chora.observability.v1.AgentDecisionLogged wire shape the Go
        consumer (chora-governance) proto.Unmarshal-es (ADR-167). The qgen
        verdict + counts ride the attributes map (field 21); the GENERIC
        proto core stays cross-agent."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_critic",
            question_type="mcq",
            tenant_id="t-1",
            gcid="g-1",
            decision="rejected",
            attempt_count=4,
            max_retries=3,
            critic_notes="qa concerns",
            quality_warning=True,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
            traceparent="",
            tracestate="",
        )
        params = _decode_row(conn.cur.executed[0])
        d = _decode_proto_payload(params["payload"])
        # decision_id == invocation_id == assist_id.
        assert d["decision_id"] == "a-1"
        assert d["invocation_id"] == "a-1"
        # agid is the per-agent registry id (NEVER the legacy "qgen_crew").
        assert d["agid"] == "qgen_critic"
        assert d["agid"] != "qgen_crew"
        assert d["decision_kind"] == 4  # DECISION_KIND_CRITIQUE
        assert d["output_summary"] == "qa concerns"  # critic_notes
        assert d["has_decided_at"] is True
        # qgen-specific detail in the attributes map (incl. question_type).
        assert d["attributes"] == {
            "decision": "rejected",
            "attempt_count": "4",
            "max_retries": "3",
            "quality_warning": "true",
            "question_type": "mcq",
        }
        # Extension fields omitted (proto3 default) when caller omits them.
        assert d["crew_name"] == ""
        assert d["crew_id"] == ""
        assert d["is_resume"] is False
        assert d["is_eval_run"] is False
        assert d["adapter_version"] == ""
        assert d["guardrail_outcome"] == ""
        assert d["prompt_tokens"] == 0
        assert d["completion_tokens"] == 0
        assert d["cached_tokens"] == 0

    @pytest.mark.asyncio
    async def test_payload_proto_forwards_extension_fields(self) -> None:
        """When the runner passes crew + cost extension fields, they land
        in proto fields 12-20 for the Go consumer to read."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t-1",
            gcid="g-1",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
            crew_name="mcq_ai_assist",
            crew_id="01970000-7777-7000-a000-000000000001",
            is_resume=False,
            is_eval_run=True,
            adapter_version="lora-tenant-acme-v3",
            guardrail_outcome="pass",
            prompt_tokens=180,
            completion_tokens=100,
            cached_tokens=15,
        )
        params = _decode_row(conn.cur.executed[0])
        d = _decode_proto_payload(params["payload"])
        assert d["crew_name"] == "mcq_ai_assist"
        assert d["crew_id"] == "01970000-7777-7000-a000-000000000001"
        assert d["is_resume"] is False
        assert d["is_eval_run"] is True
        assert d["adapter_version"] == "lora-tenant-acme-v3"
        assert d["guardrail_outcome"] == "pass"
        assert d["prompt_tokens"] == 180
        assert d["completion_tokens"] == 100
        assert d["cached_tokens"] == 15

    @pytest.mark.asyncio
    async def test_payload_proto_sets_model_id_field_6(self) -> None:
        """The concrete model that produced the decision lands in proto
        field 6 (model_id — matches token_usage.model_id) so
        chora-observability can price the per-hop token counts per model.
        A blank model_id here zeroes the cost attribution."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t-1",
            gcid="g-1",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
            model_id="gemini-3.1-pro-preview",
        )
        params = _decode_row(conn.cur.executed[0])
        d = _decode_proto_payload(params["payload"])
        assert d["model_id"] == "gemini-3.1-pro-preview"

    @pytest.mark.asyncio
    async def test_payload_proto_model_id_omitted_when_blank(self) -> None:
        """No model reported → field 6 is proto3-default-omitted (the Go
        consumer keeps the zero sentinel rather than a fabricated model)."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t-1",
            gcid="g-1",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
        )
        params = _decode_row(conn.cur.executed[0])
        d = _decode_proto_payload(params["payload"])
        assert d["model_id"] == ""

    @pytest.mark.asyncio
    async def test_payload_proto_carries_citation_hashes(self) -> None:
        """input_hash + output_hash (sha256 hex over the agent's input/output)
        ride the attributes map (field 21) so chora-observability persists a
        real PII-safe citation instead of the all-zeros sentinel. Raw content
        is never carried — only the one-way hashes."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t-1",
            gcid="g-1",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
            input_hash="c" * 64,
            output_hash="d" * 64,
        )
        params = _decode_row(conn.cur.executed[0])
        d = _decode_proto_payload(params["payload"])
        assert d["attributes"]["input_hash"] == "c" * 64
        assert d["attributes"]["output_hash"] == "d" * 64

    @pytest.mark.asyncio
    async def test_citation_hashes_omitted_when_blank(self) -> None:
        """When the caller omits the hashes (e.g. the OE crew path), no
        input_hash/output_hash attribute is emitted — proto3 map default."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t-1",
            gcid="g-1",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
        )
        params = _decode_row(conn.cur.executed[0])
        d = _decode_proto_payload(params["payload"])
        assert "input_hash" not in d["attributes"]
        assert "output_hash" not in d["attributes"]

    @pytest.mark.asyncio
    async def test_prompt_conditions_thread_into_proto_attributes(self) -> None:
        """ADR-197 M-A.3 — the runner passes a prompt_conditions map; the writer
        threads it into the event body so the encoder rides each entry in the
        field-21 attributes map under prefixed keys ``prompt_conditions.<key>``
        (no proto change)."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(conn=conn, source_project="chora-489812")
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t-1",
            gcid="g-1",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
            prompt_conditions={"intent": "new_question", "question_type": "oe"},
        )
        params = _decode_row(conn.cur.executed[0])
        d = _decode_proto_payload(params["payload"])
        assert d["attributes"]["prompt_conditions.intent"] == "new_question"
        assert d["attributes"]["prompt_conditions.question_type"] == "oe"

    @pytest.mark.asyncio
    async def test_prompt_conditions_omitted_when_not_passed(self) -> None:
        """Caller omits prompt_conditions → no prompt_conditions.* attribute keys
        (proto3 map default; the OE/legacy paths are unaffected)."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(conn=conn, source_project="chora-489812")
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t-1",
            gcid="g-1",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
        )
        params = _decode_row(conn.cur.executed[0])
        d = _decode_proto_payload(params["payload"])
        assert not any(k.startswith("prompt_conditions.") for k in d["attributes"])

    @pytest.mark.asyncio
    async def test_envelope_carries_crew_name_for_subscriber_index(self) -> None:
        """crew_name lives on the envelope (in addition to the payload)
        so D1 governance subscribers can dedupe + index by crew without
        decoding the payload — same convention as the existing
        `decision` envelope attr."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
            crew_name="mcq_ai_assist",
        )
        params = _decode_row(conn.cur.executed[0])
        env = json.loads(params["envelope"])
        assert env["crew_name"] == "mcq_ai_assist"

    @pytest.mark.asyncio
    async def test_envelope_carries_all_11_mandatory_fields(self) -> None:
        """Per CLAUDE.md §6 the envelope MUST carry: event_id +
        idempotency_key + tenant_id + gcid + occurred_at + published_at +
        traceparent + tracestate + source_project + source_service +
        schema_version. Plus chora_imda_dimension per ADR-141."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t-1",
            gcid="g-1",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
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
        assert env["tenant_id"] == "t-1"
        assert env["gcid"] == "g-1"
        assert env["traceparent"] == ("00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01")
        assert env["tracestate"] == "vendor=chora"

    @pytest.mark.asyncio
    async def test_envelope_carries_decision_for_subscriber_index(self) -> None:
        """The `decision` field lands in the envelope so D1 governance
        subscribers can index without payload decode (mirror of
        publish_refused's refusal_reason / model_armor_verdict pattern).
        """
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t",
            gcid="g",
            decision="completed_with_warning",
            attempt_count=2,
            max_retries=3,
            critic_notes="minor critic notes",
            quality_warning=True,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
            traceparent="",
            tracestate="",
        )
        params = _decode_row(conn.cur.executed[0])
        env = json.loads(params["envelope"])
        assert env["decision"] == "completed_with_warning"

    @pytest.mark.asyncio
    async def test_idempotency_key_deterministic_per_assist_id(self) -> None:
        """Idempotency key formula = ``agent_decision.{tenant_id}.{assist_id}.{agid}``
        — ONE event per (terminal run, agent). Same inputs (incl. agid) →
        same key → ON CONFLICT DO NOTHING swallows duplicate INSERTs on graph
        re-run from checkpoint (D6 P2). Distinct agids do NOT collide."""
        conn1 = _FakeAsyncConnection()
        w1 = AgentDecisionLogOutboxWriter(
            conn=conn1,
            source_project="chora-489812",
        )
        await w1.emit(
            assist_id="same-id",
            agid="qgen_question",
            tenant_id="tenant-A",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
        )
        conn2 = _FakeAsyncConnection()
        w2 = AgentDecisionLogOutboxWriter(
            conn=conn2,
            source_project="chora-489812",
        )
        await w2.emit(
            assist_id="same-id",
            agid="qgen_question",
            tenant_id="tenant-A",
            gcid="g",
            decision="rejected",  # decision varies but key stays the same
            attempt_count=2,
            max_retries=3,
            critic_notes="post-retry critic",
            quality_warning=True,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:31:00+00:00",
        )
        params1 = _decode_row(conn1.cur.executed[0])
        params2 = _decode_row(conn2.cur.executed[0])
        assert params1["idempotency_key"] == params2["idempotency_key"]
        assert params1["idempotency_key"] == "agent_decision.tenant-A.same-id.qgen_question"

    @pytest.mark.asyncio
    async def test_idempotency_key_distinct_per_tenant(self) -> None:
        """Two tenants with the same assist_id (collision in dev) get
        distinct keys (tenant_id is the prefix)."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t1",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t2",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
        )
        params_t1 = _decode_row(conn.cur.executed[0])
        params_t2 = _decode_row(conn.cur.executed[1])
        assert params_t1["idempotency_key"] == "agent_decision.t1.a-1.qgen_question"
        assert params_t2["idempotency_key"] == "agent_decision.t2.a-1.qgen_question"
        assert params_t1["idempotency_key"] != params_t2["idempotency_key"]

    @pytest.mark.asyncio
    async def test_default_dimension_when_caller_passes_blank(self) -> None:
        """If the caller passes an empty chora_imda_dimension, the writer
        coerces to the canonical accountability label so D1 evidence is
        never blank (mirror of publish_refused's empty-verdict coercion)."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="",  # blank from caller
            occurred_at="2026-05-17T15:30:00+00:00",
        )
        params = _decode_row(conn.cur.executed[0])
        env = json.loads(params["envelope"])
        assert env["chora_imda_dimension"] == IMDA_DIM_ACCOUNTABILITY

    @pytest.mark.asyncio
    async def test_occurred_at_propagated_into_envelope(self) -> None:
        """Caller-supplied occurred_at MUST be the envelope's occurred_at
        (the runner stamps it at emit-call time; we do NOT re-stamp).

        Subtle but load-bearing: BigQuery sink uses occurred_at as the
        partition key — if the writer re-stamped, the row's partition
        would be off by the queue-drain delay."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T10:00:00+00:00",
        )
        params = _decode_row(conn.cur.executed[0])
        env = json.loads(params["envelope"])
        assert env["occurred_at"] == "2026-05-17T10:00:00+00:00"

    @pytest.mark.asyncio
    async def test_malformed_occurred_at_falls_back_to_now(self) -> None:
        """Defensive: runner always stamps a valid ISO timestamp, but if a
        future caller passes a malformed string the row still lands (the
        envelope preserves the malformed value for forensic continuity;
        the SQL column gets ``now()`` so the INSERT doesn't fail)."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        row_id = await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="not-a-valid-iso",
        )
        assert row_id != ""
        params = _decode_row(conn.cur.executed[0])
        # SQL row column got a usable datetime (fallback path) ...
        import datetime as _dt

        assert isinstance(params["occurred_at"], _dt.datetime)
        # ... but the envelope keeps the malformed caller value for
        # forensic continuity.
        env = json.loads(params["envelope"])
        assert env["occurred_at"] == "not-a-valid-iso"

    @pytest.mark.asyncio
    async def test_event_id_unique_across_emits(self) -> None:
        """Each emit gets a fresh event_id (UUIDv4), even when
        idempotency_key collides — the deduplication is at the outbox
        layer via the unique index on idempotency_key, NOT event_id."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T10:00:00+00:00",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="t",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T10:00:00+00:00",
        )
        env1 = json.loads(_decode_row(conn.cur.executed[0])["envelope"])
        env2 = json.loads(_decode_row(conn.cur.executed[1])["envelope"])
        assert env1["event_id"] != env2["event_id"]
        # Idempotency key matches between rows (ON CONFLICT swallows it
        # at the DB layer in prod).
        assert (
            _decode_row(conn.cur.executed[0])["idempotency_key"] == _decode_row(conn.cur.executed[1])["idempotency_key"]
        )


# -----------------------------------------------------------------------------
# Multi-tenant isolation (D6 P3)
# -----------------------------------------------------------------------------


class TestMultiTenantIsolation:
    @pytest.mark.asyncio
    async def test_tenant_id_in_envelope_and_proto_envelope(self) -> None:
        """Pillar 3 — tenant_id MUST be reachable on the routing-attribute
        path (the JSON envelope column → Pub/Sub message attributes) AND in
        the proto payload's authoritative EventEnvelope (field 1) so the
        consumer reads the proto envelope, falling back to attrs if stripped
        by an intermediate proxy. Post-ADR-167 the proto core has no
        top-level tenant_id; it lives in the embedded envelope."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(
            conn=conn,
            source_project="chora-489812",
        )
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            tenant_id="tenant-XYZ",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
        )
        params = _decode_row(conn.cur.executed[0])
        env = json.loads(params["envelope"])
        # Routing-attribute path (JSON envelope column).
        assert env["tenant_id"] == "tenant-XYZ"
        assert params["tenant_id"] == "tenant-XYZ"
        # Proto payload's embedded EventEnvelope (field 1) carries tenant_id
        # @ envelope-field 3.
        proto_env_bytes = _walk(params["payload"])[1][0]
        proto_env = _walk(proto_env_bytes)
        assert proto_env[3][0].decode("utf-8") == "tenant-XYZ"


# -----------------------------------------------------------------------------
# Error propagation (P1 — pod-death survival ensures failure is loud)
# -----------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.asyncio
    async def test_db_error_propagates(self) -> None:
        """The runner wraps emit() in try/except and logs+continues
        (best-effort). But the writer itself MUST raise on DB failure so
        the runner's exception-handler observes the failure — silent
        swallowing here would hide a downed DB."""

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

        w = AgentDecisionLogOutboxWriter(
            conn=_FailingConn(),
            source_project="chora-489812",
        )
        with pytest.raises(RuntimeError, match="DB connection refused"):
            await w.emit(
                assist_id="a-1",
                agid="qgen_question",
                tenant_id="t",
                gcid="g",
                decision="accepted",
                attempt_count=1,
                max_retries=3,
                critic_notes="",
                quality_warning=False,
                chora_imda_dimension="accountability",
                occurred_at="2026-05-17T15:30:00+00:00",
            )


# -----------------------------------------------------------------------------
# Per-agent attribution contract (O+ /o/agents tile hydration)
#
# Field-4 agid is read from body.agid and MUST be a registry agid — one of
# qgen_question / qgen_critic / oe_evaluator / oe_moderator — NEVER the legacy
# hardcoded "qgen_crew" (which matched no /o/agents tile, leaving every qgen
# tile at 0 despite real rows). The idempotency key is suffixed with the agid
# so the two per-generation rows (qgen_question + qgen_critic) both land.
# -----------------------------------------------------------------------------


class TestPerAgentAgid:
    @pytest.mark.asyncio
    async def test_blank_agid_raises(self) -> None:
        """agid is load-bearing (it routes the /o/agents tile) — the writer
        refuses to emit a blank agid rather than silently land an unattributed
        row."""
        w = AgentDecisionLogOutboxWriter(
            conn=_FakeAsyncConnection(),
            source_project="chora-489812",
        )
        with pytest.raises(ValueError, match="agid"):
            await w.emit(
                assist_id="a-1",
                agid="",
                tenant_id="t",
                gcid="g",
                decision="accepted",
                attempt_count=1,
                max_retries=3,
                critic_notes="",
                quality_warning=False,
                chora_imda_dimension="accountability",
                occurred_at="2026-05-17T15:30:00+00:00",
            )

    @pytest.mark.asyncio
    async def test_each_registry_agid_lands_in_proto_field4(self) -> None:
        for agid in ("qgen_question", "qgen_critic", "oe_evaluator", "oe_moderator"):
            conn = _FakeAsyncConnection()
            w = AgentDecisionLogOutboxWriter(conn=conn, source_project="chora-489812")
            await w.emit(
                assist_id="a-1",
                agid=agid,
                tenant_id="t",
                gcid="g",
                decision="accepted",
                attempt_count=1,
                max_retries=3,
                critic_notes="",
                quality_warning=False,
                chora_imda_dimension="accountability",
                occurred_at="2026-05-17T15:30:00+00:00",
            )
            d = _decode_proto_payload(_decode_row(conn.cur.executed[0])["payload"])
            assert d["agid"] == agid
            assert d["agid"] != "qgen_crew"

    @pytest.mark.asyncio
    async def test_question_type_lands_in_attributes(self) -> None:
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(conn=conn, source_project="chora-489812")
        await w.emit(
            assist_id="a-1",
            agid="qgen_question",
            question_type="oe",
            tenant_id="t",
            gcid="g",
            decision="accepted",
            attempt_count=1,
            max_retries=3,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension="accountability",
            occurred_at="2026-05-17T15:30:00+00:00",
        )
        d = _decode_proto_payload(_decode_row(conn.cur.executed[0])["payload"])
        assert d["attributes"]["question_type"] == "oe"

    @pytest.mark.asyncio
    async def test_idempotency_key_distinct_per_agid(self) -> None:
        """Same tenant+assist, different agid → distinct idempotency keys so the
        qgen_question + qgen_critic rows for ONE generation both land (ON
        CONFLICT no longer collapses them into a single row)."""
        conn = _FakeAsyncConnection()
        w = AgentDecisionLogOutboxWriter(conn=conn, source_project="chora-489812")
        for agid in ("qgen_question", "qgen_critic"):
            await w.emit(
                assist_id="job-1",
                agid=agid,
                tenant_id="t-1",
                gcid="g",
                decision="accepted",
                attempt_count=1,
                max_retries=3,
                critic_notes="",
                quality_warning=False,
                chora_imda_dimension="accountability",
                occurred_at="2026-05-17T15:30:00+00:00",
            )
        k0 = _decode_row(conn.cur.executed[0])["idempotency_key"]
        k1 = _decode_row(conn.cur.executed[1])["idempotency_key"]
        assert k0 == "agent_decision.t-1.job-1.qgen_question"
        assert k1 == "agent_decision.t-1.job-1.qgen_critic"
        assert k0 != k1
