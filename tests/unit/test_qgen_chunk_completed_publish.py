"""CHO-2398 (ADR-251 D5): the chunk_completed.v1 publish.

publish_chunk gains the dedicated domain-data event: one BINARY-proto
``chora.creation.ai_assist.chunk_completed.v1`` outbox row per finished
chunk, idempotency key ``ai_assist.chunk_completed.{chunk_index}.{assist_id}``,
candidates in PUBLISHED shape (specs already stripped by the render node,
quality_warning + critic_notes included) plus the ADR-locked per-chunk
counts (warned + the render node's honest image tallies). progress.v1 stays
trace-only; the terminal completed.v1 is untouched (contract redundancy is
deliberate).

Absent publisher (None) keeps publish_chunk_node byte-identical for every
legacy caller and test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    encode_ai_assist_chunk_completed,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_publisher import (
    TOPIC_AI_ASSIST_CHUNK_COMPLETED,
    QGenCrewTerminalOutboxWriter,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    build_qgen_crew_graph,
    publish_chunk_node,
    render_image_set_node,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
    QGenBatchRunner,
)
from tests.integration.test_qgen_crew_set import (
    _FakeGuardrail,
    _FakeSetExecutor,
    _mcq,
    _wrapper,
)
from tests.unit.test_qgen_crew_runner import _started_event

TENANT = "11111111-1111-7111-8111-111111111111"
GCID = "00000000-0000-7000-8000-000000001999"


# -----------------------------------------------------------------------------
# Minimal proto3 walker (mirrors tests/unit/test_proto_wire_encoder.py)
# -----------------------------------------------------------------------------


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = data[offset]
        result |= (b & 0x7F) << shift
        offset += 1
        if not b & 0x80:
            return result, offset
        shift += 7


def _by_field(data: bytes) -> dict[int, Any]:
    out: dict[int, Any] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        fnum, wire = tag >> 3, tag & 0x07
        if wire == 0:
            value, offset = _read_varint(data, offset)
            out[fnum] = value
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            out[fnum] = data[offset : offset + length]
            offset += length
        else:
            raise ValueError(f"unsupported wire type {wire}")
    return out


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------


def _envelope() -> dict[str, Any]:
    return {
        "event_id": "evt-1",
        "idempotency_key": "ai_assist.chunk_completed.1.job-1",
        "tenant_id": TENANT,
        "gcid": GCID,
        "occurred_at": "2026-08-16T12:00:00+00:00",
        "published_at": "2026-08-16T12:00:00+00:00",
        "source_project": "chora-489812",
        "source_service": "chora-ai-kernel-orchestrator",
        "schema_version": "1",
    }


def test_encoder_emits_every_locked_field() -> None:
    body = {
        "assist_id": "job-1",
        "tenant_id": TENANT,
        "author_gcid": GCID,
        "chunk_index": 1,
        "chunk_count": 3,
        "candidates_payload_json": '[{"stem": "s"}]',
        "candidate_count": 1,
        "warned_count": 1,
        "images_rendered": 2,
        "images_dropped": 1,
        "images_failed": 0,
        "images_skipped": 3,
    }
    fields = _by_field(encode_ai_assist_chunk_completed(_envelope(), body))
    assert fields[2].decode() == "job-1"
    assert fields[3].decode() == TENANT
    assert fields[4].decode() == GCID
    assert fields[5] == 1 and fields[6] == 3
    assert json.loads(fields[7].decode())[0]["stem"] == "s"
    assert fields[8] == 1
    assert fields[10] == 1
    assert fields[11] == 2 and fields[12] == 1 and fields[14] == 3
    assert 13 not in fields, "proto3 zero-valued int32 is elided (images_failed=0)"
    env = _by_field(fields[1])
    assert env[2].decode() == "ai_assist.chunk_completed.1.job-1"
    assert 9 in fields, "chunk_completed_at synthesised from occurred_at"


# -----------------------------------------------------------------------------
# Writer.publish_chunk_completed
# -----------------------------------------------------------------------------


@dataclass
class _FakeCursor:
    executed: list[tuple[str, Any]] = field(default_factory=list)

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


@dataclass
class _FakeConn:
    cur: _FakeCursor = field(default_factory=_FakeCursor)
    commits: list[int] = field(default_factory=list)

    def cursor(self) -> _FakeCursor:
        return self.cur

    async def commit(self) -> None:
        self.commits.append(len(self.cur.executed))


class _NeverDeleteRegistry:
    async def delete_in_tx(self, assist_id: str) -> None:
        raise AssertionError("chunk_completed is NOT terminal; no registry delete")


async def test_writer_chunk_publish_key_topic_and_binary_payload() -> None:
    conn = _FakeConn()
    writer = QGenCrewTerminalOutboxWriter(
        conn=conn,
        source_project="chora-489812",
        inflight_registry=_NeverDeleteRegistry(),
    )
    await writer.publish_chunk_completed(
        assist_id="job-1",
        tenant_id=TENANT,
        author_gcid=GCID,
        chunk_index=2,
        chunk_count=3,
        candidates_payload_json='[{"stem": "s"}]',
        candidate_count=1,
        warned_count=0,
        images_rendered=0,
        images_dropped=0,
        images_failed=0,
        images_skipped=0,
    )
    assert conn.commits == [1]
    sql, params = conn.cur.executed[0]
    assert "ai_kernel_outbox_events" in sql
    assert params["idempotency_key"] == "ai_assist.chunk_completed.2.job-1"
    assert params["topic"] == TOPIC_AI_ASSIST_CHUNK_COMPLETED
    fields = _by_field(params["payload"])
    assert fields[2].decode() == "job-1"
    assert fields[5] == 2 and fields[6] == 3


# -----------------------------------------------------------------------------
# render_image_set_node: structured tallies for the chunk event
# -----------------------------------------------------------------------------


async def test_render_node_returns_structured_tallies_zero_spec_path() -> None:
    state = {
        "accepted_set": [
            {"stem": "clean", "question_type": "mcq"},
            {
                "stem": "warned",
                "question_type": "mcq",
                "quality_warning": True,
                "image_specs": [{"mode": "scene", "source": "x"}],
            },
        ],
        "chunk_plan": [{"question_type": "mcq", "count": 2, "max_images": 0}],
    }
    delta = await render_image_set_node(state, kroki=None, gcs=None)
    tallies = delta["chunk_image_tallies"]
    assert tallies == {"rendered": 0, "dropped": 0, "failed": 0, "skipped": 1}
    assert all("image_specs" not in c for c in delta["accepted_set"])


# -----------------------------------------------------------------------------
# publish_chunk_node: the publish hookup
# -----------------------------------------------------------------------------


class _FakeChunkPublisher:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def publish_chunk_completed(self, **kw: Any) -> str:
        self.calls.append(kw)
        return "row-id"


def _chunk_state(**extra: Any) -> dict[str, Any]:
    return {
        "job_id": "job-1",
        "tenant_id": TENANT,
        "author_gcid": GCID,
        "chunk_index": 0,
        "chunk_count": 2,
        "chunk_plans": [
            [{"question_type": "mcq", "count": 2, "max_images": 0}],
            [{"question_type": "mcq", "count": 1, "max_images": 0}],
        ],
        "accepted_set": [
            {"stem": "a", "question_type": "mcq"},
            {"stem": "b", "question_type": "mcq", "quality_warning": True, "critic_notes": "residual concern"},
        ],
        "chunk_image_tallies": {
            "rendered": 1,
            "dropped": 0,
            "failed": 0,
            "skipped": 1,
        },
        **extra,
    }


async def test_publish_chunk_node_publishes_the_chunk_event() -> None:
    pub = _FakeChunkPublisher()
    delta = await publish_chunk_node(_chunk_state(), publisher=pub)
    assert len(pub.calls) == 1
    call = pub.calls[0]
    assert call["assist_id"] == "job-1"
    assert call["chunk_index"] == 0 and call["chunk_count"] == 2
    cands = json.loads(call["candidates_payload_json"])
    assert [c["stem"] for c in cands] == ["a", "b"]
    assert cands[1]["quality_warning"] is True
    assert cands[1]["critic_notes"] == "residual concern"
    assert call["candidate_count"] == 2
    assert call["warned_count"] == 1
    assert call["images_rendered"] == 1 and call["images_skipped"] == 1
    # Working-state reset covers the tallies too.
    assert delta["chunk_image_tallies"] == {}


async def test_publish_chunk_node_without_publisher_is_byte_identical() -> None:
    state = _chunk_state()
    delta = await publish_chunk_node(state, publisher=None)
    assert delta["completed_candidates"], "fold still happens"
    assert delta["chunk_index"] == 1


# -----------------------------------------------------------------------------
# Full-graph proof: one chunk event per chunk with correct indices
# -----------------------------------------------------------------------------


class _TerminalAndChunkPublisher:
    """Duck-types the terminal publisher the batch runner needs PLUS the
    chunk publisher the graph node calls."""

    def __init__(self) -> None:
        self.chunk_calls: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []

    async def publish_chunk_completed(self, **kw: Any) -> str:
        self.chunk_calls.append(kw)
        return "row"

    async def publish_completed(self, **kw: Any) -> str:
        self.completed.append(kw)
        return "row"

    async def publish_refused(self, **kw: Any) -> str:  # pragma: no cover
        raise AssertionError("refusal not expected in this fixture")


@pytest.mark.asyncio
async def test_full_graph_emits_one_chunk_event_per_chunk() -> None:
    executor = _FakeSetExecutor(
        generate_queue=[
            _wrapper([_mcq(i) for i in range(10)]),
            _wrapper([_mcq(i) for i in range(10, 12)]),
        ]
    )
    pub = _TerminalAndChunkPublisher()
    graph = build_qgen_crew_graph(
        executor=executor,
        guardrail=_FakeGuardrail(),
        checkpointer=MemorySaver(),
        kroki=None,
        gcs=None,
        publisher=pub,
    )
    runner = QGenBatchRunner(graph=graph, publisher=pub)
    event = _started_event(
        assist_id="job-chunks",
        content_type="mixed",
        question_type="mixed",
        prompt="Generate a set.",
        job_kind="batch",
        requested_count=12,
        type_plan=[{"question_type": "mcq", "count": 12, "max_images": 0}],
    )
    await runner.handle_started(event)
    assert [c["chunk_index"] for c in pub.chunk_calls] == [0, 1]
    assert [c["candidate_count"] for c in pub.chunk_calls] == [10, 2]
    assert all(c["chunk_count"] == 2 for c in pub.chunk_calls)
    assert len(pub.completed) == 1, "terminal still publishes the full set"
