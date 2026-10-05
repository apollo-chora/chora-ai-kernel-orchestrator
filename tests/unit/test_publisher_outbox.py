"""Tests for ``TransactionalOutboxPublisher`` — the B.6.2.a producer-side
durable-emission adapter for the AI Kernel orchestrator.

Mirrors ``services/chora-closure-orchestrator/tests/unit/test_publisher_outbox.py``
faithfully. The AI Kernel adapter implements ``AIKernelEventPublisher`` and
writes each emitted event to ``chora_ai_kernel.ai_kernel_outbox_events``
(migration ``0003_outbox.sql``).

Topics emitted live under the ``chora.ai_kernel.*`` prefix per
``chora-contracts/CLAUDE.md`` taxonomy and the canonical event protos in
``chora-contracts/proto/events/ai_kernel/``.

Why this lives next to the closure outbox tests but in a parallel table:

* The orchestrator emits invocation lifecycle + crew lifecycle + guardrail
  events for every workflow turn. Each row is durable so a pod-death
  between the LangGraph checkpoint commit and the Pub/Sub publish does
  NOT lose the event (D6 first-class resilience per
  ``feedback_d6_resilience_first_class``).
* Multi-tenant + multi-workflow isolation is enforced at the row level —
  the D6.3 expanded scope requires tenant_id to be a queryable column,
  not just a payload field.
* Idempotency is enforced via a unique index on ``idempotency_key``.
* Dispatcher (background) polls ``status='pending'`` rows + publishes to
  Cloud Pub/Sub, then marks ``status='published'``.

Unit tests use a mocked psycopg AsyncConnection. Live DB integration
exists at ``tests/integration/test_outbox_writer_live.py`` (deferred to
a separate sub-task; marked ``live`` — skipped without cloud creds).
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.events.payloads import (
    AgentTerminated,
    CrewComposed,
    CrewExecuted,
    GuardrailEvaluated,
    ModelInvocationCompleted,
    ModelInvocationFailed,
    ModelInvoked,
)
from chora_ai_kernel_orchestrator.adapter.events.publisher_outbox import (
    TransactionalOutboxPublisher,
)

# ---------------------------------------------------------------------------
# Mock connection helpers
# ---------------------------------------------------------------------------


class _MockCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []

    async def __aenter__(self) -> _MockCursor:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))


class _MockConn:
    def __init__(self) -> None:
        self.cursor_obj = _MockCursor()
        self.committed = False

    def cursor(self) -> _MockCursor:
        return self.cursor_obj

    async def commit(self) -> None:
        self.committed = True


def _now() -> _dt.datetime:
    return _dt.datetime(2026, 5, 12, 8, 0, 0, tzinfo=_dt.UTC)


WORKFLOW_ID = "d42940c4-3edf-4a3e-83ed-938c5fef441d"
INVOCATION_ID = "01970000-7777-7000-a000-000000000001"
CREW_ID = "01970000-aaaa-7000-b000-000000000001"
TENANT_ID = "01970000-0000-7000-8000-000000000001"
GCID = "01970000-0000-7000-9000-000000000001"
AGID = "01970000-0000-7000-c000-000000000001"


# ---------------------------------------------------------------------------
# Constructor + protocol shape
# ---------------------------------------------------------------------------


class TestTransactionalOutboxPublisherInit:
    def test_constructor_captures_connection(self) -> None:
        conn = _MockConn()
        pub = TransactionalOutboxPublisher(
            conn=conn,  # type: ignore[arg-type]
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        assert pub._conn is conn  # noqa: SLF001
        assert pub._source_project == "chora-489812"  # noqa: SLF001
        assert pub._source_service == "chora-ai-kernel-orchestrator"  # noqa: SLF001

    def test_rejects_empty_source_project(self) -> None:
        with pytest.raises(ValueError, match="source_project required"):
            TransactionalOutboxPublisher(
                conn=_MockConn(),  # type: ignore[arg-type]
                source_project="",
                source_service="x",
            )

    def test_rejects_empty_source_service(self) -> None:
        with pytest.raises(ValueError, match="source_service required"):
            TransactionalOutboxPublisher(
                conn=_MockConn(),  # type: ignore[arg-type]
                source_project="x",
                source_service="",
            )


# ---------------------------------------------------------------------------
# ModelInvoked — the canonical write-path test
# ---------------------------------------------------------------------------


class TestPublishModelInvoked:
    @pytest.mark.asyncio
    async def test_writes_one_row_to_outbox(self) -> None:
        conn = _MockConn()
        pub = TransactionalOutboxPublisher(
            conn=conn,  # type: ignore[arg-type]
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        event = ModelInvoked(
            invocation_id=INVOCATION_ID,
            workflow_id=WORKFLOW_ID,
            tenant_id=TENANT_ID,
            gcid=GCID,
            agid=AGID,
            model_id="gemini-2.5-flash-001",
            model_kind="MODEL_KIND_GEMINI_FLASH",
            prompt_hash="sha256:abc",
            invoked_at=_now(),
            traceparent="00-trace-span-01",
            tracestate="vendor=chora",
        )

        await pub.publish_model_invoked(event)

        assert len(conn.cursor_obj.executed) == 1
        sql, params = conn.cursor_obj.executed[0]
        assert "INSERT INTO ai_kernel_outbox_events" in sql

    @pytest.mark.asyncio
    async def test_row_carries_correct_topic_event_type(self) -> None:
        conn = _MockConn()
        pub = TransactionalOutboxPublisher(
            conn=conn,  # type: ignore[arg-type]
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        await pub.publish_model_invoked(
            ModelInvoked(
                invocation_id=INVOCATION_ID,
                workflow_id=WORKFLOW_ID,
                tenant_id=TENANT_ID,
                gcid=GCID,
                agid=AGID,
                model_id="gemini-2.5-flash-001",
                model_kind="MODEL_KIND_GEMINI_FLASH",
                prompt_hash="sha256:abc",
                invoked_at=_now(),
            )
        )

        _, params = conn.cursor_obj.executed[0]
        params_dict = _params_to_dict(params)
        assert params_dict["topic"] == "chora.ai_kernel.invocation.invoked.v1"
        assert params_dict["event_type"] == "ai_kernel.invocation.invoked"

    @pytest.mark.asyncio
    async def test_row_carries_tenant_isolation_columns(self) -> None:
        """D6.3 multi-tenant chaos — tenant_id MUST be a queryable column."""
        conn = _MockConn()
        pub = TransactionalOutboxPublisher(
            conn=conn,  # type: ignore[arg-type]
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        await pub.publish_model_invoked(
            ModelInvoked(
                invocation_id=INVOCATION_ID,
                workflow_id=WORKFLOW_ID,
                tenant_id=TENANT_ID,
                gcid=GCID,
                agid=AGID,
                model_id="gemini-2.5-flash-001",
                model_kind="MODEL_KIND_GEMINI_FLASH",
                prompt_hash="sha256:abc",
                invoked_at=_now(),
            )
        )

        _, params = conn.cursor_obj.executed[0]
        params_dict = _params_to_dict(params)
        assert params_dict["tenant_id"] == TENANT_ID
        assert params_dict["workflow_id"] == WORKFLOW_ID
        assert params_dict["gcid"] == GCID

    @pytest.mark.asyncio
    async def test_envelope_carries_mandatory_fields(self) -> None:
        """Per CLAUDE.md cross-cutting rule, envelope MUST carry the 11
        mandatory fields. Test the 10 stable ones at write time
        (published_at is NULL until dispatched)."""
        conn = _MockConn()
        pub = TransactionalOutboxPublisher(
            conn=conn,  # type: ignore[arg-type]
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        await pub.publish_model_invoked(
            ModelInvoked(
                invocation_id=INVOCATION_ID,
                workflow_id=WORKFLOW_ID,
                tenant_id=TENANT_ID,
                gcid=GCID,
                agid=AGID,
                model_id="gemini-2.5-flash-001",
                model_kind="MODEL_KIND_GEMINI_FLASH",
                prompt_hash="sha256:abc",
                invoked_at=_now(),
                traceparent="00-trace-span-01",
                tracestate="vendor=chora",
            )
        )

        _, params = conn.cursor_obj.executed[0]
        params_dict = _params_to_dict(params)
        envelope = json.loads(params_dict["envelope"])
        for required in (
            "event_id",
            "idempotency_key",
            "tenant_id",
            "gcid",
            "occurred_at",
            "traceparent",
            "tracestate",
            "source_project",
            "source_service",
            "schema_version",
        ):
            assert required in envelope, f"envelope missing field: {required}"
        assert envelope["source_project"] == "chora-489812"
        assert envelope["source_service"] == "chora-ai-kernel-orchestrator"
        assert envelope["traceparent"] == "00-trace-span-01"

    @pytest.mark.asyncio
    async def test_status_pending_on_initial_write(self) -> None:
        conn = _MockConn()
        pub = TransactionalOutboxPublisher(
            conn=conn,  # type: ignore[arg-type]
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )
        await pub.publish_model_invoked(
            ModelInvoked(
                invocation_id=INVOCATION_ID,
                workflow_id=WORKFLOW_ID,
                tenant_id=TENANT_ID,
                gcid=GCID,
                agid=AGID,
                model_id="gemini-2.5-flash-001",
                model_kind="MODEL_KIND_GEMINI_FLASH",
                prompt_hash="sha256:abc",
                invoked_at=_now(),
            )
        )

        _, params = conn.cursor_obj.executed[0]
        params_dict = _params_to_dict(params)
        assert params_dict.get("status", "pending") == "pending"

    @pytest.mark.asyncio
    async def test_idempotency_key_unique_per_event(self) -> None:
        """Two distinct events get distinct idempotency keys (UUIDv7 random)."""
        conn = _MockConn()
        pub = TransactionalOutboxPublisher(
            conn=conn,  # type: ignore[arg-type]
            source_project="chora-489812",
            source_service="chora-ai-kernel-orchestrator",
        )

        for _ in range(2):
            await pub.publish_model_invoked(
                ModelInvoked(
                    invocation_id=INVOCATION_ID,
                    workflow_id=WORKFLOW_ID,
                    tenant_id=TENANT_ID,
                    gcid=GCID,
                    agid=AGID,
                    model_id="gemini-2.5-flash-001",
                    model_kind="MODEL_KIND_GEMINI_FLASH",
                    prompt_hash="sha256:abc",
                    invoked_at=_now(),
                )
            )

        keys = [_params_to_dict(p)["idempotency_key"] for _, p in conn.cursor_obj.executed]
        assert keys[0] != keys[1]


# ---------------------------------------------------------------------------
# Coverage of every protocol method
# ---------------------------------------------------------------------------


class TestEveryProtocolMethodWritesToOutbox:
    @pytest.mark.asyncio
    async def test_publish_model_invocation_completed(self) -> None:
        conn, pub = _fresh()
        await pub.publish_model_invocation_completed(
            ModelInvocationCompleted(
                invocation_id=INVOCATION_ID,
                workflow_id=WORKFLOW_ID,
                tenant_id=TENANT_ID,
                gcid=GCID,
                agid=AGID,
                model_id="gemini-2.5-flash-001",
                model_kind="MODEL_KIND_GEMINI_FLASH",
                prompt_hash="sha256:abc",
                response_hash="sha256:def",
                latency_ms=812,
                tokens_in=123,
                tokens_out=456,
                completed_at=_now(),
            )
        )
        assert _last_topic(conn) == "chora.ai_kernel.invocation.completed.v1"

    @pytest.mark.asyncio
    async def test_publish_model_invocation_failed(self) -> None:
        conn, pub = _fresh()
        await pub.publish_model_invocation_failed(
            ModelInvocationFailed(
                invocation_id=INVOCATION_ID,
                workflow_id=WORKFLOW_ID,
                tenant_id=TENANT_ID,
                gcid=GCID,
                agid=AGID,
                model_id="gemini-2.5-flash-001",
                model_kind="MODEL_KIND_GEMINI_FLASH",
                prompt_hash="sha256:abc",
                error_code="QUOTA_EXCEEDED",
                error_detail="HTTP 429 from Vertex",
                latency_ms=42,
                failed_at=_now(),
            )
        )
        assert _last_topic(conn) == "chora.ai_kernel.invocation.failed.v1"

    @pytest.mark.asyncio
    async def test_publish_crew_composed(self) -> None:
        conn, pub = _fresh()
        await pub.publish_crew_composed(
            CrewComposed(
                crew_id=CREW_ID,
                workflow_id=WORKFLOW_ID,
                tenant_id=TENANT_ID,
                gcid=GCID,
                crew_kind="CREW_KIND_FIXED_CORE_PER_DOMAIN",
                member_agids=[AGID],
                composed_at=_now(),
            )
        )
        assert _last_topic(conn) == "chora.ai_kernel.crew.composed.v1"

    @pytest.mark.asyncio
    async def test_publish_crew_executed(self) -> None:
        conn, pub = _fresh()
        await pub.publish_crew_executed(
            CrewExecuted(
                crew_id=CREW_ID,
                workflow_id=WORKFLOW_ID,
                tenant_id=TENANT_ID,
                gcid=GCID,
                crew_kind="CREW_KIND_FIXED_CORE_PER_DOMAIN",
                duration_ms=2410,
                outcome="completed",
                executed_at=_now(),
            )
        )
        assert _last_topic(conn) == "chora.ai_kernel.crew.executed.v1"

    @pytest.mark.asyncio
    async def test_publish_guardrail_evaluated(self) -> None:
        conn, pub = _fresh()
        await pub.publish_guardrail_evaluated(
            GuardrailEvaluated(
                invocation_id=INVOCATION_ID,
                workflow_id=WORKFLOW_ID,
                tenant_id=TENANT_ID,
                gcid=GCID,
                agid=AGID,
                tier="model_armor",
                outcome="passed",
                evaluated_at=_now(),
            )
        )
        assert _last_topic(conn) == "chora.ai_kernel.guardrail.evaluated.v1"


# ---------------------------------------------------------------------------
# AgentTerminated — W3 foundation Phase 3 (2026-05-12)
# ---------------------------------------------------------------------------


EXECUTION_ID = "thread-01970000-0000-7000-a000-000000000001"


class TestPublishAgentTerminated:
    """``publish_agent_terminated`` writes one outbox row to the canonical
    ``chora.ai_kernel.agent.terminated.v1`` topic. Reuses the workflow_id
    column as the execution-correlation key (LangGraph thread_id).
    """

    @pytest.mark.asyncio
    async def test_writes_one_row_to_outbox(self) -> None:
        conn, pub = _fresh()
        await pub.publish_agent_terminated(
            AgentTerminated(
                agent_id="ai_kernel_orchestrator",
                execution_id=EXECUTION_ID,
                termination_code="AGENT_TERMINATION_CODE_SUCCESS",
                runtime="AGENT_EXECUTION_RUNTIME_LANGGRAPH_PYTHON",
                tenant_id=TENANT_ID,
                gcid=GCID,
                terminated_at=_now(),
            )
        )
        assert len(conn.cursor_obj.executed) == 1
        sql, _ = conn.cursor_obj.executed[0]
        assert "INSERT INTO ai_kernel_outbox_events" in sql

    @pytest.mark.asyncio
    async def test_row_carries_canonical_topic_and_event_type(self) -> None:
        conn, pub = _fresh()
        await pub.publish_agent_terminated(
            AgentTerminated(
                agent_id="ai_kernel_orchestrator",
                execution_id=EXECUTION_ID,
                termination_code="AGENT_TERMINATION_CODE_SUCCESS",
                runtime="AGENT_EXECUTION_RUNTIME_LANGGRAPH_PYTHON",
                tenant_id=TENANT_ID,
                gcid=GCID,
                terminated_at=_now(),
            )
        )
        params = _params_to_dict(conn.cursor_obj.executed[0][1])
        assert params["topic"] == "chora.ai_kernel.agent.terminated.v1"
        assert params["event_type"] == "ai_kernel.agent.terminated"

    @pytest.mark.asyncio
    async def test_execution_id_lands_in_workflow_id_column(self) -> None:
        """workflow_id column is the execution-correlation key for AgentTerminated."""
        conn, pub = _fresh()
        await pub.publish_agent_terminated(
            AgentTerminated(
                agent_id="ai_kernel_orchestrator",
                execution_id=EXECUTION_ID,
                termination_code="AGENT_TERMINATION_CODE_SUCCESS",
                runtime="AGENT_EXECUTION_RUNTIME_LANGGRAPH_PYTHON",
                tenant_id=TENANT_ID,
                gcid=GCID,
                terminated_at=_now(),
            )
        )
        params = _params_to_dict(conn.cursor_obj.executed[0][1])
        assert params["workflow_id"] == EXECUTION_ID
        assert params["tenant_id"] == TENANT_ID
        assert params["gcid"] == GCID

    @pytest.mark.asyncio
    async def test_body_carries_agent_terminated_shape(self) -> None:
        conn, pub = _fresh()
        await pub.publish_agent_terminated(
            AgentTerminated(
                agent_id="ai_kernel_orchestrator",
                execution_id=EXECUTION_ID,
                termination_code="AGENT_TERMINATION_CODE_RUNTIME_ERROR",
                runtime="AGENT_EXECUTION_RUNTIME_LANGGRAPH_PYTHON",
                tenant_id=TENANT_ID,
                gcid=GCID,
                terminated_at=_now(),
                crew_pattern="P4_SIX_AGENT_GATE",
                last_state_node="governance_gate",
                last_error_message="executor unreachable",
                iteration_count=3,
                partial_state={"step": "compose"},
                current_span_id="0123456789abcdef",
                traceparent="00-trace-span-02",
                tracestate="vendor=chora",
            )
        )
        params = _params_to_dict(conn.cursor_obj.executed[0][1])
        body = json.loads(params["payload"].decode("utf-8"))
        assert body["agent_id"] == "ai_kernel_orchestrator"
        assert body["execution_id"] == EXECUTION_ID
        assert body["termination_code"] == "AGENT_TERMINATION_CODE_RUNTIME_ERROR"
        assert body["crew_pattern"] == "P4_SIX_AGENT_GATE"
        ctx = body["context"]
        assert ctx["last_state_node"] == "governance_gate"
        assert ctx["iteration_count"] == 3
        assert ctx["partial_state"] == {"step": "compose"}
        assert ctx["current_span_id"] == "0123456789abcdef"
        envelope = json.loads(params["envelope"])
        assert envelope["traceparent"] == "00-trace-span-02"
        assert envelope["tracestate"] == "vendor=chora"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> tuple[_MockConn, TransactionalOutboxPublisher]:
    conn = _MockConn()
    pub = TransactionalOutboxPublisher(
        conn=conn,  # type: ignore[arg-type]
        source_project="chora-489812",
        source_service="chora-ai-kernel-orchestrator",
    )
    return conn, pub


def _last_topic(conn: _MockConn) -> str:
    assert conn.cursor_obj.executed
    _, params = conn.cursor_obj.executed[-1]
    return _params_to_dict(params)["topic"]


def _params_to_dict(params: Any) -> dict[str, Any]:
    """The adapter binds params as a dict (psycopg supports %(name)s); the
    tests assert on that dict shape. If the adapter later switches to a
    positional tuple, this helper must be updated to keep tests stable."""
    if isinstance(params, dict):
        return params
    if isinstance(params, (list, tuple)):
        raise AssertionError("Adapter must bind INSERT params by name (dict), not positional")
    raise AssertionError(f"unexpected params type: {type(params)}")
