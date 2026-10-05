"""Tests for ``OutboxDispatcher`` — drains ``ai_kernel_outbox_events`` rows
with status='pending' to Cloud Pub/Sub.

Mirrors ``services/chora-closure-orchestrator/tests/unit/test_outbox_dispatcher.py``
faithfully. The dispatcher implements the second half of the transactional
outbox pattern (the first half is ``TransactionalOutboxPublisher`` in
B.6.2.a). Together they realise durable emission per
``feedback_d6_resilience_first_class``:

1. Saga node writes event to ``ai_kernel_outbox_events`` (DB-durable).
2. Dispatcher polls + publishes via ``GoogleCloudPubSubPublisher``.
3. On success: UPDATE status='published', published_at=now().
4. On failure: UPDATE retry_count++, last_error=..., last_attempt_at.
5. After max_attempts retries: INSERT into ``ai_kernel_outbox_dead_letters``
   + UPDATE status='deadlettered'.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub import (
    OutboxDispatcher,
    OutboxRow,
)


def _row(
    *,
    row_id: str = "row-1",
    workflow_id: str = "wf-1",
    tenant_id: str = "tenant-1",
    gcid: str = "gcid-1",
    topic: str = "chora.ai_kernel.invocation.invoked.v1",
    event_type: str = "ai_kernel.invocation.invoked",
    retry_count: int = 0,
) -> OutboxRow:
    return OutboxRow(
        id=row_id,
        workflow_id=workflow_id,
        tenant_id=tenant_id,
        gcid=gcid,
        event_type=event_type,
        topic=topic,
        payload=b'{"x":1}',
        envelope={
            "event_id": row_id,
            "idempotency_key": row_id,
            "tenant_id": tenant_id,
            "gcid": gcid,
            "occurred_at": "2026-05-12T08:00:00+00:00",
            "traceparent": "",
            "tracestate": "",
            "source_project": "chora-489812",
            "source_service": "chora-ai-kernel-orchestrator",
            "schema_version": "1",
        },
        idempotency_key=row_id,
        retry_count=retry_count,
    )


class _MockStore:
    """In-memory stand-in for the DB store the dispatcher uses."""

    def __init__(self, pending: list[OutboxRow]) -> None:
        self.pending = list(pending)
        self.marked_published: list[str] = []
        self.marked_failed: list[tuple[str, str]] = []
        self.deadlettered: list[tuple[str, str, int]] = []

    async def fetch_pending(self, *, limit: int) -> list[OutboxRow]:
        return self.pending[:limit]

    async def mark_published(self, row_id: str) -> None:
        self.marked_published.append(row_id)
        self.pending = [r for r in self.pending if r.id != row_id]

    async def mark_failed(self, row_id: str, error: str) -> None:
        self.marked_failed.append((row_id, error))
        for r in self.pending:
            if r.id == row_id:
                r.retry_count += 1
                break

    async def deadletter(self, row_id: str, failure_reason: str, attempt_count: int) -> None:
        self.deadlettered.append((row_id, failure_reason, attempt_count))
        self.pending = [r for r in self.pending if r.id != row_id]


class TestOutboxDispatcherDrain:
    @pytest.mark.asyncio
    async def test_drains_pending_rows_to_pubsub(self) -> None:
        store = _MockStore(pending=[_row(row_id="a"), _row(row_id="b")])
        publisher = AsyncMock()
        publisher.publish = AsyncMock(side_effect=["msg-a", "msg-b"])
        d = OutboxDispatcher(
            store=store,  # type: ignore[arg-type]
            publisher=publisher,
            worker_id="test-worker",
            max_attempts=5,
        )

        n = await d.drain_once(batch_size=10)

        assert n == 2
        assert store.marked_published == ["a", "b"]
        assert store.pending == []

    @pytest.mark.asyncio
    async def test_publish_failure_marks_row_failed(self) -> None:
        store = _MockStore(pending=[_row(row_id="a")])
        publisher = AsyncMock()
        publisher.publish = AsyncMock(
            side_effect=RuntimeError("Pub/Sub 503"),
        )
        d = OutboxDispatcher(
            store=store,  # type: ignore[arg-type]
            publisher=publisher,
            worker_id="test-worker",
            max_attempts=5,
        )

        n = await d.drain_once(batch_size=10)

        assert n == 0  # no successful publishes
        assert len(store.marked_failed) == 1
        assert store.marked_failed[0][0] == "a"
        assert "Pub/Sub 503" in store.marked_failed[0][1]
        # Row stays in pending for next attempt — NOT marked published or deadlettered.
        assert any(r.id == "a" for r in store.pending)
        assert store.deadlettered == []

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded_deadletters(self) -> None:
        """A row that has already failed max_attempts-1 times gets
        deadlettered on its next failure (i.e., the max_attempts-th attempt)."""
        store = _MockStore(pending=[_row(row_id="a", retry_count=4)])
        publisher = AsyncMock()
        publisher.publish = AsyncMock(
            side_effect=RuntimeError("Pub/Sub 503"),
        )
        d = OutboxDispatcher(
            store=store,  # type: ignore[arg-type]
            publisher=publisher,
            worker_id="test-worker",
            max_attempts=5,
        )

        await d.drain_once(batch_size=10)

        assert len(store.deadlettered) == 1
        assert store.deadlettered[0][0] == "a"
        assert store.deadlettered[0][2] == 5  # attempt_count
        assert store.pending == []

    @pytest.mark.asyncio
    async def test_multi_tenant_drain_preserves_isolation(self) -> None:
        """D6.3 prerequisite — when workflows from N tenants share the
        outbox, the dispatcher publishes each to the right topic with
        correct tenant attribution."""
        store = _MockStore(
            pending=[
                _row(row_id="t1-a", tenant_id="tenant-1"),
                _row(row_id="t2-a", tenant_id="tenant-2"),
                _row(row_id="t1-b", tenant_id="tenant-1"),
                _row(row_id="t3-a", tenant_id="tenant-3"),
            ]
        )
        published_tenants: list[str] = []

        async def _capture(row: OutboxRow) -> str:
            published_tenants.append(row.tenant_id)
            return f"msg-{row.id}"

        publisher = AsyncMock()
        publisher.publish = AsyncMock(side_effect=_capture)

        d = OutboxDispatcher(
            store=store,  # type: ignore[arg-type]
            publisher=publisher,
            worker_id="test-worker",
            max_attempts=5,
        )

        n = await d.drain_once(batch_size=10)

        assert n == 4
        # All 3 tenants represented, no cross-tenant bleed:
        assert sorted(set(published_tenants)) == [
            "tenant-1",
            "tenant-2",
            "tenant-3",
        ]
        assert published_tenants.count("tenant-1") == 2

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_pending(self) -> None:
        store = _MockStore(pending=[])
        publisher = AsyncMock()
        d = OutboxDispatcher(
            store=store,  # type: ignore[arg-type]
            publisher=publisher,
            worker_id="test-worker",
            max_attempts=5,
        )

        n = await d.drain_once(batch_size=10)
        assert n == 0
        publisher.publish.assert_not_called()
