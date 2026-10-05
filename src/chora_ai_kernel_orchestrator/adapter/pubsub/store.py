"""Outbox store layer for the AI Kernel orchestrator.

``OutboxStore`` is the Protocol the dispatcher uses. Two implementations:

* ``InMemoryOutboxStore``: for tests.
* ``PostgresOutboxStore``: production; wraps a psycopg async connection
  pointing at ``chora_ai_kernel`` (where ``ai_kernel_outbox_events`` lives).

Mirrors the closure-saga reference at
``services/chora-closure-orchestrator/src/chora_closure_orchestrator/adapter/pubsub/store.py``.
The only structural difference is the table name + the aggregate-root
column being ``workflow_id`` (vs ``saga_id`` in the closure outbox).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Protocol

# ADR-254 D5: a 'failed' row is retried with backoff, then dead-lettered by the
# dispatcher at max_attempts. Before this, fetch_pending read ONLY 'pending', so
# one publish failure stranded the row forever (never retried, deadletter()
# unreachable, ai_kernel_outbox_dead_letters dead code) and a dispatch-request
# row that failed to publish parked its run with no DLQ message anywhere. The
# backoff mirrors the lane subscriptions' retry_policy (10 s doubling, 600 s cap).
FAILED_RETRY_BASE_SECONDS = 10
FAILED_RETRY_MAX_SECONDS = 600


@dataclass
class OutboxRow:
    """One row from ``ai_kernel_outbox_events``."""

    id: str
    workflow_id: str
    tenant_id: str
    gcid: str
    event_type: str
    topic: str
    payload: bytes
    envelope: dict[str, str]
    idempotency_key: str
    retry_count: int = 0
    occurred_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(_dt.UTC))


class OutboxStore(Protocol):
    """Port the dispatcher uses to drain the outbox."""

    async def fetch_pending(self, *, limit: int) -> list[OutboxRow]: ...

    async def mark_published(self, row_id: str) -> None: ...

    async def mark_failed(self, row_id: str, error: str) -> None: ...

    async def deadletter(self, row_id: str, failure_reason: str, attempt_count: int) -> None: ...


class InMemoryOutboxStore:
    """Hermetic in-memory implementation for unit tests."""

    def __init__(self, pending: list[OutboxRow] | None = None) -> None:
        self.pending: list[OutboxRow] = list(pending or [])
        self.published: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.deadlettered: list[tuple[str, str, int]] = []

    async def fetch_pending(self, *, limit: int) -> list[OutboxRow]:
        return self.pending[:limit]

    async def mark_published(self, row_id: str) -> None:
        self.published.append(row_id)
        self.pending = [r for r in self.pending if r.id != row_id]

    async def mark_failed(self, row_id: str, error: str) -> None:
        self.failed.append((row_id, error))
        for r in self.pending:
            if r.id == row_id:
                r.retry_count += 1
                break

    async def deadletter(self, row_id: str, failure_reason: str, attempt_count: int) -> None:
        self.deadlettered.append((row_id, failure_reason, attempt_count))
        self.pending = [r for r in self.pending if r.id != row_id]


class PostgresOutboxStore:
    """psycopg-backed store wrapping ``ai_kernel_outbox_events`` in
    ``chora_ai_kernel``.

    The dispatcher holds one of these long-lived; per-call queries open
    a cursor on the supplied connection. Worker safety is delegated to
    ``SELECT ... FOR UPDATE SKIP LOCKED`` semantics in ``fetch_pending``
    so multiple dispatcher workers can drain concurrently without
    re-publishing the same row.
    """

    def __init__(self, *, conn: Any, worker_id: str) -> None:
        if not worker_id:
            raise ValueError("worker_id required")
        self._conn = conn
        self._worker_id = worker_id

    async def fetch_pending(self, *, limit: int) -> list[OutboxRow]:
        async with self._conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, workflow_id::TEXT, tenant_id::TEXT, gcid::TEXT,
                       event_type, topic, payload, envelope::TEXT,
                       idempotency_key, retry_count, occurred_at
                FROM ai_kernel_outbox_events
                WHERE status = 'pending'
                   OR (status = 'failed'
                       AND (last_attempt_at IS NULL
                            OR last_attempt_at
                               + LEAST(%(retry_max_s)s, %(retry_base_s)s * power(2, GREATEST(retry_count, 1) - 1))
                                 * interval '1 second'
                               < now()))
                ORDER BY occurred_at ASC
                LIMIT %(limit)s
                FOR UPDATE SKIP LOCKED
                """,
                {
                    "limit": limit,
                    "retry_base_s": FAILED_RETRY_BASE_SECONDS,
                    "retry_max_s": FAILED_RETRY_MAX_SECONDS,
                },
            )
            rows = await cur.fetchall()

        # Release the read transaction when there is nothing to drain. On the
        # QGEN wiring this connection has autocommit=OFF (raw psycopg
        # AsyncConnection), so the SELECT above opens a transaction: on the
        # non-empty path the per-row mark_published / mark_failed commits close
        # it, but on an EMPTY result NOTHING would: leaving the connection
        # "idle in transaction" indefinitely and wedging the qgen subscriber +
        # checkpointer that SHARE this connection (regression surfaced
        # 2026-06-02).
        #
        # The release MUST be a COMMIT, never a ROLLBACK. This connection is
        # shared with the outbox WRITERS (QGenCrewTerminalOutboxWriter,
        # AgentDecisionLogOutboxWriter, HITLDecisionOutboxWriter) and this
        # dispatcher runs as a CONCURRENT asyncio task (main.py create_task,
        # every 2s). psycopg serialises single statements but not a
        # multi-statement transaction across tasks, so a rollback here discards
        # any INSERT a writer landed after our SELECT returned empty:
        #
        #   dispatcher : SELECT ... FOR UPDATE SKIP LOCKED -> 0 rows
        #   writer     : INSERT ai_kernel_outbox_events    (uncommitted)
        #   dispatcher : ROLLBACK                          <-- event destroyed
        #
        # That silently ate the terminal completed event for job fd80f4e9 on
        # 2026-08-14 - the subscriber ACKed, so it was never retried and the
        # job sat at status=running forever. Zero rows matched here means zero
        # FOR UPDATE SKIP LOCKED locks were acquired and the dispatcher has
        # nothing of its own to undo, so committing is equally cheap and
        # cannot destroy a co-tenant's write. (On the OE wiring this conn is a
        # ReconnectingAsyncConnection with autocommit=ON, where the commit is a
        # harmless no-op - see oe_grading_crew_wiring.)
        if not rows:
            await self._conn.commit()
            return []

        import json as _json

        out: list[OutboxRow] = []
        for r in rows:
            (
                _id,
                workflow_id,
                tenant_id,
                gcid,
                event_type,
                topic,
                payload,
                envelope_str,
                idempotency_key,
                retry_count,
                occurred_at,
            ) = r
            envelope = _json.loads(envelope_str) if envelope_str else {}
            out.append(
                OutboxRow(
                    id=_id,
                    workflow_id=workflow_id,
                    tenant_id=tenant_id,
                    gcid=gcid,
                    event_type=event_type,
                    topic=topic,
                    payload=bytes(payload),
                    envelope={str(k): str(v) for k, v in envelope.items()},
                    idempotency_key=idempotency_key,
                    retry_count=retry_count,
                    occurred_at=occurred_at,
                )
            )
        return out

    async def mark_published(self, row_id: str) -> None:
        async with self._conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE ai_kernel_outbox_events
                SET status='published', published_at=now()
                WHERE id = %(id)s
                """,
                {"id": row_id},
            )
        await self._conn.commit()

    async def mark_failed(self, row_id: str, error: str) -> None:
        async with self._conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE ai_kernel_outbox_events
                SET status='failed',
                    retry_count = retry_count + 1,
                    last_error = %(err)s,
                    last_attempt_at = now()
                WHERE id = %(id)s
                """,
                {"id": row_id, "err": error[:1000]},
            )
        await self._conn.commit()

    async def deadletter(self, row_id: str, failure_reason: str, attempt_count: int) -> None:
        async with self._conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO ai_kernel_outbox_dead_letters
                    (outbox_event_id, failure_reason, attempt_count, worker_id)
                VALUES (%(id)s, %(reason)s, %(count)s, %(worker)s)
                ON CONFLICT (outbox_event_id) DO NOTHING
                """,
                {
                    "id": row_id,
                    "reason": failure_reason[:1000],
                    "count": attempt_count,
                    "worker": self._worker_id,
                },
            )
            await cur.execute(
                """
                UPDATE ai_kernel_outbox_events
                SET status='deadlettered',
                    last_attempt_at = now()
                WHERE id = %(id)s
                """,
                {"id": row_id},
            )
        await self._conn.commit()


__all__ = [
    "InMemoryOutboxStore",
    "OutboxRow",
    "OutboxStore",
    "PostgresOutboxStore",
]
