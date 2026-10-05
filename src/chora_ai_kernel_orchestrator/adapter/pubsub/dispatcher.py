"""OutboxDispatcher — drains ``ai_kernel_outbox_events`` to NATS JetStream.

Mirrors the closure-saga reference at
``services/chora-closure-orchestrator/src/chora_closure_orchestrator/adapter/pubsub/dispatcher.py``.

Second half of the transactional outbox pattern. Polls pending rows +
publishes via ``NatsPublisher`` + marks rows
``status='published'`` on success or accumulates retries / deadletters
on failure.

Per ``feedback_d6_resilience_first_class``: this is the producer-side
delivery resilience layer. Subscriber-side ack-after-processing is a
separate concern (B.6.2.b).

``max_attempts`` aligns with the Pub/Sub subscription's
``max_delivery_attempts`` (defaulted to 5 in ``m10-pubsub-dlq``) — but
the two semantics are distinct: this counter governs outbox→publish
attempts (DB→Pub/Sub), while the subscription's counter governs
Pub/Sub→subscriber attempts. Both ultimately route to the DLQ topic on
exhaustion.
"""

from __future__ import annotations

import logging
from typing import Any

from chora_ai_kernel_orchestrator.adapter.pubsub.store import OutboxStore

logger = logging.getLogger(__name__)


class OutboxDispatcher:
    """Coordinates store ↔ publisher with retry / deadletter."""

    def __init__(
        self,
        *,
        store: OutboxStore,
        publisher: Any,
        worker_id: str,
        max_attempts: int = 5,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._store = store
        self._publisher = publisher
        self._worker_id = worker_id
        self._max_attempts = max_attempts

    async def drain_once(self, *, batch_size: int = 100) -> int:
        """Drain one batch. Returns the count of successful publishes.

        Failures are recorded via store.mark_failed (transient) or
        store.deadletter (max_attempts exhausted) — they do NOT raise.
        Callers run this in a loop with a backoff between calls.
        """
        rows = await self._store.fetch_pending(limit=batch_size)
        published = 0
        for row in rows:
            try:
                msg_id = await self._publisher.publish(row)
            except Exception as exc:  # noqa: BLE001
                attempt = row.retry_count + 1
                if attempt >= self._max_attempts:
                    logger.warning(
                        "outbox_deadletter",
                        extra={
                            "row_id": row.id,
                            "workflow_id": row.workflow_id,
                            "tenant_id": row.tenant_id,
                            "attempts": attempt,
                            "topic": row.topic,
                            "error": str(exc)[:500],
                        },
                    )
                    await self._store.deadletter(
                        row.id,
                        failure_reason=str(exc)[:1000],
                        attempt_count=attempt,
                    )
                else:
                    logger.info(
                        "outbox_publish_failed",
                        extra={
                            "row_id": row.id,
                            "workflow_id": row.workflow_id,
                            "attempts": attempt,
                            "error": str(exc)[:500],
                        },
                    )
                    await self._store.mark_failed(row.id, str(exc))
                continue
            logger.debug(
                "outbox_published",
                extra={
                    "row_id": row.id,
                    "msg_id": msg_id,
                    "tenant_id": row.tenant_id,
                    "topic": row.topic,
                },
            )
            await self._store.mark_published(row.id)
            published += 1
        return published


__all__ = ["OutboxDispatcher"]
