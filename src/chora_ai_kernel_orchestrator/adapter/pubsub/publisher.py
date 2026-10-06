"""NATS JetStream publisher — the outbox dispatcher's transport.

Replaces ``GoogleCloudPubSubPublisher`` (the thin async wrapper around
``google.cloud.pubsub_v1.PublisherClient``). The wire contract is unchanged:
the envelope dict is encoded as message headers (event_id, idempotency_key,
tenant_id, gcid, occurred_at, traceparent, tracestate, source_project,
source_service, schema_version) so subscribers can filter without parsing the
payload, and the destination is the row's ``topic`` (a NATS subject).

The publisher owns its NATS connection: it connects lazily on the first
``publish`` (so building a wiring is cheap and side-effect free) and
``close()`` releases it. This mirrors how ``NatsConsumerLoop`` owns its
connection.
"""

from __future__ import annotations

import logging
from typing import Any

from chora_ai_kernel_orchestrator.adapter.pubsub.store import OutboxRow

logger = logging.getLogger(__name__)

# NATS subjects carry the routing hint in the subject itself; a header key
# that collides with a reserved name would confuse consumers, so strip it
# defensively (mirrors the Pub/Sub reserved-attrs strip).
_RESERVED_HEADER_KEYS = frozenset({"subject", "reply", "sid"})

# The envelope dict is snake_case; the Go eventbus (chora-common/eventbus
# `envelopeFromHeaders`) reads canonical `Chora-*` header names and rejects a
# message whose envelope it cannot reconstruct — dead-lettering it before the
# handler ever runs. Publish the canonical names so Go subscribers can read the
# envelope. Keys absent from this map are forwarded verbatim.
_ENVELOPE_HEADER_NAMES = {
    "event_id": "Chora-Event-Id",
    "idempotency_key": "Chora-Idempotency-Key",
    "tenant_id": "Chora-Tenant-Id",
    "gcid": "Chora-Gcid",
    "source_service": "Chora-Source-Service",
    "source_project": "Chora-Source-Project",
    "correlation_id": "Chora-Correlation-Id",
    "causation_id": "Chora-Causation-Id",
    "chora_imda_dimension": "Chora-Imda-Dimension",
    "imda_lifecycle_stage": "Chora-Imda-Lifecycle-Stage",
    "schema_version": "Chora-Schema-Version",
    "occurred_at": "Chora-Occurred-At",
    "published_at": "Chora-Published-At",
    "traceparent": "Traceparent",
    "tracestate": "Tracestate",
}


def envelope_headers(envelope: Any) -> dict[str, str]:
    """Map an outbox envelope dict onto the canonical NATS header names the Go
    eventbus reads. Unknown keys are forwarded verbatim; reserved keys dropped."""
    headers: dict[str, str] = {}
    for key, value in dict(envelope or {}).items():
        name = str(key)
        if name in _RESERVED_HEADER_KEYS:
            continue
        headers[_ENVELOPE_HEADER_NAMES.get(name, name)] = str(value)
    return headers


class NatsPublisher:
    """Publishes a single ``OutboxRow`` to its NATS subject."""

    def __init__(
        self,
        *,
        url: str,
        subject_prefix: str = "",
        js: Any = None,
    ) -> None:
        if not (url or "").strip():
            raise ValueError("NatsPublisher: url required")
        self._url = url.strip()
        self._subject_prefix = (subject_prefix or "").strip()
        self._nc: Any = None
        # Injectable JetStream context for tests; when None the publisher
        # connects lazily from ``url`` on the first publish.
        self._js = js

    async def _ensure_js(self) -> Any:
        if self._js is None:
            import nats  # lazy — keeps the test path SDK-free

            self._nc = await nats.connect(
                self._url,
                reconnect_time_wait=2.0,
                max_reconnect_attempts=10,
            )
            self._js = self._nc.jetstream()
        return self._js

    async def publish(self, row: OutboxRow) -> str:
        """Publish the row's payload to its subject. Returns the PubAck
        sequence as a string on success; propagates exceptions on failure so
        the dispatcher can drive retry/deadletter logic."""
        js = await self._ensure_js()
        subject = f"{self._subject_prefix}{row.topic}" if self._subject_prefix else row.topic
        headers = envelope_headers(row.envelope)
        ack = await js.publish(subject, row.payload, headers=headers)
        return str(getattr(ack, "seq", ""))

    async def close(self) -> None:
        """Release the NATS connection. Safe to call multiple times."""
        if self._nc is not None:
            try:
                await self._nc.close()
            except Exception:
                logger.exception("nats.publisher.close_failed")
            self._nc = None
            self._js = None


__all__ = ["NatsPublisher"]
