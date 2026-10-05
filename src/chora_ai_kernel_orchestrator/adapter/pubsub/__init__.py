"""NATS JetStream adapter for the AI Kernel LangGraph orchestrator.

Cloud-neutral replacement for the Google Cloud Pub/Sub adapter. Components
(per `feedback_d6_resilience_first_class` B.6.2):

* ``OutboxRow`` — DTO carrying one ``ai_kernel_outbox_events`` row from
  the store layer up to the dispatcher.
* ``OutboxStore`` — Protocol the dispatcher uses to fetch / mark
  pending rows (implementations: ``PostgresOutboxStore`` for live DB,
  ``InMemoryOutboxStore`` for tests).
* ``NatsPublisher`` — thin async wrapper around a NATS JetStream context
  (publishes payload + sets envelope as message headers).
* ``OutboxDispatcher`` — coordinates fetch → publish → mark with
  retry + deadletter semantics.

The subscriber-side ack-after-processing wrapper + DLQ handler live in
``agent_completion_subscriber`` / ``dispatch_dlq_subscriber`` and speak the
transport-agnostic ``_MessageLike`` protocol; the JetStream pull loops that
feed them are ``NatsConsumerLoop`` (``nats.py``) + the per-lane loop
wrappers.
"""

from chora_ai_kernel_orchestrator.adapter.pubsub.dispatcher import (
    OutboxDispatcher,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.publisher import (
    NatsPublisher,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.store import (
    InMemoryOutboxStore,
    OutboxRow,
    OutboxStore,
    PostgresOutboxStore,
)

__all__ = [
    "InMemoryOutboxStore",
    "NatsPublisher",
    "OutboxDispatcher",
    "OutboxRow",
    "OutboxStore",
    "PostgresOutboxStore",
]
