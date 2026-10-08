"""Tests for the prompt-promotion audit wiring's connection policy + assembly
seam (ADR-197 M-C.2 / CHO-2368).

THE defect these pin down (found live 2026-07-27, the rollback-drill final
hop): ``build_prompt_promotion_audit_from_env`` opened a PLAIN psycopg
connection (autocommit=OFF) and nothing on the audit lane ever commits - the
lane's own OutboxDispatcher is built but never started (main.py single-drain
invariant: the qgen dispatcher wins, on its OWN connection), so unlike the
qgen lane there is no co-located ``mark_published`` commit to flush the
implicit transaction psycopg opens at the inbox's first SELECT. Every
repo/emitter ``conn.transaction()`` block then degrades to a SAVEPOINT inside
that never-committed transaction: the consumer ACKS the approval, and the
activation + activation-audit + inbox mark all silently ROLL BACK when the
connection resets (observed live: app backend idle with last query ROLLBACK;
plan stuck pending_hitl after an acked O+ approve).

Contract under test (mirrors the OE grading wiring, which documents the same
lesson):

* the lane's connection is a ``ReconnectingAsyncConnection`` (cost-pause /
  DB-blip self-heal) constructed with ``autocommit=True`` so every write is
  durable + cross-connection-visible the moment it executes;
* the pure assembly seam hands that ONE connection to every adapter (repo,
  inbox, emitters, outbox store) - no second connection, no plain conn.
"""

from __future__ import annotations

from typing import Any

from chora_ai_kernel_orchestrator.adapter.pubsub.audit_recorded_consumer import (
    AuditRecordedConsumer,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.prompt_promotion_audit_wiring import (
    DEFAULT_SUBSCRIPTION,
    PromptPromotionAuditComponents,
    _assemble_components,
    _make_db_conn,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.reconnecting_connection import (
    ReconnectingAsyncConnection,
)
from chora_ai_kernel_orchestrator.domain.prompt_registry import (
    PromptPromotionApprovalHandler,
    PromptPromotionService,
)


class _FakeConn:
    """Duck-typed connection - held by adapter ctors, never exercised."""

    def cursor(self) -> Any:  # pragma: no cover - never called in unit test
        raise AssertionError("cursor should not be exercised at assembly time")


class _FakePublisher:
    """Duck-typed GoogleCloudPubSubPublisher - held, never called."""

    async def publish(self, **_: Any) -> str:  # pragma: no cover
        return ""


def test_make_db_conn_is_autocommit_reconnecting_wrapper() -> None:
    """The lane's connection MUST be the self-healing wrapper with
    autocommit=True - a plain psycopg conn (autocommit=OFF) re-opens the
    silent-rollback defect this file documents."""
    conn = _make_db_conn("postgres://unit-test-dsn/never-dialled")
    assert isinstance(conn, ReconnectingAsyncConnection)
    assert conn._autocommit is True  # noqa: SLF001 - house style (see OE tests)


def test_make_db_conn_does_not_dial() -> None:
    """Constructing the wrapper is I/O-free (the glue awaits .connect()
    separately, fail-loud at startup)."""
    conn = _make_db_conn("postgres://unit-test-dsn/never-dialled")
    assert conn._conn is None  # noqa: SLF001


def test_assemble_components_single_conn_and_shape() -> None:
    """The assembly seam threads ONE connection through every adapter and
    returns the started-together component bundle main.py expects."""
    fake_conn = _FakeConn()
    fake_pub = _FakePublisher()

    components = _assemble_components(
        db_conn=fake_conn,
        pubsub_project="unit-project",
        subscription="unit-subscription",
        publisher=fake_pub,
    )

    assert isinstance(components, PromptPromotionAuditComponents)
    assert components.db_conn is fake_conn

    # One connection everywhere - a second/plain conn would silently fork
    # commit semantics again.
    assert isinstance(components.consumer, AuditRecordedConsumer)
    assert components.consumer._inbox._conn is fake_conn  # noqa: SLF001
    assert isinstance(components.handler, PromptPromotionApprovalHandler)
    assert components.handler is components.consumer._handler  # noqa: SLF001
    assert isinstance(components.service, PromptPromotionService)
    assert components.service._repo._conn is fake_conn  # noqa: SLF001
    assert components.service._audit._conn is fake_conn  # noqa: SLF001

    assert components.pubsub_loop is not None
    assert components.outbox_dispatcher is not None


def test_default_subscription_is_the_canonical_audit_recorded_topic() -> None:
    """The default must be a NATS subject the CHORA_EVENTS ``chora.>`` stream
    captures, not the legacy Pub/Sub subscription resource name."""
    assert DEFAULT_SUBSCRIPTION == "chora.governance.audit.recorded.v1"
