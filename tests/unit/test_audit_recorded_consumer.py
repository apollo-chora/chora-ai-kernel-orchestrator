"""Unit tests for ``AuditRecordedConsumer`` (ADR-197 M-C.2).

The consumer is the Pub/Sub StreamingPull handler for
``chora.governance.audit.recorded.v1``. It:

1. Decodes the binary-proto AuditEntryRecorded (JSON fallback for synthetic).
2. STRICT filter — acks + ignores any audit that is not a prompt-promotion
   HITL decision (the vast majority of the high-volume topic) WITHOUT touching
   the inbox or the handler.
3. For a matching decision: inbox-dedupe -> handler.handle -> ack-after-process.
4. NACKs on decode failure / missing dedup key / handler exception so Pub/Sub
   redelivers (subscription DLQ catches after max_delivery_attempts).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.audit_recorded_consumer import (
    AuditRecordedConsumer,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    AUDIT_RESULT_ALLOWED,
    encode_audit_entry_recorded,
)

_PLAN = "01970000-0000-7000-8000-0000000000aa"
_URI = f"hitl_decision:prompt-override-plan:{_PLAN}"


# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


@dataclass
class _FakeHandler:
    handled: list[dict[str, Any]] = field(default_factory=list)
    raise_exc: Exception | None = None

    def matches(self, decoded: dict[str, Any]) -> bool:
        return str(decoded.get("target_resource_uri", "")).startswith("hitl_decision:prompt-override-plan:")

    async def handle(self, decoded: dict[str, Any]) -> bool:
        if self.raise_exc is not None:
            raise self.raise_exc
        self.handled.append(decoded)
        return True


@dataclass
class _FakeInbox:
    seen: set[str] = field(default_factory=set)
    processed: list[str] = field(default_factory=list)

    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool:
        if key in self.seen:
            return False
        await fn()
        self.seen.add(key)
        self.processed.append(key)
        return True


@dataclass
class _FakeMessage:
    data: bytes
    attributes: dict[str, str]
    ack_called: int = 0
    nack_called: int = 0

    def ack(self) -> None:
        self.ack_called += 1

    def nack(self) -> None:
        self.nack_called += 1


def _envelope(*, idem: str = "audit-1", with_ids: bool = True) -> dict[str, Any]:
    env = {
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "gcid": "01970000-0000-7000-9000-000000000001",
        "occurred_at": "2026-06-28T12:00:00+00:00",
        "published_at": "2026-06-28T12:00:01+00:00",
        "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        "tracestate": "vendor=chora",
        "source_project": "chora-489812",
        "source_service": "chora-governance",
        "schema_version": "1",
        "chora_imda_dimension": "accountability",
    }
    if with_ids:
        env["event_id"] = idem
        env["idempotency_key"] = idem
    return env


def _proto_message(
    *,
    uri: str = _URI,
    action: str = "decision_approve",
    idem: str = "audit-1",
    with_audit_id: bool = True,
    with_env_ids: bool = True,
    attrs: dict[str, str] | None = None,
) -> _FakeMessage:
    body: dict[str, Any] = {
        "actor_gcid": "01970000-0000-7000-9000-000000000001",
        "target_resource_uri": uri,
        "action": action,
        "result": AUDIT_RESULT_ALLOWED,
        "occurred_at": "2026-06-28T12:00:00+00:00",
    }
    if with_audit_id:
        body["audit_id"] = "01970000-0000-7000-a000-000000000abc"
    data = encode_audit_entry_recorded(_envelope(idem=idem, with_ids=with_env_ids), body)
    return _FakeMessage(
        data=data,
        attributes=attrs if attrs is not None else {"idempotency_key": idem},
    )


# -----------------------------------------------------------------------------
# Matching prompt-promotion decision — happy path
# -----------------------------------------------------------------------------


class TestMatchingDecision:
    @pytest.mark.asyncio
    async def test_approve_invokes_handler_and_acks(self) -> None:
        handler, inbox = _FakeHandler(), _FakeInbox()
        consumer = AuditRecordedConsumer(handler=handler, inbox=inbox)
        msg = _proto_message()

        await consumer.handle_message(msg)

        assert len(handler.handled) == 1
        assert handler.handled[0]["target_resource_uri"] == _URI
        assert handler.handled[0]["action"] == "decision_approve"
        assert msg.ack_called == 1
        assert msg.nack_called == 0
        assert "audit-1" in inbox.seen

    @pytest.mark.asyncio
    async def test_duplicate_delivery_dedupes_and_acks(self) -> None:
        handler, inbox = _FakeHandler(), _FakeInbox(seen={"audit-1"})
        consumer = AuditRecordedConsumer(handler=handler, inbox=inbox)
        msg = _proto_message(idem="audit-1")

        await consumer.handle_message(msg)

        assert handler.handled == []  # deduped
        assert msg.ack_called == 1
        assert msg.nack_called == 0


# -----------------------------------------------------------------------------
# Strict filtering — non-prompt-promotion audits acked, never touch inbox
# -----------------------------------------------------------------------------


class TestStrictFiltering:
    @pytest.mark.asyncio
    async def test_non_matching_audit_acks_without_inbox_or_handler(
        self,
    ) -> None:
        handler, inbox = _FakeHandler(), _FakeInbox()
        consumer = AuditRecordedConsumer(handler=handler, inbox=inbox)
        msg = _proto_message(uri="chora.creation/atom:1", action="CREATE")

        await consumer.handle_message(msg)

        assert handler.handled == []
        assert inbox.processed == []  # not even a dedupe write
        assert msg.ack_called == 1
        assert msg.nack_called == 0

    @pytest.mark.asyncio
    async def test_own_activate_audit_acked_and_ignored(self) -> None:
        handler, inbox = _FakeHandler(), _FakeInbox()
        consumer = AuditRecordedConsumer(handler=handler, inbox=inbox)
        msg = _proto_message(uri=f"chora.ai_kernel/prompt_plan:{_PLAN}", action="ACTIVATE")

        await consumer.handle_message(msg)

        assert handler.handled == []
        assert inbox.processed == []
        assert msg.ack_called == 1


# -----------------------------------------------------------------------------
# Failure paths — NACK so Pub/Sub redelivers
# -----------------------------------------------------------------------------


class TestFailurePaths:
    @pytest.mark.asyncio
    async def test_decode_failure_nacks(self) -> None:
        handler, inbox = _FakeHandler(), _FakeInbox()
        consumer = AuditRecordedConsumer(handler=handler, inbox=inbox)
        # First byte 0x0A => binary path; length 20 but body short => overrun.
        msg = _FakeMessage(
            data=bytes([(1 << 3) | 2, 20]) + b"short",
            attributes={"idempotency_key": "audit-x"},
        )

        await consumer.handle_message(msg)

        assert msg.ack_called == 0
        assert msg.nack_called == 1
        assert handler.handled == []

    @pytest.mark.asyncio
    async def test_handler_exception_nacks_and_keeps_inbox_unmarked(
        self,
    ) -> None:
        handler = _FakeHandler(raise_exc=RuntimeError("activation failed"))
        inbox = _FakeInbox()
        consumer = AuditRecordedConsumer(handler=handler, inbox=inbox)
        msg = _proto_message()

        await consumer.handle_message(msg)

        assert msg.ack_called == 0
        assert msg.nack_called == 1
        assert "audit-1" not in inbox.seen

    @pytest.mark.asyncio
    async def test_missing_dedup_key_on_matching_event_nacks(self) -> None:
        handler, inbox = _FakeHandler(), _FakeInbox()
        consumer = AuditRecordedConsumer(handler=handler, inbox=inbox)
        # Matching resource, but no attributes, no envelope ids, no audit_id.
        msg = _proto_message(with_audit_id=False, with_env_ids=False, attrs={})

        await consumer.handle_message(msg)

        assert msg.ack_called == 0
        assert msg.nack_called == 1


# -----------------------------------------------------------------------------
# Dedup key resolution + trace threading
# -----------------------------------------------------------------------------


class TestDedupAndTrace:
    @pytest.mark.asyncio
    async def test_falls_back_to_audit_id_when_no_attrs_or_envelope_ids(
        self,
    ) -> None:
        handler, inbox = _FakeHandler(), _FakeInbox()
        consumer = AuditRecordedConsumer(handler=handler, inbox=inbox)
        msg = _proto_message(with_env_ids=False, attrs={})

        await consumer.handle_message(msg)

        # audit_id is the stable last-resort dedup key.
        assert "01970000-0000-7000-a000-000000000abc" in inbox.seen
        assert msg.ack_called == 1

    @pytest.mark.asyncio
    async def test_threads_traceparent_into_decoded(self) -> None:
        handler, inbox = _FakeHandler(), _FakeInbox()
        consumer = AuditRecordedConsumer(handler=handler, inbox=inbox)
        msg = _proto_message(
            attrs={
                "idempotency_key": "audit-1",
                "traceparent": "00-cccccccccccccccccccccccccccccccc-dddddddddddddddd-01",
            }
        )

        await consumer.handle_message(msg)

        assert handler.handled[0]["traceparent"] == ("00-cccccccccccccccccccccccccccccccc-dddddddddddddddd-01")
