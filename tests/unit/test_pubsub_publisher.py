"""Tests for ``NatsPublisher`` — the async wrapper used by the AI Kernel
outbox dispatcher to push messages to NATS JetStream.

Replaces the ``GoogleCloudPubSubPublisher`` tests. The envelope is encoded as
message headers under the canonical names the Go eventbus reads
(``Chora-Event-Id``, ``Chora-Idempotency-Key``, ``Chora-Tenant-Id``,
``Chora-Gcid``, ``Chora-Occurred-At``, ``Traceparent``, ``Tracestate``,
``Chora-Source-Project``, ``Chora-Source-Service``, ``Chora-Schema-Version``)
so subscribers can filter without parsing the payload, and the destination is
the row's ``topic`` (a NATS subject).

The JetStream context is injected (the ``js`` seam) so the tests run without
a live NATS server.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub import (
    NatsPublisher,
    OutboxRow,
)


class _MockPubAck:
    def __init__(self, seq: int = 1) -> None:
        self.seq = seq


class _MockJs:
    """Duck-typed JetStream context — records publishes."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, dict[str, str]]] = []
        self._next_seq = 1

    async def publish(self, subject: str, payload: bytes, headers: dict[str, str] | None = None) -> _MockPubAck:
        self.published.append((subject, payload, dict(headers or {})))
        ack = _MockPubAck(self._next_seq)
        self._next_seq += 1
        return ack


def _envelope_dict() -> dict[str, str]:
    return {
        "event_id": "01970000-7777-7000-a000-000000000001",
        "idempotency_key": "01970000-7777-7000-a000-000000000001",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "gcid": "01970000-0000-7000-9000-000000000001",
        "occurred_at": "2026-05-12T08:00:00+00:00",
        "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        "tracestate": "vendor=chora",
        "source_project": "chora-ai-kernel-orchestrator",
        "source_service": "chora-ai-kernel-orchestrator",
        "schema_version": "1",
    }


def _row(**overrides: Any) -> OutboxRow:
    base: dict[str, Any] = {
        "id": "01970000-7777-7000-a000-000000000001",
        "workflow_id": "d42940c4-3edf-4a3e-83ed-938c5fef4401",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "gcid": "01970000-0000-7000-9000-000000000001",
        "event_type": "ai_kernel.invocation.invoked",
        "topic": "chora.ai_kernel.invocation.invoked.v1",
        "payload": json.dumps({"hello": "world"}).encode("utf-8"),
        "idempotency_key": "01970000-7777-7000-a000-000000000001",
    }
    base.update(overrides)
    base["envelope"] = {
        **_envelope_dict(),
        "tenant_id": base["tenant_id"],
        "gcid": base["gcid"],
        "event_id": base["id"],
        "idempotency_key": base["idempotency_key"],
    }
    return OutboxRow(**base)


class TestNatsPublisherInit:
    def test_rejects_empty_url(self) -> None:
        with pytest.raises(ValueError, match="url required"):
            NatsPublisher(url="")

    def test_constructor_accepts_injected_js(self) -> None:
        js = _MockJs()
        pub = NatsPublisher(url="nats://unused:4222", js=js)
        assert pub._js is js  # noqa: SLF001


class TestPublishOutboxRow:
    @pytest.mark.asyncio
    async def test_publishes_payload_to_subject(self) -> None:
        js = _MockJs()
        pub = NatsPublisher(url="nats://unused:4222", js=js)

        message_id = await pub.publish(_row())

        assert message_id == "1"
        assert len(js.published) == 1
        subject, data, _headers = js.published[0]
        assert subject == "chora.ai_kernel.invocation.invoked.v1"
        assert json.loads(data) == {"hello": "world"}

    @pytest.mark.asyncio
    async def test_envelope_lands_as_headers(self) -> None:
        """Each envelope field MUST be a message header for subscriber-side
        filtering without payload parse."""
        js = _MockJs()
        pub = NatsPublisher(url="nats://unused:4222", js=js)

        await pub.publish(_row())

        _, _, headers = js.published[0]
        for required in (
            "Chora-Event-Id",
            "Chora-Idempotency-Key",
            "Chora-Tenant-Id",
            "Chora-Gcid",
            "Chora-Occurred-At",
            "Traceparent",
            "Tracestate",
            "Chora-Source-Project",
            "Chora-Source-Service",
            "Chora-Schema-Version",
        ):
            assert required in headers, f"header missing: {required}"

    @pytest.mark.asyncio
    async def test_per_tenant_isolation_via_header(self) -> None:
        """D6.3 multi-tenant chaos: subscribers MAY filter by tenant_id
        header. The publisher MUST set it correctly per-message even when
        many tenants share the bus."""
        js = _MockJs()
        pub = NatsPublisher(url="nats://unused:4222", js=js)

        await pub.publish(_row(tenant_id="01970000-0000-7000-8000-000000000001"))
        await pub.publish(_row(tenant_id="01970000-0000-7000-8000-000000000002"))

        tenants = [h["Chora-Tenant-Id"] for _, _, h in js.published]
        assert tenants == [
            "01970000-0000-7000-8000-000000000001",
            "01970000-0000-7000-8000-000000000002",
        ]

    @pytest.mark.asyncio
    async def test_subject_is_the_row_topic(self) -> None:
        js = _MockJs()
        pub = NatsPublisher(url="nats://unused:4222", js=js)

        await pub.publish(_row(topic="chora.ai_kernel.crew.composed.v1"))

        subject, _, _ = js.published[0]
        assert subject == "chora.ai_kernel.crew.composed.v1"

    @pytest.mark.asyncio
    async def test_reserved_header_keys_do_not_collide(self) -> None:
        """An envelope key named "subject"/"reply"/"sid" collides with NATS
        subject-routing hints. The publisher MUST strip reserved keys before
        spreading so a routing attribute can never be mistaken for the
        subject. Regression for the live OE e2e."""
        js = _MockJs()
        pub = NatsPublisher(url="nats://unused:4222", js=js)

        row = _row()
        row.envelope["subject"] = "chora.delivery.grading.submission_completed.v1"
        row.envelope["event_topic"] = "chora.delivery.grading.submission_completed.v1"

        await pub.publish(row)

        _, _, headers = js.published[0]
        assert "subject" not in headers, "reserved 'subject' key must be stripped"
        assert headers.get("event_topic") == "chora.delivery.grading.submission_completed.v1"

    @pytest.mark.asyncio
    async def test_publish_error_propagates(self) -> None:
        """Dispatcher needs to see publish failures to drive retry."""

        class _FailingJs:
            async def publish(self, *args: Any, **kwargs: Any) -> _MockPubAck:
                raise RuntimeError("NATS unavailable")

        pub = NatsPublisher(url="nats://unused:4222", js=_FailingJs())

        with pytest.raises(RuntimeError, match="NATS unavailable"):
            await pub.publish(_row())
