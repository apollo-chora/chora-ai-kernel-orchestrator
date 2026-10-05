"""RED→GREEN tests for QGenCrewSubscriber.

The subscriber:
1. Receives messages from a Pub/Sub StreamingPull on
   ``chora.creation.ai_assist.started.v1``.
2. Extracts the envelope.event_id from message attributes.
3. Uses InboxIdempotencyStore to short-circuit duplicates.
4. Decodes the JSON payload into a dict + invokes QGenCrewRunner.handle_started.
5. ACKs on success (or dedupe-hit); NACKs on transient failure so Pub/Sub
   redelivers (subscription's DLQ catches after max_delivery_attempts).

Per [[feedback-d6-resilience-first-class]] D6 P3: ack-after-processing.
The ACK happens only after the runner returns nil (graph + outbox row
write succeeded).
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_subscriber import (
    QGenCrewSubscriber,
)

# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


@dataclass
class _FakeRunner:
    """Records handle_started calls; optionally raises on Nth call."""

    handled: list[dict[str, Any]] = field(default_factory=list)
    raise_on_call: int | None = None
    call_count: int = 0

    async def handle_started(self, event: dict[str, Any]) -> None:
        self.call_count += 1
        if self.raise_on_call is not None and self.call_count == self.raise_on_call:
            raise RuntimeError("simulated graph failure")
        self.handled.append(event)


@dataclass
class _FakeInbox:
    """In-memory inbox dedup."""

    seen: set[str] = field(default_factory=set)
    processed: list[str] = field(default_factory=list)

    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool:
        if key in self.seen:
            return False
        # Run fn; mark key on success.
        await fn()
        self.seen.add(key)
        self.processed.append(key)
        return True


@dataclass
class _FakeMessage:
    """Pub/Sub-style message — duck-typed for the subscriber."""

    data: bytes
    attributes: dict[str, str]
    ack_called: int = 0
    nack_called: int = 0

    def ack(self) -> None:
        self.ack_called += 1

    def nack(self) -> None:
        self.nack_called += 1


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_default_inbox_ttl_7d(self) -> None:
        sub = QGenCrewSubscriber(
            runner=_FakeRunner(),
            inbox=_FakeInbox(),
        )
        assert sub.inbox_ttl == _dt.timedelta(days=7)

    def test_inbox_ttl_overrideable(self) -> None:
        sub = QGenCrewSubscriber(
            runner=_FakeRunner(),
            inbox=_FakeInbox(),
            inbox_ttl=_dt.timedelta(days=1),
        )
        assert sub.inbox_ttl == _dt.timedelta(days=1)


# -----------------------------------------------------------------------------
# handle_message — happy path
# -----------------------------------------------------------------------------


def _make_message(
    *,
    event_id: str = "evt-1",
    assist_id: str = "01970000-7777-7000-a000-000000000001",
    tenant_id: str = "t-1",
    author_gcid: str = "g-1",
    question_type: str = "mcq",
    prompt: str = "Generate a math question",
    metadata: dict[str, str] | None = None,
    max_retries: int = 3,
    traceparent: str = "",
) -> _FakeMessage:
    body = {
        "assist_id": assist_id,
        "tenant_id": tenant_id,
        "author_gcid": author_gcid,
        "content_type": question_type,
        "question_type": question_type,
        "prompt": prompt,
        "metadata": metadata or {},
        "max_retries": max_retries,
    }
    return _FakeMessage(
        data=json.dumps(body).encode("utf-8"),
        attributes={
            "event_id": event_id,
            "idempotency_key": event_id,
            "tenant_id": tenant_id,
            "gcid": author_gcid,
            "traceparent": traceparent,
        },
    )


class TestHandleMessageHappy:
    @pytest.mark.asyncio
    async def test_first_delivery_invokes_runner_and_acks(self) -> None:
        runner = _FakeRunner()
        inbox = _FakeInbox()
        sub = QGenCrewSubscriber(runner=runner, inbox=inbox)
        msg = _make_message(event_id="evt-1")

        await sub.handle_message(msg)

        assert runner.call_count == 1
        assert msg.ack_called == 1
        assert msg.nack_called == 0
        assert "evt-1" in inbox.seen

    @pytest.mark.asyncio
    async def test_duplicate_delivery_skips_runner_and_acks(self) -> None:
        """Pub/Sub redelivery → inbox dedupe hits → fn not invoked → ACK."""
        runner = _FakeRunner()
        inbox = _FakeInbox(seen={"evt-1"})  # pre-seeded as already seen
        sub = QGenCrewSubscriber(runner=runner, inbox=inbox)
        msg = _make_message(event_id="evt-1")

        await sub.handle_message(msg)

        assert runner.call_count == 0
        assert msg.ack_called == 1
        assert msg.nack_called == 0

    @pytest.mark.asyncio
    async def test_payload_forwarded_to_runner_verbatim(self) -> None:
        runner = _FakeRunner()
        sub = QGenCrewSubscriber(runner=runner, inbox=_FakeInbox())
        msg = _make_message(
            assist_id="a-99",
            tenant_id="t-99",
            author_gcid="g-99",
            question_type="oe",
            prompt="Generate an open-ended question about photosynthesis.",
            metadata={"subject": "biology", "cognitive_level": "analyze"},
            max_retries=2,
            traceparent="00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        )

        await sub.handle_message(msg)

        event = runner.handled[0]
        assert event["assist_id"] == "a-99"
        assert event["tenant_id"] == "t-99"
        assert event["author_gcid"] == "g-99"
        assert event["question_type"] == "oe"
        assert event["prompt"] == ("Generate an open-ended question about photosynthesis.")
        assert event["metadata"] == {
            "subject": "biology",
            "cognitive_level": "analyze",
        }
        assert event["max_retries"] == 2
        # traceparent threaded from message attributes
        assert event["traceparent"] == ("00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01")


# -----------------------------------------------------------------------------
# handle_message — failure path
# -----------------------------------------------------------------------------


class TestHandleMessageFailure:
    @pytest.mark.asyncio
    async def test_runner_exception_nacks_and_does_not_mark_inbox(self) -> None:
        """Runner failure → NACK + key NOT marked → Pub/Sub will redeliver
        and the next attempt will retry. After max_delivery_attempts the
        subscription's DLQ catches.
        """
        runner = _FakeRunner(raise_on_call=1)
        inbox = _FakeInbox()
        sub = QGenCrewSubscriber(runner=runner, inbox=inbox)
        msg = _make_message(event_id="evt-bad")

        await sub.handle_message(msg)

        assert msg.ack_called == 0
        assert msg.nack_called == 1
        assert "evt-bad" not in inbox.seen

    @pytest.mark.asyncio
    async def test_invalid_json_payload_nacks(self) -> None:
        """A poison message → NACK so it cycles to the DLQ."""
        runner = _FakeRunner()
        sub = QGenCrewSubscriber(runner=runner, inbox=_FakeInbox())
        msg = _FakeMessage(
            data=b"not valid json",
            attributes={"event_id": "evt-x"},
        )

        await sub.handle_message(msg)

        assert msg.ack_called == 0
        assert msg.nack_called == 1
        assert runner.call_count == 0

    @pytest.mark.asyncio
    async def test_missing_event_id_nacks(self) -> None:
        """No event_id attribute → cannot dedupe → NACK (DLQ will catch)."""
        runner = _FakeRunner()
        sub = QGenCrewSubscriber(runner=runner, inbox=_FakeInbox())
        msg = _FakeMessage(
            data=json.dumps({"assist_id": "a"}).encode("utf-8"),
            attributes={},  # no event_id / idempotency_key
        )

        await sub.handle_message(msg)

        assert msg.ack_called == 0
        assert msg.nack_called == 1


# -----------------------------------------------------------------------------
# Dedup key resolution — prefer idempotency_key, fall back to event_id.
# -----------------------------------------------------------------------------


class TestDedupKeyResolution:
    @pytest.mark.asyncio
    async def test_prefers_idempotency_key_when_present(self) -> None:
        runner = _FakeRunner()
        inbox = _FakeInbox()
        sub = QGenCrewSubscriber(runner=runner, inbox=inbox)
        msg = _FakeMessage(
            data=json.dumps(_valid_body()).encode("utf-8"),
            attributes={
                "event_id": "evt-1",
                "idempotency_key": "idem-pref",
                "tenant_id": "t-1",
                "gcid": "g-1",
            },
        )

        await sub.handle_message(msg)

        assert "idem-pref" in inbox.seen
        assert "evt-1" not in inbox.seen

    @pytest.mark.asyncio
    async def test_falls_back_to_event_id_when_no_idempotency_key(self) -> None:
        runner = _FakeRunner()
        inbox = _FakeInbox()
        sub = QGenCrewSubscriber(runner=runner, inbox=inbox)
        msg = _FakeMessage(
            data=json.dumps(_valid_body()).encode("utf-8"),
            attributes={
                "event_id": "evt-fallback",
                "tenant_id": "t-1",
                "gcid": "g-1",
            },
        )

        await sub.handle_message(msg)

        assert "evt-fallback" in inbox.seen


def _valid_body() -> dict[str, Any]:
    return {
        "assist_id": "01970000-7777-7000-a000-000000000001",
        "tenant_id": "t-1",
        "author_gcid": "g-1",
        "content_type": "mcq",
        "question_type": "mcq",
        "prompt": "Q",
        "metadata": {},
        "max_retries": 3,
    }


# -----------------------------------------------------------------------------
# The envelope tenant is a CHECK on the body tenant (2026-08-23)
#
# The body's tenant_id decides which tenant the whole run executes as, and it
# governs the source-URI constraint the image-regen path now applies. Until
# this, the subscriber read msg.attributes["tenant_id"] for LOGGING ONLY (the
# `acked` and `runner_failed` lines) and never compared it to the body, so the
# envelope contributed nothing to authorisation.
#
# Producers checked before making disagreement fatal: chora-creation publishes
# this topic through cmd/server/job_event_bridge.go, whose envelopeAttrs()
# stamps tenant_id UNCONDITIONALLY (it is in the base map, not behind an `if`),
# on both PublishJobEvent and PublishJobEventSync, and the started event goes
# out via PublishJobEventSync to topicAiAssistStarted =
# "chora.creation.ai_assist.started.v2". So the attribute is present in
# production and this check is live rather than inert.
#
# Absence is still tolerated and logged rather than fatal: six principals hold
# project-level roles/pubsub.publisher, so a message from an unstamped producer
# is possible, and making absence fatal would turn the first one into an outage.
# -----------------------------------------------------------------------------


class TestEnvelopeTenantIsAuthoritative:
    @pytest.mark.asyncio
    async def test_a_body_tenant_that_disagrees_with_the_envelope_is_refused(
        self,
    ) -> None:
        """The run must not start. The body tenant chooses the tenant the run
        executes as, so accepting one the envelope contradicts would let a
        forged body pick its own blast radius."""
        runner, inbox = _FakeRunner(), _FakeInbox()
        sub = QGenCrewSubscriber(runner=runner, inbox=inbox)

        msg = _make_message(tenant_id="tenant-A")
        msg.attributes["tenant_id"] = "tenant-B"

        await sub.handle_message(msg)

        assert runner.handled == [], "a mismatched message must never reach the runner"
        assert msg.nack_called == 1, "a mismatch must not be silently dropped"
        assert msg.ack_called == 0

    @pytest.mark.asyncio
    async def test_a_matching_envelope_tenant_runs_normally(self) -> None:
        runner, inbox = _FakeRunner(), _FakeInbox()
        sub = QGenCrewSubscriber(runner=runner, inbox=inbox)

        await sub.handle_message(_make_message(tenant_id="tenant-A"))

        assert len(runner.handled) == 1
        assert runner.handled[0]["tenant_id"] == "tenant-A"

    @pytest.mark.asyncio
    async def test_an_absent_envelope_tenant_is_tolerated(self) -> None:
        """Loud, but not fatal. Making absence fatal would turn the first
        unstamped producer into an outage on the whole lane."""
        runner, inbox = _FakeRunner(), _FakeInbox()
        sub = QGenCrewSubscriber(runner=runner, inbox=inbox)

        msg = _make_message(tenant_id="tenant-A")
        del msg.attributes["tenant_id"]

        await sub.handle_message(msg)

        assert len(runner.handled) == 1, "an unstamped producer must still be served"
        assert msg.ack_called == 1

    @pytest.mark.asyncio
    async def test_an_empty_envelope_tenant_counts_as_absent_not_as_a_mismatch(
        self,
    ) -> None:
        """`attrs["tenant_id"] = ""` is the same situation as no key at all;
        treating it as a disagreement would fail closed on a producer that
        stamps the key with an empty value."""
        runner, inbox = _FakeRunner(), _FakeInbox()
        sub = QGenCrewSubscriber(runner=runner, inbox=inbox)

        msg = _make_message(tenant_id="tenant-A")
        msg.attributes["tenant_id"] = "   "

        await sub.handle_message(msg)

        assert len(runner.handled) == 1
        assert msg.ack_called == 1
