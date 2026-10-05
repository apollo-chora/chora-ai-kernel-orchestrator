"""RED→GREEN tests for WeaknessAnalyserCrewGraphSubscriber (ADR-205 WS-2).

The GRADUATED (graph-mode) sibling of WeaknessAnalyserCrewSubscriber. Where the
single-shot subscriber does analyse→publish synchronously, the graph subscriber
just STARTS the checkpointed crew run — it ``handle_uploaded`` drives the graph
to the HITL ``interrupt()`` (a paused, checkpointed run) and ACKs. The
publish happens later, on the FE resume (the graph's publish_analyzed node), so
this subscriber holds NO publisher.

Same envelope decode + attribute merge + upload-keyed idempotency + ack-after-
processing contract as the single-shot path; only the invoked port differs.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any

from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyser_crew_graph_subscriber import (
    WeaknessAnalyserCrewGraphSubscriber,
)
from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew_runner import (
    RunResult,
    WeaknessCrewError,
)


@dataclass
class _FakeMsg:
    data: bytes
    attributes: dict[str, str]
    acked: bool = False
    nacked: bool = False

    def ack(self) -> None:
        self.acked = True

    def nack(self) -> None:
        self.nacked = True


def _fake_panel(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "upload_id": event.get("upload_id", ""),
        "tenant_id": event.get("tenant_id", ""),
        "learner_gcid": event.get("learner_gcid", ""),
        "familiar": {},
        "proposed_edges": [{"proposed_edge_id": "pe-0", "concept_label": "x"}],
        "candidate_struggles": [],
        "available_outputs": [],
    }


@dataclass
class _FakeCrewRunner:
    """Doubles WeaknessAnalyserCrewRunner — records the event, returns RunResult."""

    events: list[dict[str, Any]] = field(default_factory=list)
    raises: bool = False
    raises_crew_error: bool = False
    interrupted: bool = True

    async def handle_uploaded(self, event: dict[str, Any]) -> RunResult:
        self.events.append(dict(event))
        if self.raises_crew_error:
            raise WeaknessCrewError("missing required field")
        if self.raises:
            raise RuntimeError("graph boom")
        return RunResult(
            interrupted=self.interrupted,
            thread_id=f"{event.get('tenant_id', '')}:{event.get('upload_id', '')}",
            run_id="run-1",
            governance_status=None if self.interrupted else "blocked",
            review_panel=_fake_panel(event) if self.interrupted else None,
        )


@dataclass
class _FakeReviewPendingPublisher:
    """Doubles WeaknessReviewPendingOutboxWriter — records emitted panels."""

    emitted: list[dict[str, Any]] = field(default_factory=list)

    async def publish_review_pending(
        self, *, panel: dict[str, Any], traceparent: str = "", tracestate: str = ""
    ) -> str:
        self.emitted.append({"panel": panel, "traceparent": traceparent, "tracestate": tracestate})
        return f"row-{len(self.emitted)}"


@dataclass
class _FakeInbox:
    """Minimal InboxIdempotencyStore double — runs fn once per key."""

    seen: set[str] = field(default_factory=set)

    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool:
        if key in self.seen:
            return False
        self.seen.add(key)
        await fn()
        return True


def _uploaded(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "upload_id": "0190aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee",
        "tenant_id": "tenant-9",
        "learner_gcid": "learner-1",
        "source_blob_uri": "gs://chora-weakness-uploads/t/u.pdf",
        "source_mime_type": "application/pdf",
        "upload_kind": "marked_test",
        "reservation_id": "res-1",
    }
    body.update(overrides)
    return body


def _msg(body: dict[str, Any] | None = None, **attrs: str) -> _FakeMsg:
    return _FakeMsg(
        data=json.dumps(_uploaded() if body is None else body).encode("utf-8"),
        attributes=attrs,
    )


def _sub(
    runner: _FakeCrewRunner, inbox: _FakeInbox, publisher: _FakeReviewPendingPublisher | None = None
) -> WeaknessAnalyserCrewGraphSubscriber:
    return WeaknessAnalyserCrewGraphSubscriber(crew_runner=runner, inbox=inbox, review_pending_publisher=publisher)


def test_default_inbox_ttl_covers_redelivery_window() -> None:
    from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyser_crew_graph_subscriber import (
        DEFAULT_INBOX_TTL,
    )

    sub = _sub(_FakeCrewRunner(), _FakeInbox())
    assert sub.inbox_ttl == DEFAULT_INBOX_TTL
    assert _dt.timedelta(days=7) <= DEFAULT_INBOX_TTL


class TestHandleMessage:
    async def test_starts_run_to_interrupt_and_acks(self) -> None:
        runner, inbox = _FakeCrewRunner(), _FakeInbox()
        sub = _sub(runner, inbox)
        msg = _msg(event_id="e1")

        await sub.handle_message(msg)

        assert msg.acked and not msg.nacked
        assert len(runner.events) == 1
        assert runner.events[0]["upload_id"] == _uploaded()["upload_id"]
        assert runner.events[0]["reservation_id"] == "res-1"

    async def test_merges_tenant_and_gcid_from_attributes(self) -> None:
        runner, inbox = _FakeCrewRunner(), _FakeInbox()
        sub = _sub(runner, inbox)
        body = _uploaded()
        body.pop("tenant_id")
        body.pop("learner_gcid")
        msg = _msg(body, event_id="e1", tenant_id="tenant-attr", gcid="gcid-attr")

        await sub.handle_message(msg)

        assert msg.acked
        assert runner.events[0]["tenant_id"] == "tenant-attr"
        assert runner.events[0]["learner_gcid"] == "gcid-attr"

    async def test_threads_traceparent_into_event(self) -> None:
        runner, inbox = _FakeCrewRunner(), _FakeInbox()
        sub = _sub(runner, inbox)
        tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        msg = _msg(event_id="e1", traceparent=tp, tracestate="rojo=1")

        await sub.handle_message(msg)

        assert runner.events[0]["traceparent"] == tp
        assert runner.events[0]["tracestate"] == "rojo=1"

    async def test_duplicate_upload_runs_once_both_ack(self) -> None:
        runner, inbox = _FakeCrewRunner(), _FakeInbox()
        sub = _sub(runner, inbox)
        m1 = _msg(event_id="e1")
        m2 = _msg(event_id="e2")

        await sub.handle_message(m1)
        await sub.handle_message(m2)

        assert m1.acked and m2.acked
        assert len(runner.events) == 1  # the paused run must not be re-started

    async def test_missing_upload_id_and_no_attr_key_nacks(self) -> None:
        runner, inbox = _FakeCrewRunner(), _FakeInbox()
        sub = _sub(runner, inbox)
        body = _uploaded()
        body.pop("upload_id")
        msg = _msg(body)

        await sub.handle_message(msg)

        assert msg.nacked and not msg.acked
        assert runner.events == []

    async def test_decode_failure_nacks(self) -> None:
        runner, inbox = _FakeCrewRunner(), _FakeInbox()
        sub = _sub(runner, inbox)
        msg = _FakeMsg(data=b"{not json", attributes={"event_id": "e1"})

        await sub.handle_message(msg)

        assert msg.nacked and not msg.acked
        assert runner.events == []

    async def test_non_dict_json_body_nacks(self) -> None:
        runner, inbox = _FakeCrewRunner(), _FakeInbox()
        sub = _sub(runner, inbox)
        msg = _FakeMsg(data=b"[1, 2, 3]", attributes={"event_id": "e1"})

        await sub.handle_message(msg)

        assert msg.nacked and not msg.acked
        assert runner.events == []

    async def test_graph_failure_nacks_and_leaves_key_unclaimed(self) -> None:
        runner, inbox = _FakeCrewRunner(raises=True), _FakeInbox()
        sub = _sub(runner, inbox)
        msg = _msg(event_id="e1")

        await sub.handle_message(msg)  # must NOT raise

        assert msg.nacked and not msg.acked

    async def test_crew_error_nacks(self) -> None:
        # handle_uploaded raises WeaknessCrewError on a malformed event → nack.
        runner, inbox = _FakeCrewRunner(raises_crew_error=True), _FakeInbox()
        sub = _sub(runner, inbox)
        msg = _msg(event_id="e1")

        await sub.handle_message(msg)

        assert msg.nacked and not msg.acked

    async def test_falls_back_to_attribute_key_when_upload_id_absent(self) -> None:
        runner, inbox = _FakeCrewRunner(), _FakeInbox()
        sub = _sub(runner, inbox)
        body = _uploaded()
        body.pop("upload_id")
        msg = _msg(body, idempotency_key="weakness_doc.uploaded.xyz")

        await sub.handle_message(msg)

        assert msg.acked
        assert len(runner.events) == 1


class TestEmitsReviewPending:
    """CHO-1973 Wave A — on the HITL interrupt the graph subscriber emits the
    review_pending panel via the outbox (it no longer discards the result)."""

    async def test_emits_review_pending_on_interrupt(self) -> None:
        runner, inbox = _FakeCrewRunner(interrupted=True), _FakeInbox()
        pub = _FakeReviewPendingPublisher()
        sub = _sub(runner, inbox, pub)
        tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        msg = _msg(event_id="e1", traceparent=tp, tracestate="rojo=1")

        await sub.handle_message(msg)

        assert msg.acked
        assert len(pub.emitted) == 1
        emitted = pub.emitted[0]
        assert emitted["panel"]["upload_id"] == _uploaded()["upload_id"]
        assert emitted["panel"]["proposed_edges"][0]["proposed_edge_id"] == "pe-0"
        # traceparent/tracestate thread through to the emitted event (P4 OTel)
        assert emitted["traceparent"] == tp
        assert emitted["tracestate"] == "rojo=1"

    async def test_does_not_emit_when_not_interrupted(self) -> None:
        # a safety BLOCK / completed run has no panel -> nothing to emit.
        runner, inbox = _FakeCrewRunner(interrupted=False), _FakeInbox()
        pub = _FakeReviewPendingPublisher()
        sub = _sub(runner, inbox, pub)
        msg = _msg(event_id="e1")

        await sub.handle_message(msg)

        assert msg.acked
        assert pub.emitted == []

    async def test_duplicate_upload_emits_review_pending_once(self) -> None:
        runner, inbox = _FakeCrewRunner(), _FakeInbox()
        pub = _FakeReviewPendingPublisher()
        sub = _sub(runner, inbox, pub)
        m1 = _msg(event_id="e1")
        m2 = _msg(event_id="e2")

        await sub.handle_message(m1)
        await sub.handle_message(m2)

        assert m1.acked and m2.acked
        assert len(pub.emitted) == 1  # dedupe: the paused run emits its panel once

    async def test_interrupt_without_publisher_still_acks(self) -> None:
        # misconfiguration guard: no publisher wired in graph mode must NOT crash
        # the StreamingPull subscriber (it logs loud + acks the start).
        runner, inbox = _FakeCrewRunner(interrupted=True), _FakeInbox()
        sub = _sub(runner, inbox, None)
        msg = _msg(event_id="e1")

        await sub.handle_message(msg)

        assert msg.acked and not msg.nacked

    async def test_publisher_failure_nacks(self) -> None:
        # the review_pending emit runs INSIDE the inbox dedupe guard: if it fails,
        # the message NACKs (Pub/Sub redelivers) — the run never half-emits + acks.
        class _BoomPublisher:
            async def publish_review_pending(
                self, *, panel: dict[str, Any], traceparent: str = "", tracestate: str = ""
            ) -> str:
                raise RuntimeError("outbox insert exploded")

        runner, inbox = _FakeCrewRunner(interrupted=True), _FakeInbox()
        sub = WeaknessAnalyserCrewGraphSubscriber(
            crew_runner=runner, inbox=inbox, review_pending_publisher=_BoomPublisher()
        )
        msg = _msg(event_id="e1")

        await sub.handle_message(msg)  # must NOT raise

        assert msg.nacked and not msg.acked
