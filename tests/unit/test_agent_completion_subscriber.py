"""RED — ADR-253 D2: the completion lane that resumes a parked run.

⚠ Verification for this lane cannot lean on logs. The `buildout-cost-pause`
exclusion on the `_Default` sink drops severity>=DEFAULT application logs
platform wide, so a silently dead subscriber is indistinguishable from a healthy
one in Cloud Logging. Every behaviour that matters is therefore asserted here on
a code-defined discriminator — ack vs nack, ran vs deduped — not on a log line.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _FakeMsg:
    data: bytes
    attributes: dict[str, str] = field(default_factory=dict)
    acked: bool = False
    nacked: bool = False

    def ack(self) -> None:
        self.acked = True

    def nack(self) -> None:
        self.nacked = True


@dataclass
class _FakeRunner:
    completions: list[dict[str, Any]] = field(default_factory=list)
    raises: Exception | None = None

    async def handle_completion(self, completion: dict[str, Any]) -> None:
        self.completions.append(completion)
        if self.raises:
            raise self.raises


@dataclass
class _FakeInbox:
    seen: set = field(default_factory=set)
    keys: list[str] = field(default_factory=list)

    async def process(self, *, key: str, ttl: _dt.timedelta, fn: Any) -> bool:
        self.keys.append(key)
        if key in self.seen:
            return False
        self.seen.add(key)
        await fn()
        return True


def _sub(runner, inbox):
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_completion_subscriber import (
        AgentCompletionSubscriber,
    )

    return AgentCompletionSubscriber(runner=runner, inbox=inbox)


_BODY = {
    "agent_role": "oe_evaluate",
    "execution_id": "sub-1:tsq-1:1",
    "thread_id": "sub-1",
    "idempotency_key": "agent_dispatch.oe_evaluate.sub-1:tsq-1:1",
    "status": "OK",
    "output_payload": json.dumps({"criterion_scores": [], "comment": "ok"}),
    "input_tokens": 10,
    "output_tokens": 4,
}


def _msg(body: dict[str, Any] | None = None, **attrs: str) -> _FakeMsg:
    base = {
        "idempotency_key": "agent_dispatch.oe_evaluate.sub-1:tsq-1:1.completed",
        "tenant_id": "11111111-1111-7111-8111-111111111111",
    }
    base.update(attrs)
    return _FakeMsg(json.dumps(body if body is not None else _BODY).encode(), base)


async def test_a_completion_resumes_the_run_and_acks() -> None:
    runner, inbox = _FakeRunner(), _FakeInbox()
    msg = _msg()

    await _sub(runner, inbox).handle_message(msg)

    assert msg.acked and not msg.nacked
    assert runner.completions[0]["thread_id"] == "sub-1"
    assert runner.completions[0]["status"] == "OK"


async def test_a_redelivered_completion_does_not_resume_twice() -> None:
    """Receive-side idempotency stays mandatory even though the publish side
    commits in one transaction — Pub/Sub is at-least-once regardless."""
    runner, inbox = _FakeRunner(), _FakeInbox()

    await _sub(runner, inbox).handle_message(_msg())
    second = _msg()
    await _sub(runner, inbox).handle_message(second)

    assert len(runner.completions) == 1, "resumed the same completion twice"
    assert second.acked, "a dedupe hit must ACK, not redeliver forever"


async def test_the_completion_dedup_key_is_distinct_from_the_request_key() -> None:
    """Request and completion share one idempotency_keys table. If the
    completion deduped on the REQUEST's key, the very first completion would be
    swallowed as an already-seen request."""
    runner, inbox = _FakeRunner(), _FakeInbox()
    await _sub(runner, inbox).handle_message(_msg())

    assert inbox.keys == ["agent_dispatch.oe_evaluate.sub-1:tsq-1:1.completed"]
    assert inbox.keys[0] != _BODY["idempotency_key"]


async def test_a_runner_failure_nacks_for_redelivery() -> None:
    runner = _FakeRunner(raises=RuntimeError("resume blew up"))
    msg = _msg()

    await _sub(runner, _FakeInbox()).handle_message(msg)

    assert msg.nacked and not msg.acked


async def test_an_undecodable_body_nacks_and_never_reaches_the_runner() -> None:
    runner = _FakeRunner()
    msg = _FakeMsg(b"not json at all", {"idempotency_key": "k1"})

    await _sub(runner, _FakeInbox()).handle_message(msg)

    assert msg.nacked
    assert runner.completions == []


async def test_a_missing_attribute_key_falls_back_to_the_body() -> None:
    """Same precedence the request-side subscriber uses: attributes, then the
    body. A publisher that forgets the attribute still gets deduped rather than
    silently losing the guard."""
    runner, inbox = _FakeRunner(), _FakeInbox()
    msg = _FakeMsg(json.dumps(_BODY).encode(), {})

    await _sub(runner, inbox).handle_message(msg)

    assert msg.acked
    assert inbox.keys == ["agent_dispatch.oe_evaluate.sub-1:tsq-1:1.completed"]


async def test_a_completion_keyed_nowhere_at_all_nacks() -> None:
    """When neither the attributes nor the body carry a key the lane cannot
    dedupe, and proceeding unguarded is worse than redelivering."""
    runner = _FakeRunner()
    body = {k: v for k, v in _BODY.items() if k != "idempotency_key"}
    msg = _FakeMsg(json.dumps(body).encode(), {})

    await _sub(runner, _FakeInbox()).handle_message(msg)

    assert msg.nacked
    assert runner.completions == []


async def test_the_trace_context_is_threaded_from_attributes() -> None:
    """One run must stay one trace across the new hop, or the platform's
    tracing claims break (ADR-253 D4)."""
    runner, inbox = _FakeRunner(), _FakeInbox()
    tp = "00-" + "c" * 32 + "-" + "d" * 16 + "-01"

    await _sub(runner, inbox).handle_message(_msg(traceparent=tp, tracestate="x=1"))

    assert runner.completions[0]["traceparent"] == tp
    assert runner.completions[0]["tracestate"] == "x=1"


async def test_a_failed_status_is_still_delivered_to_the_runner() -> None:
    """A FAILED completion is how a dead agent call reaches the graph so the run
    can end cleanly. Dropping it here would park the submission forever."""
    runner, inbox = _FakeRunner(), _FakeInbox()
    body = {**_BODY, "status": "FAILED", "error_message": "model gateway 503"}

    msg = _msg(body)
    await _sub(runner, inbox).handle_message(msg)

    assert runner.completions[0]["status"] == "FAILED"
    assert msg.acked, "a delivered failure is processed, not redelivered"


async def test_never_raises_out_of_handle_message() -> None:
    """A raise here tears down the StreamingPull subscriber for every lane."""
    runner = _FakeRunner()

    class _Exploding:
        async def process(self, **_: Any) -> bool:
            raise RuntimeError("inbox down")

    msg = _msg()
    await _sub(runner, _Exploding()).handle_message(msg)
    assert msg.nacked
