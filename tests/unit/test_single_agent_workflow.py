"""RED: the generic single-agent workflow (ADR-254 D5).

ONE code path for every "consume request -> validate -> dispatch one role ->
map completion -> emit result" lane: the two fog folds (kg_exploration,
companion_reflection) today, the five single-agent crews next. A lane is a
``LaneContract`` (decode / dispatch / result, optional rejected + cap); the
engine owns the park, the idempotent emit, the inbox, the ack discipline.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import DISPATCH_INTERRUPT_KEY
from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import AgentDispatchError
from chora_ai_kernel_orchestrator.orchestrators.single_agent_workflow import (
    DispatchSpec,
    LaneContract,
    ResultEvent,
    SingleAgentRequestSubscriber,
    SingleAgentWorkflowRunner,
    Skipped,
    build_single_agent_graph,
)

TENANT = "11111111-1111-7111-8111-111111111111"
GCID = "00000000-0000-7000-8000-000000001999"
EVENT = "01a02a3b-c5b2-7c13-bf7d-a9ecdb8ccfb1"


# --------------------------------------------------------------------------- #
# a minimal lane contract for the engine tests
# --------------------------------------------------------------------------- #


def _decode(body: dict[str, Any], attrs: dict[str, str]) -> dict[str, Any] | Skipped:
    if not attrs.get("tenant_id"):
        raise ValueError("missing tenant")
    if body.get("skip"):
        return Skipped("nothing_to_do")
    return {
        "event_id": attrs["event_id"],
        "idempotency_key": attrs.get("idempotency_key", ""),
        "tenant_id": attrs["tenant_id"],
        "gcid": attrs.get("gcid", ""),
        "traceparent": attrs.get("traceparent", ""),
        "tracestate": "",
        "request": body,
    }


def _dispatch(state: dict[str, Any]) -> DispatchSpec:
    return DispatchSpec(
        execution_id=f"{state['event_id']}:echo",
        input_payload={"text": state["request"].get("text", ""), "gcid": state["gcid"]},
        workflow_id=state["event_id"],
    )


def _result(state: dict[str, Any], completion: dict[str, Any]) -> ResultEvent | None:
    if completion["status"] != "OK":
        return ResultEvent(
            topic="chora.test.echo.completed.v1",
            event_type="test.echo.completed",
            idempotency_key=f"echo.{state['event_id']}",
            body={"status": "FAILED", "error": completion.get("error_message", "")},
            tenant_id=state["tenant_id"],
            gcid=state["gcid"],
            workflow_id=state["event_id"],
        )
    return ResultEvent(
        topic="chora.test.echo.completed.v1",
        event_type="test.echo.completed",
        idempotency_key=f"echo.{state['event_id']}",
        body={"status": "OK", "reply": json.loads(completion["output_payload"])["reply"]},
        tenant_id=state["tenant_id"],
        gcid=state["gcid"],
        workflow_id=state["event_id"],
        traceparent=state["traceparent"],
    )


def _rejected(state: dict[str, Any]) -> ResultEvent:
    return ResultEvent(
        topic="chora.test.echo.completed.v1",
        event_type="test.echo.completed",
        idempotency_key=f"echo.{state['event_id']}",
        body={"status": "REJECTED", "error_code": "tenant_in_flight_cap"},
        tenant_id=state["tenant_id"],
        gcid=state["gcid"],
        workflow_id=state["event_id"],
    )


def _contract(**over: Any) -> LaneContract:
    base = dict(
        name="echo_lane",
        role="echo",
        request_kind="echo.requested",
        decode=_decode,
        dispatch=_dispatch,
        result=_result,
        rejected=_rejected,
        tenant_inflight_cap=None,
    )
    base.update(over)
    return LaneContract(**base)


class _Executor:
    """Answers inline (no park) unless told to park (for REAL, through LangGraph's
    ``interrupt()``, exactly as ``PubSubAgentExecutor`` does: the node re-runs on
    resume and ``interrupt()`` hands back the completion) or to fail."""

    def __init__(self, *, reply: str = "hi", fail: str | None = None, park: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._reply, self._fail, self._park = reply, fail, park

    async def execute(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self._park:
            completion = interrupt({DISPATCH_INTERRUPT_KEY: {"agent_role": kw["agent_role"]}}) or {}
            return type(
                "R",
                (),
                {"output_payload": str(completion.get("output_payload", "")), "input_tokens": 0, "output_tokens": 0},
            )()
        if self._fail:
            raise AgentDispatchError(self._fail)
        return type(
            "R", (), {"output_payload": json.dumps({"reply": self._reply}), "input_tokens": 0, "output_tokens": 0}
        )()


class _Outbox:
    def __init__(self) -> None:
        self.queued: list[dict[str, Any]] = []

    async def queue_request(self, request: dict[str, Any]) -> str:
        self.queued.append(request)
        return f"row-{len(self.queued)}"


class _Ledger:
    def __init__(self, parked: int = 0) -> None:
        self.parked = parked

    async def count_parked(self, *, tenant_id: str, agent_role: str) -> int:
        return self.parked


def _attrs(**over: str) -> dict[str, str]:
    a = {
        "event_id": EVENT,
        "idempotency_key": f"echo.requested.{EVENT}",
        "tenant_id": TENANT,
        "gcid": GCID,
        "traceparent": "00-aa-bb-01",
    }
    a.update(over)
    return a


def _graph(contract: LaneContract, executor: Any, outbox: Any) -> Any:
    return build_single_agent_graph(
        contract=contract,
        executor=executor,
        outbox_writer=outbox,
        source_project="chora-489812",
        checkpointer=InMemorySaver(),
    )


def _cfg(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


# --------------------------------------------------------------------------- #
# graph: dispatch -> emit
# --------------------------------------------------------------------------- #


def test_ok_completion_emits_the_contracts_result_through_the_outbox() -> None:
    ex, ob = _Executor(reply="hello"), _Outbox()
    graph = _graph(_contract(), ex, ob)
    state = _decode({"text": "x"}, _attrs())
    out = asyncio.run(graph.ainvoke(state, config=_cfg("echo_lane:" + EVENT)))

    call = ex.calls[0]
    assert call["agent_role"] == "echo" and call["workflow_id"] == EVENT
    assert call["execution_id"] == f"{EVENT}:echo"
    assert json.loads(call["input_payload"]) == {"text": "x", "gcid": GCID}
    assert len(ob.queued) == 1
    req = ob.queued[0]
    assert req["topic"] == "chora.test.echo.completed.v1" and req["event_type"] == "test.echo.completed"
    assert req["idempotency_key"] == f"echo.{EVENT}" and req["workflow_id"] == EVENT
    assert req["body"] == {"status": "OK", "reply": "hello"}
    env = req["envelope"]
    for k in (
        "event_id",
        "idempotency_key",
        "tenant_id",
        "gcid",
        "occurred_at",
        "published_at",
        "traceparent",
        "tracestate",
        "source_project",
        "source_service",
        "schema_version",
        "event_topic",
    ):
        assert k in env, k
    assert env["event_topic"] == "chora.test.echo.completed.v1" and env["traceparent"] == "00-aa-bb-01"
    assert out["result_key"] == f"echo.{EVENT}"


def test_a_failed_dispatch_reaches_the_contract_as_failed_never_raises_out() -> None:
    ex, ob = _Executor(fail="FAILED: unknown_request_source"), _Outbox()
    graph = _graph(_contract(), ex, ob)
    out = asyncio.run(graph.ainvoke(_decode({"text": "x"}, _attrs()), config=_cfg("t1")))
    assert ob.queued[0]["body"]["status"] == "FAILED"
    assert "unknown_request_source" in ob.queued[0]["body"]["error"]
    assert out["completion"]["status"] == "FAILED"


def test_a_park_is_never_swallowed() -> None:
    ex, ob = _Executor(park=True), _Outbox()
    graph = _graph(_contract(), ex, ob)
    out = asyncio.run(graph.ainvoke(_decode({"text": "x"}, _attrs()), config=_cfg("t2")))
    assert out.get("__interrupt__"), "the park must reach LangGraph"
    assert ob.queued == [], "nothing is emitted before the completion"


def test_a_contract_that_returns_none_emits_nothing() -> None:
    ex, ob = _Executor(), _Outbox()
    graph = _graph(_contract(result=lambda s, c: None), ex, ob)
    out = asyncio.run(graph.ainvoke(_decode({"text": "x"}, _attrs()), config=_cfg("t3")))
    assert ob.queued == [] and out.get("result_key") in (None, "")


def test_a_re_run_never_emits_twice() -> None:
    ex, ob = _Executor(), _Outbox()
    graph = _graph(_contract(), ex, ob)
    cfg = _cfg("t4")
    asyncio.run(graph.ainvoke(_decode({"text": "x"}, _attrs()), config=cfg))
    asyncio.run(graph.ainvoke(Command(resume={}), config=cfg))
    assert len(ob.queued) == 1


def test_the_synthetic_traceparent_fallback_is_valid_w3c() -> None:
    ex, ob = _Executor(), _Outbox()
    graph = _graph(_contract(), ex, ob)
    asyncio.run(graph.ainvoke(_decode({"text": "x"}, _attrs(traceparent="")), config=_cfg("t5")))
    tp = ob.queued[0]["envelope"]["traceparent"]
    parts = tp.split("-")
    assert len(parts) == 4 and parts[0] == "00" and len(parts[1]) == 32 and len(parts[2]) == 16


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #


def _runner(contract: LaneContract, executor: Any, outbox: Any, ledger: Any = None) -> SingleAgentWorkflowRunner:
    return SingleAgentWorkflowRunner(
        graph=_graph(contract, executor, outbox),
        contract=contract,
        outbox_writer=outbox,
        source_project="chora-489812",
        ledger=ledger,
    )


def test_runner_reports_a_park_as_parked_and_a_finished_run_as_completed() -> None:
    parked = _runner(_contract(), _Executor(park=True), _Outbox())
    r1 = asyncio.run(parked.handle_request({"text": "x"}, _attrs()))
    assert r1.outcome == "parked" and r1.thread_id == f"echo_lane:{EVENT}"
    done = _runner(_contract(), _Executor(), _Outbox())
    r2 = asyncio.run(done.handle_request({"text": "x"}, _attrs()))
    assert r2.outcome == "completed"


def test_runner_skips_without_touching_the_graph() -> None:
    ex = _Executor()
    r = asyncio.run(_runner(_contract(), ex, _Outbox()).handle_request({"skip": True}, _attrs()))
    assert r.outcome == "skipped" and r.detail == "nothing_to_do" and ex.calls == []


def test_runner_refuses_a_request_the_contract_refuses() -> None:
    with pytest.raises(ValueError, match="tenant"):
        asyncio.run(_runner(_contract(), _Executor(), _Outbox()).handle_request({"text": "x"}, _attrs(tenant_id="")))


def test_runner_rejects_over_the_tenant_inflight_cap_with_the_contracts_rejected_event() -> None:
    ex, ob = _Executor(park=True), _Outbox()
    r = asyncio.run(
        _runner(_contract(tenant_inflight_cap=2), ex, ob, ledger=_Ledger(parked=2)).handle_request(
            {"text": "x"}, _attrs()
        )
    )
    assert r.outcome == "rejected" and ex.calls == []
    assert ob.queued[0]["body"] == {"status": "REJECTED", "error_code": "tenant_in_flight_cap"}


def test_runner_under_the_cap_dispatches() -> None:
    ex = _Executor(park=True)
    r = asyncio.run(
        _runner(_contract(tenant_inflight_cap=2), ex, _Outbox(), ledger=_Ledger(parked=1)).handle_request(
            {"text": "x"}, _attrs()
        )
    )
    assert r.outcome == "parked" and len(ex.calls) == 1


def test_runner_resumes_the_thread_a_completion_names_and_refuses_a_nameless_one() -> None:
    ex, ob = _Executor(park=True), _Outbox()
    runner = _runner(_contract(), ex, ob)
    r = asyncio.run(runner.handle_request({"text": "x"}, _attrs()))
    completion = {
        "thread_id": r.thread_id,
        "agent_role": "echo",
        "status": "OK",
        "output_payload": json.dumps({"reply": "resumed"}),
        "idempotency_key": "k",
    }
    done = asyncio.run(runner.handle_completion(completion))
    assert done.outcome == "completed"
    assert ob.queued[0]["body"] == {"status": "OK", "reply": "resumed"}
    with pytest.raises(ValueError, match="thread_id"):
        asyncio.run(runner.handle_completion({"agent_role": "echo", "status": "OK"}))


# --------------------------------------------------------------------------- #
# subscriber
# --------------------------------------------------------------------------- #


class _Msg:
    def __init__(self, body: Any, attrs: dict[str, str]) -> None:
        self.data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.attributes = attrs
        self.acked = self.nacked = False

    def ack(self) -> None:
        self.acked = True

    def nack(self) -> None:
        self.nacked = True


class _Inbox:
    def __init__(self) -> None:
        self.keys: list[str] = []

    async def process(self, *, key: str, ttl: Any, fn: Any) -> bool:
        self.keys.append(key)
        await fn()
        return True


@pytest.mark.asyncio
async def test_subscriber_acks_a_parked_request_and_keys_the_inbox_on_the_request_key() -> None:
    ex, ob, inbox = _Executor(park=True), _Outbox(), _Inbox()
    sub = SingleAgentRequestSubscriber(runner=_runner(_contract(), ex, ob), inbox=inbox, contract=_contract())
    msg = _Msg({"text": "x"}, _attrs())
    await sub.handle_message(msg)
    assert msg.acked and not msg.nacked
    assert inbox.keys == [f"echo.requested:echo.requested.{EVENT}"]


@pytest.mark.asyncio
async def test_subscriber_nacks_bad_json_and_a_refused_request() -> None:
    sub = SingleAgentRequestSubscriber(
        runner=_runner(_contract(), _Executor(), _Outbox()), inbox=_Inbox(), contract=_contract()
    )
    bad = _Msg(b"not json", _attrs())
    await sub.handle_message(bad)
    assert bad.nacked
    refused = _Msg({"text": "x"}, _attrs(tenant_id=""))
    await sub.handle_message(refused)
    assert refused.nacked


@pytest.mark.asyncio
async def test_subscriber_acks_a_skipped_request() -> None:
    sub = SingleAgentRequestSubscriber(
        runner=_runner(_contract(), _Executor(), _Outbox()), inbox=_Inbox(), contract=_contract()
    )
    msg = _Msg({"skip": True}, _attrs())
    await sub.handle_message(msg)
    assert msg.acked
