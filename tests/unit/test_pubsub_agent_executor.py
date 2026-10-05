"""RED — ADR-253 D1/D2: the Pub/Sub-backed executor parks the graph.

D1 promises the ``_ExecutorLike`` seam survives, so crew nodes are NOT rewritten.
That means ``execute()`` must keep its exact signature and its
``AgentExecutorResponse`` return type while, underneath, publishing a request and
parking the run until a completion event resumes it.

These tests drive a REAL compiled LangGraph with a REAL checkpointer, because
the whole design rests on interrupt/resume semantics that a mock would let us
assume rather than verify.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class _S(TypedDict, total=False):
    calls: list
    out: str


def _executor(**kw: Any):
    from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
        PubSubAgentExecutor,
    )

    return PubSubAgentExecutor(source_project="chora-489812", **kw)


def _graph(executor, saver, role: str = "oe_evaluate"):
    g: StateGraph = StateGraph(_S)

    async def dispatch(state: _S) -> dict:
        calls = list(state.get("calls", []))
        calls.append("body-entered")
        resp = await executor.execute(
            execution_id="01a02062-e5b4-7870-8fca-53ce363cd542:tsq-1:1",
            tenant_id="11111111-1111-7111-8111-111111111111",
            agid="",
            agent_role=role,
            input_payload=json.dumps(
                {"mode": "evaluate", "gcid": "g1", "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
            ),
        )
        return {"calls": calls, "out": resp.output_payload}

    g.add_node("dispatch", dispatch)
    g.add_edge(START, "dispatch")
    g.add_edge("dispatch", END)
    return g.compile(checkpointer=saver)


_CFG = {"configurable": {"thread_id": "01a02062-e5b4-7870-8fca-53ce363cd542"}}


async def test_execute_parks_the_graph_and_surfaces_the_dispatch_request() -> None:
    saver = InMemorySaver()
    out = await _graph(_executor(), saver).ainvoke({"calls": []}, config=_CFG)

    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
        DISPATCH_INTERRUPT_KEY,
    )

    interrupts = out.get("__interrupt__") or []
    assert interrupts, "execute() did not park the graph"
    request = interrupts[0].value[DISPATCH_INTERRUPT_KEY]
    assert request["topic"] == "chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1"
    assert request["body"]["reply_topic"] == ("chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1")
    assert request["body"]["thread_id"] == "01a02062-e5b4-7870-8fca-53ce363cd542"
    assert request["tenant_id"] == "11111111-1111-7111-8111-111111111111"
    # The traceparent is load-bearing: it rides HTTP session state today, so on
    # Pub/Sub it must ride the envelope or one run stops being one trace.
    assert request["envelope"]["traceparent"] == "00-" + "a" * 32 + "-" + "b" * 16 + "-01"


async def test_a_completion_resumes_the_run_and_maps_to_an_executor_response() -> None:
    saver = InMemorySaver()
    ex = _executor()
    await _graph(ex, saver).ainvoke({"calls": []}, config=_CFG)

    completion = {
        "status": "OK",
        "output_payload": json.dumps({"criterion_scores": [{"score": 4}], "comment": "solid"}),
        "input_tokens": 120,
        "output_tokens": 45,
        "tokens_consumed_total": 165,
    }
    out = await _graph(ex, saver).ainvoke(Command(resume=completion), config=_CFG)

    assert not (out.get("__interrupt__") or []), "still parked after a completion"
    assert json.loads(out["out"])["comment"] == "solid"


async def test_token_counts_survive_the_wire() -> None:
    """The pipeline_trace rows and the O+ per-agent tiles read these; dropping
    them would show every agent consuming zero tokens."""
    saver = InMemorySaver()
    ex = _executor()
    captured: dict[str, Any] = {}

    g: StateGraph = StateGraph(_S)

    async def node(state: _S) -> dict:
        resp = await ex.execute(
            execution_id="e1",
            tenant_id="t1",
            agid="",
            agent_role="oe_moderate",
            input_payload="{}",
        )
        captured["resp"] = resp
        return {"out": "done"}

    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    graph = g.compile(checkpointer=saver)

    await graph.ainvoke({}, config=_CFG)
    await graph.ainvoke(
        Command(
            resume={
                "status": "OK",
                "output_payload": "{}",
                "input_tokens": 7,
                "output_tokens": 3,
                "tokens_consumed_total": 10,
            }
        ),
        config=_CFG,
    )
    resp = captured["resp"]
    assert (resp.input_tokens, resp.output_tokens, resp.tokens_consumed_total) == (7, 3, 10)
    assert resp.execution_id == "e1"


async def test_a_failed_completion_raises_instead_of_fabricating_a_result() -> None:
    """ADR-253 D4 carries the outcome as a status discriminator INSIDE the
    completion. A FAILED status that returned an empty success would grade a
    learner's answer as zero on an infrastructure fault."""
    saver = InMemorySaver()
    ex = _executor()
    await _graph(ex, saver).ainvoke({"calls": []}, config=_CFG)

    with pytest.raises(Exception, match="oe_evaluate"):
        await _graph(ex, saver).ainvoke(
            Command(resume={"status": "FAILED", "error_message": "model gateway 503"}),
            config=_CFG,
        )


async def test_a_completion_with_no_status_is_refused() -> None:
    """A malformed completion must not be read as success by omission."""
    saver = InMemorySaver()
    ex = _executor()
    await _graph(ex, saver).ainvoke({"calls": []}, config=_CFG)

    with pytest.raises(Exception):  # noqa: B017 — intentionally broad: the graph must raise SOME exception on a malformed resume
        await _graph(ex, saver).ainvoke(Command(resume={"output_payload": "{}"}), config=_CFG)


async def test_the_idempotency_key_is_stable_across_node_re_execution() -> None:
    """LangGraph re-runs a node from the top on resume, so the node rebuilds the
    request. A non-deterministic key would queue a SECOND dispatch every time a
    run resumed."""
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
        DISPATCH_INTERRUPT_KEY,
    )

    saver = InMemorySaver()
    ex = _executor()
    first = await _graph(ex, saver).ainvoke({"calls": []}, config=_CFG)
    key1 = first["__interrupt__"][0].value[DISPATCH_INTERRUPT_KEY]["idempotency_key"]

    # A redelivery of the SAME trigger re-enters the same thread and re-parks.
    saver2 = InMemorySaver()
    second = await _graph(ex, saver2).ainvoke({"calls": []}, config=_CFG)
    key2 = second["__interrupt__"][0].value[DISPATCH_INTERRUPT_KEY]["idempotency_key"]

    assert key1 == key2 == "agent_dispatch.oe_evaluate.01a02062-e5b4-7870-8fca-53ce363cd542:tsq-1:1"


async def test_it_satisfies_the_executor_like_protocol_signature() -> None:
    """D1: crew nodes are not rewritten around a new seam, so the Pub/Sub
    implementation must accept every keyword the crews call it with.

    The HTTP executor this used to be compared against was deleted with the
    transport switch (RULING A, 2026-08-23), so the reference is now the
    protocol the crew nodes are typed against, which is what the parity claim
    was always really about."""
    import inspect

    from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
        PubSubAgentExecutor,
    )
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import _ExecutorLike

    protocol_params = set(inspect.signature(_ExecutorLike.execute).parameters)
    pubsub_params = set(inspect.signature(PubSubAgentExecutor.execute).parameters)
    assert protocol_params <= pubsub_params, protocol_params - pubsub_params


async def test_an_unknown_role_fails_loud_before_parking() -> None:
    """Parking on a role with no topic pair would strand the run forever."""
    saver = InMemorySaver()
    ex = _executor(allowed_roles=("oe_evaluate", "oe_moderate"))
    with pytest.raises(KeyError, match="qgen_question"):
        await _graph(ex, saver, role="qgen_question").ainvoke({"calls": []}, config=_CFG)


async def test_it_maps_an_agent_payload_through_the_shared_mapper() -> None:
    """The completion path must produce the SAME AgentExecutorResponse the
    shared mapper does. The token split in particular is derived from the
    agent's OWN emitted JSON, not from the wire, so a Pub/Sub path that
    re-derived it would quietly change every pipeline_trace row and every O+
    per-agent token tile."""
    import json as _json

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from chora_ai_kernel_orchestrator.adapter.agent_io import (
        map_agent_response as _map_response,
    )

    agent_output = _json.dumps(
        {
            "criterion_scores": [{"criterion_id": "c1", "score": 4.0}],
            "comment": "well argued",
            "model_id": "gemini-3.1-pro-preview",
            "response_id": "resp-77",
            "input_tokens": 812,
            "output_tokens": 143,
            "tokens_consumed_total": 955,
        }
    )

    expected = _map_response(execution_id="e1", terminal_text=agent_output, agent_role="oe_evaluate")

    saver, ex = InMemorySaver(), _executor()
    captured: dict[str, Any] = {}

    g: StateGraph = StateGraph(_S)

    async def node(state: _S) -> dict:
        captured["resp"] = await ex.execute(
            execution_id="e1",
            tenant_id="t1",
            agid="",
            agent_role="oe_evaluate",
            input_payload="{}",
        )
        return {"out": "done"}

    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    graph = g.compile(checkpointer=saver)

    await graph.ainvoke({}, config=_CFG)
    await graph.ainvoke(
        Command(resume={"status": "OK", "output_payload": agent_output}),
        config=_CFG,
    )

    got = captured["resp"]
    assert _json.loads(got.output_payload) == _json.loads(expected.output_payload)
    assert got.input_tokens == expected.input_tokens == 812
    assert got.output_tokens == expected.output_tokens == 143
    assert got.tokens_consumed_total == expected.tokens_consumed_total == 955
    assert got.final_state == expected.final_state
