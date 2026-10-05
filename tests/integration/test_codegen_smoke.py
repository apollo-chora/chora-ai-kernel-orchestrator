"""Smoke test for chora_contracts_gen import + AgentExecutorStub construction.

Skipped in environments where chora_contracts_gen isn't installed (e.g., a
trimmed-down dev shell). When the orchestrator's wheel is installed via
pip, the generated proto module SHOULD import cleanly + the stub SHOULD
construct over a non-running channel without raising. Live RPCs are out of
scope for unit tests; that's the M14 e2e harness's job.
"""

from __future__ import annotations

import pytest


def _has_contracts_gen() -> bool:
    try:
        import chora_contracts_gen.services  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _has_contracts_gen(), reason="chora_contracts_gen not installed in this env")


def test_agent_executor_pb2_messages_constructible() -> None:
    from chora_contracts_gen.services import agent_executor_pb2

    req = agent_executor_pb2.ExecuteAgentRequest(
        execution_id="01HXX",
        tenant_id="t1",
        agid="a1",
        agent_role="validator",
        prompt_template_id="prompt-1",
        input_payload="{}",
    )
    assert req.execution_id == "01HXX"
    assert req.agent_role == "validator"

    cm = agent_executor_pb2.ContextMessage(role="user", content="hi")
    assert cm.role == "user"


def test_model_broker_router_pb2_messages_constructible() -> None:
    from chora_contracts_gen.services import model_broker_router_pb2

    # The proto type must exist + accept the expected fields.
    cls = model_broker_router_pb2.RouteRequest
    msg = cls(
        tenant_id="t1",
        agid="a1",
        agent_role="creation-validator",
        domain="creation",
        task_kind="generate",
        prompt_size_tokens=128,
        latency_budget_ms=5000,
        cost_budget_micros=10000,
    )
    assert msg.tenant_id == "t1"


def test_model_broker_gateway_pb2_messages_constructible() -> None:
    from chora_contracts_gen.services import model_broker_gateway_pb2

    msg = model_broker_gateway_pb2.InvokeRequest(
        invocation_id="01HXX",
        tenant_id="t1",
        agid="a1",
        model_id="gemini-2.5-flash",
    )
    assert msg.model_id == "gemini-2.5-flash"


def test_agent_executor_stub_constructs_over_unconnected_channel() -> None:
    """gRPC stub must instantiate without making network calls (lazy)."""
    import grpc
    from chora_contracts_gen.services import agent_executor_pb2_grpc

    channel = grpc.insecure_channel("nonexistent.invalid:9090")
    try:
        stub = agent_executor_pb2_grpc.AgentExecutorStub(channel)
        assert stub is not None
        # Don't actually call ExecuteAgent — that would attempt the connection.
    finally:
        channel.close()
