"""Targeted tests to lift adapter coverage past the 85% gate."""

from __future__ import annotations

import os

import pytest

from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    GuardrailScreenInput,
    ModelArmorGuardrailPort,
    StubScreener,
    Verdict,
)

# -- Guardrail (ADR-152 — Cloud Model Armor) ---------------------------------


def test_guardrail_from_env_returns_none_when_yaml_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Missing mapping YAML → from_env returns None (readiness flagged)."""
    monkeypatch.setenv(
        "CHORA_AGENT_GUARDRAIL_MAPPING_PATH",
        str(tmp_path / "nonexistent" / "agent-guardrail-mapping.yaml"),  # type: ignore[operator]
    )
    assert ModelArmorGuardrailPort.from_env() is None


def test_guardrail_from_env_builds_port_with_default_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default monorepo-relative mapping path resolves and builds the port."""
    # Ensure no override path is set so we hit the bundled YAML.
    monkeypatch.delenv("CHORA_AGENT_GUARDRAIL_MAPPING_PATH", raising=False)
    port = ModelArmorGuardrailPort.from_env(screener=StubScreener())
    assert port is not None
    # default tier is strict per ADR-152
    assert port.resolver.default_tier == "strict"


@pytest.mark.asyncio
async def test_guardrail_screen_input_direction_routes_to_user_prompt() -> None:
    """direction='input' invokes sanitize_user_prompt."""
    stub = StubScreener(force_verdict=Verdict.ALLOW)
    port = ModelArmorGuardrailPort.from_components(
        screener=stub,
        project="chora-test",
        location="us-central1",
        environment="dev",
    )
    out = await port.screen(
        GuardrailScreenInput(
            tenant_id="t",
            gcid="g",
            agent_id="a",
            content="x",
            direction="input",
        )
    )
    assert out.verdict == Verdict.ALLOW
    assert stub.calls[0][0] == "sanitize_user_prompt"


@pytest.mark.asyncio
async def test_guardrail_screen_output_direction_routes_to_model_response() -> None:
    """direction='output' invokes sanitize_model_response."""
    stub = StubScreener(force_verdict=Verdict.BLOCK)
    port = ModelArmorGuardrailPort.from_components(
        screener=stub,
        project="chora-test",
        location="us-central1",
        environment="dev",
    )
    out = await port.screen(
        GuardrailScreenInput(
            tenant_id="t",
            gcid="g",
            agent_id="a",
            content="x",
            direction="output",
        )
    )
    assert out.verdict == Verdict.BLOCK
    assert stub.calls[0][0] == "sanitize_model_response"


@pytest.mark.asyncio
async def test_guardrail_screen_unknown_direction_raises() -> None:
    """Programmer error — unknown direction is loud."""
    port = ModelArmorGuardrailPort.from_components(
        screener=StubScreener(),
        project="chora-test",
        location="us-central1",
        environment="dev",
    )
    with pytest.raises(ValueError, match="unknown direction"):
        await port.screen(
            GuardrailScreenInput(
                tenant_id="t",
                gcid="g",
                agent_id="a",
                content="x",
                direction="sideways",
            )
        )


@pytest.mark.asyncio
async def test_guardrail_close_propagates_to_screener() -> None:
    stub = StubScreener()
    port = ModelArmorGuardrailPort.from_components(
        screener=stub,
        project="chora-test",
        location="us-central1",
        environment="dev",
    )
    await port.close()
    assert stub.closed is True


# -- Tracing init ------------------------------------------------------------


def test_tracing_init_no_op_when_endpoint_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from chora_ai_kernel_orchestrator.observability.tracing import init_tracing

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    init_tracing()  # must not raise

    # And a blank value also no-ops
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    init_tracing()
    assert os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") == ""
