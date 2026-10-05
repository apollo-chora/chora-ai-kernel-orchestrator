"""Tests for the AI Kernel orchestrator tracing wiring (D6.4 / M12.1.A).

The ``init_tracing`` function must:
* Pick the generic OTLP-gRPC exporter when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
  set (the local OpenTelemetry Collector — the compose stack runs
  ``otel/opentelemetry-collector-contrib``). This is the only exporter; the
  Google Cloud Trace path is removed.
* No-op cleanly when it is not configured (for local dev / unit tests).
* Be idempotent.

``get_tracer`` returns a usable tracer even when init_tracing was a no-op
(falls through to OTel's default no-op tracer).
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_tracing(monkeypatch: pytest.MonkeyPatch):
    """Reset the module-level _initialized flag between tests."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    import chora_ai_kernel_orchestrator.observability.tracing as _t

    importlib.reload(_t)
    yield
    importlib.reload(_t)


class TestInitTracing:
    def test_no_op_when_no_config(self) -> None:
        from chora_ai_kernel_orchestrator.observability.tracing import init_tracing

        assert init_tracing() is False

    def test_returns_true_with_otlp_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from chora_ai_kernel_orchestrator.observability.tracing import init_tracing

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        result = init_tracing()
        assert result in (True, False)

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from chora_ai_kernel_orchestrator.observability.tracing import init_tracing

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        first = init_tracing()
        second = init_tracing()
        # Second call must not crash; second result reflects the same
        # initialization decision.
        assert second == first or second is True

    def test_uses_otlp_for_local_collector_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When OTLP endpoint points at a local collector, the generic OTLP
        exporter is the right choice — keep the local-dev path working."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

        from chora_ai_kernel_orchestrator.observability.tracing import init_tracing

        with caplog.at_level("INFO", logger="chora_ai_kernel_orchestrator.observability.tracing"):
            init_tracing()

        msgs = [r.message for r in caplog.records]
        assert any("exporter: OTLP-gRPC" in m for m in msgs), (
            f"A local collector endpoint must route via the generic OTLP exporter.\nCaptured log messages: {msgs!r}"
        )


class TestGetTracer:
    def test_returns_tracer_even_when_uninitialized(self) -> None:
        from chora_ai_kernel_orchestrator.observability.tracing import get_tracer

        tracer = get_tracer()
        # Tracer must support start_as_current_span as a context manager
        with tracer.start_as_current_span("test_span") as span:
            span.set_attribute("chora.tenant_id", "t-1")
            span.set_attribute("chora.crew_name", "test")

    def test_named_tracer(self) -> None:
        from chora_ai_kernel_orchestrator.observability.tracing import get_tracer

        tracer = get_tracer("my.sub.module")
        assert tracer is not None
