"""OTLP tracing bootstrap for the AI Kernel orchestrator.

Per CLAUDE.md §6 + ADR-145 Outcome A: every service emits traces from day-1.
The canonical exporter is the generic OTLP-gRPC exporter, pointed at the
local OpenTelemetry Collector (the compose stack runs
``otel/opentelemetry-collector-contrib``); ``OTEL_EXPORTER_OTLP_ENDPOINT``
selects the collector. The Google Cloud Trace exporter is removed — OTLP to
the local collector is the cloud-neutral path.

D6.4 evidence (M12.1.A): spans emitted from the orchestrator HTTP handlers
carry ``chora.tenant_id``, ``chora.gcid``, ``chora.thread_id``,
``chora.run_id``, ``chora.crew_name``, ``chora.pattern``,
``chora.is_resume``, ``chora.governance.status`` for multi-tenant chaos
triage per the ``agentic-resilience-d6`` skill Pillar 4 contract.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_initialized = False


def init_tracing(
    service_name: str = "chora-ai-kernel-orchestrator",
    project_id: str | None = None,
) -> bool:
    """Initialise OTLP tracing. Returns True if a real exporter was wired;
    False if no-op (env not configured / packages missing).

    Selection logic:

    1. If ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set → generic OTLP-gRPC exporter
       (the local collector). This is the only exporter; there is no cloud-trace path.
    2. Else → no-op.

    Idempotent — calling more than once is safe; only the first call wires
    the provider.
    """
    global _initialized
    if _initialized:
        return True

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("opentelemetry-sdk not installed; tracing disabled")
        return False

    if not otlp_endpoint:
        logger.info("tracing disabled — OTEL_EXPORTER_OTLP_ENDPOINT unset")
        return False

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter()
        logger.info("tracing exporter: OTLP-gRPC", extra={"endpoint": otlp_endpoint})
    except ImportError:
        logger.warning("otlp exporter unavailable; tracing disabled")
        return False

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "chora.ai_kernel",
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _initialized = True
    return True


def get_tracer(name: str = "chora-ai-kernel-orchestrator"):
    """Return an OTel tracer. Safe to call even when init_tracing was a
    no-op — returns the default no-op tracer in that case."""
    try:
        from opentelemetry import trace as _trace

        return _trace.get_tracer(name)
    except ImportError:  # pragma: no cover

        class _NoOpSpan:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def set_attribute(self, *args, **kwargs):
                pass

            def set_attributes(self, *args, **kwargs):
                pass

        class _NoOpTracer:
            def start_as_current_span(self, *args, **kwargs):
                return _NoOpSpan()

        return _NoOpTracer()
