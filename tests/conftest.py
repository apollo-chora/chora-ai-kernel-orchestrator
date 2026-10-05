"""Shared pytest scaffolding for the chora-ai-kernel-orchestrator suite.

Test-isolation guards live here so no individual test can leak global
process state into a later test under full-suite ordering.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _reset_global_tracer_provider() -> Iterator[None]:
    """Snapshot + restore OpenTelemetry's *global* TracerProvider per test.

    Why this exists: ``opentelemetry.trace.set_tracer_provider()`` is guarded
    by a module-level ``Once`` (``_TRACER_PROVIDER_SET_ONCE``). The FIRST test
    in a run that sets a real provider wins that latch forever — every later
    ``set_tracer_provider(...)`` call is silently dropped with
    ``WARNING ... Overriding of current TracerProvider is not allowed``.

    Concretely, ``test_tracing.py::init_tracing`` installs a global provider,
    which then made ``test_qgen_crew_runner.py::
    test_handle_started_continues_inbound_traceparent`` fail ONLY under full-suite
    ordering: its own ``InMemorySpanExporter``-backed provider was never
    actually installed, so ``get_finished_spans()`` came back empty. The test
    PASSED in isolation, proving the failure was a leak, not a real bug.

    This autouse fixture saves the current global provider object AND the
    ``Once`` latch before each test, then restores both afterwards (installing
    a fresh ``Once`` so the next test starts from a clean, settable state).
    No test can leak its provider into another in either direction.
    """
    import opentelemetry.trace as trace_api

    try:
        from opentelemetry.util._once import Once
    except ImportError:  # pragma: no cover — OTel layout guard
        # If the internal Once helper ever moves, fall back to a no-op guard
        # rather than crashing the whole suite.
        yield
        return

    saved_provider = getattr(trace_api, "_TRACER_PROVIDER", None)
    saved_once = getattr(trace_api, "_TRACER_PROVIDER_SET_ONCE", None)

    # Give the test a clean latch so its own set_tracer_provider() takes effect.
    trace_api._TRACER_PROVIDER_SET_ONCE = Once()

    try:
        yield
    finally:
        trace_api._TRACER_PROVIDER = saved_provider
        if saved_once is not None:
            trace_api._TRACER_PROVIDER_SET_ONCE = saved_once
        else:  # pragma: no cover — defensive: restore a fresh latch
            trace_api._TRACER_PROVIDER_SET_ONCE = Once()
