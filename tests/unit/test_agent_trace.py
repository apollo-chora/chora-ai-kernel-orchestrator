"""Unit tests for observability.agent_trace.agent_span_traceparent — RED→GREEN
per [[feedback-strict-tdd]].

Contract: the helper starts + ends a per-agent marker span ``agent.<agid>`` as a
CHILD of the inbound run trace (continued from ``parent_traceparent``), tags it
``chora.agent_id`` (plus any extra non-empty attributes), and returns ITS W3C
traceparent. This gives each agent's AgentDecisionLog a DISTINCT span id within
the shared run trace, so the O+ /o/agents 'View in Cloud Trace' deep-link lands
on the agent's OWN span rather than the shared crew trace.

Best-effort: returns the parent traceparent unchanged when it is empty, when no
real OTel SDK is installed (no-op tracer), or on ANY exception (never raises).

A real OTel SDK is installed per test so ``start_span`` mints real span ids.
``tests/conftest.py::_reset_global_tracer_provider`` snapshots + restores the
global provider per test, so installing a provider here cannot leak into other
tests (and the no-op-default tests below get a genuine no-op tracer).
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.observability.agent_trace import (
    _MAX_REASONING_SUMMARY_CHARS,
    agent_span_traceparent,
)

# Canonical W3C parent traceparent — trace_id = 0af7651916cd43dd8448eb211c80319c,
# span_id = b7ad6b7169203331.
PARENT_TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
PARENT_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
PARENT_SPAN_ID = "b7ad6b7169203331"


def _install_real_tracer() -> object:
    """Install a real SDK TracerProvider backed by an InMemorySpanExporter and
    return the exporter. Relies on conftest's ``_reset_global_tracer_provider``
    autouse fixture for per-test isolation (a fresh ``Once`` latch is installed
    before each test, so this set_tracer_provider call takes effect, and the
    original provider is restored afterwards)."""
    pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace_api.set_tracer_provider(provider)
    return exporter


def _trace_id(tp: str) -> str:
    return tp.split("-")[1]


def _span_id(tp: str) -> str:
    return tp.split("-")[2]


def test_returns_child_span_same_trace_new_span() -> None:
    """A valid parent → returned traceparent keeps the SAME trace id (continued
    run trace) but carries a DIFFERENT span id (the agent's own marker span)."""
    _install_real_tracer()

    out = agent_span_traceparent("qgen_question", PARENT_TP)

    assert out != PARENT_TP
    assert _trace_id(out) == PARENT_TRACE_ID  # child of the run trace
    assert _span_id(out) != PARENT_SPAN_ID  # distinct span id
    assert len(_trace_id(out)) == 32
    assert len(_span_id(out)) == 16


def test_distinct_agids_yield_distinct_span_ids() -> None:
    """Two calls with different agids → DIFFERENT span ids, both still children
    of the SAME run trace (so qgen_question + qgen_critic deep-link separately)."""
    _install_real_tracer()

    a = agent_span_traceparent("qgen_question", PARENT_TP)
    b = agent_span_traceparent("qgen_critic", PARENT_TP)

    assert _span_id(a) != _span_id(b)
    assert _trace_id(a) == _trace_id(b) == PARENT_TRACE_ID


def test_marker_span_tagged_with_agent_id_and_extra_attributes() -> None:
    """The exported marker span is named ``agent.<agid>`` + tagged
    chora.agent_id==agid + any extra NON-empty attributes; empty values skipped;
    span continues the inbound run trace."""
    exporter = _install_real_tracer()

    agent_span_traceparent(
        "qgen_critic",
        PARENT_TP,
        attributes={"chora.question_type": "mcq", "blank": ""},
    )

    spans = exporter.get_finished_spans()  # type: ignore[attr-defined]
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "agent.qgen_critic"
    assert span.attributes["chora.agent_id"] == "qgen_critic"
    assert span.attributes["chora.question_type"] == "mcq"
    assert "blank" not in span.attributes  # empty attribute value skipped
    # Continues the inbound run trace + parented on the inbound span.
    assert format(span.context.trace_id, "032x") == PARENT_TRACE_ID
    assert span.parent is not None
    assert format(span.parent.span_id, "016x") == PARENT_SPAN_ID


def test_empty_parent_returns_empty_unchanged() -> None:
    """Empty parent → returned unchanged (no span minted)."""
    _install_real_tracer()
    assert agent_span_traceparent("qgen_question", "") == ""


def test_malformed_parent_does_not_raise() -> None:
    """A malformed parent never raises; returns some traceparent string."""
    _install_real_tracer()
    out = agent_span_traceparent("qgen_question", "not-a-valid-traceparent")
    assert isinstance(out, str)


def test_noop_tracer_returns_parent_unchanged() -> None:
    """With NO real SDK installed for this test, OTel's default no-op tracer
    mints an invalid span context → inject yields nothing → the helper returns
    the parent unchanged. This best-effort fallback is what keeps the existing
    runner tests green when they do not opt into a real SDK."""
    # Deliberately do NOT call _install_real_tracer().
    assert agent_span_traceparent("qgen_question", PARENT_TP) == PARENT_TP


# -----------------------------------------------------------------------------
# §9 Part B — the per-agent marker span carries the DECISION EVIDENCE (the
# verdict + the reasoning the agent used) so an auditor who deep-links from O+
# Decision-Traces → Cloud Trace lands on the agent's span AND reads what it
# decided + why (IMDA D2 transparency / explainability). PII discipline: only
# the agent's own VERDICT + bounded REASONING summary ride the span — never the
# raw candidate question / learner answer.
# -----------------------------------------------------------------------------


def test_decision_evidence_stamped_as_span_attributes() -> None:
    """``decision`` + ``reasoning_summary`` → ``chora.decision`` +
    ``chora.reasoning_summary`` attributes on the agent marker span."""
    exporter = _install_real_tracer()

    agent_span_traceparent(
        "qgen_critic",
        PARENT_TP,
        attributes={"chora.question_type": "mcq"},
        decision="rejected",
        reasoning_summary="Distractor B is implausible; answer key ambiguous.",
    )

    span = exporter.get_finished_spans()[0]  # type: ignore[attr-defined]
    assert span.attributes["chora.decision"] == "rejected"
    assert span.attributes["chora.reasoning_summary"] == "Distractor B is implausible; answer key ambiguous."
    # The pre-existing attributes still ride alongside.
    assert span.attributes["chora.agent_id"] == "qgen_critic"
    assert span.attributes["chora.question_type"] == "mcq"


def test_decision_evidence_emits_decision_span_event() -> None:
    """A non-empty verdict/reasoning also adds an ``agent.decision`` span event
    (the timeline annotation the FE wave-4 row-click can anchor on) carrying the
    same verdict + reasoning."""
    exporter = _install_real_tracer()

    agent_span_traceparent(
        "oe_evaluator",
        PARENT_TP,
        decision="completed_with_warning",
        reasoning_summary="2 of 3 questions flagged for review.",
    )

    span = exporter.get_finished_spans()[0]  # type: ignore[attr-defined]
    events = {e.name: e for e in span.events}
    assert "agent.decision" in events
    ev = events["agent.decision"]
    assert ev.attributes["chora.decision"] == "completed_with_warning"
    assert ev.attributes["chora.reasoning_summary"] == "2 of 3 questions flagged for review."


def test_reasoning_summary_truncated_to_bound() -> None:
    """An over-long reasoning summary is bounded to
    ``_MAX_REASONING_SUMMARY_CHARS`` (+ a single ellipsis marker) so a span
    attribute never carries an unbounded blob (and incidental content is
    limited). The full reasoning lives in chora_observability.agent_decision_log
    (critic_notes) — the span carries only a summary."""
    exporter = _install_real_tracer()
    long_reason = "x" * (_MAX_REASONING_SUMMARY_CHARS + 500)

    agent_span_traceparent(
        "qgen_critic",
        PARENT_TP,
        decision="accepted",
        reasoning_summary=long_reason,
    )

    span = exporter.get_finished_spans()[0]  # type: ignore[attr-defined]
    stamped = span.attributes["chora.reasoning_summary"]
    assert len(stamped) == _MAX_REASONING_SUMMARY_CHARS + 1  # bound + ellipsis
    assert stamped.endswith("…")
    assert stamped[:_MAX_REASONING_SUMMARY_CHARS] == "x" * _MAX_REASONING_SUMMARY_CHARS


def test_no_evidence_no_decision_attribute_or_event() -> None:
    """Absent verdict + reasoning (the default) → NO ``chora.decision`` /
    ``chora.reasoning_summary`` attributes and NO ``agent.decision`` event — the
    pre-§9 marker-span shape is preserved (back-compat)."""
    exporter = _install_real_tracer()

    agent_span_traceparent("qgen_question", PARENT_TP)

    span = exporter.get_finished_spans()[0]  # type: ignore[attr-defined]
    assert "chora.decision" not in span.attributes
    assert "chora.reasoning_summary" not in span.attributes
    assert all(e.name != "agent.decision" for e in span.events)


def test_decision_only_event_omits_empty_reasoning() -> None:
    """A verdict with NO reasoning still records the event + ``chora.decision``,
    but omits the empty ``chora.reasoning_summary`` (empty values are skipped on
    both the attribute and the event)."""
    exporter = _install_real_tracer()

    agent_span_traceparent("qgen_question", PARENT_TP, decision="accepted")

    span = exporter.get_finished_spans()[0]  # type: ignore[attr-defined]
    assert span.attributes["chora.decision"] == "accepted"
    assert "chora.reasoning_summary" not in span.attributes
    ev = next(e for e in span.events if e.name == "agent.decision")
    assert ev.attributes["chora.decision"] == "accepted"
    assert "chora.reasoning_summary" not in ev.attributes
