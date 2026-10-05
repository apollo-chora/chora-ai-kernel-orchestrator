"""Per-agent marker spans for O+ /o/agents + Decision-Traces Cloud Trace links.

The orchestrator emits one ``AgentDecisionLog`` event per agent in a crew run
(e.g. qgen_question + qgen_critic, oe_evaluator + oe_moderator). Historically
every one of those events was stamped with the SAME inbound ``traceparent`` —
so all agents in a run shared one trace id AND one span id, and the O+
'View in Cloud Trace' deep-link landed on the shared crew trace instead of the
individual agent. Cloud Trace also had no ``chora.agent_id`` attribute on any
span.

``agent_span_traceparent`` fixes this: it starts + ends a per-agent marker span
``agent.<agid>`` as a CHILD of the run trace (continued from the inbound
traceparent), tags it ``chora.agent_id`` (plus any caller-supplied attributes),
and returns ITS W3C traceparent. Each agent's AgentDecisionLog then carries a
DISTINCT span id within the shared run trace, so the deep-link resolves to the
agent's OWN span.

§9 (2026-06-09) — the marker span ALSO carries the DECISION EVIDENCE: the
verdict (``chora.decision``) + a bounded reasoning summary
(``chora.reasoning_summary``) + an ``agent.decision`` span event. An auditor who
deep-links from O+ Decision-Traces → Cloud Trace then lands on the agent's span
AND reads what it decided + why (IMDA D2 transparency / explainability). PII
discipline (Tier-4): ONLY the agent's own verdict + a length-bounded reasoning
summary ride the span — NEVER the raw candidate question / learner answer. The
full reasoning lives in ``chora_observability.agent_decision_log`` (critic_notes);
the span carries a summary, consistent with agent_decision.proto's "summary safe
to retain; full content in OTLP with PII redaction" intent.

Best-effort by contract: it returns the parent traceparent unchanged when the
parent is empty, when no real OTel SDK is wired (the no-op tracer mints an
invalid span context that injects to nothing), or on ANY exception. It NEVER
raises — an observability nicety must never break the agent-decision emit per
[[feedback-d6-resilience-first-class]].
"""

from __future__ import annotations

# Upper bound on the reasoning summary stamped onto a span (attribute + event).
# Bounds span size and limits incidental content exposure; the full reasoning is
# retained canonically in chora_observability.agent_decision_log. An over-long
# summary is truncated to this many chars plus a single ellipsis marker.
_MAX_REASONING_SUMMARY_CHARS = 1024


def _bounded_reasoning(summary: str) -> str:
    """Return ``summary`` bounded to ``_MAX_REASONING_SUMMARY_CHARS`` chars,
    appending a single ellipsis when truncated."""
    if len(summary) > _MAX_REASONING_SUMMARY_CHARS:
        return summary[:_MAX_REASONING_SUMMARY_CHARS] + "…"
    return summary


def agent_span_traceparent(
    agid: str,
    parent_traceparent: str,
    *,
    attributes: dict[str, str] | None = None,
    decision: str = "",
    reasoning_summary: str = "",
) -> str:
    """Start+end a per-agent marker span ``agent.<agid>`` as a CHILD of the run
    trace (continued from ``parent_traceparent``), tagged ``chora.agent_id`` (+
    any extra non-empty attributes), and return ITS W3C traceparent.

    When ``decision`` / ``reasoning_summary`` are supplied they ride the span as
    DECISION EVIDENCE: ``chora.decision`` + ``chora.reasoning_summary``
    attributes (the summary length-bounded) plus an ``agent.decision`` span
    event carrying the same. This is what an auditor reads on the per-agent span
    in Cloud Trace (§9, IMDA D2). Pass ONLY the agent's verdict + reasoning —
    never raw learner/candidate content (PII discipline).

    Gives each agent's AgentDecisionLog a DISTINCT span id within the shared run
    trace so O+ deep-links to the agent's OWN span. Best-effort: returns
    ``parent_traceparent`` unchanged when it is empty, when OTel is unavailable,
    or on ANY exception (never raises).
    """
    if not parent_traceparent:
        return parent_traceparent

    try:
        from opentelemetry import propagate, trace

        parent_ctx = propagate.extract({"traceparent": parent_traceparent})
        span = trace.get_tracer("chora_kernel.agent_decision").start_span(f"agent.{agid}", context=parent_ctx)
        try:
            span.set_attribute("chora.agent_id", agid)
            for key, value in (attributes or {}).items():
                if value:
                    span.set_attribute(key, value)

            # Decision evidence — verdict + bounded reasoning (§9). Stamp as
            # attributes (queryable + visible on the span) AND as one
            # `agent.decision` span event (the timeline annotation the FE
            # wave-4 row-click can anchor on). Empty values are skipped on both.
            event_attrs: dict[str, str] = {}
            if decision:
                span.set_attribute("chora.decision", decision)
                event_attrs["chora.decision"] = decision
            if reasoning_summary:
                bounded = _bounded_reasoning(reasoning_summary)
                span.set_attribute("chora.reasoning_summary", bounded)
                event_attrs["chora.reasoning_summary"] = bounded
            if event_attrs:
                span.add_event("agent.decision", attributes=event_attrs)
        finally:
            span.end()

        carrier: dict[str, str] = {}
        propagate.inject(carrier, context=trace.set_span_in_context(span))
        return carrier.get("traceparent") or parent_traceparent
    except Exception:
        # fail-loud-exempt: an observability marker must never break or slow
        # the path it observes, and the degraded value (the parent traceparent,
        # returned unchanged) is the correct answer. Logging here would put the
        # tracing layer's own failures on the hot path of every span it wraps.
        return parent_traceparent
