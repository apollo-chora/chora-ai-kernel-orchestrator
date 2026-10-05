"""RED→GREEN tests for WeaknessEvidenceEmitter (ADR-205 WS-3, CHO-1955).

The graduated Growth-Edge graph's terminal ``emit_evidence`` node depends on the
``EvidenceEmitter`` port (``orchestrators/weakness_analyser_crew.py``). This
adapter is the real implementation: it maps the narrow port shape (the frozen
``weakness.analyzed.v1`` body + the two Model Armor screen verdicts + the
diagnoser model) onto the canonical **D1 accountability** lane
``chora.observability.agent_decision.logged.v1``, attributed to the eval-gated
``weakness_diagnoser`` agent — the per-agent registry id the O+ /o/agents tile
keys on.

Purely ADDITIVE (the single-shot LIVE path emits neither): D2 transparency is
already emitted by the ``publish_analyzed`` node (→ ``weakness.analyzed.v1`` →
the chora-governance ``WeaknessAnalyzedConsumer`` decision_explanation
projector); D4 fairness is the offline crew-scoped ``BiasMetric`` + the HITL
gate; per-token COST is the model gateway's (ADR-163 sole producer of
``observability.token_usage.recorded.v1``). So this node's UNIQUE runtime
evidence is the D1 per-agent decision row.

No live DB, no Pub/Sub — a recorder fake stands in for the
``AgentDecisionLogOutboxWriter`` (that writer's binary-proto encode + outbox
INSERT are covered by ``test_agent_decision_outbox_writer.py``).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_evidence_emitter import (
    AGID_WEAKNESS_DIAGNOSER,
    CREW_NAME_WEAKNESS,
    WeaknessEvidenceEmitter,
)


@dataclasses.dataclass
class _RecordingAgentDecisionEmitter:
    """Duck-typed stand-in for AgentDecisionLogOutboxWriter — records emit()."""

    calls: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    async def emit(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "outbox-row-1"


def _body(**over: Any) -> dict[str, Any]:
    """The frozen weakness.analyzed.v1 body shape synthesize_edges_node builds."""
    b: dict[str, Any] = {
        "upload_id": "up-123",
        "tenant_id": "11111111-1111-1111-1111-111111111111",
        "learner_gcid": "00000000-0000-0000-0000-000000001999",
        "model_used": "gemini-2.5-pro",
        "input_token_count": 1200,
        "output_token_count": 340,
        "analyzed_at": "2026-07-18T09:00:00+00:00",
        "edges": [{"concept_key": "algebra:factoring", "concept_label": "Factoring"}],
        "output_selection": {},
    }
    b.update(over)
    return b


@pytest.mark.asyncio
async def test_emits_one_d1_agent_decision_row_for_the_diagnoser() -> None:
    rec = _RecordingAgentDecisionEmitter()
    emitter = WeaknessEvidenceEmitter(agent_decision_emitter=rec)

    await emitter.emit(
        body=_body(),
        input_decision="ALLOW",
        output_decision="ALLOW",
        model_used="gemini-2.5-pro",
        traceparent="00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        tracestate="chora=1",
    )

    assert len(rec.calls) == 1
    c = rec.calls[0]
    # D1 accountability, attributed to the eval-gated diagnoser agent.
    assert AGID_WEAKNESS_DIAGNOSER == "weakness_diagnoser"
    assert c["agid"] == AGID_WEAKNESS_DIAGNOSER
    assert c["chora_imda_dimension"] == "accountability"
    assert c["crew_name"] == CREW_NAME_WEAKNESS
    # identity + terminal-run keys come off the analyzed body.
    assert c["assist_id"] == "up-123"
    assert c["crew_id"] == "up-123"
    assert c["tenant_id"] == "11111111-1111-1111-1111-111111111111"
    assert c["gcid"] == "00000000-0000-0000-0000-000000001999"
    assert c["occurred_at"] == "2026-07-18T09:00:00+00:00"
    # token counts ride the D1 row (informational; NOT a token_usage ledger event).
    assert c["prompt_tokens"] == 1200
    assert c["completion_tokens"] == 340
    # trace correlation for the O+ Decision-Traces deep-link.
    assert c["traceparent"] == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    assert c["tracestate"] == "chora=1"
    # a clean ALLOW/ALLOW screens → guardrail pass.
    assert c["guardrail_outcome"] == "pass"


@pytest.mark.asyncio
async def test_guardrail_outcome_is_block_when_either_screen_blocked() -> None:
    rec = _RecordingAgentDecisionEmitter()
    emitter = WeaknessEvidenceEmitter(agent_decision_emitter=rec)

    await emitter.emit(body=_body(), input_decision="ALLOW", output_decision="BLOCK", model_used="m")
    assert rec.calls[0]["guardrail_outcome"] == "block"


@pytest.mark.asyncio
async def test_model_used_rides_as_a_namespaced_prompt_condition_attribute() -> None:
    # The D1 agent_decision proto has no model field; the diagnoser's model
    # rides the field-21 attributes map as prompt_conditions.model_used so O+
    # can attribute the decision without decoding the D2 event.
    rec = _RecordingAgentDecisionEmitter()
    emitter = WeaknessEvidenceEmitter(agent_decision_emitter=rec)

    await emitter.emit(
        body=_body(),
        input_decision="ALLOW",
        output_decision="ALLOW",
        model_used="gemini-2.5-pro",
    )
    assert rec.calls[0]["prompt_conditions"]["model_used"] == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_skips_and_never_raises_when_body_missing_identity() -> None:
    # A half-attributed accountability row is worse than none — fail SOFT (the
    # node swallows evidence errors anyway) but never land an unattributed row.
    rec = _RecordingAgentDecisionEmitter()
    emitter = WeaknessEvidenceEmitter(agent_decision_emitter=rec)

    await emitter.emit(
        body=_body(tenant_id=""),
        input_decision="ALLOW",
        output_decision="ALLOW",
        model_used="m",
    )
    assert rec.calls == []


@pytest.mark.asyncio
async def test_missing_analyzed_at_falls_back_to_a_parseable_iso_now() -> None:
    rec = _RecordingAgentDecisionEmitter()
    emitter = WeaknessEvidenceEmitter(agent_decision_emitter=rec)

    await emitter.emit(
        body=_body(analyzed_at=""),
        input_decision="ALLOW",
        output_decision="ALLOW",
        model_used="m",
    )
    # must be a valid ISO timestamp, never blank (BigQuery partition key).
    _dt.datetime.fromisoformat(rec.calls[0]["occurred_at"])
