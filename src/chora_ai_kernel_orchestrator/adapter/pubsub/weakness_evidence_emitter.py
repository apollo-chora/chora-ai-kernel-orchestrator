"""WeaknessEvidenceEmitter — the graph-mode ``EvidenceEmitter`` adapter (WS-3).

Implements the ``EvidenceEmitter`` port declared in
``orchestrators/weakness_analyser_crew.py`` (the terminal ``emit_evidence``
node). It maps the narrow port shape — the frozen ``weakness.analyzed.v1`` body
+ the two Cloud Model Armor screen verdicts + the diagnoser model — onto the
canonical **D1 accountability** lane
``chora.observability.agent_decision.logged.v1`` by delegating to the shared
``AgentDecisionLogOutboxWriter`` (same binary-proto encoder + outbox drain the
qgen / OE grading crews use). One row per graph-mode analysis, attributed to the
eval-gated ``weakness_diagnoser`` agent — the per-agent registry id the O+
/o/agents tile keys on.

Why ONLY D1 here (purely additive — the single-shot LIVE path emits neither
this nor token_usage):

* **D2 transparency** is already emitted one node earlier by ``publish_analyzed``
  (→ ``weakness.analyzed.v1``, tagged ``transparency``) and projected by the
  chora-governance ``WeaknessAnalyzedConsumer`` into ``decision_explanation``
  (it reads ``model_used``). Re-emitting it here would double-count.
* **D4 fairness** is the offline crew-scoped ``BiasMetric``
  (``agent_id=weakness_analyser_crew``) + the in-graph HITL gate — not a runtime
  event.
* **Per-token COST** is the model gateway's job: per ADR-163 the gateway is the
  sole producer of ``observability.token_usage.recorded.v1`` and the diagnoser's
  LLM calls flow through it. The per-agent token COUNTS still ride this D1 row
  (``prompt_tokens`` / ``completion_tokens``) as informational attributes for
  O+, but this adapter does NOT write a billing-grade token_usage ledger event
  (that per-service writer is deprecated post-cutover, per pub-sub-topology).

So the UNIQUE runtime evidence graph mode adds is the D1 per-agent decision row.

Hexagonal: depends on the narrow ``_AgentDecisionEmitter`` Protocol (the
``AgentDecisionLogOutboxWriter.emit`` shape) so the unit tests inject a recorder
without a live DB. Composition root:
``adapter/pubsub/weakness_analyser_crew_wiring.py::_assemble_graph_components``.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_decision_outbox_writer import (
    IMDA_DIM_ACCOUNTABILITY,
)

logger = logging.getLogger(__name__)

# The per-agent registry id (agid) the O+ /o/agents tile keys on for the
# Growth-Edge diagnoser. MUST match the deployed agent name + the eval-gate
# crew descriptor (chora-infra/k8s/agent-eval/crews/weakness.yaml member
# ``weakness-diagnoser``) + the guardrail-mapping key ``weakness_diagnoser``.
# NEVER the crew id ``weakness_analyser_crew`` (that is the gateway cost /
# Model-Armor id, not a per-agent tile — the qgen/OE lesson: a crew id matches
# no registry tile).
AGID_WEAKNESS_DIAGNOSER = "weakness_diagnoser"

# Crew display name for the /o/agents Crews+Agents hierarchy. Mirrors the OE
# convention (``OE_GRADING_CREW_NAME = "oe_grading"`` — drop the ``_crew``).
CREW_NAME_WEAKNESS = "weakness_analyser"

_GUARDRAIL_BLOCK = "block"
_GUARDRAIL_PASS = "pass"


class _AgentDecisionEmitter(Protocol):
    """The subset of ``AgentDecisionLogOutboxWriter.emit`` this adapter drives."""

    async def emit(
        self,
        *,
        assist_id: str,
        agid: str,
        tenant_id: str,
        gcid: str,
        decision: str,
        attempt_count: int,
        max_retries: int,
        critic_notes: str,
        quality_warning: bool,
        chora_imda_dimension: str,
        occurred_at: str,
        traceparent: str = "",
        tracestate: str = "",
        crew_name: str = "",
        crew_id: str = "",
        model_id: str = "",
        guardrail_outcome: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prompt_conditions: dict[str, str] | None = None,
    ) -> str: ...


def _guardrail_outcome(input_decision: str, output_decision: str) -> str:
    """Collapse the two Model Armor screen verdicts (``ALLOW`` / ``BLOCK`` /
    ``INSPECT_ONLY``) into the O+ guardrail surface ∈ {``block``, ``pass``}.

    ``emit_evidence`` only runs on the approved terminal path (a BLOCK routes to
    ``refund`` and never reaches here), so in practice this is ``pass`` — but a
    BLOCK on either direction maps to ``block`` for a truthful record."""
    verdicts = {
        (input_decision or "").strip().upper(),
        (output_decision or "").strip().upper(),
    }
    return _GUARDRAIL_BLOCK if "BLOCK" in verdicts else _GUARDRAIL_PASS


def _coerce_int(v: Any) -> int:
    try:
        return max(int(v), 0)
    except (TypeError, ValueError):
        return 0


class WeaknessEvidenceEmitter:
    """Adapter satisfying the crew ``EvidenceEmitter`` port (ADR-205 WS-3)."""

    def __init__(self, *, agent_decision_emitter: _AgentDecisionEmitter) -> None:
        self._emitter = agent_decision_emitter

    async def emit(
        self,
        *,
        body: dict[str, Any],
        input_decision: str,
        output_decision: str,
        model_used: str,
        traceparent: str = "",
        tracestate: str = "",
    ) -> None:
        """Emit one D1 accountability row for the diagnoser's decision.

        Best-effort: the node swallows exceptions (evidence is non-blocking), but
        we additionally refuse to land a HALF-ATTRIBUTED row — a missing
        upload/tenant/gcid skips loudly rather than writing an unattributable
        accountability record.
        """
        upload_id = str(body.get("upload_id", "")).strip()
        tenant_id = str(body.get("tenant_id", "")).strip()
        gcid = str(body.get("learner_gcid", "")).strip()
        if not (upload_id and tenant_id and gcid):
            logger.warning(
                "weakness_evidence.skipped_incomplete_body",
                extra={
                    "has_upload_id": bool(upload_id),
                    "has_tenant_id": bool(tenant_id),
                    "has_gcid": bool(gcid),
                },
            )
            return

        occurred_at = str(body.get("analyzed_at") or "").strip()
        if not occurred_at:
            occurred_at = _dt.datetime.now(tz=_dt.UTC).isoformat()

        # The diagnoser's model rides proto field 6 (model_id — the concrete
        # model that produced the decision) so chora-observability can price
        # the token counts per model. It ALSO keeps riding the namespaced
        # field-21 attribute so O+ can attribute the decision without decoding
        # the D2 analyzed event.
        model = (model_used or "").strip()
        prompt_conditions = {"model_used": model} if model else {}

        await self._emitter.emit(
            assist_id=upload_id,
            agid=AGID_WEAKNESS_DIAGNOSER,
            tenant_id=tenant_id,
            gcid=gcid,
            # The diagnoser produced a diagnosis (this node only runs post-publish
            # on the approved path). The CONTENT — edge count, "learner did well"
            # anti-mislabel — rides D2; this is the accountability action record.
            decision="analyzed",
            attempt_count=1,
            max_retries=1,
            critic_notes="",
            quality_warning=False,
            chora_imda_dimension=IMDA_DIM_ACCOUNTABILITY,
            occurred_at=occurred_at,
            traceparent=traceparent,
            tracestate=tracestate,
            crew_name=CREW_NAME_WEAKNESS,
            crew_id=upload_id,
            model_id=model,
            guardrail_outcome=_guardrail_outcome(input_decision, output_decision),
            prompt_tokens=_coerce_int(body.get("input_token_count")),
            completion_tokens=_coerce_int(body.get("output_token_count")),
            prompt_conditions=prompt_conditions,
        )
        logger.info(
            "weakness_evidence.d1_emitted",
            extra={
                "upload_id": upload_id,
                "agid": AGID_WEAKNESS_DIAGNOSER,
                "guardrail_outcome": _guardrail_outcome(input_decision, output_decision),
            },
        )


__all__ = [
    "AGID_WEAKNESS_DIAGNOSER",
    "CREW_NAME_WEAKNESS",
    "WeaknessEvidenceEmitter",
]
