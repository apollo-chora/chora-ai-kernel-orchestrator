"""Model Armor screener adapter for the Growth-Edge crew (ADR-205 WS-3 / D2).

Adapts the platform Cloud Model Armor port (``adapter.modelarmor.Screener``,
ADR-152) to the crew graph's ``Screener`` port:

  * direction ``"input"``  → ``sanitize_user_prompt`` over the extracted artifact
    text + the fenced learner-clue DATA block (the injection surface);
  * direction ``"output"`` → ``sanitize_model_response`` over the diagnosis JSON.

The graph node fails CLOSED on a BLOCK verdict (refund) and fails LOUD on a
transport error (this adapter re-raises — never a silent ALLOW). The template is
the full template name resolved at the composition root from the
``weakness_analyzer: strict`` row of ``agent-guardrail-mapping.yaml`` (WS-3).
"""

from __future__ import annotations

from chora_ai_kernel_orchestrator.adapter.modelarmor.screener import (
    Screener as ArmorScreener,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor.screener import (
    ScreenRequest,
)
from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew import (
    AGENT_ID,
    ScreenVerdict,
)


class ModelArmorScreenerAdapter:
    """Crew ``Screener`` port backed by Cloud Model Armor (one strict template)."""

    def __init__(self, *, screener: ArmorScreener, template_name: str, agent_id: str = AGENT_ID) -> None:
        if not (template_name or "").strip():
            raise ValueError("ModelArmorScreenerAdapter: template_name required (no inline config)")
        self._screener = screener
        self._template = template_name
        self._agent_id = agent_id

    async def screen(self, *, text: str, tenant_id: str, gcid: str, direction: str) -> ScreenVerdict:
        req = ScreenRequest(
            tenant_id=tenant_id,
            agent_id=self._agent_id,
            gcid=gcid,
            template_name=self._template,
            text=text,
        )
        # Fail-loud: a transport error propagates so the graph node refunds —
        # the gate must never silently degrade to ALLOW.
        if direction == "output":
            result = await self._screener.sanitize_model_response(req)
        else:
            result = await self._screener.sanitize_user_prompt(req)
        return ScreenVerdict(decision=str(result.verdict.value), explanation=result.reason)


__all__ = ["ModelArmorScreenerAdapter"]
