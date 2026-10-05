"""Cloud Model Armor adapter (canonical runtime-guardrail primitive per ADR-152).

This package supersedes the legacy HTTP client for the deprecated
`chora-guardrail` Go service. The LLM-issuing path (LangGraph
``guardrail_screen`` node, post-ADR-146) wraps every Gemini call with:

- pre-call:  :meth:`Screener.sanitize_user_prompt`
- post-call: :meth:`Screener.sanitize_model_response`

Both are awaited inline in the LLM request span (no fan-out, no Pub/Sub
ack-after-processing — Cloud Model Armor is a sync primitive). On any
``MATCH_FOUND`` verdict the orchestrator emits
``chora.governance.policy.violation_detected.v1`` via its own outbox.

See: ``docs/architecture/adrs/adr-152-chora-guardrail-superseded-by-cloud-model-armor.md``
and the Go counterpart at ``libs/chora-go-common/modelarmor/`` (parallel
parity — keep dataclasses + Verdict aligned).
"""

from chora_ai_kernel_orchestrator.adapter.modelarmor.local_screener import (
    LocalScreener,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor.port import (
    GuardrailScreenInput,
    ModelArmorGuardrailPort,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor.screener import (
    FilterHit,
    Screener,
    ScreenRequest,
    ScreenResult,
    Verdict,
    new_screener,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor.stub import StubScreener

__all__ = [
    "FilterHit",
    "GuardrailScreenInput",
    "LocalScreener",
    "ModelArmorGuardrailPort",
    "ScreenRequest",
    "ScreenResult",
    "Screener",
    "StubScreener",
    "Verdict",
    "new_screener",
]
