"""Port + dataclasses + Verdict enum for the Cloud Model Armor adapter.

Mirror of the Go canonical shape at
``libs/chora-go-common/modelarmor/`` (authored in parallel). Keep the
field set in sync — both adapters serialize into the same audit + trace
attribute schema downstream.

ADR-152 — chora-guardrail superseded by Cloud Model Armor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Verdict(str, Enum):  # noqa: UP042 — str+Enum (not StrEnum) keeps str(Verdict.X) == "Verdict.X"; consumed via .value + membership today, but avoids a subtle behavior change
    """Canonical 3-state verdict, mirrored 1:1 with the Go ``Verdict`` enum.

    Maps semantically to the legacy chora-guardrail decision space:

    - ``ALLOW``        ↔ legacy ``allow``
    - ``BLOCK``        ↔ legacy ``refuse``
    - ``INSPECT_ONLY`` ↔ NEW (Cloud Model Armor template ``INSPECT_ONLY``
                              enforcement mode — log + emit event but do not
                              gate the LLM call; orchestrator may continue)

    Legacy ``rewrite`` is no longer emitted by the runtime guardrail. Rewrite
    is a content-author responsibility post-ADR-152; the kernel does not
    synthesise rewritten content in the guardrail hop.
    """

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    INSPECT_ONLY = "INSPECT_ONLY"


@dataclass(frozen=True)
class ScreenRequest:
    """Input to a Model Armor screening call.

    ``template_name`` is the **full** template name:
    ``projects/{project}/locations/{location}/templates/{template_id}``.
    The caller (LangGraph node) resolves it from the agent metadata —
    typically via ``AgentCard.guardrail_template_name``.
    """

    tenant_id: str
    agent_id: str
    gcid: str
    template_name: str
    text: str


@dataclass(frozen=True)
class FilterHit:
    """One row of the per-filter screening verdict, flattened from the
    proto ``sanitization_result.filter_results`` map.

    ``filter_name`` is the proto map key (``"rai"`` | ``"pi_and_jailbreak"``
    | ``"sdp"`` | ``"malicious_uri"`` | ``"csam"`` | ``"virus_scan"``).
    """

    filter_name: str
    match_state: str
    severity: str | None = None
    subcategory: str | None = None


@dataclass(frozen=True)
class ScreenResult:
    """Mapped Cloud Model Armor response.

    ``raw_response`` is the proto-plus ``to_dict()`` of the underlying
    ``SanitizationResult`` — preserved for audit + the OTel span
    ``model_armor.raw`` attribute so downstream investigators can inspect
    the full SDK shape without re-deriving from filters.
    """

    verdict: Verdict
    reason: str
    filters: list[FilterHit] = field(default_factory=list)
    latency_ms: int = 0
    raw_response: dict[str, Any] = field(default_factory=dict)


class Screener(Protocol):
    """Port — implemented by :class:`.local_screener.LocalScreener` (the
    cloud-neutral local guardrail) and :class:`.stub.StubScreener` (tests)."""

    async def sanitize_user_prompt(self, req: ScreenRequest) -> ScreenResult: ...

    async def sanitize_model_response(self, req: ScreenRequest) -> ScreenResult: ...

    async def close(self) -> None: ...


async def new_screener(project: str | None = None, location: str | None = None) -> Screener:
    """Factory — returns the cloud-neutral :class:`.local_screener.LocalScreener`.

    ``project`` + ``location`` are accepted (and ignored) for call-site
    compatibility with the retired Cloud Model Armor factory; the local
    screener needs no remote resource. The blocklist is configurable via
    ``CHORA_GUARDRAIL_BLOCKLIST``.
    """
    # Lazy import — breaks the local_screener ↔ screener import cycle.
    from chora_ai_kernel_orchestrator.adapter.modelarmor.local_screener import (
        LocalScreener,
    )

    return LocalScreener.from_env()
