"""Local guardrail screener — the cloud-neutral stand-in for Cloud Model Armor.

Cloud Model Armor (Google) has no drop-in local equivalent. This module backs
the same ``Screener`` port with a minimal, configurable, tier-aware local
screener so the guardrail seam and the service's public API are unchanged.

Behaviour (per ADR-152 tier semantics, preserved):

- The screen request's ``template_name`` encodes the tier
  (``chora-guardrail-{strict|balanced|permissive}-{env}``); the local
  screener reads it so an unknown / unparseable template fails CLOSED to the
  strict tier (security wins — Principle Conflict Resolution #1).
- ``strict`` / ``balanced``: a blocklist match → ``BLOCK``.
- ``permissive``: a blocklist match → ``INSPECT_ONLY`` (audit only, do not
  gate the LLM call).
- No match → ``ALLOW``.
- Fail-LOUD: the screener never silently degrades to ALLOW on an error; it
  raises so the LangGraph node can map to a refund.

The blocklist is a small set of explicit-content patterns, overridable via
``CHORA_GUARDRAIL_BLOCKLIST`` (comma-separated substrings, case-insensitive).
It is a minimal substitute for a managed moderation API — deployments that
need production-grade screening should front this with a local moderation
service and re-point the screener.
"""

from __future__ import annotations

import logging
import os

from chora_ai_kernel_orchestrator.adapter.modelarmor.screener import (
    FilterHit,
    ScreenRequest,
    ScreenResult,
    Verdict,
)

logger = logging.getLogger(__name__)

ENV_BLOCKLIST = "CHORA_GUARDRAIL_BLOCKLIST"

# Minimal explicit-content blocklist (case-insensitive substrings). This is a
# deliberately small, auditable default — a real deployment overrides it via
# CHORA_GUARDRAIL_BLOCKLIST. It is NOT a replacement for a managed
# moderation API; it keeps the guardrail seam honest locally.
_DEFAULT_BLOCKLIST = (
    "child sexual abuse",
    "csam",
    "terrorist instruction",
    "bomb making",
    "synthesize explosives",
)

_VALID_TIERS = {"strict", "balanced", "permissive"}


def _tier_from_template(template_name: str) -> str:
    """Read the tier from a ``chora-guardrail-{tier}-{env}`` template name.

    Unknown / unparseable → ``strict`` (fail closed, security wins).
    """
    name = (template_name or "").strip().lower()
    for tier in _VALID_TIERS:
        if f"chora-guardrail-{tier}-" in name or name.endswith(f"-{tier}"):
            return tier
    return "strict"


class LocalScreener:
    """In-process, tier-aware ``Screener`` backed by a configurable blocklist."""

    def __init__(
        self,
        *,
        blocklist: tuple[str, ...] = _DEFAULT_BLOCKLIST,
    ) -> None:
        self._blocklist = tuple(p.strip().lower() for p in blocklist if p and p.strip())

    @classmethod
    def from_env(cls) -> LocalScreener:
        raw = (os.getenv(ENV_BLOCKLIST) or "").strip()
        if raw:
            patterns = tuple(p.strip() for p in raw.split(",") if p.strip())
            return cls(blocklist=patterns or _DEFAULT_BLOCKLIST)
        return cls()

    async def sanitize_user_prompt(self, req: ScreenRequest) -> ScreenResult:
        return self._screen(req)

    async def sanitize_model_response(self, req: ScreenRequest) -> ScreenResult:
        return self._screen(req)

    async def close(self) -> None:
        return None

    def _screen(self, req: ScreenRequest) -> ScreenResult:
        if not req.text:
            raise ValueError("ScreenRequest.text is required (non-empty)")
        tier = _tier_from_template(req.template_name)
        text = req.text.lower()
        hits = [p for p in self._blocklist if p and p in text]
        if not hits:
            return ScreenResult(
                verdict=Verdict.ALLOW,
                reason="local_guardrail_clean",
                filters=[],
                latency_ms=0,
                raw_response={"blocker": "local", "tier": tier},
            )
        filters = [FilterHit(filter_name="local_blocklist", match_state="MATCH_FOUND") for _ in hits]
        if tier == "permissive":
            return ScreenResult(
                verdict=Verdict.INSPECT_ONLY,
                reason="local_guardrail_advisory",
                filters=filters,
                latency_ms=0,
                raw_response={"blocker": "local", "tier": tier, "hits": hits},
            )
        return ScreenResult(
            verdict=Verdict.BLOCK,
            reason="local_guardrail_match",
            filters=filters,
            latency_ms=0,
            raw_response={"blocker": "local", "tier": tier, "hits": hits},
        )


__all__ = ["ENV_BLOCKLIST", "LocalScreener"]
