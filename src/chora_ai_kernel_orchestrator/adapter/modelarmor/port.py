"""``ModelArmorGuardrailPort`` — LangGraph-facing guardrail port.

Replaces the legacy ``adapter/http/guardrail_client.py`` per
ADR-152 (chora-guardrail superseded by a platform screener). The LangGraph
``guardrail_screen`` node depends on this port — the underlying screener
is :class:`.local_screener.LocalScreener` (the cloud-neutral local
guardrail) or :class:`.stub.StubScreener` (tests).

Design points:

- **Tier mapping loaded once at boot** from
  ``config/agent-guardrail-mapping.yaml`` (vendored; override with
  ``CHORA_AGENT_GUARDRAIL_MAPPING_PATH``). The YAML lookup is
  agent_id → ``template_tier`` (strict / balanced / permissive); the
  template label is then constructed from the tier + ``CHORA_ENVIRONMENT``.
- **Direction routing**: ``screen(direction="input")`` →
  :meth:`Screener.sanitize_user_prompt`; ``direction="output"`` →
  :meth:`Screener.sanitize_model_response`.
- **No inline config** — env vars + YAML only. Per CLAUDE.md §6 +
  ``feedback_no_inline_config``.
- **Security wins** — if the agent_id is unknown the YAML's ``default``
  entry (always ``strict`` per ADR-152 spec) applies, NOT a permissive
  fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from chora_ai_kernel_orchestrator.adapter.modelarmor.screener import (
    Screener,
    ScreenRequest,
    ScreenResult,
    new_screener,
)

# Default location of the agent → tier mapping, vendored into this repo at
# config/agent-guardrail-mapping.yaml. Resolved relative to this file when the
# env var is unset. Production callers should set
# CHORA_AGENT_GUARDRAIL_MAPPING_PATH (the Dockerfile COPYs the file to
# /etc/chora/agent-guardrail-mapping.yaml).
_DEFAULT_MAPPING_RELATIVE = "../../../../config/agent-guardrail-mapping.yaml"

# The template_name is a label the local screener parses for its tier
# (``chora-guardrail-{tier}-{env}``); project / location are provenance
# only — the local screener contacts no remote resource.
_DEFAULT_PROJECT = "chora-local"
_DEFAULT_LOCATION = "local"
_DEFAULT_ENV = "dev"

_VALID_TIERS = {"strict", "balanced", "permissive"}


@dataclass(frozen=True)
class GuardrailScreenInput:
    """Direction-tagged screen request — the LangGraph node's view.

    Kept narrow so the node's invariants (tenant_id, gcid, agent_id, prompt)
    stay obvious. The port internally constructs the wider
    :class:`ScreenRequest` once the template_name is resolved.

    ``direction`` is ``"input"`` (pre-LLM) or ``"output"`` (post-LLM); maps
    1:1 onto the two Screener methods.
    """

    tenant_id: str
    gcid: str
    agent_id: str
    content: str
    direction: str  # "input" | "output"


@dataclass(frozen=True)
class _ResolverConfig:
    """Composition-root config for :class:`ModelArmorGuardrailPort`."""

    project: str
    location: str
    environment: str
    tier_by_agent: dict[str, str]
    default_tier: str

    def template_for(self, agent_id: str) -> str:
        """Resolve the full template name for an agent.

        Pattern:
        ``projects/{project}/locations/{location}/templates/chora-guardrail-{tier}-{env}``

        Unknown agents fall back to ``default_tier`` (strict per ADR-152
        spec — security wins; never permissive).
        """
        tier = self.tier_by_agent.get(agent_id, self.default_tier)
        if tier not in _VALID_TIERS:
            tier = self.default_tier
        return f"projects/{self.project}/locations/{self.location}/templates/chora-guardrail-{tier}-{self.environment}"


class ModelArmorGuardrailPort:
    """LangGraph guardrail port backed by Cloud Model Armor.

    Instantiate via :meth:`from_env` at composition root. The constructor
    accepts an injected :class:`Screener` so tests can pass a
    :class:`StubScreener` without monkey-patching.

    The port is stateless except for the resolver config; it is safe to
    share across requests.
    """

    def __init__(self, *, screener: Screener, resolver: _ResolverConfig) -> None:
        self._screener = screener
        self._resolver = resolver

    @classmethod
    def from_env(
        cls,
        *,
        screener: Screener | None = None,
        mapping_path: str | None = None,
    ) -> ModelArmorGuardrailPort | None:
        """Build from env vars + YAML mapping.

        Reads (with defaults):

        - ``CHORA_MODELARMOR_PROJECT``     (default ``chora-local``)
        - ``CHORA_MODELARMOR_LOCATION``    (default ``local``)
        - ``CHORA_ENVIRONMENT``            (default ``dev``)
        - ``CHORA_AGENT_GUARDRAIL_MAPPING_PATH``
          (default: ``config/agent-guardrail-mapping.yaml``
           resolved relative to this module)

        When ``screener`` is omitted the port wires a **lazy** screener
        that constructs the :class:`.local_screener.LocalScreener` on first
        ``screen()`` call via :func:`new_screener`.

        Returns ``None`` only when the YAML mapping is unreadable — the
        readiness probe then surfaces the misconfiguration explicitly.
        """
        try:
            cfg = _load_resolver_config(mapping_path=mapping_path)
        except FileNotFoundError:
            return None

        if screener is None:
            screener = _LazyScreener(project=cfg.project, location=cfg.location)

        return cls(screener=screener, resolver=cfg)

    @classmethod
    def from_components(
        cls,
        *,
        screener: Screener,
        project: str | None = None,
        location: str | None = None,
        environment: str | None = None,
        mapping_path: str | None = None,
    ) -> ModelArmorGuardrailPort:
        """Explicit-construction variant for tests + bespoke composition.

        Mirrors :meth:`from_env` but takes the env-derived values as
        keyword arguments so unit tests can drive every tier mapping
        without mutating ``os.environ``.
        """
        cfg = _load_resolver_config(
            mapping_path=mapping_path,
            project=project,
            location=location,
            environment=environment,
        )
        return cls(screener=screener, resolver=cfg)

    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
        """Dispatch to the appropriate Screener method.

        ``direction == "input"``  → :meth:`Screener.sanitize_user_prompt`
        ``direction == "output"`` → :meth:`Screener.sanitize_model_response`

        Any other direction value raises ``ValueError`` — programmer error.
        """
        template_name = self._resolver.template_for(payload.agent_id)
        req = ScreenRequest(
            tenant_id=payload.tenant_id,
            agent_id=payload.agent_id,
            gcid=payload.gcid,
            template_name=template_name,
            text=payload.content,
        )
        direction = (payload.direction or "").strip().lower()
        if direction == "input":
            return await self._screener.sanitize_user_prompt(req)
        if direction == "output":
            return await self._screener.sanitize_model_response(req)
        raise ValueError(
            f"ModelArmorGuardrailPort.screen: unknown direction {direction!r} (expected 'input' or 'output')"
        )

    async def close(self) -> None:
        """Release the underlying screener transport. Safe to call twice."""
        await self._screener.close()

    # --------------------------------------------------------------- introspection

    @property
    def resolver(self) -> _ResolverConfig:
        """Expose the resolver for tests / readiness probes."""
        return self._resolver


# --------------------------------------------------------------- loader helpers


def _load_resolver_config(
    *,
    mapping_path: str | None = None,
    project: str | None = None,
    location: str | None = None,
    environment: str | None = None,
) -> _ResolverConfig:
    """Compose a :class:`_ResolverConfig` from env vars + YAML mapping.

    Falls back to the canonical defaults documented in :meth:`from_env`.
    Raises ``FileNotFoundError`` when the mapping file is unreadable.
    """
    project_value = project if project is not None else os.getenv("CHORA_MODELARMOR_PROJECT", _DEFAULT_PROJECT)
    location_value = location if location is not None else os.getenv("CHORA_MODELARMOR_LOCATION", _DEFAULT_LOCATION)
    env_value = environment if environment is not None else os.getenv("CHORA_ENVIRONMENT", _DEFAULT_ENV)

    path = _resolve_mapping_path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"ModelArmorGuardrailPort: agent-guardrail-mapping.yaml not found at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    agents = raw.get("agents") or {}
    default = raw.get("default") or {}

    tier_by_agent: dict[str, str] = {}
    for agent_id, entry in agents.items():
        tier = _safe_tier(entry)
        if tier is not None:
            tier_by_agent[str(agent_id)] = tier

    default_tier = _safe_tier(default) or "strict"

    return _ResolverConfig(
        project=str(project_value).strip() or _DEFAULT_PROJECT,
        location=str(location_value).strip() or _DEFAULT_LOCATION,
        environment=str(env_value).strip() or _DEFAULT_ENV,
        tier_by_agent=tier_by_agent,
        default_tier=default_tier,
    )


def _resolve_mapping_path(override: str | None) -> Path:
    """Pick the YAML path: explicit override → env var → container default → repo-relative dev fallback."""
    # Canonical absolute path baked into the Docker image (Dockerfile COPYs
    # chora-contracts/yaml/agent-guardrail-mapping.yaml → this location).
    # Production deployments leave the env var unset; this default Just Works.
    container_default = Path("/etc/chora/agent-guardrail-mapping.yaml")
    candidate = (
        override or os.getenv("CHORA_AGENT_GUARDRAIL_MAPPING") or os.getenv("CHORA_AGENT_GUARDRAIL_MAPPING_PATH") or ""
    )
    if candidate:
        return Path(candidate)
    if container_default.exists():
        return container_default
    here = Path(__file__).resolve().parent
    return (here / _DEFAULT_MAPPING_RELATIVE).resolve()


def _safe_tier(entry: Any) -> str | None:
    """Extract a valid tier string from a mapping entry, else None."""
    if not isinstance(entry, dict):
        return None
    raw = entry.get("template_tier")
    if not isinstance(raw, str):
        return None
    tier = raw.strip().lower()
    if tier not in _VALID_TIERS:
        return None
    return tier


class _LazyScreener:
    """Deferred-init :class:`Screener` — constructs the concrete
    :class:`LocalScreener` on first call.

    The Cloud Model Armor SDK reached out to ADC credentials on
    instantiation, so the retired lazy wrapper deferred until the first
    request. The local screener needs no remote resource, but the wrapper is
    kept so the composition root's ``from_env`` contract (and its tests) are
    unchanged: the concrete screener resolves on first ``screen()`` call.
    """

    def __init__(self, *, project: str, location: str) -> None:
        self._project = project
        self._location = location
        self._inner: Screener | None = None

    async def _resolve(self) -> Screener:
        if self._inner is None:
            self._inner = await new_screener()
        return self._inner

    async def sanitize_user_prompt(self, req: ScreenRequest) -> ScreenResult:
        inner = await self._resolve()
        return await inner.sanitize_user_prompt(req)

    async def sanitize_model_response(self, req: ScreenRequest) -> ScreenResult:
        inner = await self._resolve()
        return await inner.sanitize_model_response(req)

    async def close(self) -> None:
        if self._inner is not None:
            await self._inner.close()
            self._inner = None
