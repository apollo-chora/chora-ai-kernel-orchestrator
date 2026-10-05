"""Tests for :class:`ModelArmorGuardrailPort` template-tier resolution.

Anchors:

- ADR-152 mandates tier-based template names of shape
  ``chora-guardrail-{tier}-{env}`` where tier ∈ {strict, balanced, permissive}.
- The agent → tier mapping is loaded from
  ``chora-contracts/yaml/agent-guardrail-mapping.yaml`` at port construction.
- Unknown agents fall back to the YAML's ``default.template_tier`` — which
  ADR-152 sets to ``strict`` (security wins per Principle Conflict #1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    GuardrailScreenInput,
    ModelArmorGuardrailPort,
    ScreenRequest,
    ScreenResult,
    StubScreener,
    Verdict,
)


def _write_mapping(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "agent-guardrail-mapping.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def _port(
    tmp_path: Path,
    mapping_yaml: str,
    *,
    project: str = "chora-489812",
    location: str = "us-central1",
    environment: str = "dev",
) -> tuple[ModelArmorGuardrailPort, list[ScreenRequest]]:
    """Build a port with the supplied mapping YAML; return (port, captured_requests)."""
    captured: list[ScreenRequest] = []

    class _RecordingScreener(StubScreener):
        async def sanitize_user_prompt(self, req: ScreenRequest) -> ScreenResult:
            captured.append(req)
            return await super().sanitize_user_prompt(req)

        async def sanitize_model_response(self, req: ScreenRequest) -> ScreenResult:
            captured.append(req)
            return await super().sanitize_model_response(req)

    path = _write_mapping(tmp_path, mapping_yaml)
    port = ModelArmorGuardrailPort.from_components(
        screener=_RecordingScreener(force_verdict=Verdict.ALLOW),
        project=project,
        location=location,
        environment=environment,
        mapping_path=str(path),
    )
    return port, captured


def _input(agent_id: str) -> GuardrailScreenInput:
    return GuardrailScreenInput(
        tenant_id="00000000-0000-7000-8000-000000000001",
        gcid="00000000-0000-7000-8000-000000000002",
        agent_id=agent_id,
        content="hello",
        direction="input",
    )


# ---- tier resolution -------------------------------------------------------


_MAPPING_BODY = """
version: 1
agents:
  governance_gatekeeper:
    template_tier: strict
    rationale: high-stakes
  qgen_question_generator:
    template_tier: balanced
    rationale: learner-issued
  closure_saga_orchestrator:
    template_tier: permissive
    rationale: internal-only
default:
  template_tier: strict
  rationale: unknown → strict
"""


@pytest.mark.asyncio
async def test_strict_agent_resolves_strict_template(tmp_path: Path) -> None:
    """A `template_tier: strict` agent resolves to the strict template."""
    port, captured = _port(tmp_path, _MAPPING_BODY)
    await port.screen(_input("governance_gatekeeper"))
    assert len(captured) == 1
    assert captured[0].template_name == (
        "projects/chora-489812/locations/us-central1/templates/chora-guardrail-strict-dev"
    )


@pytest.mark.asyncio
async def test_balanced_agent_resolves_balanced_template(tmp_path: Path) -> None:
    port, captured = _port(tmp_path, _MAPPING_BODY)
    await port.screen(_input("qgen_question_generator"))
    assert captured[0].template_name == (
        "projects/chora-489812/locations/us-central1/templates/chora-guardrail-balanced-dev"
    )


@pytest.mark.asyncio
async def test_permissive_agent_resolves_permissive_template(tmp_path: Path) -> None:
    port, captured = _port(tmp_path, _MAPPING_BODY)
    await port.screen(_input("closure_saga_orchestrator"))
    assert captured[0].template_name == (
        "projects/chora-489812/locations/us-central1/templates/chora-guardrail-permissive-dev"
    )


@pytest.mark.asyncio
async def test_unknown_agent_falls_back_to_default_strict(tmp_path: Path) -> None:
    """Per ADR-152: unknown agent_id → default tier (strict) — never permissive."""
    port, captured = _port(tmp_path, _MAPPING_BODY)
    await port.screen(_input("unknown-agent-id"))
    assert captured[0].template_name.endswith("/templates/chora-guardrail-strict-dev")


@pytest.mark.asyncio
async def test_invalid_tier_in_mapping_is_ignored(tmp_path: Path) -> None:
    """A bogus tier in the YAML is dropped — unknown agent path is hit instead."""
    body = """
version: 1
agents:
  bogus_agent:
    template_tier: yolo
    rationale: invalid
default:
  template_tier: strict
"""
    port, captured = _port(tmp_path, body)
    await port.screen(_input("bogus_agent"))
    # falls through to default (strict)
    assert captured[0].template_name.endswith("/templates/chora-guardrail-strict-dev")


@pytest.mark.asyncio
async def test_default_tier_invalid_falls_back_to_strict(tmp_path: Path) -> None:
    """Even an invalid default tier collapses to strict (security wins)."""
    body = """
version: 1
agents: {}
default:
  template_tier: chaos_monkey
"""
    port, captured = _port(tmp_path, body)
    await port.screen(_input("anything"))
    assert captured[0].template_name.endswith("/templates/chora-guardrail-strict-dev")


@pytest.mark.asyncio
async def test_env_var_environment_appears_in_template_name(tmp_path: Path) -> None:
    """environment suffix flips between dev/staging/prod."""
    port, captured = _port(
        tmp_path,
        _MAPPING_BODY,
        environment="prod",
    )
    await port.screen(_input("qgen_question_generator"))
    assert captured[0].template_name.endswith("/templates/chora-guardrail-balanced-prod")


@pytest.mark.asyncio
async def test_env_var_project_location_appear_in_template_name(tmp_path: Path) -> None:
    """Project + location env vars compose into the full resource name."""
    port, captured = _port(
        tmp_path,
        _MAPPING_BODY,
        project="chora-staging",
        location="asia-southeast1",
    )
    await port.screen(_input("governance_gatekeeper"))
    assert captured[0].template_name == (
        "projects/chora-staging/locations/asia-southeast1/templates/chora-guardrail-strict-dev"
    )


def test_resolver_template_for_unknown_returns_default_strict(tmp_path: Path) -> None:
    """Resolver computation is pure — covers boot-time validation."""
    port, _ = _port(tmp_path, _MAPPING_BODY)
    name = port.resolver.template_for("never-seen")
    assert name.endswith("/templates/chora-guardrail-strict-dev")


def test_resolver_template_for_strict_agent_matches(tmp_path: Path) -> None:
    port, _ = _port(tmp_path, _MAPPING_BODY)
    name = port.resolver.template_for("governance_gatekeeper")
    assert name.endswith("/templates/chora-guardrail-strict-dev")


# ---- yaml shape edge cases -------------------------------------------------


@pytest.mark.asyncio
async def test_empty_yaml_uses_strict_default(tmp_path: Path) -> None:
    """An empty / malformed YAML still composes a working strict default."""
    port, captured = _port(tmp_path, "")
    await port.screen(_input("any"))
    assert captured[0].template_name.endswith("/templates/chora-guardrail-strict-dev")


@pytest.mark.asyncio
async def test_yaml_with_non_dict_entry_is_skipped(tmp_path: Path) -> None:
    """Defensive — a list-shaped agent entry doesn't crash the loader."""
    body = """
version: 1
agents:
  list_agent:
    - oops
  good_agent:
    template_tier: balanced
default:
  template_tier: strict
"""
    port, captured = _port(tmp_path, body)
    await port.screen(_input("list_agent"))
    # malformed agent → falls back to default strict
    assert captured[0].template_name.endswith("/templates/chora-guardrail-strict-dev")
    await port.screen(_input("good_agent"))
    assert captured[1].template_name.endswith("/templates/chora-guardrail-balanced-dev")


# ---- env-var fallback in _load_resolver_config -----------------------------


def test_load_resolver_config_uses_env_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When kwargs are None, _load_resolver_config picks up the env vars."""
    from chora_ai_kernel_orchestrator.adapter.modelarmor.port import (
        _load_resolver_config,
    )

    path = _write_mapping(tmp_path, _MAPPING_BODY)
    monkeypatch.setenv("CHORA_MODELARMOR_PROJECT", "env-project")
    monkeypatch.setenv("CHORA_MODELARMOR_LOCATION", "env-location")
    monkeypatch.setenv("CHORA_ENVIRONMENT", "env-stage")

    cfg = _load_resolver_config(mapping_path=str(path))
    assert cfg.project == "env-project"
    assert cfg.location == "env-location"
    assert cfg.environment == "env-stage"
    assert cfg.tier_by_agent["governance_gatekeeper"] == "strict"
    assert cfg.default_tier == "strict"


def test_load_resolver_config_missing_file_raises(tmp_path: Path) -> None:
    from chora_ai_kernel_orchestrator.adapter.modelarmor.port import (
        _load_resolver_config,
    )

    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        _load_resolver_config(mapping_path=str(missing))


# ---- lazy screener factory -------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_screener_defers_inner_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """from_env without an explicit screener wires a lazy screener that
    only resolves the concrete screener on first call."""
    from chora_ai_kernel_orchestrator.adapter.modelarmor import port as port_module

    constructed: list[Any] = []

    class _Inner:
        async def sanitize_user_prompt(self, req: ScreenRequest) -> ScreenResult:
            return ScreenResult(verdict=Verdict.ALLOW, reason="ok")

        async def sanitize_model_response(self, req: ScreenRequest) -> ScreenResult:
            return ScreenResult(verdict=Verdict.ALLOW, reason="ok")

        async def close(self) -> None:
            return None

    async def fake_new_screener(*args: Any, **kwargs: Any) -> Any:
        constructed.append((args, kwargs))
        return _Inner()

    monkeypatch.setattr(port_module, "new_screener", fake_new_screener)
    port = ModelArmorGuardrailPort.from_env()
    assert port is not None
    assert constructed == []  # not built yet

    out = await port.screen(_input("any"))
    assert out.verdict == Verdict.ALLOW
    assert len(constructed) == 1

    # Second call reuses the inner.
    await port.screen(_input("any"))
    assert len(constructed) == 1

    await port.close()
