"""EMBEDDED_PROMPT_VERSIONS drift pin (CHO-2364, ADR-197 read slice).

Repo-mirror pattern (same discipline as the Go seedspec drift tests): the
orchestrator ships a plain-literal ``EMBEDDED_PROMPT_VERSIONS`` map so every
AgentDecisionLog can stamp the embedded prompt version even when no override
resolved. The agents' own agentconfig YAMLs (agents/qgen_adk_go +
agents/oe_grading_adk_go, SAME repo) are the source of truth for that version -
this test parses them and asserts the map matches EXACTLY, so a YAML
``prompt_version`` bump goes RED here until the map is bumped in the same
change. The map never silently drifts from what the deployed agents embed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from chora_ai_kernel_orchestrator.domain.prompt_registry.embedded_versions import (
    EMBEDDED_PROMPT_VERSIONS,
)

# Service root = this repo (the test lives at tests/unit/ inside it). The
# agentconfig YAMLs + the familiar Go constant are vendored into
# testdata/agent_configs/ (they were originally read from the sibling ADK
# agent checkouts + chora-consumption; the standalone repo vendors the data
# it parity-pins against so there is no cross-repo reference).
_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_AGENT_CONFIG_DIR = _SERVICE_ROOT / "testdata" / "agent_configs"

# agent_id -> agentconfig YAML path (relative to the vendored config dir).
_AGENT_YAML: dict[str, str] = {
    "qgen_question": "qgen_question.yaml",
    "qgen_critic": "qgen_critic.yaml",
    "oe_evaluator": "oe_evaluator.yaml",
    "oe_moderator": "oe_moderator.yaml",
}


def _load_agentconfig(agent_id: str) -> dict[str, Any]:
    path = _AGENT_CONFIG_DIR / _AGENT_YAML[agent_id]
    assert path.is_file(), f"agentconfig YAML missing for {agent_id}: {path}"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{path} did not parse to a mapping"
    return doc


# The familiar has NO agentconfig YAML (its prompt is composed per-instance
# from FamiliarConfig); its embedded version lives in the Go constant
# FamiliarPromptVersion (vendored from the companion_chat ADK agent).
_FAMILIAR_CONDITIONS_GO = "familiar_conditions.go"


def _familiar_go_prompt_version() -> str:
    import re

    path = _AGENT_CONFIG_DIR / _FAMILIAR_CONDITIONS_GO
    assert path.is_file(), f"familiar conditions.go missing: {path}"
    m = re.search(r'FamiliarPromptVersion\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"))
    assert m, f"FamiliarPromptVersion constant not found in {path}"
    return m.group(1)


def test_map_covers_exactly_the_wave_agents() -> None:
    """The map keys are exactly the 4 YAML-configured agents plus the chat
    agent (CHO-2368 P3) - no extras, no omissions (an extra key would stamp a
    version nothing embeds; a missing key resolves to null on the resolve
    route's embedded overlay)."""
    assert set(EMBEDDED_PROMPT_VERSIONS) == set(_AGENT_YAML) | {"companion_chat"}


def test_familiar_matches_go_constant() -> None:
    """Drift pin for the YAML-less agent: the map's familiar entry must equal
    the Go FamiliarPromptVersion constant - a Go bump goes RED here until the
    map moves in the same change."""
    assert EMBEDDED_PROMPT_VERSIONS["companion_chat"] == _familiar_go_prompt_version()


def test_map_values_are_non_empty_strings() -> None:
    for agent_id, version in EMBEDDED_PROMPT_VERSIONS.items():
        assert isinstance(version, str) and version.strip(), (
            f"EMBEDDED_PROMPT_VERSIONS[{agent_id!r}] must be a non-empty string"
        )


@pytest.mark.parametrize("agent_id", sorted(_AGENT_YAML))
def test_map_matches_agentconfig_yaml_exactly(agent_id: str) -> None:
    """Drift pin: the YAML's per-sub-agent prompt_version fields must all
    equal the map's value for that agent. A YAML bump (or a sub-agent
    diverging from its siblings) goes RED until the map is updated."""
    doc = _load_agentconfig(agent_id)
    assert doc.get("agent") == agent_id, (
        f"{_AGENT_YAML[agent_id]} declares agent={doc.get('agent')!r}, expected {agent_id!r}"
    )
    sub_agents = doc.get("sub_agents")
    assert isinstance(sub_agents, dict) and sub_agents, f"{_AGENT_YAML[agent_id]} has no sub_agents map"
    versions: set[str] = set()
    for name, cfg in sub_agents.items():
        assert isinstance(cfg, dict) and "prompt_version" in cfg, (
            f"{_AGENT_YAML[agent_id]} sub_agent {name!r} lacks prompt_version"
        )
        versions.add(str(cfg["prompt_version"]))
    assert versions == {EMBEDDED_PROMPT_VERSIONS[agent_id]}, (
        f"{agent_id}: YAML prompt_version(s) {sorted(versions)} != embedded "
        f"map value {EMBEDDED_PROMPT_VERSIONS[agent_id]!r} - bump "
        f"EMBEDDED_PROMPT_VERSIONS in the same change as the YAML"
    )


def test_the_chat_key_matches_the_id_consumption_resolves_under() -> None:
    """The fail-open is a dict lookup on the agent_id the CALLER sends, so this
    key and chora-consumption's resolver id are one value in two repos. Keyed on
    the pre-rename `familiar` the lookup misses and the route answers
    version=null / source="embedded" while the agent runs a real prompt version,
    which is a transparency defect a green suite cannot see.

    Pinned against the consumption wiring (vendored into testdata/agent_configs/)
    rather than restated, so a rename on either side goes RED here instead of
    going quiet in production.
    """
    wiring = _AGENT_CONFIG_DIR / "companion_bus_wiring.go"
    assert wiring.is_file(), f"consumption companion bus wiring missing: {wiring}"
    text = wiring.read_text(encoding="utf-8")
    assert "NewPromptRegistryHTTPResolver" in text, (
        "consumption no longer builds a prompt-registry resolver; this pin needs rehoming"
    )
    import re

    m = re.search(r'NewPromptRegistryHTTPResolver\([^,]+,\s*"([^"]+)"', text)
    assert m, "could not read the agent id consumption resolves under"
    assert m.group(1) in EMBEDDED_PROMPT_VERSIONS, (
        f"consumption resolves prompts under {m.group(1)!r} but EMBEDDED_PROMPT_VERSIONS "
        f"keys {sorted(EMBEDDED_PROMPT_VERSIONS)}; the fail-open would return null while "
        "the agent runs a real prompt version"
    )
