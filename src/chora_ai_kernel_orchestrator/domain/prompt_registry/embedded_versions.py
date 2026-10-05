"""Embedded prompt versions per decision-emitting agent (CHO-2364, ADR-197).

The prompt-override registry is OVERRIDES-ONLY: when no override resolves the
agent runs its embedded default prompt, whose version each agent declares in
its own agentconfig YAML (``prompt_version`` per sub-agent, all sub-agents of
one agent share a single version today):

* ``agents/qgen_adk_go/internal/agentconfig/qgen_question.yaml``
* ``agents/qgen_adk_go/internal/agentconfig/qgen_critic.yaml``
* ``agents/oe_grading_adk_go/internal/agentconfig/oe_evaluator.yaml``
* ``agents/oe_grading_adk_go/internal/agentconfig/oe_moderator.yaml``

This module mirrors those values as a plain literal so EVERY AgentDecisionLog
can stamp ``prompt_version`` + ``prompt_source`` - including the embedded
default path where the resolver provided nothing. The mirror is drift-pinned
by ``tests/unit/test_embedded_prompt_versions.py`` (repo-mirror pattern, like
the Go seedspec drift tests): a YAML ``prompt_version`` bump goes RED there
until this map is bumped in the same change. Do NOT edit one side without the
other.
"""

from __future__ import annotations

from typing import Final

# agent_id (agid) -> embedded prompt version. Values MUST equal the
# prompt_version fields in the agent's agentconfig YAML (see module docstring).
#
# The chat agent has no agentconfig YAML - its entry mirrors the Go constant
# FamiliarPromptVersion (agents/companion_chat_adk_go/internal/agent/conditions.go),
# pinned by tests/unit/test_embedded_prompt_versions.py (CHO-2368 P3).
#
# ⚠ Its key is `companion_chat`, NOT the pre-rename `familiar`, because THIS map
# is the fail-open of the resolve route (`EMBEDDED_PROMPT_VERSIONS.get(agent_id)`
# in adapter/http/prompt_registry_router.py) and chora-consumption resolves under
# `companion_chat` (cmd/server/companion_bus_wiring.go). Keyed on the old name the
# lookup misses, and the route answers `version=null, source="embedded"` while the
# agent really runs FamiliarPromptVersion: a provenance stamp reading EMPTY beside
# a prompt version that actually executed, which is an IMDA D2 transparency defect
# rather than a naming lag.
#
# ⚠ NOT YET FINISHED: the prompt_override_plan / prompt_override_segment ROWS are
# still stored under agent_id `familiar` (seeded by the immutable migration 0009,
# plus an active 1.1.0 wave). Until the rename migration lands, a resolve under
# `companion_chat` still finds no PLAN and falls through to this map, which is the
# correct and honest answer. baseline_seedspec.py keeps saying `familiar` on
# purpose: it is byte-pinned to 0009 by tests/unit/test_prompt_catalogue_migration
# and is the record of what was seeded, not a statement about what is current.
EMBEDDED_PROMPT_VERSIONS: Final[dict[str, str]] = {
    "qgen_question": "v1",
    "qgen_critic": "v1",
    "oe_evaluator": "v1",
    "oe_moderator": "v1",
    "companion_chat": "v1",
}
