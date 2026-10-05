"""UPSERTs the AI Assist agent registry into chora_agent_registry.

The persister speaks an explicit `_ConnLike` Protocol so unit tests can
inject a fake. Production wires a `psycopg.Connection` from the
chora_ai_kernel DSN.

Idempotent: rerunning at startup updates risk_tier / autonomy_level /
guardrail_yaml_ref in place. We never DELETE — deactivate via
`active = FALSE` instead.
"""

from __future__ import annotations

from typing import Any, Protocol

from chora_ai_kernel_orchestrator.domain.ai_assist_crew import (
    AI_ASSIST_AGENT_REGISTRY,
    AIAssistAgentDescriptor,
)

_UPSERT_SQL = """
INSERT INTO chora_agent_registry (
    agent_id, role, display_name, risk_tier, autonomy_level,
    guardrail_yaml_ref, imda_dimension, hitl_imda_dimension, owner_team
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (agent_id) DO UPDATE SET
    role = EXCLUDED.role,
    display_name = EXCLUDED.display_name,
    risk_tier = EXCLUDED.risk_tier,
    autonomy_level = EXCLUDED.autonomy_level,
    guardrail_yaml_ref = EXCLUDED.guardrail_yaml_ref,
    imda_dimension = EXCLUDED.imda_dimension,
    hitl_imda_dimension = EXCLUDED.hitl_imda_dimension,
    owner_team = EXCLUDED.owner_team,
    active = TRUE,
    updated_at = now()
"""


class _CursorLike(Protocol):
    def execute(self, query: str, params: tuple[Any, ...] = ...) -> Any: ...
    def close(self) -> None: ...


class _ConnLike(Protocol):
    def cursor(self) -> _CursorLike: ...
    def commit(self) -> None: ...


class AgentRegistryPersister:
    """UPSERTs descriptors into `chora_agent_registry`."""

    def __init__(self, conn: _ConnLike) -> None:
        self._conn = conn

    def persist(self, descriptors: tuple[AIAssistAgentDescriptor, ...]) -> int:
        """UPSERT all descriptors. Returns rowcount."""
        cur = self._conn.cursor()
        try:
            for d in descriptors:
                cur.execute(
                    _UPSERT_SQL,
                    (
                        d.agent_id,
                        d.role.value,
                        d.display_name,
                        d.risk_tier,
                        d.autonomy_level.value,
                        d.guardrail_yaml_ref,
                        d.imda_dimension,
                        d.hitl_imda_dimension,
                        d.owner_team,
                    ),
                )
            self._conn.commit()
        finally:
            cur.close()
        return len(descriptors)


def persist_ai_assist_registry(conn: _ConnLike) -> int:
    """Convenience: persist the canonical 6 AI Assist gate agents."""
    persister = AgentRegistryPersister(conn)
    return persister.persist(AI_ASSIST_AGENT_REGISTRY.all())
