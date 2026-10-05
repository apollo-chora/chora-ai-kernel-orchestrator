"""Tests for the chora_agent_registry persister (S3.4 P0 RED -> GREEN)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _FakeCursor:
    executed: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    closed: bool = False

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))

    def close(self) -> None:
        self.closed = True


@dataclass
class _FakeConn:
    cursor_obj: _FakeCursor = field(default_factory=_FakeCursor)
    commits: int = 0

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1


def test_persister_upserts_all_six_agents() -> None:
    from chora_ai_kernel_orchestrator.adapter.registry import (
        persist_ai_assist_registry,
    )

    conn = _FakeConn()
    n = persist_ai_assist_registry(conn)
    assert n == 6
    # 6 INSERT/UPDATE statements + 1 commit + cursor closed.
    assert len(conn.cursor_obj.executed) == 6
    assert conn.commits == 1
    assert conn.cursor_obj.closed is True


def test_persister_passes_descriptor_fields() -> None:
    """The first row UPSERTed is `validator` with the canonical risk + autonomy."""
    from chora_ai_kernel_orchestrator.adapter.registry import (
        persist_ai_assist_registry,
    )

    conn = _FakeConn()
    persist_ai_assist_registry(conn)
    first_query, first_params = conn.cursor_obj.executed[0]
    assert "INSERT INTO chora_agent_registry" in first_query
    assert "ON CONFLICT (agent_id) DO UPDATE" in first_query
    # Tuple ordering matches _UPSERT_SQL placeholders.
    agent_id, role, display_name, risk_tier, autonomy, _yaml, _dim, _hitl, _team = first_params
    assert agent_id == "validator"
    assert role == "validator"
    assert display_name == "AI Assist — Validator"
    assert risk_tier == 2
    assert autonomy == "HOOTL"


def test_persister_explicit_descriptors_subset() -> None:
    from chora_ai_kernel_orchestrator.adapter.registry import (
        AgentRegistryPersister,
    )
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew import (
        AI_ASSIST_AGENT_REGISTRY,
        AIAssistAgentRole,
    )

    conn = _FakeConn()
    only_reporter = (AI_ASSIST_AGENT_REGISTRY.get_by_role(AIAssistAgentRole.REPORTER),)
    n = AgentRegistryPersister(conn).persist(only_reporter)
    assert n == 1
    assert len(conn.cursor_obj.executed) == 1
    _, params = conn.cursor_obj.executed[0]
    # Reporter declares HITL_LEVEL_0 + transparency.
    assert params[4] == "HITL_LEVEL_0"
    assert params[6] == "transparency"
    assert params[7] == "fairness_and_human_oversight"
