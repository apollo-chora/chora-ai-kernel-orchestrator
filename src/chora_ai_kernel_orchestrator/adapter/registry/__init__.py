"""Agent registry adapter — persists the in-process registry to Postgres.

At startup the orchestrator UPSERTs all 6 AI Assist gate agents into
`chora_agent_registry`. Production wiring uses psycopg over the chora_ai_kernel
DSN; tests use a duck-typed connection mock.
"""

from __future__ import annotations

from chora_ai_kernel_orchestrator.adapter.registry.persister import (
    AgentRegistryPersister,
    persist_ai_assist_registry,
)

__all__ = ["AgentRegistryPersister", "persist_ai_assist_registry"]
