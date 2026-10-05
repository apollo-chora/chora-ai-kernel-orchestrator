"""Tests for the env-backed DSN resolver.

Per CLAUDE.md §6 + `secrets-and-env`: DSNs come from the environment (the
Secret Manager path is removed; the local substitute is a direct DSN in the
environment). The resolver must:

- Return the direct DSN env var when set.
- Return empty string when unset (dev / unit-test fallback).
- Treat whitespace-only env vars as unset (defensive).
"""

from __future__ import annotations

import pytest


def test_resolver_returns_direct_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHORA_AI_KERNEL_PG_DSN", "postgresql://direct/db")

    from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

    assert resolve_dsn() == "postgresql://direct/db"


def test_resolver_strips_surrounding_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHORA_AI_KERNEL_PG_DSN", "  postgresql://direct/db  ")

    from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

    assert resolve_dsn() == "postgresql://direct/db"


def test_resolver_returns_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHORA_AI_KERNEL_PG_DSN", raising=False)

    from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

    assert resolve_dsn() == ""


def test_resolver_blank_dsn_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only env vars are treated as unset (defensive)."""
    monkeypatch.setenv("CHORA_AI_KERNEL_PG_DSN", "   ")

    from chora_ai_kernel_orchestrator.adapter.secrets import resolve_dsn

    assert resolve_dsn() == ""
