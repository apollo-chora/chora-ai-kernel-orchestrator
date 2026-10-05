"""Unit tests for ``PostgresPromptOverrideRepository`` (ADR-197 M-B.1).

Exercised against a fake psycopg ``AsyncConnection`` (no live DB) that records
the executed SQL + params so we can assert:

* the tenant GUC is bound tx-local via ``set_config('chora.tenant_id', ...)``
  BEFORE the SELECT (RLS) and inside an explicit transaction,
* DB rows are grouped into one ``ScopeOverride`` per scope (one active plan per
  scope → stable plan_id used as the version token),
* an empty result yields ``[]`` (resolver then returns the embedded default),
* fail-loud on a missing connection / empty agent_id.
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pg.prompt_override_repository import (
    PostgresPromptOverrideRepository,
)


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.executed: list[tuple[str, Any]] = []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeTxn:
    def __init__(self, owner: _FakeConn) -> None:
        self._owner = owner

    async def __aenter__(self) -> _FakeTxn:
        self._owner.txn_entered += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._owner.txn_exited += 1
        return None


class _FakeConn:
    """Async psycopg-connection stand-in: ``transaction()`` + ``cursor()`` are
    async context managers; the single cursor records every execute()."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.cursor_obj = _FakeCursor(rows)
        self.txn_entered = 0
        self.txn_exited = 0

    def transaction(self) -> _FakeTxn:
        return _FakeTxn(self)

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj


async def test_groups_rows_into_one_scope_override_per_scope() -> None:
    rows = [
        ("platform", "plan-plat", "role", "PLAT_ROLE"),
        ("platform", "plan-plat", "task", "PLAT_TASK"),
        ("tenant", "plan-ten", "role", "TEN_ROLE"),
    ]
    conn = _FakeConn(rows)
    repo = PostgresPromptOverrideRepository(conn=conn)

    out = await repo.fetch_active_overrides(tenant_id="11111111-1111-7111-8111-111111111111", agent_id="qgen-question")

    by_scope = {o.scope: o for o in out}
    assert set(by_scope) == {"platform", "tenant"}
    assert by_scope["platform"].version == "plan-plat"
    assert by_scope["platform"].segments == {"role": "PLAT_ROLE", "task": "PLAT_TASK"}
    assert by_scope["tenant"].version == "plan-ten"
    assert by_scope["tenant"].segments == {"role": "TEN_ROLE"}


async def test_sets_tenant_guc_before_select_inside_transaction() -> None:
    conn = _FakeConn([])
    repo = PostgresPromptOverrideRepository(conn=conn)

    await repo.fetch_active_overrides(tenant_id="ten-9", agent_id="qgen-question")

    # The transaction wrapped the work (GUC set_config(is_local=true) must hold
    # for the SELECT regardless of autocommit mode).
    assert conn.txn_entered == 1
    assert conn.txn_exited == 1

    executed = conn.cursor_obj.executed
    assert len(executed) == 2
    first_sql, first_params = executed[0]
    assert "set_config('chora.tenant_id'" in first_sql
    assert first_params == ("ten-9",)

    second_sql, second_params = executed[1]
    assert "prompt_override_plan" in second_sql
    assert "prompt_override_segment" in second_sql
    assert second_params == {"agent_id": "qgen-question", "tenant_id": "ten-9"}


async def test_empty_result_returns_empty_list() -> None:
    conn = _FakeConn([])
    repo = PostgresPromptOverrideRepository(conn=conn)

    out = await repo.fetch_active_overrides(tenant_id="ten-1", agent_id="qgen-question")

    assert out == []


async def test_blank_tenant_still_queries_platform() -> None:
    """A blank tenant is allowed (tenant-agnostic resolution); the NULLIF-safe
    query returns platform-scope overrides only."""
    rows = [("platform", "plan-plat", "examples", "PLAT_EX")]
    conn = _FakeConn(rows)
    repo = PostgresPromptOverrideRepository(conn=conn)

    out = await repo.fetch_active_overrides(tenant_id="   ", agent_id="qgen-question")

    assert len(out) == 1
    assert out[0].scope == "platform"
    # tenant GUC bound to the empty string (NULLIF makes it NULL in SQL).
    assert conn.cursor_obj.executed[0][1] == ("",)


async def test_missing_connection_raises() -> None:
    with pytest.raises(ValueError, match="requires a connection"):
        PostgresPromptOverrideRepository(conn=None)


async def test_empty_agent_id_raises() -> None:
    repo = PostgresPromptOverrideRepository(conn=_FakeConn([]))
    with pytest.raises(ValueError, match="non-empty agent_id"):
        await repo.fetch_active_overrides(tenant_id="ten-1", agent_id="  ")
