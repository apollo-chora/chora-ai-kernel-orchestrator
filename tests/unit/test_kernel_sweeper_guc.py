"""RED→GREEN: the ADR-184-shaped kernel sweeper opt-in GUC (G2).

Migrations 0055/0056 shipped a FAIL-OPEN tenant policy: an unset
``chora.tenant_id`` matched EVERY tenant's rows. Measured on the live database
2026-08-23 as ``chora_ai_kernel_app_rw`` (NOBYPASSRLS, non-owner, so the policy
genuinely applies): an unset GUC saw 41 park rows spanning 2 DISTINCT TENANTS,
while a different valid tenant UUID correctly saw 0. So the tenant arm works and
only the DEFAULT is wrong — a forgotten ``set_config`` reads cross-tenant and
looks exactly like working code.

The fix TIGHTENS: the policy becomes fail-CLOSED and the genuinely cross-tenant
machinery (the reaper scan, the boot resume sweep, the ledger backfill) opts in
EXPLICITLY through ``chora.kernel_sweeper``, the ADR-192 GUC pattern applied to
the ADR-184 shape (intra-service machinery on chora_ai_kernel). Unset
contributes nothing, so an ordinary tenant-scoped query is unchanged.

⚠ THE TRAP THIS FILE EXISTS TO PIN. ``ReconnectingAsyncConnection`` self-heals
transparently and re-applied ONLY ``autocommit``. A session-level GUC set once
at startup is therefore SILENTLY LOST on the first reconnect — and under a
fail-closed policy the sweeper then reads 0 rows and TESTS GREEN. A reaper that
reaps nothing is the false-zero failure this whole change can cause, so the
re-application on reconnect is pinned here, not left to review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.reconnecting_connection import (
    ReconnectingAsyncConnection,
)


@dataclass
class _FakeCursor:
    executed: list[tuple[str, Any]] = field(default_factory=list)

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    async def fetchone(self) -> Any:
        return None

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


@dataclass
class _FakeUnderlyingConn:
    closed: int = 0
    broken: int = 0
    autocommit_calls: list[bool] = field(default_factory=list)
    cur: _FakeCursor = field(default_factory=_FakeCursor)

    def cursor(self, *a: Any, **k: Any) -> _FakeCursor:
        if self.closed:
            raise RuntimeError("the connection is closed")
        return self.cur

    async def set_autocommit(self, value: bool) -> None:
        self.autocommit_calls.append(value)

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = 1


def _factory() -> tuple[list[_FakeUnderlyingConn], Any]:
    conns: list[_FakeUnderlyingConn] = []

    async def connect(_dsn: str) -> _FakeUnderlyingConn:
        c = _FakeUnderlyingConn()
        conns.append(c)
        return c

    return conns, connect


def _set_config_calls(conn: _FakeUnderlyingConn) -> list[tuple[str, Any]]:
    return [(s, p) for (s, p) in conn.cur.executed if "set_config" in s]


class TestOptInContributesNothingWhenUnset:
    """ADR-192's core property: not opting in must change nothing at all."""

    @pytest.mark.asyncio
    async def test_no_session_settings_issues_no_set_config(self) -> None:
        conns, connect = _factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect)
        async with w.cursor() as cur:
            await cur.execute("SELECT 1")
        assert _set_config_calls(conns[0]) == []

    @pytest.mark.asyncio
    async def test_empty_session_settings_issues_no_set_config(self) -> None:
        conns, connect = _factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect, session_settings={})
        async with w.cursor() as cur:
            await cur.execute("SELECT 1")
        assert _set_config_calls(conns[0]) == []


class TestSweeperOptIn:
    @pytest.mark.asyncio
    async def test_applies_session_setting_on_first_connect(self) -> None:
        conns, connect = _factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect, session_settings={"chora.kernel_sweeper": "on"})
        async with w.cursor() as cur:
            await cur.execute("SELECT 1")
        calls = _set_config_calls(conns[0])
        assert len(calls) == 1, f"expected exactly one set_config, got {calls}"
        sql, params = calls[0]
        # Parameterised — the GUC name/value must never be string-interpolated.
        assert "%s" in sql
        assert params == ("chora.kernel_sweeper", "on", False)

    @pytest.mark.asyncio
    async def test_session_setting_precedes_the_first_caller_statement(self) -> None:
        conns, connect = _factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect, session_settings={"chora.kernel_sweeper": "on"})
        async with w.cursor() as cur:
            await cur.execute("SELECT 1")
        executed = conns[0].cur.executed
        assert "set_config" in executed[0][0], f"GUC must be applied first, got {executed}"

    @pytest.mark.asyncio
    async def test_reapplies_session_setting_on_reconnect(self) -> None:
        """THE TRAP. A silent reconnect must not drop the sweeper opt-in."""
        conns, connect = _factory()
        w = ReconnectingAsyncConnection("dsn", connect=connect, session_settings={"chora.kernel_sweeper": "on"})
        async with w.cursor() as cur:
            await cur.execute("SELECT 1")
        # the DB blip / cost-pause resume
        conns[0].closed = 1
        async with w.cursor() as cur:
            await cur.execute("SELECT 1")
        assert len(conns) == 2, "expected a reconnect"
        assert _set_config_calls(conns[1]) == [
            (
                "SELECT set_config(%s, %s, %s)",
                ("chora.kernel_sweeper", "on", False),
            )
        ], "the sweeper GUC was NOT re-applied on reconnect: the reaper would read 0 rows and test green"

    @pytest.mark.asyncio
    async def test_reapplies_alongside_autocommit_on_reconnect(self) -> None:
        conns, connect = _factory()
        w = ReconnectingAsyncConnection(
            "dsn",
            autocommit=True,
            connect=connect,
            session_settings={"chora.kernel_sweeper": "on"},
        )
        async with w.cursor():
            pass
        conns[0].closed = 1
        async with w.cursor():
            pass
        assert conns[1].autocommit_calls == [True]
        assert len(_set_config_calls(conns[1])) == 1
