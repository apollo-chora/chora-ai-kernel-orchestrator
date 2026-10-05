"""RED→GREEN tests for InboxIdempotencyStore.

Python mirror of libs/chora-go-common/idempotent.PostgresStore. Provides
``process(key, ttl, fn)``: dedup-then-invoke. Used by per-crew Pub/Sub
subscribers in the chora-ai-kernel-orchestrator kennel to short-circuit
duplicate started.v1 deliveries per [[feedback-d6-resilience-first-class]].

Mirrors the chora-creation subscriber's ``inbox.Process(ctx, key, ttl, fn)``
contract.

Schema: `idempotency_keys` table per migration 0004_inbox_idempotency.sql.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.inbox_idempotency import (
    InboxIdempotencyStore,
)

# -----------------------------------------------------------------------------
# Fake AsyncConnection / Cursor — covers the (key, ttl_at) UPSERT pattern.
# -----------------------------------------------------------------------------


@dataclass
class _FakeCursor:
    rows_by_key: dict[str, _dt.datetime] = field(default_factory=dict)
    executed: list[tuple[str, dict[str, Any] | tuple[Any, ...]]] = field(default_factory=list)
    last_result: Any = None

    async def execute(
        self,
        sql: str,
        params: dict[str, Any] | tuple[Any, ...] | None = None,
    ) -> None:
        self.executed.append((sql, params or {}))
        sql_lower = sql.lower()
        if "select" in sql_lower and "idempotency_keys" in sql_lower:
            # Seen() — caller fetches matching un-expired row
            assert isinstance(params, dict)
            key = params["key"]
            now = _dt.datetime.now(tz=_dt.UTC)
            ttl_at = self.rows_by_key.get(key)
            self.last_result = (key,) if ttl_at is not None and ttl_at > now else None
        elif "insert" in sql_lower and "idempotency_keys" in sql_lower:
            assert isinstance(params, dict)
            self.rows_by_key[params["key"]] = params["ttl_at"]

    async def fetchone(self) -> Any:
        return self.last_result

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


@dataclass
class _FakeConn:
    cur: _FakeCursor = field(default_factory=_FakeCursor)

    def cursor(self) -> _FakeCursor:
        return self.cur


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_requires_conn(self) -> None:
        store = InboxIdempotencyStore(conn=_FakeConn(), table="idempotency_keys")
        assert store.table == "idempotency_keys"

    def test_rejects_invalid_table(self) -> None:
        """Table name must be a safe identifier — prevents SQL injection."""
        with pytest.raises(ValueError, match="table name"):
            InboxIdempotencyStore(conn=_FakeConn(), table="bad table; DROP TABLE")


# -----------------------------------------------------------------------------
# process — the high-level dedup-then-invoke contract
# -----------------------------------------------------------------------------


class TestProcessHappyPath:
    @pytest.mark.asyncio
    async def test_first_invocation_runs_fn_and_marks_key(self) -> None:
        conn = _FakeConn()
        store = InboxIdempotencyStore(conn=conn)

        fn_called = []

        async def fn() -> None:
            fn_called.append("yes")

        ran = await store.process(
            key="event-1",
            ttl=_dt.timedelta(days=7),
            fn=fn,
        )
        assert ran is True
        assert fn_called == ["yes"]
        # Key was UPSERTed.
        assert "event-1" in conn.cur.rows_by_key

    @pytest.mark.asyncio
    async def test_duplicate_invocation_skips_fn(self) -> None:
        conn = _FakeConn()
        # Pre-seed the table with a non-expired row.
        conn.cur.rows_by_key["event-1"] = _dt.datetime.now(tz=_dt.UTC) + _dt.timedelta(days=1)
        store = InboxIdempotencyStore(conn=conn)

        fn_called = []

        async def fn() -> None:
            fn_called.append("yes")

        ran = await store.process(
            key="event-1",
            ttl=_dt.timedelta(days=7),
            fn=fn,
        )
        assert ran is False
        assert fn_called == []  # fn never invoked

    @pytest.mark.asyncio
    async def test_expired_key_re_runs_fn(self) -> None:
        """TTL-expired rows are treated as 'not seen' — fn re-runs."""
        conn = _FakeConn()
        # Expired row.
        conn.cur.rows_by_key["event-1"] = _dt.datetime.now(tz=_dt.UTC) - _dt.timedelta(days=1)
        store = InboxIdempotencyStore(conn=conn)

        fn_called = []

        async def fn() -> None:
            fn_called.append("yes")

        ran = await store.process(
            key="event-1",
            ttl=_dt.timedelta(days=7),
            fn=fn,
        )
        assert ran is True
        assert fn_called == ["yes"]


# -----------------------------------------------------------------------------
# Error semantics
# -----------------------------------------------------------------------------


class TestProcessErrors:
    @pytest.mark.asyncio
    async def test_fn_exception_propagates_and_key_not_marked(self) -> None:
        """If the side-effect fn raises, the key MUST NOT be marked — Pub/Sub
        will redeliver, and the inbox should let the next attempt try again."""
        conn = _FakeConn()
        store = InboxIdempotencyStore(conn=conn)

        async def fn() -> None:
            raise RuntimeError("downstream broke")

        with pytest.raises(RuntimeError, match="downstream broke"):
            await store.process(
                key="event-1",
                ttl=_dt.timedelta(days=7),
                fn=fn,
            )
        # Key NOT marked.
        assert "event-1" not in conn.cur.rows_by_key

    @pytest.mark.asyncio
    async def test_empty_key_rejected(self) -> None:
        store = InboxIdempotencyStore(conn=_FakeConn())

        async def fn() -> None:
            pass

        with pytest.raises(ValueError, match="key"):
            await store.process(
                key="",
                ttl=_dt.timedelta(days=7),
                fn=fn,
            )

    @pytest.mark.asyncio
    async def test_non_positive_ttl_rejected(self) -> None:
        store = InboxIdempotencyStore(conn=_FakeConn())

        async def fn() -> None:
            pass

        with pytest.raises(ValueError, match="ttl"):
            await store.process(
                key="k",
                ttl=_dt.timedelta(seconds=0),
                fn=fn,
            )
