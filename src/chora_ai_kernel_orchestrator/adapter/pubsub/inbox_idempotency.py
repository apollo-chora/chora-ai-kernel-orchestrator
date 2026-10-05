"""InboxIdempotencyStore — Python mirror of libs/chora-go-common/idempotent.

Provides ``process(key, ttl, fn)``: dedup-then-invoke for Pub/Sub
subscribers per [[feedback-d6-resilience-first-class]] D6 P2
idempotency contract. Mirrors the chora-creation Go subscriber's
``inbox.Process(ctx, key, ttl, fn)`` (used in
``services/chora-creation/internal/adapter/events/ai_assist_subscriber.go``).

Schema: ``idempotency_keys`` table per migration
``0004_inbox_idempotency.sql`` in this service.

Contract:
1. If key exists AND ttl_at > now() → fn is NOT invoked, return False.
2. Else, invoke fn:
   - If fn raises → propagate; key NOT marked (so Pub/Sub redelivery can retry).
   - If fn returns → mark key (UPSERT row with ttl_at = now() + ttl).
3. Returns True iff fn ran, False on dedupe-hit.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

_SAFE_TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


class InboxIdempotencyStore:
    """Postgres-backed inbox dedupe store.

    Connection is caller-owned. The store does NOT commit explicitly —
    the caller's transaction boundary controls commit (the typical
    pattern: each Pub/Sub message gets its own auto-commit per
    handle_message).
    """

    def __init__(
        self,
        *,
        conn: Any,
        table: str = "idempotency_keys",
    ) -> None:
        if not _SAFE_TABLE_NAME.match(table):
            raise ValueError(f"InboxIdempotencyStore: invalid table name {table!r} — must match ^[a-z_][a-z0-9_]*$")
        self._conn = conn
        self._table = table

    @property
    def table(self) -> str:
        return self._table

    async def process(
        self,
        *,
        key: str,
        ttl: _dt.timedelta,
        fn: Callable[[], Awaitable[None]],
    ) -> bool:
        """Dedup-then-invoke.

        Returns:
            True iff fn was invoked (and completed without raising).
            False iff the key was already seen (within TTL) — fn skipped.

        Raises:
            Whatever fn raises (key NOT marked on raise).
            ValueError on empty key or non-positive ttl.
        """
        if not (key or "").strip():
            raise ValueError("InboxIdempotencyStore.process: key required")
        if ttl <= _dt.timedelta(0):
            raise ValueError("InboxIdempotencyStore.process: ttl must be > 0")

        async with self._conn.cursor() as cur:
            seen = await self._seen(cur, key)
            if seen:
                logger.info(
                    "inbox_idempotency.dedupe_hit",
                    extra={"key": key, "table": self._table},
                )
                return False

        # fn runs OUTSIDE the cursor context — keeps long-running side
        # effects (graph ainvoke + outbox INSERT in another cursor) from
        # holding row locks unnecessarily.
        await fn()

        # Mark after success. (If a duplicate delivery races with the
        # first attempt's fn, both attempts will run fn — that's expected
        # at-least-once semantics; downstream idempotency on the outbox
        # writer's `ON CONFLICT (idempotency_key) DO NOTHING` collapses
        # the duplicate side effect.)
        now = _dt.datetime.now(tz=_dt.UTC)
        ttl_at = now + ttl
        async with self._conn.cursor() as cur:
            await self._mark(cur, key, processed_at=now, ttl_at=ttl_at)
        return True

    # ---- Internal queries -------------------------------------------------

    async def _seen(self, cur: Any, key: str) -> bool:
        sql = (
            f"SELECT key FROM {self._table} "  # noqa: S608 — table validated in __init__
            f"WHERE key = %(key)s AND ttl_at > now()"
        )
        await cur.execute(sql, {"key": key})
        row = await cur.fetchone()
        return row is not None

    async def _mark(
        self,
        cur: Any,
        key: str,
        *,
        processed_at: _dt.datetime,
        ttl_at: _dt.datetime,
    ) -> None:
        sql = (
            f"INSERT INTO {self._table} (key, processed_at, ttl_at) "  # noqa: S608 — table validated
            f"VALUES (%(key)s, %(processed_at)s, %(ttl_at)s) "
            f"ON CONFLICT (key) DO UPDATE SET ttl_at = EXCLUDED.ttl_at"
        )
        await cur.execute(
            sql,
            {"key": key, "processed_at": processed_at, "ttl_at": ttl_at},
        )


__all__ = ["InboxIdempotencyStore"]
