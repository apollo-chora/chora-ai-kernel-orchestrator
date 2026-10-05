"""Park ledger adapters over ``ai_kernel_agent_dispatch_parks`` (migration 0056).

* ``ParkLedgerWriter``: called by ``TransactionalDispatchSaver`` INSIDE its
  transaction, on that saver's connection. Never commits. Idempotent on the
  dispatch key because LangGraph re-executes a node from the top on resume and
  the rebuilt request must collide with the queued row rather than write twice.
* ``ParkLedgerStore``: the completion router's and the reaper's view. Runs on
  its own autocommit connection (the OE pattern: a lane that commits nothing
  on a non-autocommit connection acks-then-rolls-back, see
  ``[[reusable_gotcha_an_ack_is_not_a_commit_lane_without_committer]]``).
  Every write commits explicitly; every read releases with COMMIT, never
  ROLLBACK, because a rollback on a shared connection destroys co-tenant
  writes (``store.py`` carries the 2026-08-14 incident).

Tenant GUC (G2): these queries run in sweeper mode, opted in EXPLICITLY via
``chora.kernel_sweeper`` on the connection, because the reaper scan, the
completion router's key lookups and the backfill span every tenant by design.
They no longer rely on an unset ``chora.tenant_id`` matching every row: under
the tightened 0056 policy a session that opts in to neither tenant nor sweeper
sees NOTHING rather than EVERYTHING.

⚠ The previous note here claimed this was "exactly like the outbox dispatcher".
It is not: 0003_outbox keys its fail-open arm on ``app.current_tenant``, a
DIFFERENT GUC that a session setting ``chora.tenant_id`` never touches, and
describes itself as POC scaffolding. See adapter/postgres/kernel_sweeper.py.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from collections.abc import Mapping
from typing import Any

from chora_ai_kernel_orchestrator.domain.agent_dispatch.park import (
    ParkRecord,
    ParkState,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.reaper import ReaperArm

logger = logging.getLogger(__name__)

PARK_LEDGER_TABLE = "ai_kernel_agent_dispatch_parks"

PARK_COLUMNS: tuple[str, ...] = (
    "idempotency_key",
    "workflow_id",
    "thread_id",
    "crew",
    "tenant_id",
    "gcid",
    "agent_role",
    "request_topic",
    "completion_topic",
    "traceparent",
    "tracestate",
    "parked_at",
    "deadline_at",
    "state",
    "settled_at",
    "settled_by",
    "late_completion_at",
)

_SELECT_COLUMNS = ", ".join(PARK_COLUMNS)
_SELECT_COLUMNS_P = ", ".join(f"p.{c}" for c in PARK_COLUMNS)

_INSERT_SQL = f"""
INSERT INTO {PARK_LEDGER_TABLE} (
    idempotency_key, workflow_id, thread_id, crew, tenant_id, gcid,
    agent_role, request_topic, completion_topic, traceparent, tracestate,
    parked_at, deadline_at
) VALUES (
    %(idempotency_key)s, %(workflow_id)s, %(thread_id)s, %(crew)s, %(tenant_id)s, %(gcid)s,
    %(agent_role)s, %(request_topic)s, %(completion_topic)s, %(traceparent)s, %(tracestate)s,
    %(parked_at)s, %(deadline_at)s
)
ON CONFLICT (idempotency_key) DO NOTHING
"""

_GET_SQL = f"""
SELECT {_SELECT_COLUMNS}
FROM {PARK_LEDGER_TABLE}
WHERE idempotency_key = %(key)s
"""

_MARK_COMPLETED_SQL = f"""
UPDATE {PARK_LEDGER_TABLE}
SET state = 'completed', settled_by = 'completion', settled_at = now()
WHERE idempotency_key = %(key)s AND state = 'parked'
"""

_MARK_REAPED_SQL = f"""
UPDATE {PARK_LEDGER_TABLE}
SET state = 'reaped', settled_by = %(settled_by)s, settled_at = now()
WHERE idempotency_key = %(key)s AND state = 'parked'
"""

_LATE_COMPLETION_SQL = f"""
UPDATE {PARK_LEDGER_TABLE}
SET late_completion_at = now()
WHERE idempotency_key = %(key)s AND state <> 'parked'
"""

_FETCH_EXPIRED_SQL = f"""
SELECT {_SELECT_COLUMNS}
FROM {PARK_LEDGER_TABLE}
WHERE state = 'parked' AND deadline_at < now()
ORDER BY deadline_at ASC
LIMIT %(limit)s
"""

_FETCH_OUTBOX_DEAD_LETTERED_SQL = f"""
SELECT {_SELECT_COLUMNS_P}
FROM {PARK_LEDGER_TABLE} p
JOIN ai_kernel_outbox_events o ON o.idempotency_key = p.idempotency_key
WHERE p.state = 'parked' AND o.status = 'deadlettered'
ORDER BY p.parked_at ASC
LIMIT %(limit)s
"""

_COUNT_PARKED_SQL = f"""
SELECT count(*)
FROM {PARK_LEDGER_TABLE}
WHERE tenant_id = %(tenant_id)s AND agent_role = %(agent_role)s AND state = 'parked'
"""

_THREAD_PARKED_SQL = f"""
SELECT 1
FROM {PARK_LEDGER_TABLE}
WHERE thread_id = %(thread_id)s AND state = 'parked'
LIMIT 1
"""

# Startup backfill: a published dispatch request whose completion was never
# consumed (the completion subscriber marks '<key>.completed' in the inbox only
# after a successful resume) and that has no ledger row yet is a park the
# kennel is still holding from before migration 0056 (the OE requests parked
# during the 2026-08-21 outage). ON CONFLICT on the insert makes re-running it
# on every boot safe.
_BACKFILL_CANDIDATES_SQL = f"""
SELECT o.idempotency_key, o.workflow_id::TEXT, o.tenant_id::TEXT, o.gcid::TEXT,
       o.topic, convert_from(o.payload, 'UTF8') AS body, o.occurred_at
FROM ai_kernel_outbox_events o
WHERE o.event_type LIKE 'ai_kernel.agent_dispatch.%_requested'
  AND o.status = 'published'
  AND NOT EXISTS (SELECT 1 FROM idempotency_keys k WHERE k.key = o.idempotency_key || '.completed')
  AND NOT EXISTS (SELECT 1 FROM {PARK_LEDGER_TABLE} p WHERE p.idempotency_key = o.idempotency_key)
ORDER BY o.occurred_at ASC
"""


class ParkLedgerWriter:
    """INSERTs the park row on the caller's open transaction (the saver's)."""

    def __init__(self, *, conn: Any, crew: str) -> None:
        crew_name = (crew or "").strip()
        if not crew_name:
            raise ValueError("ParkLedgerWriter: crew is required (which runner resumes this park)")
        self._conn = conn
        self._crew = crew_name

    @property
    def crew(self) -> str:
        return self._crew

    async def queue_park(self, request: dict[str, Any], *, deadline_at: _dt.datetime) -> str:
        """Queue one park for a request built by ``build_dispatch_request``.

        Does NOT commit: the saver owns the boundary, so the ledger row, the
        outbox row and the LangGraph park commit together or not at all.
        """
        body = request["body"]
        envelope = request["envelope"]
        thread_id = str(body.get("thread_id") or "").strip()
        if not thread_id:
            raise ValueError("park ledger: request body carries no thread_id")
        parked_at = _dt.datetime.fromisoformat(str(envelope["occurred_at"]))
        if deadline_at.tzinfo is None or deadline_at.utcoffset() is None:
            raise ValueError("park ledger: deadline_at must be timezone-aware")
        if deadline_at <= parked_at:
            raise ValueError(
                f"park ledger: deadline_at {deadline_at.isoformat()} is not after parked_at {parked_at.isoformat()}"
            )
        params = {
            "idempotency_key": request["idempotency_key"],
            "workflow_id": request["workflow_id"],
            "thread_id": thread_id,
            "crew": self._crew,
            "tenant_id": request["tenant_id"],
            "gcid": request.get("gcid", ""),
            "agent_role": body["agent_role"],
            "request_topic": request["topic"],
            "completion_topic": body["reply_topic"],
            "traceparent": str(body.get("traceparent") or ""),
            "tracestate": str(body.get("tracestate") or ""),
            "parked_at": parked_at,
            "deadline_at": deadline_at,
        }
        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        logger.info(
            "park_ledger.queued",
            extra={
                "idempotency_key": params["idempotency_key"],
                "thread_id": thread_id,
                "agent_role": params["agent_role"],
                "crew": self._crew,
                "deadline_at": deadline_at.isoformat(),
            },
        )
        return str(params["idempotency_key"])


class ParkLedgerStore:
    """Reads and settles park rows on its own autocommit connection."""

    def __init__(self, *, conn: Any) -> None:
        self._conn = conn

    async def get(self, idempotency_key: str) -> ParkRecord | None:
        key = _require(idempotency_key, "idempotency_key")
        async with self._conn.cursor() as cur:
            await cur.execute(_GET_SQL, {"key": key})
            row = await cur.fetchone()
        await self._conn.commit()
        return _to_record(row) if row is not None else None

    async def mark_completed(self, idempotency_key: str) -> bool:
        key = _require(idempotency_key, "idempotency_key")
        async with self._conn.cursor() as cur:
            await cur.execute(_MARK_COMPLETED_SQL, {"key": key})
            changed = _rowcount(cur)
        await self._conn.commit()
        return changed > 0

    async def mark_reaped(self, idempotency_key: str, *, arm: ReaperArm) -> bool:
        key = _require(idempotency_key, "idempotency_key")
        settled_by = f"reaper:{ReaperArm(arm).value}"
        async with self._conn.cursor() as cur:
            await cur.execute(_MARK_REAPED_SQL, {"key": key, "settled_by": settled_by})
            changed = _rowcount(cur)
        await self._conn.commit()
        return changed > 0

    async def record_late_completion(self, idempotency_key: str) -> None:
        key = _require(idempotency_key, "idempotency_key")
        async with self._conn.cursor() as cur:
            await cur.execute(_LATE_COMPLETION_SQL, {"key": key})
        await self._conn.commit()

    async def fetch_expired(self, *, limit: int) -> list[ParkRecord]:
        async with self._conn.cursor() as cur:
            await cur.execute(_FETCH_EXPIRED_SQL, {"limit": _limit(limit)})
            rows = await cur.fetchall()
        await self._conn.commit()
        return [_to_record(r) for r in rows]

    async def fetch_outbox_dead_lettered(self, *, limit: int) -> list[ParkRecord]:
        async with self._conn.cursor() as cur:
            await cur.execute(_FETCH_OUTBOX_DEAD_LETTERED_SQL, {"limit": _limit(limit)})
            rows = await cur.fetchall()
        await self._conn.commit()
        return [_to_record(r) for r in rows]

    async def count_parked(self, *, tenant_id: str, agent_role: str) -> int:
        params = {
            "tenant_id": _require(tenant_id, "tenant_id"),
            "agent_role": _require(agent_role, "agent_role"),
        }
        async with self._conn.cursor() as cur:
            await cur.execute(_COUNT_PARKED_SQL, params)
            row = await cur.fetchone()
        await self._conn.commit()
        return int(row[0]) if row else 0

    async def thread_is_parked(self, thread_id: str) -> bool:
        params = {"thread_id": _require(thread_id, "thread_id")}
        async with self._conn.cursor() as cur:
            await cur.execute(_THREAD_PARKED_SQL, params)
            row = await cur.fetchone()
        await self._conn.commit()
        return row is not None

    async def backfill_from_outbox(
        self,
        *,
        crew_for_role: Mapping[str, str],
        deadline_policy: Any,
    ) -> int:
        """Insert a ledger row for every park made before the ledger existed.

        ``crew_for_role`` maps a dispatch role to the crew whose runner resumes
        it (the same binding the completion router holds). A candidate whose
        role has no crew, or whose body carries no thread, is logged LOUD and
        skipped: the run is still parked and still reachable by its completion,
        it just cannot be reaped by deadline, which is the state it was in
        before this code existed. Returns the number of rows inserted.
        """
        async with self._conn.cursor() as cur:
            await cur.execute(_BACKFILL_CANDIDATES_SQL)
            candidates = await cur.fetchall()
        await self._conn.commit()

        inserted = 0
        for key, workflow_id, tenant_id, gcid, topic, body_text, occurred_at in candidates:
            try:
                body = json.loads(body_text or "{}")
                if not isinstance(body, dict):
                    raise ValueError("request body is not a JSON object")
            except (TypeError, ValueError) as exc:
                logger.error(
                    "park_ledger.backfill.undecodable_request",
                    extra={"idempotency_key": key, "err": str(exc)},
                )
                continue
            role = str(body.get("agent_role") or "").strip()
            thread_id = str(body.get("thread_id") or "").strip()
            crew = (crew_for_role.get(role) or "").strip() if role else ""
            if not crew:
                logger.error(
                    "park_ledger.backfill.no_crew_for_role",
                    extra={"idempotency_key": key, "agent_role": role},
                )
                continue
            if not thread_id:
                logger.error(
                    "park_ledger.backfill.request_without_thread",
                    extra={"idempotency_key": key, "agent_role": role},
                )
                continue
            parked_at = occurred_at
            params = {
                "idempotency_key": key,
                "workflow_id": workflow_id,
                "thread_id": thread_id,
                "crew": crew,
                "tenant_id": tenant_id,
                "gcid": gcid,
                "agent_role": role,
                "request_topic": topic,
                "completion_topic": str(body.get("reply_topic") or ""),
                "traceparent": str(body.get("traceparent") or ""),
                "tracestate": str(body.get("tracestate") or ""),
                "parked_at": parked_at,
                "deadline_at": deadline_policy.deadline_for(role, parked_at=parked_at),
            }
            async with self._conn.cursor() as cur:
                await cur.execute(_INSERT_SQL, params)
                if _rowcount(cur) > 0:
                    inserted += 1
            await self._conn.commit()
            logger.warning(
                "park_ledger.backfill.inserted",
                extra={
                    "idempotency_key": key,
                    "agent_role": role,
                    "crew": crew,
                    "thread_id": thread_id,
                    "parked_at": parked_at.isoformat() if hasattr(parked_at, "isoformat") else str(parked_at),
                    "deadline_at": params["deadline_at"].isoformat(),
                },
            )
        if inserted:
            logger.warning("park_ledger.backfill.summary", extra={"inserted": inserted})
        return inserted


def _to_record(row: Any) -> ParkRecord:
    values = dict(zip(PARK_COLUMNS, row, strict=True))
    values["state"] = ParkState(str(values["state"]))
    # psycopg hands the UUID columns back as uuid.UUID objects. Everything that
    # leaves this record travels as JSON (the run_failed body, the synthesized
    # completion), so coerce to str here, once. Measured live 2026-08-22: the
    # first reap marked the row and then lost its terminal to
    # "Object of type UUID is not JSON serializable".
    for column in (
        "idempotency_key",
        "workflow_id",
        "thread_id",
        "crew",
        "tenant_id",
        "gcid",
        "agent_role",
        "request_topic",
        "completion_topic",
    ):
        values[column] = "" if values.get(column) is None else str(values[column])
    values["tracestate"] = str(values.get("tracestate") or "")
    values["traceparent"] = str(values.get("traceparent") or "")
    values["settled_by"] = str(values.get("settled_by") or "")
    return ParkRecord(**values)


def _rowcount(cur: Any) -> int:
    count = getattr(cur, "rowcount", -1)
    try:
        return int(count)
    except (TypeError, ValueError):
        return 0


def _limit(limit: int) -> int:
    if int(limit) < 1:
        raise ValueError("park ledger: limit must be >= 1")
    return int(limit)


def _require(value: str, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"park ledger: {field} is required")
    return text


__all__ = ["PARK_COLUMNS", "PARK_LEDGER_TABLE", "ParkLedgerStore", "ParkLedgerWriter"]
