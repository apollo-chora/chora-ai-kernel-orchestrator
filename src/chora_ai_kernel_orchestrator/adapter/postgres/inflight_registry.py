"""InflightRegistry - the ADR-251 D4 in-flight job registry adapter.

Wraps ``ai_assist_inflight_jobs`` (migration 0055, chora_ai_kernel): the
durable-acceptance record that lets the orchestrator ACK a started event
BEFORE driving the graph, and lets the boot sweep resume every unfinished
job after a pod death or a cost-pause resume.

Connection discipline mirrors ``QGenCrewTerminalOutboxWriter``: the caller
passes the shared psycopg ``AsyncConnection`` and this adapter never owns
its lifecycle. ``register`` / ``mark_resumed`` / ``delete`` commit their own
single-statement transactions (durability before ACK; the shared-conn drain
lesson). ``delete_in_tx`` deliberately does NOT commit: the terminal outbox
writer executes it inside its own transaction so the registry removal and
the terminal event become durable in the SAME commit (ADR-251 D4 locked
semantics).

RLS (G2): the connection this adapter rides opts in EXPLICITLY to the
kernel-sweeper arm of the 0055 policy via ``chora.kernel_sweeper``, so
acceptance writes and the boot sweep span every tenant deliberately rather
than by an unset ``chora.tenant_id`` matching everything. Tenant-scoped
sessions elsewhere stay confined, and a session that opts in to neither now
sees NOTHING instead of EVERYTHING.

⚠ The previous note here cited "the 0003_outbox dispatcher precedent". That
citation does not hold: 0003_outbox keys its fail-open arm on
``app.current_tenant``, a DIFFERENT GUC, and describes itself as POC
scaffolding. See adapter/postgres/kernel_sweeper.py for the full reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_INSERT_SQL = """
INSERT INTO ai_assist_inflight_jobs (
    assist_id, tenant_id, author_gcid, started_payload_json,
    traceparent, tracestate
) VALUES (
    %(assist_id)s, %(tenant_id)s, %(author_gcid)s, %(started_payload_json)s,
    %(traceparent)s, %(tracestate)s
)
ON CONFLICT (assist_id) DO NOTHING
""".strip()

_DELETE_SQL = "DELETE FROM ai_assist_inflight_jobs WHERE assist_id = %(assist_id)s"

_SWEEP_SQL = """
SELECT assist_id, tenant_id, author_gcid, started_payload_json, resume_count
  FROM ai_assist_inflight_jobs
 ORDER BY accepted_at
""".strip()

_GET_SQL = """
SELECT assist_id, tenant_id, author_gcid, started_payload_json, resume_count
  FROM ai_assist_inflight_jobs
 WHERE assist_id = %(assist_id)s
""".strip()

_MARK_RESUMED_SQL = """
UPDATE ai_assist_inflight_jobs
   SET resume_count = resume_count + 1,
       last_resumed_at = now(),
       updated_at = now()
 WHERE assist_id = %(assist_id)s
""".strip()


class InflightRegistry:
    """Postgres adapter for the ADR-251 D4 in-flight job registry."""

    def __init__(self, *, conn: Any) -> None:
        self._conn = conn

    async def register(
        self,
        *,
        assist_id: str,
        tenant_id: str,
        author_gcid: str,
        started_event: dict[str, Any],
    ) -> None:
        """Durably record an accepted job BEFORE the subscriber ACKs.

        Idempotent (ON CONFLICT DO NOTHING) so a redelivered started event
        re-registers harmlessly. Commits its own single-statement
        transaction: the ACK that follows must never outrun the registry
        row. A failure raises to the caller, which NACKs (acceptance keeps
        the redelivery + DLQ pillar).
        """
        params = {
            "assist_id": assist_id,
            "tenant_id": tenant_id,
            "author_gcid": author_gcid,
            "started_payload_json": json.dumps(started_event),
            "traceparent": str(started_event.get("traceparent") or ""),
            "tracestate": str(started_event.get("tracestate") or ""),
        }
        async with self._conn.cursor() as cur:
            await cur.execute(_INSERT_SQL, params)
        await self._conn.commit()
        logger.info(
            "inflight_registry.registered",
            extra={"assist_id": assist_id, "tenant_id": tenant_id},
        )

    async def delete_in_tx(self, assist_id: str) -> None:
        """Execute the registry delete WITHOUT committing.

        The terminal outbox writer calls this between its outbox INSERT and
        its single commit, so the terminal event and the registry removal
        share one transaction boundary (ADR-251 D4). Callers other than the
        writer must use :meth:`delete`.
        """
        async with self._conn.cursor() as cur:
            await cur.execute(_DELETE_SQL, {"assist_id": assist_id})

    async def delete(self, assist_id: str) -> None:
        """Delete + commit (the sweep's stale-terminal cleanup path)."""
        async with self._conn.cursor() as cur:
            await cur.execute(_DELETE_SQL, {"assist_id": assist_id})
        await self._conn.commit()

    async def mark_resumed(self, assist_id: str) -> None:
        async with self._conn.cursor() as cur:
            await cur.execute(_MARK_RESUMED_SQL, {"assist_id": assist_id})
        await self._conn.commit()

    async def get(self, assist_id: str) -> dict[str, Any] | None:
        """ONE in-flight row by assist_id, or None once the job is no longer in
        flight (its terminal deleted the row in the same transaction).

        ADR-254 D5: a completion resumes a PARKED qgen job on any pod; the
        runner that settles it reads the job's started payload back from here
        (traceparent, requested_count, type_plan, regen spec, source_files), the
        durable copy that outlives the drive. Releases its read transaction
        with a commit (the shared-connection rule: never an empty-path rollback).
        """
        async with self._conn.cursor() as cur:
            await cur.execute(_GET_SQL, {"assist_id": assist_id})
            row = await cur.fetchone()
        await self._conn.commit()
        if row is None:
            return None
        return _row_to_dict(row)

    async def sweep(self) -> list[dict[str, Any]]:
        """All in-flight rows, oldest accepted first (the boot resume order).

        Runs in sweeper mode, opted in EXPLICITLY via ``chora.kernel_sweeper``
        on this connection (G2), so the rows span every tenant by design.

        Releases its read transaction with a COMMIT, exactly as ``get`` does.
        Without it the backend sat "idle in transaction" holding AccessShareLock
        on ai_assist_inflight_jobs for the WHOLE LIFE OF THE POD (observed live
        2026-08-23: 15m18s and counting on a pod 16m old), which blocks the
        ACCESS EXCLUSIVE that any policy DDL on this table needs — and a queued
        ACCESS EXCLUSIVE then blocks the kennel's own writes behind it. COMMIT
        and never ROLLBACK: an empty-path rollback on this shared connection
        destroys co-tenant writes made by the adapters sharing it.
        """
        async with self._conn.cursor() as cur:
            await cur.execute(_SWEEP_SQL)
            rows = await cur.fetchall()
        await self._conn.commit()
        return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: Any) -> dict[str, Any]:
    assist_id, tenant_id, author_gcid, payload, resume_count = row
    parsed = payload if isinstance(payload, dict) else json.loads(payload)
    return {
        "assist_id": str(assist_id),
        "tenant_id": str(tenant_id),
        "author_gcid": str(author_gcid),
        "started_payload": parsed,
        "resume_count": int(resume_count or 0),
    }


__all__ = ["InflightRegistry"]
