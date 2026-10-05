"""QGenResumeSweeper - the ADR-251 D4 boot sweep (CHO-2398).

After a pod death or a cost-pause resume, redelivery can no longer recover an
ACCEPTED job (the message was ACKed at acceptance). The sweep is the only
recovery: every in-flight registry row either

  * already has its terminal event in the outbox (the pod died between the
    terminal commit and nothing at all - the same-tx delete makes this window
    near-empty, but near is not zero) -> the row is stale; delete it, never
    re-drive;
  * is currently driving in this process -> skip (double-resume idempotence);
  * is genuinely unfinished -> mark_resumed + re-drive from the started
    payload; the LangGraph checkpointer continues the thread from wherever
    it stopped (or from the beginning when the pod died pre-first-checkpoint,
    which is exactly what the durable started payload is for).

Runs at orchestrator startup BEFORE the streaming pull begins, so a resumed
drive never races a redelivered duplicate of its own started event; the
acceptance layer's in-process guard covers the residual overlap.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_TERMINAL_LOOKUP_SQL = """
SELECT 1
  FROM ai_kernel_outbox_events
 WHERE idempotency_key = ANY(%(keys)s)
 LIMIT 1
""".strip()


class _RegistryLike(Protocol):
    async def sweep(self) -> list[dict[str, Any]]: ...
    async def mark_resumed(self, assist_id: str) -> None: ...
    async def delete(self, assist_id: str) -> None: ...


class _AcceptanceLike(Protocol):
    def is_driving(self, assist_id: str) -> bool: ...
    def spawn_drive(self, assist_id: str, event: dict[str, Any]) -> None: ...


class _TerminalIndexLike(Protocol):
    async def has_terminal(self, assist_id: str) -> bool: ...


class _ParkLedgerLike(Protocol):
    async def thread_is_parked(self, thread_id: str) -> bool: ...


class OutboxTerminalIndex:
    """Answers "is this job's terminal already durable?" from the outbox.

    The terminal idempotency keys are deterministic
    (``ai_assist.completed.{assist_id}`` / ``ai_assist.refused.{assist_id}``,
    per QGenCrewTerminalOutboxWriter._write), so presence is one indexed
    lookup. Read-only; no commit.
    """

    def __init__(self, *, conn: Any) -> None:
        self._conn = conn

    async def has_terminal(self, assist_id: str) -> bool:
        keys = [
            f"ai_assist.completed.{assist_id}",
            f"ai_assist.refused.{assist_id}",
        ]
        async with self._conn.cursor() as cur:
            await cur.execute(_TERMINAL_LOOKUP_SQL, {"keys": keys})
            row = await cur.fetchone()
        return row is not None


class QGenResumeSweeper:
    """Resume-on-boot over the in-flight registry."""

    def __init__(
        self,
        *,
        registry: _RegistryLike,
        acceptance: _AcceptanceLike,
        terminal_index: _TerminalIndexLike,
        park_ledger: _ParkLedgerLike | None = None,
    ) -> None:
        self._registry = registry
        self._acceptance = acceptance
        self._terminal_index = terminal_index
        # ADR-254 D5: a job parked on an agent dispatch (ADR-253) is NOT
        # unfinished work for the sweep to re-drive. Re-driving it would
        # rebuild the started payload and dispatch AGAIN (a second paid model
        # call), while the completion that resumes the parked thread is still
        # on its way. The ledger says which threads are parked.
        self._park_ledger = park_ledger

    async def sweep_and_resume(self) -> dict[str, int]:
        tallies = {
            "resumed": 0,
            "cleaned_terminal": 0,
            "skipped_driving": 0,
            "skipped_parked": 0,
        }
        rows = await self._registry.sweep()
        for row in rows:
            assist_id = row["assist_id"]
            if await self._terminal_index.has_terminal(assist_id):
                await self._registry.delete(assist_id)
                tallies["cleaned_terminal"] += 1
                logger.info(
                    "qgen_resume_sweeper.stale_row_cleaned",
                    extra={"assist_id": assist_id},
                )
                continue
            if self._park_ledger is not None and await self._park_ledger.thread_is_parked(assist_id):
                tallies["skipped_parked"] += 1
                logger.info(
                    "qgen_resume_sweeper.skipped_parked",
                    extra={"assist_id": assist_id},
                )
                continue
            if self._acceptance.is_driving(assist_id):
                tallies["skipped_driving"] += 1
                continue
            await self._registry.mark_resumed(assist_id)
            self._acceptance.spawn_drive(assist_id, row["started_payload"])
            tallies["resumed"] += 1
            logger.info(
                "qgen_resume_sweeper.resumed",
                extra={
                    "assist_id": assist_id,
                    "resume_count": row.get("resume_count", 0) + 1,
                    "tenant_id": row.get("tenant_id", ""),
                },
            )
        if any(tallies.values()):
            logger.info("qgen_resume_sweeper.summary", extra=tallies)
        return tallies


__all__ = ["OutboxTerminalIndex", "QGenResumeSweeper"]
