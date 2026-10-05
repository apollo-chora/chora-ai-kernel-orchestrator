"""RED: ADR-254 D5, the qgen boot sweep must not re-dispatch a PARKED job.

``QGenResumeSweeper`` treats "no terminal row, not driving in-process" as
"re-drive from the started payload". Once qgen parks on the bus, a per-chunk
parked job matches that predicate on every pod restart, and a re-drive is a
re-DISPATCH (a second paid generate call), not a resume. The completion that
arrives later resumes the thread; the sweep must leave parked threads alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.orchestrators.qgen_resume_sweeper import (
    QGenResumeSweeper,
)


class _Registry:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.resumed: list[str] = []
        self.deleted: list[str] = []

    async def sweep(self) -> list[dict[str, Any]]:
        return list(self.rows)

    async def mark_resumed(self, assist_id: str) -> None:
        self.resumed.append(assist_id)

    async def delete(self, assist_id: str) -> None:
        self.deleted.append(assist_id)


class _Acceptance:
    def __init__(self) -> None:
        self.spawned: list[str] = []

    def is_driving(self, assist_id: str) -> bool:
        return False

    def spawn_drive(self, assist_id: str, event: dict[str, Any]) -> None:
        self.spawned.append(assist_id)


class _NoTerminal:
    async def has_terminal(self, assist_id: str) -> bool:
        return False


class _Ledger:
    def __init__(self, parked: set[str]) -> None:
        self._parked = parked
        self.asked: list[str] = []

    async def thread_is_parked(self, thread_id: str) -> bool:
        self.asked.append(thread_id)
        return thread_id in self._parked


def _rows() -> list[dict[str, Any]]:
    return [
        {"assist_id": "parked-1", "started_payload": {"a": 1}, "tenant_id": "t"},
        {"assist_id": "free-2", "started_payload": {"a": 2}, "tenant_id": "t"},
    ]


@pytest.mark.asyncio
async def test_a_parked_job_is_skipped_and_tallied_not_re_driven() -> None:
    registry, acceptance = _Registry(_rows()), _Acceptance()
    ledger = _Ledger(parked={"parked-1"})
    sweeper = QGenResumeSweeper(
        registry=registry,
        acceptance=acceptance,
        terminal_index=_NoTerminal(),
        park_ledger=ledger,
    )

    tallies = await sweeper.sweep_and_resume()

    assert acceptance.spawned == ["free-2"]
    assert registry.resumed == ["free-2"]
    assert tallies["resumed"] == 1
    assert tallies["skipped_parked"] == 1
    assert "parked-1" in ledger.asked


@pytest.mark.asyncio
async def test_without_a_ledger_the_legacy_behaviour_is_unchanged() -> None:
    registry, acceptance = _Registry(_rows()), _Acceptance()
    sweeper = QGenResumeSweeper(registry=registry, acceptance=acceptance, terminal_index=_NoTerminal())

    tallies = await sweeper.sweep_and_resume()

    assert acceptance.spawned == ["parked-1", "free-2"]
    assert tallies["resumed"] == 2
    assert tallies.get("skipped_parked", 0) == 0
