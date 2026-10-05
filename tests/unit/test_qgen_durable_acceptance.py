"""CHO-2398 (ADR-251 D4): durable acceptance + background drive + resume-on-boot.

handle_started splits at a durable boundary: ACCEPTANCE (register in the
ai_assist_inflight_jobs registry, then return so the subscriber ACKs) and
DRIVE (the graph runs as a background task). Recovery is the boot sweep:
every registry row without a terminal outbox event resumes from its
checkpoint. The registry row is removed in the SAME transaction that makes
the terminal outbox row durable (the writer executes the delete before its
single commit).

Owner-ruled invariants pinned here:
  * acceptance failures still NACK (registry insert raising propagates);
  * a second concurrent resume is idempotent (in-flight guard);
  * a job whose terminal is already in the outbox is NOT re-driven;
  * a drive failure leaves the registry row for the next sweep (no silent
    drop, no fabricated terminal).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.postgres.inflight_registry import (
    InflightRegistry,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_publisher import (
    QGenCrewTerminalOutboxWriter,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_durable_acceptance import (
    DurableQGenAcceptance,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_resume_sweeper import (
    QGenResumeSweeper,
)

TENANT = "11111111-1111-7111-8111-111111111111"
GCID = "00000000-0000-7000-8000-000000001999"


def _event(assist_id: str = "job-1") -> dict[str, Any]:
    return {
        "assist_id": assist_id,
        "tenant_id": TENANT,
        "author_gcid": GCID,
        "prompt": "Generate a set",
        "traceparent": "00-abc-def-01",
    }


# -----------------------------------------------------------------------------
# Fakes - mirror the psycopg AsyncConnection surface used by the outbox writer
# -----------------------------------------------------------------------------


@dataclass
class _FakeCursor:
    executed: list[tuple[str, Any]] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self.rows)

    async def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


@dataclass
class _FakeConn:
    cur: _FakeCursor = field(default_factory=_FakeCursor)
    commits: list[int] = field(default_factory=list)

    def cursor(self) -> _FakeCursor:
        return self.cur

    async def execute(self, sql: str, params: Any = None) -> None:
        await self.cur.execute(sql, params)

    async def commit(self) -> None:
        self.commits.append(len(self.cur.executed))


# -----------------------------------------------------------------------------
# InflightRegistry SQL surface
# -----------------------------------------------------------------------------


async def test_register_inserts_on_conflict_do_nothing_and_commits() -> None:
    conn = _FakeConn()
    reg = InflightRegistry(conn=conn)
    await reg.register(
        assist_id="job-1",
        tenant_id=TENANT,
        author_gcid=GCID,
        started_event=_event(),
    )
    assert len(conn.cur.executed) == 1
    sql, params = conn.cur.executed[0]
    assert "ai_assist_inflight_jobs" in sql
    assert "ON CONFLICT" in sql and "DO NOTHING" in sql
    assert params["assist_id"] == "job-1"
    assert json.loads(params["started_payload_json"])["prompt"] == "Generate a set"
    assert conn.commits == [1], "register owns its commit (durable before ACK)"


async def test_delete_in_tx_executes_without_committing() -> None:
    conn = _FakeConn()
    reg = InflightRegistry(conn=conn)
    await reg.delete_in_tx("job-1")
    assert len(conn.cur.executed) == 1
    sql, params = conn.cur.executed[0]
    assert "DELETE FROM ai_assist_inflight_jobs" in sql
    assert conn.commits == [], "delete_in_tx must leave the commit to the caller"


# -----------------------------------------------------------------------------
# Writer: terminal publish deletes the registry row in the SAME transaction
# -----------------------------------------------------------------------------


def _completed_kwargs(assist_id: str = "job-1") -> dict[str, Any]:
    return {
        "assist_id": assist_id,
        "tenant_id": TENANT,
        "author_gcid": GCID,
        "candidate_payload_json": "[]",
        "pipeline_trace_json": "[]",
        "quality_warning": False,
        "attempt_count": 1,
        "critic_notes": "",
        "mana_charged": 0,
    }


async def test_publish_completed_deletes_registry_row_before_the_single_commit() -> None:
    conn = _FakeConn()
    reg = InflightRegistry(conn=conn)
    writer = QGenCrewTerminalOutboxWriter(
        conn=conn,
        source_project="chora-489812",
        inflight_registry=reg,
    )
    await writer.publish_completed(**_completed_kwargs())
    sqls = [sql for sql, _ in conn.cur.executed]
    assert any("ai_kernel_outbox_events" in s for s in sqls)
    assert any("DELETE FROM ai_assist_inflight_jobs" in s for s in sqls)
    assert sqls.index(next(s for s in sqls if "ai_kernel_outbox_events" in s)) < sqls.index(
        next(s for s in sqls if "DELETE FROM ai_assist_inflight_jobs" in s)
    )
    assert conn.commits == [2], (
        "exactly ONE commit covering outbox insert + registry delete (the same-transaction boundary of ADR-251 D4)"
    )


async def test_publish_completed_without_registry_is_byte_identical() -> None:
    conn = _FakeConn()
    writer = QGenCrewTerminalOutboxWriter(conn=conn, source_project="chora-489812")
    await writer.publish_completed(**_completed_kwargs())
    sqls = [sql for sql, _ in conn.cur.executed]
    assert not any("ai_assist_inflight_jobs" in s for s in sqls)
    assert conn.commits == [1]


# -----------------------------------------------------------------------------
# DurableQGenAcceptance: ack-fast + background drive + fail-loud acceptance
# -----------------------------------------------------------------------------


class _SlowRouter:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[str] = []
        self.gate = asyncio.Event()

    async def handle_started(self, event: dict[str, Any]) -> None:
        self.started.append(event["assist_id"])
        await self.gate.wait()
        self.finished.append(event["assist_id"])


class _FakeRegistry:
    def __init__(self, fail: bool = False) -> None:
        self.registered: list[str] = []
        self.fail = fail

    async def register(self, **kw: Any) -> None:
        if self.fail:
            raise RuntimeError("registry down")
        self.registered.append(kw["assist_id"])


async def test_handle_started_returns_before_the_drive_finishes() -> None:
    router = _SlowRouter()
    reg = _FakeRegistry()
    acc = DurableQGenAcceptance(registry=reg, router=router)

    await asyncio.wait_for(acc.handle_started(_event()), timeout=1.0)
    assert reg.registered == ["job-1"], "registered durably before returning"
    assert acc.is_driving("job-1"), "drive continues as a background task"
    assert router.finished == []

    router.gate.set()
    await acc.wait_idle(timeout=2.0)
    assert router.finished == ["job-1"]
    assert not acc.is_driving("job-1")


async def test_registry_failure_raises_so_the_subscriber_nacks() -> None:
    router = _SlowRouter()
    acc = DurableQGenAcceptance(registry=_FakeRegistry(fail=True), router=router)
    with pytest.raises(RuntimeError, match="registry down"):
        await acc.handle_started(_event())
    assert router.started == [], "no drive on failed acceptance"


async def test_drive_failure_is_loud_and_leaves_resume_to_the_sweep() -> None:
    class _BoomRouter:
        async def handle_started(self, event: dict[str, Any]) -> None:
            raise RuntimeError("mid-graph death")

    acc = DurableQGenAcceptance(registry=_FakeRegistry(), router=_BoomRouter())
    await acc.handle_started(_event())
    await acc.wait_idle(timeout=2.0)
    assert not acc.is_driving("job-1")
    assert acc.failed_drives.get("job-1"), "the failure is recorded loudly"


async def test_duplicate_start_while_driving_does_not_double_drive() -> None:
    router = _SlowRouter()
    acc = DurableQGenAcceptance(registry=_FakeRegistry(), router=router)
    await acc.handle_started(_event())
    await asyncio.sleep(0)  # let the spawned drive task reach the gate
    await acc.handle_started(_event())
    await asyncio.sleep(0)
    assert router.started == ["job-1"], "second start while driving is a no-op"
    router.gate.set()
    await acc.wait_idle(timeout=2.0)


# -----------------------------------------------------------------------------
# QGenResumeSweeper: orphan resumes; terminal-in-outbox cleans; driving skips
# -----------------------------------------------------------------------------


class _SweepRegistry:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.deleted: list[str] = []
        self.resumed_marks: list[str] = []

    async def sweep(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["assist_id"] not in self.deleted]

    async def mark_resumed(self, assist_id: str) -> None:
        self.resumed_marks.append(assist_id)

    async def delete(self, assist_id: str) -> None:
        self.deleted.append(assist_id)


class _TerminalIndex:
    """Duck-typed outbox terminal lookup."""

    def __init__(self, terminal_ids: set[str]) -> None:
        self.terminal_ids = terminal_ids

    async def has_terminal(self, assist_id: str) -> bool:
        return assist_id in self.terminal_ids


def _row(assist_id: str) -> dict[str, Any]:
    return {
        "assist_id": assist_id,
        "tenant_id": TENANT,
        "started_payload": _event(assist_id),
        "resume_count": 0,
    }


async def test_orphan_row_resumes_to_terminal() -> None:
    router = _SlowRouter()
    router.gate.set()
    acc = DurableQGenAcceptance(registry=_FakeRegistry(), router=router)
    reg = _SweepRegistry([_row("job-a")])
    sweeper = QGenResumeSweeper(
        registry=reg,
        acceptance=acc,
        terminal_index=_TerminalIndex(set()),
    )
    tallies = await sweeper.sweep_and_resume()
    await acc.wait_idle(timeout=2.0)
    assert router.finished == ["job-a"]
    assert reg.resumed_marks == ["job-a"]
    assert tallies["resumed"] == 1


async def test_terminal_already_in_outbox_cleans_without_redriving() -> None:
    router = _SlowRouter()
    acc = DurableQGenAcceptance(registry=_FakeRegistry(), router=router)
    reg = _SweepRegistry([_row("job-done")])
    sweeper = QGenResumeSweeper(
        registry=reg,
        acceptance=acc,
        terminal_index=_TerminalIndex({"job-done"}),
    )
    tallies = await sweeper.sweep_and_resume()
    assert router.started == [], "terminal job must NOT be re-driven"
    assert reg.deleted == ["job-done"], "stale registry row cleaned"
    assert tallies["cleaned_terminal"] == 1


async def test_double_resume_is_idempotent_while_driving() -> None:
    router = _SlowRouter()
    acc = DurableQGenAcceptance(registry=_FakeRegistry(), router=router)
    reg = _SweepRegistry([_row("job-a")])
    sweeper = QGenResumeSweeper(
        registry=reg,
        acceptance=acc,
        terminal_index=_TerminalIndex(set()),
    )
    await sweeper.sweep_and_resume()
    await asyncio.sleep(0)  # let the resumed drive task reach the gate
    tallies2 = await sweeper.sweep_and_resume()
    assert router.started == ["job-a"], "second sweep must not double-drive"
    assert tallies2["skipped_driving"] == 1
    router.gate.set()
    await acc.wait_idle(timeout=2.0)


async def test_registry_sweep_parses_rows_and_orders_by_sql() -> None:
    conn = _FakeConn()
    conn.cur.rows = [
        ("job-a", TENANT, GCID, json.dumps(_event("job-a")), 2),
        ("job-b", TENANT, GCID, {"assist_id": "job-b"}, 0),
    ]
    reg = InflightRegistry(conn=conn)
    rows = await reg.sweep()
    sql, _ = conn.cur.executed[0]
    assert "ORDER BY accepted_at" in sql
    assert rows[0]["assist_id"] == "job-a"
    assert rows[0]["started_payload"]["assist_id"] == "job-a"
    assert rows[0]["resume_count"] == 2
    assert rows[1]["started_payload"] == {"assist_id": "job-b"}, "native-dict JSONB rows parse without a second decode"
    # The SQL is read-only; the TRANSACTION is not free. On this non-autocommit
    # shared connection a read still opens a transaction, and leaving it open
    # parks the backend "idle in transaction" holding AccessShareLock for the
    # life of the pod (observed live 2026-08-23: 15m18s on a 16m-old pod), which
    # blocks the ACCESS EXCLUSIVE that policy DDL on this table needs. Release
    # with COMMIT, never ROLLBACK (an empty-path rollback on a shared connection
    # destroys co-tenant writes). This assertion previously read
    # `conn.commits == []` and pinned the defect as intended behaviour.
    assert conn.commits == [1], "sweep must RELEASE its read transaction with a commit"


async def test_registry_delete_and_mark_resumed_commit_their_writes() -> None:
    conn = _FakeConn()
    reg = InflightRegistry(conn=conn)
    await reg.delete("job-a")
    await reg.mark_resumed("job-b")
    sqls = [sql for sql, _ in conn.cur.executed]
    assert any("DELETE FROM ai_assist_inflight_jobs" in s for s in sqls)
    assert any("resume_count = resume_count + 1" in s for s in sqls)
    assert conn.commits == [1, 2]
