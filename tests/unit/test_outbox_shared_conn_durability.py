"""RED→GREEN: the shared-connection outbox lane must never DESTROY a write.

Live incident 2026-08-14 (job ``fd80f4e9-db27-434b-9f61-685a8a2c47f6``): the
qgen graph ran to ``publish_completed``, the terminal writer logged
``qgen_crew_outbox.queued`` at 02:07:06.910, the subscriber ACKed the Pub/Sub
message - and NO ``creation.ai_assist.completed`` row ever existed in
``ai_kernel_outbox_events``. The job sat at ``status=running`` in
``chora_creation`` forever and the FE polled it forever.

Mechanism - one psycopg ``AsyncConnection`` with **autocommit OFF** is shared by
(``qgen_crew_wiring.build_qgen_crew`` docstring):

  * ``QGenCrewTerminalOutboxWriter``   - INSERT the terminal event
  * ``AgentDecisionLogOutboxWriter``   - INSERT the accountability row
  * ``HITLDecisionOutboxWriter``       - INSERT the oversight gate row
  * ``InboxIdempotencyStore``          - inbox dedupe
  * ``PostgresOutboxStore``            - the DISPATCHER, a *concurrent*
    ``asyncio`` task draining every 2s (``main.py`` ``create_task``)

``fetch_pending`` released its read transaction with ``rollback()`` on the empty
path. psycopg serialises individual statements but NOT a multi-statement
transaction across tasks, so this interleaving loses the event:

    dispatcher : SELECT ... FOR UPDATE SKIP LOCKED  -> 0 rows
    writer     : INSERT INTO ai_kernel_outbox_events ...   (uncommitted)
    dispatcher : ROLLBACK                            <-- INSERT destroyed
    runner     : log set_completed; subscriber ACKs   <-- never retried

Two independent invariants close it - either alone would have saved the event,
and both are pinned here because the lane must not depend on someone else's
commit ([[reusable_gotcha_an_ack_is_not_a_commit_lane_without_committer]]):

  1. A writer COMMITS its own INSERT. "Queued" must mean durable.
  2. The dispatcher's empty-path release must be non-destructive (COMMIT, not
     ROLLBACK). Zero rows matched, so zero locks were taken and there is
     nothing of the dispatcher's own to undo.
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_decision_outbox_writer import (
    AgentDecisionLogOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.hitl_decision_outbox_writer import (
    HITLDecisionOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_publisher import (
    QGenCrewTerminalOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.store import PostgresOutboxStore

# -----------------------------------------------------------------------------
# A connection double with REAL transaction semantics: statements land in an
# uncommitted buffer, commit() promotes them to durable, rollback() DISCARDS
# them. That is the only way a test can see the defect - a spy that merely
# counts commit/rollback calls cannot tell "released the txn" from "destroyed
# the pending write" ([[reusable_gotcha_the_harness_cannot_see_the_failure_mode]]).
# -----------------------------------------------------------------------------


class _TxnCursor:
    def __init__(self, conn: _TxnConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _TxnCursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, sql: str, params: Any = None) -> None:
        self._conn.executed.append(sql)
        if sql.lstrip().upper().startswith("INSERT"):
            self._conn.uncommitted.append((sql, params))

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self._conn.select_rows


class _TxnConn:
    """psycopg-shaped async connection with autocommit OFF semantics."""

    def __init__(self, select_rows: list[tuple[Any, ...]] | None = None) -> None:
        self.select_rows = list(select_rows or [])
        self.uncommitted: list[tuple[str, Any]] = []
        self.durable: list[tuple[str, Any]] = []
        self.executed: list[str] = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _TxnCursor:
        return _TxnCursor(self)

    async def commit(self) -> None:
        self.commits += 1
        self.durable.extend(self.uncommitted)
        self.uncommitted.clear()

    async def rollback(self) -> None:
        self.rollbacks += 1
        self.uncommitted.clear()


def _completed_kwargs(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "assist_id": "fd80f4e9-db27-434b-9f61-685a8a2c47f6",
        "tenant_id": "11111111-1111-7111-8111-111111111111",
        "author_gcid": "00000000-0000-7000-8000-000000001999",
        "candidate_payload_json": '{"candidates": []}',
        "pipeline_trace_json": "[]",
        "quality_warning": False,
        "attempt_count": 3,
        "critic_notes": "",
        "mana_charged": 0,
        "generated_count": 3,
    }
    base.update(over)
    return base


# -----------------------------------------------------------------------------
# Invariant 1 - a writer commits its own INSERT.
# -----------------------------------------------------------------------------


async def test_completed_write_is_durable_without_an_external_committer() -> None:
    """``publish_completed`` returning MUST mean the row is COMMITTED.

    The live loss: the writer logged "queued" while the row was still only
    uncommitted, and nothing of its own ever committed it.
    """
    conn = _TxnConn()
    writer = QGenCrewTerminalOutboxWriter(conn=conn, source_project="chora-489812")

    await writer.publish_completed(**_completed_kwargs())

    assert conn.uncommitted == [], (
        "publish_completed returned with the INSERT still uncommitted - any "
        "concurrent rollback on the SHARED connection destroys the terminal event"
    )
    assert len(conn.durable) == 1


async def test_refused_and_progress_writes_are_durable() -> None:
    """The other two terminal shapes carry the same guarantee."""
    conn = _TxnConn()
    writer = QGenCrewTerminalOutboxWriter(conn=conn, source_project="chora-489812")

    await writer.publish_refused(
        assist_id="a1",
        tenant_id="t1",
        author_gcid="g1",
        refusal_reason="VALIDATION",
        model_armor_verdict="",
        user_facing_message="nope",
        last_candidate_payload_json="{}",
        pipeline_trace_json="[]",
        attempt_count=1,
        mana_charged=0,
    )
    await writer.publish_progress(
        assist_id="a1",
        tenant_id="t1",
        author_gcid="g1",
        pipeline_trace_json="[]",
        step_index=4,
        step_name="critique",
        step_status="ACCEPTED",
    )

    assert conn.uncommitted == []
    assert len(conn.durable) == 2


async def test_sibling_writers_on_the_shared_conn_are_durable() -> None:
    """``AgentDecisionLogOutboxWriter`` / ``HITLDecisionOutboxWriter`` bind the
    SAME connection (see qgen_crew_wiring) and carry the same exposure."""
    conn = _TxnConn()
    agent = AgentDecisionLogOutboxWriter(conn=conn, source_project="chora-489812")
    await agent.emit(
        assist_id="a1",
        agid="qgen_question",
        tenant_id="11111111-1111-7111-8111-111111111111",
        gcid="00000000-0000-7000-8000-000000001999",
        decision="ACCEPTED",
        attempt_count=1,
        max_retries=3,
        critic_notes="",
        quality_warning=False,
        chora_imda_dimension="accountability",
        occurred_at="2026-08-14T02:07:06+00:00",
    )
    assert conn.uncommitted == [], "agent-decision row left uncommitted"
    assert len(conn.durable) == 1

    conn2 = _TxnConn()
    hitl = HITLDecisionOutboxWriter(conn=conn2, source_project="chora-489812")
    await hitl.emit(
        decision_id="a1",
        run_id="a1",
        tenant_id="11111111-1111-7111-8111-111111111111",
        gcid="00000000-0000-7000-8000-000000001999",
        agent_id="qgen_question",
        autonomy_level="HITL_L1",
        summary="1 candidate shipped with a quality warning",
        occurred_at="2026-08-14T02:07:06+00:00",
    )
    assert conn2.uncommitted == [], "HITL gate row left uncommitted"
    assert len(conn2.durable) == 1


# -----------------------------------------------------------------------------
# Invariant 2 - the dispatcher's empty-path release is non-destructive.
# -----------------------------------------------------------------------------


async def test_empty_drain_does_not_destroy_a_concurrent_pending_insert() -> None:
    """THE regression: an empty drain must not ROLLBACK a writer's INSERT.

    Reproduces the exact interleaving - the dispatcher's SELECT already
    returned empty, a writer's INSERT lands, then the dispatcher releases.
    """
    conn = _TxnConn(select_rows=[])
    store = PostgresOutboxStore(conn=conn, worker_id="w1")

    # A concurrent writer's INSERT is sitting uncommitted on the shared conn.
    async with conn.cursor() as cur:
        await cur.execute("INSERT INTO ai_kernel_outbox_events ...", {"id": "row-1"})
    assert conn.uncommitted, "precondition: the write is pending"

    assert await store.fetch_pending(limit=100) == []

    assert conn.rollbacks == 0, (
        "fetch_pending ROLLED BACK the shared connection on the empty path - "
        "this is what destroyed the completed event for job fd80f4e9"
    )
    assert len(conn.durable) == 1, "the concurrent writer's row did not survive the empty drain cycle"


async def test_empty_drain_still_releases_the_read_transaction() -> None:
    """The 2026-06-02 wedge fix must survive: an empty drain still releases
    the txn so the shared connection never sits idle-in-transaction."""
    conn = _TxnConn(select_rows=[])
    store = PostgresOutboxStore(conn=conn, worker_id="w1")

    for _ in range(3):
        assert await store.fetch_pending(limit=100) == []

    assert conn.commits >= 3
    assert conn.rollbacks == 0


@pytest.mark.parametrize("n_pending", [1, 5])
async def test_non_empty_drain_path_is_unchanged(n_pending: int) -> None:
    """The non-empty path must NOT gain a release - the dispatcher still holds
    its ``FOR UPDATE SKIP LOCKED`` locks until mark_published/mark_failed."""
    rows = [
        (
            f"row-{i}",
            "wf",
            "11111111-1111-7111-8111-111111111111",
            "00000000-0000-7000-8000-000000001999",
            "creation.ai_assist.completed",
            "chora.creation.ai_assist.completed.v1",
            b"\x08\x01",
            "{}",
            f"idem-{i}",
            0,
            None,
        )
        for i in range(n_pending)
    ]
    conn = _TxnConn(select_rows=rows)
    store = PostgresOutboxStore(conn=conn, worker_id="w1")

    fetched = await store.fetch_pending(limit=100)

    assert len(fetched) == n_pending
    assert conn.commits == 0 and conn.rollbacks == 0
