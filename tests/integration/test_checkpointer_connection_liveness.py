"""Connection liveness + single-transaction property for TransactionalDispatchSaver.

Both tests need a REAL PostgreSQL. Nothing here is mocked at the database
boundary: the connection is killed SERVER-SIDE with ``pg_terminate_backend``,
which is what a Cloud SQL cost-pause resume does to this service, and the
rollback assertion is read back on an INDEPENDENT connection.

⚠ THESE SKIP WITHOUT ``CHORA_SAVER_TEST_DSN``, AND A SKIPPED TEST GATES NOTHING.
CI has no Postgres today (see tests/integration/test_crash_resume.py, which
falls back to InMemorySaver for exactly that reason), so until CI gains one
these run locally only and must not be counted as protection in CI.

Run:
    docker exec chora-local-postgres psql -U postgres -c 'CREATE DATABASE chora_saver_red'
    CHORA_SAVER_TEST_DSN=postgresql://postgres:postgres@127.0.0.1:5432/chora_saver_red \
      .venv/bin/python -m pytest tests/integration/test_checkpointer_connection_liveness.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import psycopg
import pytest
from langgraph.types import Interrupt

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    DISPATCH_INTERRUPT_KEY,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_wiring import (
    build_transactional_saver,
)

DSN = os.getenv("CHORA_SAVER_TEST_DSN", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="CHORA_SAVER_TEST_DSN not set (needs a real Postgres)"),
    pytest.mark.asyncio,
]


@pytest.fixture(autouse=True, scope="module")
def _bootstrap_checkpoint_schema() -> None:
    """Create LangGraph's checkpoint tables on an AUTOCOMMIT connection first.

    Mirrors production, where agent_dispatch_wiring.py:146 notes "the checkpoint
    tables are created by the shared saver's setup already". They must pre-exist:
    AsyncPostgresSaver.setup() issues CREATE INDEX CONCURRENTLY, which Postgres
    refuses inside a transaction block, and the saver's own connection is opened
    autocommit=False. (Minor separate finding: that makes the same line's "makes a
    fresh database work" claim false on a genuinely empty database.)
    """
    import asyncio

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async def _setup() -> None:
        conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
        try:
            await AsyncPostgresSaver(conn).setup()
        finally:
            await conn.close()

    asyncio.run(_setup())


def _dispatch_request(thread_id: str) -> dict[str, Any]:
    return {
        "idempotency_key": f"agent_dispatch.qgen_generate.{thread_id}",
        "envelope": {
            "occurred_at": "2026-08-23T17:02:42+00:00",
            "tenant_id": str(uuid.uuid4()),
            "gcid": str(uuid.uuid4()),
        },
        "body": {"agent_role": "qgen_generate", "execution_id": f"{thread_id}:generate:1"},
        "topic": "chora.ai_kernel.agent_dispatch.qgen_generate_requested.v1",
        "workflow_id": str(uuid.uuid4()),
    }


def _park_writes(thread_id: str) -> list[tuple[str, Any]]:
    """A park as LangGraph actually records it: put_writes on ``__interrupt__``
    (agent_dispatch.py:228-232), NOT put."""
    # The REAL langgraph Interrupt, not a stub: it is what the serde must
    # persist, and a stub is silently unserialisable by msgpack.
    return [("__interrupt__", (Interrupt(value={DISPATCH_INTERRUPT_KEY: _dispatch_request(thread_id)}),))]


async def _backend_pid(conn: Any) -> int:
    async with conn.cursor() as cur:
        await cur.execute("SELECT pg_backend_pid()")
        return (await cur.fetchone())[0]


async def _kill_backend(pid: int) -> None:
    """Terminate the backend from ANOTHER session: a server-side close, the same
    shape as a cost-pause resume. Not a mock, not conn.close()."""
    killer = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    try:
        async with killer.cursor() as cur:
            await cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
    finally:
        await killer.close()


# ---------------------------------------------------------------------------
# RED 1: liveness. FAILS TODAY.
# ---------------------------------------------------------------------------
async def test_saver_survives_a_server_side_connection_kill() -> None:
    """The saver's dedicated connection is opened raw at agent_dispatch_wiring.py:126
    with no reconnect wrapper, so the first operation after a server-side close
    raises instead of reconnecting.

    Production instance 2026-08-23T17:02:42Z: psycopg.OperationalError
    ("consuming input failed: server closed the connection unexpectedly") out of
    transactional_dispatch_saver.py:164, which killed the qgen drive.
    """
    saver = await build_transactional_saver(dsn=DSN, source_project="chora-test")
    pid = await _backend_pid(saver._conn)
    await saver._conn.commit()

    await _kill_backend(pid)

    config = {"configurable": {"thread_id": str(uuid.uuid4()), "checkpoint_ns": ""}}
    # Must reconnect and answer. Today this raises psycopg.OperationalError.
    assert await saver.aget_tuple(config) is None


# ---------------------------------------------------------------------------
# GUARD 2: atomicity. PASSES TODAY; must FAIL against a pooled saver.
# ---------------------------------------------------------------------------
async def test_checkpoint_write_rolls_back_with_the_dispatch_row() -> None:
    """park + outbox row + ledger row are ONE transaction, so a failure after the
    checkpoint write must roll the checkpoint write back too.

    ⚠ THIS IS THE TRAP-2 GUARD. `AsyncConnectionPool` passes langgraph's type
    gate (_ainternal.py:17-21) and looks like the textbook repair, but every
    method here calls the inner saver INSIDE self._tx() (transactional_dispatch_
    saver.py:179-180) and the inner saver then acquires its OWN connection. At
    max_size=1 that deadlocks; at max_size>1 the checkpoint commits on a
    DIFFERENT connection from the outbox row and this assertion fails. Without
    this test that repair passes review, passes every unit test and passes a
    manual walk, and only diverges under a crash between the two commits.
    """
    saver = await build_transactional_saver(dsn=DSN, source_project="chora-test")
    thread_id = str(uuid.uuid4())
    # checkpoint_id is required by AsyncPostgresSaver.aput_writes (aio.py:332).
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": str(uuid.uuid4()),
        }
    }

    class _FailingOutbox:
        """Fails where a real outbox INSERT would (constraint, disconnect, crash).
        The DB interaction under test stays real; only the trigger is injected."""

        async def queue_request(self, request: dict[str, Any]) -> str:
            raise RuntimeError("outbox insert failed after the checkpoint write")

    saver._outbox = _FailingOutbox()

    with pytest.raises(RuntimeError, match="outbox insert failed"):
        await saver.aput_writes(config, _park_writes(thread_id), task_id="task-1")

    # Read back on an INDEPENDENT connection: nothing the failed transaction
    # wrote may be visible.
    probe = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    try:
        async with probe.cursor() as cur:
            await cur.execute("SELECT count(*) FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
            rows = (await cur.fetchone())[0]
    finally:
        await probe.close()

    assert rows == 0, (
        f"{rows} checkpoint_writes row(s) survived a rolled-back dispatch. The "
        "checkpoint committed independently of the outbox row, so park + outbox "
        "are NO LONGER ATOMIC (see the pooled-saver trap in this test's docstring)."
    )
