"""Composition helpers for the ADR-253 Pub/Sub dispatch lane.

One job, and it is about making the wrong configuration LOUD:
``require_transactional_saver`` is the guard that keeps the ratified property
honest. An agent-dispatch lane REFUSES to run on the shared checkpointer.

Why that is worth a whole function. The shared ``LazyPostgresSaver`` opens its
own autocommit connection (ADR-253 1.4), so a lane running on it would write
the dispatch row in a different transaction from the park. That is exactly the
split ADR-253 D3a was ruled against, and it would still WORK, nearly all of the
time. The failure is not a crash; it is that the single-transaction property
the ADR claims, the report describes and the deck draws would simply be false,
with nothing on any screen to say so. A silent fallback here is therefore worse
than a refusal to start.

2026-08-23 (RULING A): the per-deployment transport switch went with the HTTP
executor. Pub/Sub dispatch is the only transport, so there is no arm left to
select and the guard is unconditional. The rollback of record is the previous
kennel image (or the previous agent image on the agent side), never a
transport flip.
"""

from __future__ import annotations

import logging
from typing import Any

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    ROLE_OE_EVALUATE,
    ROLE_OE_MODERATE,
)

logger = logging.getLogger(__name__)

# The two roles the OE lane dispatches. These are ROLE ids (oe_evaluate), not
# agent ids (oe_evaluator) - the topics are keyed on the role, and the two
# deliberately differ.
OE_DISPATCH_ROLES: tuple[str, ...] = (ROLE_OE_EVALUATE, ROLE_OE_MODERATE)


def require_transactional_saver(saver: Any, *, lane: str = "OE grading") -> Any:
    """Return ``saver``, refusing one that is not the transactional saver.

    See the module docstring: the point is that the wrong saver here does not
    fail, it just quietly makes the ADR's atomicity claim untrue. ``lane``
    names the caller in the refusal (every dispatch lane shares this guard).
    """
    from chora_ai_kernel_orchestrator.adapter.checkpointer.transactional_dispatch_saver import (  # noqa: E501
        TransactionalDispatchSaver,
    )

    if not isinstance(saver, TransactionalDispatchSaver):
        raise ValueError(
            f"{lane} dispatches agents over Pub/Sub but its checkpointer is "
            f"{type(saver).__name__}, not a TransactionalDispatchSaver. "
            "ADR-253 D3a requires the park and the dispatch outbox row to "
            "commit in ONE transaction; the shared saver opens its own "
            "autocommit connection, so running on it would split them and the "
            "single-transaction property would be silently false. "
            "Refusing to start."
        )
    return saver


def assemble_transactional_saver(
    *,
    inner: Any,
    conn: Any,
    source_project: str,
    crew: str | None = None,
    deadline_policy: Any = None,
    connect: Any = None,
) -> Any:
    """Compose the D3a saver from an already-open connection (pure; unit-tested).

    With ``crew`` + ``deadline_policy`` (ADR-254 D5) the park ledger writer
    rides the SAME connection, so the ledger row (parked_at, deadline_at)
    commits in the park's transaction. Crew and policy come together or not
    at all; the saver itself enforces that too.
    """
    from chora_ai_kernel_orchestrator.adapter.checkpointer.transactional_dispatch_saver import (  # noqa: E501
        TransactionalDispatchSaver,
    )
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_outbox_writer import (  # noqa: E501
        AgentDispatchOutboxWriter,
    )

    if (crew is None) != (deadline_policy is None):
        raise ValueError(
            "assemble_transactional_saver: crew and deadline_policy must be given "
            "together (ADR-254 D5: every parked run carries a deadline) or both omitted"
        )
    ledger_writer = None
    if crew is not None:
        from chora_ai_kernel_orchestrator.adapter.pubsub.park_ledger import ParkLedgerWriter

        ledger_writer = ParkLedgerWriter(conn=conn, crew=crew)
    return TransactionalDispatchSaver(
        inner=inner,
        conn=conn,
        outbox_writer=AgentDispatchOutboxWriter(conn=conn, source_project=source_project),
        park_ledger_writer=ledger_writer,
        deadline_policy=deadline_policy,
        connect=connect,
    )


async def open_saver_connection(dsn: str) -> Any:
    """Open THE saver's dedicated connection, session-configured and COMMITTED.

    ⚠ THIS IS THE ONE PLACE A SAVER CONNECTION IS BORN. It is used for the first
    connect AND for every reconnect (TransactionalDispatchSaver._reconnect), so
    the two cannot drift. A reconnect that skipped any step here would be the
    half-right repair that is worse than the bug, per the GUC note below.

    autocommit is deliberately OFF: the saver owns every transaction boundary.

    Sweeper mode, EXPLICIT (G2). The park ledger INSERT rides this connection
    inside the saver's transaction; under the fail-CLOSED 0056 policy the
    policy's WITH CHECK would otherwise REJECT every park, which is the
    reaper-that-reaps-nothing failure that TESTS GREEN. The GUC is applied here
    and COMMITTED before any saver transaction opens: a session GUC set inside a
    transaction is reverted when that transaction rolls back, and the saver rolls
    back on failure by design. Committing it puts it out of reach of those
    rollbacks for the life of the session, INCLUDING a session created by a
    reconnect.
    """
    import psycopg

    from chora_ai_kernel_orchestrator.adapter.postgres.kernel_sweeper import (
        KERNEL_SWEEPER_GUC,
        KERNEL_SWEEPER_ON,
    )

    conn = await psycopg.AsyncConnection.connect(dsn, autocommit=False)
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT set_config(%s, %s, %s)",
            (KERNEL_SWEEPER_GUC, KERNEL_SWEEPER_ON, False),
        )
    await conn.commit()
    return conn


async def build_transactional_saver(
    *,
    dsn: str,
    source_project: str,
    crew: str | None = None,
    deadline_policy: Any = None,
) -> Any:  # pragma: no cover — integration glue; unit-tested via its components
    """Open the dedicated connection and assemble the D3a saver.

    The connection is this saver's OWN, deliberately not the crew's. The crew
    connection is autocommit=True on purpose (no dispatcher runs on it, so an
    open transaction there would never flush and a learner's submission would
    sit in PENDING_OE_GRADING forever). Borrowing it would either destroy the
    single-transaction property or silently retire that contract; a dedicated
    connection into the SAME database and the SAME ai_kernel_outbox_events table
    delivers the property with no blast radius on the live lane.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    conn = await open_saver_connection(dsn)
    inner = AsyncPostgresSaver(conn)
    # The checkpoint tables are created by the shared saver's setup already;
    # calling setup here is idempotent and makes a fresh database work.
    await inner.setup()
    await conn.commit()
    saver = assemble_transactional_saver(
        inner=inner,
        conn=conn,
        source_project=source_project,
        crew=crew,
        deadline_policy=deadline_policy,
        # The saver reconnects itself through the SAME procedure that opened
        # this connection, so a reconnected session is configured identically.
        connect=lambda: open_saver_connection(dsn),
    )
    logger.info(
        "agent_dispatch_wiring.transactional_saver_open",
        extra={"source_project": source_project, "crew": crew or ""},
    )
    return saver


__all__ = [
    "OE_DISPATCH_ROLES",
    "assemble_transactional_saver",
    "build_transactional_saver",
    "require_transactional_saver",
]
