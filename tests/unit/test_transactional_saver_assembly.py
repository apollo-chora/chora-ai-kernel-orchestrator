"""RED: the D3a saver is assembled WITH the park ledger writer per crew.

``agent_dispatch_wiring.assemble_transactional_saver`` is the pure composition
``build_transactional_saver`` (the psycopg glue) delegates to: the inner
AsyncPostgresSaver, the connection, the outbox writer, and, when a crew is
given, the ParkLedgerWriter on the SAME connection plus the deadline policy, so
the ledger row commits in the park's transaction (ADR-254 D5).
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.checkpointer.transactional_dispatch_saver import (
    TransactionalDispatchSaver,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch_wiring import (
    assemble_transactional_saver,
    build_transactional_saver,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.park_ledger import ParkLedgerWriter
from chora_ai_kernel_orchestrator.domain.agent_dispatch.deadline_policy import (
    ParkDeadlinePolicy,
)


class _Inner:
    serde = None

    def get_next_version(self, current: Any, channel: Any = None) -> Any:
        return 1


def test_assembly_with_a_crew_attaches_the_ledger_writer_and_the_policy() -> None:
    policy = ParkDeadlinePolicy()
    conn = object()
    saver = assemble_transactional_saver(
        inner=_Inner(),
        conn=conn,
        source_project="chora-489812",
        crew="oe_grading",
        deadline_policy=policy,
    )
    assert isinstance(saver, TransactionalDispatchSaver)
    assert isinstance(saver._ledger, ParkLedgerWriter)  # noqa: SLF001
    assert saver._ledger.crew == "oe_grading"  # noqa: SLF001
    assert saver._deadline_policy is policy  # noqa: SLF001
    assert saver._outbox.source_project == "chora-489812"  # noqa: SLF001


def test_assembly_without_a_crew_is_the_plain_d3a_saver() -> None:
    saver = assemble_transactional_saver(
        inner=_Inner(),
        conn=object(),
        source_project="chora-489812",
    )
    assert saver._ledger is None  # noqa: SLF001
    assert saver._deadline_policy is None  # noqa: SLF001


def test_assembly_refuses_a_crew_without_a_policy_and_vice_versa() -> None:
    with pytest.raises(ValueError):
        assemble_transactional_saver(
            inner=_Inner(),
            conn=object(),
            source_project="chora-489812",
            crew="oe_grading",
        )
    with pytest.raises(ValueError):
        assemble_transactional_saver(
            inner=_Inner(),
            conn=object(),
            source_project="chora-489812",
            deadline_policy=ParkDeadlinePolicy(),
        )


def test_build_transactional_saver_accepts_crew_and_policy() -> None:
    params = inspect.signature(build_transactional_saver).parameters
    assert "crew" in params and "deadline_policy" in params
