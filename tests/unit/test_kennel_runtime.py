"""RED: ADR-254 D5, ONE kennel runtime per process.

The per-process singletons every lane shares, composed once and started once:
the park ledger store, the completion router, the reaper (scan loop + the DLQ
loop over every lane's dead-letter pull subscriptions), and the single outbox
drain. Lanes REGISTER into it (name, crew, roles, runner); the runtime refuses
a duplicate lane and binds each role to exactly one runner.

``readiness()`` is what ``/readyz`` reports: a pod is ready only when the
runtime started, every registered lane reported started, the drain and scan
tasks are alive, and (when any role is registered) the completion and DLQ
loops started. Today ``/readyz`` checks only the guardrail while ``main.py``
degrades five lanes to ``None`` on a startup failure: a READY pod that consumes
nothing. That is the defect this closes.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.kennel_runtime import (
    KennelRuntime,
    LaneRecord,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.deadline_policy import (
    ParkDeadlinePolicy,
)


class _FakeLedger:
    def __init__(self) -> None:
        self.backfills: list[dict[str, Any]] = []

    async def backfill_from_outbox(self, *, crew_for_role: Any, deadline_policy: Any) -> int:
        self.backfills.append({"crew_for_role": dict(crew_for_role), "policy": deadline_policy})
        return 3

    async def get(self, key: str) -> Any:
        return None

    async def mark_completed(self, key: str) -> bool:
        return True

    async def record_late_completion(self, key: str) -> None:
        return None


class _FakeReaper:
    def __init__(self) -> None:
        self.scans = 0

    async def scan_once(self, *, limit: int) -> dict[str, int]:
        self.scans += 1
        return {"request_expired": 0, "outbox_dead_lettered": 0, "errors": 0}


class _FakeDispatcher:
    def __init__(self) -> None:
        self.drains = 0

    async def drain_once(self, *, batch_size: int = 100) -> int:
        self.drains += 1
        return 0


class _FakeLoop:
    def __init__(self, kind: str, subscriber: Any, subscriptions: list[str]) -> None:
        self.kind = kind
        self.subscriber = subscriber
        self.subscriptions = list(subscriptions)
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1


class _FakeRunner:
    async def handle_completion(self, completion: dict[str, Any]) -> None:
        return None


class _Harness:
    def __init__(self) -> None:
        self.ledger = _FakeLedger()
        self.reaper = _FakeReaper()
        self.dispatcher = _FakeDispatcher()
        self.loops: list[_FakeLoop] = []
        self.savers: list[str] = []
        self.policy = ParkDeadlinePolicy(per_role_seconds={"companion_chat": 120})

        from chora_ai_kernel_orchestrator.adapter.pubsub.completion_router import (
            CompletionRouter,
        )

        self.router = CompletionRouter(ledger=self.ledger)

    async def _saver(self, *, crew: str) -> str:
        self.savers.append(crew)
        return f"saver:{crew}"

    def _completion_loop(self, subscriber: Any, roles: list[str]) -> _FakeLoop:
        loop = _FakeLoop("completion", subscriber, roles)
        self.loops.append(loop)
        return loop

    def _dlq_loop(self, subscriber: Any, subscriptions: list[str]) -> _FakeLoop:
        loop = _FakeLoop("dlq", subscriber, subscriptions)
        self.loops.append(loop)
        return loop

    def runtime(self) -> KennelRuntime:
        return KennelRuntime(
            pubsub_project="chora-489812",
            deadline_policy=self.policy,
            ledger=self.ledger,
            router=self.router,
            reaper=self.reaper,
            inbox=object(),
            dispatcher=self.dispatcher,
            saver_factory=self._saver,
            completion_loop_factory=self._completion_loop,
            dlq_loop_factory=self._dlq_loop,
            scan_interval_s=0.05,
            scan_batch=10,
            drain_interval_s=0.05,
        )


def test_register_lane_binds_roles_to_the_router_and_refuses_duplicates() -> None:
    h = _Harness()
    runtime = h.runtime()
    runner = _FakeRunner()
    record = runtime.register_lane("oe_grading", crew="oe_grading", roles=["oe_evaluate", "oe_moderate"], runner=runner)
    assert isinstance(record, LaneRecord)
    assert record.roles == ("oe_evaluate", "oe_moderate")
    assert h.router.roles == ["oe_evaluate", "oe_moderate"]
    assert runtime.crew_for_role == {"oe_evaluate": "oe_grading", "oe_moderate": "oe_grading"}
    with pytest.raises(ValueError):
        runtime.register_lane("oe_grading", crew="oe_grading", roles=[], runner=None)
    # ADR-254 D5: a role MAY be shared by several crews (the router resolves a
    # completion by the ledger row's crew); the SAME (role, crew) twice is refused.
    other = runtime.register_lane("other", crew="other", roles=["oe_evaluate"], runner=runner)
    assert other.roles == ("oe_evaluate",)
    assert runtime.crew_for_role["oe_evaluate"] == "oe_grading", "the backfill map keeps the first crew"
    with pytest.raises(ValueError):
        runtime.register_lane("dup", crew="oe_grading", roles=["oe_evaluate"], runner=runner)


def test_a_lane_with_roles_needs_a_runner() -> None:
    runtime = _Harness().runtime()
    with pytest.raises(ValueError):
        runtime.register_lane("x", crew="x", roles=["recommend"], runner=None)


@pytest.mark.asyncio
async def test_transactional_saver_is_built_per_crew_through_the_factory() -> None:
    h = _Harness()
    runtime = h.runtime()
    assert await runtime.transactional_saver(crew="oe_grading") == "saver:oe_grading"
    assert h.savers == ["oe_grading"]


@pytest.mark.asyncio
async def test_start_backfills_then_starts_loops_scan_and_one_drain() -> None:
    h = _Harness()
    runtime = h.runtime()
    runtime.register_lane("oe_grading", crew="oe_grading", roles=["oe_evaluate", "oe_moderate"], runner=_FakeRunner())
    runtime.register_lane("weakness", crew="weakness_analyser", roles=["weakness_diagnose"], runner=_FakeRunner())

    summary = await runtime.start()
    try:
        assert summary["backfilled"] == 3
        assert h.ledger.backfills[0]["crew_for_role"] == {
            "oe_evaluate": "oe_grading",
            "oe_moderate": "oe_grading",
            "weakness_diagnose": "weakness_analyser",
        }
        kinds = {loop.kind: loop for loop in h.loops}
        assert kinds["completion"].subscriptions == ["oe_evaluate", "oe_moderate", "weakness_diagnose"]
        assert kinds["completion"].started == 1
        assert kinds["dlq"].subscriptions == [
            "chora.dlq.ai_kernel.agent_dispatch.oe_evaluate_requested.v1.pull",
            "chora.dlq.ai_kernel.agent_dispatch.oe_evaluate_completed.v1.pull",
            "chora.dlq.ai_kernel.agent_dispatch.oe_moderate_requested.v1.pull",
            "chora.dlq.ai_kernel.agent_dispatch.oe_moderate_completed.v1.pull",
            "chora.dlq.ai_kernel.agent_dispatch.weakness_diagnose_requested.v1.pull",
            "chora.dlq.ai_kernel.agent_dispatch.weakness_diagnose_completed.v1.pull",
        ]
        assert kinds["dlq"].started == 1
        assert runtime.drain_task is not None and not runtime.drain_task.done()
        assert runtime.scan_task is not None and not runtime.scan_task.done()
        await asyncio.sleep(0.15)
        assert h.dispatcher.drains >= 1, "the single outbox drain must be ticking"
        assert h.reaper.scans >= 1, "the reaper scan must be ticking"
    finally:
        await runtime.stop()
    assert runtime.drain_task is not None and runtime.drain_task.done()
    assert runtime.scan_task is not None and runtime.scan_task.done()
    assert all(loop.stopped == 1 for loop in h.loops)


@pytest.mark.asyncio
async def test_start_without_roles_still_drains_and_scans_but_opens_no_loops() -> None:
    h = _Harness()
    runtime = h.runtime()
    summary = await runtime.start()
    try:
        assert summary["roles"] == []
        assert h.loops == []
        assert runtime.drain_task is not None and runtime.scan_task is not None
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_readiness_reports_not_ready_until_started_and_every_lane_is_up() -> None:
    h = _Harness()
    runtime = h.runtime()
    runtime.register_lane("oe_grading", crew="oe_grading", roles=["oe_evaluate"], runner=_FakeRunner())

    before = runtime.readiness()
    assert before["ready"] is False
    assert "runtime_not_started" in before["reasons"]

    await runtime.start()
    try:
        mid = runtime.readiness()
        assert mid["ready"] is False
        assert "lane_not_started:oe_grading" in mid["reasons"]

        runtime.mark_lane_started("oe_grading")
        after = runtime.readiness()
        assert after["ready"] is True
        assert after["reasons"] == []
        assert after["lanes"]["oe_grading"]["started"] is True
        assert after["roles"] == ["oe_evaluate"]

        runtime.mark_lane_failed("oe_grading", "subscriber died")
        failed = runtime.readiness()
        assert failed["ready"] is False
        assert "lane_failed:oe_grading" in failed["reasons"]
        assert failed["lanes"]["oe_grading"]["detail"] == "subscriber died"
    finally:
        await runtime.stop()

    stopped = runtime.readiness()
    assert stopped["ready"] is False


def test_mark_unknown_lane_is_refused() -> None:
    runtime = _Harness().runtime()
    with pytest.raises(ValueError):
        runtime.mark_lane_started("nobody")


def test_runtime_refuses_blank_project() -> None:
    h = _Harness()
    with pytest.raises(ValueError):
        KennelRuntime(
            pubsub_project=" ",
            deadline_policy=h.policy,
            ledger=h.ledger,
            router=h.router,
            reaper=h.reaper,
            inbox=object(),
            dispatcher=h.dispatcher,
            saver_factory=h._saver,
            completion_loop_factory=h._completion_loop,
            dlq_loop_factory=h._dlq_loop,
        )


def test_deadline_policy_is_exposed_for_the_lanes() -> None:
    h = _Harness()
    runtime = h.runtime()
    assert runtime.deadline_policy is h.policy
    assert runtime.deadline_policy.deadline_for(
        "companion_chat", parked_at=_dt.datetime(2026, 8, 22, tzinfo=_dt.UTC)
    ) == _dt.datetime(2026, 8, 22, 0, 2, tzinfo=_dt.UTC)
