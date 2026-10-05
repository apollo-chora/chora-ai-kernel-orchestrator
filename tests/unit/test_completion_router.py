"""RED: ADR-254 D5, ONE completion router for every dispatch role.

Today each lane builds its own ``AgentCompletionPubsubLoop`` bound to ONE
runner and a hard-coded role bundle (``OE_DISPATCH_ROLES``), which is how the
growth-edge conversion shipped with nothing consuming its completions
(docs/TODO-DEVELOPMENT.md, 2026-08-21). The router is the one place a role is
bound to the runner that resumes it; the completion loop, the reaper and the
generic workflow all route through it.

The router also settles the park ledger: a real completion flips the row to
``completed``; a completion for a row the reaper already settled is recorded
(``late_completion_at``) and NEVER resumed; a completion for a thread that has
no ledger row (parked before migration 0056 landed) still resumes, because
refusing it would strand the OE runs parked during the 2026-08-21 outage.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.completion_router import (
    CompletionRouter,
)
from chora_ai_kernel_orchestrator.domain.agent_dispatch.park import (
    ParkRecord,
    ParkState,
)

_KEY = "agent_dispatch.oe_evaluate.01a02062-e5b4-7870-8fca-53ce363cd542:tsq-1:1"
_NOW = _dt.datetime(2026, 8, 22, 15, 0, tzinfo=_dt.UTC)


def _park(state: ParkState = ParkState.PARKED) -> ParkRecord:
    return ParkRecord(
        idempotency_key=_KEY,
        workflow_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        thread_id="01a02062-e5b4-7870-8fca-53ce363cd542",
        crew="oe_grading",
        tenant_id="11111111-1111-7111-8111-111111111111",
        gcid="00000000-0000-7000-8000-000000001999",
        agent_role="oe_evaluate",
        request_topic="chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1",
        completion_topic="chora.ai_kernel.agent_dispatch.oe_evaluate_completed.v1",
        traceparent="",
        tracestate="",
        parked_at=_NOW,
        deadline_at=_NOW + _dt.timedelta(days=7),
        state=state,
    )


class _FakeLedger:
    def __init__(self, park: ParkRecord | None) -> None:
        self._park = park
        self.completed: list[str] = []
        self.late: list[str] = []

    async def get(self, key: str) -> ParkRecord | None:
        return self._park

    async def mark_completed(self, key: str) -> bool:
        self.completed.append(key)
        return True

    async def record_late_completion(self, key: str) -> None:
        self.late.append(key)


class _FakeRunner:
    def __init__(self, name: str) -> None:
        self.name = name
        self.completions: list[dict[str, Any]] = []

    async def handle_completion(self, completion: dict[str, Any]) -> None:
        self.completions.append(completion)


def _completion(role: str = "oe_evaluate", key: str = _KEY) -> dict[str, Any]:
    return {"agent_role": role, "idempotency_key": key, "thread_id": "t", "status": "OK"}


def test_register_binds_a_role_once_and_lists_roles_sorted() -> None:
    router = CompletionRouter(ledger=_FakeLedger(None))
    oe = _FakeRunner("oe")
    router.register("oe_moderate", oe)
    router.register("oe_evaluate", oe)
    assert router.roles == ["oe_evaluate", "oe_moderate"]
    assert router.runner_for("oe_evaluate") is oe
    with pytest.raises(ValueError):
        router.register("oe_evaluate", _FakeRunner("other"))
    with pytest.raises(ValueError):
        router.register("  ", oe)


def test_unknown_role_is_refused_loudly() -> None:
    router = CompletionRouter(ledger=_FakeLedger(None))
    with pytest.raises(ValueError):
        router.runner_for("nobody")


def test_require_roles_registered_names_the_gap() -> None:
    router = CompletionRouter(ledger=_FakeLedger(None))
    router.register("oe_evaluate", _FakeRunner("oe"))
    router.require_roles_registered(["oe_evaluate"])
    with pytest.raises(ValueError, match="oe_moderate"):
        router.require_roles_registered(["oe_evaluate", "oe_moderate"])


@pytest.mark.asyncio
async def test_a_real_completion_resumes_the_runner_then_settles_the_ledger() -> None:
    ledger = _FakeLedger(_park())
    runner = _FakeRunner("oe")
    router = CompletionRouter(ledger=ledger)
    router.register("oe_evaluate", runner)

    outcome = await router.handle_completion(_completion())

    assert outcome == "resumed"
    assert runner.completions == [_completion()]
    assert ledger.completed == [_KEY]
    assert ledger.late == []


@pytest.mark.asyncio
async def test_a_completion_after_a_reap_is_recorded_and_never_resumed() -> None:
    ledger = _FakeLedger(_park(ParkState.REAPED))
    runner = _FakeRunner("oe")
    router = CompletionRouter(ledger=ledger)
    router.register("oe_evaluate", runner)

    outcome = await router.handle_completion(_completion())

    assert outcome == "late"
    assert runner.completions == []
    assert ledger.late == [_KEY]
    assert ledger.completed == []


@pytest.mark.asyncio
async def test_an_unledgered_completion_still_resumes_the_legacy_park() -> None:
    """Parks made before 0056 landed have no ledger row; their completions
    must still resume, or the OE runs parked during the outage stay parked."""
    ledger = _FakeLedger(None)
    runner = _FakeRunner("oe")
    router = CompletionRouter(ledger=ledger)
    router.register("oe_evaluate", runner)

    outcome = await router.handle_completion(_completion())

    assert outcome == "resumed_unledgered"
    assert runner.completions == [_completion()]
    assert ledger.completed == []


@pytest.mark.asyncio
async def test_resume_reaped_routes_without_touching_the_ledger() -> None:
    """The reaper flips the ledger itself (state='reaped'); the router must not
    race it to 'completed'."""
    ledger = _FakeLedger(_park())
    runner = _FakeRunner("oe")
    router = CompletionRouter(ledger=ledger)
    router.register("oe_evaluate", runner)

    failed = {**_completion(), "status": "FAILED", "reaper_arm": "request_expired"}
    await router.resume_reaped(failed)

    assert runner.completions == [failed]
    assert ledger.completed == []
    assert ledger.late == []


@pytest.mark.asyncio
async def test_a_completion_without_role_or_key_is_refused() -> None:
    router = CompletionRouter(ledger=_FakeLedger(_park()))
    router.register("oe_evaluate", _FakeRunner("oe"))
    with pytest.raises(ValueError):
        await router.handle_completion({"idempotency_key": _KEY})
    with pytest.raises(ValueError):
        await router.handle_completion({"agent_role": "oe_evaluate"})


@pytest.mark.asyncio
async def test_a_runner_failure_does_not_settle_the_ledger() -> None:
    """If the resume raises, the subscriber NACKs and redelivers; the row must
    still read 'parked' so the retry (or the reaper) can act on it."""
    ledger = _FakeLedger(_park())

    class _Boom:
        async def handle_completion(self, completion: dict[str, Any]) -> None:
            raise RuntimeError("resume exploded")

    router = CompletionRouter(ledger=ledger)
    router.register("oe_evaluate", _Boom())
    with pytest.raises(RuntimeError):
        await router.handle_completion(_completion())
    assert ledger.completed == []


# --------------------------------------------------------------------------- #
# ADR-254 D5: a ROLE is shared by several crews (companion_chat resumes the
# reflection workflow, the diagnosis voice step and the typed-chat lane), so
# the router binds (role, crew) and routes a completion by the ledger row's crew.
# --------------------------------------------------------------------------- #


def _park_for(crew: str, key: str = _KEY) -> ParkRecord:
    import dataclasses

    return dataclasses.replace(_park(), crew=crew, idempotency_key=key)


def test_two_crews_can_bind_the_same_role() -> None:
    router = CompletionRouter(ledger=_FakeLedger(None))
    reflect, voice = _FakeRunner("reflection"), _FakeRunner("diagnosis")
    router.register("companion_chat", reflect, crew="companion_reflection")
    router.register("companion_chat", voice, crew="weakness_analyser")
    assert router.roles == ["companion_chat"]
    assert router.runner_for("companion_chat", crew="companion_reflection") is reflect
    assert router.runner_for("companion_chat", crew="weakness_analyser") is voice
    with pytest.raises(ValueError, match="already bound"):
        router.register("companion_chat", _FakeRunner("dup"), crew="companion_reflection")


@pytest.mark.asyncio
async def test_a_completion_routes_by_the_ledger_rows_crew() -> None:
    ledger = _FakeLedger(_park_for("weakness_analyser"))
    router = CompletionRouter(ledger=ledger)
    reflect, voice = _FakeRunner("reflection"), _FakeRunner("diagnosis")
    router.register("companion_chat", reflect, crew="companion_reflection")
    router.register("companion_chat", voice, crew="weakness_analyser")
    await router.handle_completion(_completion(role="companion_chat"))
    assert voice.completions and not reflect.completions
    assert ledger.completed == [_KEY]


@pytest.mark.asyncio
async def test_an_unledgered_completion_on_a_shared_role_is_refused_not_guessed() -> None:
    router = CompletionRouter(ledger=_FakeLedger(None))
    router.register("companion_chat", _FakeRunner("a"), crew="companion_reflection")
    router.register("companion_chat", _FakeRunner("b"), crew="weakness_analyser")
    with pytest.raises(ValueError, match="ambiguous"):
        await router.handle_completion(_completion(role="companion_chat"))


def test_a_single_crew_role_still_resolves_without_a_crew_hint() -> None:
    router = CompletionRouter(ledger=_FakeLedger(None))
    oe = _FakeRunner("oe")
    router.register("oe_evaluate", oe, crew="oe_grading")
    assert router.runner_for("oe_evaluate") is oe
    # a ledger row from a renamed crew still lands on the only runner, loudly
    assert router.runner_for("oe_evaluate", crew="old_name") is oe


@pytest.mark.asyncio
async def test_resume_reaped_routes_by_the_crew_the_reaper_stamps() -> None:
    router = CompletionRouter(ledger=_FakeLedger(None))
    reflect, voice = _FakeRunner("reflection"), _FakeRunner("diagnosis")
    router.register("companion_chat", reflect, crew="companion_reflection")
    router.register("companion_chat", voice, crew="weakness_analyser")
    await router.resume_reaped({**_completion(role="companion_chat"), "crew": "companion_reflection"})
    assert reflect.completions and not voice.completions
