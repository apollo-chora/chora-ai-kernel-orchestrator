"""RED: nothing consumes the growth-edge completion, so a parked run never resumes.

Measured live on 2026-08-21. One diagnosis produced exactly one dispatch (the
ack-on-park fix working), the agent handled it and logged
``agentdispatch.completed`` with the right thread, and then the run stopped.
The upload sat at QUEUED.

Cause: ``AgentCompletionPubsubLoop`` is only ever built in
``oe_grading_crew_wiring`` with ``agent_roles=OE_DISPATCH_ROLES``. The
growth-edge completion subscription exists and receives the message, and nobody
is listening on it. OE's own wiring calls the loop's absence "fatal, not a
degradation", and it is just as fatal here.

Two halves are needed and neither existed:

  1. ``WeaknessAnalyserCrewRunner.handle_completion``, resume the thread the
     completion names. The completion subscriber calls exactly this method.
  2. The review_pending emit has to MOVE for this transport. On HTTP the inbound
     subscriber drove the graph all the way to the HITL interrupt and emitted the
     panel itself. On the bus it parks at the dispatch long before HITL, so the
     run now reaches the human gate inside the COMPLETION resume, and that is
     where the panel has to come from. Leave it only on the inbound path and the
     learner is never told their review is ready.
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew_runner import (
    WeaknessAnalyserCrewRunner,
)

_TENANT = "11111111-1111-7111-8111-111111111111"
_UPLOAD = "01a0236e-a22f-7119-a1f2-5fedf07af073"
_THREAD = f"{_TENANT}:{_UPLOAD}"


class _Interrupt:
    def __init__(self, value: Any) -> None:
        self.value = value


class _Graph:
    """Records the resume and returns a terminal parked at the HITL review."""

    def __init__(self, terminal: dict[str, Any]) -> None:
        self.terminal = terminal
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, command: Any, config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"command": command, "config": config})
        return self.terminal


class _Publisher:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def publish_review_pending(self, *, panel: Any, traceparent: str = "", tracestate: str = "") -> None:
        self.published.append({"panel": panel})


def _runner(graph: Any, publisher: Any = None) -> WeaknessAnalyserCrewRunner:
    r = WeaknessAnalyserCrewRunner.__new__(WeaknessAnalyserCrewRunner)
    r._graph = graph
    r._review_pending_publisher = publisher

    async def _panel(payload: Any) -> dict[str, Any]:
        return {"proposed_edges": payload.get("edges", [])}

    r._build_panel = _panel  # type: ignore[method-assign]
    return r


def _hitl_terminal() -> dict[str, Any]:
    review = {"tenant_id": _TENANT, "learner_gcid": "g", "upload_id": _UPLOAD, "edges": [{"concept": "meiosis"}]}
    return {"__interrupt__": [_Interrupt(review)], "run_id": "run-1"}


@pytest.mark.asyncio
async def test_a_completion_resumes_the_thread_it_names() -> None:
    graph = _Graph(_hitl_terminal())
    await _runner(graph).handle_completion({"thread_id": _THREAD, "status": "OK", "output_payload": "{}"})

    assert graph.calls, "the graph was never resumed, so the run stays parked"
    assert graph.calls[0]["config"]["configurable"]["thread_id"] == _THREAD


@pytest.mark.asyncio
async def test_a_completion_with_no_thread_is_refused() -> None:
    """Resuming a guessed thread would inject one learner's diagnosis into
    another learner's run."""
    graph = _Graph(_hitl_terminal())
    with pytest.raises(ValueError, match="thread_id"):
        await _runner(graph).handle_completion({"status": "OK"})
    assert not graph.calls


@pytest.mark.asyncio
async def test_reaching_the_human_gate_on_resume_emits_the_panel() -> None:
    """The emit that used to happen on the inbound path now happens here."""
    pub = _Publisher()
    await _runner(_Graph(_hitl_terminal()), pub).handle_completion(
        {"thread_id": _THREAD, "status": "OK", "output_payload": "{}"}
    )

    assert pub.published, (
        "the run reached the HITL review and nothing told the learner; on this "
        "transport the inbound subscriber parked long before HITL and cannot emit"
    )
    assert pub.published[0]["panel"] == {"proposed_edges": [{"concept": "meiosis"}]}


@pytest.mark.asyncio
async def test_a_resume_that_parks_again_emits_nothing() -> None:
    """A crew with several hops parks again on its next dispatch. That is not a
    review and must not be announced as one."""
    from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
        DISPATCH_INTERRUPT_KEY,
    )

    pub = _Publisher()
    parked = {"__interrupt__": [_Interrupt({DISPATCH_INTERRUPT_KEY: {"agent_role": "x"}})]}
    result = await _runner(_Graph(parked), pub).handle_completion(
        {"thread_id": _THREAD, "status": "OK", "output_payload": "{}"}
    )

    assert result.awaiting_agent is True
    assert not pub.published
