"""RED: a dispatch park is not a review, and must not be handled as one.

Observed live on 2026-08-21, driving one real diagnosis through the converted
lane. The upload produced FIVE dispatch rows instead of one, spaced 21 to 39
seconds apart, and stopped at exactly five, which is the inbound subscription's
``maxDeliveryAttempts``. Each redelivery re-ran the WHOLE graph, including the
non-deterministic extract, so every attempt dispatched a different payload.

Mechanism. Under the Pub/Sub transport the graph parks at the DIAGNOSE dispatch,
which is earlier than the HITL review it used to park at. LangGraph returns that
park in ``__interrupt__`` exactly as it returns the review one, so ``_result``
cannot tell them apart: it treats the dispatch payload as a review payload and
builds a review panel out of it, with empty tenant_id and learner_gcid. The
graph subscriber's ``except Exception`` then NACKs the inbound message, the
broker redelivers, and the run restarts from the top.

Neither half is wrong on its own. The subscriber is right to NACK a failed
start, and the runner was right to treat an interrupt as a review while the
review was the only interrupt in the graph. The conversion added a second kind.

The correct behaviour is to ACK: the park is committed with its dispatch outbox
row in one transaction (D3a), and the completion consumer resumes the thread.
A redelivery is not recovery here, it is duplicate work against a paid model.
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    DISPATCH_INTERRUPT_KEY,
)
from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew_runner import (
    WeaknessAnalyserCrewRunner,
)

_TENANT = "11111111-1111-7111-8111-111111111111"
_UPLOAD = "01a0235d-2487-7263-8754-92993f0cd59e"
_THREAD = f"{_TENANT}:{_UPLOAD}"


class _Interrupt:
    def __init__(self, value: Any) -> None:
        self.value = value


def _runner() -> WeaknessAnalyserCrewRunner:
    return WeaknessAnalyserCrewRunner.__new__(WeaknessAnalyserCrewRunner)


def _dispatch_terminal() -> dict[str, Any]:
    """What LangGraph hands back when the diagnose hop parks on the bus."""
    return {
        "__interrupt__": [
            _Interrupt(
                {
                    DISPATCH_INTERRUPT_KEY: {
                        "agent_role": "weakness_diagnose",
                        "execution_id": f"{_UPLOAD}:diagnose:b6b94a709804",
                        "topic": "chora.ai_kernel.agent_dispatch.weakness_diagnose_requested.v1",
                    }
                }
            )
        ],
        "run_id": "run-1",
    }


@pytest.mark.asyncio
async def test_a_dispatch_park_is_reported_as_awaiting_an_agent() -> None:
    """The discriminator the runner currently lacks."""
    result = await _runner()._result(_dispatch_terminal(), thread_id=_THREAD, fallback_run_id="run-1")

    assert result.awaiting_agent is True, (
        "the runner cannot tell a dispatch park from a review interrupt, so the "
        "subscriber has no way to ACK one and emit a panel for the other"
    )
    assert result.interrupted is True, "the run IS suspended, just not by a human"


@pytest.mark.asyncio
async def test_a_dispatch_park_builds_no_review_panel() -> None:
    """Building a panel from a dispatch payload is what raised, and the raise is
    what became a NACK, and the NACK is what became five model calls."""
    result = await _runner()._result(_dispatch_terminal(), thread_id=_THREAD, fallback_run_id="run-1")

    assert result.review_panel is None
    assert result.review_payload is None, (
        "a dispatch envelope must not be surfaced as a review payload; the FE resume route would try to render it"
    )


@pytest.mark.asyncio
async def test_a_real_review_interrupt_is_unchanged() -> None:
    """Regression: the HITL path is the one that already worked."""
    called: dict[str, Any] = {}

    async def _fake_panel(payload: Any) -> dict[str, Any]:
        called["payload"] = payload
        return {"proposed_edges": []}

    runner = _runner()
    runner._build_panel = _fake_panel  # type: ignore[method-assign]
    review = {"tenant_id": _TENANT, "learner_gcid": "g", "upload_id": _UPLOAD, "edges": []}

    result = await runner._result(
        {"__interrupt__": [_Interrupt(review)], "run_id": "run-1"},
        thread_id=_THREAD,
        fallback_run_id="run-1",
    )

    assert result.awaiting_agent is False
    assert result.review_panel == {"proposed_edges": []}
    assert called["payload"] == review
