"""CHO-1973 (ADR-205 D4/D5) — crew runner: drive the graph to the HITL pause on
upload (building the review PANEL), then resume it from the panel-shape decision.

The runner is the load-bearing seam between the Pub/Sub subscription (one
WeaknessDocUploaded → ainvoke to the interrupt → build panel → ack) and the FE
resume route (Command(resume=panel-shape) → ainvoke to completion → publish, or
re-pause on reiterate with a refreshed panel). thread_id is DETERMINISTIC on
{tenant}:{upload} ONLY (no run_id) so a resume after a pod restart finds the
checkpointed run from tenant + upload alone.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew import (
    build_weakness_analyser_graph,
)
from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew_runner import (
    WeaknessAnalyserCrewRunner,
)
from tests.unit.test_weakness_analyser_crew_graph import _ports, _uploaded_state


def _runner(
    *, output_prices: dict[str, int] | None = None, **port_overrides: Any
) -> tuple[WeaknessAnalyserCrewRunner, dict[str, Any]]:
    ports = _ports(**port_overrides)
    graph = build_weakness_analyser_graph(checkpointer=InMemorySaver(), **ports)
    runner = WeaknessAnalyserCrewRunner(
        graph=graph, output_prices=output_prices or {"practice_test": 120, "study_aids": 60}
    )
    return runner, ports


def _event(**overrides: Any) -> dict[str, Any]:
    ev = _uploaded_state()
    # the subscriber hands the runner the decoded event (no run_id — the runner
    # mints it internally for attribution; it does NOT key the thread).
    ev.pop("run_id", None)
    ev.update(overrides)
    return ev


def test_handle_uploaded_runs_to_hitl_and_builds_panel() -> None:
    runner, ports = _runner()
    result = asyncio.run(runner.handle_uploaded(_event()))

    assert result.interrupted is True
    # deterministic 2-segment thread (no run_id tail)
    assert result.thread_id == "tenant-1:upload-1"
    assert ports["diagnoser"].calls == 1
    assert ports["publisher"].published == []
    # the raw interrupt payload still carries the diagnosed edges
    assert result.review_payload is not None
    assert any(e["concept_key"] == "adding-fractions" for e in result.review_payload["edges"])
    # CHO-1973 Wave A: the runner builds the canonical FE PANEL from candidate_edges
    panel = result.review_panel
    assert panel is not None
    assert panel["upload_id"] == "upload-1"
    assert panel["learner_gcid"] == "gcid-1"
    assert [pe["proposed_edge_id"] for pe in panel["proposed_edges"]] == ["pe-0"]
    assert panel["proposed_edges"][0]["concept_label"] == "adding fractions"
    # available_outputs priced from the injected price table (metered) / 0 (free)
    by_kind = {o["kind"]: o for o in panel["available_outputs"]}
    assert by_kind["focused_dose"]["mana_price"] == 0
    assert by_kind["practice_test"]["mana_price"] == 120
    assert by_kind["study_aids"]["mana_price"] == 60
    # no Familiar resolver wired -> empty familiar (consumption enriches on read)
    assert panel["familiar"] == {}


def test_panel_candidate_struggles_from_retained_below_threshold() -> None:
    # CHO-1973 Q2: end-to-end, a sub-threshold diagnosed concept surfaces in the
    # built panel as a bounded candidate struggle ({key, label}); the promoted
    # edge is unaffected.
    from tests.unit.test_weakness_analyser_crew_graph import FakeDiagnoserWithStruggle

    runner, _ports = _runner(diagnoser=FakeDiagnoserWithStruggle())
    result = asyncio.run(runner.handle_uploaded(_event()))

    panel = result.review_panel
    assert panel is not None
    assert [pe["concept_label"] for pe in panel["proposed_edges"]] == ["adding fractions"]
    assert panel["candidate_struggles"] == [{"concept_key": "borrowing", "concept_label": "borrowing"}]


def test_resume_confirm_completes_and_publishes() -> None:
    runner, ports = _runner()
    started = asyncio.run(runner.handle_uploaded(_event()))
    decision = {"action": "confirm", "edges": [{"proposed_edge_id": "pe-0", "decision": "accept"}]}

    final = asyncio.run(runner.resume(thread_id=started.thread_id, decision=decision))

    assert final.interrupted is False
    assert final.governance_status == "approved"
    assert final.review_panel is None  # completed -> no pending panel
    assert len(ports["publisher"].published) == 1


def test_resume_reiterate_repauses_with_refreshed_panel() -> None:
    runner, ports = _runner()
    started = asyncio.run(runner.handle_uploaded(_event()))
    decision = {"action": "reiterate", "added_struggles": ["long-division"]}

    again = asyncio.run(runner.resume(thread_id=started.thread_id, decision=decision))

    assert again.interrupted is True
    assert ports["diagnoser"].calls == 2
    assert ports["publisher"].published == []
    # the FE re-enters review with a fresh panel
    assert again.review_panel is not None
    assert again.review_panel["upload_id"] == "upload-1"


def test_resume_runs_the_selected_outputs_and_surfaces_them() -> None:
    # ADR-254 D5: the learner outputs are dispatched AFTER the analyzed publish
    # (one park per node in production; the fake task runner answers inline
    # here), then the outputs event fires from the graph. The edges are durable
    # first; the resume result carries what came back.
    runner, ports = _runner()
    started = asyncio.run(runner.handle_uploaded(_event()))
    decision = {
        "action": "confirm",
        "edges": [{"proposed_edge_id": "pe-0", "decision": "accept"}],
        "selected_outputs": ["study_aids"],
    }

    final = asyncio.run(runner.resume(thread_id=started.thread_id, decision=decision))

    assert final.governance_status == "approved"
    assert len(ports["publisher"].published) == 1  # edges durable (published)
    assert [c["task_kind"] for c in ports["task_runner"].calls] == ["study_aids"]
    assert [o["type"] for o in final.generated_outputs] == ["study_aids"]
    assert len(ports["outputs_publisher"].bodies) == 1
    assert final.interrupted is False and final.awaiting_agent is False


def test_blocked_upload_completes_without_pause_and_refunds() -> None:
    from tests.unit.test_weakness_analyser_crew_graph import FakeScreener

    runner, ports = _runner(screener=FakeScreener(input_decision="BLOCK"))
    result = asyncio.run(runner.handle_uploaded(_event()))

    # a safety BLOCK terminates the first ainvoke (no interrupt to wait on)
    assert result.interrupted is False
    assert result.governance_status == "blocked"
    assert result.review_panel is None
    assert len(ports["mana"].refunds) == 1
    assert ports["publisher"].published == []


class FakeFamiliarResolver:
    """Doubles a (future) consumption-gRPC Familiar resolver port."""

    async def resolve(self, *, tenant_id: str, gcid: str) -> dict[str, str]:
        return {"familiar_id": "fam-7", "name": "Ember", "species": "dragon"}


def test_runner_uses_familiar_resolver_when_wired() -> None:
    runner, _ports = _runner()
    runner_with_fam = WeaknessAnalyserCrewRunner(
        graph=runner._graph,
        output_prices={"practice_test": 1, "study_aids": 1},
        familiar_resolver=FakeFamiliarResolver(),
    )
    started = asyncio.run(runner_with_fam.handle_uploaded(_event()))
    assert started.review_panel is not None
    assert started.review_panel["familiar"] == {"familiar_id": "fam-7", "name": "Ember", "species": "dragon"}
