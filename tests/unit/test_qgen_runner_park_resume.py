"""RED: the qgen runners on the ADR-254 lanes (park / resume / settle).

On the Pub/Sub transport a qgen graph run ends at its first agent dispatch
(the park) and continues only when the completion arrives on another call,
possibly on another pod. The runners therefore split into:

  * ``handle_started``: build the state, drive the graph until it PARKS (no
    terminal published) or reaches a terminal (settle), and re-drive
    idempotently (an already-terminal thread settles without re-invoking, a
    parked one is left for its completion);
  * ``handle_completion(completion, started_event=...)``: resume the parked
    thread inline and hand back a SETTLE thunk when the graph reached a
    terminal (the acceptance layer runs it as a tracked background drive, so
    no model call ever sits on the completion ack path).

The router resolves the started event for a completion from the in-flight
registry (the durable copy that outlives the drive) and routes exactly as it
routes a start. The legacy per-candidate N-loop is retired: every batch rides
the chunked set lane (a plan-less legacy batch gets the same single-quota
plan chora-creation itself stamps, image flags included).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from chora_ai_kernel_orchestrator.adapter.agent_io import LANE_ROLE_RENDER
from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    GuardrailScreenInput,
    ScreenResult,
    Verdict,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    DISPATCH_INTERRUPT_KEY,
    STATUS_FAILED,
    STATUS_OK,
    build_dispatch_request,
    wrap_for_interrupt,
)
from chora_ai_kernel_orchestrator.orchestrators.image_regen_graph import (
    build_image_regen_graph,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_CRITIQUE,
    ROLE_GENERATE,
    build_qgen_crew_graph,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
    ImageRegenRunner,
    QGenBatchRunner,
    QGenCrewRunner,
    QGenRunnerRouter,
    ResumeOutcome,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_durable_acceptance import (
    DurableQGenAcceptance,
)

_JOB = "0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"
_TENANT = "11111111-1111-7111-8111-111111111111"
_GCID = "00000000-0000-7000-8000-000000001999"


def _mcq(stem: str, *, specs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cand: dict[str, Any] = {
        "stem": stem,
        "question_type": "mcq",
        "mcq_payload": {
            "options": [
                {"option_id": "a", "label": "A", "text": "x", "is_correct": True, "explainer": "x"},
                {"option_id": "b", "label": "B", "text": "y", "is_correct": False, "explainer": "y"},
            ],
            "scoring_mode": "single_correct",
        },
    }
    if specs is not None:
        cand["image_specs"] = specs
    return cand


def _wrapper(cands: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "candidates": cands,
            "generation_summary": {
                "requested_total": 0,
                "generated_total": len(cands),
                "generated_per_type": {},
                "shortfall_reason": "",
            },
        }
    )


def _verdicts(decoded: dict[str, Any]) -> str:
    return json.dumps(
        {
            "verdicts": [
                {"candidate_id": c["candidate_id"], "accepted": True, "critique_notes": "", "suggested_revisions": []}
                for c in decoded.get("candidates") or []
            ]
        }
    )


def _accept() -> str:
    return json.dumps({"accepted": True, "critique_notes": "ok", "suggested_revisions": []})


class _ParkingExecutor:
    """Every role parks through the REAL ``interrupt()`` (as PubSubAgentExecutor
    does) and maps the resume value like it: OK -> output_payload; FAILED ->
    AgentDispatchError."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        agid: str,
        agent_role: str,
        input_payload: str,
        workflow_id: str = "",
        prompt_template_id: str = "",
        context_window: Any = None,
        available_tools: Any = None,
    ) -> AgentExecutorResponse:
        self.calls.append(
            {
                "execution_id": execution_id,
                "agent_role": agent_role,
                "workflow_id": workflow_id,
                "input_payload": input_payload,
            }
        )
        from langgraph.config import get_config

        thread_id = str(get_config()["configurable"]["thread_id"])
        request = build_dispatch_request(
            agent_role=agent_role,
            execution_id=execution_id,
            tenant_id=tenant_id,
            gcid=_GCID,
            thread_id=thread_id,
            input_payload=input_payload,
            workflow_id=workflow_id,
            agid=agid,
            source_project="chora-489812",
        )
        completion = interrupt(wrap_for_interrupt(request))
        from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import AgentDispatchError

        if str(completion.get("status")) != STATUS_OK:
            raise AgentDispatchError(f"{agent_role} {execution_id} status={completion.get('status')}")
        return AgentExecutorResponse(execution_id=execution_id, output_payload=str(completion["output_payload"]))


@dataclass
class _Guardrail:
    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
        return ScreenResult(verdict=Verdict.ALLOW, reason="clean")


@dataclass
class _Publisher:
    completed: list[dict[str, Any]] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)

    async def publish_completed(self, **kw: Any) -> str:
        self.completed.append(kw)
        return "c"

    async def publish_refused(self, **kw: Any) -> str:
        self.refused.append(kw)
        return "r"


@dataclass
class _Kroki:
    calls: list[str] = field(default_factory=list)

    async def render(self, *, source: str, output_format: str = "png") -> bytes:
        self.calls.append(source)
        return b"PNG"


@dataclass
class _Gcs:
    # Mirrors GcsImageUploadAdapter.bucket_name. The runner pins a caller-supplied
    # source URI to the bucket it writes to, so a fake without this fails closed.
    bucket_name: str = "bkt"
    uploads: int = 0
    signed: list[str] = field(default_factory=list)

    async def upload_and_sign(self, *, tenant_id: str, job_id: str, data: bytes, content_type: str) -> tuple[str, str]:
        self.uploads += 1
        return f"gs://bkt/{job_id}/{self.uploads}.png", f"https://signed/{self.uploads}"

    async def sign_read_url(self, gs_uri: str) -> str:
        self.signed.append(gs_uri)
        return f"https://signed-for/{gs_uri.rsplit('/', 1)[-1]}"


@dataclass
class _Downloader:
    present: set[str] = field(default_factory=set)

    async def exists(self, gs_uri: str) -> bool:
        return gs_uri in self.present


def _event(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "assist_id": _JOB,
        "tenant_id": _TENANT,
        "author_gcid": _GCID,
        "content_type": "mcq",
        "question_type": "mcq",
        "prompt": "Photosynthesis",
        "metadata": {"subject": "Biology"},
        "max_retries": 3,
        "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01",
    }
    base.update(over)
    return base


def _completion(park: Any, *, output: str, status: str = STATUS_OK) -> dict[str, Any]:
    req = park.value[DISPATCH_INTERRUPT_KEY]
    return {
        "agent_role": req["body"]["agent_role"],
        "execution_id": req["body"]["execution_id"],
        "thread_id": req["body"]["thread_id"],
        "idempotency_key": req["idempotency_key"],
        "status": status,
        "output_payload": output,
        "tenant_id": _TENANT,
    }


async def _parks(graph: Any, thread_id: str) -> list[Any]:
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return [i for t in snap.tasks for i in t.interrupts]


# -----------------------------------------------------------------------------
# single lane
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_runner_parks_on_generate_then_resumes_to_a_published_terminal() -> None:
    ex, pub = _ParkingExecutor(), _Publisher()
    graph = build_qgen_crew_graph(executor=ex, guardrail=_Guardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=pub)
    event = _event()
    await runner.handle_started(event)
    assert pub.completed == [] and pub.refused == [], "a parked run publishes no terminal"
    parks = await _parks(graph, _JOB)
    assert len(parks) == 1 and parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]["agent_role"] == ROLE_GENERATE

    out = await runner.handle_completion(_completion(parks[0], output=json.dumps(_mcq("q"))), started_event=event)
    assert isinstance(out, ResumeOutcome) and out.parked and out.settle is None, "parked again on critique"
    parks = await _parks(graph, _JOB)
    assert parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]["agent_role"] == ROLE_CRITIQUE

    out = await runner.handle_completion(_completion(parks[0], output=_accept()), started_event=event)
    assert not out.parked and out.settle is not None
    assert pub.completed == [], "the settle is handed back, not run inline"
    await out.settle()
    assert len(pub.completed) == 1
    ev = pub.completed[0]
    assert ev["assist_id"] == _JOB and ev["traceparent"] == event["traceparent"]
    assert json.loads(ev["candidate_payload_json"])["stem"] == "q"


@pytest.mark.asyncio
async def test_single_runner_redrive_of_a_parked_thread_does_not_dispatch_again() -> None:
    ex, pub = _ParkingExecutor(), _Publisher()
    graph = build_qgen_crew_graph(executor=ex, guardrail=_Guardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=pub)
    await runner.handle_started(_event())
    n = len(ex.calls)
    await runner.handle_started(_event())  # the boot sweep re-driving a parked job
    assert len(ex.calls) == n and pub.completed == []


@pytest.mark.asyncio
async def test_single_runner_redrive_of_a_terminal_thread_settles_without_reinvoking() -> None:
    ex, pub = _ParkingExecutor(), _Publisher()
    graph = build_qgen_crew_graph(executor=ex, guardrail=_Guardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=pub)
    event = _event()
    await runner.handle_started(event)
    parks = await _parks(graph, _JOB)
    await runner.handle_completion(_completion(parks[0], output=json.dumps(_mcq("q"))), started_event=event)
    parks = await _parks(graph, _JOB)
    out = await runner.handle_completion(_completion(parks[0], output=_accept()), started_event=event)
    assert out.settle is not None
    # The pod died before the settle ran; the sweep re-drives from the started payload.
    n = len(ex.calls)
    await runner.handle_started(event)
    assert len(ex.calls) == n, "a finished thread is never re-invoked"
    assert len(pub.completed) == 1, "the re-drive settles the finished run"


@pytest.mark.asyncio
async def test_single_runner_failed_completion_surfaces_as_the_crew_failure_branch() -> None:
    ex, pub = _ParkingExecutor(), _Publisher()
    graph = build_qgen_crew_graph(executor=ex, guardrail=_Guardrail(), checkpointer=MemorySaver())
    runner = QGenCrewRunner(graph=graph, publisher=pub)
    event = _event()
    await runner.handle_started(event)
    parks = await _parks(graph, _JOB)
    out = await runner.handle_completion(
        _completion(parks[0], output="", status=STATUS_FAILED),
        started_event=event,
    )
    assert out.settle is not None
    await out.settle()
    assert len(pub.refused) == 1 and pub.refused[0]["refusal_reason"] == "VALIDATION"


# -----------------------------------------------------------------------------
# set lane (every batch rides it; the N-loop is retired)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_runner_set_lane_parks_per_dispatch_and_settles_with_summary() -> None:
    ex, pub = _ParkingExecutor(), _Publisher()
    graph = build_qgen_crew_graph(
        executor=ex, guardrail=_Guardrail(), checkpointer=MemorySaver(), kroki=_Kroki(), gcs=_Gcs()
    )
    runner = QGenBatchRunner(graph=graph, publisher=pub)
    event = _event(
        content_type="mixed",
        question_type="mixed",
        job_kind="batch",
        requested_count=2,
        type_plan=[{"question_type": "mcq", "count": 2, "max_images": 1}],
    )
    await runner.handle_started(event)
    assert pub.completed == []
    parks = await _parks(graph, _JOB)
    assert parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]["execution_id"] == f"{_JOB}:generate_set:c0:r0"
    cands = [_mcq("a", specs=[{"mode": "scene", "source": "pic a", "placement": "stem"}]), _mcq("b")]
    out = await runner.handle_completion(_completion(parks[0], output=_wrapper(cands)), started_event=event)
    assert out.parked
    parks = await _parks(graph, _JOB)
    req = parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]
    assert req["execution_id"] == f"{_JOB}:critique_set:c0:r0"
    out = await runner.handle_completion(
        _completion(parks[0], output=_verdicts(json.loads(req["input_payload"]))),
        started_event=event,
    )
    assert out.parked, "parked on the scene render"
    parks = await _parks(graph, _JOB)
    req = parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]
    assert req["agent_role"] == LANE_ROLE_RENDER and req["execution_id"] == f"{_JOB}:render:c0:i0:s0:stem"
    out = await runner.handle_completion(
        _completion(parks[0], output=json.dumps({"image_uri": "gs://b/a.png", "mime_type": "image/png"})),
        started_event=event,
    )
    assert not out.parked and out.settle is not None
    await out.settle()
    assert len(pub.completed) == 1
    ev = pub.completed[0]
    payload = json.loads(ev["candidate_payload_json"])
    assert [c["stem"] for c in payload["candidates"]] == ["a", "b"]
    assert payload["candidates"][0]["image_gcs_uri"] == "gs://b/a.png"
    assert ev["generated_count"] == 2 and ev["generation_summary"]["requested_total"] == 2


@pytest.mark.asyncio
async def test_set_lane_with_compose_parks_on_the_compose_dispatch_and_publishes_the_proposal() -> None:
    """ADR-254 D2: the test-set composer is ONE mode=compose park on the
    qgen_generate lane after finalize_set; its completion settles the object
    wrapper with the repaired proposal and the compose_test_set trace row."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import ROLE_GENERATE
    from chora_ai_kernel_orchestrator.orchestrators.qgen_grounding import deterministic_draft_id

    ex, pub = _ParkingExecutor(), _Publisher()
    graph = build_qgen_crew_graph(executor=ex, guardrail=_Guardrail(), checkpointer=MemorySaver())
    runner = QGenBatchRunner(graph=graph, publisher=pub, compose_enabled=True)
    event = _event(
        content_type="mcq",
        question_type="mcq",
        job_kind="batch",
        requested_count=2,
        type_plan=[{"question_type": "mcq", "count": 2, "max_images": 0}],
    )
    await runner.handle_started(event)
    parks = await _parks(graph, _JOB)
    out = await runner.handle_completion(
        _completion(parks[0], output=_wrapper([_mcq("a"), _mcq("b")])),
        started_event=event,
    )
    assert out.parked
    parks = await _parks(graph, _JOB)
    req = parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]
    out = await runner.handle_completion(
        _completion(parks[0], output=_verdicts(json.loads(req["input_payload"]))),
        started_event=event,
    )
    assert out.parked, "parked on the compose dispatch"
    parks = await _parks(graph, _JOB)
    req = parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]
    assert req["execution_id"] == f"{_JOB}:compose" and req["agent_role"] == ROLE_GENERATE
    d0, d1 = deterministic_draft_id(_JOB, 0), deterministic_draft_id(_JOB, 1)
    sent = json.loads(req["input_payload"])
    assert sent["mode"] == "compose"
    assert [c["draft_id"] for c in sent["candidates"]] == [d0, d1]
    assert sent["author_prompt"] == "Photosynthesis" and sent["metadata"] == {"subject": "Biology"}
    proposal = {"title": "T", "description": "D", "order": [d1, d0], "points": {d0: 3, d1: 4}}
    out = await runner.handle_completion(
        _completion(parks[0], output=json.dumps({"proposed_test_set": proposal})),
        started_event=event,
    )
    assert not out.parked and out.settle is not None
    await out.settle()
    payload = json.loads(pub.completed[0]["candidate_payload_json"])
    assert payload["proposed_test_set"] == proposal
    assert [c["draft_id"] for c in payload["candidates"]] == [d0, d1]
    rows = [r for r in json.loads(pub.completed[0]["pipeline_trace_json"]) if r["name"] == "compose_test_set"]
    assert rows and rows[-1]["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_batch_without_type_plan_rides_the_set_lane_with_a_synthesized_plan() -> None:
    """The legacy N-loop is retired. A plan-less batch (pre-compose v1 shape)
    gets the SAME single-quota plan chora-creation stamps for count>1 today,
    including the author's image opt-ins, and runs chunked like any set."""
    ex, pub = _ParkingExecutor(), _Publisher()
    graph = build_qgen_crew_graph(executor=ex, guardrail=_Guardrail(), checkpointer=MemorySaver())
    runner = QGenBatchRunner(graph=graph, publisher=pub)
    event = _event(
        job_kind="batch",
        requested_count=3,
        image_for_stem=True,
        grounding_mode="strict",
        source_blob_uri="gs://uploads/t/j/material.pdf",
        source_mime_type="application/pdf",
    )
    await runner.handle_started(event)
    parks = await _parks(graph, _JOB)
    req = parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]
    assert req["execution_id"] == f"{_JOB}:generate_set:c0:r0"
    shipped = json.loads(req["input_payload"])
    assert shipped["set_mode"] is True
    assert shipped["type_plan"] == [{"question_type": "mcq", "count": 3, "max_images": 0, "image_for_stem": True}]
    assert shipped["source_blob_uri"] == "gs://uploads/t/j/material.pdf"
    assert not [c for c in ex.calls if c["execution_id"].endswith(":generate:1")], "no per-candidate loop"


# -----------------------------------------------------------------------------
# image regen (a small graph: author the spec, render it)
# -----------------------------------------------------------------------------


def _regen_event(**over: Any) -> dict[str, Any]:
    base = _event(
        job_kind="image_regen",
        prompt="Regenerate the stem illustration.",
        regen={"draft_id": "draft-7", "placement": "stem", "prompt": "clearer diagram", "mode": "scene"},
    )
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_image_regen_parks_on_author_then_render_then_publishes_the_patch() -> None:
    ex, pub, gcs = _ParkingExecutor(), _Publisher(), _Gcs()
    graph = build_image_regen_graph(executor=ex, kroki=_Kroki(), gcs=gcs, checkpointer=MemorySaver())
    runner = ImageRegenRunner(graph=graph, publisher=pub, kroki=_Kroki(), gcs=gcs, image_downloader=_Downloader())
    event = _regen_event()
    await runner.handle_started(event)
    assert pub.completed == [] and pub.refused == []
    parks = await _parks(graph, _JOB)
    req = parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]
    assert req["agent_role"] == ROLE_GENERATE and req["execution_id"] == f"{_JOB}:image_regen:author"
    assert json.loads(req["input_payload"])["intent"] == "image_regen"
    out = await runner.handle_completion(
        _completion(parks[0], output=json.dumps({"image_spec": {"mode": "scene", "source": "a crisp diagram"}})),
        started_event=event,
    )
    assert out.parked
    parks = await _parks(graph, _JOB)
    req = parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]
    assert req["agent_role"] == LANE_ROLE_RENDER and req["execution_id"] == f"{_JOB}:image_regen:render"
    shipped = json.loads(req["input_payload"])
    assert shipped["render_prompt"] == "a crisp diagram" and "source_image_uri" not in shipped
    out = await runner.handle_completion(
        _completion(parks[0], output=json.dumps({"image_uri": "gs://b/new.png", "mime_type": "image/png"})),
        started_event=event,
    )
    assert not out.parked and out.settle is not None
    await out.settle()
    assert len(pub.completed) == 1
    patch = json.loads(pub.completed[0]["candidate_payload_json"])
    assert patch == [
        {
            "draft_id": "draft-7",
            "placement": "stem",
            "image_url": "https://signed-for/new.png",
            "image_gcs_uri": "gs://b/new.png",
        }
    ]
    assert gcs.signed == ["gs://b/new.png"]


@pytest.mark.asyncio
async def test_image_regen_edit_carries_the_source_image_by_reference() -> None:
    ex, pub, gcs = _ParkingExecutor(), _Publisher(), _Gcs()
    graph = build_image_regen_graph(executor=ex, kroki=_Kroki(), gcs=gcs, checkpointer=MemorySaver())
    dl = _Downloader(present={f"gs://bkt/tenants/{_TENANT}/jobs/{_JOB}/old.png"})
    runner = ImageRegenRunner(graph=graph, publisher=pub, kroki=_Kroki(), gcs=gcs, image_downloader=dl)
    event = _regen_event(
        regen={
            "draft_id": "d",
            "placement": "answer",
            "prompt": "warmer colours",
            "mode": "scene",
            "original_image_gcs_uri": f"gs://bkt/tenants/{_TENANT}/jobs/{_JOB}/old.png",
        }
    )
    await runner.handle_started(event)
    parks = await _parks(graph, _JOB)
    await runner.handle_completion(
        _completion(parks[0], output=json.dumps({"image_spec": {"mode": "scene", "source": "warm diagram"}})),
        started_event=event,
    )
    parks = await _parks(graph, _JOB)
    shipped = json.loads(parks[0].value[DISPATCH_INTERRUPT_KEY]["body"]["input_payload"])
    assert shipped["source_image_uri"] == f"gs://bkt/tenants/{_TENANT}/jobs/{_JOB}/old.png"
    assert shipped["source_image_mime"] == "image/png"


@pytest.mark.asyncio
async def test_image_regen_refuses_when_the_original_is_gone_before_any_dispatch() -> None:
    ex, pub, gcs = _ParkingExecutor(), _Publisher(), _Gcs()
    graph = build_image_regen_graph(executor=ex, kroki=_Kroki(), gcs=gcs, checkpointer=MemorySaver())
    runner = ImageRegenRunner(graph=graph, publisher=pub, kroki=_Kroki(), gcs=gcs, image_downloader=_Downloader())
    event = _regen_event(
        regen={
            "draft_id": "d",
            "placement": "stem",
            "prompt": "p",
            "mode": "scene",
            "original_image_gcs_uri": f"gs://bkt/tenants/{_TENANT}/jobs/{_JOB}/gone.png",
        }
    )
    await runner.handle_started(event)
    assert ex.calls == [], "no model call for an edit whose original is gone (ADR-210 D3)"
    assert len(pub.refused) == 1 and pub.refused[0]["refusal_reason"] == "image_regen_original_unavailable"


@pytest.mark.asyncio
async def test_image_regen_mermaid_spec_renders_in_process_without_a_second_park() -> None:
    ex, pub, gcs, kroki = _ParkingExecutor(), _Publisher(), _Gcs(), _Kroki()
    graph = build_image_regen_graph(executor=ex, kroki=kroki, gcs=gcs, checkpointer=MemorySaver())
    runner = ImageRegenRunner(graph=graph, publisher=pub, kroki=kroki, gcs=gcs, image_downloader=_Downloader())
    event = _regen_event()
    await runner.handle_started(event)
    parks = await _parks(graph, _JOB)
    out = await runner.handle_completion(
        _completion(parks[0], output=json.dumps({"image_spec": {"mode": "mermaid", "source": "graph TD; A-->B"}})),
        started_event=event,
    )
    assert not out.parked and out.settle is not None
    await out.settle()
    assert kroki.calls == ["graph TD; A-->B"] and gcs.uploads == 1
    assert json.loads(pub.completed[0]["candidate_payload_json"])[0]["image_url"] == "https://signed/1"


@pytest.mark.asyncio
async def test_image_regen_failed_author_completion_refuses_loudly() -> None:
    ex, pub, gcs = _ParkingExecutor(), _Publisher(), _Gcs()
    graph = build_image_regen_graph(executor=ex, kroki=_Kroki(), gcs=gcs, checkpointer=MemorySaver())
    runner = ImageRegenRunner(graph=graph, publisher=pub, kroki=_Kroki(), gcs=gcs, image_downloader=_Downloader())
    event = _regen_event()
    await runner.handle_started(event)
    parks = await _parks(graph, _JOB)
    out = await runner.handle_completion(_completion(parks[0], output="", status=STATUS_FAILED), started_event=event)
    assert out.settle is not None
    await out.settle()
    assert pub.refused[0]["refusal_reason"] == "image_regen_author_failed"


# -----------------------------------------------------------------------------
# router + acceptance
# -----------------------------------------------------------------------------


@dataclass
class _Registry:
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def register(
        self,
        *,
        assist_id: str,
        tenant_id: str,
        author_gcid: str,
        started_event: dict[str, Any],
    ) -> None:
        self.rows[assist_id] = {
            "assist_id": assist_id,
            "tenant_id": tenant_id,
            "author_gcid": author_gcid,
            "started_payload": started_event,
            "resume_count": 0,
        }

    async def get(self, assist_id: str) -> dict[str, Any] | None:
        return self.rows.get(assist_id)


class _RecordingRunner:
    def __init__(self, *, settle: bool) -> None:
        self.started: list[dict[str, Any]] = []
        self.completions: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.settled = 0
        self._settle = settle

    async def handle_started(self, event: dict[str, Any]) -> None:
        self.started.append(event)

    async def handle_completion(self, completion: dict[str, Any], *, started_event: dict[str, Any]) -> ResumeOutcome:
        self.completions.append((completion, started_event))
        if not self._settle:
            return ResumeOutcome(parked=True, settle=None)

        async def _settle() -> None:
            self.settled += 1

        return ResumeOutcome(parked=False, settle=_settle)


@pytest.mark.asyncio
async def test_router_resolves_the_started_event_from_the_registry_and_routes_by_it() -> None:
    single = _RecordingRunner(settle=False)
    batch, regen = _RecordingRunner(settle=False), _RecordingRunner(settle=False)
    registry = _Registry()
    router = QGenRunnerRouter(single=single, batch=batch, image_regen=regen, inflight_registry=registry)
    await registry.register(
        assist_id=_JOB,
        tenant_id=_TENANT,
        author_gcid=_GCID,
        started_event=_event(job_kind="batch", requested_count=2, type_plan=[{"question_type": "mcq", "count": 2}]),
    )
    completion = {
        "agent_role": ROLE_GENERATE,
        "thread_id": _JOB,
        "idempotency_key": "k",
        "status": STATUS_OK,
        "output_payload": "{}",
        "execution_id": f"{_JOB}:generate_set:c0:r0",
    }
    out = await router.handle_completion(completion)
    assert out.parked and batch.completions and batch.completions[0][1]["job_kind"] == "batch"
    assert single.completions == [] and regen.completions == []


@pytest.mark.asyncio
async def test_router_refuses_a_completion_whose_job_is_not_in_flight() -> None:
    router = QGenRunnerRouter(
        single=_RecordingRunner(settle=False),
        batch=_RecordingRunner(settle=False),
        image_regen=None,
        inflight_registry=_Registry(),
    )
    with pytest.raises(LookupError):
        await router.handle_completion(
            {
                "agent_role": ROLE_GENERATE,
                "thread_id": "unknown-job",
                "idempotency_key": "k",
                "status": STATUS_OK,
                "output_payload": "{}",
            }
        )


@pytest.mark.asyncio
async def test_router_refuses_a_completion_without_a_thread_id() -> None:
    router = QGenRunnerRouter(
        single=_RecordingRunner(settle=False),
        batch=_RecordingRunner(settle=False),
        image_regen=None,
        inflight_registry=_Registry(),
    )
    with pytest.raises(ValueError):
        await router.handle_completion({"agent_role": ROLE_GENERATE, "idempotency_key": "k", "status": STATUS_OK})


@pytest.mark.asyncio
async def test_acceptance_runs_the_settle_as_a_tracked_background_drive() -> None:
    single = _RecordingRunner(settle=True)
    registry = _Registry()
    router = QGenRunnerRouter(
        single=single, batch=_RecordingRunner(settle=False), image_regen=None, inflight_registry=registry
    )
    acceptance = DurableQGenAcceptance(registry=registry, router=router)
    await registry.register(assist_id=_JOB, tenant_id=_TENANT, author_gcid=_GCID, started_event=_event())
    await acceptance.handle_completion(
        {
            "agent_role": ROLE_CRITIQUE,
            "thread_id": _JOB,
            "idempotency_key": "k",
            "status": STATUS_OK,
            "output_payload": _accept(),
        }
    )
    assert acceptance.is_driving(_JOB), "the settle runs as a drive the shutdown drain waits for"
    await acceptance.wait_idle()
    assert single.settled == 1
