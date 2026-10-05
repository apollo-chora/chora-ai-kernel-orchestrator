"""RED: the qgen graph on the ADR-254 lanes.

Three contract changes the flip needs at the node level, pinned here against
the REAL compiled graph (MemorySaver, fake clients):

1. every dispatch passes ``workflow_id=job_id`` explicitly (qgen thread ids
   are assist ids, the outbox key is the job) and the set lane's execution
   ids carry the chunk index + regen round, so two chunks of one job can no
   longer collide on one idempotency key;
2. the scene image is no longer a gateway call inside the node: it is ONE
   ``qgen_render`` dispatch per image from a loop node (one park per image),
   the agent returns the gs:// object, the kennel signs it (ADR-254 D12);
   Mermaid stays in-process on Kroki + the kennel's own upload;
3. a render failure stays fail-soft per spec and LOUD (quality_warning), and
   the loop really parks and resumes through LangGraph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

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
    STATUS_OK,
    build_dispatch_request,
    wrap_for_interrupt,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
    AgentDispatchError,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_CRITIQUE,
    ROLE_GENERATE,
    build_qgen_crew_graph,
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


def _accept() -> str:
    return json.dumps({"accepted": True, "critique_notes": "fine", "suggested_revisions": []})


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


def _verdicts(decoded: dict[str, Any], *, reject_marker: str = "[REJECT]") -> str:
    out = []
    for c in decoded.get("candidates") or []:
        rej = reject_marker in str(c.get("stem") or "")
        out.append(
            {
                "candidate_id": c["candidate_id"],
                "accepted": not rej,
                "critique_notes": "dup" if rej else "",
                "suggested_revisions": [],
            }
        )
    return json.dumps({"verdicts": out})


@dataclass
class _Recorder:
    """Canned executor: generate/critique/render answers; records every call."""

    generate: list[str] = field(default_factory=list)
    critique: list[str] = field(default_factory=list)
    render: list[Any] = field(default_factory=list)  # str output or Exception
    set_mode: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)

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
                "agid": agid,
                "workflow_id": workflow_id,
                "input_payload": input_payload,
                "prompt_template_id": prompt_template_id,
            }
        )
        if agent_role == ROLE_GENERATE:
            out = self.generate.pop(0)
        elif agent_role == ROLE_CRITIQUE:
            if self.set_mode:
                out = _verdicts(json.loads(input_payload))
            else:
                out = self.critique.pop(0) if self.critique else _accept()
        elif agent_role == LANE_ROLE_RENDER:
            nxt = self.render.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            out = nxt
        else:
            raise RuntimeError(f"unexpected role {agent_role}")
        return AgentExecutorResponse(
            execution_id=execution_id, output_payload=out, tokens_consumed_total=11, input_tokens=7, output_tokens=4
        )

    def calls_for(self, role: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["agent_role"] == role]


@dataclass
class _Guardrail:
    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
        return ScreenResult(verdict=Verdict.ALLOW, reason="clean")


@dataclass
class _Kroki:
    calls: list[str] = field(default_factory=list)

    async def render(self, *, source: str, output_format: str = "png") -> bytes:
        self.calls.append(source)
        return b"PNG"


@dataclass
class _Gcs:
    uploads: list[tuple[str, str, bytes, str]] = field(default_factory=list)
    signed: list[str] = field(default_factory=list)

    async def upload_and_sign(self, *, tenant_id: str, job_id: str, data: bytes, content_type: str) -> tuple[str, str]:
        self.uploads.append((tenant_id, job_id, data, content_type))
        return (
            f"gs://bkt/tenants/{tenant_id}/jobs/{job_id}/{len(self.uploads)}.png",
            f"https://signed/{len(self.uploads)}",
        )

    async def sign_read_url(self, gs_uri: str) -> str:
        self.signed.append(gs_uri)
        return f"https://signed-for/{gs_uri.rsplit('/', 1)[-1]}"


def _render_ok(uri: str) -> str:
    return json.dumps(
        {
            "image_uri": uri,
            "mime_type": "image/png",
            "model_id": "gemini-3-pro-image",
            "prompt_version": "v1",
            "bytes": 3,
            "edit": False,
        }
    )


def _single_state(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": _JOB,
        "tenant_id": _TENANT,
        "gcid": _GCID,
        "prompt": "Photosynthesis",
        "question_type": "mcq",
        "metadata": {},
        "max_retries": 3,
        "pipeline_trace": [],
        "errors": [],
    }
    base.update(over)
    return base


def _set_state(count: int, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": _JOB,
        "tenant_id": _TENANT,
        "gcid": _GCID,
        "prompt": "Mixed set",
        "question_type": "mixed",
        "metadata": {},
        "set_mode": True,
        "type_plan": [{"question_type": "mcq", "count": count, "max_images": 2}],
        "max_regen_rounds": 1,
        "pipeline_trace": [],
        "errors": [],
    }
    base.update(over)
    return base


def _graph(executor: Any, *, kroki: Any = None, gcs: Any = None) -> Any:
    return build_qgen_crew_graph(
        executor=executor, guardrail=_Guardrail(), checkpointer=MemorySaver(), kroki=kroki, gcs=gcs
    )


# -----------------------------------------------------------------------------
# 1. workflow_id + chunk-scoped execution ids
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_lane_passes_workflow_id_and_keeps_attempt_scoped_ids() -> None:
    ex = _Recorder(generate=[json.dumps(_mcq("q"))])
    await _graph(ex).ainvoke(_single_state(), config={"configurable": {"thread_id": _JOB}})
    gen = ex.calls_for(ROLE_GENERATE)[0]
    crit = ex.calls_for(ROLE_CRITIQUE)[0]
    assert gen["workflow_id"] == _JOB and crit["workflow_id"] == _JOB
    assert gen["execution_id"] == f"{_JOB}:generate:1" and crit["execution_id"] == f"{_JOB}:critique:1"
    assert gen["agid"] == "qgen_question" and crit["agid"] == "qgen_critic"


@pytest.mark.asyncio
async def test_set_lane_execution_ids_carry_chunk_and_round(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QGEN_SET_MAX_PER_CALL", "2")
    ex = _Recorder(
        set_mode=True,
        generate=[
            _wrapper([_mcq("a"), _mcq("b [REJECT]")]),  # chunk 0 round 0
            _wrapper([_mcq("b2")]),  # chunk 0 round 1 (regen of the rejected one)
            _wrapper([_mcq("c")]),  # chunk 1 round 0
        ],
    )
    out = await _graph(ex).ainvoke(_set_state(3), config={"configurable": {"thread_id": _JOB}})
    assert not out.get("refusal_reason"), out.get("errors")
    gen_ids = [c["execution_id"] for c in ex.calls_for(ROLE_GENERATE)]
    crit_ids = [c["execution_id"] for c in ex.calls_for(ROLE_CRITIQUE)]
    assert gen_ids == [f"{_JOB}:generate_set:c0:r0", f"{_JOB}:generate_set:c0:r1", f"{_JOB}:generate_set:c1:r0"]
    assert crit_ids == [f"{_JOB}:critique_set:c0:r0", f"{_JOB}:critique_set:c0:r1", f"{_JOB}:critique_set:c1:r0"]
    assert all(c["workflow_id"] == _JOB for c in ex.calls)
    assert len(out["accepted_set"]) == 3


# -----------------------------------------------------------------------------
# 2. render: scene via qgen_render, mermaid in-process
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_lane_scene_image_is_a_qgen_render_dispatch_signed_by_the_kennel() -> None:
    spec = [{"mode": "scene", "source": "a red apple on a desk", "placement": "stem"}]
    ex = _Recorder(generate=[json.dumps(_mcq("q", specs=spec))], render=[_render_ok("gs://bkt/t/j/img1.png")])
    kroki, gcs = _Kroki(), _Gcs()
    out = await _graph(ex, kroki=kroki, gcs=gcs).ainvoke(_single_state(), config={"configurable": {"thread_id": _JOB}})
    render = ex.calls_for(LANE_ROLE_RENDER)
    assert len(render) == 1
    call = render[0]
    assert call["execution_id"] == f"{_JOB}:render:a1:s0:stem"
    assert call["workflow_id"] == _JOB and call["agid"] == "qgen_renderer"
    payload = json.loads(call["input_payload"])
    assert payload["render_prompt"] == "a red apple on a desk" and payload["mode"] == "scene"
    assert payload["job_id"] == _JOB and payload["gcid"] == _GCID
    assert "source_image_uri" not in payload
    # The kennel signs the agent's object; nothing was uploaded by the kennel.
    assert gcs.signed == ["gs://bkt/t/j/img1.png"] and gcs.uploads == [] and kroki.calls == []
    cand = json.loads(out["completed_candidate"].payload_json)
    assert cand["image_url"] == "https://signed-for/img1.png"
    assert cand["image_gcs_uri"] == "gs://bkt/t/j/img1.png"
    assert "image_specs" not in cand
    row = [r for r in out["pipeline_trace"] if r["name"] == "render_image"][-1]
    assert row["status"] == "COMPLETED" and row["notes"] == "rendered 1/1 image(s)"
    assert not out.get("quality_warning")


@pytest.mark.asyncio
async def test_single_lane_mermaid_stays_in_process_on_kroki() -> None:
    spec = [{"mode": "mermaid", "source": "graph TD; A-->B", "placement": "answer"}]
    ex = _Recorder(generate=[json.dumps(_mcq("q", specs=spec))])
    kroki, gcs = _Kroki(), _Gcs()
    out = await _graph(ex, kroki=kroki, gcs=gcs).ainvoke(_single_state(), config={"configurable": {"thread_id": _JOB}})
    assert ex.calls_for(LANE_ROLE_RENDER) == []
    assert kroki.calls == ["graph TD; A-->B"] and len(gcs.uploads) == 1 and gcs.signed == []
    cand = json.loads(out["completed_candidate"].payload_json)
    assert cand["answer_image_url"] == "https://signed/1" and cand["answer_image_gcs_uri"].startswith("gs://bkt/")


@pytest.mark.asyncio
async def test_render_failure_is_fail_soft_per_spec_and_raises_quality_warning() -> None:
    specs = [
        {"mode": "scene", "source": "stem pic", "placement": "stem"},
        {"mode": "mermaid", "source": "graph TD; A-->B", "placement": "answer"},
    ]
    ex = _Recorder(
        generate=[json.dumps(_mcq("q", specs=specs))],
        render=[AgentDispatchError("qgen_render dispatch returned status=FAILED: image_blocked")],
    )
    kroki, gcs = _Kroki(), _Gcs()
    out = await _graph(ex, kroki=kroki, gcs=gcs).ainvoke(_single_state(), config={"configurable": {"thread_id": _JOB}})
    cand = json.loads(out["completed_candidate"].payload_json)
    assert "image_url" not in cand and cand["answer_image_url"] == "https://signed/1"
    row = [r for r in out["pipeline_trace"] if r["name"] == "render_image"][-1]
    assert row["status"] == "DEGRADED" and row["notes"].startswith("rendered 1/2 image(s); 1 failed (fail-soft)")
    assert out["quality_warning"] is True
    assert any("render_image failed" in e for e in out["errors"])


@pytest.mark.asyncio
async def test_set_lane_renders_one_dispatch_per_image_and_reassembles() -> None:
    cands = [
        _mcq("a", specs=[{"mode": "scene", "source": "pic a", "placement": "stem"}]),
        _mcq("b"),
        _mcq("c", specs=[{"mode": "scene", "source": "pic c", "placement": "answer"}]),
    ]
    ex = _Recorder(
        set_mode=True,
        generate=[_wrapper(cands)],
        render=[_render_ok("gs://bkt/t/j/a.png"), _render_ok("gs://bkt/t/j/c.png")],
    )
    kroki, gcs = _Kroki(), _Gcs()
    out = await _graph(ex, kroki=kroki, gcs=gcs).ainvoke(_set_state(3), config={"configurable": {"thread_id": _JOB}})
    ids = [c["execution_id"] for c in ex.calls_for(LANE_ROLE_RENDER)]
    assert ids == [f"{_JOB}:render:c0:i0:s0:stem", f"{_JOB}:render:c0:i2:s0:answer"]
    payloads = [json.loads(c["input_payload"]) for c in ex.calls_for(LANE_ROLE_RENDER)]
    assert [p["render_prompt"] for p in payloads] == ["pic a", "pic c"]
    assert all(p["chunk_id"] == "c0" and p["job_id"] == _JOB for p in payloads)
    final = out["accepted_set"]
    assert [c["stem"] for c in final] == ["a", "b", "c"]
    assert final[0]["image_url"] == "https://signed-for/a.png" and final[0]["image_gcs_uri"] == "gs://bkt/t/j/a.png"
    assert "image_url" not in final[1] and "image_specs" not in final[1]
    assert final[2]["answer_image_url"] == "https://signed-for/c.png"
    row = [r for r in out["pipeline_trace"] if r["name"] == "render_image"][-1]
    assert row["notes"] == "rendered 2/2 image(s)" and row["status"] == "COMPLETED"
    assert not out.get("quality_warning")


# -----------------------------------------------------------------------------
# 3. the loop really parks: one interrupt per image, resumable
# -----------------------------------------------------------------------------


class _ParkingRenderer:
    """Generate/critique answer inline; qgen_render PARKS through the real
    ``interrupt()`` exactly as PubSubAgentExecutor does, so the loop node's
    park-per-image is exercised through LangGraph, not through a stub."""

    def __init__(self, generate: str) -> None:
        self._generate = generate
        self.resumed: list[Any] = []

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
        if agent_role == ROLE_GENERATE:
            return AgentExecutorResponse(execution_id=execution_id, output_payload=self._generate)
        if agent_role == ROLE_CRITIQUE:
            return AgentExecutorResponse(execution_id=execution_id, output_payload=_verdicts(json.loads(input_payload)))
        assert agent_role == LANE_ROLE_RENDER
        request = build_dispatch_request(
            agent_role=agent_role,
            execution_id=execution_id,
            tenant_id=tenant_id,
            gcid=_GCID,
            thread_id=_JOB,
            input_payload=input_payload,
            workflow_id=workflow_id,
            agid=agid,
            source_project="chora-489812",
        )
        completion = interrupt(wrap_for_interrupt(request))
        self.resumed.append(completion)
        return AgentExecutorResponse(execution_id=execution_id, output_payload=str(completion["output_payload"]))


@pytest.mark.asyncio
async def test_render_loop_parks_once_per_image_and_resumes_to_terminal() -> None:
    cands = [
        _mcq("a", specs=[{"mode": "scene", "source": "pic a", "placement": "stem"}]),
        _mcq("b", specs=[{"mode": "scene", "source": "pic b", "placement": "stem"}]),
    ]
    ex = _ParkingRenderer(_wrapper(cands))
    graph = _graph(ex, kroki=_Kroki(), gcs=_Gcs())
    cfg = {"configurable": {"thread_id": _JOB}}
    first = await graph.ainvoke(_set_state(2), config=cfg)
    parks = first.get("__interrupt__") or []
    assert len(parks) == 1, "one image, one park"
    req = parks[0].value[DISPATCH_INTERRUPT_KEY]
    assert req["body"]["execution_id"] == f"{_JOB}:render:c0:i0:s0:stem"
    second = await graph.ainvoke(
        Command(
            resume={"status": STATUS_OK, "output_payload": _render_ok("gs://b/a.png"), "agent_role": LANE_ROLE_RENDER}
        ),
        config=cfg,
    )
    parks2 = second.get("__interrupt__") or []
    assert len(parks2) == 1
    assert parks2[0].value[DISPATCH_INTERRUPT_KEY]["body"]["execution_id"] == f"{_JOB}:render:c0:i1:s0:stem"
    final = await graph.ainvoke(
        Command(
            resume={"status": STATUS_OK, "output_payload": _render_ok("gs://b/b.png"), "agent_role": LANE_ROLE_RENDER}
        ),
        config=cfg,
    )
    assert not final.get("__interrupt__")
    assert [c["image_gcs_uri"] for c in final["accepted_set"]] == ["gs://b/a.png", "gs://b/b.png"]
    assert len(ex.resumed) == 2


@pytest.mark.asyncio
async def test_render_with_specs_but_no_kroki_or_gcs_is_a_misconfig() -> None:
    spec = [{"mode": "scene", "source": "x", "placement": "stem"}]
    ex = _Recorder(generate=[json.dumps(_mcq("q", specs=spec))], render=[_render_ok("gs://b/x.png")])
    with pytest.raises(RuntimeError):
        await _graph(ex, kroki=None, gcs=None).ainvoke(_single_state(), config={"configurable": {"thread_id": _JOB}})


@pytest.mark.asyncio
async def test_graph_builder_has_no_gateway_client_any_more() -> None:
    import inspect

    params = inspect.signature(build_qgen_crew_graph).parameters
    assert "gateway" not in params, "the scene render is a qgen_render dispatch; no in-process gateway client"
