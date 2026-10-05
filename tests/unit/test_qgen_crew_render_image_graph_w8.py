"""W8 — render_image graph wiring + generate-payload opt-in threading.

RED→GREEN per [[feedback-strict-tdd]].

Asserts:
  - build_qgen_crew_graph accepts image clients (kroki / gcs); a scene image is a qgen_render dispatch
    and wires render_image between quality_gate (ACCEPTED) and
    publish_completed.
  - The DORMANT default (no image_specs) leaves the live MCQ loop's trace
    shape unchanged (no extra trace row) — the accepted path still produces
    the canonical 7-row trace.
  - With image_specs present on the accepted candidate, the rendered image
    URL reaches the published candidate.
  - generate_node threads image_for_stem / image_for_answer into the
    generate executor payload (author opt-in, default False).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    GuardrailScreenInput,
    ScreenResult,
    Verdict,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_CRITIQUE,
    ROLE_GENERATE,
    build_qgen_crew_graph,
)


@dataclass
class _FakeExecutor:
    responses: dict[str, list[str]]
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
        context_window: list[dict[str, str]] | None = None,
        available_tools: list[str] | None = None,
    ) -> AgentExecutorResponse:
        self.calls.append({"agent_role": agent_role, "input_payload": input_payload})
        out = self.responses.get(agent_role, []).pop(0)
        return AgentExecutorResponse(
            execution_id=execution_id,
            output_payload=out,
            input_tokens=10,
            output_tokens=5,
            tokens_consumed_total=15,
            final_state="EXECUTION_FINAL_STATE_SUCCESS",
        )


@dataclass
class _FakeGuardrail:
    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
        return ScreenResult(verdict=Verdict.ALLOW, reason="clean")


@dataclass
class _FakeKroki:
    calls: list[Any] = field(default_factory=list)

    async def render(self, *, source: str, output_format: str = "png") -> bytes:
        self.calls.append(source)
        return b"<svg/>"


@dataclass
class _FakeGcs:
    url: str = "https://signed/x.png"
    calls: list[Any] = field(default_factory=list)

    async def upload_and_sign(self, **kwargs: Any) -> tuple[str, str]:
        self.calls.append(kwargs)
        return "gs://chora-ai-assist-images-dev/tenants/t/jobs/j/x.png", self.url


def _mcq(stem: str = "What is 2+2?", image_specs: Any = None) -> str:
    payload: dict[str, Any] = {
        "stem": stem,
        "question_type": "mcq",
        "mcq_payload": {"options": [], "scoring_mode": "single_correct"},
    }
    if image_specs is not None:
        payload["image_specs"] = image_specs
    return json.dumps(payload)


def _critic_accept() -> str:
    return json.dumps({"accepted": True, "critique_notes": "", "suggested_revisions": []})


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": "job-w8g",
        "tenant_id": "tenant-w8g",
        "gcid": "gcid-w8g",
        "prompt": "Generate a question",
        "question_type": "mcq",
        "metadata": {},
        "max_retries": 3,
    }
    base.update(overrides)
    return base


def _cfg(job_id: str = "job-w8g") -> dict[str, Any]:
    return {"configurable": {"thread_id": job_id}}


def _build(executor: Any, guardrail: Any, **image_clients: Any) -> Any:
    return build_qgen_crew_graph(
        executor=executor,
        guardrail=guardrail,
        checkpointer=MemorySaver(),
        **image_clients,
    )


# -----------------------------------------------------------------------------
# DORMANT — accepted path trace shape unchanged when no image_specs
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepted_path_trace_shape_unchanged_without_image_specs() -> None:
    """The live MCQ loop: render_image is wired but a no-op (no image_specs),
    so the canonical 7-row trace is byte-for-byte unchanged."""
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_mcq()], ROLE_CRITIQUE: [_critic_accept()]})
    kroki, gcs = _FakeKroki(), _FakeGcs()
    graph = _build(executor, _FakeGuardrail(), kroki=kroki, gcs=gcs)

    out = await graph.ainvoke(_state(), config=_cfg())

    names = [r["name"] for r in out["pipeline_trace"]]
    assert names == [
        "validate_input",
        "guardrail_pre",
        "generate",
        "guardrail_post",
        "critique",
        "quality_gate",
        "publish_completed",
    ]
    assert out.get("completed_candidate") is not None
    # No image clients touched on the dormant path.
    assert kroki.calls == [] and gcs.calls == []


@pytest.mark.asyncio
async def test_graph_builds_without_image_clients() -> None:
    """Back-compat: build_qgen_crew_graph still works with NO image clients
    (the live MCQ loop predates W8). Accepted path completes normally."""
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_mcq()], ROLE_CRITIQUE: [_critic_accept()]})
    graph = _build(executor, _FakeGuardrail())  # no image clients
    out = await graph.ainvoke(_state(), config=_cfg())
    assert out.get("completed_candidate") is not None
    names = [r["name"] for r in out["pipeline_trace"]]
    assert names[-1] == "publish_completed"


# -----------------------------------------------------------------------------
# Accepted path WITH image_specs → rendered URL on the published candidate
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepted_path_with_image_specs_sets_image_url() -> None:
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_mcq(image_specs=[{"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"}])],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    kroki, gcs = _FakeKroki(), _FakeGcs(url="https://signed/diag.png")
    graph = _build(executor, _FakeGuardrail(), kroki=kroki, gcs=gcs)

    out = await graph.ainvoke(_state(), config=_cfg())

    completed = out["completed_candidate"]
    decoded = json.loads(completed.payload_json)
    assert decoded["image_url"] == "https://signed/diag.png"
    assert "image_specs" not in decoded
    assert len(kroki.calls) == 1
    # render_image trace row is present when it actually rendered.
    names = [r["name"] for r in out["pipeline_trace"]]
    assert "render_image" in names


# -----------------------------------------------------------------------------
# generate_node threads the author per-image opt-in
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_payload_threads_image_opt_in_flags() -> None:
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_mcq()], ROLE_CRITIQUE: [_critic_accept()]})
    graph = _build(executor, _FakeGuardrail())
    await graph.ainvoke(_state(image_for_stem=True, image_for_answer=False), config=_cfg())
    gen = next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)
    payload = json.loads(gen["input_payload"])
    assert payload["image_for_stem"] is True
    assert payload["image_for_answer"] is False


@pytest.mark.asyncio
async def test_generate_payload_image_opt_in_defaults_false() -> None:
    """When the author opt-in keys are absent from state (the live MCQ loop),
    both flags default to False — no behaviour change."""
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_mcq()], ROLE_CRITIQUE: [_critic_accept()]})
    graph = _build(executor, _FakeGuardrail())
    await graph.ainvoke(_state(), config=_cfg())
    gen = next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)
    payload = json.loads(gen["input_payload"])
    assert payload["image_for_stem"] is False
    assert payload["image_for_answer"] is False


# -----------------------------------------------------------------------------
# CHO-1658 — generate_node threads intent + existing_question (model_answer_fill)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_payload_threads_intent_and_existing_question() -> None:
    """model_answer_fill: generate_node forwards intent + the author's
    existing_question into the executor input so the executor surfaces the
    author_stem / author_options / author_rubric / model_answer fill keys."""
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_mcq()], ROLE_CRITIQUE: [_critic_accept()]})
    graph = _build(executor, _FakeGuardrail())
    existing = {
        "stem": "Explain photosynthesis.",
        "oe_rubric": [{"criterion": "accuracy", "weight": 1.0}],
        "model_answer": "placeholder",
    }
    await graph.ainvoke(_state(intent="model_answer_fill", existing_question=existing), config=_cfg())
    gen = next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)
    payload = json.loads(gen["input_payload"])
    assert payload["intent"] == "model_answer_fill"
    assert payload["existing_question"] == existing


@pytest.mark.asyncio
async def test_generate_payload_intent_existing_question_absent_by_default() -> None:
    """The live new_question loop carries neither key in state ⇒ NEITHER is added
    to the generate payload (byte-stable; the executor then defaults intent to
    new_question and reads no existing_question)."""
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_mcq()], ROLE_CRITIQUE: [_critic_accept()]})
    graph = _build(executor, _FakeGuardrail())
    await graph.ainvoke(_state(), config=_cfg())
    gen = next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)
    payload = json.loads(gen["input_payload"])
    assert "existing_question" not in payload
    assert "intent" not in payload
