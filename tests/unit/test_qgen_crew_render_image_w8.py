"""W8 / ADR-254 D12: the render loop (additive + DORMANT) unit tests.

The render stage sits on the ACCEPTED path:
    quality_gate (ACCEPTED) -> render_image (PLAN) -> [render_next ...] ->
    render_finalize -> publish_completed

Behaviour:
  - NO image_specs on the accepted candidate => pure pass-through (no-op, no
    trace row, candidate unchanged). This is the DEFAULT: the live MCQ loop
    must be byte-for-byte unchanged.
  - image_specs present (a LIST of 0-2 entries, each {mode, source, placement}):
      * mode=="mermaid" -> POST source to Kroki -> the kennel's own upload + URL
      * mode=="scene"   -> ONE qgen_render dispatch (a park on the bus) -> the
        agent's gs:// object, SIGNED by the kennel (no upload)
      * placement=="stem"   -> candidate.image_url       (in payload_json)
      * placement=="answer" -> candidate.answer_image_url (in payload_json)
      * REMOVE image_specs from the published candidate
  - Fail-soft + loud: a render error logs (logger.exception) + publishes the
    candidate WITHOUT that image (the question is still valid) and raises
    quality_warning. Fail-LOUD only when a client is missing AND an image_spec
    IS present (mis-config).

These tests drive the three nodes the way the graph does (plan -> one
render per superstep -> finalize) and assert on the ACCUMULATED delta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.domain.qgen_crew import CandidatePayload
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_RENDER,
    render_finalize_node,
    render_image_node,
    render_next_node,
)

# -----------------------------------------------------------------------------
# Fakes for the injected clients (Kroki + GCS) and the qgen_render lane
# -----------------------------------------------------------------------------


@dataclass
class _FakeKroki:
    calls: list[dict[str, Any]] = field(default_factory=list)
    raises: BaseException | None = None
    out: bytes = b"<svg>diagram</svg>"

    async def render(self, *, source: str, output_format: str = "png") -> bytes:
        self.calls.append({"source": source, "output_format": output_format})
        if self.raises is not None:
            raise self.raises
        return self.out


@dataclass
class _FakeRenderExecutor:
    """The qgen_render lane: ONE dispatch per scene image; answers with the
    gs:// object the renderer wrote (the kennel signs it, never uploads)."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    raises: BaseException | None = None
    image_uri: str = "gs://chora-ai-assist-images-dev/tenants/t/jobs/j/scene.png"

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
        payload = json.loads(input_payload)
        self.calls.append(
            {
                "execution_id": execution_id,
                "tenant_id": tenant_id,
                "agid": agid,
                "agent_role": agent_role,
                "workflow_id": workflow_id,
                "prompt": payload.get("render_prompt"),
                "gcid": payload.get("gcid"),
                "mode": payload.get("mode"),
            }
        )
        if self.raises is not None:
            raise self.raises
        return AgentExecutorResponse(
            execution_id=execution_id,
            output_payload=json.dumps({"image_uri": self.image_uri, "mime_type": "image/png"}),
            tokens_consumed_total=0,
            cost_micros_total=0,
            final_state="EXECUTION_FINAL_STATE_SUCCEEDED",
        )


@dataclass
class _FakeGcs:
    calls: list[dict[str, Any]] = field(default_factory=list)
    signed: list[str] = field(default_factory=list)
    raises: BaseException | None = None
    url: str = "https://storage.googleapis.com/signed/url.png?X-Goog-Signature=abc"
    gs_uri: str = "gs://chora-ai-assist-images-dev/tenants/t/jobs/j/rendered.png"

    async def upload_and_sign(self, *, tenant_id: str, job_id: str, data: bytes, content_type: str) -> tuple[str, str]:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "job_id": job_id,
                "len": len(data),
                "content_type": content_type,
            }
        )
        if self.raises is not None:
            raise self.raises
        return self.gs_uri, self.url

    async def sign_read_url(self, gs_uri: str) -> str:
        self.signed.append(gs_uri)
        if self.raises is not None:
            raise self.raises
        return self.url


def _candidate(image_specs: Any = None, *, stem: str = "What is 2+2?") -> CandidatePayload:
    payload: dict[str, Any] = {
        "stem": stem,
        "question_type": "mcq",
        "mcq_payload": {"options": [], "scoring_mode": "single_correct"},
    }
    if image_specs is not None:
        payload["image_specs"] = image_specs
    return CandidatePayload(stem=stem, question_type="mcq", payload_json=json.dumps(payload))


def _state(candidate: CandidatePayload, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": "job-w8",
        "tenant_id": "tenant-w8",
        "gcid": "gcid-w8",
        "question_type": "mcq",
        "current_candidate": candidate,
        "attempt_count": 1,
    }
    base.update(overrides)
    return base


async def _render(state: dict[str, Any], *, executor: Any, kroki: Any, gcs: Any) -> dict[str, Any]:
    """Drive plan -> render_next (one image per superstep) -> finalize exactly
    like the graph's edges do; return the ACCUMULATED delta."""
    merged = dict(state)
    acc: dict[str, Any] = {}
    delta = await render_image_node(merged, kroki=kroki, gcs=gcs)
    merged.update(delta)
    acc.update(delta)
    while merged.get("pending_renders"):
        delta = await render_next_node(merged, executor=executor, kroki=kroki, gcs=gcs)
        merged.update(delta)
        acc.update(delta)
    delta = await render_finalize_node(merged)
    merged.update(delta)
    acc.update(delta)
    return acc


def _decoded_completed(delta: dict[str, Any]) -> dict[str, Any]:
    cand = delta["current_candidate"]
    return json.loads(cand.payload_json)


# -----------------------------------------------------------------------------
# DORMANT: no image_specs => pure pass-through
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_image_no_specs_is_pure_passthrough() -> None:
    """No image_specs => no client calls, candidate unchanged, EMPTY delta
    (no trace row added: the live MCQ loop's trace shape is unchanged)."""
    kroki, renderer, gcs = _FakeKroki(), _FakeRenderExecutor(), _FakeGcs()
    cand = _candidate(image_specs=None)
    delta = await _render(_state(cand), executor=renderer, kroki=kroki, gcs=gcs)
    assert delta == {}
    assert kroki.calls == []
    assert renderer.calls == []
    assert gcs.calls == [] and gcs.signed == []


@pytest.mark.asyncio
async def test_render_image_empty_list_is_pure_passthrough() -> None:
    """An explicit empty image_specs list is also a no-op."""
    kroki, renderer, gcs = _FakeKroki(), _FakeRenderExecutor(), _FakeGcs()
    cand = _candidate(image_specs=[])
    delta = await _render(_state(cand), executor=renderer, kroki=kroki, gcs=gcs)
    assert delta == {}
    assert kroki.calls == [] and renderer.calls == [] and gcs.calls == []


@pytest.mark.asyncio
async def test_render_image_no_current_candidate_passthrough() -> None:
    """Defensive: no current_candidate (shouldn't happen on accepted path) =>
    empty delta, no crash."""
    kroki, renderer, gcs = _FakeKroki(), _FakeRenderExecutor(), _FakeGcs()
    state = _state(_candidate())
    state["current_candidate"] = None
    delta = await _render(state, executor=renderer, kroki=kroki, gcs=gcs)
    assert delta == {}


# -----------------------------------------------------------------------------
# mermaid (stem) -> Kroki, in-process, the kennel uploads + signs
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_image_mermaid_stem_calls_kroki_and_sets_image_url() -> None:
    kroki, renderer, gcs = _FakeKroki(), _FakeRenderExecutor(), _FakeGcs()
    cand = _candidate(image_specs=[{"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"}])
    delta = await _render(_state(cand), executor=renderer, kroki=kroki, gcs=gcs)
    # Kroki called with the mermaid source; the render lane NOT dispatched.
    assert len(kroki.calls) == 1
    assert kroki.calls[0]["source"] == "graph TD; A-->B"
    assert renderer.calls == []
    # Uploaded + signed by the kennel.
    assert len(gcs.calls) == 1
    assert gcs.calls[0]["tenant_id"] == "tenant-w8"
    assert gcs.calls[0]["job_id"] == "job-w8"
    # image_url set on the candidate; image_specs removed.
    decoded = _decoded_completed(delta)
    assert decoded["image_url"] == gcs.url
    # ADR-210 B2: the canonical gs:// object path is stamped alongside the
    # signed display URL so the regen producer never parses the signed URL.
    assert decoded["image_gcs_uri"] == gcs.gs_uri
    assert "answer_image_url" not in decoded
    assert "answer_image_gcs_uri" not in decoded
    assert "image_specs" not in decoded
    names = [r["name"] for r in delta["pipeline_trace"]]
    assert names[-1] == "render_image"


# -----------------------------------------------------------------------------
# scene (answer) -> ONE qgen_render dispatch, signed by the kennel
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_image_scene_answer_dispatches_qgen_render_and_sets_answer_url() -> None:
    kroki, renderer, gcs = _FakeKroki(), _FakeRenderExecutor(), _FakeGcs()
    cand = _candidate(image_specs=[{"mode": "scene", "source": "a diagram of the water cycle", "placement": "answer"}])
    delta = await _render(_state(cand), executor=renderer, kroki=kroki, gcs=gcs)
    # The render lane dispatched ONCE with the scene prompt; Kroki NOT called.
    assert len(renderer.calls) == 1
    call = renderer.calls[0]
    assert call["agent_role"] == ROLE_RENDER and call["mode"] == "scene"
    assert call["prompt"] == "a diagram of the water cycle"
    assert call["tenant_id"] == "tenant-w8"
    assert call["gcid"] == "gcid-w8"
    assert call["workflow_id"] == "job-w8"
    assert call["execution_id"] == "job-w8:render:a1:s0:answer"
    assert kroki.calls == []
    # The agent's gs:// is SIGNED here, never uploaded.
    assert gcs.calls == []
    assert gcs.signed == [renderer.image_uri]
    # answer_image_url set; image_url NOT set; image_specs removed.
    decoded = _decoded_completed(delta)
    assert decoded["answer_image_url"] == gcs.url
    assert decoded["answer_image_gcs_uri"] == renderer.image_uri
    assert "image_url" not in decoded
    assert "image_gcs_uri" not in decoded
    assert "image_specs" not in decoded


# -----------------------------------------------------------------------------
# Both stem + answer (2 entries): the W8 refinement two-image path
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_image_two_specs_stem_and_answer() -> None:
    kroki, renderer, gcs = _FakeKroki(), _FakeRenderExecutor(), _FakeGcs()
    gcs.url = "https://signed/x.png"
    cand = _candidate(
        image_specs=[
            {"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"},
            {"mode": "scene", "source": "answer illustration", "placement": "answer"},
        ]
    )
    delta = await _render(_state(cand), executor=renderer, kroki=kroki, gcs=gcs)
    assert len(kroki.calls) == 1  # stem mermaid
    assert len(renderer.calls) == 1  # answer scene, ONE dispatch
    assert len(gcs.calls) == 1  # one upload (mermaid)
    assert len(gcs.signed) == 1  # one signing (scene)
    decoded = _decoded_completed(delta)
    assert decoded["image_url"] == "https://signed/x.png"
    assert decoded["answer_image_url"] == "https://signed/x.png"
    assert "image_specs" not in decoded
    assert not delta.get("quality_warning")


# -----------------------------------------------------------------------------
# Fail-soft: a render error publishes WITHOUT the image (question still valid)
# and LOUD (quality_warning + the reason recorded)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_image_kroki_error_fail_soft() -> None:
    kroki = _FakeKroki(raises=RuntimeError("kroki down"))
    renderer, gcs = _FakeRenderExecutor(), _FakeGcs()
    cand = _candidate(image_specs=[{"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"}])
    delta = await _render(_state(cand), executor=renderer, kroki=kroki, gcs=gcs)
    # The candidate is still published (image_specs stripped) WITHOUT image_url.
    decoded = _decoded_completed(delta)
    assert "image_url" not in decoded
    assert "image_specs" not in decoded
    # No GCS upload happened (render failed before upload).
    assert gcs.calls == []
    assert delta.get("quality_warning") is True
    assert "kroki down" in "; ".join(delta.get("errors") or [])


@pytest.mark.asyncio
async def test_render_image_gcs_error_fail_soft() -> None:
    kroki = _FakeKroki()
    gcs = _FakeGcs(raises=RuntimeError("gcs write failed"))
    renderer = _FakeRenderExecutor()
    cand = _candidate(image_specs=[{"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"}])
    delta = await _render(_state(cand), executor=renderer, kroki=kroki, gcs=gcs)
    decoded = _decoded_completed(delta)
    assert "image_url" not in decoded  # upload failed -> no URL
    assert "image_specs" not in decoded  # but spec still stripped


@pytest.mark.asyncio
async def test_render_image_failed_dispatch_fail_soft() -> None:
    """A FAILED qgen_render completion (the executor raises) keeps the
    question and records the reason; the job never strands on an image."""
    kroki = _FakeKroki()
    renderer = _FakeRenderExecutor(raises=RuntimeError("qgen_render dispatch returned status=FAILED: image_blocked"))
    gcs = _FakeGcs()
    cand = _candidate(image_specs=[{"mode": "scene", "source": "a junction", "placement": "stem"}])
    delta = await _render(_state(cand), executor=renderer, kroki=kroki, gcs=gcs)
    decoded = _decoded_completed(delta)
    assert "image_url" not in decoded and "image_specs" not in decoded
    assert gcs.signed == []
    assert delta.get("quality_warning") is True
    assert "image_blocked" in "; ".join(delta.get("errors") or [])


@pytest.mark.asyncio
async def test_render_image_one_fails_one_succeeds() -> None:
    """When one of two specs fails, the OTHER still renders (per-spec
    fail-soft, not all-or-nothing)."""
    kroki = _FakeKroki(raises=RuntimeError("kroki down"))  # stem fails
    renderer = _FakeRenderExecutor()  # answer succeeds
    gcs = _FakeGcs(url="https://signed/answer.png")
    cand = _candidate(
        image_specs=[
            {"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"},
            {"mode": "scene", "source": "answer illustration", "placement": "answer"},
        ]
    )
    delta = await _render(_state(cand), executor=renderer, kroki=kroki, gcs=gcs)
    decoded = _decoded_completed(delta)
    assert "image_url" not in decoded  # stem failed
    assert decoded["answer_image_url"] == "https://signed/answer.png"  # answer ok


# -----------------------------------------------------------------------------
# Fail-LOUD: env vars missing (client is None) AND image_spec present
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_image_missing_client_with_spec_fails_loud() -> None:
    """A None client (env var unset at composition root) + an image_spec
    present is a mis-config (NOT a transient error) -> raise at PLAN time."""
    cand = _candidate(image_specs=[{"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"}])
    with pytest.raises(RuntimeError, match="unconfigured"):
        await render_image_node(_state(cand), kroki=None, gcs=_FakeGcs())


@pytest.mark.asyncio
async def test_render_image_missing_client_no_spec_is_passthrough() -> None:
    """None clients + NO image_spec => still a pure pass-through (the
    common production case when the image feature is unconfigured)."""
    cand = _candidate(image_specs=None)
    delta = await _render(_state(cand), executor=None, kroki=None, gcs=None)
    assert delta == {}


@pytest.mark.asyncio
async def test_render_image_unknown_mode_fail_soft() -> None:
    """An unrecognised mode is fail-soft (logged, skipped): never crashes
    the job; the question publishes without that image."""
    kroki, renderer, gcs = _FakeKroki(), _FakeRenderExecutor(), _FakeGcs()
    cand = _candidate(image_specs=[{"mode": "hologram", "source": "x", "placement": "stem"}])
    delta = await _render(_state(cand), executor=renderer, kroki=kroki, gcs=gcs)
    decoded = _decoded_completed(delta)
    assert "image_url" not in decoded
    assert "image_specs" not in decoded
    assert kroki.calls == [] and renderer.calls == []


# -----------------------------------------------------------------------------
# One image per superstep (a shared-quota image model is never fanned into)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_next_renders_exactly_one_image_per_superstep() -> None:
    kroki, renderer, gcs = _FakeKroki(), _FakeRenderExecutor(), _FakeGcs()
    cand = _candidate(
        image_specs=[
            {"mode": "scene", "source": "stem scene", "placement": "stem"},
            {"mode": "scene", "source": "answer scene", "placement": "answer"},
        ]
    )
    state = _state(cand)
    plan = await render_image_node(state, kroki=kroki, gcs=gcs)
    assert [p["execution_id"] for p in plan["pending_renders"]] == [
        "job-w8:render:a1:s0:stem",
        "job-w8:render:a1:s1:answer",
    ]
    state.update(plan)
    step = await render_next_node(state, executor=renderer, kroki=kroki, gcs=gcs)
    assert len(renderer.calls) == 1, "one dispatch per superstep"
    assert [p["placement"] for p in step["pending_renders"]] == ["answer"]
    assert len(step["render_results"]) == 1
