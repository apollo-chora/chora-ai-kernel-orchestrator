"""ImageRegenRunner unit tests (CHO-1822 / ADR-210 / ADR-254 D2, D12).

The runner consumes an ai_assist.started.v1 with job_kind="image_regen" + a
``regen`` spec and drives the image-regen graph (``image_regen_graph.py``):
the qgen generator is dispatched with ``intent=image_regen`` to author ONE
image_spec {mode, source} from the author's CURRENT (edited) stem (+ model
answer for answer placement) + the refinement instruction; the spec is then
rendered (``mode=="mermaid"`` -> Kroki + the kennel's upload + signed URL;
``mode=="scene"`` -> ONE qgen_render dispatch carrying the CURRENT image by
reference for an EDIT, the returned gs:// signed by the kennel) and the runner
publishes ai_assist.completed.v1 whose candidate_payload_json is a 1-element
image-patch array ``[{draft_id, placement, image_url, image_gcs_uri}]`` the
creation terminal subscriber applies to the parent candidate.

Fail-soft (never strand the job): a bad request / unwired client / missing
original / author error / empty spec / render error publishes refused.v1.
On the bus an agent call is a PARK, never a hang: the graph is checkpointed and
its completion resumes it (``test_qgen_runner_park_resume.py`` drives the
park/resume path; these tests use an in-process executor so each run reaches
its terminal inside ``handle_started``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from chora_ai_kernel_orchestrator.orchestrators.image_regen_graph import (
    build_image_regen_graph,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import ROLE_RENDER
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
    AiAssistStartedPayload,
    ImageRegenRunner,
    QGenRunnerRouter,
)


def _regen_event(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "assist_id": "regen-job-1",
        "tenant_id": "TEN",
        "author_gcid": "GCID",
        "content_type": "mcq",
        "question_type": "mcq",
        "prompt": "Regenerate the stem illustration.",
        "job_kind": "image_regen",
        "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "regen": {
            "draft_id": "draft-7",
            "placement": "stem",
            "prompt": "A clearer diagram of the light-dependent reactions",
            "mode": "mermaid",
        },
    }
    base.update(overrides)
    return base


class _Resp:
    def __init__(self, output_payload: str) -> None:
        self.output_payload = output_payload
        self.output_tokens = 0
        self.input_tokens = 0
        self.tokens_consumed_total = 0


class _FakeExecutor:
    """Serves BOTH lanes the regen graph dispatches on: the author call
    (qgen_question, intent=image_regen) returns a canned image_spec; a scene
    render (qgen_render) returns the gs:// object the renderer wrote."""

    def __init__(
        self,
        *,
        image_spec: dict[str, Any] | None = None,
        output_payload: str | None = None,
        raise_exc: Exception | None = None,
        render_uri: str = "gs://chora-ai-assist-images-dev/tenants/TEN/jobs/regen-job-1/scene.png",
        render_exc: Exception | None = None,
    ) -> None:
        if output_payload is None:
            spec = image_spec if image_spec is not None else {"mode": "scene", "source": "a clear photo"}
            output_payload = json.dumps({"image_spec": spec})
        self.output_payload = output_payload
        self.raise_exc = raise_exc
        self.render_uri = render_uri
        self.render_exc = render_exc
        self.calls: list[dict[str, Any]] = []
        self.render_calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> _Resp:
        if kwargs.get("agent_role") == ROLE_RENDER:
            self.render_calls.append(json.loads(kwargs["input_payload"]))
            if self.render_exc is not None:
                raise self.render_exc
            return _Resp(json.dumps({"image_uri": self.render_uri, "mime_type": "image/png"}))
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        return _Resp(self.output_payload)


class _FakeKroki:
    def __init__(self, *, data: bytes = b"MERMAIDPNG", raise_exc: Exception | None = None) -> None:
        self.data = data
        self.raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    async def render(self, *, source: str, output_format: str = "png") -> bytes:
        self.calls.append({"source": source, "output_format": output_format})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.data


class _FakeGcs:
    # Mirrors GcsImageUploadAdapter.bucket_name: the runner pins a caller-supplied
    # source URI to the SAME bucket it writes to, so the fake has to expose it.
    bucket_name = "chora-ai-assist-images-dev"

    def __init__(self, *, url: str = "https://signed.example/img.png") -> None:
        self.url = url
        self.calls: list[dict[str, Any]] = []
        self.signed: list[str] = []

    async def upload_and_sign(self, *, tenant_id: str, job_id: str, data: bytes, content_type: str) -> tuple[str, str]:
        self.calls.append({"tenant_id": tenant_id, "job_id": job_id, "content_type": content_type})
        gs_uri = f"gs://chora-ai-assist-images-dev/tenants/{tenant_id}/jobs/{job_id}/regen.png"
        return gs_uri, self.url

    async def sign_read_url(self, gs_uri: str) -> str:
        self.signed.append(gs_uri)
        return self.url


class _FakeDownloader:
    """ADR-210 D3: the ORIGINAL is checked by reference (``exists``) before any
    dispatch; the renderer reads it by gs:// itself, so no bytes move here."""

    def __init__(self, *, present: bool = True, raise_exc: Exception | None = None) -> None:
        self.present = present
        self.raise_exc = raise_exc
        self.calls: list[str] = []

    async def exists(self, gs_uri: str) -> bool:
        self.calls.append(gs_uri)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.present


class _FakeTerminalPublisher:
    def __init__(self) -> None:
        self.completed: list[dict[str, Any]] = []
        self.refused: list[dict[str, Any]] = []

    async def publish_completed(self, **kwargs: Any) -> str:
        self.completed.append(kwargs)
        return "outbox-completed"

    async def publish_refused(self, **kwargs: Any) -> str:
        self.refused.append(kwargs)
        return "outbox-refused"


def _runner(
    *,
    executor: Any,
    kroki: Any,
    gcs: Any,
    pub: Any,
    downloader: Any = None,
) -> ImageRegenRunner:
    graph = build_image_regen_graph(executor=executor, kroki=kroki, gcs=gcs, checkpointer=MemorySaver())
    return ImageRegenRunner(
        graph=graph,
        publisher=pub,
        kroki=kroki,
        gcs=gcs,
        image_downloader=downloader,
    )


# -----------------------------------------------------------------------------
# Payload decode (unchanged contract)
# -----------------------------------------------------------------------------


def test_payload_from_event_surfaces_regen() -> None:
    p = AiAssistStartedPayload.from_event(_regen_event())
    assert p.job_kind == "image_regen"
    assert p.regen == {
        "draft_id": "draft-7",
        "placement": "stem",
        "prompt": "A clearer diagram of the light-dependent reactions",
        "mode": "mermaid",
    }


def test_payload_from_event_regen_absent_defaults_none() -> None:
    p = AiAssistStartedPayload.from_event({"assist_id": "j", "content_type": "mcq"})
    assert p.regen is None


# -----------------------------------------------------------------------------
# Mode-aware render: mermaid -> kroki (in-process), scene -> qgen_render lane
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_regen_mermaid_renders_via_kroki() -> None:
    executor = _FakeExecutor(image_spec={"mode": "mermaid", "source": "flowchart LR; egg-->tadpole;"})
    kroki, gcs = _FakeKroki(), _FakeGcs(url="https://s/x.png")
    pub = _FakeTerminalPublisher()
    runner = _runner(executor=executor, kroki=kroki, gcs=gcs, pub=pub)

    await runner.handle_started(_regen_event())

    # The agent authored a mermaid spec -> kroki rendered it; no render dispatch.
    assert len(kroki.calls) == 1
    assert kroki.calls[0]["source"] == "flowchart LR; egg-->tadpole;"
    assert executor.render_calls == []
    assert len(gcs.calls) == 1
    assert len(pub.completed) == 1 and len(pub.refused) == 0
    patch = json.loads(pub.completed[0]["candidate_payload_json"])
    assert len(patch) == 1
    entry = patch[0]
    assert entry["draft_id"] == "draft-7"
    assert entry["placement"] == "stem"
    assert entry["image_url"] == "https://s/x.png"
    # ADR-210: the patch carries the NEW durable gs:// so a subsequent regen
    # edits THIS result (iterative image-to-image), not the stale original.
    assert entry["image_gcs_uri"].startswith("gs://chora-ai-assist-images-dev/")
    assert pub.completed[0]["traceparent"] == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


@pytest.mark.asyncio
async def test_image_regen_scene_renders_via_the_render_lane() -> None:
    executor = _FakeExecutor(image_spec={"mode": "scene", "source": "a photorealistic leaf cross-section"})
    kroki, gcs, pub = _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    runner = _runner(executor=executor, kroki=kroki, gcs=gcs, pub=pub)

    await runner.handle_started(
        _regen_event(regen={"draft_id": "d", "placement": "answer", "prompt": "x", "mode": "scene"})
    )

    assert len(executor.render_calls) == 1
    sent = executor.render_calls[0]
    assert sent["render_prompt"] == "a photorealistic leaf cross-section"
    assert sent["mode"] == "scene"
    assert sent["job_id"] == "regen-job-1"
    assert len(kroki.calls) == 0
    # The agent's gs:// is SIGNED by the kennel, never uploaded.
    assert gcs.calls == [] and gcs.signed == [executor.render_uri]
    assert len(pub.completed) == 1
    patch = json.loads(pub.completed[0]["candidate_payload_json"])
    assert len(patch) == 1
    entry = patch[0]
    assert entry["draft_id"] == "d"
    assert entry["placement"] == "answer"
    assert entry["image_url"] == "https://signed.example/img.png"
    assert entry["image_gcs_uri"] == executor.render_uri


# -----------------------------------------------------------------------------
# ADR-210: true image-to-image by reference + fail-loud on a missing original
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_regen_image_to_image_passes_the_original_by_reference() -> None:
    # A regen carrying original_image_gcs_uri hands the renderer the CURRENT
    # image by gs:// reference (+ mime) so the model EDITS it.
    #
    # The URI shape matters now that the runner pins the source (2026-08-23).
    # These fixtures used to name gs://chora-atom-media-dev/tenants/t/atoms/...,
    # a shape production never produces: the candidate's image_gcs_uri is
    # STAMPED BY THIS ORCHESTRATOR via GcsImageUploadAdapter, which always
    # writes tenants/{tenant}/jobs/{job}/ in the render bucket, and
    # chora-creation copies it verbatim into candidate_questions_jsonb. A
    # fixture encoding a shape the producer cannot emit is how a constraint
    # looks like a regression when it is actually correct.
    executor = _FakeExecutor(image_spec={"mode": "scene", "source": "restyle in crayon"})
    kroki, gcs, pub = _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    downloader = _FakeDownloader(present=True)
    runner = _runner(executor=executor, kroki=kroki, gcs=gcs, pub=pub, downloader=downloader)

    await runner.handle_started(
        _regen_event(
            regen={
                "draft_id": "d",
                "placement": "stem",
                "prompt": "crayon",
                "mode": "scene",
                "original_image_gcs_uri": "gs://chora-ai-assist-images-dev/tenants/TEN/jobs/parent-job/stem.png",
            }
        )
    )

    assert downloader.calls == ["gs://chora-ai-assist-images-dev/tenants/TEN/jobs/parent-job/stem.png"]
    assert len(executor.render_calls) == 1
    sent = executor.render_calls[0]
    assert sent["source_image_uri"] == "gs://chora-ai-assist-images-dev/tenants/TEN/jobs/parent-job/stem.png"
    assert sent["source_image_mime"] == "image/png"
    assert len(pub.completed) == 1 and len(pub.refused) == 0


@pytest.mark.asyncio
async def test_image_regen_original_missing_refuses_not_text_redraw() -> None:
    # ADR-210 D3 FAIL-LOUD: an original WAS requested but is gone (aged out of
    # the transient bucket TTL): REFUSE with an explicit author message BEFORE
    # any dispatch, NEVER a silent text-redraw of a different image.
    executor = _FakeExecutor(image_spec={"mode": "scene", "source": "restyle"})
    kroki, gcs, pub = _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    downloader = _FakeDownloader(present=False)
    runner = _runner(executor=executor, kroki=kroki, gcs=gcs, pub=pub, downloader=downloader)

    await runner.handle_started(
        _regen_event(
            regen={
                "draft_id": "d",
                "placement": "stem",
                "prompt": "crayon",
                "mode": "scene",
                "original_image_gcs_uri": "gs://chora-ai-assist-images-dev/tenants/TEN/jobs/parent-job/gone.png",
            }
        )
    )

    assert len(pub.refused) == 1 and len(pub.completed) == 0
    assert pub.refused[0]["refusal_reason"] == "image_regen_original_unavailable"
    assert executor.calls == [] and executor.render_calls == []  # nothing dispatched
    assert "no longer available" in pub.refused[0]["user_facing_message"].lower()


@pytest.mark.asyncio
async def test_image_regen_original_check_error_refuses() -> None:
    # A failing existence check is not a licence to redraw: refuse loud.
    executor = _FakeExecutor(image_spec={"mode": "scene", "source": "restyle"})
    pub = _FakeTerminalPublisher()
    downloader = _FakeDownloader(raise_exc=RuntimeError("storage 503"))
    runner = _runner(executor=executor, kroki=_FakeKroki(), gcs=_FakeGcs(), pub=pub, downloader=downloader)

    await runner.handle_started(
        _regen_event(
            regen={
                "draft_id": "d",
                "placement": "stem",
                "prompt": "crayon",
                "mode": "scene",
                "original_image_gcs_uri": "gs://chora-ai-assist-images-dev/tenants/TEN/jobs/parent-job/x.png",
            }
        )
    )

    assert len(pub.refused) == 1 and executor.calls == []


@pytest.mark.asyncio
async def test_image_regen_no_original_is_text_to_image_not_refusal() -> None:
    # No original_image_gcs_uri => first-pass text-to-image (NOT a refusal): the
    # render dispatch carries no source image and no existence check runs.
    executor = _FakeExecutor(image_spec={"mode": "scene", "source": "a leaf"})
    kroki, gcs, pub = _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    downloader = _FakeDownloader()
    runner = _runner(executor=executor, kroki=kroki, gcs=gcs, pub=pub, downloader=downloader)

    await runner.handle_started(
        _regen_event(regen={"draft_id": "d", "placement": "stem", "prompt": "x", "mode": "scene"})
    )

    assert downloader.calls == []  # no check when no original requested
    assert len(executor.render_calls) == 1
    assert "source_image_uri" not in executor.render_calls[0]
    assert len(pub.completed) == 1 and len(pub.refused) == 0


# -----------------------------------------------------------------------------
# Current edited context (Item 3) is dispatched to the agent
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_regen_dispatch_carries_current_context() -> None:
    executor = _FakeExecutor(image_spec={"mode": "scene", "source": "s"})
    runner = _runner(executor=executor, kroki=_FakeKroki(), gcs=_FakeGcs(), pub=_FakeTerminalPublisher())

    await runner.handle_started(
        _regen_event(
            question_type="oe",
            regen={
                "draft_id": "d9",
                "placement": "answer",
                "prompt": "show each step",
                "mode": "mermaid",
                "current_stem": "Explain photosynthesis.",
                "current_model_answer": "Plants convert light to glucose via the Calvin cycle.",
                "original_source": "flowchart TD; light-->glucose;",
            },
        )
    )

    assert len(executor.calls) == 1
    sent = json.loads(executor.calls[0]["input_payload"])
    assert sent["intent"] == "image_regen"
    assert sent["placement"] == "answer"
    assert sent["question_type"] == "oe"
    assert sent["refinement_prompt"] == "show each step"
    assert sent["current_stem"] == "Explain photosynthesis."
    assert sent["current_model_answer"] == "Plants convert light to glucose via the Calvin cycle."
    assert sent["original_mode"] == "mermaid"
    assert sent["original_source"] == "flowchart TD; light-->glucose;"
    # Same agent role as generation (no new deployment); the regen job is the
    # workflow the dispatch rides under.
    assert executor.calls[0]["agent_role"] == "qgen_question"
    assert executor.calls[0]["workflow_id"] == "regen-job-1"
    assert executor.calls[0]["execution_id"] == "regen-job-1:image_regen:author"


# -----------------------------------------------------------------------------
# Fail-soft (never strand the job)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_regen_author_failure_refuses_without_render() -> None:
    executor = _FakeExecutor(raise_exc=RuntimeError("agent dispatch failed"))
    kroki, gcs, pub = _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    runner = _runner(executor=executor, kroki=kroki, gcs=gcs, pub=pub)

    await runner.handle_started(_regen_event())

    assert len(pub.refused) == 1 and len(pub.completed) == 0
    assert pub.refused[0]["refusal_reason"] == "image_regen_author_failed"
    assert len(kroki.calls) == 0 and executor.render_calls == [] and gcs.calls == []


@pytest.mark.asyncio
async def test_image_regen_render_failure_refuses() -> None:
    executor = _FakeExecutor(image_spec={"mode": "mermaid", "source": "graph TD; A-->B;"})
    kroki = _FakeKroki(raise_exc=RuntimeError("kroki down"))
    gcs, pub = _FakeGcs(), _FakeTerminalPublisher()
    runner = _runner(executor=executor, kroki=kroki, gcs=gcs, pub=pub)

    await runner.handle_started(_regen_event())

    assert len(pub.completed) == 0 and len(pub.refused) == 1
    assert pub.refused[0]["refusal_reason"] == "image_regen_render_failed"
    assert gcs.calls == []


@pytest.mark.asyncio
async def test_image_regen_failed_render_dispatch_refuses() -> None:
    executor = _FakeExecutor(
        image_spec={"mode": "scene", "source": "a leaf"},
        render_exc=RuntimeError("qgen_render dispatch returned status=FAILED: image_blocked"),
    )
    pub = _FakeTerminalPublisher()
    runner = _runner(executor=executor, kroki=_FakeKroki(), gcs=_FakeGcs(), pub=pub)

    await runner.handle_started(
        _regen_event(regen={"draft_id": "d", "placement": "stem", "prompt": "x", "mode": "scene"})
    )

    assert len(pub.refused) == 1 and pub.refused[0]["refusal_reason"] == "image_regen_render_failed"


@pytest.mark.asyncio
async def test_image_regen_empty_author_output_refuses() -> None:
    executor = _FakeExecutor(image_spec={"mode": "mermaid", "source": ""})  # no source
    kroki, gcs, pub = _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    runner = _runner(executor=executor, kroki=kroki, gcs=gcs, pub=pub)

    await runner.handle_started(_regen_event())

    assert len(pub.refused) == 1 and len(pub.completed) == 0
    assert pub.refused[0]["refusal_reason"] == "image_regen_author_empty"
    assert len(kroki.calls) == 0


@pytest.mark.asyncio
async def test_image_regen_bad_request_refuses_without_dispatch() -> None:
    executor = _FakeExecutor()
    kroki, gcs, pub = _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    runner = _runner(executor=executor, kroki=kroki, gcs=gcs, pub=pub)

    await runner.handle_started(_regen_event(regen={"draft_id": "d", "placement": "sidebar", "prompt": "x"}))

    assert len(pub.refused) == 1 and len(pub.completed) == 0
    assert pub.refused[0]["refusal_reason"] == "image_regen_bad_request"
    assert len(executor.calls) == 0


@pytest.mark.asyncio
async def test_image_regen_unwired_refuses() -> None:
    pub = _FakeTerminalPublisher()
    executor = _FakeExecutor()
    runner = _runner(executor=executor, kroki=None, gcs=None, pub=pub)

    await runner.handle_started(_regen_event())

    assert len(pub.refused) == 1 and len(pub.completed) == 0
    assert pub.refused[0]["refusal_reason"] == "image_regen_unwired"
    assert executor.calls == []


# -----------------------------------------------------------------------------
# On the bus a slow agent is a PARK, not a hang (the live stuck-regen bug was an
# SSE read that never returned; a park is checkpointed and resumed/reaped).
# -----------------------------------------------------------------------------


class _ParkingExecutor:
    """execute() parks the run on a LangGraph interrupt, as the Pub/Sub
    executor does; nothing is awaited across the agent's turn."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return interrupt({"dispatch": kwargs["execution_id"]})


@pytest.mark.asyncio
async def test_image_regen_author_dispatch_parks_instead_of_hanging() -> None:
    pub = _FakeTerminalPublisher()
    executor = _ParkingExecutor()
    graph = build_image_regen_graph(executor=executor, kroki=_FakeKroki(), gcs=_FakeGcs(), checkpointer=MemorySaver())
    runner = ImageRegenRunner(graph=graph, publisher=pub, kroki=_FakeKroki(), gcs=_FakeGcs())

    await runner.handle_started(_regen_event())  # returns: the job is parked

    assert len(executor.calls) == 1
    assert pub.completed == [] and pub.refused == []
    snapshot = await graph.aget_state({"configurable": {"thread_id": "regen-job-1"}})
    assert snapshot.next, "the thread is parked on the author dispatch, awaiting its completion"


@pytest.mark.asyncio
async def test_router_routes_image_regen() -> None:
    class _Recording:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        async def handle_started(self, event: dict[str, Any]) -> None:
            self.events.append(event)

    single, batch, regen = _Recording(), _Recording(), _Recording()
    router = QGenRunnerRouter(single=single, batch=batch, image_regen=regen)

    await router.handle_started(_regen_event())  # job_kind=image_regen -> regen
    await router.handle_started({"assist_id": "s"})  # no job_kind -> single

    assert len(regen.events) == 1
    assert len(single.events) == 1
    assert len(batch.events) == 0


# -----------------------------------------------------------------------------
# The source URI is pinned to the caller's own objects BEFORE any dispatch
# (cross-tenant read, found by WP-A, routed 2026-08-23).
#
# `original_image_gcs_uri` is echoed from the client's request body; `tenant_id`
# is stamped by chora-creation from the authenticated request context. Only the
# first is attacker-shaped, which is what makes pinning to the second a real
# boundary. Behavioural half; the policy's own edges are in
# test_regen_source_uri_is_tenant_scoped.py.
# -----------------------------------------------------------------------------

_OTHER_TENANT = "22222222-2222-7222-8222-222222222222"


def _regen_event_with_original(uri: str, *, tenant_id: str = "TEN") -> dict[str, Any]:
    ev = _regen_event(tenant_id=tenant_id)
    ev["regen"] = {**ev["regen"], "mode": "scene", "original_image_gcs_uri": uri}
    return ev


@pytest.mark.asyncio
async def test_another_tenants_original_is_refused_before_any_dispatch() -> None:
    """The run must not reach the agent at all: a dispatch would hand the URI
    to a renderer whose GSA can read any bucket in the project."""
    ex, kroki, gcs, pub = _FakeExecutor(), _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    dl = _FakeDownloader(present=True)
    runner = _runner(executor=ex, kroki=kroki, gcs=gcs, pub=pub, downloader=dl)

    hostile = f"gs://chora-ai-assist-images-dev/tenants/{_OTHER_TENANT}/jobs/j/x.png"
    await runner.handle_started(_regen_event_with_original(hostile))

    assert len(pub.refused) == 1, "the job must settle as refused, not strand"
    assert pub.refused[0]["refusal_reason"] == "image_regen_source_not_permitted"
    assert ex.calls == [], "no dispatch may be published for a refused source URI"


@pytest.mark.asyncio
async def test_a_refused_source_uri_is_never_probed() -> None:
    """The existence probe confirms readability, so running it on an
    attacker-shaped URI is itself the discovery oracle. Constraint FIRST."""
    ex, kroki, gcs, pub = _FakeExecutor(), _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    dl = _FakeDownloader(present=True)
    runner = _runner(executor=ex, kroki=kroki, gcs=gcs, pub=pub, downloader=dl)

    await runner.handle_started(_regen_event_with_original("gs://someone-elses-bucket/tenants/TEN/jobs/j/x.png"))

    assert dl.calls == [], f"the probe touched a refused URI: {dl.calls}"


@pytest.mark.asyncio
async def test_the_refusal_message_does_not_echo_the_rejected_uri() -> None:
    """The message reaches the author; echoing the URI hands back the shape."""
    ex, kroki, gcs, pub = _FakeExecutor(), _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    runner = _runner(executor=ex, kroki=kroki, gcs=gcs, pub=pub, downloader=_FakeDownloader())

    await runner.handle_started(
        _regen_event_with_original(f"gs://secret-bucket/tenants/{_OTHER_TENANT}/jobs/secret-job/x.png")
    )

    blob = json.dumps(pub.refused[0])
    assert "secret-bucket" not in blob
    assert "secret-job" not in blob
    assert _OTHER_TENANT not in blob


@pytest.mark.asyncio
async def test_the_tenants_own_original_still_reaches_the_probe() -> None:
    """The fix must not break the legitimate edit path, and the probe still
    earns its place there: the bucket has a 7-day TTL, so an aged-out original
    is a real and common case that deserves its own clear message."""
    ex, kroki, gcs, pub = _FakeExecutor(), _FakeKroki(), _FakeGcs(), _FakeTerminalPublisher()
    dl = _FakeDownloader(present=False)
    runner = _runner(executor=ex, kroki=kroki, gcs=gcs, pub=pub, downloader=dl)

    own = "gs://chora-ai-assist-images-dev/tenants/TEN/jobs/older-job/x.png"
    await runner.handle_started(_regen_event_with_original(own))

    assert dl.calls == [own], "the tenant's own URI must still be probed"
    assert pub.refused[0]["refusal_reason"] == "image_regen_original_unavailable"
