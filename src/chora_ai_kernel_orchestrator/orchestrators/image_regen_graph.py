"""Image regenerate graph (CHO-1822 / ADR-210 / ADR-254 D2, D12).

Re-authors + re-renders ONE image (stem|answer) for a single candidate draft of
a parent batch job, THROUGH the qgen agents on the Pub/Sub lanes:

    author_spec  (qgen_generate, intent=image_regen)  -> image_spec {mode, source}
         |
    render       (mermaid: Kroki + the kennel's upload; scene: ONE qgen_render
                  dispatch carrying the CURRENT image by reference for an EDIT,
                  the agent's gs:// object signed by the kennel)
         |
        END      (the runner publishes the 1-element image patch, or refused.v1)

Before ADR-254 this was a plain async handler awaiting the agent over HTTP
and the gateway in-process. On the dispatch transport each agent call is a
PARK, so the flow is a small LangGraph StateGraph (checkpointed, thread id =
the regen job's assist id): the author dispatch parks, its completion resumes
into the render step, a scene render parks again, its completion reaches END.
Fail-soft at the job level (never strand it): an author/render failure is
recorded as a ``refusal_reason`` the runner turns into refused.v1; a park is
control flow and is never swallowed (``reraise_if_dispatch_park`` first).
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    reraise_if_dispatch_park,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_GENERATE,
    _current_w3c_trace_context,
    render_spec,
)

logger = logging.getLogger(__name__)

# Refusal reasons (the creation terminal subscriber + the A+ author banner
# render these; keep the pre-ADR-254 vocabulary byte-identical).
REFUSAL_AUTHOR_FAILED = "image_regen_author_failed"
REFUSAL_AUTHOR_EMPTY = "image_regen_author_empty"
REFUSAL_RENDER_FAILED = "image_regen_render_failed"

_MSG_TRY_AGAIN = "The image could not be regenerated. Please try again."


class ImageRegenState(TypedDict, total=False):
    """Checkpointed state of one image-regen job."""

    assist_id: str
    tenant_id: str
    gcid: str
    question_type: str
    traceparent: str
    tracestate: str
    draft_id: str
    placement: str
    refinement_prompt: str
    current_stem: str
    current_model_answer: str
    original_mode: str
    original_source: str
    # ADR-210 image-to-image: the CURRENT image's gs:// object (+ mime), read
    # by the renderer by reference so the refinement EDITS it.
    original_image_gcs_uri: str
    original_image_mime: str
    # authored by qgen_generate (intent=image_regen): {"mode", "source"}
    spec: dict[str, str]
    image_url: str
    image_gcs_uri: str
    refusal_reason: str
    refusal_message: str


def _trace_ctx(state: ImageRegenState) -> dict[str, str]:
    ctx: dict[str, str] = {}
    if state.get("traceparent"):
        ctx["traceparent"] = str(state["traceparent"])
        if state.get("tracestate"):
            ctx["tracestate"] = str(state["tracestate"])
        return ctx
    return _current_w3c_trace_context()


async def author_spec_node(state: ImageRegenState, *, executor: Any) -> dict[str, Any]:
    """Dispatch the qgen generator (intent=image_regen) -> ``{mode, source}``.

    Reuses the qgen_question agent + the generate lane (no new deployment).
    ``original_mode`` rides the regen ``mode`` field; ``current_*`` +
    ``original_source`` are the CHO-1822 fields. The agent decides the final
    mode; we backstop to the original mode (then scene) if it emits an unknown
    one. The execution id is fixed per job so a resume rebuilds the same
    dispatch key."""
    assist_id = str(state.get("assist_id") or "")
    gen_input: dict[str, Any] = {
        "intent": "image_regen",
        "question_type": str(state.get("question_type") or "mcq"),
        "placement": str(state.get("placement") or ""),
        "refinement_prompt": str(state.get("refinement_prompt") or ""),
        "current_stem": str(state.get("current_stem") or ""),
        "current_model_answer": str(state.get("current_model_answer") or ""),
        "original_mode": str(state.get("original_mode") or ""),
        "original_source": str(state.get("original_source") or ""),
        "gcid": str(state.get("gcid") or ""),
        **_trace_ctx(state),
    }
    try:
        resp = await executor.execute(
            execution_id=f"{assist_id}:image_regen:author",
            tenant_id=str(state.get("tenant_id") or ""),
            agid=ROLE_GENERATE,
            agent_role=ROLE_GENERATE,
            input_payload=json.dumps(gen_input),
            workflow_id=assist_id,
            prompt_template_id="qgen_crew::image_regen",
        )
    except Exception as exc:
        reraise_if_dispatch_park(exc)
        logger.exception("image_regen_graph.author_failed", extra={"assist_id": assist_id})
        return {"refusal_reason": REFUSAL_AUTHOR_FAILED, "refusal_message": _MSG_TRY_AGAIN}
    raw: dict[str, Any] = {}
    try:
        parsed = json.loads(getattr(resp, "output_payload", "") or "{}")
        if isinstance(parsed, dict):
            raw = parsed
    except (TypeError, ValueError):
        raw = {}
    spec = raw.get("image_spec")
    if not isinstance(spec, dict):
        # Tolerate a flat {mode, source} emission (no image_spec wrapper).
        spec = raw if ("mode" in raw or "source" in raw) else {}
    mode = str(spec.get("mode") or "").strip().lower()
    source = str(spec.get("source") or "").strip()
    if mode not in ("mermaid", "scene"):
        mode = str(state.get("original_mode") or "").strip().lower()
        if mode not in ("mermaid", "scene"):
            mode = "scene"
    if not source:
        return {"refusal_reason": REFUSAL_AUTHOR_EMPTY, "refusal_message": _MSG_TRY_AGAIN}
    return {"spec": {"mode": mode, "source": source}}


async def render_node(state: ImageRegenState, *, executor: Any, kroki: Any, gcs: Any) -> dict[str, Any]:
    """Render the authored spec: Mermaid in-process (Kroki + upload), a scene
    as ONE qgen_render dispatch (a park) carrying the original image by
    reference for an edit; the returned gs:// is signed here."""
    if state.get("refusal_reason"):
        return {}
    spec = dict(state.get("spec") or {})
    assist_id = str(state.get("assist_id") or "")
    try:
        gs_uri, url = await render_spec(
            spec=spec,
            executor=executor,
            kroki=kroki,
            gcs=gcs,
            tenant_id=str(state.get("tenant_id") or ""),
            job_id=assist_id,
            gcid=str(state.get("gcid") or ""),
            trace_ctx=_trace_ctx(state),
            execution_id=f"{assist_id}:image_regen:render",
            workflow_id=assist_id,
            source_image_uri=str(state.get("original_image_gcs_uri") or ""),
            source_image_mime=str(state.get("original_image_mime") or ""),
        )
    except Exception as exc:
        reraise_if_dispatch_park(exc)
        logger.exception("image_regen_graph.render_failed", extra={"assist_id": assist_id})
        return {"refusal_reason": REFUSAL_RENDER_FAILED, "refusal_message": _MSG_TRY_AGAIN}
    return {"image_url": url, "image_gcs_uri": gs_uri}


def build_image_regen_graph(*, executor: Any, kroki: Any, gcs: Any, checkpointer: Any | None = None) -> Any:
    """Compile the image-regen StateGraph (thread id = the regen job's assist id)."""
    graph: StateGraph = StateGraph(ImageRegenState)

    async def _author(state: ImageRegenState) -> dict[str, Any]:
        return await author_spec_node(state, executor=executor)

    async def _render(state: ImageRegenState) -> dict[str, Any]:
        return await render_node(state, executor=executor, kroki=kroki, gcs=gcs)

    graph.add_node("author_spec", _author)
    graph.add_node("render", _render)
    graph.add_edge(START, "author_spec")
    graph.add_edge("author_spec", "render")
    graph.add_edge("render", END)
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


__all__ = [
    "REFUSAL_AUTHOR_EMPTY",
    "REFUSAL_AUTHOR_FAILED",
    "REFUSAL_RENDER_FAILED",
    "ImageRegenState",
    "author_spec_node",
    "build_image_regen_graph",
    "render_node",
]
