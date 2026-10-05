"""companion_diagnosis crew (ex Growth-Edge weakness analyser): the governed
multimodal HITL StateGraph, fully on the bus (ADR-205 WS-2 -> ADR-254 D2/D5/D12).

    reserve_mana -> safesearch -> extract -> screen_input -> diagnose
    -> critic_verify -> screen_output -> HITL_review (interrupt)
    -> synthesize_edges -> publish_analyzed -> emit_evidence
    -> study_aids -> practice_test -> publish_outputs -> END

"Reiterate" loops HITL_review -> diagnose carrying the learner's STRUCTURED
decisions as constraints (never free text). Any fail-loud node and any safety
BLOCK route to ``refund`` (the reservation taken at the upload door is
returned, ADR-205 D6) -> END.

What moved onto the bus (ADR-254 D5, deterministic kernel):
  * ``extract`` is DISPATCHED to the ``companion_extract`` agent BY REFERENCE
    (``source_blob_uri`` + ``source_mime_type``); no bytes ride the bus and no
    bytes enter the checkpoint. Cloud Vision SafeSearch runs in the kennel on
    the RAW bytes BEFORE that dispatch (D12), so an image is downloaded for the
    inspection only and the bytes stay node-local; a non-image upload is never
    downloaded at all.
  * ``diagnose`` is dispatched to ``companion_diagnose`` (``task_kind=diagnose``).
  * the learner outputs ``study_aids`` and ``practice_test`` are dispatched to
    the same agent by ``task_kind``, ONE PARK PER NODE, after the analyzed event
    and the D1 evidence are durable; the outputs event fires from the graph
    (``publish_outputs``), not from a detached task. A FAILED / BLOCKED /
    malformed output is dropped LOUDLY and never sinks the analysis; an empty
    outputs list still publishes so consumption can tell "generated nothing"
    from "not generated yet".

Every dispatch PARKS the run (LangGraph ``interrupt()`` raises
``GraphInterrupt``, an Exception subclass): each ``except Exception`` around a
dispatch re-raises the park FIRST (``reraise_if_dispatch_park``); the AST guard
in ``tests/unit/test_dispatch_park_not_swallowed.py`` enumerates the verbs.

Hexagonal: the graph depends ONLY on the port Protocols below. The real
adapters fill them: ``adapter/weakness/pubsub_dispatch.py`` (extractor,
diagnoser, output tasks), Cloud Model Armor (ADR-152), Cloud Vision, the
binary outbox writers. The pure parse + clue rendering live in
``domain/weakness_analyser_crew.analysis``.

The contract this honours is the FROZEN ``weakness.analyzed.v1`` body shape and
the ``weakness.outputs_generated.v1`` outputs shape (``{type, content,
metered}``), additive proto fields only.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    reraise_if_dispatch_park,
)
from chora_ai_kernel_orchestrator.domain.weakness_analyser_crew.analysis import (
    MAX_EDGES,
    PANEL_DIFFICULTY_STRENGTH,
    UPLOAD_KIND_MARKED_TEST,
    parse_analysis,
    render_structured_clues_block,
)
from chora_ai_kernel_orchestrator.domain.weakness_analyser_crew.panel import (
    OUTPUT_KINDS,
    index_from_proposed_edge_id,
)

logger = logging.getLogger(__name__)

# Calling-agent id (cost attribution + the Model Armor template mapping key).
AGENT_ID = "weakness_analyser_crew"

# Cap reiterations so a stuck review loop cannot burn the recursion budget; after
# this the crew proceeds with the latest diagnosis (the learner still reviews).
MAX_REITERATIONS = 3

# Governance terminal states surfaced to O+ / the resume route.
GOV_APPROVED = "approved"
GOV_BLOCKED = "blocked"
GOV_FAILED = "failed"

# The two orchestrator-produced learner outputs, dispatched by task_kind on the
# companion_diagnose role (ADR-254 D6 addendum). The other selectable kinds
# (focused_dose, familiar_coaching) are CONSUMPTION-SIDE seams performed from
# the published edges; they are logged here, never produced.
OUTPUT_STUDY_AIDS = "study_aids"
OUTPUT_PRACTICE_TEST = "practice_test"
DEFAULT_PRACTICE_TEST_MAX_QUESTIONS = 8
_SEAM_SELECTIONS = ("focused_dose", "familiar_coaching")


# --------------------------------------------------------------------------- #
# port results + Protocols (adapters satisfy these)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExtractResult:
    """Vision-to-text extraction of the uploaded artifact (the extractor agent)."""

    text: str
    model_used: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class DiagnoseResult:
    """The diagnoser agent's structured output (the edges JSON) + attribution."""

    diagnosis_json: str
    model_used: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ScreenVerdict:
    """Cloud Model Armor verdict for one screened text (ADR-152)."""

    decision: str  # "ALLOW" | "BLOCK" | "INSPECT_ONLY"
    explanation: str = ""


class ManaPort(Protocol):
    """Reserve -> refund mana economy (ADR-205 D6 / ADR-178). The reservation
    is taken at the upload HTTP door (WS-4); the crew confirms it then settles
    on success / refunds on any fail-loud or BLOCK."""

    async def ensure_reserved(self, *, reservation_id: str, tenant_id: str, gcid: str) -> None: ...
    async def refund(self, *, reservation_id: str, tenant_id: str, gcid: str, reason: str) -> None: ...


class BlobDownloader(Protocol):
    """Downloads the raw upload for the SafeSearch inspection ONLY (images)."""

    async def download(self, gs_uri: str) -> bytes: ...


class Extractor(Protocol):
    """The ``companion_extract`` agent, BY REFERENCE: gs:// uri + mime in,
    plain transcription text out (ADR-254 D6 addendum). The call PARKS."""

    async def extract(
        self,
        *,
        source_blob_uri: str,
        source_mime_type: str,
        tenant_id: str,
        gcid: str,
        traceparent: str,
        tracestate: str,
    ) -> ExtractResult: ...


class SafeSearchPort(Protocol):
    """Cloud Vision SafeSearch on the RAW image (explicit/CSAM gate, ADR-205 D2)."""

    async def inspect(self, *, blob_bytes: bytes, mime_type: str) -> bool: ...


class Screener(Protocol):
    """Cloud Model Armor on extracted text + clues (input), the diagnosis and
    every learner-facing generated output (output)."""

    async def screen(self, *, text: str, tenant_id: str, gcid: str, direction: str) -> ScreenVerdict: ...


class Diagnoser(Protocol):
    """The ``companion_diagnose`` agent (task_kind=diagnose): extracted text +
    clue DATA -> edges JSON. The call PARKS."""

    async def diagnose(
        self,
        *,
        extracted_text: str,
        structured_clues_block: str,
        upload_kind: str,
        tenant_id: str,
        gcid: str,
        traceparent: str,
        tracestate: str,
    ) -> DiagnoseResult: ...


class OutputTaskRunner(Protocol):
    """The ``companion_diagnose`` agent by ``task_kind`` (study_aids /
    practice_test): published edges in, the agent's output TEXT out. The call
    PARKS; a FAILED completion surfaces as an exception."""

    async def run_task(
        self,
        *,
        task_kind: str,
        edges: list[dict[str, Any]],
        tenant_id: str,
        gcid: str,
        traceparent: str,
        tracestate: str,
        max_questions: int = DEFAULT_PRACTICE_TEST_MAX_QUESTIONS,
    ) -> str: ...


class OutputsPublisher(Protocol):
    """Binary transactional-outbox emitter for weakness.outputs_generated.v1."""

    async def publish_outputs_generated(
        self, *, body: dict[str, Any], traceparent: str = "", tracestate: str = ""
    ) -> str: ...


class EvidenceEmitter(Protocol):
    """D1 accountability evidence for the ``emit_evidence`` node (WS-3).

    The real adapter (``adapter/pubsub/weakness_evidence_emitter.py``) emits one
    ``observability.agent_decision.logged.v1`` row attributed to the diagnoser.
    The other IMDA dimensions are NOT emitted here: **D2** transparency rides
    ``publish_analyzed`` (-> ``weakness.analyzed.v1`` -> the governance
    decision_explanation projector); **D4** fairness is the offline crew-scoped
    ``BiasMetric`` + the HITL gate; per-token **cost** is the model gateway
    (ADR-163 sole producer of ``token_usage.recorded.v1``)."""

    async def emit(
        self,
        *,
        body: dict[str, Any],
        input_decision: str,
        output_decision: str,
        model_used: str,
        traceparent: str = "",
        tracestate: str = "",
    ) -> None: ...


class AnalyzedPublisher(Protocol):
    """Binary transactional-outbox emitter for weakness.analyzed.v1 (unchanged)."""

    async def publish_weakness_analyzed(
        self, *, body: dict[str, Any], traceparent: str = "", tracestate: str = ""
    ) -> str: ...


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #


class WeaknessAnalyserState(TypedDict, total=False):
    """LangGraph state for one Growth-Edge analysis. Partial dicts merge.

    Deliberately NO ``blob_bytes``: the raw upload never enters the checkpoint
    (a park at extract would otherwise persist a multi-MB blob per run)."""

    # caller-supplied (from WeaknessDocUploaded)
    run_id: str
    tenant_id: str
    learner_gcid: str
    upload_id: str
    source_blob_uri: str
    source_mime_type: str
    upload_kind: str
    structured_clues: dict[str, Any]
    requested_outputs: dict[str, Any]
    reservation_id: str
    context_hint: str
    traceparent: str
    tracestate: str

    # stage outputs
    extracted_text: str
    extract_tokens_in: int
    extract_tokens_out: int
    input_decision: str
    diagnosis_json: str
    diagnose_model_used: str
    diagnose_tokens_in: int
    diagnose_tokens_out: int
    candidate_edges: list[dict[str, Any]]
    # diagnosed-but-not-promoted concepts (confidence below MIN_CONFIDENCE),
    # retained as the bounded HITL "add a struggle" picker source (CHO-1973 Q2,
    # ADR-205 D4). Each: {concept_key, concept_label, confidence}.
    below_threshold_concepts: list[dict[str, Any]]
    output_decision: str
    reviewed_edges: list[dict[str, Any]]
    output_selection: dict[str, Any]
    generated_outputs: list[dict[str, Any]]
    analyzed_body: dict[str, Any]
    published_row_id: str
    outputs_published_row_id: str

    # control / bookkeeping
    reiterate_requested: bool
    reiterate_count: int
    diagnose_constraints: dict[str, Any]
    governance_status: str
    error_message: str
    trace: list[dict[str, Any]]


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


def _trace(state: WeaknessAnalyserState, node: str, **fields: Any) -> list[dict[str, Any]]:
    entry = {"node": node, **fields}
    return [*(state.get("trace") or []), entry]


def _edge_to_body_dict(edge: Any) -> dict[str, Any]:
    return {
        "concept_label": edge.concept_label,
        "concept_key": edge.concept_key,
        "category": edge.category,
        "tags": list(edge.tags),
        "confidence": edge.confidence,
        "strength": edge.strength,
        "descriptor_json": edge.descriptor_json,
    }


def _struggle_to_body_dict(struggle: Any) -> dict[str, Any]:
    """Serialise a retained ``CandidateStruggle`` for checkpoint state.
    ``confidence`` rides along so the panel builder can rank + cap the bounded
    picker; it is stripped from the proto wire shape there."""
    return {
        "concept_key": struggle.concept_key,
        "concept_label": struggle.concept_label,
        "confidence": struggle.confidence,
    }


def apply_panel_review(candidates: list[dict[str, Any]], edge_decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the panel-shape per-edge review to the candidate edges (CHO-1973).

    ``edge_decisions`` is the resume body's ``edges`` list: each item addresses a
    candidate by its ``proposed_edge_id`` (minted in the panel; ``pe-{index}``)
    with a bounded ``decision`` (accept / reject / merge), an optional
    ``merge_into_id`` (another proposed_edge_id), and an optional ``difficulty``
    (easier / standard / harder). All controls are STRUCTURED, never free text.

    Policy: an edge the learner did NOT touch is KEPT (the panel proposed it);
    only an explicit ``reject`` or a ``merge`` source is dropped. A ``merge``
    keeps its target. An unresolvable id (out-of-range / garbled) is logged +
    skipped: one bad id never sinks the rest of the review (it is the FE's bug
    to fix, not a learner-data loss). The proposed_edge_id -> candidate mapping
    is the inverse of the id minted in the panel builder; the orchestrator
    alone holds it.
    """
    drop: set[int] = set()
    keep_target: set[int] = set()
    difficulty: dict[int, str] = {}
    for d in edge_decisions:
        if not isinstance(d, dict):
            continue
        idx = index_from_proposed_edge_id(d.get("proposed_edge_id"))
        decision = str(d.get("decision", "")).strip().lower()
        if idx is None or not (0 <= idx < len(candidates)):
            logger.warning(
                "weakness_crew.hitl.unresolved_proposed_edge_id",
                extra={"proposed_edge_id": d.get("proposed_edge_id"), "decision": decision},
            )
            continue
        if decision == "reject":
            drop.add(idx)
        elif decision == "merge":
            drop.add(idx)  # the source folds into the target
            tgt = index_from_proposed_edge_id(d.get("merge_into_id"))
            if tgt is not None and 0 <= tgt < len(candidates):
                keep_target.add(tgt)
            else:
                logger.warning(
                    "weakness_crew.hitl.unresolved_merge_target",
                    extra={"merge_into_id": d.get("merge_into_id")},
                )
        else:  # accept (default): may carry a difficulty override
            bucket = str(d.get("difficulty", "")).strip().lower()
            if bucket:
                difficulty[idx] = bucket

    out: list[dict[str, Any]] = []
    for i, edge in enumerate(candidates):
        if i in drop and i not in keep_target:
            continue
        e2 = dict(edge)
        strength = PANEL_DIFFICULTY_STRENGTH.get(difficulty.get(i, ""))
        if strength is not None:
            e2["strength"] = strength
        out.append(e2)
        if len(out) >= MAX_EDGES:
            break
    return out


def selection_from_kinds(kinds: list[str]) -> dict[str, bool]:
    """Project the resume body's ``selected_outputs`` (a list of output KINDS)
    into the full ``{kind: bool}`` selection the output nodes + the analyzed
    encoder consume. An absent kind is False (not selected)."""
    chosen = {str(k).strip() for k in (kinds or []) if str(k).strip()}
    return {kind: (kind in chosen) for kind in OUTPUT_KINDS}


# --------------------------------------------------------------------------- #
# output parsing (pure): the agent's text -> the FROZEN outputs entry content
# --------------------------------------------------------------------------- #


def _loads_object(text: str) -> dict[str, Any] | None:
    """Parse STRICT JSON; tolerate the ``json`` code fences the model sometimes
    wraps an object in (exactly the tolerant parse the diagnose path keeps)."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_study_aids(text: str) -> str | None:
    """The STRICT study-aids JSON ``{advice, glossary[{term,definition}],
    cheat_sheet[]}``; returns the text for the outputs entry (the wire carries
    it as content_json) or None when the shape is not honoured (dropped loudly,
    never fabricated)."""
    obj = _loads_object(text)
    if obj is None:
        return None
    if not isinstance(obj.get("advice"), str) or not obj.get("advice", "").strip():
        return None
    if not isinstance(obj.get("glossary"), list) or not isinstance(obj.get("cheat_sheet"), list):
        return None
    return text


def parse_practice_test(text: str, *, edges: list[dict[str, Any]]) -> dict[str, Any] | None:
    """``{title, questions[], rejected_reason?}`` from the agent (actor -> critic
    -> on-edge gate, one bounded regenerate inside ONE dispatch). An
    all-rejected set is ``OK`` with ``questions: []`` and a ``rejected_reason``:
    that is NOT an output (logged), never fabricated. The entry content keeps
    the shape consumption already projects: ``{title, questions, edge_keys}``."""
    obj = _loads_object(text)
    if obj is None:
        return None
    questions = [q for q in (obj.get("questions") or []) if isinstance(q, dict)]
    if not questions:
        logger.warning(
            "weakness_crew.practice_test.rejected",
            extra={"rejected_reason": str(obj.get("rejected_reason", "") or "no questions")},
        )
        return None
    title = str(obj.get("title") or "Practice test")
    edge_keys = sorted({str(e.get("concept_key", "")) for e in edges if e.get("concept_key")})
    return {"title": title, "questions": questions, "edge_keys": edge_keys}


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #


async def reserve_mana_node(state: WeaknessAnalyserState, *, mana: ManaPort) -> dict[str, Any]:
    reservation_id = str(state.get("reservation_id", "")).strip()
    if not reservation_id:
        # free tier (e.g. graded-assessment-derived): nothing to reserve.
        return {"trace": _trace(state, "reserve_mana", reserved=False)}
    try:
        await mana.ensure_reserved(
            reservation_id=reservation_id,
            tenant_id=state["tenant_id"],
            gcid=state["learner_gcid"],
        )
    except Exception as exc:  # noqa: BLE001 - fail loud, route to refund
        logger.exception("weakness_crew.reserve_mana.failed")
        return {
            "governance_status": GOV_FAILED,
            "error_message": f"reserve_mana: {exc}",
            "trace": _trace(state, "reserve_mana", error=str(exc)),
        }
    return {"trace": _trace(state, "reserve_mana", reserved=True)}


async def safesearch_node(
    state: WeaknessAnalyserState, *, downloader: BlobDownloader, safesearch: SafeSearchPort
) -> dict[str, Any]:
    """Cloud Vision SafeSearch on the RAW image BEFORE the extract dispatch
    (ADR-254 D12): fail closed. Only an image is downloaded, and only here; the
    bytes are node-local and never enter the checkpoint. A non-image upload
    (PDF / text) needs no bytes in the kennel at all: its extracted text is
    Model-Armor-screened downstream (``screen_input``)."""
    mime = str(state.get("source_mime_type", "") or "")
    if not mime.lower().startswith("image/"):
        return {"trace": _trace(state, "safesearch", skipped="non_image", mime=mime)}
    try:
        blob = await downloader.download(state["source_blob_uri"])
    except Exception as exc:  # noqa: BLE001
        logger.exception("weakness_crew.ingest.failed")
        return {
            "governance_status": GOV_FAILED,
            "error_message": f"ingest: {exc}",
            "trace": _trace(state, "safesearch", error=str(exc)),
        }
    try:
        ok = await safesearch.inspect(blob_bytes=blob, mime_type=mime)
    except Exception as exc:  # noqa: BLE001
        logger.exception("weakness_crew.safesearch.failed")
        return {
            "governance_status": GOV_FAILED,
            "error_message": f"safesearch: {exc}",
            "trace": _trace(state, "safesearch", error=str(exc)),
        }
    if not ok:
        return {
            "governance_status": GOV_BLOCKED,
            "error_message": "safesearch: explicit content",
            "trace": _trace(state, "safesearch", safesearch="blocked"),
        }
    return {"trace": _trace(state, "safesearch", bytes=len(blob), safesearch="ok")}


async def extract_node(state: WeaknessAnalyserState, *, extractor: Extractor) -> dict[str, Any]:
    """Dispatch the extract BY REFERENCE (the run PARKS here on the bus)."""
    try:
        res = await extractor.extract(
            source_blob_uri=state["source_blob_uri"],
            source_mime_type=str(state.get("source_mime_type", "") or ""),
            tenant_id=state["tenant_id"],
            gcid=state["learner_gcid"],
            traceparent=state.get("traceparent", ""),
            tracestate=state.get("tracestate", ""),
        )
    except Exception as exc:  # noqa: BLE001
        # A PARK IS NOT A FAILURE (ADR-253): re-raise it FIRST; only a genuine
        # dispatch failure (a FAILED completion, a refused uri) gets here.
        reraise_if_dispatch_park(exc)
        logger.exception("weakness_crew.extract.failed")
        return {
            "governance_status": GOV_FAILED,
            "error_message": f"extract: {exc}",
            "trace": _trace(state, "extract", error=str(exc)),
        }
    return {
        "extracted_text": res.text,
        "extract_tokens_in": res.input_tokens,
        "extract_tokens_out": res.output_tokens,
        "trace": _trace(state, "extract", chars=len(res.text)),
    }


async def screen_input_node(state: WeaknessAnalyserState, *, screener: Screener) -> dict[str, Any]:
    clues_block = render_structured_clues_block(state.get("structured_clues"))
    text = state.get("extracted_text", "")
    if clues_block:
        text = f"{text}\n\n{clues_block}"
    try:
        verdict = await screener.screen(
            text=text, tenant_id=state["tenant_id"], gcid=state["learner_gcid"], direction="input"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("weakness_crew.screen_input.failed")
        return {
            "governance_status": GOV_FAILED,
            "error_message": f"screen_input: {exc}",
            "trace": _trace(state, "screen_input", error=str(exc)),
        }
    if verdict.decision == "BLOCK":
        return {
            "input_decision": verdict.decision,
            "governance_status": GOV_BLOCKED,
            "error_message": f"screen_input blocked: {verdict.explanation}",
            "trace": _trace(state, "screen_input", decision=verdict.decision),
        }
    return {"input_decision": verdict.decision, "trace": _trace(state, "screen_input", decision=verdict.decision)}


async def diagnose_node(state: WeaknessAnalyserState, *, diagnoser: Diagnoser) -> dict[str, Any]:
    clues = dict(state.get("structured_clues") or {})
    # reiterate constraints (from a prior HITL review) ride as ADDITIONAL clue
    # data: bounded, structured, never free-form reprompt.
    constraints = state.get("diagnose_constraints") or {}
    if constraints.get("focus_topic_keys"):
        existing = list(clues.get("weak_topic_keys") or [])
        clues["weak_topic_keys"] = existing + [k for k in constraints["focus_topic_keys"] if k not in existing]
    clues_block = render_structured_clues_block(clues)
    try:
        res = await diagnoser.diagnose(
            extracted_text=state.get("extracted_text", ""),
            structured_clues_block=clues_block,
            upload_kind=state.get("upload_kind", UPLOAD_KIND_MARKED_TEST),
            tenant_id=state["tenant_id"],
            gcid=state["learner_gcid"],
            traceparent=state.get("traceparent", ""),
            tracestate=state.get("tracestate", ""),
        )
    except Exception as exc:  # noqa: BLE001
        # A PARK IS NOT A FAILURE (ADR-253). Re-raise the park FIRST; only a
        # genuine dispatch failure gets here.
        reraise_if_dispatch_park(exc)
        logger.exception("weakness_crew.diagnose.failed")
        return {
            "governance_status": GOV_FAILED,
            "error_message": f"diagnose: {exc}",
            "trace": _trace(state, "diagnose", error=str(exc)),
        }
    return {
        "diagnosis_json": res.diagnosis_json,
        "diagnose_model_used": res.model_used,
        "diagnose_tokens_in": res.input_tokens,
        "diagnose_tokens_out": res.output_tokens,
        "trace": _trace(state, "diagnose", model=res.model_used),
    }


async def critic_verify_node(state: WeaknessAnalyserState) -> dict[str, Any]:
    """Parse the diagnosis (confidence-floored, fail-soft) into candidate edges.
    Unreadable output -> zero edges (never fabricate), ADR-205 D3.

    Additively RETAINS the sub-threshold diagnosed concepts
    (``below_threshold_concepts``) so the HITL panel can offer them as bounded
    candidate struggles (CHO-1973 Q2, ADR-205 D4)."""
    result = parse_analysis(
        state.get("diagnosis_json", ""),
        model_used=state.get("diagnose_model_used", ""),
        input_tokens=state.get("diagnose_tokens_in", 0),
        output_tokens=state.get("diagnose_tokens_out", 0),
    )
    edges = [_edge_to_body_dict(e) for e in result.edges]
    struggles = [_struggle_to_body_dict(s) for s in result.below_threshold]
    return {
        "candidate_edges": edges,
        "below_threshold_concepts": struggles,
        "trace": _trace(state, "critic_verify", edge_count=len(edges), struggle_count=len(struggles)),
    }


async def screen_output_node(state: WeaknessAnalyserState, *, screener: Screener) -> dict[str, Any]:
    try:
        verdict = await screener.screen(
            text=state.get("diagnosis_json", ""),
            tenant_id=state["tenant_id"],
            gcid=state["learner_gcid"],
            direction="output",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("weakness_crew.screen_output.failed")
        return {
            "governance_status": GOV_FAILED,
            "error_message": f"screen_output: {exc}",
            "trace": _trace(state, "screen_output", error=str(exc)),
        }
    if verdict.decision == "BLOCK":
        return {
            "output_decision": verdict.decision,
            "governance_status": GOV_BLOCKED,
            "error_message": f"screen_output blocked: {verdict.explanation}",
            "trace": _trace(state, "screen_output", decision=verdict.decision),
        }
    return {"output_decision": verdict.decision, "trace": _trace(state, "screen_output", decision=verdict.decision)}


def hitl_review_node(state: WeaknessAnalyserState) -> dict[str, Any]:
    """Pause for the learner to review their own diagnosis via BOUNDED controls.

    The interrupt payload carries the diagnosed edges (reframed positively in
    the FE) + identity (incl. ``learner_gcid``, CHO-1973 Wave A) so the runner
    can build the FE panel without re-reading state. The resume value is the
    PANEL-SHAPE decision (CHO-1973 Wave C):

        {action: "confirm"|"reiterate",
         edges: [{proposed_edge_id, decision, merge_into_id?, difficulty?}],
         added_struggles: [concept_key], selected_outputs: [output_kind]}

    Zero free-form reprompt (ADR-205 D4): the panel state IS the resume payload.
    """
    resume = (
        interrupt(
            {
                "node": "hitl_review",
                "run_id": state.get("run_id"),
                "tenant_id": state.get("tenant_id"),
                "upload_id": state.get("upload_id"),
                "learner_gcid": state.get("learner_gcid"),
                "edges": state.get("candidate_edges", []),
                # CHO-1973 Q2: the sub-threshold diagnosed concepts ride the interrupt
                # so the runner can offer them in the bounded "add a struggle" picker
                # (ADR-205 D4: analyser-suggested, never free-form).
                "candidate_struggles": state.get("below_threshold_concepts", []),
                "requested_outputs": state.get("requested_outputs", {}),
            }
        )
        or {}
    )

    action = str(resume.get("action", "")).strip().lower()
    if action == "reiterate" and state.get("reiterate_count", 0) < MAX_REITERATIONS:
        # added_struggles (concept_keys) ride into the re-diagnose as the bounded
        # focus constraint (clue DATA, never free-text reprompt; ADR-205 D2).
        added = [str(s).strip() for s in (resume.get("added_struggles") or []) if str(s).strip()]
        return {
            "reiterate_requested": True,
            "reiterate_count": state.get("reiterate_count", 0) + 1,
            "diagnose_constraints": {"focus_topic_keys": added} if added else {},
            "trace": _trace(state, "hitl_review", decision="reiterate"),
        }
    reviewed = apply_panel_review(state.get("candidate_edges", []), resume.get("edges") or [])
    selection = selection_from_kinds(resume.get("selected_outputs") or [])
    return {
        "reiterate_requested": False,
        "reviewed_edges": reviewed,
        "output_selection": selection,
        "governance_status": GOV_APPROVED,
        "trace": _trace(state, "hitl_review", decision=action or "confirm", kept=len(reviewed)),
    }


async def synthesize_edges_node(state: WeaknessAnalyserState, *, clock: Callable[[], _dt.datetime]) -> dict[str, Any]:
    """Build the FROZEN weakness.analyzed.v1 body from the reviewed edges."""
    edges = state.get("reviewed_edges", [])
    body = {
        "upload_id": state["upload_id"],
        "tenant_id": state["tenant_id"],
        "learner_gcid": state["learner_gcid"],
        "model_used": state.get("diagnose_model_used", ""),
        "input_token_count": state.get("extract_tokens_in", 0) + state.get("diagnose_tokens_in", 0),
        "output_token_count": state.get("extract_tokens_out", 0) + state.get("diagnose_tokens_out", 0),
        "analyzed_at": clock().isoformat(),
        "edges": edges,
        # Echo the post-HITL output selection onto the analyzed event (ADR-205 D5
        # / CHO-1966) so chora-consumption can gate the Familiar-RAG write on
        # familiar_coaching without a cross-DB read. Empty/all-false -> the
        # encoder drops field 10 (default-off, the safe single-shot-equivalent
        # default).
        "output_selection": state.get("output_selection") or {},
    }
    return {"analyzed_body": body, "trace": _trace(state, "synthesize_edges", edge_count=len(edges))}


async def publish_analyzed_node(state: WeaknessAnalyserState, *, publisher: AnalyzedPublisher) -> dict[str, Any]:
    if state.get("published_row_id"):  # idempotent: a resumed/retried run never double-publishes
        return {"trace": _trace(state, "publish_analyzed", skipped="already_published")}
    body = state.get("analyzed_body") or {}
    row_id = await publisher.publish_weakness_analyzed(
        body=body,
        traceparent=state.get("traceparent", ""),
        tracestate=state.get("tracestate", ""),
    )
    return {"published_row_id": row_id, "trace": _trace(state, "publish_analyzed", row_id=row_id)}


async def emit_evidence_node(
    state: WeaknessAnalyserState, *, evidence_emitter: EvidenceEmitter | None
) -> dict[str, Any]:
    if evidence_emitter is not None:
        try:
            await evidence_emitter.emit(
                body=state.get("analyzed_body") or {},
                input_decision=state.get("input_decision", "ALLOW"),
                output_decision=state.get("output_decision", "ALLOW"),
                model_used=state.get("diagnose_model_used", ""),
                traceparent=state.get("traceparent", ""),
                tracestate=state.get("tracestate", ""),
            )
        except Exception:  # noqa: BLE001 - evidence is non-blocking; never fail the publish
            logger.exception("weakness_crew.emit_evidence.failed")
    return {"trace": _trace(state, "emit_evidence")}


async def _run_output_task(
    state: WeaknessAnalyserState,
    *,
    kind: str,
    task_runner: OutputTaskRunner,
    screener: Screener,
    parse: Callable[[str], Any],
    max_questions: int,
) -> dict[str, Any]:
    """ONE learner output = ONE parked dispatch + screen + parse.

    Skips (no dispatch) when the kind was not selected, when there are no edges
    to work from, or when the outputs event already published (a duplicate
    resume of a finished thread). A FAILED completion, a malformed answer or a
    Model Armor BLOCK drops THIS output loudly; the analysis is already durable
    and the run goes on to the next kind / the outputs event.
    """
    selection = state.get("output_selection") or {}
    edges = list(state.get("reviewed_edges") or [])
    if not selection.get(kind) or not edges or state.get("outputs_published_row_id"):
        return {"trace": _trace(state, kind, skipped=True)}
    try:
        text = await task_runner.run_task(
            task_kind=kind,
            edges=edges,
            tenant_id=state["tenant_id"],
            gcid=state["learner_gcid"],
            traceparent=state.get("traceparent", ""),
            tracestate=state.get("tracestate", ""),
            max_questions=max_questions,
        )
    except Exception as exc:  # noqa: BLE001
        # A PARK IS NOT A FAILURE (ADR-253): re-raise it FIRST.
        reraise_if_dispatch_park(exc)
        logger.error(
            "weakness_crew.output_task.failed",
            extra={"kind": kind, "upload_id": state.get("upload_id", ""), "err": f"{exc.__class__.__name__}: {exc}"},
        )
        return {"trace": _trace(state, kind, error=str(exc))}
    content = parse(text)
    if content is None:
        logger.error(
            "weakness_crew.output_task.malformed",
            extra={"kind": kind, "upload_id": state.get("upload_id", ""), "chars": len(text or "")},
        )
        return {"trace": _trace(state, kind, malformed=True)}
    # Output-screen the learner-facing generated content (ADR-205 D5 / ADR-152).
    # A BLOCK drops it; a screener failure drops it too (never surface unscreened).
    screen_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    try:
        verdict = await screener.screen(
            text=screen_text,
            tenant_id=state["tenant_id"],
            gcid=state["learner_gcid"],
            direction="output",
        )
    except Exception as exc:  # noqa: BLE001 - fail safe: drop rather than surface unscreened
        logger.exception("weakness_crew.output_task.screen_failed", extra={"kind": kind})
        return {"trace": _trace(state, kind, screen_error=str(exc))}
    if verdict.decision == "BLOCK":
        logger.warning(
            "weakness_crew.output_task.blocked", extra={"kind": kind, "upload_id": state.get("upload_id", "")}
        )
        return {"trace": _trace(state, kind, blocked=True)}
    outputs = [*(state.get("generated_outputs") or []), {"type": kind, "content": content, "metered": True}]
    return {"generated_outputs": outputs, "trace": _trace(state, kind, generated=True)}


async def study_aids_node(
    state: WeaknessAnalyserState, *, task_runner: OutputTaskRunner, screener: Screener
) -> dict[str, Any]:
    return await _run_output_task(
        state,
        kind=OUTPUT_STUDY_AIDS,
        task_runner=task_runner,
        screener=screener,
        parse=parse_study_aids,
        max_questions=DEFAULT_PRACTICE_TEST_MAX_QUESTIONS,
    )


async def practice_test_node(
    state: WeaknessAnalyserState, *, task_runner: OutputTaskRunner, screener: Screener, max_questions: int
) -> dict[str, Any]:
    edges = list(state.get("reviewed_edges") or [])
    return await _run_output_task(
        state,
        kind=OUTPUT_PRACTICE_TEST,
        task_runner=task_runner,
        screener=screener,
        parse=lambda text: parse_practice_test(text, edges=edges),
        max_questions=max_questions,
    )


async def publish_outputs_node(
    state: WeaknessAnalyserState, *, outputs_publisher: OutputsPublisher, clock: Callable[[], _dt.datetime]
) -> dict[str, Any]:
    """Emit ``weakness.outputs_generated.v1`` for the learner's selection.

    Fires whenever ANY kind was selected (an empty list is a real outcome:
    every kind dropped, or only consumption-side seams selected); never fires
    when nothing was selected; idempotent on a duplicate resume. A publish
    failure RAISES: the outbox INSERT is ``ON CONFLICT DO NOTHING`` so the
    redelivery that follows is safe, and a swallowed drop is exactly how the
    WS-7 gap stayed invisible.
    """
    selection = state.get("output_selection") or {}
    outputs = list(state.get("generated_outputs") or [])
    if not any(selection.values()):
        return {"generated_outputs": outputs, "trace": _trace(state, "publish_outputs", skipped="no_selection")}
    if state.get("outputs_published_row_id"):
        return {"generated_outputs": outputs, "trace": _trace(state, "publish_outputs", skipped="already_published")}
    deferred = [s for s in _SEAM_SELECTIONS if selection.get(s)]
    if deferred:
        logger.info(
            "weakness_outputs.deferred_to_consumption",
            extra={"upload_id": state.get("upload_id", ""), "deferred": deferred},
        )
    row_id = await outputs_publisher.publish_outputs_generated(
        body={
            "upload_id": state["upload_id"],
            "tenant_id": state["tenant_id"],
            "learner_gcid": state["learner_gcid"],
            "generated_at": clock().isoformat(),
            "outputs": outputs,
        },
        # traceparent is a MANDATORY envelope field; chora-consumption's inbound
        # validator REJECTS an envelope without it.
        traceparent=state.get("traceparent", ""),
        tracestate=state.get("tracestate", ""),
    )
    return {
        "outputs_published_row_id": row_id,
        "generated_outputs": outputs,
        "trace": _trace(state, "publish_outputs", row_id=row_id, generated=len(outputs)),
    }


async def refund_node(state: WeaknessAnalyserState, *, mana: ManaPort) -> dict[str, Any]:
    """Terminal fail-loud / BLOCK handler: return the reservation (ADR-205 D6)."""
    reservation_id = str(state.get("reservation_id", "")).strip()
    if reservation_id:
        try:
            await mana.refund(
                reservation_id=reservation_id,
                tenant_id=state["tenant_id"],
                gcid=state["learner_gcid"],
                reason=state.get("error_message", "weakness_crew_failed"),
            )
        except Exception:  # noqa: BLE001 - operator reconcile; never mask the original failure
            logger.exception("weakness_crew.refund.failed")
    status = state.get("governance_status") or GOV_FAILED
    return {"governance_status": status, "trace": _trace(state, "refund", status=status)}


# --------------------------------------------------------------------------- #
# routers
# --------------------------------------------------------------------------- #


def _is_terminal(state: WeaknessAnalyserState) -> bool:
    return state.get("governance_status") in (GOV_FAILED, GOV_BLOCKED)


def _route_or_refund(next_node: str) -> Callable[[WeaknessAnalyserState], str]:
    def router(state: WeaknessAnalyserState) -> str:
        return "refund" if _is_terminal(state) else next_node

    return router


def route_after_hitl(state: WeaknessAnalyserState) -> str:
    return "diagnose" if state.get("reiterate_requested") else "synthesize_edges"


# --------------------------------------------------------------------------- #
# graph builder
# --------------------------------------------------------------------------- #


def build_weakness_analyser_graph(
    *,
    mana: ManaPort,
    downloader: BlobDownloader,
    extractor: Extractor,
    safesearch: SafeSearchPort,
    screener: Screener,
    diagnoser: Diagnoser,
    publisher: AnalyzedPublisher,
    task_runner: OutputTaskRunner,
    outputs_publisher: OutputsPublisher,
    evidence_emitter: EvidenceEmitter | None = None,
    clock: Callable[[], _dt.datetime] | None = None,
    checkpointer: Any | None = None,
    practice_test_max_questions: int = DEFAULT_PRACTICE_TEST_MAX_QUESTIONS,
) -> Any:
    """Compile the companion_diagnosis StateGraph (ADR-205 WS-2 on the bus)."""
    clk = clock or _utc_now
    sg: StateGraph = StateGraph(WeaknessAnalyserState)

    sg.add_node("reserve_mana", partial(reserve_mana_node, mana=mana))
    sg.add_node("safesearch", partial(safesearch_node, downloader=downloader, safesearch=safesearch))
    sg.add_node("extract", partial(extract_node, extractor=extractor))
    sg.add_node("screen_input", partial(screen_input_node, screener=screener))
    sg.add_node("diagnose", partial(diagnose_node, diagnoser=diagnoser))
    sg.add_node("critic_verify", critic_verify_node)
    sg.add_node("screen_output", partial(screen_output_node, screener=screener))
    sg.add_node("hitl_review", hitl_review_node)
    sg.add_node("synthesize_edges", partial(synthesize_edges_node, clock=clk))
    sg.add_node("publish_analyzed", partial(publish_analyzed_node, publisher=publisher))
    sg.add_node("emit_evidence", partial(emit_evidence_node, evidence_emitter=evidence_emitter))
    sg.add_node("study_aids", partial(study_aids_node, task_runner=task_runner, screener=screener))
    sg.add_node(
        "practice_test",
        partial(
            practice_test_node, task_runner=task_runner, screener=screener, max_questions=practice_test_max_questions
        ),
    )
    sg.add_node("publish_outputs", partial(publish_outputs_node, outputs_publisher=outputs_publisher, clock=clk))
    sg.add_node("refund", partial(refund_node, mana=mana))

    # Each fallible/gating step routes to "refund" on failure or BLOCK; the
    # happy successor otherwise. "refund" is the single terminal fail-loud sink.
    def _gate(node: str, nxt: str) -> None:
        sg.add_conditional_edges(node, _route_or_refund(nxt), {nxt: nxt, "refund": "refund"})

    sg.add_edge(START, "reserve_mana")
    _gate("reserve_mana", "safesearch")
    _gate("safesearch", "extract")
    _gate("extract", "screen_input")
    _gate("screen_input", "diagnose")
    _gate("diagnose", "critic_verify")
    sg.add_edge("critic_verify", "screen_output")
    _gate("screen_output", "hitl_review")
    sg.add_conditional_edges(
        "hitl_review",
        route_after_hitl,
        {"diagnose": "diagnose", "synthesize_edges": "synthesize_edges"},
    )
    # Publish + emit the edges FIRST (durable), THEN the selected learner
    # outputs one park per node, THEN the outputs event. The HITL resume
    # returns at the first output park (the FE already treats confirm as done;
    # the artifacts ride the outputs event whenever they land).
    sg.add_edge("synthesize_edges", "publish_analyzed")
    sg.add_edge("publish_analyzed", "emit_evidence")
    sg.add_edge("emit_evidence", "study_aids")
    sg.add_edge("study_aids", "practice_test")
    sg.add_edge("practice_test", "publish_outputs")
    sg.add_edge("publish_outputs", END)
    sg.add_edge("refund", END)

    compiled = sg.compile(checkpointer=checkpointer) if checkpointer is not None else sg.compile()
    return compiled.with_config({"recursion_limit": 50})


__all__ = [
    "AGENT_ID",
    "DEFAULT_PRACTICE_TEST_MAX_QUESTIONS",
    "OUTPUT_PRACTICE_TEST",
    "OUTPUT_STUDY_AIDS",
    "AnalyzedPublisher",
    "BlobDownloader",
    "DiagnoseResult",
    "Diagnoser",
    "EvidenceEmitter",
    "ExtractResult",
    "Extractor",
    "ManaPort",
    "OutputTaskRunner",
    "OutputsPublisher",
    "SafeSearchPort",
    "ScreenVerdict",
    "Screener",
    "WeaknessAnalyserState",
    "apply_panel_review",
    "build_weakness_analyser_graph",
    "parse_practice_test",
    "parse_study_aids",
    "selection_from_kinds",
]
