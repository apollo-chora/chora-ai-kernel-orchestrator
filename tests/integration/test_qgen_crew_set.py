"""Single-pass SET-native qgen crew — graph-level proof (CHO-1819 P1b).

Compiles the REAL ``build_qgen_crew_graph`` with FAKE clients (executor /
guardrail / kroki / gateway / gcs) and ainvokes it in SET MODE. This is the
load-bearing proof the handoff demands before the orchestrator rolls: the
single generate-set call yields a diverse mixed-type set, strict-shortfall is
honestly summarised, a non-strict short set FAILS LOUD, the per-type image cap
backstops an over-emitting agent, the bounded regenerate loop re-generates only
the rejected, and N=1 collapses onto the same path.

No gRPC, no Vertex AI, no Pub/Sub — the fakes return the confirmed P1c agent
wrapper (``{"candidates": [...], "generation_summary": {...}}``) so the test
exercises the orchestrator's OWN set-native logic, not the agent's.
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
    ROLE_RENDER,
    build_qgen_crew_graph,
)

# Marker substring: a candidate whose stem contains it is REJECTED by the fake
# critic (content-aware → robust to asyncio.gather scheduling order).
_REJECT_MARKER = "[REJECT]"


# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


def _mcq(stem: str, *, image: bool = False, placement: str = "stem") -> dict[str, Any]:
    cand: dict[str, Any] = {
        "stem": stem,
        "question_type": "mcq",
        "intent": "new_question",
        "options": [
            {"option_id": "a", "label": "A", "is_correct": True, "explainer": "x"},
            {"option_id": "b", "label": "B", "is_correct": False, "explainer": "y"},
        ],
    }
    if image:
        cand["image_specs"] = [{"mode": "scene", "source": f"render {stem}", "placement": placement}]
    return cand


def _oe(stem: str, *, image: bool = False, placement: str = "answer") -> dict[str, Any]:
    cand: dict[str, Any] = {
        "stem": stem,
        "question_type": "oe",
        "intent": "new_question",
        "oe_payload": {
            "model_answer": "A sufficiently long model answer " * 4,
            "rubric": [{"criterion_id": "a", "title": "t", "description": "d", "weight": 100}],
            "grader_tier": "T2",
        },
    }
    if image:
        cand["image_specs"] = [{"mode": "mermaid", "source": "graph TD; A-->B", "placement": placement}]
    return cand


def _wrapper(candidates: list[dict[str, Any]], *, shortfall_reason: str = "") -> str:
    summary: dict[str, Any] = {
        "requested_total": 0,
        "generated_total": len(candidates),
        "generated_per_type": {},
        "shortfall_reason": shortfall_reason,
    }
    return json.dumps({"candidates": candidates, "generation_summary": summary})


@dataclass
class _FakeSetExecutor:
    """Set-native fake AgentExecutor.

    ROLE_GENERATE: pops the next wrapper from ``generate_queue`` (one per
    generate round). ROLE_CRITIQUE: content-aware — REJECTS a candidate whose
    stem contains ``_REJECT_MARKER`` (robust to gather ordering), else accepts.
    """

    generate_queue: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)
    # ADR-254 D12: a scene image is ONE qgen_render dispatch; the fake answers
    # with the gs:// object the renderer would have written (the kennel signs).
    renders: int = 0
    # ADR-254 D2: mode=compose on the generate lane. None => a default answer
    # (order reversed, points 7) built from the candidates the kennel sent; a
    # dict => returned as the output payload; an Exception => raised (the
    # executor surfaces a FAILED completion that way).
    compose_answer: Any = None
    compose_calls: list[dict[str, Any]] = field(default_factory=list)

    def _critique_set(self, decoded: dict[str, Any]) -> str:
        """Batch critique (CHO-2397): echo one verdict per candidate_id,
        rejecting stems that carry the marker."""
        verdicts = []
        for cand in decoded.get("candidates") or []:
            rejected = _REJECT_MARKER in str(cand.get("stem") or "")
            verdicts.append(
                {
                    "candidate_id": str(cand.get("candidate_id") or ""),
                    "accepted": not rejected,
                    "critique_notes": "near-duplicate" if rejected else "",
                    "suggested_revisions": ["differentiate"] if rejected else [],
                }
            )
        return json.dumps({"verdicts": verdicts})

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
        self.calls.append(
            {
                "execution_id": execution_id,
                "agent_role": agent_role,
                "input_payload": input_payload,
            }
        )
        decoded_in = json.loads(input_payload) if input_payload else {}
        if agent_role == ROLE_GENERATE and decoded_in.get("mode") == "compose":
            self.compose_calls.append(decoded_in)
            if isinstance(self.compose_answer, Exception):
                raise self.compose_answer
            if isinstance(self.compose_answer, dict):
                out = json.dumps(self.compose_answer)
            else:
                ids = [str(c.get("draft_id") or "") for c in decoded_in.get("candidates") or []]
                out = json.dumps(
                    {
                        "proposed_test_set": {
                            "title": "Composed set",
                            "description": "By the agent.",
                            "order": list(reversed(ids)),
                            "points": {d: 7 for d in ids},
                        }
                    }
                )
        elif agent_role == ROLE_GENERATE:
            if not self.generate_queue:
                raise RuntimeError("_FakeSetExecutor: generate_queue exhausted")
            out = self.generate_queue.pop(0)
        elif agent_role == ROLE_CRITIQUE:
            out = self._critique_set(json.loads(input_payload))
        elif agent_role == ROLE_RENDER:
            self.renders += 1
            out = json.dumps(
                {
                    "image_uri": (
                        f"gs://chora-ai-assist-images-dev/tenants/{tenant_id}/jobs/"
                        f"{workflow_id or 'job'}/scene-{self.renders}.png"
                    ),
                    "mime_type": "image/png",
                }
            )
        else:  # pragma: no cover (defensive)
            raise RuntimeError(f"unexpected role {agent_role}")
        return AgentExecutorResponse(
            execution_id=execution_id,
            output_payload=out,
            tokens_consumed_total=11,
            cost_micros_total=5,
            final_state="EXECUTION_FINAL_STATE_SUCCESS",
            input_tokens=7,
            output_tokens=4,
        )


@dataclass
class _FakeGuardrail:
    block_substrings: list[str] = field(default_factory=list)
    calls: list[GuardrailScreenInput] = field(default_factory=list)

    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
        self.calls.append(payload)
        for sub in self.block_substrings:
            if sub in payload.content:
                return ScreenResult(verdict=Verdict.BLOCK, reason="pii_high_risk_block")
        return ScreenResult(verdict=Verdict.ALLOW, reason="clean")


@dataclass
class _FakeKroki:
    async def render(self, *, source: str, output_format: str = "png") -> bytes:
        return b"PNGBYTES"


@dataclass
class _FakeGcs:
    uploads: int = 0
    # scene renders come back as a gs:// from the qgen_render agent; the kennel
    # only SIGNS them (no upload), so the image count for scenes is ``signed``.
    signed: int = 0

    async def sign_read_url(self, gs_uri: str) -> str:
        self.signed += 1
        return f"https://signed.example/{gs_uri.rsplit('/', 1)[-1]}"

    async def upload_and_sign(self, *, tenant_id: str, job_id: str, data: bytes, content_type: str) -> tuple[str, str]:
        self.uploads += 1
        return (
            f"gs://chora-ai-assist-images-dev/tenants/{tenant_id}/jobs/{job_id}/{self.uploads}.png",
            f"https://signed.example/{job_id}/{self.uploads}.png",
        )


def _set_state(type_plan: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": "job-set-001",
        "tenant_id": "tenant-set",
        "gcid": "gcid-author",
        "prompt": "Generate a mixed question set from the source.",
        "question_type": "mixed",
        "metadata": {"subject": "Biology"},
        "set_mode": True,
        "type_plan": type_plan,
        "grounding_mode": "",
        "pipeline_trace": [],
        "errors": [],
    }
    base.update(overrides)
    return base


def _build(
    executor: Any,
    guardrail: Any,
    *,
    kroki: Any = None,
    gcs: Any = None,
) -> Any:
    return build_qgen_crew_graph(
        executor=executor,
        guardrail=guardrail,
        checkpointer=MemorySaver(),
        kroki=kroki,
        gcs=gcs,
    )


def _cfg(job_id: str = "job-set-001") -> dict[str, Any]:
    return {"configurable": {"thread_id": job_id}}


_MIXED_PLAN = [
    {"question_type": "mcq", "count": 8, "max_images": 0},
    {"question_type": "oe", "count": 2, "max_images": 0},
]


# -----------------------------------------------------------------------------
# 1. Mixed 8 MCQ + 2 OE — single generate-set, all accepted.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_set_8mcq_2oe_all_accepted() -> None:
    cands = [_mcq(f"MCQ {i}") for i in range(8)] + [_oe(f"OE {i}") for i in range(2)]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    graph = _build(executor, _FakeGuardrail())

    out = await graph.ainvoke(_set_state(_MIXED_PLAN), config=_cfg())

    assert out.get("refusal_reason", "") == ""
    accepted = out.get("accepted_set") or []
    assert len(accepted) == 10
    assert sum(1 for c in accepted if c["question_type"] == "mcq") == 8
    assert sum(1 for c in accepted if c["question_type"] == "oe") == 2

    summary = out.get("generation_summary") or {}
    assert summary["requested_total"] == 10
    assert summary["generated_total"] == 10
    assert summary["generated_per_type"] == {"mcq": 8, "oe": 2}
    assert summary["shortfall_reason"] == ""
    assert out.get("quality_warning") is False

    # ONE generate call (single-pass), ONE batched critique call for the
    # whole chunk (CHO-2397, ADR-251 D2).
    gen_calls = [c for c in executor.calls if c["agent_role"] == ROLE_GENERATE]
    crit_calls = [c for c in executor.calls if c["agent_role"] == ROLE_CRITIQUE]
    assert len(gen_calls) == 1
    assert len(crit_calls) == 1

    # The generate payload carried set_mode + the type_plan (executor stamps it).
    gen_payload = json.loads(gen_calls[0]["input_payload"])
    assert gen_payload["set_mode"] is True
    assert gen_payload["type_plan"] == _MIXED_PLAN

    # The critique payload carried set_mode + all 10 candidates with ids.
    crit_payload = json.loads(crit_calls[0]["input_payload"])
    assert crit_payload["set_mode"] is True
    assert [c["candidate_id"] for c in crit_payload["candidates"]] == [f"c{i}" for i in range(10)]

    # Trace ABI: generate + ONE tokened critique row + a verdict row per
    # candidate + quality_gate rows present.
    names = [r["name"] for r in out["pipeline_trace"]]
    assert names.count("generate") == 1
    assert names.count("critique") == 1
    assert names.count("critique_verdict") == 10
    assert "quality_gate" in names


# -----------------------------------------------------------------------------
# 2. Strict shortfall — agent honestly returns 7 of 10 with a reason.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strict_shortfall_carries_reason_and_count() -> None:
    cands = [_mcq(f"MCQ {i}") for i in range(6)] + [_oe("OE 0")]
    reason = "The source supported 6 distinct MCQ concepts and 1 OE prompt."
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands, shortfall_reason=reason)])
    graph = _build(executor, _FakeGuardrail())

    out = await graph.ainvoke(_set_state(_MIXED_PLAN, grounding_mode="strict"), config=_cfg())

    assert out.get("refusal_reason", "") == ""
    summary = out["generation_summary"]
    assert summary["requested_total"] == 10
    assert summary["generated_total"] == 7
    assert summary["generated_per_type"] == {"mcq": 6, "oe": 1}
    assert summary["shortfall_reason"] == reason
    assert len(out["accepted_set"]) == 7


# -----------------------------------------------------------------------------
# 3. Non-strict short set — FAIL LOUD (VALIDATION refusal, never padded).
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_strict_short_set_fails_loud() -> None:
    cands = [_mcq(f"MCQ {i}") for i in range(7)]
    # No shortfall_reason + grounding_mode != strict ⇒ illegal short set.
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    graph = _build(executor, _FakeGuardrail())

    out = await graph.ainvoke(_set_state(_MIXED_PLAN, grounding_mode="starting_point"), config=_cfg())

    assert out.get("refusal_reason") == "VALIDATION"
    assert out.get("accepted_set") in (None, [])
    # No candidate was published — fail-loud, no padding.
    names = [r["name"] for r in out["pipeline_trace"]]
    assert "publish_refused" in names
    assert "critique" not in names  # never reached the critic


# -----------------------------------------------------------------------------
# 4. Image cap backstop — agent over-emits; deterministic drop + LOUD warning.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_cap_backstop_drops_over_budget_with_warning() -> None:
    # Plan: 4 MCQ, cap 3 images. Agent emits an image on ALL 4 → 1 dropped.
    plan = [{"question_type": "mcq", "count": 4, "max_images": 3}]
    cands = [_mcq(f"MCQ {i}", image=True) for i in range(4)]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    kroki, gcs = _FakeKroki(), _FakeGcs()
    graph = _build(executor, _FakeGuardrail(), kroki=kroki, gcs=gcs)

    out = await graph.ainvoke(_set_state(plan, question_type="mcq"), config=_cfg())

    assert out.get("refusal_reason", "") == ""
    accepted = out["accepted_set"]
    assert len(accepted) == 4
    # Exactly 3 images rendered (cap), 1 candidate ships imageless.
    rendered = sum(1 for c in accepted if c.get("image_url") or c.get("answer_image_url"))
    assert rendered == 3
    assert gcs.signed == 3 and executor.renders == 3
    # No raw image_specs leak to the published candidate.
    assert all("image_specs" not in c for c in accepted)
    # LOUD over-cap warning trace row.
    render_rows = [r for r in out["pipeline_trace"] if r["name"] == "render_image"]
    assert any(r["status"] == "WARNING" for r in render_rows)


@pytest.mark.asyncio
async def test_forced_stem_image_renders_on_every_question_zero_budget() -> None:
    # CHO-1825 deterministic toggle: image_for_stem forces a stem image on EVERY
    # MCQ even though the discretionary budget (max_images) is 0. The render
    # backstop must admit the forced images (effective_image_caps) — NOT drop
    # them as over-budget. Both questions render; no over-cap warning.
    plan = [{"question_type": "mcq", "count": 2, "max_images": 0, "image_for_stem": True}]
    cands = [_mcq(f"MCQ {i}", image=True, placement="stem") for i in range(2)]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    kroki, gcs = _FakeKroki(), _FakeGcs()
    graph = _build(executor, _FakeGuardrail(), kroki=kroki, gcs=gcs)

    out = await graph.ainvoke(_set_state(plan, question_type="mcq"), config=_cfg())

    assert out.get("refusal_reason", "") == ""
    accepted = out["accepted_set"]
    assert len(accepted) == 2
    # BOTH forced stem images survive the backstop and render.
    rendered = sum(1 for c in accepted if c.get("image_url"))
    assert rendered == 2
    assert gcs.signed == 2 and executor.renders == 2
    # No raw image_specs leak; no over-cap warning (nothing was dropped).
    assert all("image_specs" not in c for c in accepted)
    render_rows = [r for r in out["pipeline_trace"] if r["name"] == "render_image"]
    assert not any(r["status"] == "WARNING" for r in render_rows)


@pytest.mark.asyncio
async def test_regenerate_preserves_forced_image_flag_on_round_plan() -> None:
    # The bounded regenerate round must carry the per-type image_for_stem flag
    # into the REDUCED round plan, or regenerated questions ship imageless.
    plan = [{"question_type": "mcq", "count": 2, "max_images": 0, "image_for_stem": True}]
    round0 = [
        _mcq(f"{_REJECT_MARKER} dup A", image=True, placement="stem"),
        _mcq(f"{_REJECT_MARKER} dup B", image=True, placement="stem"),
    ]
    round1 = [_mcq("fresh A", image=True, placement="stem"), _mcq("fresh B", image=True, placement="stem")]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(round0), _wrapper(round1)])
    kroki, gcs = _FakeKroki(), _FakeGcs()
    graph = _build(executor, _FakeGuardrail(), kroki=kroki, gcs=gcs)

    out = await graph.ainvoke(_set_state(plan, question_type="mcq", max_regen_rounds=1), config=_cfg())

    assert out.get("refusal_reason", "") == ""
    gen_calls = [c for c in executor.calls if c["agent_role"] == ROLE_GENERATE]
    assert len(gen_calls) == 2
    regen_payload = json.loads(gen_calls[1]["input_payload"])
    assert regen_payload["type_plan"] == [{"question_type": "mcq", "count": 2, "max_images": 0, "image_for_stem": True}]


# -----------------------------------------------------------------------------
# 5. N=1 single set-mode — collapses onto the same path.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_n1_single_element_set() -> None:
    plan = [{"question_type": "mcq", "count": 1, "max_images": 0}]
    executor = _FakeSetExecutor(generate_queue=[_wrapper([_mcq("only one")])])
    graph = _build(executor, _FakeGuardrail())

    out = await graph.ainvoke(_set_state(plan, question_type="mcq"), config=_cfg())

    assert out.get("refusal_reason", "") == ""
    assert len(out["accepted_set"]) == 1
    assert out["generation_summary"]["generated_total"] == 1
    assert out["generation_summary"]["generated_per_type"] == {"mcq": 1}


# -----------------------------------------------------------------------------
# 6. Bounded regenerate — critic rejects 2; regenerate ONLY those, then accept.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_regenerate_replaces_rejected_only() -> None:
    # Round 0: 8 good MCQ + 2 OE, but 2 MCQ carry the reject marker.
    round0 = [_mcq(f"MCQ {i}") for i in range(6)] + [
        _mcq(f"{_REJECT_MARKER} dup A"),
        _mcq(f"{_REJECT_MARKER} dup B"),
        _oe("OE 0"),
        _oe("OE 1"),
    ]
    # Round 1: 2 fresh MCQ replacements (no marker → accepted).
    round1 = [_mcq("fresh A"), _mcq("fresh B")]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(round0), _wrapper(round1)])
    graph = _build(executor, _FakeGuardrail())

    out = await graph.ainvoke(_set_state(_MIXED_PLAN, max_regen_rounds=1), config=_cfg())

    assert out.get("refusal_reason", "") == ""
    accepted = out["accepted_set"]
    # 6 good MCQ + 2 OE + 2 regenerated MCQ = 10, all accepted, no warning.
    assert len(accepted) == 10
    assert out.get("quality_warning") is False
    assert out["generation_summary"]["generated_per_type"] == {"mcq": 8, "oe": 2}

    # TWO generate calls (round 0 + regenerate). The regenerate generate
    # requested ONLY the 2 rejected MCQ (reduced round plan).
    gen_calls = [c for c in executor.calls if c["agent_role"] == ROLE_GENERATE]
    assert len(gen_calls) == 2
    regen_payload = json.loads(gen_calls[1]["input_payload"])
    assert regen_payload["type_plan"] == [{"question_type": "mcq", "count": 2, "max_images": 0}]
    # The regenerate generate carried a dedup hint (avoid_concepts = accepted stems).
    assert regen_payload.get("avoid_concepts")
    # Trace shows a regenerate_rejected row + a RETRY quality_gate.
    names = [r["name"] for r in out["pipeline_trace"]]
    assert "regenerate_rejected" in names
    statuses = [(r["name"], r["status"]) for r in out["pipeline_trace"]]
    assert ("quality_gate", "RETRY") in statuses


@pytest.mark.asyncio
async def test_regenerate_exhausted_includes_rejected_with_quality_warning() -> None:
    # max_regen_rounds=0 ⇒ no regenerate; rejected candidate ships with warning.
    plan = [{"question_type": "mcq", "count": 2, "max_images": 0}]
    cands = [_mcq("good"), _mcq(f"{_REJECT_MARKER} weak")]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    graph = _build(executor, _FakeGuardrail())

    out = await graph.ainvoke(_set_state(plan, question_type="mcq", max_regen_rounds=0), config=_cfg())

    assert out.get("refusal_reason", "") == ""
    assert out.get("quality_warning") is True
    accepted = out["accepted_set"]
    assert len(accepted) == 2  # both included (rejected one flagged)
    flagged = [c for c in accepted if c.get("quality_warning")]
    assert len(flagged) == 1
    assert flagged[0].get("critic_notes") == "near-duplicate"
    statuses = [(r["name"], r["status"]) for r in out["pipeline_trace"]]
    assert ("quality_gate", "QUALITY_WARNING") in statuses


# -----------------------------------------------------------------------------
# 7. Whole-batch post-screen — one Armor BLOCK refuses the entire set.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardrail_post_set_block_refuses_whole_batch() -> None:
    cands = [_mcq("clean A"), _mcq("contains TOXIN here")]
    executor = _FakeSetExecutor(generate_queue=[_wrapper(cands)])
    guardrail = _FakeGuardrail(block_substrings=["TOXIN"])
    graph = _build(executor, guardrail)

    plan = [{"question_type": "mcq", "count": 2, "max_images": 0}]
    out = await graph.ainvoke(_set_state(plan, question_type="mcq"), config=_cfg())

    assert out.get("refusal_reason") == "GUARDRAIL_POST"
    names = [r["name"] for r in out["pipeline_trace"]]
    assert "critique" not in names  # blocked before critique
    assert "publish_refused" in names


# -----------------------------------------------------------------------------
# 8. Cross-service round-trip — the orchestrator decoder reads the type_plan
#    wire shape chora-creation's protomarshal emits (belt-and-braces vs P2↔P0).
# -----------------------------------------------------------------------------


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld(field_no: int, data: bytes) -> bytes:
    return _varint((field_no << 3) | 2) + _varint(len(data)) + data


def _vf(field_no: int, value: int) -> bytes:
    return _varint(field_no << 3) + _varint(value)


def test_cross_service_type_plan_wire_round_trip() -> None:
    from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire import (
        decode_ai_assist_started,
    )

    # Hand-encode an AiAssistStarted with two field-21 GenerationTypeQuota
    # submessages, EXACTLY as chora-creation protomarshal emits them
    # ({1:str question_type, 2:varint count, 3:varint max_images} under tag 21).
    quota_mcq = _ld(1, b"mcq") + _vf(2, 8) + _vf(3, 3)
    quota_oe = _ld(1, b"oe") + _vf(2, 2) + _vf(3, 1)
    msg = (
        _ld(2, b"assist-xyz")  # field 2 assist_id
        + _ld(3, b"tenant-1")  # field 3 tenant_id
        + _ld(6, b"mixed")  # field 6 content_type
        + _vf(8, 10)  # field 8 requested_count
        + _ld(21, quota_mcq)  # field 21 type_plan[0]
        + _ld(21, quota_oe)  # field 21 type_plan[1]
    )

    decoded = decode_ai_assist_started(msg)

    assert decoded["assist_id"] == "assist-xyz"
    assert decoded["content_type"] == "mixed"
    assert decoded["requested_count"] == 10
    assert decoded["type_plan"] == [
        {"question_type": "mcq", "count": 8, "max_images": 3},
        {"question_type": "oe", "count": 2, "max_images": 1},
    ]
