"""RED→GREEN: an image that never rendered must be LOUD, not silent.

Live incident 2026-08-14, job ``b4d12cd0``. The author ticked "stem image" and
"answer image"; both renders 429'd; the render node fail-softed each spec and
the batch shipped as ``status=accepted`` with **no warning of any kind** on the
delivered item. The question that reached the author reads:

    "Based on the first-person perspective shown in the illustration below,
     you are cycling towards a junction..."

...with no illustration. The forced-image contract is explicit that this is not
decoration (``integralImageRule`` in composer_question.go): "the question MUST
explicitly reference and depend on it ... the learner cannot fully answer
without it". So a dropped image does not degrade the question, it BREAKS it.

Per-spec fail-soft is still right - losing 7 good questions because 1 image
was throttled would be worse. What was missing is the loud half:

  * ``quality_warning`` set on the terminal, which is already wired end-to-end
    (proto field -> chora-creation -> the A+ author banner + the O+ oversight
    gate). No contract change needed; the signal existed and was not raised.
  * a ``render_image`` trace row that names the REAL reason, so the author and
    O+ see "throttled", not a bare "failed".
  * the failure recorded in ``errors``.

Pinned for both lanes: the set-native plan (``render_image_set_node``) and the
single-candidate plan (``render_image_node``), each driven through the ADR-254
render loop (one ``render_next`` per image, then ``render_finalize``) exactly as
the graph's edges drive it; a scene image is one qgen_render dispatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.domain.qgen_crew import CandidatePayload
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    publish_completed_node,
    render_finalize_node,
    render_image_node,
    render_image_set_node,
    render_next_node,
)


@dataclass
class _FakeKroki:
    fail: bool = False

    async def render(self, *, source: str, output_format: str = "png") -> bytes:
        if self.fail:
            raise RuntimeError("Kroki render failed: POST /mermaid/png -> 503")
        return b"PNGBYTES"


@dataclass
class _ThrottledRenderer:
    """The live failure on the bus: the qgen_render agent reports a FAILED
    completion (the vendor 429'd past its retry budget); the executor raises
    it into the node, which fail-softs the spec."""

    calls: int = 0

    async def execute(self, **kwargs: Any) -> AgentExecutorResponse:
        self.calls += 1
        raise RuntimeError(
            "qgen_render dispatch returned status=FAILED: invoke vendor_error: all 1 "
            'vendor attempts failed; last vendor "vertex_ai_gemini": gemini: HTTP 429: '
            "Resource has been exhausted"
        )


@dataclass
class _OkRenderer:
    calls: int = 0

    async def execute(self, *, execution_id: str, **_: Any) -> AgentExecutorResponse:
        self.calls += 1
        return AgentExecutorResponse(
            execution_id=execution_id,
            output_payload=json.dumps({"image_uri": f"gs://b/scene/{self.calls}.png", "mime_type": "image/png"}),
            tokens_consumed_total=0,
            cost_micros_total=0,
            final_state="EXECUTION_FINAL_STATE_SUCCEEDED",
        )


@dataclass
class _FakeGcs:
    uploads: int = 0
    signed: int = 0

    async def upload_and_sign(self, *, tenant_id: str, job_id: str, data: bytes, content_type: str) -> tuple[str, str]:
        self.uploads += 1
        return (f"gs://b/{job_id}/{self.uploads}.png", f"https://s/{self.uploads}.png")

    async def sign_read_url(self, gs_uri: str) -> str:
        self.signed += 1
        return f"https://s/{gs_uri.rsplit('/', 1)[-1]}"


async def _loop(
    plan_delta: dict[str, Any], merged: dict[str, Any], *, renderer: Any, kroki: Any, gcs: Any
) -> dict[str, Any]:
    """render_next (ONE image per superstep) until the queue drains, then
    render_finalize: the graph's edges, in-process. Returns the ACCUMULATED
    delta the old single-node assertions read."""
    acc: dict[str, Any] = {}
    merged.update(plan_delta)
    acc.update(plan_delta)
    while merged.get("pending_renders"):
        delta = await render_next_node(merged, executor=renderer, kroki=kroki, gcs=gcs)
        merged.update(delta)
        acc.update(delta)
    delta = await render_finalize_node(merged)
    merged.update(delta)
    acc.update(delta)
    return acc


async def _render_set(state: dict[str, Any], *, renderer: Any, kroki: Any, gcs: Any) -> dict[str, Any]:
    merged = dict(state)
    plan = await render_image_set_node(merged, kroki=kroki, gcs=gcs)
    return await _loop(plan, merged, renderer=renderer, kroki=kroki, gcs=gcs)


async def _render_single(state: dict[str, Any], *, renderer: Any, kroki: Any, gcs: Any) -> dict[str, Any]:
    merged = dict(state)
    plan = await render_image_node(merged, kroki=kroki, gcs=gcs)
    return await _loop(plan, merged, renderer=renderer, kroki=kroki, gcs=gcs)


def _state(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": "b4d12cd0-64fd-42d4-bfe4-d29afb459848",
        "tenant_id": "11111111-1111-7111-8111-111111111111",
        "gcid": "00000000-0000-7000-8000-000000001999",
        "set_mode": True,
        "type_plan": [
            {
                "question_type": "mcq",
                "count": 1,
                "max_images": 0,
                "image_for_stem": True,
                "image_for_answer": True,
            }
        ],
        "pipeline_trace": [],
        "errors": [],
    }
    base.update(over)
    return base


def _candidate(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question_type": "mcq",
        "stem": "Based on the illustration below, what must the cyclist do?",
        "image_specs": [
            {"mode": "scene", "source": "a cyclist at a junction", "placement": "stem"},
            {"mode": "scene", "source": "the correct lane position", "placement": "answer"},
        ],
    }
    base.update(over)
    return base


def _trace_rows(delta: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [r for r in (delta.get("pipeline_trace") or []) if r.get("name") == name]


# -----------------------------------------------------------------------------
# Set lane
# -----------------------------------------------------------------------------


async def test_set_lane_raises_quality_warning_when_every_image_fails() -> None:
    """b4d12cd0 exactly: 0/2 rendered, shipped clean. Must now warn."""
    delta = await _render_set(
        _state(accepted_set=[_candidate()]),
        kroki=_FakeKroki(),
        renderer=_ThrottledRenderer(),
        gcs=_FakeGcs(),
    )

    assert delta.get("quality_warning") is True, (
        "a question whose stem says 'the illustration below' shipped with no "
        "image and no warning - the author had no way to know"
    )
    rows = _trace_rows(delta, "render_image")
    assert rows and rows[-1]["status"] == "DEGRADED"
    assert "429" in str(delta.get("errors")) or "exhausted" in str(delta.get("errors")).lower(), (
        f"the vendor reason was not recorded: {delta.get('errors')}"
    )


async def test_set_lane_warns_on_a_partial_failure_too() -> None:
    """a07383a3 rendered 5/16 and still shipped clean. One broken item in a
    batch is still a broken item."""
    ok_candidate = _candidate(
        stem="A clean question",
        image_specs=[{"mode": "mermaid", "source": "graph TD; A-->B;", "placement": "stem"}],
    )
    delta = await _render_set(
        _state(
            accepted_set=[ok_candidate, _candidate()],
            # count=2 so BOTH candidates sit inside the forced-image budget -
            # otherwise enforce_image_caps drops the surplus spec and the run
            # measures the cap backstop instead of the render failure.
            type_plan=[
                {
                    "question_type": "mcq",
                    "count": 2,
                    "max_images": 0,
                    "image_for_stem": True,
                    "image_for_answer": True,
                }
            ],
        ),
        kroki=_FakeKroki(),
        renderer=_ThrottledRenderer(),
        gcs=_FakeGcs(),
    )

    assert delta.get("quality_warning") is True
    rows = _trace_rows(delta, "render_image")
    assert "1/3" in str(rows[-1].get("notes")), rows[-1]


async def test_set_lane_stays_clean_when_every_image_renders() -> None:
    """No regression on the happy path - the 08-10/08-11 runs rendered 2/2 and
    must keep shipping without a warning."""
    delta = await _render_set(
        _state(accepted_set=[_candidate()]),
        kroki=_FakeKroki(),
        renderer=_OkRenderer(),
        gcs=_FakeGcs(),
    )

    assert not delta.get("quality_warning")
    rows = _trace_rows(delta, "render_image")
    assert rows[-1]["status"] == "COMPLETED"
    assert not delta.get("errors")


async def test_set_lane_pass_through_when_no_images_requested() -> None:
    """The dominant imageless path must not gain a warning."""
    delta = await _render_set(
        _state(
            accepted_set=[_candidate(image_specs=[])],
            type_plan=[{"question_type": "mcq", "count": 1, "max_images": 0}],
        ),
        kroki=_FakeKroki(),
        renderer=_OkRenderer(),
        gcs=_FakeGcs(),
    )
    assert not delta.get("quality_warning")


async def test_set_quality_warning_survives_the_terminal_node() -> None:
    """publish_completed_node must PRESERVE a render-raised warning - a warning
    the terminal drops is a warning the author never sees."""
    delta = await publish_completed_node(_state(set_mode=True, quality_warning=True))
    assert delta.get("quality_warning") is True


# -----------------------------------------------------------------------------
# Legacy single-candidate lane
# -----------------------------------------------------------------------------


async def test_single_lane_raises_quality_warning_when_the_image_fails() -> None:
    payload = _candidate(image_specs=[{"mode": "scene", "source": "a cyclist at a junction", "placement": "stem"}])
    delta = await _render_single(
        _state(
            set_mode=False,
            current_candidate=CandidatePayload(
                stem=payload["stem"],
                question_type="mcq",
                payload_json=json.dumps(payload),
            ),
        ),
        kroki=_FakeKroki(),
        renderer=_ThrottledRenderer(),
        gcs=_FakeGcs(),
    )

    assert delta.get("quality_warning") is True
    rows = _trace_rows(delta, "render_image")
    assert rows and rows[-1]["status"] == "DEGRADED"


async def test_single_lane_terminal_does_not_clobber_a_render_warning() -> None:
    """The single lane's terminal RE-DERIVES quality_warning from the critic;
    it must OR in the render-raised warning rather than overwrite it."""
    delta = await publish_completed_node(
        _state(
            set_mode=False,
            quality_warning=True,  # raised by render_image_node
            critic_result=None,
            evaluator_below_threshold=False,
            current_candidate={"stem": "x"},
        )
    )
    assert delta.get("quality_warning") is True, (
        "publish_completed_node overwrote the render-raised warning with the "
        "critic verdict - the missing image became invisible again"
    )


# -----------------------------------------------------------------------------
# CHO-2399 - the CHO-2395 sibling-screen fix: the SINGLE lane's exhausted
# candidate (critic rejected with the retry budget spent, or the evaluator
# self-rejected on the last attempt) is PUBLISHED with quality_warning but must
# never render its image. Found by the report session's qgen dossier
# 2026-08-16: the set lane skipped, the single lane still paid ~22s of
# gemini-3-pro-image quota per exhausted candidate.
# -----------------------------------------------------------------------------


def _single_candidate(specs: list[dict[str, Any]]) -> CandidatePayload:
    payload = {
        "stem": "Based on the illustration below, what must the cyclist do?",
        "question_type": "mcq",
        "mcq_payload": {"options": [], "scoring_mode": "single_correct"},
        "image_specs": specs,
    }
    return CandidatePayload(
        stem=payload["stem"],
        question_type="mcq",
        payload_json=json.dumps(payload),
    )


async def test_single_lane_critic_exhausted_candidate_skips_render() -> None:
    """Reaching render_image with a REJECTING critic verdict means the retry
    budget is spent (the gate only routes terminal then): the candidate ships
    warned and must not spend image quota."""
    from chora_ai_kernel_orchestrator.domain.qgen_crew import CritiqueResult

    renderer = _OkRenderer()
    gcs = _FakeGcs()
    delta = await _render_single(
        _state(
            set_mode=False,
            critic_result=CritiqueResult(accepted=False, critique_notes="dup"),
            current_candidate=_single_candidate(
                [
                    {"mode": "scene", "source": "a junction", "placement": "stem"},
                    {"mode": "scene", "source": "the lane", "placement": "answer"},
                ]
            ),
        ),
        kroki=_FakeKroki(),
        renderer=renderer,
        gcs=gcs,
    )

    assert renderer.calls == 0, "an exhausted candidate dispatched a render"
    assert gcs.uploads == 0
    decoded = json.loads(delta["current_candidate"].payload_json)
    assert "image_specs" not in decoded
    assert "image_url" not in decoded and "answer_image_url" not in decoded
    rows = _trace_rows(delta, "render_image")
    assert rows[-1]["status"] == "COMPLETED"
    assert str(rows[-1]["notes"]) == "rendered 0/0 image(s); 2 skipped (quality_warning)"
    assert not delta.get("errors")
    # publish_completed_node re-derives the warning from the critic verdict;
    # the deliberate skip itself is not an error and raises nothing here.
    assert "quality_warning" not in delta


async def test_single_lane_evaluator_exhausted_also_skips() -> None:
    delta = await _render_single(
        _state(
            set_mode=False,
            critic_result=None,
            evaluator_below_threshold=True,
            current_candidate=_single_candidate(
                [
                    {"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"},
                ]
            ),
        ),
        kroki=_FakeKroki(),
        renderer=_OkRenderer(),
        gcs=_FakeGcs(),
    )
    decoded = json.loads(delta["current_candidate"].payload_json)
    assert "image_specs" not in decoded and "image_url" not in decoded
    notes = str(_trace_rows(delta, "render_image")[-1]["notes"])
    assert notes == "rendered 0/0 image(s); 1 skipped (quality_warning)", notes


async def test_single_lane_accepted_candidate_still_renders() -> None:
    """Parity pin: an ACCEPTED candidate renders exactly as today."""
    from chora_ai_kernel_orchestrator.domain.qgen_crew import CritiqueResult

    gcs = _FakeGcs()
    delta = await _render_single(
        _state(
            set_mode=False,
            critic_result=CritiqueResult(accepted=True),
            current_candidate=_single_candidate(
                [
                    {"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"},
                ]
            ),
        ),
        kroki=_FakeKroki(),
        renderer=_OkRenderer(),
        gcs=gcs,
    )
    assert gcs.uploads == 1
    decoded = json.loads(delta["current_candidate"].payload_json)
    assert decoded.get("image_url")
    assert "image_specs" not in decoded


async def test_single_lane_exhausted_skip_needs_no_image_wiring() -> None:
    """A rejected-only spec never turns unconfigured image clients into a
    mis-config: the skip precedes the fail-loud wiring check, mirroring the
    set lane (CHO-2395)."""
    from chora_ai_kernel_orchestrator.domain.qgen_crew import CritiqueResult

    delta = await _render_single(
        _state(
            set_mode=False,
            critic_result=CritiqueResult(accepted=False, critique_notes="dup"),
            current_candidate=_single_candidate(
                [
                    {"mode": "scene", "source": "a junction", "placement": "stem"},
                ]
            ),
        ),
        kroki=None,
        renderer=None,
        gcs=None,
    )
    decoded = json.loads(delta["current_candidate"].payload_json)
    assert "image_specs" not in decoded
    notes = str(_trace_rows(delta, "render_image")[-1]["notes"])
    assert notes == "rendered 0/0 image(s); 1 skipped (quality_warning)", notes


# -----------------------------------------------------------------------------
# Scene-lane pacing: a shared-quota image model (gemini-3-pro-image, dynamic
# shared quota, ~22s/image) is never fanned into. ADR-254 D12 replaced the
# bounded in-node fan-out with the render loop: ONE qgen_render dispatch per
# superstep, each a park the completion resumes. That is the pacing now, and it
# is structural (no concurrency knob to mis-tune).
# -----------------------------------------------------------------------------


async def test_scene_renders_are_sequential_one_dispatch_per_superstep() -> None:
    renderer = _OkRenderer()
    state = _state(accepted_set=[_candidate()])
    plan = await render_image_set_node(state, kroki=_FakeKroki(), gcs=_FakeGcs())
    assert len(plan["pending_renders"]) == 2
    state.update(plan)
    step = await render_next_node(state, executor=renderer, kroki=_FakeKroki(), gcs=_FakeGcs())
    assert renderer.calls == 1, "the loop dispatches exactly one scene per superstep"
    assert len(step["pending_renders"]) == 1 and len(step["render_results"]) == 1


# -----------------------------------------------------------------------------
# Honest counting when the cap backstop trims (live run ab740e67, 2026-08-14).
#
# The agent over-generated (6 questions for a requested 3), so it emitted 6
# forced-stem specs and enforce_image_caps trimmed 3. The first cut appended the
# cap-drop message to the same `warnings` list as real render errors, which
# produced the self-contradictory trace note
#
#     "rendered 3/3 image(s); 1 failed (fail-soft)"
#
# 3 of 3 succeeded AND 1 failed cannot both be true. Worse, the ratio was
# computed against the CAPPED set, so it read as complete while 3 delivered
# questions carried no image at all - exactly the silent-missing-image defect
# the warning exists to prevent, hidden by the warning's own arithmetic.
#
# The counts are now reported against what the agent EMITTED, so a dropped spec
# and a failed render are both visible as "an image the question does not have".
# -----------------------------------------------------------------------------


async def test_dropped_over_cap_is_counted_against_what_was_emitted() -> None:
    # 2 candidates, each with 2 specs = 4 emitted; the plan budgets 1 question
    # with both placements, so the backstop trims to 2.
    delta = await _render_set(
        _state(accepted_set=[_candidate(), _candidate()]),
        kroki=_FakeKroki(),
        renderer=_OkRenderer(),
        gcs=_FakeGcs(),
    )
    rows = _trace_rows(delta, "render_image")
    notes = str(rows[-1].get("notes"))
    assert "/4" in notes, f"ratio not stated against what the agent emitted: {notes}"
    assert "dropped" in notes, f"the cap drop is not named: {notes}"
    # A dropped spec means a delivered question has no image, so it warns.
    assert delta.get("quality_warning") is True


async def test_a_cap_drop_is_never_described_as_a_render_failure() -> None:
    """The two are different things and must not share a tally."""
    delta = await _render_set(
        _state(accepted_set=[_candidate(), _candidate()]),
        kroki=_FakeKroki(),
        renderer=_OkRenderer(),
        gcs=_FakeGcs(),
    )
    notes = str(_trace_rows(delta, "render_image")[-1].get("notes"))
    assert "failed" not in notes, f"a deterministic over-budget trim was reported as a render failure: {notes}"


async def test_clean_run_still_reports_a_whole_ratio_and_no_warning() -> None:
    delta = await _render_set(
        _state(accepted_set=[_candidate()]),
        kroki=_FakeKroki(),
        renderer=_OkRenderer(),
        gcs=_FakeGcs(),
    )
    notes = str(_trace_rows(delta, "render_image")[-1].get("notes"))
    assert notes == "rendered 2/2 image(s)", notes
    assert not delta.get("quality_warning")


# -----------------------------------------------------------------------------
# CHO-2395 - a candidate the critic REJECTED (included with quality_warning by
# quality_gate_set_node under the locked exhausted semantics) is PUBLISHED,
# never RENDERED. A question that failed baseline QC must not spend ~22s of
# gemini-3-pro-image dynamic shared quota (x8 retries, 60s backoff) on an
# illustration for a question the author is being warned about anyway.
#
# The deliberate skip is a THIRD tally: never folded into the over-cap trim
# (us, by design) nor a render failure (the vendor). This story changes what
# is RENDERED, never what is PUBLISHED.
# -----------------------------------------------------------------------------


def _warned_candidate(**over: Any) -> dict[str, Any]:
    """A critic-rejected candidate as quality_gate_set_node includes it on a
    terminal pass: quality_warning stamped, critic notes carried, _critic_notes
    already stripped."""
    cand = _candidate(**over)
    cand["quality_warning"] = True
    cand["critic_notes"] = "stem is ambiguous; the distractors overlap"
    return cand


def _plan(count: int, max_images: int) -> list[dict[str, Any]]:
    return [{"question_type": "mcq", "count": count, "max_images": max_images}]


async def test_rejected_candidates_do_not_render_and_still_publish_warned() -> None:
    """2 of 5 rejected: exactly the 3 accepted render; the 2 warned ship
    imageless, in place, still warned, with no orphaned image_specs."""
    accepted = [
        _candidate(
            stem="ok-0",
            image_specs=[{"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"}],
        ),
        _warned_candidate(
            stem="warned-1",
            image_specs=[{"mode": "scene", "source": "a junction", "placement": "stem"}],
        ),
        _candidate(
            stem="ok-2",
            image_specs=[{"mode": "mermaid", "source": "graph TD; B-->C", "placement": "stem"}],
        ),
        _warned_candidate(
            stem="warned-3",
            image_specs=[{"mode": "scene", "source": "a roundabout", "placement": "stem"}],
        ),
        _candidate(
            stem="ok-4",
            image_specs=[{"mode": "mermaid", "source": "graph TD; C-->D", "placement": "stem"}],
        ),
    ]
    renderer = _OkRenderer()
    gcs = _FakeGcs()
    delta = await _render_set(
        _state(accepted_set=accepted, type_plan=_plan(5, 5)),
        kroki=_FakeKroki(),
        renderer=renderer,
        gcs=gcs,
    )

    # Cost: only the 3 accepted candidates' images were produced; the render
    # lane was never dispatched for a rejected candidate.
    assert gcs.uploads == 3, "expected exactly the 3 accepted images"
    assert renderer.calls == 0, "a critic-rejected candidate reached the render lane"

    out = delta["accepted_set"]
    assert [c["stem"] for c in out] == [
        "ok-0",
        "warned-1",
        "ok-2",
        "warned-3",
        "ok-4",
    ], "set order must be preserved"
    for cand in out:
        assert "image_specs" not in cand, "an unrendered spec leaked to a consumer"
    for idx in (1, 3):
        warned = out[idx]
        assert warned.get("quality_warning") is True
        assert warned.get("critic_notes")
        for key in ("image_url", "image_gcs_uri", "answer_image_url", "answer_image_gcs_uri"):
            assert key not in warned, f"rejected candidate carries {key}"
    for idx in (0, 2, 4):
        assert out[idx].get("image_url"), "an accepted candidate lost its image"

    notes = str(_trace_rows(delta, "render_image")[-1].get("notes"))
    assert notes == "rendered 3/3 image(s); 2 skipped (quality_warning)", notes
    assert _trace_rows(delta, "render_image")[-1]["status"] == "COMPLETED"
    # The deliberate skip is not an error and must not re-raise the job-level
    # warning through the render node (quality_gate already owns that signal).
    assert "quality_warning" not in delta
    assert not delta.get("errors")


async def test_all_accepted_set_renders_exactly_as_today() -> None:
    """No rejects: the skip machinery must be invisible (no clause, same
    counts, same status)."""
    accepted = [
        _candidate(
            stem=f"ok-{i}",
            image_specs=[{"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"}],
        )
        for i in range(5)
    ]
    gcs = _FakeGcs()
    delta = await _render_set(
        _state(accepted_set=accepted, type_plan=_plan(5, 5)),
        kroki=_FakeKroki(),
        renderer=_OkRenderer(),
        gcs=gcs,
    )
    assert gcs.uploads == 5
    notes = str(_trace_rows(delta, "render_image")[-1].get("notes"))
    assert notes == "rendered 5/5 image(s)", notes
    assert "skipped" not in notes
    assert not delta.get("quality_warning")


async def test_skip_is_a_third_tally_beside_drop_and_failure() -> None:
    """All three events at once stay separately named and separately counted:
    a deterministic over-cap trim, a vendor render failure, a deliberate
    quality_warning skip."""
    accepted = [
        _candidate(
            stem="ok-two-mermaid",
            image_specs=[
                {"mode": "mermaid", "source": "graph TD; A-->B", "placement": "stem"},
                {"mode": "mermaid", "source": "graph TD; B-->C", "placement": "answer"},
            ],
        ),
        _candidate(
            stem="ok-two-scene",
            image_specs=[
                {"mode": "scene", "source": "a junction", "placement": "stem"},
                {"mode": "scene", "source": "the lane", "placement": "answer"},
            ],
        ),
        _warned_candidate(
            stem="warned",
            image_specs=[{"mode": "scene", "source": "never rendered", "placement": "stem"}],
        ),
    ]
    renderer = _ThrottledRenderer()
    delta = await _render_set(
        # Effective cap 3: the first candidate keeps 2, the second keeps its
        # stem spec (1 dropped), and that kept scene spec then 429s.
        _state(accepted_set=accepted, type_plan=_plan(3, 3)),
        kroki=_FakeKroki(),
        renderer=renderer,
        gcs=_FakeGcs(),
    )

    assert renderer.calls == 1, "only the accepted candidate's surviving scene spec dispatches"
    notes = str(_trace_rows(delta, "render_image")[-1].get("notes"))
    assert notes == (
        "rendered 2/4 image(s); 1 dropped over budget; 1 failed (fail-soft); 1 skipped (quality_warning)"
    ), notes
    # The renderable set really is degraded (a drop + a failure), so the loud
    # half still fires; the skip contributed nothing to `missing`.
    assert _trace_rows(delta, "render_image")[-1]["status"] == "DEGRADED"
    assert delta.get("quality_warning") is True
    assert "429" in str(delta.get("errors")) or "exhausted" in str(delta.get("errors")).lower()


async def test_specs_only_on_rejected_candidates_need_no_image_wiring() -> None:
    """When every image_spec belongs to a rejected candidate there is NO image
    work, so unconfigured clients are not a mis-config: nothing renders,
    nothing raises, and the skip is still visible in the trace."""
    accepted = [
        _candidate(stem="ok-imageless", image_specs=[]),
        _warned_candidate(
            stem="warned",
            image_specs=[
                {"mode": "scene", "source": "a junction", "placement": "stem"},
                {"mode": "scene", "source": "the lane", "placement": "answer"},
            ],
        ),
    ]
    delta = await _render_set(
        _state(accepted_set=accepted, type_plan=_plan(2, 2)),
        kroki=None,
        renderer=None,
        gcs=None,
    )
    out = delta["accepted_set"]
    assert [c["stem"] for c in out] == ["ok-imageless", "warned"]
    assert all("image_specs" not in c for c in out)
    assert out[1].get("quality_warning") is True
    notes = str(_trace_rows(delta, "render_image")[-1].get("notes"))
    assert notes == "rendered 0/0 image(s); 2 skipped (quality_warning)", notes
    assert not delta.get("errors")
    assert "quality_warning" not in delta
