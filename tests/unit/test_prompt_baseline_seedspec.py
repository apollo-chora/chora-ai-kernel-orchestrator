"""ADR-197 baseline catalogue drift pin (CHO-2368).

The v1 baseline seed transcribes each scoped agent's CURRENT prompt segments
into the prompt registry. This suite pins the seed fixture to the embedded
repo sources so that EITHER side moving alone goes RED:

- OE segments byte-pin to ``agents/oe_grading_adk_go/internal/agent/prompts/v1``
  files (the go:embed runtime authority) and to the OE goldens.
- qgen segments byte-pin to the Go-test-pinned goldens under
  ``agents/qgen_adk_go/internal/agent/testdata`` (the composer strings are the
  runtime authority; the goldens are their frozen rendering).
- The parameterized fill templates keep placeholder form from the prompts/v1
  mirrors and are ADDITIONALLY regex-pinned (placeholders wildcarded,
  whitespace normalized) against the runtime goldens, so a mirror that goes
  inert relative to the composer is caught here.
- familiar segment TEMPLATES regex-pin against the familiar golden
  (``golden_instruction_canonical.txt``), which its own Go test pins to
  ComposeInstruction.

Same discipline as the Go seedspec drift tests (chora-consumption 0106).
"""

from __future__ import annotations

import hashlib

import pytest

from chora_ai_kernel_orchestrator.domain.prompt_registry import baseline_seedspec as spec

EXPECTED_SEGMENTS: dict[str, dict[str, bool]] = {
    # segment_id -> locked
    "qgen_question": {
        "context": True,
        "role": False,
        "examples": False,
        "audience": True,
        "task_new_mcq": False,
        "task_new_oe": False,
        "task_fill_mcq": False,
        "task_fill_oe": False,
        "output_new_mcq": True,
        "output_new_oe": True,
        "output_fill_mcq": True,
        "output_fill_oe": True,
        "safety_tail": True,
    },
    "qgen_critic": {
        "context": True,
        "role": False,
        "examples": False,
        "audience": True,
        "task_mcq": False,
        "task_oe": False,
        "candidate_frame": True,
        "output_mcq": True,
        "output_oe": True,
        "safety_tail": True,
    },
    "oe_evaluator": {
        "context": True,
        "role": False,
        "examples": False,
        "audience": True,
        "task": False,
        "output": True,
        "summary_context": True,
        "summary_role": False,
        "summary_examples": False,
        "summary_audience": True,
        "summary_task": False,
        "summary_output": True,
    },
    "oe_moderator": {
        "context": True,
        "role": False,
        "examples": False,
        "audience": True,
        "task": False,
        "output": True,
    },
    "familiar": {
        "context_frame": True,
        "role_frame": False,
        "examples_frame": False,
        "audience_frame": True,
        "task_frame": False,
        "expected_output_frame": True,
        "untrusted_fence": True,
        "skills_frame": True,
    },
}


def test_covers_exactly_the_five_agents() -> None:
    segs = spec.baseline_segments()
    assert {s.agent_id for s in segs} == set(EXPECTED_SEGMENTS)


@pytest.mark.parametrize("agent_id", sorted(EXPECTED_SEGMENTS))
def test_segment_sets_exact(agent_id: str) -> None:
    segs = [s for s in spec.baseline_segments() if s.agent_id == agent_id]
    got = {s.segment_id: s.locked for s in segs}
    assert got == EXPECTED_SEGMENTS[agent_id]


def test_bodies_non_empty_and_hashes_are_sha256_of_body() -> None:
    for s in spec.baseline_segments():
        assert s.body.strip(), f"{s.agent_id}/{s.segment_id} body is empty"
        want = hashlib.sha256(s.body.encode("utf-8")).hexdigest()
        assert s.content_hash == want, f"{s.agent_id}/{s.segment_id} hash mismatch"


def test_bodies_do_not_contain_the_dollar_quote_marker() -> None:
    for s in spec.baseline_segments():
        assert "$chora_seed$" not in s.body, f"{s.agent_id}/{s.segment_id}"


def test_positions_are_unique_and_ordered_per_agent() -> None:
    for agent_id in EXPECTED_SEGMENTS:
        positions = [s.position for s in spec.baseline_segments() if s.agent_id == agent_id]
        assert positions == sorted(positions)
        assert len(positions) == len(set(positions))


# ---------------------------------------------------------------------------
# Drift pins proper. baseline_segments() slices the repo sources at call time,
# so the byte-pin against the sources is inherent; what remains is (a) the
# mirror-vs-runtime inertness pins and (b) the familiar template pins, both of
# which compare fixture TEMPLATES to a DIFFERENT file than the one they were
# authored from.
# ---------------------------------------------------------------------------


def test_qgen_shared_segments_identical_across_all_four_goldens() -> None:
    """role/examples/audience/context/safety_tail must not depend on template."""
    spec.assert_qgen_shared_segments_consistent()


def test_qgen_fill_templates_match_runtime_goldens() -> None:
    """The prompts/v1 fill mirrors must still describe what the composer emits.

    Placeholders ({{author_stem}}, {{author_options}}, {{author_rubric}}) are
    wildcarded and whitespace is normalized; every static run of the mirror
    template must appear in the golden's [TASK] section in order.
    """
    spec.assert_fill_templates_match_goldens()


def test_critic_safety_tail_identical_across_both_goldens() -> None:
    spec.assert_critic_shared_segments_consistent()


def test_oe_txt_sources_match_composed_goldens() -> None:
    """The go:embed .txt files and the composed goldens agree (trim-equal)."""
    spec.assert_oe_sources_match_goldens()


def test_familiar_templates_match_golden() -> None:
    """Every familiar segment template regex-matches the composed golden."""
    spec.assert_familiar_templates_match_golden()


def test_familiar_untrusted_fence_preamble_verbatim_in_golden() -> None:
    """The fence preamble is safety-critical: byte-verbatim, not just regexed."""
    fence = next(s for s in spec.baseline_segments() if s.agent_id == "familiar" and s.segment_id == "untrusted_fence")
    golden = spec.familiar_golden_text()
    preamble = fence.body.split("\n")[0]
    assert preamble.startswith("[UNTRUSTED DATA]")
    assert preamble in golden
