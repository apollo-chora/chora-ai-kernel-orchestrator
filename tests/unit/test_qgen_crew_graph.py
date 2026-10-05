"""qgen_crew StateGraph unit tests — RED→GREEN per [[feedback-strict-tdd]].

Covers the 2-agent quality-loop state machine in
`chora_ai_kernel_orchestrator.orchestrators.qgen_crew`:

  - Happy path (critic accepts on first attempt) → publish_completed
  - Retry loop (critic rejects then accepts) → publish_completed
  - Max retries exhausted (critic rejects all) → publish_completed +
    quality_warning=True (NOT refused — refused reserved for guardrails)
  - Guardrail pre block → publish_refused (reason=GUARDRAIL_PRE)
  - Guardrail post block → publish_refused (reason=GUARDRAIL_POST)
  - Validation: unsupported question_type → publish_refused (VALIDATION)
  - Validation: missing required field → publish_refused (VALIDATION)
  - pipeline_trace populated end-to-end (IMDA D2 transparency)

Tests use:
  - MemorySaver checkpointer (D6 P1 unit-level seam; PostgresSaver in prod)
  - In-memory fake executor (duck-typed _ExecutorLike)
  - In-memory fake guardrail (duck-typed _GuardrailLike)

NO gRPC, NO Vertex AI, NO Pub/Sub — those are integration-test concerns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    FilterHit,
    GuardrailScreenInput,
    ScreenResult,
    Verdict,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_CRITIQUE,
    ROLE_GENERATE,
    build_qgen_crew_graph,
)

# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


@dataclass
class _FakeExecutor:
    """Duck-typed _ExecutorLike. Returns canned responses keyed by agent_role.

    For role-specific multi-response sequences (critic rejects then
    accepts), pass a list — the fake pops in order.
    """

    responses: dict[str, list[str]]  # role -> list of JSON output_payloads
    calls: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

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
                "tenant_id": tenant_id,
                "agent_role": agent_role,
                "input_payload": input_payload,
            }
        )
        queue = self.responses.get(agent_role, [])
        if not queue:
            raise RuntimeError(f"_FakeExecutor: no queued response for {agent_role}")
        out = queue.pop(0)
        return AgentExecutorResponse(
            execution_id=execution_id,
            output_payload=out,
            tokens_consumed_total=42,
            cost_micros_total=100,
            final_state="EXECUTION_FINAL_STATE_SUCCESS",
        )


@dataclass
class _FakeGuardrail:
    """Duck-typed ModelArmorGuardrailPort. By default allows everything;
    specific content substrings can be configured to BLOCK.

    Per ADR-169 the qgen_crew nodes now call the tier-mapped
    ``ModelArmorGuardrailPort`` with a :class:`GuardrailScreenInput` payload
    and receive a :class:`ScreenResult` back (verdict ALLOW / BLOCK /
    INSPECT_ONLY) — the node adapts that into the downstream
    ``GuardrailResult`` contract. This fake mirrors that Port shape.
    """

    block_substrings: list[str] = None  # type: ignore[assignment]
    block_reason: str = "pii_high_risk_block"
    calls: list[GuardrailScreenInput] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.block_substrings is None:
            self.block_substrings = []
        if self.calls is None:
            self.calls = []

    async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
        self.calls.append(payload)
        for sub in self.block_substrings:
            if sub in payload.content:
                return ScreenResult(
                    verdict=Verdict.BLOCK,
                    reason=self.block_reason,
                    filters=[
                        FilterHit(
                            filter_name="pi_and_jailbreak",
                            match_state="MATCH_FOUND",
                            severity="HIGH",
                        )
                    ],
                )
        return ScreenResult(verdict=Verdict.ALLOW, reason="clean")


def _good_mcq_payload() -> str:
    """Canonical MCQ candidate JSON the qgen_question agent would return."""
    return json.dumps(
        {
            "stem": "What gas do plants release as a byproduct of photosynthesis?",
            "question_type": "mcq",
            "mcq_payload": {
                "options": [
                    {
                        "option_id": "a",
                        "label": "A",
                        "text": "Oxygen",
                        "is_correct": True,
                        "explainer": "O2 from water-splitting.",
                    },
                    {
                        "option_id": "b",
                        "label": "B",
                        "text": "Carbon dioxide",
                        "is_correct": False,
                        "explainer": "CO2 is the INPUT, not the byproduct.",
                    },
                    {
                        "option_id": "c",
                        "label": "C",
                        "text": "Nitrogen",
                        "is_correct": False,
                        "explainer": "Plants don't fix N2 from photosynthesis.",
                    },
                    {
                        "option_id": "d",
                        "label": "D",
                        "text": "Methane",
                        "is_correct": False,
                        "explainer": "Methane comes from anaerobic decomposition.",
                    },
                ],
                "scoring_mode": "single_correct",
            },
        }
    )


def _good_mcq_payload_variant() -> str:
    """A SECOND distinct MCQ candidate (different stem) so multi-attempt
    tests can assert per-attempt candidate_json rows differ when the
    regenerator yields a materially different candidate."""
    return json.dumps(
        {
            "stem": "Which molecule is the immediate oxygen source in photosynthesis?",
            "question_type": "mcq",
            "mcq_payload": {
                "options": [
                    {
                        "option_id": "a",
                        "label": "A",
                        "text": "Water",
                        "is_correct": True,
                        "explainer": "Photolysis of H2O at PSII.",
                    },
                    {
                        "option_id": "b",
                        "label": "B",
                        "text": "Carbon dioxide",
                        "is_correct": False,
                        "explainer": "CO2 is fixed in the Calvin cycle, not split for O2.",
                    },
                    {
                        "option_id": "c",
                        "label": "C",
                        "text": "Glucose",
                        "is_correct": False,
                        "explainer": "Glucose is a product, not the O2 source.",
                    },
                    {
                        "option_id": "d",
                        "label": "D",
                        "text": "ATP",
                        "is_correct": False,
                        "explainer": "ATP is an energy carrier.",
                    },
                ],
                "scoring_mode": "single_correct",
            },
        }
    )


def _critic_accept() -> str:
    return json.dumps({"accepted": True, "critique_notes": "", "suggested_revisions": []})


def _critic_reject(notes: str = "stem ambiguous") -> str:
    return json.dumps(
        {
            "accepted": False,
            "critique_notes": notes,
            "suggested_revisions": ["clarify the stem"],
        }
    )


def _starting_state(**overrides: Any) -> dict[str, Any]:
    """Caller-supplied state mirroring AiAssistStarted event."""
    base: dict[str, Any] = {
        "job_id": "job-aaa-bbb-ccc",
        "tenant_id": "tenant-test",
        "gcid": "gcid-author",
        "prompt": "Generate a question on photosynthesis byproducts",
        "question_type": "mcq",
        "metadata": {"subject": "Biology", "cognitive_level": "knowledge"},
        "max_retries": 3,
    }
    base.update(overrides)
    return base


def _build(executor: Any, guardrail: Any) -> Any:
    """Build a compiled graph with InMemorySaver for unit tests."""
    return build_qgen_crew_graph(
        executor=executor,
        guardrail=guardrail,
        checkpointer=MemorySaver(),
    )


def _invoke_config(job_id: str = "job-aaa-bbb-ccc") -> dict[str, Any]:
    return {"configurable": {"thread_id": job_id}}


# -----------------------------------------------------------------------------
# Happy path
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_mcq_critic_accepts_first_attempt() -> None:
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(), config=_invoke_config())

    assert out.get("completed_candidate") is not None
    assert out["completed_candidate"].question_type == "mcq"
    assert out.get("quality_warning") is False
    assert out.get("refusal_reason", "") == ""
    assert out.get("attempt_count") == 1

    # pipeline_trace covers all happy-path nodes.
    names = [row["name"] for row in out["pipeline_trace"]]
    assert names == [
        "validate_input",
        "guardrail_pre",
        "generate",
        "guardrail_post",
        "critique",
        "quality_gate",
        "publish_completed",
    ]
    # No residual queued responses (one of each role consumed).
    assert executor.responses[ROLE_GENERATE] == []
    assert executor.responses[ROLE_CRITIQUE] == []


# -----------------------------------------------------------------------------
# Retry loop — critic rejects then accepts
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_loop_rejects_then_accepts() -> None:
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload(), _good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_reject("option_id=b is implausible"), _critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(), config=_invoke_config())

    assert out.get("completed_candidate") is not None
    assert out.get("quality_warning") is False
    assert out.get("attempt_count") == 2

    # Trace shows: validate → pre → generate(1) → post → critique(1: REJECTED) →
    # quality_gate(RETRY) → generate(2) → post → critique(2: ACCEPTED) →
    # quality_gate(ACCEPTED) → publish_completed
    statuses = [(r["name"], r["status"]) for r in out["pipeline_trace"]]
    assert ("critique", "REJECTED") in statuses
    assert ("critique", "ACCEPTED") in statuses
    assert ("quality_gate", "RETRY") in statuses
    assert ("quality_gate", "ACCEPTED") in statuses

    assert executor.responses[ROLE_GENERATE] == []
    assert executor.responses[ROLE_CRITIQUE] == []


# -----------------------------------------------------------------------------
# Max retries exhausted — completed with quality_warning
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_retries_exhausted_completes_with_quality_warning() -> None:
    """User-locked semantics 2026-05-17: retries exhausted → completed.v1
    with quality_warning=True + last critic_notes. NOT refused.v1."""
    # max_retries=2 → up to 3 attempts. All critic rejects.
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload(), _good_mcq_payload(), _good_mcq_payload()],
            ROLE_CRITIQUE: [
                _critic_reject("issue 1"),
                _critic_reject("issue 2"),
                _critic_reject("issue 3 final"),
            ],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(max_retries=2), config=_invoke_config())

    assert out.get("completed_candidate") is not None
    assert out.get("quality_warning") is True
    assert out.get("refusal_reason", "") == ""  # NOT refused
    assert out.get("attempt_count") == 3

    # Trace has 3 critique REJECTED rows + 1 quality_gate QUALITY_WARNING.
    statuses = [(r["name"], r["status"]) for r in out["pipeline_trace"]]
    assert statuses.count(("critique", "REJECTED")) == 3
    assert ("quality_gate", "QUALITY_WARNING") in statuses


# -----------------------------------------------------------------------------
# Guardrail pre block — publish_refused with reason=GUARDRAIL_PRE
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardrail_pre_blocks_routes_to_publish_refused() -> None:
    executor = _FakeExecutor(responses={ROLE_GENERATE: [], ROLE_CRITIQUE: []})
    guardrail = _FakeGuardrail(block_substrings=["SSN"])
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(
        _starting_state(prompt="Generate an OE about SSN 123-45-6789"),
        config=_invoke_config(),
    )

    assert out.get("completed_candidate") is None
    assert out.get("refusal_reason") == "GUARDRAIL_PRE"
    assert out.get("refusal_armor_verdict") == "armor:pii_high_risk_block"
    assert out.get("refusal_user_facing_message")

    names = [r["name"] for r in out["pipeline_trace"]]
    assert "generate" not in names  # generator never invoked
    assert "publish_refused" in names

    # Executor never called — no engine cost burned.
    assert len(executor.calls) == 0


# -----------------------------------------------------------------------------
# Guardrail post block — publish_refused with reason=GUARDRAIL_POST
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardrail_post_blocks_routes_to_publish_refused() -> None:
    # Pre-screen allows; post-screen blocks on the candidate payload.
    executor = _FakeExecutor(responses={ROLE_GENERATE: [_good_mcq_payload()], ROLE_CRITIQUE: []})
    guardrail = _FakeGuardrail(block_substrings=["Methane"])  # appears in candidate
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(), config=_invoke_config())

    assert out.get("completed_candidate") is None
    assert out.get("refusal_reason") == "GUARDRAIL_POST"
    assert out.get("last_candidate_on_refusal") is not None
    names = [r["name"] for r in out["pipeline_trace"]]
    assert "generate" in names  # generator did run
    assert "critique" not in names  # critic skipped
    assert "publish_refused" in names


# -----------------------------------------------------------------------------
# Validation: unsupported question_type → publish_refused (VALIDATION)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_refuses_unsupported_question_type() -> None:
    executor = _FakeExecutor(responses={ROLE_GENERATE: [], ROLE_CRITIQUE: []})
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(
        _starting_state(question_type="flashcard"),  # reserved sentinel
        config=_invoke_config(),
    )

    assert out.get("refusal_reason") == "VALIDATION"
    assert "flashcard" in (out.get("refusal_user_facing_message") or "")
    names = [r["name"] for r in out["pipeline_trace"]]
    assert names == ["validate_input", "publish_refused"]
    # Guardrail never invoked.
    assert len(guardrail.calls) == 0


# -----------------------------------------------------------------------------
# Validation: missing required field → publish_refused (VALIDATION)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_refuses_missing_required_field() -> None:
    executor = _FakeExecutor(responses={ROLE_GENERATE: [], ROLE_CRITIQUE: []})
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(
        _starting_state(prompt=""),  # missing prompt
        config=_invoke_config(),
    )

    assert out.get("refusal_reason") == "VALIDATION"
    assert "prompt" in (out.get("refusal_user_facing_message") or "")
    names = [r["name"] for r in out["pipeline_trace"]]
    assert names == ["validate_input", "publish_refused"]


# -----------------------------------------------------------------------------
# Pipeline trace shape (IMDA D2 transparency)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_trace_carries_attempt_and_engine_metadata() -> None:
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(), config=_invoke_config())

    # `generate` and `critique` trace rows must carry attempt + token info.
    generate_row = next(r for r in out["pipeline_trace"] if r["name"] == "generate")
    critique_row = next(r for r in out["pipeline_trace"] if r["name"] == "critique")
    for r in (generate_row, critique_row):
        assert "attempt" in r
        assert r["attempt"] == 1
        assert "input_tokens" in r
        assert r["input_tokens"] == 42


# -----------------------------------------------------------------------------
# Executor dispatched with correct role identifiers
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_dispatched_with_role_identifiers() -> None:
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    await graph.ainvoke(_starting_state(), config=_invoke_config())

    roles = [c["agent_role"] for c in executor.calls]
    assert roles == [ROLE_GENERATE, ROLE_CRITIQUE]
    # job_id is in execution_id for trace correlation.
    assert all("job-aaa-bbb-ccc" in c["execution_id"] for c in executor.calls)


# -----------------------------------------------------------------------------
# Critic notes loop back to next regenerator attempt
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critic_notes_passed_to_regenerator() -> None:
    """The regenerator on attempt N must see the critic_notes from attempt N-1
    in its input_payload (orchestrator's "loop with critic_notes as context"
    contract per docs/m13/ack-oe-ai-assist-plan-2026-05-17.md §2)."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload(), _good_mcq_payload()],
            ROLE_CRITIQUE: [
                _critic_reject("option_id=b distractor too implausible"),
                _critic_accept(),
            ],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    await graph.ainvoke(_starting_state(), config=_invoke_config())

    # First generate call has empty prior_critic_notes.
    first_gen = next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)
    first_gen_payload = json.loads(first_gen["input_payload"])
    assert first_gen_payload["prior_critic_notes"] == ""
    assert first_gen_payload["attempt_index"] == 0

    # Second generate call carries the prior critic_notes.
    gen_calls = [c for c in executor.calls if c["agent_role"] == ROLE_GENERATE]
    assert len(gen_calls) == 2
    second_gen_payload = json.loads(gen_calls[1]["input_payload"])
    assert "option_id=b distractor too implausible" in second_gen_payload["prior_critic_notes"]
    assert second_gen_payload["attempt_index"] == 1


# -----------------------------------------------------------------------------
# W6 per-attempt loop-state capture — generate + critique trace rows are
# self-sufficient to reconstruct the actor↔critic trajectory tuple
#   (attempt_index, candidate_produced, critic_accepted, critique_notes,
#    suggested_revisions)
# for the downstream managed-Eval harness, WITHOUT a new event field or proto
# change (additive opaque keys on the existing pipeline_trace rows). The
# harness zips the `generate` + `critique` rows by their shared `attempt`.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_rows_capture_per_attempt_loop_state_reject_then_accept() -> None:
    """Multi-attempt (reject→accept): each `generate` row carries
    `candidate_json` (non-empty, and DIFFERS across attempts when the
    candidates differ); each `critique` row carries `accepted` (bool) +
    `suggested_revisions` (list). The per-attempt zip reconstructs a
    coherent trajectory."""
    # Attempt 1 candidate differs from attempt 2 candidate (distinct stems).
    cand1 = _good_mcq_payload()
    cand2 = _good_mcq_payload_variant()
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [cand1, cand2],
            ROLE_CRITIQUE: [
                _critic_reject("distractor d implausible"),
                _critic_accept(),
            ],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(), config=_invoke_config())
    trace = out["pipeline_trace"]

    generate_rows = [r for r in trace if r["name"] == "generate"]
    critique_rows = [r for r in trace if r["name"] == "critique"]
    assert len(generate_rows) == 2
    assert len(critique_rows) == 2

    # --- generate rows carry candidate_json (verbatim serialized candidate)
    for r in generate_rows:
        assert "candidate_json" in r
        assert r["candidate_json"]  # non-empty
    # The candidate_json on each attempt's generate row is the exact
    # payload_json string the orchestrator carried downstream.
    g_by_attempt = {r["attempt"]: r for r in generate_rows}
    assert json.loads(g_by_attempt[1]["candidate_json"])["stem"] == json.loads(cand1)["stem"]
    assert json.loads(g_by_attempt[2]["candidate_json"])["stem"] == json.loads(cand2)["stem"]
    # Candidates DIFFER across attempts (the regenerator produced a new one).
    assert g_by_attempt[1]["candidate_json"] != g_by_attempt[2]["candidate_json"]

    # --- critique rows carry accepted + suggested_revisions
    c_by_attempt = {r["attempt"]: r for r in critique_rows}
    assert c_by_attempt[1]["accepted"] is False
    assert c_by_attempt[2]["accepted"] is True
    assert c_by_attempt[1]["suggested_revisions"] == ["clarify the stem"]
    assert c_by_attempt[2]["suggested_revisions"] == []
    # critique_notes already lives in `notes` (CHANGE 1 does not duplicate it).
    assert c_by_attempt[1]["notes"] == "distractor d implausible"

    # --- canonical per-attempt reconstruction: zip generate+critique by attempt
    trajectory = []
    for attempt in sorted(g_by_attempt):
        g = g_by_attempt[attempt]
        c = c_by_attempt[attempt]
        trajectory.append(
            (
                g["attempt"] - 1,  # 0-based attempt_index
                g["candidate_json"],  # candidate_produced
                c["accepted"],  # critic_accepted
                c.get("notes", ""),  # critique_notes
                c["suggested_revisions"],  # suggested_revisions
            )
        )
    # Attempt 0: candidate_1 rejected with a revision suggestion.
    assert trajectory[0][0] == 0
    assert json.loads(trajectory[0][1])["stem"] == json.loads(cand1)["stem"]
    assert trajectory[0][2] is False
    assert trajectory[0][3] == "distractor d implausible"
    assert trajectory[0][4] == ["clarify the stem"]
    # Attempt 1: candidate_2 accepted, no revisions.
    assert trajectory[1][0] == 1
    assert json.loads(trajectory[1][1])["stem"] == json.loads(cand2)["stem"]
    assert trajectory[1][2] is True
    assert trajectory[1][4] == []


@pytest.mark.asyncio
async def test_trace_rows_loop_state_happy_path_single_attempt() -> None:
    """Single-attempt happy path: the lone generate row carries
    candidate_json and the lone critique row carries accepted=True +
    empty suggested_revisions."""
    cand = _good_mcq_payload()
    executor = _FakeExecutor(responses={ROLE_GENERATE: [cand], ROLE_CRITIQUE: [_critic_accept()]})
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(), config=_invoke_config())
    trace = out["pipeline_trace"]

    generate_row = next(r for r in trace if r["name"] == "generate")
    critique_row = next(r for r in trace if r["name"] == "critique")
    assert json.loads(generate_row["candidate_json"])["stem"] == json.loads(cand)["stem"]
    assert critique_row["accepted"] is True
    assert critique_row["suggested_revisions"] == []
    # Attempt stamped on both (harness zip key).
    assert generate_row["attempt"] == 1
    assert critique_row["attempt"] == 1


# -----------------------------------------------------------------------------
# CHANGE 2 — debt-free critic context: critique_node threads the executor-
# expected envelope keys (author_prompt / attempt_index / prior_critic_notes)
# into the candidate payload the qgen_critic executor receives, so its
# _build_session_state has real context to read (not empty defaults).
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critique_payload_carries_author_prompt_attempt_and_prior_notes() -> None:
    """The qgen_critic executor input_payload merges author_prompt (= state
    prompt), attempt_index (0-based), and prior_critic_notes (= state
    critic_notes carried from the prior attempt)."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload(), _good_mcq_payload_variant()],
            ROLE_CRITIQUE: [
                _critic_reject("distractor d implausible"),
                _critic_accept(),
            ],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    await graph.ainvoke(
        _starting_state(prompt="Generate a question on photosynthesis byproducts"),
        config=_invoke_config(),
    )

    crit_calls = [c for c in executor.calls if c["agent_role"] == ROLE_CRITIQUE]
    assert len(crit_calls) == 2

    # First critique (attempt 1, 0-based index 0): author_prompt present,
    # prior_critic_notes empty (no prior attempt), attempt_index 0.
    first = json.loads(crit_calls[0]["input_payload"])
    assert first["author_prompt"] == "Generate a question on photosynthesis byproducts"
    assert first["attempt_index"] == 0
    assert first["prior_critic_notes"] == ""
    # The candidate payload fields are still present (merge, not replace).
    assert first["question_type"] == "mcq"
    assert "mcq_payload" in first

    # Second critique (attempt 2, 0-based index 1): attempt_index 1. The
    # prior_critic_notes reflect the critic's verdict on attempt 1, which the
    # generator cleared into critic_notes="" when it produced the new
    # candidate (generate_node clears critic_notes per-attempt); so on the
    # critic's own input it is the state's critic_notes at critique time.
    second = json.loads(crit_calls[1]["input_payload"])
    assert second["author_prompt"] == "Generate a question on photosynthesis byproducts"
    assert second["attempt_index"] == 1
    assert "prior_critic_notes" in second


@pytest.mark.asyncio
async def test_critique_payload_uses_exact_executor_key_names() -> None:
    """Fail-loud key-name match: the executor's _build_session_state reads
    EXACTLY author_prompt / attempt_index / prior_critic_notes. Any rename
    silently breaks the critic's context window — assert the literal keys."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    await graph.ainvoke(_starting_state(), config=_invoke_config())

    crit_call = next(c for c in executor.calls if c["agent_role"] == ROLE_CRITIQUE)
    payload = json.loads(crit_call["input_payload"])
    for key in ("author_prompt", "attempt_index", "prior_critic_notes"):
        assert key in payload, f"critic payload missing executor-expected key {key!r}"


@pytest.mark.asyncio
async def test_critique_payload_carries_metadata_hints() -> None:
    """Authoring-metadata wire-through (2026-06-03): critique_node merges
    state['metadata'] into the critic's input_payload so the executor stamps
    subject_hint / cognitive_level_hint / difficulty_hint, which the ADK Go
    critic reads (critic.go:477-479 → metadataHints). Without this merge the
    critic judged with empty hints (the audited dead-wire)."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    await graph.ainvoke(
        _starting_state(
            metadata={
                "subject": "Physics — Newtonian mechanics",
                "cognitive_level": "synthesis",
                "difficulty": "advanced",
            }
        ),
        config=_invoke_config(),
    )

    crit_call = next(c for c in executor.calls if c["agent_role"] == ROLE_CRITIQUE)
    payload = json.loads(crit_call["input_payload"])
    assert payload.get("metadata") == {
        "subject": "Physics — Newtonian mechanics",
        "cognitive_level": "synthesis",
        "difficulty": "advanced",
    }


@pytest.mark.asyncio
async def test_generate_and_critique_payloads_forward_author_gcid() -> None:
    """W7 follow-up: BOTH the qgen_question (generate) and qgen_critic
    (critique) executor input_payloads carry the author gcid from state, so the
    executor's _build_session_state attributes per-USER rather than the
    synthetic ``qgen-anon:{tenant_id}`` fallback W7 observed at the gateway."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    graph = _build(executor, _FakeGuardrail())

    await graph.ainvoke(_starting_state(), config=_invoke_config())

    gen_call = next(c for c in executor.calls if c["agent_role"] == ROLE_GENERATE)
    crit_call = next(c for c in executor.calls if c["agent_role"] == ROLE_CRITIQUE)
    assert json.loads(gen_call["input_payload"])["gcid"] == "gcid-author"
    assert json.loads(crit_call["input_payload"])["gcid"] == "gcid-author"


# -----------------------------------------------------------------------------
# Evaluator self-rejection — {"scored": null, "reason": "below_threshold"}
# -----------------------------------------------------------------------------
#
# Bug #5 from the 2026-05-17 parallel smoke wave (commit 03c2ae2b
# the MCQ live smoke, deleted with the HTTP executor 2026-08-23):
# qgen_question's terminal evaluator legitimately emits
#   {"scored": null, "reason": "below_threshold"}
# on composite < 0.6. Before the fix, generate_node constructed an
# empty-stem CandidatePayload from this; the critic was then handed
# an empty candidate and the quality loop wasted attempts. The fix
# detects the shape in generate_node, sets evaluator_below_threshold
# + evaluator_reason, and routes via the new guardrail_post →
# quality_gate edge (skipping critique). Retries follow the same
# budget as the critic-rejected path.


def _eval_below_threshold(*, reason: str = "below_threshold", details: str = "") -> str:
    """Mirror the live qgen_question evaluator's below-threshold wire shape."""
    payload: dict[str, Any] = {"scored": None, "reason": reason}
    if details:
        payload["details"] = details
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_evaluator_below_threshold_first_attempt_retries() -> None:
    """generate_node detects {"scored": null, "reason": "below_threshold"};
    quality_gate routes to generate (retry, same as critic-rejected). The
    critic is NEVER invoked for the below-threshold attempt."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_eval_below_threshold(), _good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],  # only consumed on attempt 2
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(), config=_invoke_config())

    # Terminal: published completed (NOT refused) with the good candidate.
    assert out.get("completed_candidate") is not None
    assert out["completed_candidate"].question_type == "mcq"
    # Attempt 2's good payload reset the flag in generate_node.
    assert out.get("evaluator_below_threshold") is False
    assert out.get("quality_warning") is False
    assert out.get("refusal_reason", "") == ""
    assert out.get("attempt_count") == 2

    # Critic invoked exactly once — for the good candidate on attempt 2.
    # The below-threshold attempt skipped the critique node.
    critic_calls = [c for c in executor.calls if c["agent_role"] == ROLE_CRITIQUE]
    assert len(critic_calls) == 1

    # Trace shows generate REJECTED (attempt 1) + RETRY + generate COMPLETED
    # (attempt 2) + critique ACCEPTED + quality_gate ACCEPTED + publish.
    statuses = [(r["name"], r["status"]) for r in out["pipeline_trace"]]
    assert ("generate", "REJECTED") in statuses
    assert ("generate", "COMPLETED") in statuses
    assert ("critique", "ACCEPTED") in statuses
    assert ("quality_gate", "RETRY") in statuses
    assert ("quality_gate", "ACCEPTED") in statuses
    # critique appears once (only after the good attempt) — not for the
    # below-threshold attempt 1.
    critique_rows = [s for s in statuses if s[0] == "critique"]
    assert len(critique_rows) == 1


@pytest.mark.asyncio
async def test_evaluator_below_threshold_exhausted_completes_with_warning() -> None:
    """All attempts emit {scored: null, below_threshold} — terminal must
    be publish_completed with quality_warning=True (NOT refused.v1).
    Mirrors the critic-rejected-exhausted semantics per user lock
    2026-05-17.
    """
    # max_retries=2 → up to 3 attempts; all 3 below-threshold.
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [
                _eval_below_threshold(reason="below_threshold"),
                _eval_below_threshold(reason="below_threshold", details="weak stem"),
                _eval_below_threshold(reason="below_threshold"),
            ],
            ROLE_CRITIQUE: [],  # never invoked
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(max_retries=2), config=_invoke_config())

    assert out.get("completed_candidate") is not None
    assert out.get("quality_warning") is True
    assert out.get("refusal_reason", "") == ""
    assert out.get("attempt_count") == 3
    assert out.get("evaluator_below_threshold") is True
    assert out.get("evaluator_reason") == "below_threshold"

    # Trace: 3 generate REJECTED + 3 quality_gate (2 RETRY + 1 QUALITY_WARNING).
    statuses = [(r["name"], r["status"]) for r in out["pipeline_trace"]]
    assert statuses.count(("generate", "REJECTED")) == 3
    assert statuses.count(("quality_gate", "RETRY")) == 2
    assert ("quality_gate", "QUALITY_WARNING") in statuses
    # Critic never appears.
    assert not any(name == "critique" for name, _ in statuses)
    # publish_completed terminal note cites the evaluator reason.
    publish_row = next(r for r in out["pipeline_trace"] if r["name"] == "publish_completed")
    assert "below_threshold" in publish_row["notes"]

    # Critic executor was never dispatched — no engine cost.
    assert all(c["agent_role"] != ROLE_CRITIQUE for c in executor.calls)


@pytest.mark.asyncio
async def test_evaluator_below_threshold_with_details_propagates_reason() -> None:
    """The optional `details` field is preserved into the trace notes
    so IMDA D2 transparency captures the evaluator's diagnosis verbatim."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [
                _eval_below_threshold(
                    reason="below_threshold",
                    details="stem references undefined variable",
                ),
                _good_mcq_payload(),
            ],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(), config=_invoke_config())

    # The first generate row's notes carries both the reason + details.
    generate_rows = [r for r in out["pipeline_trace"] if r["name"] == "generate"]
    assert len(generate_rows) == 2
    rejected_row = generate_rows[0]
    assert rejected_row["status"] == "REJECTED"
    assert "below_threshold" in rejected_row["notes"]
    assert "stem references undefined variable" in rejected_row["notes"]


@pytest.mark.asyncio
async def test_evaluator_below_threshold_then_good_candidate_clears_flag() -> None:
    """When a below-threshold attempt is followed by a good candidate,
    generate_node clears evaluator_below_threshold so quality_gate routes
    on the critic's verdict, not a stale flag."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_eval_below_threshold(), _good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(), config=_invoke_config())

    # Final state: critic-acceptance was the gating decision (NOT the
    # earlier evaluator below_threshold).
    assert out.get("evaluator_below_threshold") is False
    assert out.get("evaluator_reason") == ""
    assert out.get("quality_warning") is False
    assert out.get("completed_candidate") is not None
    # The quality_gate row immediately before publish reflects ACCEPTED,
    # not QUALITY_WARNING (the flag was cleared by attempt 2).
    quality_rows = [r for r in out["pipeline_trace"] if r["name"] == "quality_gate"]
    assert quality_rows[-1]["status"] == "ACCEPTED"


@pytest.mark.asyncio
async def test_evaluator_below_threshold_route_skips_critique_node() -> None:
    """Verify the new guardrail_post → quality_gate conditional edge
    bypasses the critique node when evaluator_below_threshold is True.
    Critic executor is never invoked; pipeline_trace contains no critique
    row for the below-threshold attempt.
    """
    # max_retries=1 → up to 2 attempts; both below-threshold → exhausted.
    # Uses 2 attempts because max_retries=0 falls back to DEFAULT_MAX_RETRIES
    # via the `or` coalesce in validate_input_node (pre-existing semantics).
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_eval_below_threshold(), _eval_below_threshold()],
            ROLE_CRITIQUE: [],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(max_retries=1), config=_invoke_config())

    names = [r["name"] for r in out["pipeline_trace"]]
    # Trace skips critique entirely. Order:
    # validate_input → guardrail_pre →
    #   generate → guardrail_post → quality_gate (RETRY) →
    #   generate → guardrail_post → quality_gate (QUALITY_WARNING) →
    #   publish_completed
    assert "critique" not in names
    assert names == [
        "validate_input",
        "guardrail_pre",
        "generate",
        "guardrail_post",
        "quality_gate",
        "generate",
        "guardrail_post",
        "quality_gate",
        "publish_completed",
    ]
    assert out.get("quality_warning") is True
    assert out.get("attempt_count") == 2
    # No critic dispatch.
    assert all(c["agent_role"] != ROLE_CRITIQUE for c in executor.calls)


# -----------------------------------------------------------------------------
# Generate node — direct unit checks on the below-threshold detection
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_node_populates_below_threshold_state_keys() -> None:
    """generate_node sets evaluator_below_threshold + evaluator_reason
    when the executor returns the {"scored": null, "reason": ...} shape.
    This is the canonical wire shape from the qgen_question evaluator
    sub-agent (StepEvaluation3 in composer_question.go)."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import generate_node

    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_eval_below_threshold(reason="below_threshold")],
            ROLE_CRITIQUE: [],
        }
    )
    state = _starting_state()
    delta = await generate_node(state, executor=executor)

    assert delta["evaluator_below_threshold"] is True
    assert delta["evaluator_reason"] == "below_threshold"
    assert delta["attempt_count"] == 1
    # Placeholder candidate preserves the evaluator's raw payload so
    # audit can replay verdict provenance.
    assert delta["current_candidate"] is not None
    assert delta["current_candidate"].stem == ""
    # The trace entry is REJECTED (not COMPLETED) for the below-threshold attempt.
    trace = delta["pipeline_trace"]
    assert trace[-1]["status"] == "REJECTED"


@pytest.mark.asyncio
async def test_generate_node_good_candidate_clears_flags() -> None:
    """generate_node clears the below-threshold flags when the next
    attempt yields a valid candidate. Critical for the
    below-threshold-then-accepted path."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import generate_node

    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [],
        }
    )
    # Pretend attempt 1 was below-threshold; attempt 2 produces a real candidate.
    state = _starting_state()
    state["attempt_count"] = 1
    state["evaluator_below_threshold"] = True
    state["evaluator_reason"] = "below_threshold"

    delta = await generate_node(state, executor=executor)

    assert delta["evaluator_below_threshold"] is False
    assert delta["evaluator_reason"] == ""
    assert delta["attempt_count"] == 2
    assert delta["current_candidate"].stem  # non-empty
    assert delta["pipeline_trace"][-1]["status"] == "COMPLETED"


# -----------------------------------------------------------------------------
# Routing functions — direct unit checks
# -----------------------------------------------------------------------------


def test_route_after_guardrail_post_below_threshold_skips_critique() -> None:
    """The new conditional edge: when evaluator_below_threshold is True
    AND no refusal_reason, route_after_guardrail_post returns
    'quality_gate' (NOT 'critique')."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
        route_after_guardrail_post,
    )

    state: dict[str, Any] = {
        "evaluator_below_threshold": True,
        "evaluator_reason": "below_threshold",
    }
    assert route_after_guardrail_post(state) == "quality_gate"


def test_route_after_guardrail_post_default_to_critique() -> None:
    """When evaluator_below_threshold is False/absent, default to critique."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
        route_after_guardrail_post,
    )

    assert route_after_guardrail_post({}) == "critique"
    assert route_after_guardrail_post({"evaluator_below_threshold": False}) == "critique"


def test_route_after_guardrail_post_refusal_wins() -> None:
    """refusal_reason set ⇒ publish_refused regardless of below-threshold."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
        route_after_guardrail_post,
    )

    state: dict[str, Any] = {
        "refusal_reason": "GUARDRAIL_POST",
        "evaluator_below_threshold": True,  # ignored under refusal
    }
    assert route_after_guardrail_post(state) == "publish_refused"


def test_route_after_quality_gate_below_threshold_retries_when_budget_left() -> None:
    """evaluator_below_threshold + attempt_count <= max_retries → generate."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
        route_after_quality_gate,
    )

    state: dict[str, Any] = {
        "evaluator_below_threshold": True,
        "evaluator_reason": "below_threshold",
        "attempt_count": 1,
        "max_retries": 3,
        "critic_result": None,
    }
    assert route_after_quality_gate(state) == "generate"


def test_route_after_quality_gate_below_threshold_exhausted_publishes() -> None:
    """evaluator_below_threshold + attempt_count > max_retries →
    render_image (then publish_completed; quality_warning is set by
    publish_completed_node). W8 reroutes the terminal-completed path through
    the render_image no-op node first; the render_image→publish_completed edge
    is unconditional."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
        route_after_quality_gate,
    )

    # max_retries=2 → max attempt = 3. attempt_count=3 → exhausted.
    state: dict[str, Any] = {
        "evaluator_below_threshold": True,
        "evaluator_reason": "below_threshold",
        "attempt_count": 3,
        "max_retries": 2,
        "critic_result": None,
    }
    assert route_after_quality_gate(state) == "render_image"


def test_route_after_quality_gate_critic_accept_when_no_eval_reject() -> None:
    """Sanity: critic.accepted=True + no eval rejection → render_image (W8
    terminal-completed path; render_image is a no-op without image_specs and
    edges unconditionally to publish_completed)."""
    from chora_ai_kernel_orchestrator.domain.qgen_crew import CritiqueResult
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
        route_after_quality_gate,
    )

    state: dict[str, Any] = {
        "evaluator_below_threshold": False,
        "attempt_count": 1,
        "max_retries": 3,
        "critic_result": CritiqueResult(accepted=True),
    }
    assert route_after_quality_gate(state) == "render_image"


# -----------------------------------------------------------------------------
# Below-threshold detection — pure helper test (edge cases)
# -----------------------------------------------------------------------------


def test_is_evaluator_below_threshold_canonical_shape() -> None:
    """Canonical {"scored": null, "reason": "below_threshold"} matches."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
        _is_evaluator_below_threshold,
    )

    assert _is_evaluator_below_threshold({"scored": None, "reason": "below_threshold"}) is True
    assert _is_evaluator_below_threshold({"scored": None, "reason": "below_threshold", "details": "weak"}) is True


def test_is_evaluator_below_threshold_negative_cases() -> None:
    """Non-below-threshold shapes do NOT match."""
    from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
        _is_evaluator_below_threshold,
    )

    # Good candidate — scored absent.
    assert _is_evaluator_below_threshold({"stem": "What is 1+1?"}) is False
    # scored dict (post-unwrap path in _map_response) — not below-threshold.
    assert _is_evaluator_below_threshold({"scored": {"candidate": {}}}) is False
    # scored: null but no reason — defensive: do NOT treat as below-threshold.
    assert _is_evaluator_below_threshold({"scored": None}) is False
    assert _is_evaluator_below_threshold({"scored": None, "reason": ""}) is False
    # Not a dict.
    assert _is_evaluator_below_threshold(None) is False
    assert _is_evaluator_below_threshold("not a dict") is False
    assert _is_evaluator_below_threshold([1, 2, 3]) is False


# -----------------------------------------------------------------------------
# ADR-169 — guardrail routed through tier-mapped ModelArmorGuardrailPort
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardrail_pre_calls_port_with_qgen_question_input_direction() -> None:
    """Pre-screen builds a GuardrailScreenInput(agent_id='qgen_question',
    direction='input') carrying the prompt + tenant_id + gcid from state."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    await graph.ainvoke(_starting_state(), config=_invoke_config())

    pre = guardrail.calls[0]
    assert isinstance(pre, GuardrailScreenInput)
    assert pre.agent_id == "qgen_question"
    assert pre.direction == "input"
    assert pre.tenant_id == "tenant-test"
    assert pre.gcid == "gcid-author"
    assert "photosynthesis" in pre.content


@pytest.mark.asyncio
async def test_guardrail_post_calls_port_with_qgen_critic_output_direction() -> None:
    """Post-screen builds a GuardrailScreenInput(agent_id='qgen_critic',
    direction='output') carrying the candidate payload_json as content."""
    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _FakeGuardrail()
    graph = _build(executor, guardrail)

    await graph.ainvoke(_starting_state(), config=_invoke_config())

    post = guardrail.calls[1]
    assert isinstance(post, GuardrailScreenInput)
    assert post.agent_id == "qgen_critic"
    assert post.direction == "output"
    assert post.tenant_id == "tenant-test"
    assert post.gcid == "gcid-author"
    # candidate payload JSON carried verbatim as content.
    assert "mcq_payload" in post.content


@pytest.mark.asyncio
async def test_guardrail_pre_inspect_only_is_allowed() -> None:
    """A Port INSPECT_ONLY verdict is treated as allowed (audit-only) —
    parity with the legacy _GuardrailAdapter collapse rule."""

    @dataclass
    class _InspectGuardrail:
        calls: list[GuardrailScreenInput] = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            if self.calls is None:
                self.calls = []

        async def screen(self, payload: GuardrailScreenInput) -> ScreenResult:
            self.calls.append(payload)
            return ScreenResult(verdict=Verdict.INSPECT_ONLY, reason="advisory")

    executor = _FakeExecutor(
        responses={
            ROLE_GENERATE: [_good_mcq_payload()],
            ROLE_CRITIQUE: [_critic_accept()],
        }
    )
    guardrail = _InspectGuardrail()
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(_starting_state(), config=_invoke_config())

    assert out.get("refusal_reason", "") == ""
    assert out.get("completed_candidate") is not None


@pytest.mark.asyncio
async def test_guardrail_block_armor_verdict_prefix_preserved() -> None:
    """A BLOCK verdict surfaces 'armor:<reason>' in refusal_armor_verdict —
    identical downstream contract to the retired _GuardrailAdapter."""
    executor = _FakeExecutor(responses={ROLE_GENERATE: [], ROLE_CRITIQUE: []})
    guardrail = _FakeGuardrail(block_substrings=["SSN"], block_reason="pii_high_risk_block")
    graph = _build(executor, guardrail)

    out = await graph.ainvoke(
        _starting_state(prompt="Generate an OE about SSN 123-45-6789"),
        config=_invoke_config(),
    )

    assert out.get("refusal_reason") == "GUARDRAIL_PRE"
    assert out.get("refusal_armor_verdict") == "armor:pii_high_risk_block"
    assert out.get("refusal_user_facing_message")


def test_build_qgen_crew_graph_rejects_armor_template_name_param() -> None:
    """ADR-169 retires the armor_template_name parameter — the tier is now
    resolved by the Port from agent-guardrail-mapping.yaml."""
    import inspect

    sig = inspect.signature(build_qgen_crew_graph)
    assert "armor_template_name" not in sig.parameters


def test_contract_mapping_resolves_qgen_agents_to_balanced() -> None:
    """The real chora-contracts mapping resolves qgen_question + qgen_critic
    to the balanced template (NOT the strict default fallback)."""
    from chora_ai_kernel_orchestrator.adapter.modelarmor import (
        ModelArmorGuardrailPort,
        StubScreener,
    )

    port = ModelArmorGuardrailPort.from_components(
        screener=StubScreener(force_verdict=Verdict.ALLOW),
        project="chora-489812",
        location="us-central1",
        environment="dev",
        # mapping_path omitted → resolves the bundled real contract file.
    )
    q = port.resolver.template_for("qgen_question")
    c = port.resolver.template_for("qgen_critic")
    assert q.endswith("/templates/chora-guardrail-balanced-dev"), q
    assert c.endswith("/templates/chora-guardrail-balanced-dev"), c
