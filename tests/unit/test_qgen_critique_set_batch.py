"""CHO-2397 (ADR-251 D2/D3): batch critique per chunk.

critique_set_node makes ONE qgen_critic call for the whole chunk (set_mode +
candidates array with orchestrator-assigned candidate_ids) and
parse_critique_set_response enforces the candidate_id echo contract STRICTLY:
a missing, unknown, or duplicated id fails the chunk loudly (raise ->
subscriber NACK -> redelivery resumes at critique). There is deliberately NO
per-candidate fallback and NO catch on executor errors: a mis-correlated
verdict silently swaps an accept and a reject, and a swallowed transport error
turns a retryable chunk into a fabricated rejection.

Trace ABI (owner-ruled): ONE row named ``critique`` carries the call's real
token counts (so the runner's per-hop aggregation and the model-gateway's
TokenUsageLedger event stay one-to-one with real calls); per-candidate
``critique_verdict`` rows carry the verdicts and NO token fields.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.orchestrators.qgen_crew import (
    ROLE_CRITIQUE,
    critique_set_node,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
    _aggregate_tokens_for,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_set_plan import (
    AgentContractViolation,
    parse_critique_set_response,
)

# ---------------------------------------------------------------------------
# parse_critique_set_response: strict echo-contract parse
# ---------------------------------------------------------------------------


def _verdict(cid: str, accepted: bool, notes: str = "n") -> dict[str, Any]:
    return {
        "candidate_id": cid,
        "accepted": accepted,
        "critique_notes": notes,
        "suggested_revisions": [] if accepted else ["fix it"],
    }


def test_parse_happy_path_order_independent() -> None:
    raw = {"verdicts": [_verdict("c1", False, "bad"), _verdict("c0", True, "good")]}
    got = parse_critique_set_response(raw, ["c0", "c1"])
    assert set(got) == {"c0", "c1"}
    assert got["c0"]["accepted"] is True
    assert got["c1"]["accepted"] is False
    assert got["c1"]["critique_notes"] == "bad"
    assert got["c1"]["suggested_revisions"] == ["fix it"]


def test_parse_missing_id_raises_naming_it() -> None:
    raw = {"verdicts": [_verdict("c0", True)]}
    with pytest.raises(AgentContractViolation, match="c1"):
        parse_critique_set_response(raw, ["c0", "c1"])


def test_parse_unknown_id_raises_naming_it() -> None:
    raw = {"verdicts": [_verdict("c0", True), _verdict("c9", False)]}
    with pytest.raises(AgentContractViolation, match="c9"):
        parse_critique_set_response(raw, ["c0", "c1"])


def test_parse_duplicate_id_raises_naming_it() -> None:
    raw = {"verdicts": [_verdict("c0", True), _verdict("c0", False)]}
    with pytest.raises(AgentContractViolation, match="c0"):
        parse_critique_set_response(raw, ["c0", "c1"])


def test_parse_non_object_raw_raises() -> None:
    with pytest.raises(AgentContractViolation):
        parse_critique_set_response("nope", ["c0"])


def test_parse_verdicts_not_a_list_raises() -> None:
    with pytest.raises(AgentContractViolation):
        parse_critique_set_response({"verdicts": {"c0": True}}, ["c0"])


def test_parse_entry_not_an_object_raises() -> None:
    with pytest.raises(AgentContractViolation):
        parse_critique_set_response({"verdicts": ["c0"]}, ["c0"])


def test_parse_verdict_fields_coerce_like_single_path() -> None:
    # Field tolerance mirrors the single-path parse (bool/str/list coercion);
    # only the ID correlation is strict.
    raw = {"verdicts": [{"candidate_id": "c0", "accepted": 1, "critique_notes": None}]}
    got = parse_critique_set_response(raw, ["c0"])
    assert got["c0"]["accepted"] is True
    assert got["c0"]["critique_notes"] == ""
    assert got["c0"]["suggested_revisions"] == []


# ---------------------------------------------------------------------------
# critique_set_node: one call per chunk + trace ABI + fail-loud
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, payload: str, input_tokens: int = 111, output_tokens: int = 22) -> None:
        self.output_payload = payload
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _EchoExecutor:
    """Returns a valid verdicts payload echoing the ids it was sent,
    rejecting the ids listed in ``reject``."""

    def __init__(self, reject: set[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._reject = reject or set()

    async def execute(self, **kwargs: Any) -> _Resp:
        self.calls.append(kwargs)
        sent = json.loads(kwargs["input_payload"])
        verdicts = [
            {
                "candidate_id": c["candidate_id"],
                "accepted": c["candidate_id"] not in self._reject,
                "critique_notes": f"notes-{c['candidate_id']}",
                "suggested_revisions": [],
            }
            for c in sent["candidates"]
        ]
        return _Resp(json.dumps({"verdicts": verdicts}))


def _chunk_state(n: int = 3, **extra: Any) -> dict[str, Any]:
    return {
        "job_id": "job-1",
        "tenant_id": "tenant-test",
        "prompt": "Generate a set on photosynthesis",
        "candidate_set": [{"stem": f"s{i}", "question_type": "mcq"} for i in range(n)],
        "chunk_index": 0,
        "chunk_count": 2,
        **extra,
    }


async def test_one_call_carries_all_candidates_with_ids() -> None:
    ex = _EchoExecutor()
    await critique_set_node(_chunk_state(3), executor=ex)
    assert len(ex.calls) == 1, "batch critique must be ONE executor call per chunk"
    call = ex.calls[0]
    assert call["agent_role"] == ROLE_CRITIQUE
    sent = json.loads(call["input_payload"])
    assert sent["set_mode"] is True
    assert [c["candidate_id"] for c in sent["candidates"]] == ["c0", "c1", "c2"]
    assert sent["candidates"][1]["stem"] == "s1"


async def test_verdicts_route_accept_and_reject_with_notes() -> None:
    ex = _EchoExecutor(reject={"c1"})
    out = await critique_set_node(_chunk_state(3), executor=ex)
    assert [c["stem"] for c in out["accepted_set"]] == ["s0", "s2"]
    assert [c["stem"] for c in out["rejected_set"]] == ["s1"]
    assert out["rejected_set"][0]["_critic_notes"] == "notes-c1"


async def test_trace_one_tokened_critique_row_and_tokenless_verdict_rows() -> None:
    ex = _EchoExecutor(reject={"c2"})
    out = await critique_set_node(_chunk_state(3), executor=ex)
    rows = out["pipeline_trace"]
    critique_rows = [r for r in rows if r.get("name") == "critique"]
    verdict_rows = [r for r in rows if r.get("name") == "critique_verdict"]
    assert len(critique_rows) == 1, "exactly ONE tokened critique row per real call"
    assert critique_rows[0]["input_tokens"] == 111
    assert critique_rows[0]["output_tokens"] == 22
    assert len(verdict_rows) == 3
    for r in verdict_rows:
        assert "input_tokens" not in r and "output_tokens" not in r
    statuses = [r["status"] for r in verdict_rows]
    assert statuses == ["ACCEPTED", "ACCEPTED", "REJECTED"]
    # The runner's per-hop aggregation counts the ONE call, undistorted by the
    # verdict rows.
    assert _aggregate_tokens_for(rows, "critique") == (111, 22, 0)


async def test_echo_violation_raises_out_of_the_node() -> None:
    class _DropsOne:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def execute(self, **kwargs: Any) -> _Resp:
            self.calls.append(kwargs)
            sent = json.loads(kwargs["input_payload"])
            verdicts = [
                {"candidate_id": c["candidate_id"], "accepted": True, "critique_notes": "n", "suggested_revisions": []}
                for c in sent["candidates"][1:]  # drops c0
            ]
            return _Resp(json.dumps({"verdicts": verdicts}))

    with pytest.raises(AgentContractViolation, match="c0"):
        await critique_set_node(_chunk_state(3), executor=_DropsOne())


async def test_executor_error_propagates_no_silent_fallback() -> None:
    class _Boom:
        async def execute(self, **kwargs: Any) -> _Resp:
            raise RuntimeError("transport down")

    with pytest.raises(RuntimeError, match="transport down"):
        await critique_set_node(_chunk_state(2), executor=_Boom())


async def test_empty_candidate_set_makes_no_call() -> None:
    ex = _EchoExecutor()
    out = await critique_set_node(_chunk_state(0), executor=ex)
    assert ex.calls == []
    assert out.get("rejected_set", []) == []


async def test_accepted_set_accumulates_across_rounds() -> None:
    ex = _EchoExecutor()
    prior = [{"stem": "earlier", "question_type": "mcq"}]
    out = await critique_set_node(_chunk_state(2, accepted_set=prior), executor=ex)
    assert [c["stem"] for c in out["accepted_set"]] == ["earlier", "s0", "s1"]


# ---------------------------------------------------------------------------
# Executor session-state: the critic branch stamps set_mode
# ---------------------------------------------------------------------------


def test_build_session_state_critic_stamps_set_mode() -> None:
    from chora_ai_kernel_orchestrator.adapter.agent_io import (
        ROLE_QGEN_CRITIC,
    )
    from chora_ai_kernel_orchestrator.adapter.agent_io import (
        build_session_state as _build_session_state,
    )

    with_flag = _build_session_state(
        agent_role=ROLE_QGEN_CRITIC,
        execution_id="job-1:critique_set:0",
        tenant_id="tenant-test",
        input_obj={"set_mode": True, "candidates": [], "prompt": "p"},
    )
    assert with_flag["set_mode"] is True

    without_flag = _build_session_state(
        agent_role=ROLE_QGEN_CRITIC,
        execution_id="job-1:critique:0",
        tenant_id="tenant-test",
        input_obj={"prompt": "p", "question_type": "mcq"},
    )
    assert "set_mode" not in without_flag
