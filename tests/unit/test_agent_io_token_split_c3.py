"""C3: per-hop input/output token capture in agent_io.map_agent_response.

RED→GREEN per [[feedback-strict-tdd]]. Closes the W6 eval C3/C4-marginal gap:
the eval found `pipeline_trace` generate/critique rows carried
`input_tokens=0`/`output_tokens=0`, so the token-compounding criterion read 0.

Per the SHARED CONTRACT (CR qgen 2026-06-01): the qgen agent response JSON
carries top-level ``input_tokens`` (int) + ``output_tokens`` (int);
``tokens_consumed_total`` stays = their sum. ``_map_response`` must surface the
split onto :class:`AgentExecutorResponse` while keeping ``tokens_consumed_total``
working (fallback = input + output when the total is absent).
"""

from __future__ import annotations

import json

from chora_ai_kernel_orchestrator.adapter import agent_io as exe
from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)

# -----------------------------------------------------------------------------
# AgentExecutorResponse carries the per-hop split
# -----------------------------------------------------------------------------


def test_agent_executor_response_has_token_split_fields() -> None:
    """The DTO exposes input_tokens + output_tokens (default 0) so the
    qgen_crew nodes can stamp the per-hop split onto the trace rows."""
    resp = AgentExecutorResponse(
        execution_id="x",
        output_payload="{}",
        input_tokens=120,
        output_tokens=75,
        tokens_consumed_total=195,
    )
    assert resp.input_tokens == 120
    assert resp.output_tokens == 75
    assert resp.tokens_consumed_total == 195


def test_agent_executor_response_token_split_defaults_zero() -> None:
    """Back-compat: callers that don't set the split get 0 (the M11
    baseline shape — only tokens_consumed_total was ever populated)."""
    resp = AgentExecutorResponse(execution_id="x", output_payload="{}")
    assert resp.input_tokens == 0
    assert resp.output_tokens == 0


# -----------------------------------------------------------------------------
# _map_response extracts the top-level split from the agent JSON
# -----------------------------------------------------------------------------


def test_map_response_extracts_input_output_tokens_from_candidate() -> None:
    """Critic-shape (flat dict) with top-level input/output tokens — the
    mapper surfaces them onto the response."""
    terminal_text = json.dumps(
        {
            "accepted": True,
            "critique_notes": "",
            "input_tokens": 800,
            "output_tokens": 42,
        }
    )
    resp = exe.map_agent_response(
        execution_id="exec-1",
        terminal_text=terminal_text,
        agent_role=exe.ROLE_QGEN_CRITIC,
    )
    assert resp.input_tokens == 800
    assert resp.output_tokens == 42
    # tokens_consumed_total honoured as the sum when total absent.
    assert resp.tokens_consumed_total == 842


def test_map_response_total_present_wins_over_sum() -> None:
    """When the agent emits an explicit tokens_consumed_total it is
    preserved verbatim (NOT recomputed from the split) — the agent is the
    authority on its own accounting."""
    terminal_text = json.dumps(
        {
            "stem": "x",
            "question_type": "mcq",
            "input_tokens": 100,
            "output_tokens": 20,
            "tokens_consumed_total": 999,
        }
    )
    resp = exe.map_agent_response(
        execution_id="exec-1",
        terminal_text=terminal_text,
        agent_role=exe.ROLE_QGEN_QUESTION,
    )
    assert resp.input_tokens == 100
    assert resp.output_tokens == 20
    assert resp.tokens_consumed_total == 999


def test_map_response_total_fallback_when_split_absent() -> None:
    """No split + no total → all zero (the M11 baseline shape)."""
    terminal_text = json.dumps({"stem": "x", "question_type": "mcq"})
    resp = exe.map_agent_response(
        execution_id="exec-1",
        terminal_text=terminal_text,
        agent_role=exe.ROLE_QGEN_QUESTION,
    )
    assert resp.input_tokens == 0
    assert resp.output_tokens == 0
    assert resp.tokens_consumed_total == 0


def test_map_response_split_survives_scored_unwrap() -> None:
    """qgen_question evaluation-shape ({"scored": {"candidate": {...}}})
    carries the token split at the TOP level (sibling to ``scored``), so
    the split survives the candidate-unwrap path."""
    terminal_text = json.dumps(
        {
            "scored": {
                "candidate": {"stem": "x", "question_type": "mcq", "options": []},
                "composite": 0.9,
            },
            "input_tokens": 500,
            "output_tokens": 60,
        }
    )
    resp = exe.map_agent_response(
        execution_id="exec-1",
        terminal_text=terminal_text,
        agent_role=exe.ROLE_QGEN_QUESTION,
    )
    # Candidate flattened (existing behaviour) AND the split surfaced.
    decoded = json.loads(resp.output_payload)
    assert decoded["stem"] == "x"
    assert resp.input_tokens == 500
    assert resp.output_tokens == 60
    assert resp.tokens_consumed_total == 560


def test_map_response_extracts_model_id_from_top_level() -> None:
    """The agent's top-level ``model_id`` (the concrete model that produced
    the hop) rides AgentExecutorResponse so the qgen_crew nodes can stamp it
    on the pipeline_trace rows → the AgentDecisionLog proto field 6.
    chora-observability prices the per-hop token counts per model, so a
    dropped model_id zeroes the cost attribution."""
    terminal_text = json.dumps(
        {
            "stem": "x",
            "question_type": "mcq",
            "model_id": "gemini-3.1-pro-preview",
            "input_tokens": 100,
            "output_tokens": 20,
        }
    )
    resp = exe.map_agent_response(
        execution_id="exec-1",
        terminal_text=terminal_text,
        agent_role=exe.ROLE_QGEN_QUESTION,
    )
    assert resp.model_id == "gemini-3.1-pro-preview"


def test_map_response_model_id_defaults_empty() -> None:
    """No model_id in the agent JSON → blank (proto3-default-omit downstream;
    the consumer keeps the zero sentinel rather than a fabricated model)."""
    terminal_text = json.dumps({"stem": "x", "question_type": "mcq"})
    resp = exe.map_agent_response(
        execution_id="exec-1",
        terminal_text=terminal_text,
        agent_role=exe.ROLE_QGEN_QUESTION,
    )
    assert resp.model_id == ""
