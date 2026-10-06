"""Agent terminal-output mapper, shared by every transport (ADR-253 D6).

The agent's terminal JSON is normalised to the ``AgentExecutorResponse`` the
crew nodes read by ONE mapper, whichever transport carried the answer (the
HTTP executor's SSE terminal text or the Pub/Sub completion's
``output_payload``). Re-deriving the unwrap or the token split per transport
would quietly rewrite pipeline_trace rows and the O+ per-agent token tiles;
that is what makes ADR-253 D6 rollback a rollback. The qgen_question unwraps
apply to the generator's agent id AND its ADR-254 lane role.
"""

from __future__ import annotations

import json
import re
from typing import Any

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.agent_io.session_state import QUESTION_ROLES

MD_FENCE_RE = re.compile(
    r"^\s*```(?:[a-zA-Z0-9_+\-]+)?\s*\n?(?P<body>.*?)\n?\s*```\s*$",
    re.DOTALL,
)


def strip_markdown_fences(text: str) -> str:
    """Peel leading + trailing ```lang … ``` fences if present.

    Gemini occasionally wraps structured output in ```json … ``` despite
    instruction to emit pure JSON. Strip those fences before json.loads
    so we don't silently collapse to {raw: <fenced>} and lose the
    structured shape (surfaced 2026-05-17 MCQ+OE smoke wave).
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    m = MD_FENCE_RE.match(stripped)
    if not m:
        return text
    return m.group("body")


def parse_terminal_json(terminal_text: str) -> Any:
    """Parse the terminal agent's JSON output, tolerating three
    Gemini-emission quirks:

    1. ```json … ``` (or plain ```… ```) markdown-fence wrapping
       (peeled via strip_markdown_fences).
    2. ONE extra trailing character past the valid JSON object
       (e.g., a stray closing brace): handled via
       JSONDecoder().raw_decode() which consumes the first valid
       value and ignores trailing content. Observed in MCQ smoke 1×.
    3. **Prompt-echo prefix**: the qgen_question agent's [EXAMPLES]
       block in ``composer_question.go`` dumps the step name +
       description as a literal "example output," and the LLM
       occasionally echoes that header verbatim ahead of the JSON
       body. Observed 2026-05-17 smoke (job 8cfc8b6e-3d9c-…), shape:

           "Step name: qgen_question_evaluation
            Step description: LLM-as-judge: score the candidate …
            scored: {<inner JSON>}"

       We scan for the first ``{`` after the strict parse fails and
       try ``raw_decode`` from that position. The unwrapped inner
       JSON (which is the value of ``scored``) is re-wrapped as
       ``{"scored": <inner>}`` so downstream ``map_agent_response`` sees
       the canonical envelope.

    Falls back to {raw: <text>} on hard parse failure so downstream
    callers see the verbatim emission and can fail loud per
    [[feedback-no-stubs-real-wiring]].
    """
    if not terminal_text:
        return {}
    peeled = strip_markdown_fences(terminal_text).strip()
    if not peeled:
        return {}
    # Try strict parse first (the success path).
    try:
        return json.loads(peeled)
    except (ValueError, TypeError):
        pass
    # Tolerate ONE trailing-extra-char quirk via raw_decode.
    try:
        decoder = json.JSONDecoder()
        value, end = decoder.raw_decode(peeled)
        # If raw_decode consumed any content, accept it.
        if end > 0:
            return value
    except (ValueError, TypeError):
        pass
    # Prompt-echo recovery: scan for first '{' and raw_decode from
    # there. The qgen_question terminal evaluator's [EXAMPLES] block
    # in composer_question.go currently emits a "Step name: <name> /
    # Step description: <desc>" placeholder that the LLM occasionally
    # follows verbatim, prefixing the JSON body. See docstring §3.
    first_brace = peeled.find("{")
    if first_brace > 0:
        try:
            decoder = json.JSONDecoder()
            value, end = decoder.raw_decode(peeled[first_brace:])
            if end > 0:
                # Re-wrap unwrapped `scored` content. Detection: the
                # extracted object has BOTH "candidate" AND at least
                # one of the scoring axes ("factuality" / "clarity" /
                # "difficulty" / "composite"). Matches the qgen_question
                # evaluator schema (composer_question.go::StepEvaluation3
                # outputBlock: `{"scored": {"candidate": ..., factuality, …}}`).
                if (
                    isinstance(value, dict)
                    and "candidate" in value
                    and any(k in value for k in ("factuality", "clarity", "difficulty", "composite"))
                ):
                    return {"scored": value}
                return value
        except (ValueError, TypeError):
            pass
    return {"raw": terminal_text}


def map_agent_response(
    *,
    execution_id: str,
    terminal_text: str,
    agent_role: str,
) -> AgentExecutorResponse:
    """Normalise the terminal agent's JSON output to the qgen_crew
    nodes' expected shape (a flat candidate dict for generate; a
    qualitative critique dict for critique).
    """
    raw = parse_terminal_json(terminal_text)

    if not isinstance(raw, dict):
        raw = {"value": raw}

    # C3 (CR qgen 2026-06-01): capture the per-hop token split from the
    # TOP-LEVEL of the agent's emitted JSON BEFORE the candidate-unwrap
    # branches below reassign `raw` to the inner candidate (which does not
    # carry the sibling token fields). Per the SHARED CONTRACT the agent
    # reports `input_tokens` + `output_tokens` at the top level alongside
    # `scored` / `candidate`; `tokens_consumed_total` (when present) is their
    # authoritative sum. The qgen_crew generate/critique nodes stamp these
    # onto the pipeline_trace rows so the W6 managed-Eval token-compounding
    # criterion reads real numbers instead of 0.
    input_tokens = int(raw.get("input_tokens", 0) or 0)
    output_tokens = int(raw.get("output_tokens", 0) or 0)
    # The concrete model the agent reported for the hop (top-level
    # ``model_id`` — the same wire key the OE evaluator captures as
    # ``grading_model_id``). Captured BEFORE the candidate-unwrap branches
    # reassign ``raw`` to the inner candidate, which does not carry the
    # sibling model field. Rides AgentExecutorResponse → the pipeline_trace
    # rows → the AgentDecisionLog proto field 6 (chora-observability prices
    # the per-hop token counts per model).
    model_id = str(raw.get("model_id") or "").strip()

    # Unwrap qgen_question evaluation output:
    # {"scored": {"candidate": {...}, ...}} → candidate dict (preserve
    # the score fields under a sibling key so audit can pick them up).
    #
    # Bug #4 (surfaced 2026-05-17 follow-on to M14.2 commit 2ddc9867):
    # the LLM occasionally emits `scored.candidate` as a non-dict
    # (string / null / list). `dict(<non-dict>)` raises TypeError or
    # ValueError and crashes the adapter: wrap in {"raw": <value>}
    # envelope so downstream crew sees a flat dict shape and can
    # fail loud per [[feedback-no-stubs-real-wiring]].
    # Set-mode (CHO-1819): the qgen_question agent emits an OBJECT WRAPPER
    # {"candidates": [...], "generation_summary": {...}} for single-pass set
    # generation. Pass it through UNCHANGED so the set graph's generate_set node
    # receives the array + summary (the top-level token split was already
    # captured above). The legacy {"scored": {...}} / {"candidate": {...}}
    # single shapes fall through to the unwraps below: rollout-safe while ONE
    # deployed agent serves both the live single path and the new set path.
    if agent_role in QUESTION_ROLES and isinstance(raw.get("candidates"), list):
        pass  # raw already carries candidates + generation_summary
    elif agent_role in QUESTION_ROLES and "scored" in raw:
        scored = raw["scored"]
        if isinstance(scored, dict) and "candidate" in scored:
            inner = scored["candidate"]
            candidate = dict(inner) if isinstance(inner, dict) else {"raw": inner}
            # Preserve sibling score keys (factuality / clarity /
            # composite / …) verbatim under _scored: values may be
            # numeric, string, or null; downstream decides how to
            # interpret them.
            candidate["_scored"] = {k: v for k, v in scored.items() if k != "candidate"}
            raw = candidate
    # Generation is the TERMINAL sub-agent after the 2026-06-01 trim (assurance
    # + evaluation dropped). Its OutputBlock wraps the candidate under a
    # "candidate" key; the dropped evaluator used to re-emit the inner as
    # scored.candidate (unwrapped above). Without it, unwrap the bare
    # {"candidate": {...}} here: otherwise the candidate reaches the FE
    # double-nested (job.candidate.candidate.*) and the MCQ render is skipped.
    # Mirrors the scored-unwrap; flattens to stem/options/question_type on top.
    elif agent_role in QUESTION_ROLES and isinstance(raw.get("candidate"), dict) and "stem" not in raw:
        raw = raw["candidate"]

    # Total: an explicit agent-emitted total wins (the agent is the authority
    # on its own accounting); otherwise fall back to input+output per the C3
    # SHARED CONTRACT. `raw` may be the flattened candidate now, so check both
    # the (post-unwrap) raw and recompute from the captured split.
    tokens = int(raw.get("tokens_consumed_total", 0) or 0)
    if tokens == 0 and (input_tokens or output_tokens):
        tokens = input_tokens + output_tokens
    cost = int(raw.get("cost_micros_total", 0) or 0)

    return AgentExecutorResponse(
        execution_id=execution_id,
        output_payload=json.dumps(raw),
        tokens_consumed_total=tokens,
        cost_micros_total=cost,
        final_state="EXECUTION_FINAL_STATE_SUCCESS",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_id=model_id,
    )


__all__ = ["MD_FENCE_RE", "map_agent_response", "parse_terminal_json", "strip_markdown_fences"]
