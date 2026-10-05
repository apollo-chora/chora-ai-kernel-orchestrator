"""Test-set composition helpers (CHO-1703 W2, ADR-180 D4/D5; ADR-254 D2).

The composer's ONE LLM call now rides the ``qgen_generate`` dispatch lane as
``mode=compose`` (``orchestrators/qgen_crew.compose_test_set_node``): the
chora-qgen-question agent reads the accepted candidates + the source files by
reference and answers ``{"proposed_test_set": {...}}``. What stays here is
the pure, deterministic half the node needs on both the happy and the
degraded path:

    {"title": str<=256, "description": str<=2048,
     "order": [draft_id, ...], "points": {draft_id: int 1..100}}

* :func:`fallback_proposal`: the DETERMINISTIC degraded proposal (D4): title
  from the job/source convention, candidates in submission order, uniform
  :data:`DEFAULT_POINTS`. The batch must never fail because composing failed.
* :func:`normalise_proposal`: repair the agent's proposal into a contract-valid
  ProposedTestSet (order = a permutation of the draft ids, points clamped,
  title/description clamped with the fallback title when blank).
* :func:`parse_completion_json`: tolerant JSON recovery for a raw completion
  (fences, prose around the object) for callers that still hold raw text.

The kennel no longer holds a model-gateway client for this: the last direct
Invoke went with the qgen flip (ADR-254 D2/D12), and ``MODEL_GATEWAY_GRPC_TARGET``
has no kennel reader. ``QGEN_TESTSET_COMPOSE_ENABLED`` remains the switch
(read by the qgen wiring, threaded into the set-lane state as
``compose_enabled``).
"""

from __future__ import annotations

import json
import re
from typing import Any

from chora_ai_kernel_orchestrator.orchestrators.qgen_grounding import (
    split_source_files,
)

# Uniform per-question points fallback (D5): mirrors the test-set editor's
# default (delivery test_set_questions.points default 10).
DEFAULT_POINTS = 10

# Contract clamps per chora-contracts/openapi/creation-questions.yaml
# ProposedTestSet.
TITLE_MAX_CHARS = 256
DESCRIPTION_MAX_CHARS = 2048
POINTS_MIN = 1
POINTS_MAX = 100

# Env knob (per [[secrets-and-env]]; read by the qgen wiring).
ENV_TESTSET_COMPOSE_ENABLED = "QGEN_TESTSET_COMPOSE_ENABLED"


def fallback_proposal(
    *,
    payload: Any,
    candidates: list[Any],
    files: list[dict[str, str]],
) -> dict[str, Any]:
    """The DETERMINISTIC degraded proposal (D4): title from the job/source
    convention, candidates in submission order, uniform default points.
    Pure + total — never raises on any candidate shape."""
    draft_ids = _draft_ids_of(candidates)
    sources, _rubric = split_source_files(files or [])
    title = ""
    if sources:
        base = sources[0]["blob_uri"].rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0].strip()
        if stem:
            title = f"Test set — {stem}"
    if not title:
        prompt_snippet = str(getattr(payload, "prompt", "") or "").strip()
        title = f"Test set — {prompt_snippet[:64].strip()}" if prompt_snippet else "Generated test set"
    description = (
        f"Auto-proposed from {len(draft_ids)} generated question(s)"
        + (f" grounded on {sources[0]['blob_uri'].rsplit('/', 1)[-1]}" if sources else "")
        + "."
    )
    return {
        "title": title[:TITLE_MAX_CHARS],
        "description": description[:DESCRIPTION_MAX_CHARS],
        "order": draft_ids,
        "points": {d: DEFAULT_POINTS for d in draft_ids},
    }


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def parse_completion_json(completion: str) -> dict[str, Any]:
    """Parse the model completion into a dict. Tolerates ```json fences and
    leading/trailing prose (first ``{`` .. last ``}``). Raises ValueError
    when no JSON object can be recovered — caller falls back."""
    text = (completion or "").strip()
    if not text:
        raise ValueError("empty completion")
    for candidate_text in _candidate_json_texts(text):
        try:
            parsed = json.loads(candidate_text)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("completion carried no JSON object")


def _candidate_json_texts(text: str) -> list[str]:
    out = [text]
    m = _FENCE_RE.search(text)
    if m:
        out.append(m.group(1))
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        out.append(text[first : last + 1])
    return out


def normalise_proposal(
    parsed: dict[str, Any],
    *,
    candidates: list[Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Repair the model's proposal into a contract-valid ProposedTestSet:
    order must be a permutation of the candidate draft_ids (unknowns dropped,
    dupes dropped, missing appended in submission order); points clamped to
    1..100 with the uniform default for missing/invalid; title/description
    clamped with the deterministic fallback title when blank."""
    draft_ids = _draft_ids_of(candidates)
    known = set(draft_ids)

    title = str(parsed.get("title") or "").strip()[:TITLE_MAX_CHARS]
    if not title:
        title = fallback["title"]
    description = str(parsed.get("description") or "").strip()[:DESCRIPTION_MAX_CHARS]
    if not description:
        description = fallback["description"]

    order: list[str] = []
    raw_order = parsed.get("order")
    if isinstance(raw_order, list):
        for entry in raw_order:
            did = str(entry or "").strip()
            if did in known and did not in order:
                order.append(did)
    for did in draft_ids:
        if did not in order:
            order.append(did)

    points: dict[str, int] = {}
    raw_points = parsed.get("points")
    raw_points = raw_points if isinstance(raw_points, dict) else {}
    for did in draft_ids:
        points[did] = _coerce_points(raw_points.get(did))

    return {
        "title": title,
        "description": description,
        "order": order,
        "points": points,
    }


def _coerce_points(value: Any) -> int:
    if isinstance(value, bool):
        return DEFAULT_POINTS
    try:
        pts = int(value)
    except (TypeError, ValueError):
        return DEFAULT_POINTS
    if pts < POINTS_MIN:
        return POINTS_MIN
    if pts > POINTS_MAX:
        return POINTS_MAX
    return pts


def _draft_ids_of(candidates: list[Any]) -> list[str]:
    out: list[str] = []
    for c in candidates or []:
        if isinstance(c, dict):
            did = str(c.get("draft_id") or "").strip()
            if did:
                out.append(did)
    return out


# Backwards-compatible private aliases (tests + older callers).
_normalise_proposal = normalise_proposal
_parse_completion_json = parse_completion_json

__all__ = [
    "DEFAULT_POINTS",
    "DESCRIPTION_MAX_CHARS",
    "ENV_TESTSET_COMPOSE_ENABLED",
    "POINTS_MAX",
    "POINTS_MIN",
    "TITLE_MAX_CHARS",
    "fallback_proposal",
    "normalise_proposal",
    "parse_completion_json",
]
