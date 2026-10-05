"""Pure Growth-Edge review-panel builder (CHO-1973 Wave A, ADR-205 D4/D5).

The panel is the CANONICAL UX shape the A+ FE renders at the bounded HITL review,
and the exact body the ``chora.consumption.weakness.review_pending.v1`` event
carries. It is built from the checkpoint ``candidate_edges`` so it is fully
deterministic and ROUND-TRIPS: the ``proposed_edge_id`` minted here resolves back
to the SAME candidate edge on resume — the orchestrator alone holds the
checkpoint candidate_edges, so it alone owns the ``proposed_edge_id → edge``
mapping (the inverse lives in the graph's hitl_review node).

NO I/O / clock / network here — a pure transform, trivially unit-tested. The
caller (the crew runner) resolves the learner's Familiar + the metered output
prices and passes them in; this module never fabricates either.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

# The four learner-selectable outputs (ADR-205 D5), in canonical panel order.
FREE_OUTPUT_KINDS: tuple[str, ...] = ("focused_dose", "familiar_coaching")
METERED_OUTPUT_KINDS: tuple[str, ...] = ("practice_test", "study_aids")
OUTPUT_KINDS: tuple[str, ...] = (*FREE_OUTPUT_KINDS, *METERED_OUTPUT_KINDS)

# Cap the bounded "add a struggle the analyser missed" picker (ADR-205 D4) so it
# stays a short, reviewable shortlist — a flood of low-signal concepts defeats
# the point of a BOUNDED control. The highest-confidence concepts win the cap.
MAX_CANDIDATE_STRUGGLES = 6

# proposed_edge_id prefix — opaque-but-deterministic id minted per candidate edge.
_PROPOSED_EDGE_ID_PREFIX = "pe-"

# Suggested-difficulty cut-points on the shakiness ``strength`` (0..1, 1 = weak):
# a near-mastered edge wants easier practice; a very weak one wants harder.
_DIFFICULTY_EASIER_MAX = 0.4
_DIFFICULTY_STANDARD_MAX = 0.7


def proposed_edge_id_for_index(index: int) -> str:
    """Mint the stable ``proposed_edge_id`` for a candidate edge at ``index``.

    Index-based (``pe-{i}``) because the checkpoint preserves ``candidate_edges``
    order, so this id round-trips to the identical edge on resume with NO stored
    map. (Stable WITHIN one review cycle; a reiterate re-diagnoses and re-mints.)
    """
    return f"{_PROPOSED_EDGE_ID_PREFIX}{index}"


def index_from_proposed_edge_id(proposed_edge_id: str | None) -> int | None:
    """Inverse of :func:`proposed_edge_id_for_index`. Returns None for any id that
    is not a canonical ``pe-{non-negative-int}`` (so an unknown/garbled id from a
    resume body is rejected loudly by the caller rather than silently mapped)."""
    if not proposed_edge_id or not proposed_edge_id.startswith(_PROPOSED_EDGE_ID_PREFIX):
        return None
    tail = proposed_edge_id[len(_PROPOSED_EDGE_ID_PREFIX) :]
    if not tail.isdigit():  # rejects "", "x", "-1", " 1", leading "+"
        return None
    return int(tail)


def suggested_difficulty_for_strength(strength: float) -> str:
    """Map shakiness ``strength`` → a bounded practice-difficulty suggestion."""
    if strength <= _DIFFICULTY_EASIER_MAX:
        return "easier"
    if strength <= _DIFFICULTY_STANDARD_MAX:
        return "standard"
    return "harder"


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _descriptor(edge: Mapping[str, Any]) -> dict[str, Any]:
    """Parse an edge's ``descriptor_json`` fail-soft (our own graph serialised it;
    a malformed value loses summary/angles but never sinks the panel)."""
    raw = edge.get("descriptor_json")
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if str(x).strip()]


def _proposed_edge(index: int, edge: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = _descriptor(edge)
    strength = _coerce_float(edge.get("strength"))
    return {
        "proposed_edge_id": proposed_edge_id_for_index(index),
        "concept_label": str(edge.get("concept_label", "")),
        "summary": str(descriptor.get("summary", "")),
        "suggested_angles": _str_list(descriptor.get("suggested_angles")),
        "strength": strength,
        "suggested_difficulty": suggested_difficulty_for_strength(strength),
    }


def _candidate_struggle(raw: Any) -> dict[str, str] | None:
    """Clean one retained below-threshold concept into the proto ``CandidateStruggle``
    panel shape ``{concept_key, concept_label}``. Returns None (the caller logs +
    skips) when it lacks a usable key OR label — never fabricate one."""
    if not isinstance(raw, Mapping):
        return None
    key = str(raw.get("concept_key", "")).strip()
    label = str(raw.get("concept_label", "")).strip()
    if not key or not label:
        return None
    return {"concept_key": key, "concept_label": label}


def _build_candidate_struggles(retained: Any, *, edge_keys: set[str]) -> list[dict[str, str]]:
    """Build the bounded candidate-struggle shortlist from the retained
    below-threshold concepts (CHO-1973 Q2 / ADR-205 D4).

    Rank by descending diagnosis confidence, then: skip any concept missing a
    usable key/label (fail-loud log + skip, never fabricate), de-dup by
    concept_key against BOTH the proposed edges (``edge_keys`` — never re-offer a
    concept already proposed as an edge) and the list itself, and cap to
    ``MAX_CANDIDATE_STRUGGLES``. Dedup runs BEFORE the cap so the shortlist is
    full whenever enough genuine non-edge struggles exist. Each entry is the wire
    shape ``{concept_key, concept_label}`` (the ranking confidence is stripped)."""
    if not retained:
        return []
    ordered = sorted(retained, key=_struggle_rank, reverse=True)
    out: list[dict[str, str]] = []
    seen: set[str] = set(edge_keys)  # seed: a proposed-edge key is already taken
    for raw in ordered:
        cs = _candidate_struggle(raw)
        if cs is None:
            logger.warning(
                "weakness_panel.candidate_struggle_skipped: retained concept lacks "
                "a usable key/label — skipping (never fabricated)",
                extra={"raw": raw},
            )
            continue
        if cs["concept_key"] in seen:
            continue
        seen.add(cs["concept_key"])
        out.append(cs)
        if len(out) >= MAX_CANDIDATE_STRUGGLES:
            break
    return out


def _struggle_rank(raw: Any) -> float:
    return _coerce_float(raw.get("confidence")) if isinstance(raw, Mapping) else 0.0


def _available_output(
    kind: str, *, requested_outputs: Mapping[str, Any], output_prices: Mapping[str, int]
) -> dict[str, Any]:
    mana_price = 0 if kind in FREE_OUTPUT_KINDS else int(output_prices.get(kind, 0) or 0)
    # nil requested_outputs ⇒ "core edges + dose only" (uploaded proto default):
    # focused_dose pre-selected, the rest off, unless the upload set them.
    default_selected = bool(requested_outputs.get(kind, kind == "focused_dose"))
    return {"kind": kind, "mana_price": mana_price, "default_selected": default_selected}


def build_review_panel(
    *,
    candidate_edges: list[dict[str, Any]],
    upload_id: str,
    tenant_id: str,
    learner_gcid: str,
    requested_outputs: Mapping[str, Any] | None,
    output_prices: Mapping[str, int] | None,
    familiar: Mapping[str, Any] | None = None,
    candidate_struggles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the bounded HITL review panel from the checkpoint candidate edges.

    ``candidate_edges`` are the body-dicts the graph stored in state (concept_label
    / concept_key / category / tags / confidence / strength / descriptor_json).
    ``output_prices`` maps a metered output kind → mana price (free kinds are
    always 0); an absent/0 price encodes 0 (the caller logs the unconfigured
    case — this pure builder never fabricates a price). ``familiar`` defaults to
    empty when the caller cannot resolve it. ``candidate_struggles`` is the RAW
    retained below-threshold list (each ``{concept_key, concept_label, confidence}``);
    this builder ranks, de-dups (against the proposed edges + itself), caps, and
    cleans it into the bounded ``{concept_key, concept_label}`` picker shortlist.
    """
    req = requested_outputs or {}
    prices = output_prices or {}
    edge_keys = {
        str(e.get("concept_key", "")).strip() for e in candidate_edges if str(e.get("concept_key", "")).strip()
    }
    return {
        "upload_id": upload_id,
        "tenant_id": tenant_id,
        "learner_gcid": learner_gcid,
        "familiar": dict(familiar) if familiar else {},
        "proposed_edges": [_proposed_edge(i, edge) for i, edge in enumerate(candidate_edges)],
        "candidate_struggles": _build_candidate_struggles(candidate_struggles, edge_keys=edge_keys),
        "available_outputs": [
            _available_output(kind, requested_outputs=req, output_prices=prices) for kind in OUTPUT_KINDS
        ],
    }


__all__ = [
    "FREE_OUTPUT_KINDS",
    "MAX_CANDIDATE_STRUGGLES",
    "METERED_OUTPUT_KINDS",
    "OUTPUT_KINDS",
    "build_review_panel",
    "index_from_proposed_edge_id",
    "proposed_edge_id_for_index",
    "suggested_difficulty_for_strength",
]
