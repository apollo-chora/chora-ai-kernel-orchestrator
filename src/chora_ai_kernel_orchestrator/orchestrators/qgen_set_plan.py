"""Pure helpers for single-pass SET generation (CHO-1819).

No LLM, no LangGraph — these are the LLM-free seams the qgen set-native graph +
runner compose: type-plan construction/validation, the per-type image-cap
backstop, honest GenerationSummary reconciliation, the regenerate dedup-context
block, token-limit chunking, and the FAIL-LOUD agent-output parser. Kept pure so
every one is unit-testable without invoking the model.
"""

from __future__ import annotations

from typing import Any

_KNOWN_TYPES = ("mcq", "oe")


class AgentContractViolation(Exception):  # noqa: N818 — public exception name, raised/caught across the qgen crew
    """The qgen_question agent returned a set that violates the requested plan —
    a malformed wrapper, an off-plan question_type, an over-count, or an illegal
    short set. Raised FAIL-LOUD; the runner turns it into a VALIDATION refusal.
    The generator output is NEVER padded, truncated, or silently coerced.
    """


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_type_plan(typed_quotas: list[Any], *, question_type: str, requested_count: int) -> list[dict[str, Any]]:
    """Normalise the typed type_plan into ``[{question_type, count, max_images}]``.

    EMPTY ⇒ legacy single-type path: ONE quota from content_type +
    requested_count with images off (the W8 ``image_for_*`` booleans drive
    images on the legacy path). NON-EMPTY ⇒ the mixed plan, int-coerced.
    """
    if not typed_quotas:
        return [
            {
                "question_type": question_type or "mcq",
                "count": max(0, _coerce_int(requested_count)),
                "max_images": 0,
            }
        ]
    plan: list[dict[str, Any]] = []
    for q in typed_quotas:
        if isinstance(q, dict):
            qt = str(q.get("question_type") or "")
            count = _coerce_int(q.get("count"))
            max_images = _coerce_int(q.get("max_images"))
            stem = bool(q.get("image_for_stem"))
            answer = bool(q.get("image_for_answer"))
        else:  # tolerate a generated protobuf message object
            qt = str(getattr(q, "question_type", "") or "")
            count = _coerce_int(getattr(q, "count", 0))
            max_images = _coerce_int(getattr(q, "max_images", 0))
            stem = bool(getattr(q, "image_for_stem", False))
            answer = bool(getattr(q, "image_for_answer", False))
        entry: dict[str, Any] = {
            "question_type": qt,
            "count": count,
            "max_images": max_images,
        }
        # CHO-1825 — carry the per-type author image opt-ins into the resolved
        # plan so they reach the Go set-mode composer via type_plan_json. Written
        # ONLY when set (proto3-omit-false parity): an off quota keeps the legacy
        # 3-key shape, so the existing mixed-batch path is byte-for-byte unchanged.
        if stem:
            entry["image_for_stem"] = True
        if answer:
            entry["image_for_answer"] = True
        plan.append(entry)
    return plan


def total_count(type_plan: list[dict[str, Any]]) -> int:
    return sum(_coerce_int(q.get("count")) for q in type_plan)


def image_caps(type_plan: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate per-question-type DISCRETIONARY image budget (sums max_images of
    quotas of the same type). This is the AI-decide budget the regenerate round
    apportions; the render backstop uses :func:`effective_image_caps` (which adds
    the deterministic author-forced allowance on top)."""
    caps: dict[str, int] = {}
    for q in type_plan:
        qt = str(q.get("question_type") or "")
        caps[qt] = caps.get(qt, 0) + _coerce_int(q.get("max_images"))
    return caps


def effective_image_caps(type_plan: list[dict[str, Any]]) -> dict[str, int]:
    """Per-type image cap for the render backstop (CHO-1825) = the discretionary
    AI-decide budget (``max_images``) PLUS the deterministic author-forced
    allowance. When a type sets ``image_for_stem`` / ``image_for_answer``, EVERY
    question of that type MUST carry that image, i.e. ``count`` images per toggled
    part — so the backstop must admit them ON TOP of ``max_images``, else
    :func:`enforce_image_caps` would drop the very images the author forced.
    Aggregates quotas of the same type."""
    caps: dict[str, int] = {}
    for q in type_plan:
        qt = str(q.get("question_type") or "")
        count = _coerce_int(q.get("count"))
        forced_parts = int(bool(q.get("image_for_stem"))) + int(bool(q.get("image_for_answer")))
        caps[qt] = caps.get(qt, 0) + _coerce_int(q.get("max_images")) + count * forced_parts
    return caps


def validate_type_plan(type_plan: list[dict[str, Any]], requested_count: int, max_batch: int = 200) -> list[str]:
    """Defence-in-depth invariant check (the chora-creation domain is
    authoritative; the orchestrator re-checks and fails loud). Returns a list of
    human-readable issues — EMPTY means valid.
    """
    if not type_plan:
        return ["type_plan must contain at least one quota"]
    issues: list[str] = []
    running = 0
    for q in type_plan:
        qt = str(q.get("question_type") or "")
        count = _coerce_int(q.get("count"))
        max_images = _coerce_int(q.get("max_images"))
        if qt not in _KNOWN_TYPES:
            issues.append(f"unknown question_type {qt!r}")
        if count < 1:
            issues.append(f"count must be >= 1 for {qt!r} (got {count})")
        if max_images < 0 or max_images > count:
            issues.append(f"max_images must be 0..count for {qt!r} (got {max_images}, count {count})")
        running += count
    if running > max_batch:
        issues.append(f"sum(count)={running} exceeds max_batch={max_batch}")
    if running != requested_count:
        issues.append(f"sum(count)={running} != requested_count={requested_count}")
    return issues


def chunk_type_plan(type_plan: list[dict[str, Any]], max_per_call: int) -> list[list[dict[str, Any]]]:
    """Split a plan into sub-plans whose per-chunk total count <= max_per_call
    (token-limit mitigation), apportioning each quota's max_images so the
    per-type cap is preserved in aggregate. Question types are preserved. The
    ``enforce_image_caps`` backstop is the authoritative cap; this is best-effort.
    """
    if max_per_call < 1 or total_count(type_plan) <= max_per_call:
        return [type_plan]
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_total = 0
    for q in type_plan:
        qt = str(q.get("question_type") or "")
        remaining = _coerce_int(q.get("count"))
        remaining_images = _coerce_int(q.get("max_images"))
        while remaining > 0:
            room = max_per_call - current_total
            if room <= 0:
                chunks.append(current)
                current = []
                current_total = 0
                room = max_per_call
            take = min(remaining, room)
            take_images = min(remaining_images, take)
            entry: dict[str, Any] = {
                "question_type": qt,
                "count": take,
                "max_images": take_images,
            }
            # CHO-1825's lesson recurs at this seam (ADR-251 D1): the
            # deterministic author-forced image flags force ONE image per
            # question of the type, so every CHUNK plan must carry them or
            # effective_image_caps computes a zero forced allowance for later
            # chunks and enforce_image_caps drops the very images the author
            # forced.
            if q.get("image_for_stem"):
                entry["image_for_stem"] = True
            if q.get("image_for_answer"):
                entry["image_for_answer"] = True
            current.append(entry)
            remaining -= take
            remaining_images -= take_images
            current_total += take
    if current:
        chunks.append(current)
    return chunks


def enforce_image_caps(candidates: list[dict[str, Any]], caps: dict[str, int]) -> tuple[list[dict[str, Any]], int]:
    """Backstop: drop image_specs beyond each type's cap (deterministic — keep
    by array order, prefer placement=='stem' over 'answer' on ties within a
    candidate). Returns ``(candidates, dropped_count)`` over shallow copies; the
    inputs are not mutated. A non-zero dropped_count is an agent contract
    violation the render node logs LOUD (it still trims defensively).
    """
    used: dict[str, int] = {}
    dropped = 0
    out: list[dict[str, Any]] = []
    for cand in candidates:
        qt = str(cand.get("question_type") or "")
        specs = list(cand.get("image_specs") or [])
        if not specs:
            out.append(dict(cand))
            continue
        cap = caps.get(qt, 0)
        ordered = sorted(specs, key=lambda s: 0 if str(s.get("placement")) == "stem" else 1)
        kept: list[Any] = []
        for spec in ordered:
            if used.get(qt, 0) < cap:
                kept.append(spec)
                used[qt] = used.get(qt, 0) + 1
            else:
                dropped += 1
        new_cand = dict(cand)
        new_cand["image_specs"] = kept
        out.append(new_cand)
    return out, dropped


def compute_generation_summary(
    type_plan: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    *,
    grounding_mode: str,
    model_shortfall_reason: str,
) -> dict[str, Any]:
    """Reconcile requested (from the plan) vs generated (the accepted set) into
    the honest GenerationSummary. ``shortfall_reason`` is carried ONLY for a
    legal strict shortfall (generated < requested under strict grounding).
    """
    requested_per_type: dict[str, int] = {}
    for q in type_plan:
        qt = str(q.get("question_type") or "")
        requested_per_type[qt] = requested_per_type.get(qt, 0) + _coerce_int(q.get("count"))
    counted: dict[str, int] = {}
    for cand in accepted:
        qt = str(cand.get("question_type") or "")
        counted[qt] = counted.get(qt, 0) + 1
    # FLAT map<question_type,int> matching the typed proto
    # chora.creation.v1.GenerationSummary.generated_per_type (the per-type
    # REQUESTED count is derivable from type_plan, so the proto does not carry
    # it). All requested types appear (a strict shortfall that produced zero of
    # a type surfaces it as 0 rather than dropping the key).
    generated_per_type = {qt: counted.get(qt, 0) for qt in sorted(set(requested_per_type) | set(counted))}
    requested_total = total_count(type_plan)
    generated_total = len(accepted)
    shortfall = ""
    if generated_total < requested_total and grounding_mode == "strict":
        shortfall = model_shortfall_reason or (
            f"The uploaded source supported {generated_total} of {requested_total} requested questions."
        )
    return {
        "requested_total": requested_total,
        "generated_total": generated_total,
        "generated_per_type": generated_per_type,
        "shortfall_reason": shortfall,
    }


def dedup_context_block(accepted_stems: list[str], rejected_notes: list[str]) -> str:
    """The regenerate-rejected prompt extension: which concepts are already
    covered (do not repeat) + why the prior attempts were rejected. Empty when
    there is nothing to say.
    """
    parts: list[str] = []
    covered = "; ".join(s for s in accepted_stems if s)
    if covered:
        parts.append("Already covered — do NOT repeat these concepts: " + covered)
    notes = "; ".join(n for n in rejected_notes if n)
    if notes:
        parts.append("Previous attempts were rejected because: " + notes)
    return "\n".join(parts)


def parse_set_response(
    raw: dict[str, Any],
    type_plan: list[dict[str, Any]],
    *,
    grounding_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """FAIL-LOUD parse of the qgen_question set output. Returns
    ``(candidates, model_summary)``. Raises ``AgentContractViolation`` on a
    malformed wrapper, an off-plan question_type, an over-count, or an illegal
    short set (generated < requested when NOT a strict shortfall). Tolerates the
    legacy single-candidate shapes during the ordered rollout by wrapping them
    as a 1-element set.
    """
    if not isinstance(raw, dict):
        raise AgentContractViolation(f"agent output is not an object: {type(raw)!r}")

    candidates = raw.get("candidates")
    model_summary: dict[str, Any] = raw.get("generation_summary") or {}

    if not isinstance(candidates, list):
        # Rollout tolerance — a not-yet-redeployed agent emits the single shape.
        if isinstance(raw.get("candidate"), dict):
            candidates = [raw["candidate"]]
        elif "stem" in raw:
            candidates = [raw]
        else:
            raise AgentContractViolation(
                "agent output has no 'candidates' array and is not a legacy single-candidate shape"
            )
        model_summary = {}

    requested_per_type: dict[str, int] = {}
    for q in type_plan:
        qt = str(q.get("question_type") or "")
        requested_per_type[qt] = requested_per_type.get(qt, 0) + _coerce_int(q.get("count"))
    plan_types = set(requested_per_type)

    seen_per_type: dict[str, int] = {}
    for cand in candidates:
        if not isinstance(cand, dict):
            raise AgentContractViolation("candidate is not an object")
        qt = str(cand.get("question_type") or "")
        if qt not in plan_types:
            raise AgentContractViolation(f"candidate question_type {qt!r} not in requested plan {sorted(plan_types)}")
        seen_per_type[qt] = seen_per_type.get(qt, 0) + 1

    for qt, seen in seen_per_type.items():
        if seen > requested_per_type.get(qt, 0):
            raise AgentContractViolation(f"generated {seen} {qt!r} exceeds requested {requested_per_type.get(qt, 0)}")

    requested_total = total_count(type_plan)
    if len(candidates) < requested_total:
        shortfall_reason = str(model_summary.get("shortfall_reason") or "")
        if not (grounding_mode == "strict" and shortfall_reason):
            raise AgentContractViolation(
                f"generated {len(candidates)} < requested {requested_total} but "
                "this is not a legal strict shortfall (requires strict grounding "
                "+ a non-empty shortfall_reason) — refusing to pad"
            )
    return candidates, model_summary


def parse_critique_set_response(
    raw: Any,
    expected_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """FAIL-LOUD parse of the qgen_critic set output (CHO-2397, ADR-251 D2).

    Returns ``{candidate_id: {"accepted", "critique_notes",
    "suggested_revisions"}}`` covering EXACTLY ``expected_ids``. Raises
    ``AgentContractViolation`` on a malformed wrapper, a verdict that is not
    an object, an unknown candidate_id, a duplicate candidate_id, or a
    missing candidate_id, always naming the offending id(s).

    The ID correlation is deliberately strict with NO tolerance and NO
    per-candidate fallback: a mis-correlated verdict silently swaps an
    accept and a reject, which is the failure mode this parse exists to
    prevent (owner-ruled). Verdict FIELD coercion mirrors the single-path
    parse (bool/str/list), so only the correlation is stricter.
    """
    if not isinstance(raw, dict):
        raise AgentContractViolation(f"critique set output is not an object: {type(raw)!r}")
    verdicts = raw.get("verdicts")
    if not isinstance(verdicts, list):
        raise AgentContractViolation("critique set output has no 'verdicts' array")

    expected = set(expected_ids)
    by_id: dict[str, dict[str, Any]] = {}
    for entry in verdicts:
        if not isinstance(entry, dict):
            raise AgentContractViolation(f"verdict entry is not an object: {entry!r}")
        cid = str(entry.get("candidate_id") or "")
        if cid not in expected:
            raise AgentContractViolation(f"verdict for unknown candidate_id {cid!r} (sent {sorted(expected)})")
        if cid in by_id:
            raise AgentContractViolation(f"duplicate verdict for candidate_id {cid!r}")
        by_id[cid] = {
            "accepted": bool(entry.get("accepted", False)),
            "critique_notes": str(entry.get("critique_notes") or ""),
            "suggested_revisions": list(entry.get("suggested_revisions") or []),
        }

    missing = expected - set(by_id)
    if missing:
        raise AgentContractViolation(
            f"no verdict for candidate_id(s) {sorted(missing)} ({len(by_id)} of {len(expected)} echoed)"
        )
    return by_id
