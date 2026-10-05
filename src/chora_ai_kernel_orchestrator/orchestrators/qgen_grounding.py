"""Lane 1c multi-file + rubric grounding helpers (CHO-1703 W2, ADR-180).

Pure functions shared by the qgen batch path:

* ``normalise_source_files`` — defensive wire→typed parse of the
  ``AiAssistStarted.source_files`` (f20) list decoded by
  ``adapter/pubsub/proto_wire.py``.
* ``effective_source_files`` — the W0 back-compat resolution: prefer
  ``source_files[]`` when non-empty, else synthesize a single role="source"
  entry from the f17/18 mirror (``source_blob_uri`` / ``source_mime_type``);
  ``[]`` when the job is ungrounded.
* ``split_source_files`` — partition into question-material sources and the
  ≤1 mark-scheme rubric file (D6).
* ``grounding_prompt_block`` — the generation-prompt extension for grounded
  BATCH jobs: lists the grounding corpus, demands per-candidate
  MODEL-REPORTED citations ``{source_file, page, excerpt}`` (D9 — the
  orchestrator never verifies; chora-creation's completed-event subscriber
  stamps ``verified``/``chunk_id`` per D15), and — when a rubric file rides
  the job — explicit mark-scheme alignment instructions (OE rubric criterion
  weights + marks cues like "[5 marks]" / "Question 3 (10 points)" per D5/D6).

Kept free of LangGraph / adapter imports so the helpers stay unit-testable
and reusable by both ``generate_node`` (prompt side) and the
``TestSetComposer`` (composer side).
"""

from __future__ import annotations

import uuid
from typing import Any

# Canonical SourceFileRef.role values per chora-contracts
# proto/events/creation/ai_assist.proto §SourceFileRef.
ROLE_SOURCE = "source"
ROLE_RUBRIC = "rubric"

# Verbatim-excerpt cap per chora-contracts/openapi/creation-questions.yaml
# §QuestionCitation.excerpt maxLength.
CITATION_EXCERPT_MAX_CHARS = 1024

# Stable uuid5 namespace for batch-candidate draft ids. The composer's
# proposed_test_set keys (order[] / points{}) reference these ids, so they
# MUST be deterministic per (assist_id, index): a checkpoint-replayed batch
# re-publish mints IDENTICAL ids and creation-side idempotent re-processing
# maps the same candidates (D10 idempotency posture).
_DRAFT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "chora:qgen:batch-draft")


def normalise_source_files(raw: Any) -> list[dict[str, str]]:
    """Parse a decoded ``source_files`` value into a clean, ordered list of
    ``{"blob_uri", "mime_type", "role"}`` dicts.

    Defensive per the decoder contract: non-list input → ``[]``; non-dict
    entries and entries without a usable ``blob_uri`` are dropped; ``role``
    is lower-cased/stripped and anything other than ``rubric`` collapses to
    ``source`` (unknown future roles must never silently become mark
    schemes).
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        blob_uri = str(entry.get("blob_uri") or "").strip()
        if not blob_uri:
            continue
        role = str(entry.get("role") or "").strip().lower()
        if role != ROLE_RUBRIC:
            role = ROLE_SOURCE
        out.append(
            {
                "blob_uri": blob_uri,
                "mime_type": str(entry.get("mime_type") or "").strip(),
                "role": role,
            }
        )
    return out


def effective_source_files(
    *,
    source_files: Any,
    source_blob_uri: str,
    source_mime_type: str,
) -> list[dict[str, str]]:
    """Resolve the job's EFFECTIVE grounding-file list per the W0 contract:
    ``source_files`` (f20) is canonical when non-empty; the f17/18 scalar
    mirror is the single-file fallback for events published by pre-1c
    publishers; ``[]`` when the job carries no grounding material at all.
    """
    files = normalise_source_files(source_files)
    if files:
        return files
    uri = (source_blob_uri or "").strip()
    if uri:
        return [
            {
                "blob_uri": uri,
                "mime_type": (source_mime_type or "").strip(),
                "role": ROLE_SOURCE,
            }
        ]
    return []


def split_source_files(
    files: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    """Partition ``files`` into ``(sources, rubric)``.

    The upload handler caps jobs at ≤1 role="rubric" file; defensively the
    FIRST rubric wins and any extra rubric entries are DROPPED entirely — a
    mark scheme must never masquerade as question source material.
    """
    sources: list[dict[str, str]] = []
    rubric: dict[str, str] | None = None
    for f in files:
        if f.get("role") == ROLE_RUBRIC:
            if rubric is None:
                rubric = f
            continue
        sources.append(f)
    return sources, rubric


def grounding_prompt_block(
    *,
    files: list[dict[str, str]],
    grounding_mode: str,
    question_type: str,
) -> str:
    """Build the grounding + citations instruction block appended to the
    author prompt for grounded BATCH generation runs.

    Returns ``""`` when ``files`` is empty so ungrounded jobs keep their
    prompt byte-for-byte unchanged. The block is purely additive text — the
    qgen_question agent's instruction template renders the author prompt
    verbatim, so these instructions reach the model without any agent-side
    change.
    """
    if not files:
        return ""

    sources, rubric = split_source_files(files)

    lines: list[str] = ["", "", "## [GROUNDING MATERIAL]"]
    if sources:
        lines.append(
            "The following uploaded file(s) are the question source material "
            "(the primary file is attached to this request; refer to every "
            "listed file by its exact URI):"
        )
        for i, f in enumerate(sources, start=1):
            mime = f" ({f['mime_type']})" if f.get("mime_type") else ""
            lines.append(f"  {i}. {f['blob_uri']}{mime}")
    mode = (grounding_mode or "").strip().lower()
    if mode == "strict":
        lines.append(
            "Grounding mode STRICT: generate ONLY from facts stated in the "
            "source material — do not introduce outside facts."
        )
    elif sources:
        lines.append(
            "Grounding mode starting-point: use the source material as the seed and springboard for the questions."
        )

    lines += [
        "",
        "## [CITATIONS — REQUIRED]",
        'Your candidate JSON MUST include a top-level "citations" array on '
        "the candidate object — one entry per source passage that grounds "
        "this question:",
        '  "citations": [{"source_file": "<exact gs:// URI from the list '
        'above>", "page": <1-based page number, or null when the file has '
        'no pages>, "excerpt": "<short VERBATIM excerpt copied from the '
        f'source, at most {CITATION_EXCERPT_MAX_CHARS} characters>"}}]',
        "Quote excerpts verbatim — they are matched against the source text "
        "downstream; paraphrased excerpts will be flagged as unverified.",
    ]

    if rubric is not None:
        lines += [
            "",
            "## [MARK SCHEME / RUBRIC]",
            f"A mark scheme (grading rubric) file rides this job: "
            f"{rubric['blob_uri']}" + (f" ({rubric['mime_type']})" if rubric.get("mime_type") else "") + ".",
            "Align every question to it:",
            '  - Extract marks cues such as "[5 marks]" or "Question 3 '
            "(10 points)\" and match each question's depth and difficulty "
            "to the marks on offer.",
        ]
        if (question_type or "").strip().lower() == "oe":
            lines.append(
                "  - For open-ended questions, align the rubric criteria AND "
                "their weight values to the mark scheme's allocation (a "
                "criterion worth more marks carries proportionally more "
                "weight)."
            )

    return "\n".join(lines)


def deterministic_draft_id(assist_id: str, index: int) -> str:
    """UUIDv5 draft id for the batch candidate at ``index`` of ``assist_id``.

    Deterministic so (a) the composer's ``proposed_test_set.order``/``points``
    keys always match the ``candidates[].draft_id`` they were minted with and
    (b) a checkpoint-replayed terminal re-publish carries identical ids
    (creation's idempotent subscriber maps the same candidates).
    """
    return str(uuid.uuid5(_DRAFT_ID_NAMESPACE, f"{assist_id}:{index}"))


def _core_candidate(obj: dict[str, Any]) -> dict[str, Any]:
    """Locate the candidate dict the model's citations ride on. The qgen
    agent may emit the candidate flat, wrapped as ``{"candidate": {...}}``,
    or evaluator-wrapped as ``{"scored": {"candidate": {...}}}``.
    """
    inner = obj.get("candidate")
    if isinstance(inner, dict):
        return inner
    scored = obj.get("scored")
    if isinstance(scored, dict):
        inner = scored.get("candidate")
        if isinstance(inner, dict):
            return inner
    return obj


def normalise_candidate_citations(
    candidate: Any,
    files: list[dict[str, str]],
) -> Any:
    """Make a batch candidate's MODEL-REPORTED ``citations`` safe + canonical
    (D9). NEVER verifies — chora-creation's completed-event subscriber stamps
    ``verified``/``chunk_id`` against the chunk store (D15).

    Grounded jobs (``files`` non-empty):

    * non-list ``citations`` value → key removed
    * entries kept only when dicts with a non-blank ``source_file`` or
      ``excerpt``; excerpt clamped to :data:`CITATION_EXCERPT_MAX_CHARS`
    * bare-filename ``source_file`` mentions resolve to the job's matching
      gs:// ``blob_uri`` (unknown mentions are LEFT AS-IS — creation flags
      them unverified; never silently dropped)
    * ``page`` coerced to a 1-based int, else the key is dropped (nullable)
    * absent ``citations`` stays absent — best-effort model-reported, never
      fabricated

    Ungrounded jobs (``files`` empty): a hallucinated ``citations`` key is
    stripped entirely (the contract says no citations key for ungrounded
    jobs).

    Non-dict candidates pass through untouched (defensive — the batch array
    builder already minimum-shapes them).
    """
    if not isinstance(candidate, dict):
        return candidate
    core = _core_candidate(candidate)

    if not files:
        core.pop("citations", None)
        return candidate

    raw = core.get("citations")
    if raw is None:
        return candidate
    if not isinstance(raw, list):
        core.pop("citations", None)
        return candidate

    by_uri = {f["blob_uri"]: f for f in files}
    by_basename: dict[str, list[str]] = {}
    for f in files:
        base = f["blob_uri"].rsplit("/", 1)[-1].strip().lower()
        if base:
            by_basename.setdefault(base, []).append(f["blob_uri"])

    cleaned: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        source_file = str(entry.get("source_file") or "").strip()
        excerpt = str(entry.get("excerpt") or "").strip()
        if not source_file and not excerpt:
            continue
        # Resolve bare-filename mentions to the job's blob_uri when
        # unambiguous; exact URIs and unknown mentions pass through.
        if source_file and source_file not in by_uri:
            matches = by_basename.get(source_file.strip().lower(), [])
            if len(matches) == 1:
                source_file = matches[0]
        out: dict[str, Any] = {
            "source_file": source_file,
            "excerpt": excerpt[:CITATION_EXCERPT_MAX_CHARS],
        }
        page = _coerce_page(entry.get("page"))
        if page is not None:
            out["page"] = page
        cleaned.append(out)
    core["citations"] = cleaned
    return candidate


def _coerce_page(value: Any) -> int | None:
    """1-based page int, or None when absent/invalid (schema: nullable)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page >= 1 else None


__all__ = [
    "CITATION_EXCERPT_MAX_CHARS",
    "ROLE_RUBRIC",
    "ROLE_SOURCE",
    "deterministic_draft_id",
    "effective_source_files",
    "grounding_prompt_block",
    "normalise_candidate_citations",
    "normalise_source_files",
    "split_source_files",
]
