"""Python port of consumption's ``proseNarrative`` predicate.

Source of truth for the BEHAVIOUR is the Go original (consumption's
``func proseNarrative``). Source of truth for the CASES is the shared
vector file, vendored into this repo at
``testdata/prose_narrative_vectors.json``.

The file is read by BOTH sides, so adding a vector turns RED on whichever
implementation does not handle it. That was NOT true when this port was written:
until ``TestProseNarrative_SharedVectors`` landed in
``companion_handlers_narrative_test.go`` (2026-08-23) the Go test declared its
cases inline and read no external file, so a change to the Go original turned
nothing red and the "shared vectors" claim here was false. It is now real rather
than believed, which is the only reason this docstring may assert it.

The extras in that file (bare scalars, a quoted string, the fence-tag rules,
NaN/Infinity) pin semantics the Go ORIGINAL had no test for, and the original is
what renders to a learner. NaN/Infinity in particular pin the ONE place the two
languages genuinely diverge: ``json.loads`` accepts them as a Python extension
and Go's ``json.Valid`` does not, which is what ``_reject_constant`` below
closes.

Why a port exists at all, since a second implementation of a predicate is a
second thing that can drift: the kennel dose lane must decide OK vs FAILED
before it publishes, and that decision cannot be delegated across a process
boundary to a Go function in another service. Consumption drops a non-prose
narrative server side, so a kennel that answered OK on a JSON-shaped rationale
would hand the learner a silently empty dose. The drift risk is real and is
mitigated by the shared vectors, not by wishing.

⚠ THE FENCE UNWRAP IS NOT OPTIONAL. A port that only rejects blank / leading
brace / leading bracket / ``json.Valid`` still passes every naive vector while
accepting ```` ```json {...}``` ```` at a learner, because the fence hides the
brace from a prefix check. The original unwraps first and then tests.
"""

from __future__ import annotations

import json

__all__ = ["prose_narrative"]


def _reject_constant(_name: str) -> None:
    """json.loads accepts NaN, Infinity and -Infinity as a documented PYTHON
    EXTENSION; Go's encoding/json rejects all three. Without this hook the port
    would drop a "NaN" rationale that consumption would have RENDERED, so the
    two sides would disagree about the same string."""
    raise ValueError("not JSON to encoding/json")


def _is_json_value(text: str) -> bool:
    """Go's ``json.Valid``: true for objects, arrays AND bare scalars.

    ``json.loads`` accepts ALMOST the same grammar: ``42``, ``true``, ``null``
    and ``"quoted"`` are JSON values on both sides and therefore not prose. The
    ONE divergence is NaN/Infinity/-Infinity, closed by ``parse_constant``
    above and pinned by the shared vectors.
    """
    try:
        json.loads(text, parse_constant=_reject_constant)
    except (ValueError, TypeError):
        return False
    return True


def _unwrap_fence(text: str) -> str:
    """Strip ONE leading markdown fence and its trailing partner.

    Mirrors the original exactly, including the language-tag rule: a first line
    with no interior whitespace is a language tag and is dropped; a first line
    that contains a space or tab is CONTENT and is kept.
    """
    if not text.startswith("```"):
        return text
    out = text[3:]
    nl = out.find("\n")
    if nl >= 0:
        first_line = out[:nl].strip()
        if first_line == "" or not any(ch in first_line for ch in " \t"):
            out = out[nl + 1 :]
    out = out.strip()
    if out.endswith("```"):
        out = out[: -len("```")]
    return out.strip()


def prose_narrative(s: str | None) -> str:
    """Return ``s`` only when it is genuine learner-facing prose, else ``""``."""
    out = (s or "").strip()
    out = _unwrap_fence(out)
    if out == "" or out.startswith("{") or out.startswith("[") or _is_json_value(out):
        return ""
    return out
