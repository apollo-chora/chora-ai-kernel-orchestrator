"""CHO-2379 - picking the next free prompt version label (pure domain rule).

ADR-197 makes ARCHIVED terminal, so a rollback plus a re-promotion always mints
a FRESH plan row. Before this rule the fresh row reused the embedded revision's
constant label, which produced several catalogue entries all reading 1.1.0.
That is not cosmetic:

- the catalogue's single-version read disambiguates duplicates with
  ``ORDER BY created_at DESC LIMIT 1``, so every superseded attempt sharing a
  label became UNREACHABLE in the O+ prompt modal;
- the Vertex eval-run name is derived from the label, and reuse collided with
  a 409 AlreadyExists (worked around with an r2 suffix in CHO-2368).

The label is DISPLAY plus stamp identity only. The resolver still picks the
active plan by (agent, status), so bumping a label never moves a resolution,
and historical StepStamps keep the label they were written with as a
point-in-time fact.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: A label the auto-bump understands: strict major.minor.patch. The seeded
#: baselines carry 'v1' and are deliberately OUTSIDE this shape - a baseline is
#: an immutable transcription of the embedded prompt, never a promotion attempt.
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

#: The label the FIRST override attempt of the current wave takes. Mirrors
#: ``revisions_110.REVISION_VERSION_LABEL``, which stays the CONTENT revision
#: identity; only the PLAN label bumps off it.
DEFAULT_BASE_LABEL = "1.1.0"


def next_version_label(existing: Iterable[str | None], base: str = DEFAULT_BASE_LABEL) -> str:
    """Return the next free patch label in ``base``'s major.minor family.

    ``existing`` is every label already present for the agent in the platform
    catalogue (baselines included; they are filtered here rather than at the
    call site so a caller can hand over the catalogue rows verbatim). Labels
    outside ``base``'s family, ``None`` and malformed values are ignored: the
    promotion lane must not die on a hand-inserted row.

    Holes are NEVER refilled. A re-promotion has to sort ABOVE the attempt it
    supersedes, so the family maximum plus one wins even when a lower patch is
    free. ``base`` acts as the floor, so the result can never land below it.

    Raises ``ValueError`` when ``base`` is not major.minor.patch - a malformed
    base would silently produce a family nothing can ever match.
    """
    base_match = _SEMVER_RE.match(base) if isinstance(base, str) else None
    if base_match is None:
        raise ValueError(f"next_version_label: base must be major.minor.patch, got {base!r}")
    major, minor, base_patch = (int(part) for part in base_match.groups())

    patches = [
        int(match.group(3))
        for match in (_SEMVER_RE.match(label) for label in existing if isinstance(label, str))
        if match is not None and int(match.group(1)) == major and int(match.group(2)) == minor
    ]
    if not patches:
        return base
    return f"{major}.{minor}.{max(max(patches) + 1, base_patch)}"
