"""Local SafeSearch adapter for the Growth-Edge crew (ADR-205 WS-3 / D2).

Cloud Vision SafeSearch (Google) has no drop-in local equivalent. This module
backs the same ``SafeSearchPort`` (``orchestrators/weakness_analyser_crew.py``)
with a local adapter so the seam and the crew graph are unchanged.

Contract (per ADR-205 WS-3 / D2), preserved:

  * non-image MIME (PDF / text / unknown) → SafeSearch does not apply; PASS.
    The extracted text is Model-Armor-screened downstream (``screen_input``).
  * image MIME → the explicit-content gate. Cloud Vision is gone; the local
    adapter FAILS OPEN (PASS) by default so the weakness feature keeps working,
    and logs LOUD that the Vision gate is not enforced locally. Set
    ``CHORA_SAFESEARCH_FAIL_CLOSED=1`` to fail CLOSED (BLOCK every image) for
    a deployment that prefers the feature be unavailable over ungated.
  * fail-CLOSED on empty image bytes (nothing to inspect) — unchanged.

The explicit-content screening a managed Vision API provided is NOT enforced
locally. A deployment that needs it should front this with a local
image-moderation service and re-point the port.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Cloud Vision ``Likelihood`` enum → ordinal rank. Kept for interface parity
# with the retired Cloud Vision adapter (the local adapter does not call a
# moderation API, but the threshold/categories config is still accepted).
_LIKELIHOOD_RANK = {
    "UNKNOWN": 0,
    "VERY_UNLIKELY": 1,
    "UNLIKELY": 2,
    "POSSIBLE": 3,
    "LIKELY": 4,
    "VERY_LIKELY": 5,
}

_DEFAULT_CATEGORIES = ("adult", "violence", "racy")
_DEFAULT_THRESHOLD = "LIKELY"

ENV_THRESHOLD = "WEAKNESS_SAFESEARCH_BLOCK_THRESHOLD"
ENV_CATEGORIES = "WEAKNESS_SAFESEARCH_CATEGORIES"
ENV_FAIL_CLOSED = "CHORA_SAFESEARCH_FAIL_CLOSED"

_TRUTHY = {"1", "true", "yes"}


def _rank(value: Any) -> int:
    """Coerce a likelihood (enum member / int / name) to an ordinal."""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return _LIKELIHOOD_RANK.get(name.upper(), 0)
    if isinstance(value, bool):  # bool is an int subclass — guard first
        return 0
    if isinstance(value, int):
        return value if 0 <= value <= 5 else 0
    return _LIKELIHOOD_RANK.get(str(value).upper(), 0)


def _threshold_rank(name: str) -> int:
    """Resolve a threshold name to its rank; UNKNOWN / unparseable → default."""
    rank = _LIKELIHOOD_RANK.get((name or "").strip().upper())
    if not rank:  # None or 0 (UNKNOWN) is never a sensible gate
        return _LIKELIHOOD_RANK[_DEFAULT_THRESHOLD]
    return rank


class LocalSafeSearchAdapter:
    """Crew ``SafeSearchPort`` — local, Vision-free explicit-content gate.

    ``client`` is accepted (and ignored) for constructor compatibility with
    the retired Cloud Vision adapter; the local adapter contacts no remote
    service. Production builds via :meth:`from_env`.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        categories: tuple[str, ...] = _DEFAULT_CATEGORIES,
        block_threshold: str = _DEFAULT_THRESHOLD,
        fail_closed: bool = False,
    ) -> None:
        self._client = client
        self._categories = tuple(c.strip().lower() for c in categories if c.strip())
        self._threshold_rank = _threshold_rank(block_threshold)
        self._fail_closed = bool(fail_closed)

    @classmethod
    def from_env(cls) -> LocalSafeSearchAdapter:
        """Build from optional env overrides (defaults keep the feature up)."""
        threshold = os.getenv(ENV_THRESHOLD, "").strip() or _DEFAULT_THRESHOLD
        raw_cats = os.getenv(ENV_CATEGORIES, "").strip()
        categories = (
            tuple(c.strip().lower() for c in raw_cats.split(",") if c.strip()) if raw_cats else _DEFAULT_CATEGORIES
        )
        fail_closed = os.getenv(ENV_FAIL_CLOSED, "").strip().lower() in _TRUTHY
        return cls(
            categories=categories or _DEFAULT_CATEGORIES,
            block_threshold=threshold,
            fail_closed=fail_closed,
        )

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories

    @property
    def block_threshold_rank(self) -> int:
        return self._threshold_rank

    async def inspect(self, *, blob_bytes: bytes, mime_type: str) -> bool:
        """True = safe to read; False = explicit (the crew BLOCKs → refund)."""
        if not (mime_type or "").lower().startswith("image/"):
            logger.debug("weakness_safesearch.skip_non_image", extra={"mime_type": mime_type})
            return True
        if not blob_bytes:
            raise ValueError("safesearch: empty image bytes (nothing to inspect)")
        if self._fail_closed:
            logger.warning("weakness_safesearch.blocked_fail_closed")
            return False
        # Cloud Vision is gone: fail OPEN so the weakness feature keeps working,
        # and say so loudly — a silent "safe" is the failure mode this line
        # exists to prevent.
        logger.warning(
            "weakness_safesearch.vision_disabled_fail_open",
            extra={"bytes": len(blob_bytes), "mime_type": mime_type},
        )
        return True


__all__ = [
    "ENV_CATEGORIES",
    "ENV_FAIL_CLOSED",
    "ENV_THRESHOLD",
    "LocalSafeSearchAdapter",
]
