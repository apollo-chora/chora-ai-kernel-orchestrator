"""DSN resolver — env-backed (cloud-neutral).

Per CLAUDE.md §6 + memory ``feedback_no_inline_config``: DSNs come from the
environment. The Google Cloud Secret Manager path is removed; the local
substitute is a direct DSN in the environment (the compose stack injects it).

Environment contract:

- ``CHORA_AI_KERNEL_PG_DSN`` — direct DSN. The only path.

Behaviour:

- Direct DSN set → return it immediately.
- Unset (or whitespace-only) → return empty string. Caller is expected to
  fall through to InMemorySaver in dev / unit tests.
"""

from __future__ import annotations

import os


def resolve_dsn() -> str:
    """Resolve the chora_ai_kernel DSN from the environment.

    Returns the empty string when unset — caller MUST treat empty as
    ``unset``.
    """
    return (os.getenv("CHORA_AI_KERNEL_PG_DSN") or "").strip()
