"""Run-id helper shared by the crews.

Per CLAUDE.md section 6: every event-bearing payload uses UUIDv7 ids. A crew
generates its run id at the boundary so downstream ledgers and traces sort
lexicographically by id.

The generic ``OrchestratorState`` TypedDict and ``OrchestratorDecision`` enum
that used to live here went with the legacy /orchestrate graph they typed
(RULING A, 2026-08-23). Each crew owns its own state module, and the guardrail
verdict is read from ``adapter.modelarmor.Verdict`` directly.
"""

from __future__ import annotations

import os
import secrets
import time


def new_request_id() -> str:
    """Generate a UUIDv7 string.

    Uses time-ordered prefix per RFC 9562 v7 layout. We avoid an external
    dependency to keep the request_id helper trivial; the Python `uuid7`
    package is in pyproject for tests/runtime parity.
    """
    # Prefer external uuid7 if available — keeps parity with rest of fleet.
    try:
        import uuid7 as _u7

        return str(_u7.uuid7())
    except Exception:  # pragma: no cover
        # fail-loud-exempt: an optional-import fallback, not an error path. The
        # RFC 9562 construction below is a complete implementation, so the
        # caller gets a correct UUIDv7 either way and nothing is degraded.
        # Fallback: manual UUIDv7 construction.
        ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
        rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF  # 12 bits
        rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)  # 62 bits
        # Layout: 48-bit ts | 4-bit ver(7) | 12-bit rand_a | 2-bit var(10) | 62-bit rand_b
        version = 7 << 12
        variant = 0b10 << 62
        hi = (ts_ms << 16) | version | rand_a
        lo = variant | rand_b
        full = (hi << 64) | lo
        # Sprinkle some final entropy if the OS RNG was zero (defensive).
        full ^= secrets.randbits(64)
        # Re-set version + variant after entropy mix.
        full &= ~(0xF000 << 64)
        full |= 0x7000 << 64
        full &= ~(0b11 << 62)
        full |= 0b10 << 62

        hexs = f"{full:032x}"
        return f"{hexs[0:8]}-{hexs[8:12]}-{hexs[12:16]}-{hexs[16:20]}-{hexs[20:32]}"
