"""In-process stub :class:`.screener.Screener` for unit tests.

No SDK calls, no network. Configurable to return a fixed verdict or raise
a fixed exception so downstream LangGraph node tests can drive every
branch (ALLOW / BLOCK / INSPECT_ONLY / error). Records every call for
post-hoc assertion (``calls`` attribute).

The constructor mirrors test-fixture conventions used elsewhere in this
service (see e.g. ``tests/fakes/`` for parallel patterns).
"""

from __future__ import annotations

from typing import Any

from chora_ai_kernel_orchestrator.adapter.modelarmor.screener import (
    FilterHit,
    ScreenRequest,
    ScreenResult,
    Verdict,
)


class StubScreener:
    """Deterministic in-process Screener for unit tests.

    Parameters
    ----------
    force_verdict:
        If set, every successful call returns a ScreenResult with this
        verdict and the supplied ``force_filters`` / ``force_reason``.
    force_error:
        If set, every call raises this exception (caller drives the
        fail-loud / map-to-BLOCK branch in the LangGraph node).
    force_filters:
        Filter rows to surface alongside ``force_verdict``. Defaults to
        ``[]`` for ALLOW and a single synthesised row for BLOCK /
        INSPECT_ONLY.
    force_reason:
        Reason string. Defaults to a sensible stub label per verdict.
    force_latency_ms:
        Latency to report in the ScreenResult.
    force_raw_response:
        Optional raw response dict to surface for audit + span attribute
        verification.
    """

    def __init__(
        self,
        *,
        force_verdict: Verdict | None = Verdict.ALLOW,
        force_error: BaseException | None = None,
        force_filters: list[FilterHit] | None = None,
        force_reason: str | None = None,
        force_latency_ms: int = 1,
        force_raw_response: dict[str, Any] | None = None,
    ) -> None:
        self._verdict = force_verdict
        self._error = force_error
        self._filters = force_filters
        self._reason = force_reason
        self._latency_ms = force_latency_ms
        self._raw = force_raw_response

        # Audit log — list of (method, ScreenRequest) tuples
        self.calls: list[tuple[str, ScreenRequest]] = []
        self.closed = False

    async def sanitize_user_prompt(self, req: ScreenRequest) -> ScreenResult:
        return await self._respond("sanitize_user_prompt", req)

    async def sanitize_model_response(self, req: ScreenRequest) -> ScreenResult:
        return await self._respond("sanitize_model_response", req)

    async def close(self) -> None:
        self.closed = True

    # ---------------------------------------------------------------- internal

    async def _respond(self, method: str, req: ScreenRequest) -> ScreenResult:
        self.calls.append((method, req))

        if self._error is not None:
            raise self._error

        verdict = self._verdict or Verdict.ALLOW
        filters = self._filters
        if filters is None:
            if verdict == Verdict.ALLOW:
                filters = []
            else:
                filters = [
                    FilterHit(
                        filter_name="rai",
                        match_state="MATCH_FOUND",
                        severity="HIGH",
                        subcategory="HATE_SPEECH",
                    )
                ]
        reason = self._reason
        if reason is None:
            reason = {
                Verdict.ALLOW: "stub_clean",
                Verdict.BLOCK: "stub_block",
                Verdict.INSPECT_ONLY: "stub_advisory",
            }[verdict]
        raw = self._raw if self._raw is not None else {"stub": True}

        return ScreenResult(
            verdict=verdict,
            reason=reason,
            filters=list(filters),
            latency_ms=self._latency_ms,
            raw_response=raw,
        )
