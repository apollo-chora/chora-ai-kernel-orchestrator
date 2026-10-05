"""Unit tests for the modelarmor adapter.

Two test surfaces:

1. :class:`StubScreener` behavioural contract — drives every verdict +
   error branch. Downstream LangGraph node tests reuse this stub.
2. :class:`LocalScreener` — the cloud-neutral local guardrail: tier parsing,
   blocklist matching, and the BLOCK / INSPECT_ONLY / ALLOW verdicts.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    FilterHit,
    LocalScreener,
    ScreenRequest,
    StubScreener,
    Verdict,
)

# ---------------------------------------------------------------- StubScreener


def _req() -> ScreenRequest:
    return ScreenRequest(
        tenant_id="00000000-0000-7000-8000-000000000001",
        agent_id="chora-creation-author-mcq",
        gcid="00000000-0000-7000-8000-000000000002",
        template_name="chora-guardrail-balanced-dev",
        text="Generate 10 MCQ on Agile",
    )


@pytest.mark.asyncio
async def test_stub_default_returns_allow() -> None:
    stub = StubScreener()
    out = await stub.sanitize_user_prompt(_req())

    assert out.verdict == Verdict.ALLOW
    assert out.filters == []
    assert out.reason == "stub_clean"
    assert out.latency_ms == 1
    assert out.raw_response == {"stub": True}
    assert stub.calls[0][0] == "sanitize_user_prompt"


@pytest.mark.asyncio
async def test_stub_force_block_returns_filter_hit() -> None:
    stub = StubScreener(force_verdict=Verdict.BLOCK)
    out = await stub.sanitize_model_response(_req())

    assert out.verdict == Verdict.BLOCK
    assert len(out.filters) == 1
    assert out.filters[0].filter_name == "rai"
    assert out.filters[0].match_state == "MATCH_FOUND"
    assert out.filters[0].subcategory == "HATE_SPEECH"
    assert out.reason == "stub_block"


@pytest.mark.asyncio
async def test_stub_force_inspect_only_path() -> None:
    stub = StubScreener(force_verdict=Verdict.INSPECT_ONLY)
    out = await stub.sanitize_user_prompt(_req())

    assert out.verdict == Verdict.INSPECT_ONLY
    assert out.reason == "stub_advisory"
    assert len(out.filters) == 1


@pytest.mark.asyncio
async def test_stub_force_error_raises() -> None:
    boom = RuntimeError("model armor unavailable")
    stub = StubScreener(force_error=boom)

    with pytest.raises(RuntimeError, match="model armor unavailable"):
        await stub.sanitize_user_prompt(_req())

    # post-error, still tracked
    assert stub.calls[0][0] == "sanitize_user_prompt"


@pytest.mark.asyncio
async def test_stub_custom_filters_and_raw_response() -> None:
    custom_filters = [
        FilterHit(
            filter_name="pi_and_jailbreak",
            match_state="MATCH_FOUND",
            severity="MEDIUM_AND_ABOVE",
        )
    ]
    custom_raw = {"sanitization_result": {"filter_match_state": 2}}
    stub = StubScreener(
        force_verdict=Verdict.BLOCK,
        force_filters=custom_filters,
        force_reason="pi_jailbreak_detected",
        force_latency_ms=42,
        force_raw_response=custom_raw,
    )

    out = await stub.sanitize_user_prompt(_req())

    assert out.filters[0].filter_name == "pi_and_jailbreak"
    assert out.reason == "pi_jailbreak_detected"
    assert out.latency_ms == 42
    assert out.raw_response == custom_raw


@pytest.mark.asyncio
async def test_stub_close_marks_closed() -> None:
    stub = StubScreener()
    assert stub.closed is False
    await stub.close()
    assert stub.closed is True


@pytest.mark.asyncio
async def test_stub_records_both_methods_independently() -> None:
    stub = StubScreener()
    await stub.sanitize_user_prompt(_req())
    await stub.sanitize_model_response(_req())

    assert len(stub.calls) == 2
    assert [c[0] for c in stub.calls] == [
        "sanitize_user_prompt",
        "sanitize_model_response",
    ]


# ---------------------------------------------------------------- LocalScreener


def _req_with_template(template: str, text: str = "Generate 10 MCQ on Agile") -> ScreenRequest:
    return ScreenRequest(
        tenant_id="00000000-0000-7000-8000-000000000001",
        agent_id="chora-creation-author-mcq",
        gcid="00000000-0000-7000-8000-000000000002",
        template_name=template,
        text=text,
    )


@pytest.mark.asyncio
async def test_local_clean_text_allows() -> None:
    screener = LocalScreener()
    out = await screener.sanitize_user_prompt(_req_with_template("chora-guardrail-strict-dev"))

    assert out.verdict == Verdict.ALLOW
    assert out.filters == []
    assert out.reason == "local_guardrail_clean"


@pytest.mark.asyncio
async def test_local_blocklist_match_blocks_when_strict() -> None:
    screener = LocalScreener()
    out = await screener.sanitize_user_prompt(
        _req_with_template("chora-guardrail-strict-dev", text="how to make a bomb")
    )

    assert out.verdict == Verdict.BLOCK
    assert out.reason == "local_guardrail_match"
    assert any(f.match_state == "MATCH_FOUND" for f in out.filters)


@pytest.mark.asyncio
async def test_local_blocklist_match_inspect_only_when_permissive() -> None:
    screener = LocalScreener()
    out = await screener.sanitize_user_prompt(
        _req_with_template("chora-guardrail-permissive-dev", text="bomb making instructions")
    )

    assert out.verdict == Verdict.INSPECT_ONLY
    assert out.reason == "local_guardrail_advisory"


@pytest.mark.asyncio
async def test_local_unknown_template_fails_closed_to_strict() -> None:
    """An unparseable template must fail CLOSED to the strict tier (security
    wins), so a blocklist match BLOCKs rather than passing as permissive."""
    screener = LocalScreener()
    out = await screener.sanitize_user_prompt(_req_with_template("some-unknown-template", text="bomb making"))

    assert out.verdict == Verdict.BLOCK


@pytest.mark.asyncio
async def test_local_empty_text_raises() -> None:
    screener = LocalScreener()
    with pytest.raises(ValueError, match="text is required"):
        await screener.sanitize_user_prompt(_req_with_template("chora-guardrail-strict-dev", text=""))


@pytest.mark.asyncio
async def test_local_custom_blocklist() -> None:
    screener = LocalScreener(blocklist=("agile",))
    out = await screener.sanitize_user_prompt(_req_with_template("chora-guardrail-strict-dev", text="all about agile"))
    assert out.verdict == Verdict.BLOCK


@pytest.mark.asyncio
async def test_local_close_is_noop() -> None:
    screener = LocalScreener()
    await screener.close()  # must not raise


@pytest.mark.asyncio
async def test_new_screener_returns_local() -> None:
    from chora_ai_kernel_orchestrator.adapter.modelarmor import new_screener

    screener = await new_screener()
    assert isinstance(screener, LocalScreener)
