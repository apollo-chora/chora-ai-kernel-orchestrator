"""Tests for the local SafeSearch adapter (ADR-205 WS-3 / D2).

The graduated Growth-Edge crew screens the RAW uploaded image for explicit
content BEFORE the extractor reads it (``extract_node`` calls
``safesearch.inspect`` first). This adapter fills the graph's ``SafeSearchPort``:

  * non-image MIME (PDF / text) → SafeSearch does not apply; pass (the text is
    screened downstream by the guardrail). Never calls a remote API.
  * image MIME → the explicit-content gate. Cloud Vision is gone; the local
    adapter FAILS OPEN (pass) by default so the feature keeps working, and
    logs LOUD that the Vision gate is not enforced. Set
    CHORA_SAFESEARCH_FAIL_CLOSED=1 to fail CLOSED (block every image).
  * fail-CLOSED: empty image bytes raise (nothing to inspect).
"""

from __future__ import annotations

import asyncio

import pytest

from chora_ai_kernel_orchestrator.adapter.weakness.safesearch import (
    LocalSafeSearchAdapter,
)


def _inspect(adapter: LocalSafeSearchAdapter, blob: bytes, mime: str) -> bool:
    return asyncio.run(adapter.inspect(blob_bytes=blob, mime_type=mime))


# --------------------------------------------------------------------------- #
# non-image MIME — the gate is never applied
# --------------------------------------------------------------------------- #


def test_pdf_mime_passes() -> None:
    adapter = LocalSafeSearchAdapter()
    assert _inspect(adapter, b"%PDF-1.7", "application/pdf") is True


def test_text_mime_passes() -> None:
    adapter = LocalSafeSearchAdapter()
    assert _inspect(adapter, b"my notes", "text/plain") is True


def test_empty_mime_passes() -> None:
    adapter = LocalSafeSearchAdapter()
    assert _inspect(adapter, b"bytes", "") is True


# --------------------------------------------------------------------------- #
# image MIME — the local gate
# --------------------------------------------------------------------------- #


def test_image_mime_fails_open_by_default() -> None:
    """Cloud Vision is gone: the local adapter fails OPEN (pass) so the
    weakness feature keeps working, and logs that the gate is not enforced."""
    adapter = LocalSafeSearchAdapter()
    assert _inspect(adapter, b"\x89PNG...", "image/png") is True


def test_image_mime_fail_closed_blocks() -> None:
    adapter = LocalSafeSearchAdapter(fail_closed=True)
    assert _inspect(adapter, b"\x89PNG...", "image/png") is False


def test_empty_image_bytes_raises_fail_closed() -> None:
    adapter = LocalSafeSearchAdapter()
    with pytest.raises(ValueError):
        _inspect(adapter, b"", "image/png")


# --------------------------------------------------------------------------- #
# from_env
# --------------------------------------------------------------------------- #


def test_from_env_fail_closed_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHORA_SAFESEARCH_FAIL_CLOSED", "1")
    adapter = LocalSafeSearchAdapter.from_env()
    assert _inspect(adapter, b"\x89PNG...", "image/png") is False


def test_from_env_fail_closed_flag_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHORA_SAFESEARCH_FAIL_CLOSED", "0")
    adapter = LocalSafeSearchAdapter.from_env()
    assert _inspect(adapter, b"\x89PNG...", "image/png") is True


def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHORA_SAFESEARCH_FAIL_CLOSED", raising=False)
    monkeypatch.delenv("WEAKNESS_SAFESEARCH_BLOCK_THRESHOLD", raising=False)
    monkeypatch.delenv("WEAKNESS_SAFESEARCH_CATEGORIES", raising=False)
    adapter = LocalSafeSearchAdapter.from_env()
    assert adapter.block_threshold_rank == 4  # LIKELY
    assert "adult" in adapter.categories
    assert _inspect(adapter, b"\x89PNG...", "image/png") is True
