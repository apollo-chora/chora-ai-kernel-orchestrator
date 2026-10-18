"""Source-material retrieval + rendering.

The behaviour under test is what a grounded batch depends on: page-labelled
text that keeps its structure, and a HARD refusal instead of an ungrounded
generation when the material is unavailable.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.adapter.creation.source_material import (
    DEFAULT_MAX_BYTES,
    ENV_CREATION_BASE_URL,
    REASON_BAD_RESPONSE,
    REASON_NO_TEXT,
    REASON_NOT_CONFIGURED,
    REASON_UNREACHABLE,
    CreationSourceMaterialClient,
    SourceMaterialChunk,
    SourceMaterialError,
    render_source_material,
)


class _Resp:
    def __init__(self, status_code: int = 200, body=None, raise_json: bool = False) -> None:
        self.status_code = status_code
        self._body = body
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._body


class _Client:
    def __init__(self, resp=None, exc: Exception | None = None) -> None:
        self._resp = resp
        self._exc = exc
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url: str, params: dict) -> object:
        self.calls.append((url, params))
        if self._exc is not None:
            raise self._exc
        return self._resp


def _chunk(**over) -> dict:
    base = {
        "file_uri": "s3://b/jobs/j1/source",
        "file_role": "source",
        "chunk_index": 0,
        "page_no": 1,
        "text": "Photosynthesis.",
    }
    base.update(over)
    return base


# ---- rendering ---------------------------------------------------------------


def test_render_tags_pages_and_roles() -> None:
    out = render_source_material(
        [
            SourceMaterialChunk("s3://b/p.pdf", "source", 0, 1, "page one"),
            SourceMaterialChunk("s3://b/m.pdf", "rubric", 0, 3, "1 mark each"),
        ]
    )
    assert '<SOURCE file="s3://b/p.pdf" page="1">' in out
    assert "page one" in out
    assert "</SOURCE>" in out
    # A mark scheme GRADES the material; it must stay distinguishable.
    assert '<RUBRIC file="s3://b/m.pdf" page="3">' in out
    assert "1 mark each" in out


def test_render_omits_page_for_unpaged_formats() -> None:
    out = render_source_material([SourceMaterialChunk("s3://b/n.md", "source", 0, None, "notes")])
    # A zero page would be a lie dressed as data.
    assert "page=" not in out
    assert '<SOURCE file="s3://b/n.md">' in out


def test_render_keeps_every_chunk_in_order() -> None:
    chunks = [SourceMaterialChunk("s3://b/p.pdf", "source", i, i + 1, f"text {i}") for i in range(3)]
    out = render_source_material(chunks)
    assert out.index("text 0") < out.index("text 1") < out.index("text 2")
    assert out.count("<SOURCE") == 3


# ---- fetching ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_renders_and_reports_truncation() -> None:
    http = _Client(_Resp(body={"chunks": [_chunk()], "chunks_complete": False}))
    client = CreationSourceMaterialClient("http://creation:8080", http_client=http)

    text, complete = await client.fetch(tenant_id="t1", job_id="j1", author_gcid="g1")

    assert '<SOURCE file="s3://b/jobs/j1/source" page="1">' in text
    assert complete is False
    url, params = http.calls[0]
    assert url == "http://creation:8080/api/internal/question-jobs/j1/source-material"
    assert params["tenant_id"] == "t1"
    assert params["author_gcid"] == "g1"
    assert params["max_bytes"] == str(DEFAULT_MAX_BYTES)


@pytest.mark.asyncio
async def test_fetch_omits_author_when_unknown() -> None:
    http = _Client(_Resp(body={"chunks": [_chunk()], "chunks_complete": True}))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    await client.fetch(tenant_id="t1", job_id="j1")
    assert "author_gcid" not in http.calls[0][1]


@pytest.mark.asyncio
async def test_empty_material_is_a_refusal_not_a_noop() -> None:
    # An image-only upload or failed extraction must NOT become an ungrounded
    # generation the author believes came from their document.
    http = _Client(_Resp(body={"chunks": [], "chunks_complete": True}))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    assert err.value.reason == REASON_NO_TEXT


@pytest.mark.asyncio
async def test_whitespace_only_chunks_are_empty_material() -> None:
    http = _Client(_Resp(body={"chunks": [_chunk(text="   \n ")], "chunks_complete": True}))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    assert err.value.reason == REASON_NO_TEXT


@pytest.mark.asyncio
async def test_http_error_is_unreachable_with_a_reason() -> None:
    http = _Client(_Resp(status_code=500))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    # Machine-readable: the caller must branch on reason, never on English.
    assert err.value.reason == REASON_UNREACHABLE


@pytest.mark.asyncio
async def test_transport_error_is_unreachable() -> None:
    http = _Client(exc=TimeoutError("boom"))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    assert err.value.reason == REASON_UNREACHABLE


@pytest.mark.asyncio
async def test_unparseable_body_is_bad_response() -> None:
    http = _Client(_Resp(raise_json=True))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    assert err.value.reason == REASON_BAD_RESPONSE


@pytest.mark.asyncio
async def test_missing_chunks_array_is_bad_response() -> None:
    http = _Client(_Resp(body={"nope": 1}))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    assert err.value.reason == REASON_BAD_RESPONSE


def test_empty_base_url_refuses_construction() -> None:
    with pytest.raises(SourceMaterialError) as err:
        CreationSourceMaterialClient("   ")
    assert err.value.reason == REASON_NOT_CONFIGURED


def test_from_env_missing_returns_none(monkeypatch) -> None:
    monkeypatch.delenv(ENV_CREATION_BASE_URL, raising=False)
    assert CreationSourceMaterialClient.from_env() is None


def test_from_env_reads_base_and_tolerates_bad_timeout(monkeypatch) -> None:
    monkeypatch.setenv(ENV_CREATION_BASE_URL, "http://creation:8080/")
    monkeypatch.setenv("CREATION_SOURCE_MATERIAL_TIMEOUT_SECONDS", "not-a-number")
    client = CreationSourceMaterialClient.from_env()
    assert client is not None
    # trailing slash must not double up in the path
    assert client._base == "http://creation:8080"
