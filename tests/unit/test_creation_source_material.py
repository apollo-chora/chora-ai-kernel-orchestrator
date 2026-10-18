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
    REASON_FAILED,
    REASON_NO_TEXT,
    REASON_NOT_CONFIGURED,
    REASON_REJECTED,
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
@pytest.mark.parametrize("status", [400, 403, 404])
async def test_client_errors_are_rejected_not_unreachable(status: int) -> None:
    # A response is proof creation was REACHED. An unknown job (404) must never
    # look like a job with zero text, and neither may look like a network fault.
    http = _Client(_Resp(status_code=status))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    assert err.value.reason == REASON_REJECTED
    assert str(status) in err.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 503])
async def test_server_errors_are_failed_not_unreachable(status: int) -> None:
    http = _Client(_Resp(status_code=status))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    assert err.value.reason == REASON_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["false", 0, 1, [], None])
async def test_non_boolean_chunks_complete_is_rejected(value) -> None:
    # "false" is a truthy string: a permissive bool() would report a truncated
    # corpus as complete.
    http = _Client(_Resp(body={"chunks": [_chunk()], "chunks_complete": value}))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    assert err.value.reason == REASON_BAD_RESPONSE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    [
        "not-an-object",
        {"text": 5},
        {"text": "x", "file_uri": 7},
        {"text": "x", "chunk_index": "1"},
        {"text": "x", "page_no": "1"},
    ],
)
async def test_structurally_malformed_chunks_are_rejected_not_coerced(entry) -> None:
    http = _Client(_Resp(body={"chunks": [entry], "chunks_complete": True}))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    assert err.value.reason == REASON_BAD_RESPONSE


@pytest.mark.asyncio
async def test_whitespace_chunk_is_skipped_not_rejected() -> None:
    # Skipping blank text is legitimate; COERCING a malformed field is not.
    http = _Client(_Resp(body={"chunks": [_chunk(text="  "), _chunk(text="real")], "chunks_complete": True}))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    text, complete = await client.fetch(tenant_id="t1", job_id="j1")
    assert "real" in text and complete is True


@pytest.mark.asyncio
async def test_malformed_chunk_field_stays_typed() -> None:
    # A 200 carrying garbage must leave through the typed contract; a raw
    # ValueError would escape every caller's except SourceMaterialError.
    http = _Client(_Resp(body={"chunks": [_chunk(chunk_index="garbage")], "chunks_complete": True}))
    client = CreationSourceMaterialClient("http://c", http_client=http)
    with pytest.raises(SourceMaterialError) as err:
        await client.fetch(tenant_id="t1", job_id="j1")
    assert err.value.reason == REASON_BAD_RESPONSE


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


# ---- state glue --------------------------------------------------------------

from chora_ai_kernel_orchestrator.adapter.creation.source_material import (  # noqa: E402
    declares_source_material,
    source_material_text_for_state,
)


class _FakeMaterialClient:
    def __init__(self, text: str = "MATERIAL", complete: bool = True, exc=None) -> None:
        self.calls: list[dict] = []
        self._text = text
        self._complete = complete
        self._exc = exc

    async def fetch(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._text, self._complete


def test_declares_material_matches_the_go_plugin_rule() -> None:
    assert declares_source_material({"source_files": [{"blob_uri": "s3://b/x"}]}) is True
    assert declares_source_material({"source_blob_uri": "s3://b/x"}) is True
    assert declares_source_material({"source_material_text": "already here"}) is True
    # A list whose entries carry no blob_uri is NOT material in the Go plugin,
    # so it must not be material here either.
    assert declares_source_material({"source_files": [{}]}) is False
    assert declares_source_material({"source_files": [{"blob_uri": "   "}]}) is False
    assert declares_source_material({"source_files": []}) is False
    assert declares_source_material({"source_blob_uri": "   "}) is False
    assert declares_source_material({}) is False


@pytest.mark.asyncio
async def test_ungrounded_state_is_untouched_and_never_calls_creation() -> None:
    client = _FakeMaterialClient()
    text, complete = await source_material_text_for_state({"tenant_id": "t", "job_id": "j"}, client=client)
    # The prompt-only and single-candidate paths must stay byte-identical.
    assert (text, complete) == ("", True)
    assert client.calls == []


@pytest.mark.asyncio
async def test_grounded_state_passes_tenant_job_and_author() -> None:
    client = _FakeMaterialClient()
    text, complete = await source_material_text_for_state(
        {"tenant_id": "t1", "job_id": "j1", "gcid": "g1", "source_blob_uri": "s3://b/x"},
        client=client,
    )
    assert text == "MATERIAL" and complete is True
    assert client.calls[0] == {"tenant_id": "t1", "job_id": "j1", "author_gcid": "g1", "max_bytes": DEFAULT_MAX_BYTES}


@pytest.mark.asyncio
async def test_gcid_wins_over_user_gcid() -> None:
    client = _FakeMaterialClient()
    await source_material_text_for_state(
        {
            "tenant_id": "t",
            "job_id": "j",
            "gcid": "from-state",
            "user_gcid": "from-payload",
            "source_files": [{"blob_uri": "s3://b/x"}],
        },
        client=client,
    )
    # `gcid` is the authoritative graph-state author; user_gcid is a fallback.
    assert client.calls[0]["author_gcid"] == "from-state"


@pytest.mark.asyncio
async def test_declared_material_without_a_client_is_a_refusal(monkeypatch) -> None:
    monkeypatch.delenv(ENV_CREATION_BASE_URL, raising=False)
    with pytest.raises(SourceMaterialError) as err:
        await source_material_text_for_state({"tenant_id": "t", "job_id": "j", "source_blob_uri": "s3://b/x"})
    assert err.value.reason == REASON_NOT_CONFIGURED


@pytest.mark.asyncio
async def test_fetch_failure_propagates_as_a_refusal() -> None:
    client = _FakeMaterialClient(exc=SourceMaterialError(REASON_NO_TEXT, "empty"))
    with pytest.raises(SourceMaterialError) as err:
        await source_material_text_for_state(
            {"tenant_id": "t", "job_id": "j", "source_blob_uri": "s3://b/x"}, client=client
        )
    assert err.value.reason == REASON_NO_TEXT
