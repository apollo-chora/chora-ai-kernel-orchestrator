"""W8: thin image adapters (Kroki / GCS) unit tests.

RED→GREEN per [[feedback-strict-tdd]]. Each adapter is injectable (constructor
deps) so the render-loop tests mock them; here we exercise the adapters
themselves with fake transports, mirroring the executor/guardrail port-adapter
style. No real HTTP / GCS. (The W8 model-gateway image client retired with
ADR-254 D12: a scene render is a qgen_render dispatch, the kennel only signs.)
"""

from __future__ import annotations

from datetime import timedelta

from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.gcs import GcsImageUploadAdapter
from chora_ai_kernel_orchestrator.adapter.kroki import KrokiClient

# =============================================================================
# KrokiClient
# =============================================================================


@dataclass
class _FakeHttpResp:
    status_code: int
    content: bytes = b""
    text: str = ""


@dataclass
class _FakeHttpClient:
    resp: _FakeHttpResp
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def post(
        self,
        url: str,
        *,
        content: Any = None,
        headers: dict[str, str] | None = None,
        **_: Any,
    ) -> _FakeHttpResp:
        self.calls.append({"url": url, "content": content, "headers": headers})
        return self.resp


@pytest.mark.asyncio
async def test_kroki_render_posts_source_and_returns_bytes() -> None:
    http = _FakeHttpClient(resp=_FakeHttpResp(status_code=200, content=b"<svg>ok</svg>"))
    client = KrokiClient(endpoint="http://kroki:8000", http_client=http)
    out = await client.render(source="graph TD; A-->B", output_format="svg")
    assert out == b"<svg>ok</svg>"
    assert len(http.calls) == 1
    # POSTs to {endpoint}/mermaid/{format} with the raw source as body.
    assert http.calls[0]["url"] == "http://kroki:8000/mermaid/svg"
    assert http.calls[0]["content"] == b"graph TD; A-->B"


@pytest.mark.asyncio
async def test_kroki_render_default_format_png() -> None:
    http = _FakeHttpClient(resp=_FakeHttpResp(status_code=200, content=b"PNG"))
    client = KrokiClient(endpoint="http://kroki:8000/", http_client=http)
    out = await client.render(source="graph TD; A-->B")
    assert out == b"PNG"
    # Trailing slash on endpoint is normalised (no double slash).
    assert http.calls[0]["url"] == "http://kroki:8000/mermaid/png"


@pytest.mark.asyncio
async def test_kroki_render_raises_on_http_error() -> None:
    http = _FakeHttpClient(resp=_FakeHttpResp(status_code=400, text="bad mermaid"))
    client = KrokiClient(endpoint="http://kroki:8000", http_client=http)
    with pytest.raises(RuntimeError):
        await client.render(source="not mermaid", output_format="png")


def test_kroki_from_env_none_when_unset(monkeypatch: Any) -> None:
    monkeypatch.delenv("KROKI_ENDPOINT", raising=False)
    assert KrokiClient.from_env() is None


def test_kroki_from_env_builds_when_set(monkeypatch: Any) -> None:
    monkeypatch.setenv("KROKI_ENDPOINT", "http://chora-kroki.ai-kernel.svc:8000")
    client = KrokiClient.from_env()
    assert client is not None
    assert client.endpoint == "http://chora-kroki.ai-kernel.svc:8000"


# =============================================================================
# GcsImageUploadAdapter (MinIO-backed)
# =============================================================================


@dataclass
class _FakeMinio:
    """Duck-typed Minio client — records put_object / presigned_get_object."""

    puts: list[tuple[str, str, bytes, str]] = field(default_factory=list)
    signed: list[tuple[str, str, int]] = field(default_factory=list)

    def put_object(
        self,
        bucket: str,
        key: str,
        data: Any,
        length: int,
        content_type: str = "",
    ) -> None:
        self.puts.append((bucket, key, data.read(), content_type))

    def presigned_get_object(self, bucket: str, key: str, expires: int) -> str:
        self.signed.append((bucket, key, expires))
        return "https://minio.local/signed?sig=abc"


@pytest.mark.asyncio
async def test_gcs_upload_and_sign_writes_and_signs() -> None:
    client = _FakeMinio()
    adapter = GcsImageUploadAdapter(
        bucket_name="chora-ai-assist-images-dev",
        client=client,
    )
    gs_uri, url = await adapter.upload_and_sign(
        tenant_id="t1", job_id="job-1", data=b"PNGDATA", content_type="image/png"
    )
    assert url == "https://minio.local/signed?sig=abc"
    # One object written under the tenant/job-scoped key.
    assert len(client.puts) == 1
    bucket, key, data, content_type = client.puts[0]
    assert bucket == "chora-ai-assist-images-dev"
    assert key.startswith("tenants/t1/jobs/job-1/")
    assert key.endswith(".png")
    assert data == b"PNGDATA"
    assert content_type == "image/png"
    # ADR-210 B2 — the canonical durable gs:// object path is returned.
    assert gs_uri == "gs://chora-ai-assist-images-dev/" + key
    # Presigned GET URL minted (7-day expiry).
    assert client.signed == [(bucket, key, timedelta(seconds=604800))]


def test_gcs_from_env_none_when_unset(monkeypatch: Any) -> None:
    monkeypatch.delenv("QGEN_IMAGE_GCS_BUCKET", raising=False)
    assert GcsImageUploadAdapter.from_env() is None
