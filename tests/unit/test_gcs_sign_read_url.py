"""GcsImageUploadAdapter.sign_read_url (ADR-254 D12) — MinIO-backed.

On the bus the qgen_render agent writes the scene image and returns its
gs:// object; the kennel keeps the presigned-URL step, so the adapter needs a
sign-only method that signs an EXISTING object by gs:// reference (the bucket
is taken from the URI, which may differ from the upload bucket).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from chora_ai_kernel_orchestrator.adapter.gcs import GcsImageUploadAdapter


@dataclass
class _FakeMinio:
    """Duck-typed Minio client — records presigned_get_object calls."""

    signed: list[tuple[str, str, int]] = field(default_factory=list)

    def presigned_get_object(self, bucket: str, key: str, expires: int) -> str:
        self.signed.append((bucket, key, expires))
        return f"https://minio.local/{bucket}/{key}?sig=abc"


@pytest.mark.asyncio
async def test_sign_read_url_signs_the_named_object() -> None:
    client = _FakeMinio()
    adapter = GcsImageUploadAdapter(bucket_name="chora-ai-assist-images-dev", client=client)
    url = await adapter.sign_read_url("gs://chora-ai-assist-images-dev/tenants/t/jobs/j/abc.png")
    assert url == "https://minio.local/chora-ai-assist-images-dev/tenants/t/jobs/j/abc.png?sig=abc"
    assert client.signed == [("chora-ai-assist-images-dev", "tenants/t/jobs/j/abc.png", timedelta(seconds=604800))]


@pytest.mark.asyncio
async def test_sign_read_url_takes_the_bucket_from_the_uri() -> None:
    client = _FakeMinio()
    adapter = GcsImageUploadAdapter(bucket_name="upload-bucket", client=client)
    await adapter.sign_read_url("gs://other-bucket/k.png")
    assert client.signed == [("other-bucket", "k.png", timedelta(seconds=604800))]


@pytest.mark.asyncio
async def test_sign_read_url_refuses_a_non_gs_reference() -> None:
    adapter = GcsImageUploadAdapter(bucket_name="b", client=_FakeMinio())
    with pytest.raises(ValueError):
        await adapter.sign_read_url("https://storage.googleapis.com/b/k.png")
    with pytest.raises(ValueError):
        await adapter.sign_read_url("gs://bucket-only")


@pytest.mark.asyncio
async def test_sign_read_url_accepts_the_s3_scheme_the_deployment_emits() -> None:
    """The cloud-neutral deployment's qgen_render agent returns an s3:// URI
    (its S3ObjectStore writes MinIO), while the adapter's legacy spelling is
    gs://. Rejecting s3:// made every render look failed, so nothing was
    stamped onto the candidate."""
    client = _FakeMinio()
    adapter = GcsImageUploadAdapter(bucket_name="chora-ai-assist-images", client=client)

    url = await adapter.sign_read_url("s3://chora-ai-assist-images/tenants/t/jobs/j/abc.png")

    assert url == "https://minio.local/chora-ai-assist-images/tenants/t/jobs/j/abc.png?sig=abc"
    assert client.signed == [("chora-ai-assist-images", "tenants/t/jobs/j/abc.png", timedelta(seconds=604800))]
