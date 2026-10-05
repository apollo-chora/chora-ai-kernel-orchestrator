"""Unit tests for the MinIO blob downloader (W2 adapter — satisfies the
weakness-analyser runner's BlobDownloader port). Verified with a fake
MinIO client; no real object store."""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.adapter.gcs.weakness_downloader import (
    GcsBlobDownloader,
    parse_gs_uri,
)

pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        return None

    def release_conn(self) -> None:
        return None


class _FakeMinio:
    """Duck-typed Minio client — records get_object / stat_object calls."""

    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs
        self.requested_bucket: str | None = None
        self.requested_key: str | None = None

    def get_object(self, bucket: str, key: str) -> _FakeResponse:
        self.requested_bucket = bucket
        self.requested_key = key
        return _FakeResponse(self._blobs[f"{bucket}/{key}"])

    def stat_object(self, bucket: str, key: str) -> None:
        if f"{bucket}/{key}" not in self._blobs:
            raise FileNotFoundError(f"{bucket}/{key}")


def test_parse_gs_uri_ok() -> None:
    assert parse_gs_uri("gs://my-bucket/path/to/doc.pdf") == ("my-bucket", "path/to/doc.pdf")


@pytest.mark.parametrize("bad", ["", "http://x/y", "gs://", "gs://only-bucket", "gs:///no-bucket"])
def test_parse_gs_uri_rejects_bad(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_gs_uri(bad)


async def test_download_reads_blob_bytes() -> None:
    client = _FakeMinio({"chora-weakness-docs/t/upl-1.pdf": b"%PDF bytes"})
    dl = GcsBlobDownloader(client=client)

    data = await dl.download("gs://chora-weakness-docs/t/upl-1.pdf")
    assert data == b"%PDF bytes"
    assert client.requested_bucket == "chora-weakness-docs"
    assert client.requested_key == "t/upl-1.pdf"


async def test_download_rejects_bad_uri() -> None:
    dl = GcsBlobDownloader(client=_FakeMinio({}))
    with pytest.raises(ValueError):
        await dl.download("not-a-gs-uri")


@pytest.mark.asyncio
async def test_exists_reports_presence_by_gs_reference() -> None:
    """ADR-210 D3 / ADR-254: the image-regen runner checks the original
    object BEFORE dispatching an edit by reference."""
    client = _FakeMinio({"bkt/tenants/t/old.png": b"x", "bkt/tenants/t/gone.png": b"x"})
    # gone.png is present in the map; remove it to model absence.
    del client._blobs["bkt/tenants/t/gone.png"]
    dl = GcsBlobDownloader(client=client)
    assert await dl.exists("gs://bkt/tenants/t/old.png") is True
    assert await dl.exists("gs://bkt/tenants/t/gone.png") is False
