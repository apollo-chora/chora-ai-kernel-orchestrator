"""MinIO blob downloader — satisfies the weakness-analyser runner's
``BlobDownloader`` port. Downloads the uploaded weakness document
(``gs://bucket/path``) to bytes so the runner can stream it to the model
gateway as a multimodal ``inlineData`` part.

Cloud Storage (Google) is replaced by MinIO (S3-compatible). The ``gs://``
scheme is retained as the canonical object reference; the adapter translates
it to a MinIO bucket/key. The minio client is lazily built from env
(S3_ENDPOINT / S3_ACCESS_KEY_ID / S3_ACCESS_KEY) so unit tests inject a fake
client and never drag the dep in. ``download_as_bytes`` is blocking, so it
runs on a worker thread via ``asyncio.to_thread`` to avoid stalling the
event loop.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from chora_ai_kernel_orchestrator.adapter.gcs.image_upload import (
    ENV_S3_ACCESS_KEY,
    ENV_S3_ENDPOINT,
    ENV_S3_SECRET_KEY,
    ENV_S3_SECURE,
    _env_flag,
)


def parse_gs_uri(gs_uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/object/path`` or ``s3://bucket/object/path`` into
    ``(bucket, object_path)``. Raises ``ValueError`` for anything that is not a
    well-formed reference with a non-empty bucket and object path.

    ``gs://`` is the legacy GCP spelling; the cloud-neutral deployment emits
    ``s3://`` for the same object-store location.
    """
    text = (gs_uri or "").strip()
    for scheme in ("gs://", "s3://"):
        if text.startswith(scheme):
            rest = text[len(scheme) :]
            bucket, sep, path = rest.partition("/")
            if not bucket or not sep or not path:
                raise ValueError(f"{scheme} URI missing bucket or object path: {gs_uri!r}")
            return bucket, path
    raise ValueError(f"not a gs:// or s3:// URI: {gs_uri!r}")


class GcsBlobDownloader:
    """Downloads a gs:// blob to bytes. ``client`` is injectable for tests;
    production lazily builds a Minio client from env."""

    def __init__(self, *, client: Any = None) -> None:
        self._client = client

    async def download(self, gs_uri: str) -> bytes:
        bucket_name, object_path = parse_gs_uri(gs_uri)
        client = self._resolve_client()
        response = client.get_object(bucket_name, object_path)
        try:
            return await asyncio.to_thread(response.read)
        finally:
            response.close()
            response.release_conn()

    async def exists(self, gs_uri: str) -> bool:
        """Cheap existence check by gs:// reference (ADR-210 D3: an image EDIT
        whose original is gone is refused with an explicit author message
        BEFORE any dispatch, instead of being discovered by the renderer after
        a paid round trip)."""
        bucket_name, object_path = parse_gs_uri(gs_uri)
        client = self._resolve_client()
        return bool(await asyncio.to_thread(lambda: _stat(client, bucket_name, object_path)))

    def _resolve_client(self) -> Any:
        if self._client is not None:
            return self._client
        from minio import Minio  # lazy — keeps the test path SDK-free

        endpoint = os.getenv(ENV_S3_ENDPOINT, "").strip()
        if not endpoint:
            raise RuntimeError("S3_ENDPOINT not set; GcsBlobDownloader requires it (declared in pyproject.toml).")
        access_key = os.getenv(ENV_S3_ACCESS_KEY, "").strip()
        secret_key = os.getenv(ENV_S3_SECRET_KEY, "").strip()
        secure = _env_flag(ENV_S3_SECURE)
        if secure is None:
            secure = False
        self._client = Minio(
            endpoint,
            access_key=access_key or None,
            secret_key=secret_key or None,
            secure=secure,
        )
        return self._client


def _stat(client: Any, bucket_name: str, object_path: str) -> bool:
    try:
        client.stat_object(bucket_name, object_path)
        return True
    except Exception:
        # fail-loud-exempt: exists() is a best-effort pre-check where "not
        # found" (the common case) IS the return value; a real transport error
        # is surfaced loud by the subsequent download, which fail-louds.
        return False


__all__ = ["GcsBlobDownloader", "parse_gs_uri"]
