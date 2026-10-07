"""``GcsImageUploadAdapter`` — upload rendered image bytes → presigned GET URL.

The qgen ``render_image`` node uploads each rendered artifact here and stamps
the returned presigned URL onto the candidate (``image_url`` /
``answer_image_url``).

Cloud Storage (Google) is replaced by MinIO (S3-compatible), the local
object store in the compose stack. The ``gs://`` URI scheme is RETAINED as the
canonical object reference — the FE sends ``gs://`` URIs and the source-URI
tenant policy keys on it — but the bytes now live in MinIO. The adapter
translates ``gs://bucket/object`` to a MinIO bucket/key.

Object key (per the storage layout):
    tenants/{tenant_id}/jobs/{job_id}/{artifact_id}.{ext}

Read approach = PRESIGNED GET URL. The bucket has no public read (drafts are
pre-moderation), so the FE fetches via a short-lived presigned GET.

The minio client is lazily built from env (S3_ENDPOINT / S3_ACCESS_KEY_ID /
S3_SECRET_ACCESS_KEY, the same convention as the Go objectstore) so unit tests
inject a fake client (constructor dep) and never drag the dep in.

Per [[secrets-and-env]] the bucket NAME comes from ``QGEN_IMAGE_GCS_BUCKET``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from io import BytesIO
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

ENV_QGEN_IMAGE_GCS_BUCKET = "QGEN_IMAGE_GCS_BUCKET"

# MinIO / S3 connection env (mirrors chora-common/objectstore's convention).
ENV_S3_ENDPOINT = "S3_ENDPOINT"
ENV_S3_ACCESS_KEY = "S3_ACCESS_KEY_ID"
ENV_S3_SECRET_KEY = "S3_SECRET_ACCESS_KEY"
ENV_S3_SECURE = "S3_SECURE"

# Presigned-URL maximum (7 days). Persisted question/answer images must survive
# the author→test-set→assessment→learner-take journey; a 15-min link would 404
# before a learner reaches the assessment.
_SIGNED_URL_EXPIRY_SECONDS = 604800

# MIME → file extension for the object key suffix.
_MIME_EXT = {
    "image/png": "png",
    "image/svg+xml": "svg",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}
_DEFAULT_EXT = "png"

_TRUTHY = {"1", "true", "yes"}


class GcsImageUploadAdapter:
    """Uploads bytes to the transient image bucket + mints a presigned GET URL.

    ``client`` is injectable for tests; production lazily builds a Minio
    client from env.
    """

    def __init__(
        self,
        *,
        bucket_name: str,
        client: Any | None = None,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool | None = None,
    ) -> None:
        if not (bucket_name or "").strip():
            raise ValueError(
                f"GcsImageUploadAdapter requires a non-empty bucket "
                f"(set {ENV_QGEN_IMAGE_GCS_BUCKET} per [[secrets-and-env]])"
            )
        self._bucket_name = bucket_name.strip()
        self._client = client
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = secure

    @classmethod
    def from_env(cls) -> GcsImageUploadAdapter | None:
        """Build from ``QGEN_IMAGE_GCS_BUCKET`` + the S3/MinIO env. None when
        the bucket is unset."""
        bucket = os.getenv(ENV_QGEN_IMAGE_GCS_BUCKET, "").strip()
        if not bucket:
            return None
        return cls(
            bucket_name=bucket,
            endpoint=os.getenv(ENV_S3_ENDPOINT, "").strip() or None,
            access_key=os.getenv(ENV_S3_ACCESS_KEY, "").strip() or None,
            secret_key=os.getenv(ENV_S3_SECRET_KEY, "").strip() or None,
            secure=_env_flag(ENV_S3_SECURE),
        )

    @property
    def bucket_name(self) -> str:
        return self._bucket_name

    async def upload_and_sign(self, *, tenant_id: str, job_id: str, data: bytes, content_type: str) -> tuple[str, str]:
        """Upload ``data`` under a tenant/job-scoped key + return
        ``(gs_uri, signed_url)``:

          * ``gs_uri`` — the canonical ``gs://`` OBJECT path of the uploaded
            blob (ADR-210 B2). The producer PERSISTS this on the candidate so
            image-to-image regen reads the object directly instead of parsing
            the (transient) signed URL back into a ``gs://``.
          * ``signed_url`` — a presigned GET URL for the author's ``<img>``.

        Synchronous SDK calls run on a thread to keep the event loop free.
        Raises on any SDK error (the render_image node treats that as a
        fail-soft render failure)."""
        ext = _MIME_EXT.get((content_type or "").lower(), _DEFAULT_EXT)
        key = f"tenants/{tenant_id}/jobs/{job_id}/{uuid4().hex}.{ext}"
        return await asyncio.to_thread(self._upload_and_sign_sync, key, data, content_type)

    async def sign_read_url(self, gs_uri: str) -> str:
        """Mint a presigned GET URL for an EXISTING object by ``gs://`` reference
        (ADR-254 D12: the qgen_render agent writes the scene image and returns
        its object; the kennel keeps the signing step). The bucket comes from
        the URI, not from this adapter's upload bucket, so a renderer writing to
        its own bucket still signs. Refuses anything that is not
        ``gs://bucket/object``. Never uploads."""
        bucket_name, key = _split_gs_uri(gs_uri)
        return await asyncio.to_thread(self._sign_sync, bucket_name, key)

    def _sign_sync(self, bucket_name: str, key: str) -> str:
        client = self._resolve_client()
        return str(client.presigned_get_object(bucket_name, key, expires=_SIGNED_URL_EXPIRY_SECONDS))

    def _upload_and_sign_sync(self, key: str, data: bytes, content_type: str) -> tuple[str, str]:
        client = self._resolve_client()
        client.put_object(
            self._bucket_name,
            key,
            BytesIO(data),
            length=len(data),
            content_type=content_type or "image/png",
        )
        gs_uri = f"gs://{self._bucket_name}/{key}"
        signed = client.presigned_get_object(self._bucket_name, key, expires=_SIGNED_URL_EXPIRY_SECONDS)
        return gs_uri, str(signed)

    def _resolve_client(self) -> Any:
        if self._client is not None:
            return self._client
        from minio import Minio  # lazy — keeps the test path SDK-free

        endpoint = self._endpoint or os.getenv(ENV_S3_ENDPOINT, "").strip()
        if not endpoint:
            raise RuntimeError("S3_ENDPOINT not set; GcsImageUploadAdapter requires it (declared in pyproject.toml).")
        access_key = self._access_key or os.getenv(ENV_S3_ACCESS_KEY, "").strip()
        secret_key = self._secret_key or os.getenv(ENV_S3_SECRET_KEY, "").strip()
        secure = self._secure
        # The deployment sets S3_ENDPOINT with a scheme (`http://minio:9000`),
        # which the Go objectstore accepts; the minio SDK wants a bare
        # `host:port` and rejects the URL form inside BaseURL. Normalise here and
        # let the scheme decide TLS when S3_SECURE is not set explicitly.
        if "://" in endpoint:
            scheme, _, hostport = endpoint.partition("://")
            if secure is None:
                secure = scheme.strip().lower() == "https"
            endpoint = hostport.strip("/")
        if secure is None:
            secure = os.getenv(ENV_S3_SECURE, "").strip().lower() in _TRUTHY
        self._client = Minio(
            endpoint,
            access_key=access_key or None,
            secret_key=secret_key or None,
            secure=secure,
        )
        return self._client


def _env_flag(name: str) -> bool | None:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return None
    return raw in _TRUTHY


def _split_gs_uri(gs_uri: str) -> tuple[str, str]:
    """``gs://bucket/object`` or ``s3://bucket/object`` -> (bucket, object).

    Both schemes name the same object-store location. ``gs://`` is the legacy
    GCP spelling; the cloud-neutral deployment's MinIO/S3 adapters (and the
    qgen_render agent's S3ObjectStore) emit ``s3://``, so a reader that only
    accepted ``gs://`` rejected every render it was handed.
    """
    text = (gs_uri or "").strip()
    for scheme in ("gs://", "s3://"):
        if text.startswith(scheme):
            rest = text[len(scheme) :]
            bucket, _, key = rest.partition("/")
            if not bucket or not key:
                raise ValueError(
                    f"sign_read_url: {scheme} reference needs a bucket and an object: {text[:120]!r}"
                )
            return bucket, key
    raise ValueError(f"sign_read_url: not a gs:// or s3:// reference: {text[:120]!r}")


__all__ = [
    "ENV_QGEN_IMAGE_GCS_BUCKET",
    "ENV_S3_ACCESS_KEY",
    "ENV_S3_ENDPOINT",
    "ENV_S3_SECRET_KEY",
    "ENV_S3_SECURE",
    "GcsImageUploadAdapter",
]
