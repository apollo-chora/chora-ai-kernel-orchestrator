"""Image-upload adapter (W8) — MinIO-backed.

Backs the qgen ``render_image`` node: uploads rendered diagram / scene bytes to
the AI-Kernel transient generation-artifact bucket and mints a short-lived
presigned GET URL the FE fetches directly.

Cloud Storage (Google) is replaced by MinIO (S3-compatible), the local
object store in the compose stack. The ``gs://`` URI scheme is RETAINED as the
canonical object reference (the FE sends ``gs://`` URIs and the source-URI
tenant policy keys on it); the adapter translates ``gs://bucket/object`` to a
MinIO bucket/key.

Storage layout:
  - bucket = ``chora-ai-assist-images-{env}`` (env ``QGEN_IMAGE_GCS_BUCKET``),
    7-day TTL, no public read (these are unreviewed pre-moderation DRAFT
    artifacts).
  - read approach = PRESIGNED GET URL.
  - object key = ``tenants/{tenant_id}/jobs/{job_id}/{artifact_id}.{ext}``.

Per [[secrets-and-env]] the bucket NAME is the only config the service reads —
object keys are computed in code. The MinIO connection (S3_ENDPOINT /
S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY) mirrors chora-common/objectstore.
"""

from chora_ai_kernel_orchestrator.adapter.gcs.image_upload import (
    GcsImageUploadAdapter,
)

__all__ = ["GcsImageUploadAdapter"]
