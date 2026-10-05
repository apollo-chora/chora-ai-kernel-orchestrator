"""Pin a caller-supplied ``gs://`` source image URI to the caller's own objects.

The image-regen request carries ``original_image_gcs_uri`` from the client. It
used to reach the qgen_render dispatch unchanged, checked only by an existence
probe, and the agent validated the SCHEME before reading the object from
whatever bucket the URI named. Since the render GSA holds project-level
``roles/storage.objectViewer``, neither the kennel nor IAM contained it: an
authenticated author who could shape a regen request had a cross-tenant read
with an image-to-image output channel.

The write path was never the problem. Objects are written under
``tenants/{tenant_id}/jobs/{job_id}/{artifact}.{ext}`` (see
``adapter/gcs/image_upload.py``), so "the caller's own objects" already has an
exact definition. This module applies that same definition on the way IN.

On provenance, because a constraint keyed on the wrong field would be circular:
``original_image_gcs_uri`` is echoed from the client's request body, but
``tenant_id`` is stamped by chora-creation from the authenticated request
context before the event is published. An attacker shapes the URI; they do not
choose the tenant they run as. That asymmetry is what makes the tenant prefix a
real boundary rather than a restatement of the attacker's own input.

Fail CLOSED throughout: an absent bucket or tenant refuses rather than widening
the constraint to ``tenants/`` (which would admit every tenant), and every
comparison is on whole path SEGMENTS, never a string prefix, so neither
``{bucket}-evil`` nor ``tenants/{tenant}-other`` can satisfy it.
"""

from __future__ import annotations

_SCHEME = "gs://"


class SourceUriNotPermittedError(ValueError):
    """A caller-supplied source URI is not one of this tenant's own objects.

    Deliberately carries no detail about what was rejected. The message reaches
    the author, and echoing the URI (or the bucket, or the tenant it named)
    would confirm the shape they probed with, which is the oracle this module
    exists to remove.
    """


def require_tenant_scoped_source_uri(uri: str, *, bucket: str, tenant_id: str) -> str:
    """Return ``uri`` when it names an object this tenant owns, else refuse.

    The URI must be ``gs://{bucket}/tenants/{tenant_id}/...`` with at least one
    object path segment below the tenant prefix, matched segment by segment.
    """
    bucket = (bucket or "").strip()
    tenant_id = (tenant_id or "").strip()
    # Fail closed. An empty constraint must never mean "anything matches": that
    # is how a misconfigured deployment silently reopens the hole.
    if not bucket:
        raise SourceUriNotPermittedError("the image bucket is not configured")
    if not tenant_id:
        raise SourceUriNotPermittedError("the request carries no tenant")

    candidate = (uri or "").strip()
    if not candidate.startswith(_SCHEME):
        raise SourceUriNotPermittedError("the source image reference is not supported")

    remainder = candidate[len(_SCHEME) :]
    segments = remainder.split("/")
    # gs://{bucket}/tenants/{tenant}/{at least one object segment}
    if len(segments) < 4:
        raise SourceUriNotPermittedError("the source image reference is not supported")

    # Whole-segment equality, never startswith: `{bucket}-evil` and
    # `tenants/{tenant}-other` are the two shapes a prefix check would admit.
    if segments[0] != bucket:
        raise SourceUriNotPermittedError("the source image is not one of your images")
    if segments[1] != "tenants" or segments[2] != tenant_id:
        raise SourceUriNotPermittedError("the source image is not one of your images")

    # No traversal or empty segments anywhere below the prefix: `..` would climb
    # straight back out of the tenant prefix the segments above just pinned.
    object_segments = segments[3:]
    if any(seg in ("", ".", "..") for seg in object_segments):
        raise SourceUriNotPermittedError("the source image reference is not supported")

    return candidate


__all__ = ["SourceUriNotPermittedError", "require_tenant_scoped_source_uri"]
