"""RED: a caller-supplied source image URI must be pinned to the caller's own
bucket and tenant prefix before it is ever dispatched.

The defect this closes (found by WP-A on the qgen_render adversarial rows,
verified hop by hop by the coordinator, routed here 2026-08-23):

``original_image_gcs_uri`` arrives on the wire in the ai_assist_started request
(proto_wire field 8). The kennel's only check was an existence probe,
``_original_available`` -> ``image_downloader.exists(gs_uri)``: no bucket
allowlist, no tenant prefix, no binding to the draft or job the URI claims to
belong to. It was then passed through unchanged as ``source_image_uri`` on the
qgen_render dispatch, and the agent side validated the SCHEME only before
reading the object from whatever bucket the URI named. The render GSA holds
project-level ``roles/storage.objectViewer``, so IAM did not contain it either.

Net effect: an authenticated author who could shape a regen request had a
cross-tenant read primitive with an image-to-image output channel. The WRITE
path was always partitioned (``tenants/{tenant}/jobs/{job}/...``); the read path
was not partitioned at all. Worse, the existence probe CONFIRMED readability
before proceeding, so it doubled as a discovery oracle.

Why keying the constraint on the payload's ``tenant_id`` is sound rather than
circular, which is the trap in a fix like this: the two fields do NOT have the
same provenance despite riding the same message. ``original_image_gcs_uri`` is
echoed from the client's request body. ``tenant_id`` is stamped by
chora-creation from the authenticated request context
(``tenantFromContext(r.Context())`` / ``tenantAndGCIDFromRequest``), so it is
server-derived upstream. An attacker shapes the URI; they do not choose the
tenant they run as.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.adapter.gcs.source_uri_policy import (
    SourceUriNotPermittedError,
    require_tenant_scoped_source_uri,
)

_BUCKET = "chora-ai-assist-images-dev"
_TENANT = "11111111-1111-7111-8111-111111111111"
_OTHER = "22222222-2222-7222-8222-222222222222"


def _ok(uri: str) -> str:
    return require_tenant_scoped_source_uri(uri, bucket=_BUCKET, tenant_id=_TENANT)


def _refused(uri: str) -> str:
    with pytest.raises(SourceUriNotPermittedError) as excinfo:
        require_tenant_scoped_source_uri(uri, bucket=_BUCKET, tenant_id=_TENANT)
    return str(excinfo.value)


# -----------------------------------------------------------------------------
# What must be accepted
# -----------------------------------------------------------------------------


def test_the_tenants_own_object_in_the_render_bucket_is_accepted() -> None:
    uri = f"gs://{_BUCKET}/tenants/{_TENANT}/jobs/job-1/abc.png"
    assert _ok(uri) == uri


def test_any_job_of_the_same_tenant_is_accepted() -> None:
    """The constraint is the TENANT prefix, not the job: a learner legitimately
    edits an image produced by an earlier job of theirs."""
    uri = f"gs://{_BUCKET}/tenants/{_TENANT}/jobs/some-older-job/def.png"
    assert _ok(uri) == uri


# -----------------------------------------------------------------------------
# What must be refused
# -----------------------------------------------------------------------------


def test_another_tenants_object_is_refused() -> None:
    """The whole point. Same bucket, wrong tenant.

    Note what is NOT asserted: that the message says "tenant". The first draft
    of this test did assert that, and it was the test that was wrong. A refusal
    naming the reason tells a prober whether they got the bucket right and only
    missed the tenant, which is half the oracle back. The refusals are
    deliberately uniform; see test_the_refusal_never_echoes_the_rejected_uri.
    """
    _refused(f"gs://{_BUCKET}/tenants/{_OTHER}/jobs/job-1/abc.png")


def test_another_bucket_is_refused_even_with_a_matching_tenant_prefix() -> None:
    """The GSA can read any bucket in the project, so the bucket has to be
    pinned too: a matching-looking prefix in someone else's bucket is exactly
    the shape an attacker would construct."""
    _refused(f"gs://chora-atom-media/tenants/{_TENANT}/jobs/job-1/abc.png")


def test_a_bucket_whose_name_merely_starts_with_the_render_bucket_is_refused() -> None:
    """``gs://bucket-evil/...`` must not pass a startswith check on
    ``gs://bucket``. Prefix comparisons on names are how this class of fix
    usually leaks."""
    _refused(f"gs://{_BUCKET}-evil/tenants/{_TENANT}/jobs/job-1/abc.png")


def test_a_tenant_prefix_that_merely_starts_with_the_tenant_id_is_refused() -> None:
    """Same failure mode one segment down: ``tenants/{tenant}-other/`` must not
    satisfy a startswith on ``tenants/{tenant}``."""
    _refused(f"gs://{_BUCKET}/tenants/{_TENANT}-other/jobs/job-1/abc.png")


def test_traversal_out_of_the_tenant_prefix_is_refused() -> None:
    _refused(f"gs://{_BUCKET}/tenants/{_TENANT}/../{_OTHER}/jobs/j/abc.png")


def test_a_non_gs_scheme_is_refused() -> None:
    _refused(f"https://storage.googleapis.com/{_BUCKET}/tenants/{_TENANT}/a.png")
    _refused("file:///etc/passwd")
    _refused(f"//{_BUCKET}/tenants/{_TENANT}/a.png")


def test_an_object_directly_under_the_bucket_is_refused() -> None:
    """No tenant segment at all."""
    _refused(f"gs://{_BUCKET}/abc.png")


def test_a_uri_that_is_only_the_bucket_is_refused() -> None:
    _refused(f"gs://{_BUCKET}")
    _refused(f"gs://{_BUCKET}/")


def test_an_empty_or_whitespace_uri_is_refused() -> None:
    _refused("")
    _refused("   ")


def test_a_missing_tenant_id_refuses_rather_than_matching_everything() -> None:
    """An empty tenant must never widen the prefix to ``tenants/``. Fail closed:
    the caller cannot end up with a constraint that admits every tenant."""
    with pytest.raises(SourceUriNotPermittedError):
        require_tenant_scoped_source_uri(
            f"gs://{_BUCKET}/tenants/{_TENANT}/jobs/j/a.png",
            bucket=_BUCKET,
            tenant_id="",
        )


def test_a_missing_bucket_refuses_rather_than_matching_everything() -> None:
    with pytest.raises(SourceUriNotPermittedError):
        require_tenant_scoped_source_uri(
            f"gs://{_BUCKET}/tenants/{_TENANT}/jobs/j/a.png",
            bucket="   ",
            tenant_id=_TENANT,
        )


def test_the_refusal_never_echoes_the_rejected_uri() -> None:
    """The refusal reaches the author. Echoing the URI back would confirm the
    shape they probed with, which is the oracle this fix exists to remove."""
    hostile = f"gs://someone-elses-bucket/tenants/{_OTHER}/jobs/secret-job/x.png"
    msg = _refused(hostile)
    assert "someone-elses-bucket" not in msg
    assert _OTHER not in msg
    assert "secret-job" not in msg
