"""RED: the dispatch outbox rejects a non-UUID workflow_id, and it looks like silence.

``migrations/0003_outbox.sql:39`` declares ``workflow_id UUID NOT NULL``, and
``adapter/pubsub/agent_dispatch.py`` sets the dispatch row's ``workflow_id`` from
``thread_id``.

The OE lane satisfies that by accident of naming: ``oe_grading_crew_wiring.py:16``
records ``thread_id = submission_id``, which is already a UUID.

The growth-edge lane does not. ``checkpointer/factory.py:105``
``build_weakness_thread_id`` returns ``"{tenant_id}:{upload_id}"``, two
colon-separated segments, deliberately NOT a UUID, because the FE resume route
reconstructs the thread from tenant + upload alone (ADR-205 D4/D5).

So the INSERT fails inside the same transaction as the park. The symptom is a
dispatch that never publishes and a run that never resumes, NOT a type error at
the call site. That is expensive to debug from the outside, which is why the
request builder refuses it here instead, where the failure names its own cause.

The crew's sibling writers already establish the right value: both
``weakness_review_pending_outbox_writer.py:151`` and
``weakness_outputs_outbox_writer.py:163`` use ``"workflow_id": upload_id``.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    build_dispatch_request,
)

_TENANT = "11111111-1111-7111-8111-111111111111"
_UPLOAD = "01a02062-e5b4-7870-8fca-53ce363cd542"
_GCID = "00000000-0000-7000-8000-000000001999"


def _weakness_thread_id() -> str:
    """The real 2-segment growth-edge thread key."""
    return f"{_TENANT}:{_UPLOAD}"


def test_a_non_uuid_thread_id_is_refused_at_the_dispatch() -> None:
    """Fail LOUD at the request, not silently at the INSERT.

    Without this the growth-edge conversion produces a lane that publishes
    nothing and parks forever, with no error naming the cause.
    """
    with pytest.raises(ValueError, match="workflow_id"):
        build_dispatch_request(
            agent_role="weakness_diagnose",
            execution_id=f"{_UPLOAD}:diagnose",
            tenant_id=_TENANT,
            gcid=_GCID,
            thread_id=_weakness_thread_id(),
            input_payload="{}",
        )


def test_an_explicit_workflow_id_is_carried_separately_from_the_thread_id() -> None:
    """The growth-edge lane keys the outbox on upload_id, the checkpoint on the thread."""
    req = build_dispatch_request(
        agent_role="weakness_diagnose",
        execution_id=f"{_UPLOAD}:diagnose",
        tenant_id=_TENANT,
        gcid=_GCID,
        thread_id=_weakness_thread_id(),
        workflow_id=_UPLOAD,
        input_payload="{}",
    )

    assert req["workflow_id"] == _UPLOAD, "the outbox row keys on the upload"
    assert req["body"]["thread_id"] == _weakness_thread_id(), (
        "the checkpoint key must stay the 2-segment thread the FE resume "
        "reconstructs, it is NOT interchangeable with workflow_id"
    )


def test_the_oe_shape_is_unchanged_when_no_workflow_id_is_given() -> None:
    """Regression: OE passes no workflow_id and must keep deriving it from the thread.

    OE is LIVE on this lane. A change that altered its workflow_id would rewrite
    what the dispatcher and every downstream trace key on.
    """
    submission_id = "01a02062-e5b4-7870-8fca-53ce363cd542"
    req = build_dispatch_request(
        agent_role="oe_evaluate",
        execution_id=f"{submission_id}:q1",
        tenant_id=_TENANT,
        gcid=_GCID,
        thread_id=submission_id,
        input_payload="{}",
    )

    assert req["workflow_id"] == submission_id
    assert req["body"]["thread_id"] == submission_id
