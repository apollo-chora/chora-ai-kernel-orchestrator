"""CHO-2368 P2 — the operator promotion CLI (the drafts' only entry into the gate).

Pins the load-bearing behaviors:
- create-draft creates a platform plan with agent_id + an auto-bumped
  version_label (CHO-2379) and adds EXACTLY the revision segments (bodies from
  revisions_110);
- request-hitl calls request_hitl_approval (NEVER mark_eval_passed) threading
  eval_run_id + agent/segment ids + a human-readable summary carrying the
  PLAN's own label (CHO-2379), not the embedded revision constant;
- archive drives the guarded ACTIVE -> ARCHIVED edge (the rollback lever);
- the CLI never constructs an outbox dispatcher (single-drain invariant: the
  live pod's dispatcher drains the rows this tool writes).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.domain.prompt_registry import revisions_110 as rev
from chora_ai_kernel_orchestrator.domain.prompt_registry.state_machine import (
    PromptPlanState,
)
from chora_ai_kernel_orchestrator.ops import promote_prompt_plan as cli


class _FakeRepo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        #: Catalogue rows create-draft reads to pick the next free label.
        #: Empty is the golden path (a first attempt takes the base label).
        self.agent_versions: list[Any] = []
        #: version_label carried by the get_plan row.
        self.plan_version_label: str | None = "1.1.0"
        #: True => get_plan returns None (the fail-loud branch).
        self.plan_missing = False

    async def list_agent_versions(self, **kw: Any) -> list[Any]:
        self.calls.append(("list_agent_versions", kw))
        return list(self.agent_versions)

    async def create_draft_plan(self, **kw: Any) -> str:
        self.calls.append(("create_draft_plan", kw))
        return "plan-new-1"

    async def add_segment(self, **kw: Any) -> str:
        self.calls.append(("add_segment", kw))
        return f"seg-{len(self.calls)}"

    async def update_plan_status(self, **kw: Any) -> None:
        self.calls.append(("update_plan_status", kw))

    async def get_plan(self, plan_id: str) -> Any:
        self.calls.append(("get_plan", {"plan_id": plan_id}))
        if self.plan_missing:
            return None
        return SimpleNamespace(
            plan_id=plan_id,
            scope="platform",
            tenant_id=None,
            status="pending_eval",
            agent_id="qgen_question",
            version_label=self.plan_version_label,
        )


class _FakeSvc:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def submit_for_eval(self, plan_id: str) -> None:
        self.calls.append(("submit_for_eval", {"plan_id": plan_id}))

    async def request_hitl_approval(self, plan_id: str, **kw: Any) -> None:
        self.calls.append(("request_hitl_approval", {"plan_id": plan_id, **kw}))


async def test_create_draft_creates_plan_and_all_segments() -> None:
    repo, svc = _FakeRepo(), _FakeSvc()
    plan_id = await cli.run_command(
        ["create-draft", "--agent", "qgen_question", "--operator", "gcid-op"],
        repo=repo,
        svc=svc,
    )
    assert plan_id == "plan-new-1"
    create = next(kw for name, kw in repo.calls if name == "create_draft_plan")
    assert create["agent_id"] == "qgen_question"
    assert create["version_label"] == rev.REVISION_VERSION_LABEL
    assert create["scope"] == "platform"
    assert create["plan_code"] == "qgen_question-1.1.0"
    assert create["created_by"] == "gcid-op"

    seg_calls = [kw for name, kw in repo.calls if name == "add_segment"]
    assert {c["segment_id"] for c in seg_calls} == {"role", "examples"}
    expected = rev.overrides_for("qgen_question")
    for c in seg_calls:
        assert c["plan_id"] == "plan-new-1"
        assert c["agent_id"] == "qgen_question"
        assert c["body"] == expected[c["segment_id"]]
        assert c["note"].strip()
    assert svc.calls == []


async def test_create_draft_first_attempt_keeps_the_base_label() -> None:
    """CHO-2379 golden path: an agent with no 1.1.x plan still mints 1.1.0."""
    repo, svc = _FakeRepo(), _FakeSvc()
    # Only the seeded immutable baseline exists; it is not a promotion attempt.
    repo.agent_versions = [SimpleNamespace(version_label="v1")]
    await cli.run_command(
        ["create-draft", "--agent", "qgen_question", "--operator", "gcid-op"],
        repo=repo,
        svc=svc,
    )
    create = next(kw for name, kw in repo.calls if name == "create_draft_plan")
    assert create["version_label"] == rev.REVISION_VERSION_LABEL
    assert create["plan_code"] == "qgen_question-1.1.0"


async def test_create_draft_auto_bumps_past_the_superseded_attempt() -> None:
    """CHO-2379: ARCHIVED is terminal, so a re-promotion needs a FRESH label.

    Reusing 1.1.0 hides the superseded plan behind the catalogue's
    newest-wins single-version read and collides the eval-run name.
    """
    repo, svc = _FakeRepo(), _FakeSvc()
    repo.agent_versions = [
        SimpleNamespace(version_label="v1"),
        SimpleNamespace(version_label="1.1.0"),
    ]
    await cli.run_command(
        ["create-draft", "--agent", "qgen_question", "--operator", "gcid-op"],
        repo=repo,
        svc=svc,
    )
    listed = next(kw for name, kw in repo.calls if name == "list_agent_versions")
    assert listed == {"agent_id": "qgen_question"}
    create = next(kw for name, kw in repo.calls if name == "create_draft_plan")
    assert create["version_label"] == "1.1.1"
    # plan_code FOLLOWS the label; a stale code would re-introduce the dupe by
    # the other axis.
    assert create["plan_code"] == "qgen_question-1.1.1"


async def test_create_draft_bump_reads_the_catalogue_before_inserting() -> None:
    """The read must precede the write, else the bump is computed off nothing."""
    repo, svc = _FakeRepo(), _FakeSvc()
    repo.agent_versions = [SimpleNamespace(version_label="1.1.2")]
    await cli.run_command(
        ["create-draft", "--agent", "familiar", "--operator", "gcid-op"],
        repo=repo,
        svc=svc,
    )
    names = [name for name, _ in repo.calls]
    assert names.index("list_agent_versions") < names.index("create_draft_plan")
    create = next(kw for name, kw in repo.calls if name == "create_draft_plan")
    assert create["version_label"] == "1.1.3"
    assert create["plan_code"] == "familiar-1.1.3"


async def test_submit_for_eval_delegates_to_service() -> None:
    repo, svc = _FakeRepo(), _FakeSvc()
    await cli.run_command(["submit-for-eval", "--plan-id", "plan-9"], repo=repo, svc=svc)
    assert svc.calls == [("submit_for_eval", {"plan_id": "plan-9"})]


async def test_request_hitl_uses_the_gated_edge_with_run_id() -> None:
    repo, svc = _FakeRepo(), _FakeSvc()
    await cli.run_command(
        [
            "request-hitl",
            "--plan-id",
            "plan-9",
            "--agent",
            "qgen_question",
            "--eval-run-id",
            "cho2368-qgen-question-prompt-1-1-0-r1",
            "--requester",
            "gcid-op",
        ],
        repo=repo,
        svc=svc,
    )
    name, kw = svc.calls[0]
    assert name == "request_hitl_approval"
    assert kw["plan_id"] == "plan-9"
    assert kw["eval_run_id"] == "cho2368-qgen-question-prompt-1-1-0-r1"
    assert kw["requester_gcid"] == "gcid-op"
    assert kw["scope"] == "platform"
    # A platform plan has no tenant, but the gate must still land in a
    # tenant's O+ queue (outbox tenant_id is UUID NOT NULL and
    # /api/hitl/pending filters by tenant), so it routes to the platform
    # tenant's queue where the operator reviews it.
    assert kw["tenant_id"] == "00000000-0000-7000-8000-000000000001"
    assert kw["agent_ids"] == ["qgen_question"]
    assert set(kw["segment_ids"]) == {"role", "examples"}
    assert "1.1.0" in kw["summary"] and "qgen_question" in kw["summary"]


async def test_request_hitl_summary_carries_the_plans_own_label() -> None:
    """CHO-2379: the O+ gate card must name the plan actually being approved.

    The summary used to interpolate the embedded revision CONSTANT, so every
    gate card read 1.1.0 no matter which plan was on the gate.
    """
    repo, svc = _FakeRepo(), _FakeSvc()
    repo.plan_version_label = "1.1.2"
    await cli.run_command(
        [
            "request-hitl",
            "--plan-id",
            "plan-9",
            "--agent",
            "qgen_question",
            "--eval-run-id",
            "cho2379-run-a",
            "--requester",
            "gcid-op",
        ],
        repo=repo,
        svc=svc,
    )
    assert ("get_plan", {"plan_id": "plan-9"}) in repo.calls
    _, kw = svc.calls[0]
    assert "1.1.2" in kw["summary"]
    assert rev.REVISION_VERSION_LABEL not in kw["summary"]


async def test_request_hitl_falls_back_to_the_revision_label_when_unlabelled() -> None:
    """A legacy plan row predating the label column still gates."""
    repo, svc = _FakeRepo(), _FakeSvc()
    repo.plan_version_label = None
    await cli.run_command(
        [
            "request-hitl",
            "--plan-id",
            "plan-9",
            "--agent",
            "qgen_question",
            "--eval-run-id",
            "cho2379-run-b",
            "--requester",
            "gcid-op",
        ],
        repo=repo,
        svc=svc,
    )
    _, kw = svc.calls[0]
    assert rev.REVISION_VERSION_LABEL in kw["summary"]


async def test_request_hitl_fails_loud_on_a_missing_plan() -> None:
    """No silent gate on a plan_id typo: refuse before touching the service."""
    repo, svc = _FakeRepo(), _FakeSvc()
    repo.plan_missing = True
    with pytest.raises(ValueError, match="plan-missing"):
        await cli.run_command(
            [
                "request-hitl",
                "--plan-id",
                "plan-missing",
                "--agent",
                "qgen_question",
                "--eval-run-id",
                "cho2379-run-c",
                "--requester",
                "gcid-op",
            ],
            repo=repo,
            svc=svc,
        )
    assert svc.calls == []


async def test_archive_drives_the_guarded_edge() -> None:
    repo, svc = _FakeRepo(), _FakeSvc()
    await cli.run_command(["archive", "--plan-id", "plan-9"], repo=repo, svc=svc)
    name, kw = repo.calls[-1]
    assert name == "update_plan_status"
    assert kw["expected_from"] == PromptPlanState.ACTIVE
    assert kw["to"] == PromptPlanState.ARCHIVED


async def test_unknown_agent_fails_loud() -> None:
    # argparse `choices` is the fail-loud gate: an agent outside the wave
    # (the familiar JOINED in P3, so use a genuinely unknown id) exits before
    # any repo call.
    repo, svc = _FakeRepo(), _FakeSvc()
    with pytest.raises(SystemExit):
        await cli.run_command(
            ["create-draft", "--agent", "weakness_analyser", "--operator", "g"],
            repo=repo,
            svc=svc,
        )
    assert repo.calls == [] and svc.calls == []


def test_module_never_imports_the_dispatcher() -> None:
    import inspect

    src = inspect.getsource(cli)
    assert "OutboxDispatcher" not in src, "single-drain invariant: the live pod drains"
