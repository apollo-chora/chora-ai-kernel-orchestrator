"""AUDIT-G3: the qgen checkpoint must not serialise UNREGISTERED domain types.

LangGraph warns "Deserializing unregistered type ... This will be blocked in a
future version." Today it is a warning; on a LangGraph upgrade it becomes an
error, and it would break resume-from-checkpoint for EVERY parked qgen job,
which is exactly the durability ADR-251 D4 exists to provide.

⚠ THE TWO TYPES THAT WARNED ARE A SAMPLE, NOT THE POPULATION. A live drive
warned for GuardrailResult and CandidatePayload only, but CritiqueResult is the
same kind of frozen dataclass on the same graph state and simply did not happen
to deserialise in that window. Registering only the two observed leaves the
third to break on the same upgrade, so this pins ALL THREE.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from chora_ai_kernel_orchestrator.adapter.checkpointer.qgen_serde import (
    QGEN_MSGPACK_MODULES,
    qgen_checkpoint_serde,
)
from chora_ai_kernel_orchestrator.domain.qgen_crew.state import (
    CandidatePayload,
    CritiqueResult,
    GuardrailResult,
)

SAMPLES = (
    CandidatePayload(stem="s", question_type="mcq", payload_json="{}", critic_notes="n"),
    CritiqueResult(accepted=True, critique_notes="ok", suggested_revisions=["a", "b"]),
    GuardrailResult(allowed=False, armor_verdict="armor:pii_high_risk_block", user_facing_message="blocked"),
)


@pytest.mark.parametrize("obj", SAMPLES, ids=lambda o: type(o).__name__)
def test_every_qgen_state_type_round_trips_under_a_strict_serde(obj) -> None:
    serde = qgen_checkpoint_serde()
    assert serde.loads_typed(serde.dumps_typed(obj)) == obj


@pytest.mark.parametrize("obj", SAMPLES, ids=lambda o: type(o).__name__)
def test_negative_control_without_the_allowlist_the_type_is_lost_not_raised(obj) -> None:
    """Proves the allowlist is what makes the test above pass, AND pins the real
    failure mode, which is worse than the warning implies.

    A blocked deserialisation does NOT raise. LangGraph logs and hands back a
    PLAIN DICT carrying the same fields. So on the upgrade that enforces this, a
    resumed qgen job does not fail at boot: it resumes with `dict` where the
    graph expects a dataclass, and dies later on attribute access, or silently
    reads a default through a getattr. That is a fail-quiet, which is why this
    is registered ahead of the upgrade rather than after it.
    """
    bare = JsonPlusSerializer(allowed_msgpack_modules=())
    back = bare.loads_typed(bare.dumps_typed(obj))
    assert isinstance(back, dict), "expected the degraded plain-dict form"
    assert not isinstance(back, type(obj))
    assert back != obj
    # the DATA survives; only the TYPE is lost, which is what makes it quiet
    assert back == vars(obj)


def test_the_allowlist_covers_all_three_and_names_them_by_module() -> None:
    assert set(QGEN_MSGPACK_MODULES) == {
        ("chora_ai_kernel_orchestrator.domain.qgen_crew.state", "CandidatePayload"),
        ("chora_ai_kernel_orchestrator.domain.qgen_crew.state", "CritiqueResult"),
        ("chora_ai_kernel_orchestrator.domain.qgen_crew.state", "GuardrailResult"),
    }


# ── the wiring guard: a serde nobody applies is a config file with no reader ──


class _FakeInner:
    """Stands in for the upstream PostgresSaver, carrying a STRICT serde so the
    merge is observable. Under the permissive default the merge is a deliberate
    no-op, which would make this test vacuous."""

    def __init__(self) -> None:
        self.serde = JsonPlusSerializer(allowed_msgpack_modules=())


class _FakeCM:
    def __init__(self, inner: _FakeInner) -> None:
        self._inner = inner

    def __enter__(self) -> _FakeInner:
        return self._inner

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeBuilder:
    def __init__(self, inner: _FakeInner) -> None:
        self._inner = inner

    def from_conn_string(self, dsn: str) -> _FakeCM:
        return _FakeCM(self._inner)


def test_the_proxy_applies_the_qgen_allowlist_to_the_serde_it_adopts() -> None:
    """The adopted serde must carry our types, not merely exist.

    Without this the allowlist could be defined, exported, tested in isolation
    and never reach the object that actually decodes a checkpoint.
    """
    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        LazyPostgresSaver,
    )

    inner = _FakeInner()
    proxy = LazyPostgresSaver(dsn="postgresql://x/y", builder=_FakeBuilder(inner))
    proxy.open()

    obj = GuardrailResult(allowed=True, armor_verdict="v", user_facing_message="m")
    # the inner's own strict serde LOSES the type (negative control)
    assert isinstance(inner.serde.loads_typed(inner.serde.dumps_typed(obj)), dict)
    # the proxy's adopted serde must NOT
    assert proxy.serde.loads_typed(proxy.serde.dumps_typed(obj)) == obj
