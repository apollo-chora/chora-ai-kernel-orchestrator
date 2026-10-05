"""companion_diagnosis crew composition root on the bus (ADR-254 D2/D5/D12).

The live SDK/DB composition (``_build_graph_live``) is integration glue
(pragma); these verify the env guards (None => the lifespan aborts the lane),
the STRICT template resolution, and the pure assembly seam
(``_assemble_graph_components``): the crew is built on the Pub/Sub extractor +
diagnoser (one executor pinned to the two roles, the diagnoser also the output
task runner), its roles (plus the legacy drain role) are registered on the
kennel runtime, no model-gateway client is constructed anywhere.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyser_crew_wiring as wiring_mod
import chora_ai_kernel_orchestrator.adapter.secrets as secrets_mod
from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyser_crew_graph_subscriber import (
    WeaknessAnalyserCrewGraphSubscriber,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.weakness_analyser_crew_wiring import (
    COMPLETION_ROLES,
    DISPATCH_ROLES,
    WEAKNESS_CREW,
    WeaknessAnalyserCrewComponents,
    _assemble_graph_components,
    _resolve_practice_test_max_questions,
    _resolve_weakness_guardrail_template,
    build_weakness_analyser_crew_from_env,
)
from chora_ai_kernel_orchestrator.adapter.weakness.pubsub_dispatch import (
    PubSubDiagnoserAdapter,
    PubSubExtractorAdapter,
)
from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew_runner import (
    WeaknessAnalyserCrewRunner,
)


class _FakeScreener:
    async def sanitize_user_prompt(self, req: Any) -> Any:  # pragma: no cover
        raise AssertionError("screener not exercised in unit test")

    async def sanitize_model_response(self, req: Any) -> Any:  # pragma: no cover
        raise AssertionError("screener not exercised in unit test")

    async def close(self) -> None:  # pragma: no cover
        ...


class _FakeConn:
    def cursor(self) -> Any:  # pragma: no cover - never called
        raise AssertionError("cursor should not be exercised in unit test")


class _FakeOutPublisher:
    async def publish(self, **_: Any) -> str:  # pragma: no cover
        return ""


class _FakeRuntime:
    """Records the lane registration the wiring must make."""

    def __init__(self) -> None:
        self.lanes: list[dict[str, Any]] = []

    def register_lane(self, name: str, *, crew: str, roles: Any, runner: Any) -> None:
        self.lanes.append({"name": name, "crew": crew, "roles": tuple(roles), "runner": runner})


TEMPLATE = "projects/p/locations/l/templates/chora-guardrail-strict-dev"


# --------------------------------------------------------------------------- #
# env guards
# --------------------------------------------------------------------------- #


async def test_none_when_pubsub_project_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NATS_URL", raising=False)
    assert await build_weakness_analyser_crew_from_env(screener=_FakeScreener(), runtime=_FakeRuntime()) is None


async def test_the_kennel_runtime_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NATS_URL", "nats://nats:4222")
    with pytest.raises(RuntimeError, match="runtime"):
        await build_weakness_analyser_crew_from_env(screener=_FakeScreener(), runtime=None)


async def test_none_when_screener_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NATS_URL", "nats://nats:4222")
    assert await build_weakness_analyser_crew_from_env(screener=None, runtime=_FakeRuntime()) is None


async def test_none_when_template_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NATS_URL", "nats://nats:4222")
    monkeypatch.setattr(wiring_mod, "_resolve_weakness_guardrail_template", lambda screener: None)
    assert await build_weakness_analyser_crew_from_env(screener=_FakeScreener(), runtime=_FakeRuntime()) is None


async def test_none_when_dsn_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NATS_URL", "nats://nats:4222")
    monkeypatch.setattr(wiring_mod, "_resolve_weakness_guardrail_template", lambda screener: TEMPLATE)
    monkeypatch.setattr(secrets_mod, "resolve_dsn", lambda: "")
    assert await build_weakness_analyser_crew_from_env(screener=_FakeScreener(), runtime=_FakeRuntime()) is None


def test_no_gateway_client_is_imported_by_the_wiring() -> None:
    """ADR-254 D5 deterministic kernel: the lane constructs no model-gateway
    client; every model call is a dispatch."""
    import inspect

    src = inspect.getsource(wiring_mod)
    assert "ModelGatewayMultimodalClient" not in src
    assert "ModelGatewayTextClient" not in src
    assert "ModelGatewayImageClient" not in src
    # the retired variables are named in the module docstring as history, but
    # never READ: no getenv on any of them
    for var in (
        "MODEL_GATEWAY_GRPC_TARGET",
        "WEAKNESS_DISPATCH_TRANSPORT",
        "WEAKNESS_CREW_MODE",
        "WEAKNESS_DIAGNOSER_ENGINE_RESOURCE",
    ):
        assert f'getenv("{var}")' not in src and f"getenv('{var}')" not in src, var
        assert f'os.environ["{var}"]' not in src, var


# --------------------------------------------------------------------------- #
# config resolution
# --------------------------------------------------------------------------- #


def test_resolve_weakness_guardrail_template_is_strict() -> None:
    template = _resolve_weakness_guardrail_template(_FakeScreener())
    assert template is not None
    assert "/templates/chora-guardrail-strict-" in template


def test_template_resolution_uses_canonical_crew_agent_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class _SpyResolver:
        def template_for(self, agent_id: str) -> str:
            captured["agent_id"] = agent_id
            return TEMPLATE

    class _SpyPort:
        resolver = _SpyResolver()

        @classmethod
        def from_env(cls, *, screener: Any) -> _SpyPort:
            return cls()

    monkeypatch.setattr(wiring_mod, "ModelArmorGuardrailPort", _SpyPort)
    wiring_mod._resolve_weakness_guardrail_template(_FakeScreener())
    assert captured["agent_id"] == "weakness_analyser_crew"


def test_practice_test_max_questions_defaults_and_refuses_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEAKNESS_PRACTICE_TEST_MAX_QUESTIONS", raising=False)
    assert _resolve_practice_test_max_questions() == 8
    monkeypatch.setenv("WEAKNESS_PRACTICE_TEST_MAX_QUESTIONS", "5")
    assert _resolve_practice_test_max_questions() == 5
    monkeypatch.setenv("WEAKNESS_PRACTICE_TEST_MAX_QUESTIONS", "five")
    with pytest.raises(ValueError, match="WEAKNESS_PRACTICE_TEST_MAX_QUESTIONS"):
        _resolve_practice_test_max_questions()
    monkeypatch.setenv("WEAKNESS_PRACTICE_TEST_MAX_QUESTIONS", "0")
    with pytest.raises(ValueError, match=">= 1"):
        _resolve_practice_test_max_questions()


def test_resolve_output_prices_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEAKNESS_OUTPUT_PRICE_PRACTICE_TEST", "120")
    monkeypatch.setenv("WEAKNESS_OUTPUT_PRICE_STUDY_AIDS", "60")
    assert wiring_mod._resolve_output_prices() == {"practice_test": 120, "study_aids": 60}


def test_resolve_output_prices_omits_unset_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEAKNESS_OUTPUT_PRICE_PRACTICE_TEST", raising=False)
    monkeypatch.setenv("WEAKNESS_OUTPUT_PRICE_STUDY_AIDS", "not-an-int")
    assert wiring_mod._resolve_output_prices() == {}


# --------------------------------------------------------------------------- #
# pure assembly seam
# --------------------------------------------------------------------------- #


def _assemble(monkeypatch: pytest.MonkeyPatch, *, runtime: Any, max_questions: int = 8) -> Any:
    monkeypatch.setenv("NATS_URL", "nats://nats:4222")  # for loop.from_env
    monkeypatch.setenv("WEAKNESS_OUTPUT_PRICE_PRACTICE_TEST", "120")
    monkeypatch.setenv("WEAKNESS_OUTPUT_PRICE_STUDY_AIDS", "60")
    return _assemble_graph_components(
        pubsub_project="chora-ai-kernel-orchestrator",
        db_conn=_FakeConn(),
        screener=_FakeScreener(),
        checkpointer=None,
        diagnoser_model_id="gemini-2.5-pro",
        guardrail_template=TEMPLATE,
        out_publisher=_FakeOutPublisher(),
        runtime=runtime,
        practice_test_max_questions=max_questions,
    )


def test_assemble_builds_the_bus_crew_and_registers_its_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict[str, Any] = {}
    real_builder = wiring_mod.build_weakness_analyser_graph

    def _spy(**kwargs: Any) -> Any:
        built.update(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(wiring_mod, "build_weakness_analyser_graph", _spy)
    runtime = _FakeRuntime()
    components = _assemble(monkeypatch, runtime=runtime, max_questions=5)

    assert components is not None
    # the ports are the Pub/Sub adapters; the diagnoser is also the task runner
    assert isinstance(built["extractor"], PubSubExtractorAdapter)
    assert isinstance(built["diagnoser"], PubSubDiagnoserAdapter)
    assert built["task_runner"] is built["diagnoser"]
    assert built["practice_test_max_questions"] == 5
    assert built["outputs_publisher"] is not None and built["evidence_emitter"] is not None
    # one executor, pinned to exactly this lane's two dispatch roles
    executor = built["extractor"]._executor
    assert executor is built["diagnoser"]._executor
    assert set(DISPATCH_ROLES) == {"companion_extract", "companion_diagnose"}
    assert executor._allowed_roles == set(DISPATCH_ROLES)
    # the runner drives that graph and carries the panel prices
    assert isinstance(components.crew_runner, WeaknessAnalyserCrewRunner)
    assert components.crew_runner._graph is components.graph
    assert components.crew_runner._output_prices == {"practice_test": 120, "study_aids": 60}
    # the inbound loop drives the GRAPH subscriber with the review_pending writer
    assert isinstance(components.subscriber, WeaknessAnalyserCrewGraphSubscriber)
    assert components.subscriber._runner is components.crew_runner
    assert components.subscriber._review_pending_publisher is not None
    # the lane is registered on the runtime: the two dispatch roles, bound to
    # this runner under the crew name (the legacy drain role was dropped once
    # its window closed, 2026-08-23)
    assert runtime.lanes == [
        {
            "name": WEAKNESS_CREW,
            "crew": WEAKNESS_CREW,
            "roles": COMPLETION_ROLES,
            "runner": components.crew_runner,
        }
    ]
    assert COMPLETION_ROLES == ("companion_extract", "companion_diagnose")
    assert components.pubsub_loop is not None and components.outbox_dispatcher is not None


def test_assemble_requires_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="runtime"):
        _assemble(monkeypatch, runtime=None)


def test_components_dataclass_has_no_gateway_and_no_mode() -> None:
    fields = {f.name for f in dataclasses.fields(WeaknessAnalyserCrewComponents)}
    assert {"crew_runner", "graph", "pubsub_loop", "outbox_dispatcher", "db_conn", "subscriber"} <= fields
    assert "gateway" not in fields and "mode" not in fields and "completion_loop" not in fields
