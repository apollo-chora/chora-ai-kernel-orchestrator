"""Composition-root smoke tests for qgen_crew_wiring.

The full ``build_qgen_crew_from_env`` is integration glue (psycopg
connect + Pub/Sub PublisherClient + LangGraph build). What we DO want
unit coverage on:

* ``QGenCrewComponents`` exposes ``agent_decision_writer`` (Gate #8) so the
  lifespan + tests can see it.
* When constructed with the writer, ``QGenCrewRunner`` records
  ``agent_decision_emitter`` so the runner's ``_emit_agent_decision_log``
  is reachable end-to-end.

(The Gate #7 token-usage writer was RETIRED 2026-07-23 -- see
tests/unit/test_token_usage_emitter_retired.py.)

This guards against silent regressions where a refactor drops the
wiring without breaking the existing terminal-publish path.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.agent_decision_outbox_writer import (
    AgentDecisionLogOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_publisher import (
    QGenCrewTerminalOutboxWriter,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_wiring import (
    QGenCrewComponents,
)
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
    QGenCrewRunner,
)


class _FakeConn:
    """Mock connection — never exercised; passed to writer ctor only."""

    def cursor(self) -> Any:  # pragma: no cover — never called
        raise AssertionError("cursor should not be exercised in unit test")


class _FakeGraph:
    """Mock LangGraph CompiledStateGraph — never exercised."""

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:  # pragma: no cover — never called
        return {}


class _FakePublisher:
    """Mock terminal publisher — never exercised."""

    async def publish_completed(self, **_: Any) -> str:  # pragma: no cover
        return ""

    async def publish_refused(self, **_: Any) -> str:  # pragma: no cover
        return ""


class TestQGenCrewComponentsAgentDecision:
    def test_components_dataclass_has_agent_decision_writer_field(self) -> None:
        """Gate #8 — composition root surfaces agent_decision_writer."""
        fields = {f.name for f in dataclasses.fields(QGenCrewComponents)}
        assert "agent_decision_writer" in fields, (
            "QGenCrewComponents missing agent_decision_writer field — Gate #8 composition root not wired"
        )


class TestQGenCrewComponentsTerminalPublish:
    def test_components_outbox_writer_field_still_present(self) -> None:
        """Regression guard — retiring the Gate #7 token-usage writer
        (2026-07-23) MUST NOT disturb the terminal-publish wiring."""
        fields = QGenCrewComponents.__dataclass_fields__
        assert "outbox_writer" in fields
        assert "QGenCrewTerminalOutboxWriter" in str(fields["outbox_writer"].type)
        assert QGenCrewTerminalOutboxWriter is not None


class TestQGenCrewComponentsHITL:
    def test_components_dataclass_has_hitl_decision_writer_field(self) -> None:
        """Human-Oversight gate — composition root surfaces
        hitl_decision_writer so the qgen runner can escalate a low-quality
        terminal into the O+ Human-Oversight queue."""
        fields = {f.name for f in dataclasses.fields(QGenCrewComponents)}
        assert "hitl_decision_writer" in fields, (
            "QGenCrewComponents missing hitl_decision_writer field — "
            "HITL gate composition root not wired (queue would stay empty)"
        )

    def test_components_hitl_decision_writer_field_type(self) -> None:
        """Field MUST be typed to HITLDecisionOutboxWriter so a refactor
        that swaps it for a stub gets caught at type-check."""
        f = QGenCrewComponents.__dataclass_fields__["hitl_decision_writer"]
        assert "HITLDecisionOutboxWriter" in str(f.type)


class TestArmorTemplateRetired:
    """ADR-169 — the bare-env _GuardrailAdapter + ARMOR_TEMPLATE_AI_ASSIST
    handling are retired in favour of the tier-mapped ModelArmorGuardrailPort.
    """

    def test_guardrail_adapter_class_removed(self) -> None:
        from chora_ai_kernel_orchestrator.adapter.pubsub import qgen_crew_wiring

        assert not hasattr(qgen_crew_wiring, "_GuardrailAdapter"), (
            "_GuardrailAdapter must be deleted — guardrail now routes through ModelArmorGuardrailPort (ADR-169)"
        )

    def test_armor_template_env_handling_removed(self) -> None:
        from chora_ai_kernel_orchestrator.adapter.pubsub import qgen_crew_wiring

        # The bare-env template constants are no longer part of the wiring
        # surface — the Port resolves the template from the tier mapping.
        assert not hasattr(qgen_crew_wiring, "ENV_ARMOR_TEMPLATE")
        assert not hasattr(qgen_crew_wiring, "DEFAULT_ARMOR_TEMPLATE")

    def test_wiring_module_imports_port(self) -> None:
        """The wiring builds the guardrail via ModelArmorGuardrailPort."""
        from chora_ai_kernel_orchestrator.adapter.pubsub import qgen_crew_wiring

        src = inspect.getsource(qgen_crew_wiring.build_qgen_crew_from_env)
        assert "ModelArmorGuardrailPort" in src
        assert "armor_template_name" not in src
        assert "_GuardrailAdapter" not in src


class TestRunnerEmitterBinding:
    def test_runner_records_agent_decision_emitter(self) -> None:
        """Gate #8 load-bearing wiring — runner._agent_decision_emitter
        is what the runner's _emit_agent_decision_log uses to fire the
        terminal AgentDecisionLog row."""
        writer = AgentDecisionLogOutboxWriter(
            conn=_FakeConn(),
            source_project="chora-489812",
        )
        runner = QGenCrewRunner(
            graph=_FakeGraph(),
            publisher=_FakePublisher(),
            agent_decision_emitter=writer,
        )
        assert runner._agent_decision_emitter is not None
        assert runner._agent_decision_emitter is writer

    def test_batch_runner_records_agent_decision_emitter(self) -> None:
        """CHO-2364 - the batch runner accepts + records the SAME Gate #8
        writer so the batch lane's per-item decisions ride the same outbox."""
        from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
            QGenBatchRunner,
        )

        writer = AgentDecisionLogOutboxWriter(
            conn=_FakeConn(),
            source_project="chora-489812",
        )
        runner = QGenBatchRunner(
            graph=_FakeGraph(),
            publisher=_FakePublisher(),
            agent_decision_emitter=writer,
        )
        assert runner._agent_decision_emitter is writer

    def test_wiring_threads_decision_writer_into_batch_runner(self) -> None:
        """CHO-2364 - build_qgen_crew_from_env must construct QGenBatchRunner
        WITH agent_decision_emitter=agent_decision_writer. This is the live
        seam whose absence made the batch lane (incl. the daily-dose type_plan
        path) emit zero qgen decision rows since 2026-07-04."""
        import re

        from chora_ai_kernel_orchestrator.adapter.pubsub import qgen_crew_wiring

        src = inspect.getsource(qgen_crew_wiring.build_qgen_crew_from_env)
        match = re.search(r"QGenBatchRunner\((?s:.*?)\n    \)", src)
        assert match is not None, "QGenBatchRunner construction not found"
        assert "agent_decision_emitter=agent_decision_writer" in match.group(0)


class TestQGenOnTheBus:
    """ADR-254 D2: the qgen crew dispatches on the three Pub/Sub lanes through
    the kennel runtime; no HTTP executor, no engine resource, no gateway image
    client. Read the composition root's source so a refactor that quietly
    re-introduces the HTTP path (or forgets the lane registration) is caught."""

    def _src(self) -> str:
        from chora_ai_kernel_orchestrator.adapter.pubsub import qgen_crew_wiring

        return inspect.getsource(qgen_crew_wiring.build_qgen_crew_from_env)

    def test_executor_is_the_dispatch_adapter_over_the_pubsub_executor(self) -> None:
        src = self._src()
        assert "QGenDispatchAdapter(" in src
        assert "PubSubAgentExecutor(source_project=pubsub_project, allowed_roles=QGEN_LANE_ROLES)" in src
        assert "ReasoningEngineExecutor" not in src
        assert "QGEN_QUESTION_ENGINE_RESOURCE" not in src

    def test_graph_rides_the_transactional_saver_for_crew_qgen(self) -> None:
        src = self._src()
        assert 'runtime.transactional_saver(crew="qgen")' in src
        assert "require_transactional_saver(" in src
        assert "checkpointer=saver" in src

    def test_lane_registers_the_three_roles_with_the_acceptance(self) -> None:
        src = self._src()
        assert 'runtime.register_lane("qgen", crew="qgen", roles=QGEN_LANE_ROLES, runner=acceptance)' in src

    def test_router_resolves_completions_through_the_inflight_registry(self) -> None:
        src = self._src()
        assert "inflight_registry=inflight_registry," in src
        assert "build_image_regen_graph(" in src

    def test_no_gateway_client_is_built(self) -> None:
        from chora_ai_kernel_orchestrator.adapter.pubsub import qgen_crew_wiring

        module_src = inspect.getsource(qgen_crew_wiring)
        assert "ModelGatewayImageClient" not in module_src
        assert "TestSetComposer" not in module_src
        assert "MODEL_GATEWAY_GRPC_TARGET" not in module_src
        assert "gateway=" not in self._src()

    def test_compose_is_a_flag_threaded_into_the_batch_runner(self) -> None:
        src = self._src()
        assert "ENV_TESTSET_COMPOSE_ENABLED" in src
        assert "compose_enabled=compose_enabled," in src

    def test_runtime_is_required_not_optional(self) -> None:
        import asyncio

        from chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_wiring import (
            build_qgen_crew_from_env,
        )

        with pytest.raises(RuntimeError, match="kennel runtime"):
            asyncio.run(build_qgen_crew_from_env(screener=object(), runtime=None))
