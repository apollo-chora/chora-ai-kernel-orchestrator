"""WS-3 (ADR-205 / CHO-1955) — Model Armor screener adapter for the crew graph.

Adapts the existing Cloud Model Armor port (sanitize_user_prompt /
sanitize_model_response, ADR-152) to the graph's `Screener` port: direction
"input" runs the prompt sanitiser over the extracted text + clue DATA;
"output" runs the response sanitiser over the diagnosis JSON. A non-ALLOW
verdict on input fails the crew closed (ADR-205 D2). Fail-loud: a Model Armor
transport error raises (the graph node catches → refund), never a silent ALLOW.
"""

from __future__ import annotations

import asyncio

import pytest

from chora_ai_kernel_orchestrator.adapter.modelarmor.screener import (
    ScreenRequest,
    ScreenResult,
    Verdict,
)
from chora_ai_kernel_orchestrator.adapter.weakness.screener_adapter import (
    ModelArmorScreenerAdapter,
)


class FakeArmor:
    def __init__(
        self,
        *,
        user_verdict: Verdict = Verdict.ALLOW,
        resp_verdict: Verdict = Verdict.ALLOW,
        raise_on: str | None = None,
    ) -> None:
        self._uv = user_verdict
        self._rv = resp_verdict
        self._raise = raise_on
        self.user_calls: list[ScreenRequest] = []
        self.resp_calls: list[ScreenRequest] = []

    async def sanitize_user_prompt(self, req: ScreenRequest) -> ScreenResult:
        if self._raise == "input":
            raise RuntimeError("armor transport down")
        self.user_calls.append(req)
        return ScreenResult(verdict=self._uv, reason="user-screened")

    async def sanitize_model_response(self, req: ScreenRequest) -> ScreenResult:
        self.resp_calls.append(req)
        return ScreenResult(verdict=self._rv, reason="resp-screened")

    async def close(self) -> None:
        pass


_TEMPLATE = "projects/chora-489812/locations/asia-southeast1/templates/weakness-strict"


def test_input_direction_uses_user_prompt_sanitiser() -> None:
    armor = FakeArmor()
    adapter = ModelArmorScreenerAdapter(screener=armor, template_name=_TEMPLATE)
    v = asyncio.run(adapter.screen(text="2+2=5", tenant_id="t1", gcid="g1", direction="input"))
    assert v.decision == "ALLOW"
    assert len(armor.user_calls) == 1 and len(armor.resp_calls) == 0
    assert armor.user_calls[0].template_name == _TEMPLATE
    assert armor.user_calls[0].text == "2+2=5"
    assert armor.user_calls[0].tenant_id == "t1"


def test_output_direction_uses_response_sanitiser() -> None:
    armor = FakeArmor()
    adapter = ModelArmorScreenerAdapter(screener=armor, template_name=_TEMPLATE)
    v = asyncio.run(adapter.screen(text='{"edges":[]}', tenant_id="t1", gcid="g1", direction="output"))
    assert v.decision == "ALLOW"
    assert len(armor.resp_calls) == 1 and len(armor.user_calls) == 0


def test_block_verdict_maps_through() -> None:
    armor = FakeArmor(user_verdict=Verdict.BLOCK)
    adapter = ModelArmorScreenerAdapter(screener=armor, template_name=_TEMPLATE)
    v = asyncio.run(adapter.screen(text="jailbreak", tenant_id="t1", gcid="g1", direction="input"))
    assert v.decision == "BLOCK"


def test_transport_error_raises_never_silent_allow() -> None:
    armor = FakeArmor(raise_on="input")
    adapter = ModelArmorScreenerAdapter(screener=armor, template_name=_TEMPLATE)
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.screen(text="x", tenant_id="t1", gcid="g1", direction="input"))
