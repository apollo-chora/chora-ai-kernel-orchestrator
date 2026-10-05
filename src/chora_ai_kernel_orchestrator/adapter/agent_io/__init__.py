"""Transport-neutral agent I/O (ADR-253 D6, ADR-254 D2): the session-state
builder the crew nodes feed an agent with and the terminal-output mapper they
read an agent's answer through. One implementation for the HTTP executor and
the Pub/Sub dispatch, so a transport swap cannot change what an agent sees or
what a crew reads."""

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.agent_io.response_mapping import (
    MD_FENCE_RE,
    map_agent_response,
    parse_terminal_json,
    strip_markdown_fences,
)
from chora_ai_kernel_orchestrator.adapter.agent_io.session_state import (
    CRITIC_ROLES,
    LANE_ROLE_CRITIQUE,
    LANE_ROLE_GENERATE,
    LANE_ROLE_RENDER,
    MERGE_ROLES,
    QUESTION_ROLES,
    ROLE_OE_EVALUATE,
    ROLE_OE_MODERATE,
    ROLE_QGEN_CRITIC,
    ROLE_QGEN_QUESTION,
    build_session_state,
    current_w3c_trace_context,
    decode_input,
    stamp_metadata_hints,
    stamp_set_plan,
)

__all__ = [
    "CRITIC_ROLES",
    "AgentExecutorResponse",
    "LANE_ROLE_CRITIQUE",
    "LANE_ROLE_GENERATE",
    "LANE_ROLE_RENDER",
    "MD_FENCE_RE",
    "MERGE_ROLES",
    "QUESTION_ROLES",
    "ROLE_OE_EVALUATE",
    "ROLE_OE_MODERATE",
    "ROLE_QGEN_CRITIC",
    "ROLE_QGEN_QUESTION",
    "build_session_state",
    "current_w3c_trace_context",
    "decode_input",
    "map_agent_response",
    "parse_terminal_json",
    "stamp_metadata_hints",
    "stamp_set_plan",
    "strip_markdown_fences",
]
