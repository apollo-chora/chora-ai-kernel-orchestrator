"""AUDIT-G3: the msgpack allowlist for the qgen crew's checkpointed state.

LangGraph's JsonPlusSerializer msgpack-encodes arbitrary dataclasses today and
warns on the way back in:

    Deserializing unregistered type ...qgen_crew.state.GuardrailResult from
    checkpoint. This will be blocked in a future version.

⚠ WHAT ACTUALLY HAPPENS ONCE IT IS BLOCKED. MEASURED against
langgraph-checkpoint 4.2.0 on 2026-08-23, because an earlier version of this
docstring asserted it and was WRONG IN THE DANGEROUS DIRECTION: it said every
parked job becomes "unresumable", which promises a loud failure at the
checkpoint read. IT DOES NOT RAISE THERE. The read SUCCEEDS and the frozen
dataclass comes back as a PLAIN DICT carrying the SAME FIELD VALUES. The run
then fails at the first ATTRIBUTE access on that value, in whatever node
touches it, with a traceback that never mentions serialisation.

⚠⚠ AND THAT LOUDNESS IS A PROPERTY OF THE CALLERS, NOT OF THIS SERDE.
It fails loudly only because the crews currently reach these values by
attribute (`res.allowed`, `q.question_id`). Because the degraded dict holds the
same values under the same keys, a caller written with `.get()` or a truthiness
test would NOT raise: it would silently take a different branch, on a graph
whose conditional edges route on exactly these fields. Today that is luck, not
design. That is why this is a correctness dependency on a library version and
not a lint warning: it protects the durability ADR-251 D4 exists to provide.

⚠ THE OBSERVED WARNINGS ARE A SAMPLE, NOT THE POPULATION. A live drive on
2026-08-23 warned for GuardrailResult and CandidatePayload only. CritiqueResult
is the same frozen dataclass on the same graph state and simply did not
deserialise in that window. Registering only what was observed would leave it to
fail on the same upgrade, so the allowlist is built from the MODULE CENSUS
(every dataclass in domain/qgen_crew/state.py), not from the log.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

#: (module, qualname) pairs LangGraph may reconstruct from a qgen checkpoint.
#: Census of domain/qgen_crew/state.py: three frozen dataclasses. QGenCrewState
#: is a TypedDict and rides as a plain mapping, so it needs no entry.
_STATE_MODULE = "chora_ai_kernel_orchestrator.domain.qgen_crew.state"

QGEN_MSGPACK_MODULES: tuple[tuple[str, str], ...] = (
    (_STATE_MODULE, "CandidatePayload"),
    (_STATE_MODULE, "CritiqueResult"),
    (_STATE_MODULE, "GuardrailResult"),
)


def qgen_checkpoint_serde() -> JsonPlusSerializer:
    """A serializer that stays correct when LANGGRAPH_STRICT_MSGPACK is on.

    Declaring the allowlist EXPLICITLY rather than relying on the permissive
    default means the behaviour does not change under us when the library
    flips its default, which is the whole point.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=QGEN_MSGPACK_MODULES)


def register_qgen_types(serde: Any) -> Any:
    """MERGE the qgen allowlist into an existing serde, returning the result.

    Deliberately a merge and not a replacement: this serde is shared by EVERY
    lane, so handing it a qgen-only allowlist would restrict the others. While
    LangGraph's default is permissive, ``with_msgpack_allowlist`` returns the
    serde unchanged, so this is a NO-OP TODAY and correct the moment the
    library flips its default. That is the point: the fix lands before the
    upgrade rather than after a resume has already degraded.

    Returns ``serde`` untouched if it does not support the merge, so an
    unexpected serializer type cannot break checkpointing.
    """
    merge = getattr(serde, "with_msgpack_allowlist", None)
    if merge is None:
        return serde
    return merge(QGEN_MSGPACK_MODULES)


__all__ = ["QGEN_MSGPACK_MODULES", "qgen_checkpoint_serde", "register_qgen_types"]
