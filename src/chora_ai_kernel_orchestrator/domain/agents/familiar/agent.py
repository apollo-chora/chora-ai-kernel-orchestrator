"""Agent #19 Familiar — composition logic.

Pure functions that take a typed FamiliarRequest and return a typed
FamiliarResponse. No LLM SDK imports, no HTTP, no I/O. The orchestrator
pipeline runs these inside its LangGraph dispatch node, threading
guardrail + model-broker adapters around them.

Comic Ch6 P14 P1 anchor (proactive nudge): the generated text MUST
contain a topic-aware suggestion mentioning {topic}. Tested via
test_proactive_nudge_mentions_topic_for_authoring_signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# Default Familiar voice when the request omits the in-game name. Mirrors
# the comic's "no-bond" placeholder companion.
_DEFAULT_FAMILIAR_NAME = "Familiar"

# Canonical comic anchor for Daily Dose curation — must be present in the
# composed message so UI can match it deterministically.
_DAILY_DOSE_PREFIX = "Your Daily Dose is ready! 5 atoms"


class FamiliarKind(StrEnum):
    """Which Familiar agent call is being requested."""

    PROACTIVE_NUDGE = "proactive_nudge"
    DAILY_DOSE_CURATION = "daily_dose_curation"
    RPG_DIALOGUE = "rpg_dialogue"


@dataclass(frozen=True)
class FamiliarRequest:
    """Typed input to a Familiar agent call.

    Attributes:
        kind:            Which call to make (drives dispatch()).
        owner_gcid:      Learner's GCID (UUIDv7, opaque).
        familiar_name:   In-game Familiar name (the entity's name, NOT
                         the model name). Falls back to "Familiar" when
                         empty.
        topic:           Topic the agent should reference. Required for
                         proactive_nudge + rpg_dialogue.
        signal:          What triggered the proactive nudge — one of
                         "authoring", "review-due", "curiosity", or "".
                         "" defaults to authoring phrasing.
        dose_topics:     For daily_dose_curation, the topics in the dose.
        learner_message: For rpg_dialogue, the learner's input string.
    """

    kind: FamiliarKind
    owner_gcid: str
    familiar_name: str = ""
    topic: str = ""
    signal: str = ""
    dose_topics: list[str] = field(default_factory=list)
    learner_message: str = ""


@dataclass(frozen=True)
class FamiliarResponse:
    """Typed output from a Familiar agent call.

    Attributes:
        kind:           The originating kind (round-tripped for clients).
        message:        Generated text — what the orchestrator hands to
                        the guardrail screen + then back to the caller.
        familiar_name:  Echoed for UI personalisation.
    """

    kind: FamiliarKind
    message: str
    familiar_name: str


# --- helpers -----------------------------------------------------------------


def _require(value: str, field_name: str) -> str:
    """Trim and reject blank required fields."""
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError(f"familiar agent: {field_name} required (got blank)")
    return cleaned


def _voice_name(req: FamiliarRequest) -> str:
    name = (req.familiar_name or "").strip()
    return name if name else _DEFAULT_FAMILIAR_NAME


# --- compose_proactive_nudge -------------------------------------------------


def compose_proactive_nudge(req: FamiliarRequest) -> FamiliarResponse:
    """Build the Comic Ch6 P14 P1 proactive nudge text.

    Invariant: ``{topic}`` appears verbatim in the message. Default
    phrasing for the "authoring" signal: "I noticed you're authoring on
    {topic}". Other signals get their own opening but always preserve
    the topic mention.

    Args:
        req: kind must be PROACTIVE_NUDGE; owner_gcid + topic required.

    Raises:
        ValueError: owner_gcid or topic blank.
    """
    _require(req.owner_gcid, "owner_gcid")
    topic = _require(req.topic, "topic")

    voice = _voice_name(req)
    signal = (req.signal or "").strip().lower()

    if signal == "review-due":
        # Review branch — comic anchor: "Time to review {topic}!"
        message = (
            f"Hi! It's {voice}. Time to review {topic} — your memory's fading "
            f"on this topic and a quick recap will lock it back in."
        )
    elif signal == "curiosity":
        # Curiosity branch — invitation tone.
        message = (
            f"Hi! It's {voice}. I noticed your curiosity sparking on "
            f"{topic}. Want to explore an adjacent atom together?"
        )
    else:
        # Default + "authoring" branch — the canonical Comic Ch6 P14 P1
        # anchor wording must appear verbatim.
        message = (
            f"Hi! It's {voice}. I noticed you're authoring on {topic} — want me to suggest a related atom or quiz?"
        )

    return FamiliarResponse(
        kind=FamiliarKind.PROACTIVE_NUDGE,
        message=message,
        familiar_name=voice,
    )


# --- compose_daily_dose_curation ---------------------------------------------


def compose_daily_dose_curation(req: FamiliarRequest) -> FamiliarResponse:
    """Build the Comic Ch6 P14 P4 daily-dose curation message.

    Always opens with the canonical "Your Daily Dose is ready! 5 atoms"
    anchor; appends a topic preview when ``dose_topics`` is non-empty.
    """
    _require(req.owner_gcid, "owner_gcid")
    voice = _voice_name(req)

    topic_list = [t.strip() for t in (req.dose_topics or []) if t and t.strip()]
    if topic_list:
        # Dedupe while preserving insertion order so the preview is stable.
        seen: set[str] = set()
        unique: list[str] = []
        for t in topic_list:
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(t)
        topics_phrase = ", ".join(unique)
        message = f"{_DAILY_DOSE_PREFIX} — today's mix covers {topics_phrase}. {voice} is ready when you are."
    else:
        message = f"{_DAILY_DOSE_PREFIX} — fresh atoms picked just for you. {voice} is ready when you are."

    return FamiliarResponse(
        kind=FamiliarKind.DAILY_DOSE_CURATION,
        message=message,
        familiar_name=voice,
    )


# --- compose_rpg_dialogue ----------------------------------------------------


def compose_rpg_dialogue(req: FamiliarRequest) -> FamiliarResponse:
    """Build a Familiar-voice reply to a learner message.

    Personalises the reply with the in-game familiar name + the topic so
    the companion-voice fiction holds together. Real personalisation
    (RAG memory + tone-from-archetype) is post-MVP; this is the
    deterministic, audit-friendly skeleton.
    """
    _require(req.owner_gcid, "owner_gcid")
    _require(req.learner_message, "learner_message")
    topic = _require(req.topic, "topic")

    voice = _voice_name(req)
    message = f"{voice} chirps: 'Together let's unpack {topic} — that's a great question to dig into next.'"
    return FamiliarResponse(
        kind=FamiliarKind.RPG_DIALOGUE,
        message=message,
        familiar_name=voice,
    )


# --- dispatch ----------------------------------------------------------------


def dispatch(req: FamiliarRequest) -> FamiliarResponse:
    """Route a FamiliarRequest to the matching compose_* helper.

    Used by the orchestrator's LangGraph node to invoke agent #19. The
    enum-keyed branch keeps the call site declarative and avoids string
    mismatches.
    """
    if req.kind == FamiliarKind.PROACTIVE_NUDGE:
        return compose_proactive_nudge(req)
    if req.kind == FamiliarKind.DAILY_DOSE_CURATION:
        return compose_daily_dose_curation(req)
    if req.kind == FamiliarKind.RPG_DIALOGUE:
        return compose_rpg_dialogue(req)
    # StrEnum guarantees the only valid values are the three above; this
    # guard is defensive — future kinds must be wired here.
    raise ValueError(f"familiar agent: unknown kind {req.kind!r}")
