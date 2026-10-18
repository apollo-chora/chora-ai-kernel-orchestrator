"""Source-material retrieval from chora-creation, and its prompt rendering.

WHY THIS EXISTS

A grounded batch used to reach the model as a ``genai.FileData`` part carrying
the object-store URI "by reference", on the assumption that the gateway would
dereference it. The Python model gateway's canonical contract admits only
text | image | refusal parts, so the Go client refuses such a part and every
file-based generation failed before a model call.

The replacement is deterministic extracted TEXT. chora-creation already parses
text-bearing uploads at job creation (ADR-180 D9) into page-referenced chunks
and serves them from a tenant/job-scoped endpoint:

    GET {creation}/api/internal/question-jobs/{job_id}/source-material
        ?tenant_id=…&author_gcid=…&max_bytes=…

This module fetches those chunks and renders them with explicit delimiters.
It is the ONLY place document semantics are reconstructed: the Go grounding
plugin and the qgen composer inject the rendered string verbatim, because they
cannot see chunk structure.

FAIL LOUD, NEVER GROUND ON NOTHING

Every failure mode raises :class:`SourceMaterialError` with a machine-readable
``reason``. The caller must refuse BEFORE dispatching, rather than let a model
call proceed ungrounded: questions that look valid but ignore the author's
document are worse than an error. Do not recover by parsing English text —
branch on ``reason``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx

logger = logging.getLogger(__name__)

ENV_CREATION_BASE_URL = "CREATION_BASE_URL"
ENV_SOURCE_MATERIAL_TIMEOUT = "CREATION_SOURCE_MATERIAL_TIMEOUT_SECONDS"
DEFAULT_TIMEOUT_SECONDS = 5.0

# Bounds handed to creation. It enforces its own ceiling too; the caller is
# never the only bound. Roughly 30k tokens of English — a recovery default, not
# a number derived from any model's context window.
DEFAULT_MAX_BYTES = 120_000

# Reasons are a closed vocabulary so callers can branch without string matching.
# An HTTP RESPONSE is proof creation was reached, so never report one as
# "unreachable": 4xx means it refused this request, 5xx means it accepted it and
# failed. Those are different operator actions.
REASON_NOT_CONFIGURED = "creation_not_configured"
REASON_UNREACHABLE = "creation_unreachable"
REASON_REJECTED = "creation_rejected"
REASON_FAILED = "creation_failed"
REASON_BAD_RESPONSE = "creation_bad_response"
REASON_NO_TEXT = "source_material_empty"


class SourceMaterialError(RuntimeError):
    """A grounded batch could not obtain its grounding text.

    ``reason`` is machine-readable; ``detail`` is for logs, never for branching.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SourceMaterialChunk:
    file_uri: str
    file_role: str
    chunk_index: int
    page_no: int | None
    text: str


class _HttpClientLike(Protocol):
    """Subset of httpx.AsyncClient this adapter uses."""

    async def get(self, url: str, params: dict[str, str]) -> Any: ...


def render_source_material(chunks: list[SourceMaterialChunk]) -> str:
    """Render chunks into the delimited block the agents ground on.

    Structure is preserved: one tag per chunk, carrying the file it came from
    and its page (omitted when the format is unpaged, roughly DOCX/MD/TXT). A
    rubric keeps its own tag because a mark scheme GRADES the material, it is
    not question material.

    Order is the storage order creation returned, NOT semantic priority — a
    rubric can therefore precede the paper it grades. Callers that need role
    priority must apply it themselves; do not infer importance from position.

    Chunk text is inserted verbatim. That is deliberate: escaping would alter
    the author's content, and the upstream text is already untrusted input the
    model is told to treat as material rather than instructions.
    """
    blocks: list[str] = []
    for c in chunks:
        tag = "RUBRIC" if c.file_role.strip().lower() == "rubric" else "SOURCE"
        page = f' page="{c.page_no}"' if c.page_no is not None else ""
        blocks.append(f'<{tag} file="{c.file_uri}"{page}>\n{c.text}\n</{tag}>')
    return "\n\n".join(blocks)


class CreationSourceMaterialClient:
    """Fetches a job's extracted source material from chora-creation."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: _HttpClientLike | None = None,
    ) -> None:
        base = (base_url or "").strip().rstrip("/")
        if not base:
            raise SourceMaterialError(
                REASON_NOT_CONFIGURED,
                f"creation base URL is empty (set {ENV_CREATION_BASE_URL})",
            )
        self._base = base
        self._timeout = timeout_seconds
        self._http = http_client

    @classmethod
    def from_env(cls) -> CreationSourceMaterialClient | None:
        base = os.getenv(ENV_CREATION_BASE_URL, "").strip()
        if not base:
            logger.warning("creation_source_material.not_configured env=%s", ENV_CREATION_BASE_URL)
            return None
        raw = os.getenv(ENV_SOURCE_MATERIAL_TIMEOUT, "").strip()
        try:
            timeout = float(raw) if raw else DEFAULT_TIMEOUT_SECONDS
        except ValueError:
            logger.warning(
                "creation_source_material.bad_timeout env=%s value=%s", ENV_SOURCE_MATERIAL_TIMEOUT, raw[:32]
            )
            timeout = DEFAULT_TIMEOUT_SECONDS
        return cls(base, timeout_seconds=timeout)

    def _client(self) -> _HttpClientLike:
        if self._http is None:
            self._http = cast(_HttpClientLike, httpx.AsyncClient(timeout=self._timeout))
        return self._http

    async def fetch(
        self,
        *,
        tenant_id: str,
        job_id: str,
        author_gcid: str = "",
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> tuple[str, bool]:
        """Return ``(rendered_text, chunks_complete)`` for one job.

        Raises :class:`SourceMaterialError` when the material cannot be
        obtained or is empty — a caller must not treat that as "no grounding
        needed". ``author_gcid`` is sent whenever known: creation rejects a job
        owned by a different author, which turns a wrong job id into a refusal
        instead of another author's document.
        """
        if not tenant_id.strip():
            raise SourceMaterialError(REASON_BAD_RESPONSE, "tenant_id is required")
        if not job_id.strip():
            raise SourceMaterialError(REASON_BAD_RESPONSE, "job_id is required")

        url = f"{self._base}/api/internal/question-jobs/{job_id}/source-material"
        params = {"tenant_id": tenant_id, "max_bytes": str(int(max_bytes))}
        if author_gcid.strip():
            params["author_gcid"] = author_gcid.strip()

        try:
            resp = await self._client().get(url, params=params)
        except Exception as exc:  # transport: DNS, refused, timeout, TLS
            raise SourceMaterialError(REASON_UNREACHABLE, f"{type(exc).__name__}: {exc}") from exc

        status = int(getattr(resp, "status_code", 0) or 0)
        if status != 200:
            # Do NOT collapse this into "unreachable": a response proves creation
            # was reached. 4xx = it refused this request (bad tenant, unknown
            # job, author mismatch, unwired store); 5xx = it accepted the
            # request and failed. An unknown job must never look like a job with
            # zero extracted text, so neither becomes source_material_empty.
            reason = REASON_REJECTED if 400 <= status < 500 else REASON_FAILED
            raise SourceMaterialError(reason, f"creation returned HTTP {status}")

        try:
            body = resp.json()
        except Exception as exc:
            raise SourceMaterialError(REASON_BAD_RESPONSE, f"unparseable body: {exc}") from exc
        if not isinstance(body, dict):
            raise SourceMaterialError(REASON_BAD_RESPONSE, f"body is {type(body).__name__}, want object")

        raw_chunks = body.get("chunks")
        if not isinstance(raw_chunks, list):
            raise SourceMaterialError(REASON_BAD_RESPONSE, "body has no chunks array")

        # creation emits a JSON boolean; "false" is a truthy string, so a
        # permissive bool() would silently mark a truncated corpus as complete.
        raw_complete = body.get("chunks_complete")
        if not isinstance(raw_complete, bool):
            raise SourceMaterialError(REASON_BAD_RESPONSE, "body.chunks_complete is not boolean")

        chunks: list[SourceMaterialChunk] = []
        for idx, entry in enumerate(raw_chunks):
            # Strict, not coercing: this adapter promises malformed wire
            # responses become creation_bad_response, and str()/int() would
            # quietly rewrite the document instead of reporting it.
            if not isinstance(entry, dict):
                raise SourceMaterialError(REASON_BAD_RESPONSE, f"chunk {idx} is not an object")
            for field in ("text", "file_uri", "file_role"):
                value = entry.get(field)
                if value is not None and not isinstance(value, str):
                    raise SourceMaterialError(REASON_BAD_RESPONSE, f"chunk {idx}.{field} is not a string")
            chunk_index = entry.get("chunk_index", 0)
            if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
                raise SourceMaterialError(REASON_BAD_RESPONSE, f"chunk {idx}.chunk_index is not an integer")
            page = entry.get("page_no")
            if page is not None and (not isinstance(page, int) or isinstance(page, bool)):
                raise SourceMaterialError(REASON_BAD_RESPONSE, f"chunk {idx}.page_no is not an integer")

            text = entry.get("text") or ""
            if not text.strip():
                continue  # a whitespace chunk cannot ground anything

            chunks.append(
                SourceMaterialChunk(
                    file_uri=entry.get("file_uri") or "",
                    file_role=entry.get("file_role") or "source",
                    chunk_index=chunk_index,
                    page_no=page,
                    text=text,
                )
            )

        if not chunks:
            # Image-only uploads have no text layer by design, and a failed
            # extraction stores nothing. Neither may become an ungrounded
            # generation.
            raise SourceMaterialError(
                REASON_NO_TEXT,
                f"job {job_id} has no extractable text (image-only upload or failed extraction)",
            )

        rendered = render_source_material(chunks)
        complete = raw_complete
        if not complete:
            # Grounding on a prefix is legitimate; pretending it is the whole
            # document is not. The caller records this in the trace.
            logger.warning("creation_source_material.truncated job=%s chunks=%d", job_id, len(chunks))
        return rendered, complete


def _state_str(state: Any, key: str) -> str:
    value = state.get(key) if hasattr(state, "get") else None
    return value.strip() if isinstance(value, str) else ""


def declares_source_material(state: Any) -> bool:
    """True when the job says it has uploaded material, text or not.

    Mirrors the Go grounding plugin's rule so the two sides cannot disagree
    about which jobs owe grounding: a non-empty ``source_material_text``, OR any
    ``source_files`` ENTRY carrying a non-empty ``blob_uri``, OR a non-empty
    ``source_blob_uri``. A non-empty list of empty entries is NOT material —
    groundingplugin treats it the same way.
    """
    if not hasattr(state, "get"):
        return False
    # Mirror the Go plugin check-for-check, including the entries: a
    # source_files LIST whose entries all lack blob_uri is NOT material there,
    # so treating list-length alone as the signal would make the two sides
    # disagree and one of them would silently skip grounding.
    if _state_str(state, "source_material_text") != "":
        return True
    files = state.get("source_files")
    if isinstance(files, (list, tuple)):
        for entry in files:
            if isinstance(entry, dict) and str(entry.get("blob_uri") or "").strip():
                return True
    return _state_str(state, "source_blob_uri") != ""


async def source_material_text_for_state(
    state: Any,
    *,
    client: CreationSourceMaterialClient | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[str, bool]:
    """Return ``(grounding_text, chunks_complete)`` for the job in graph state.

    Returns ``("", True)`` when the state declares NO source material, so the
    ungrounded and single-candidate paths stay byte-identical.

    Raises :class:`SourceMaterialError` when material IS declared but cannot be
    obtained. The caller must turn that into a refusal BEFORE dispatching: a
    model call that proceeds without the material produces questions that look
    valid while ignoring the author's document, which is worse than an error.
    """
    if not declares_source_material(state):
        return "", True

    # Already-resolved material is honoured rather than refetched: a resumed or
    # replayed run must not pay a second round-trip, and a caller that has
    # already done the work should not be forced through the network again.
    existing = _state_str(state, "source_material_text")
    if existing:
        return existing, True

    resolved = client if client is not None else CreationSourceMaterialClient.from_env()
    if resolved is None:
        raise SourceMaterialError(
            REASON_NOT_CONFIGURED,
            f"job declares source material but {ENV_CREATION_BASE_URL} is unset",
        )

    return await resolved.fetch(
        tenant_id=_state_str(state, "tenant_id"),
        job_id=_state_str(state, "job_id"),
        # `gcid` is the authoritative author field in graph state; `user_gcid`
        # is only a fallback for payloads that carry the other spelling.
        author_gcid=_state_str(state, "gcid") or _state_str(state, "user_gcid"),
        max_bytes=max_bytes,
    )
