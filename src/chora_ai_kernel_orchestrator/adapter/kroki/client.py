"""``KrokiClient`` — thin HTTP adapter to the cluster-local Kroki renderer.

The qgen ``render_image`` node's ``mode=="mermaid"`` path POSTs the Mermaid
source here and gets back diagram bytes (PNG/SVG). Injectable HTTP client so
the node + adapter tests mock the transport without a live Kroki.

Kroki HTTP API (per chora-infra/k8s/services/chora-kroki/service.yaml):

    POST {KROKI_ENDPOINT}/mermaid/{output_format}
    Body: the raw Mermaid source (text/plain)
    → 200 with the rendered diagram bytes

Per [[secrets-and-env]] the endpoint is sourced from ``KROKI_ENDPOINT``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)

ENV_KROKI_ENDPOINT = "KROKI_ENDPOINT"

# Default Kroki output format when the image_spec doesn't pin one. PNG is the
# safe FE-renderable default (SVG also works; the FE <img> handles both).
_DEFAULT_OUTPUT_FORMAT = "png"

# Kroki's Mermaid diagram-type path segment.
_DIAGRAM_TYPE = "mermaid"


class _HttpResponseLike(Protocol):
    status_code: int
    content: bytes
    text: str


class _HttpClientLike(Protocol):
    """Subset of httpx.AsyncClient that we use (raw-body POST)."""

    async def post(
        self,
        url: str,
        *,
        content: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> _HttpResponseLike: ...

    async def aclose(self) -> None: ...


class KrokiClient:
    """POSTs Mermaid source to Kroki → diagram bytes."""

    def __init__(
        self,
        *,
        endpoint: str,
        http_client: _HttpClientLike | None = None,
    ) -> None:
        if not (endpoint or "").strip():
            raise ValueError(
                f"KrokiClient requires a non-empty endpoint (set {ENV_KROKI_ENDPOINT} per [[secrets-and-env]])"
            )
        # Normalise: strip trailing slash so {endpoint}/mermaid/{fmt} never
        # double-slashes.
        self._endpoint = endpoint.strip().rstrip("/")
        self._http_client = http_client

    @classmethod
    def from_env(cls) -> KrokiClient | None:
        """Build from ``KROKI_ENDPOINT``. None when unset (image feature
        unconfigured — the render_image node fails loud only if an image_spec
        is actually present)."""
        endpoint = os.getenv(ENV_KROKI_ENDPOINT, "").strip()
        if not endpoint:
            return None
        return cls(endpoint=endpoint)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def render(self, *, source: str, output_format: str = _DEFAULT_OUTPUT_FORMAT) -> bytes:
        """Render the Mermaid ``source`` → diagram bytes.

        Raises ``RuntimeError`` on a non-2xx response (the caller treats a
        render failure as fail-soft: publish the question without the image).
        """
        fmt = (output_format or _DEFAULT_OUTPUT_FORMAT).strip().lower()
        url = f"{self._endpoint}/{_DIAGRAM_TYPE}/{fmt}"
        client = await self._resolve_http_client()
        resp = await client.post(
            url,
            content=source.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Kroki render failed: POST {url} → {resp.status_code} {getattr(resp, 'text', '')[:300]}"
            )
        return bytes(resp.content)

    async def _resolve_http_client(self) -> _HttpClientLike:
        if self._http_client is not None:
            return self._http_client
        try:  # pragma: no cover — production
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "httpx not installed; KrokiClient requires httpx>=0.27 (already present per pyproject.toml)."
            ) from exc
        # httpx.AsyncClient.post has a wider signature than our narrow
        # _HttpClientLike Protocol (which we deliberately keep minimal); the
        # call sites only use the subset, so cast for the structural match.
        client = cast(_HttpClientLike, httpx.AsyncClient(timeout=30.0))  # pragma: no cover
        self._http_client = client  # pragma: no cover
        return client  # pragma: no cover

    async def aclose(self) -> None:  # pragma: no cover — lifecycle
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                logger.exception("kroki_client.aclose_failed")


__all__ = ["ENV_KROKI_ENDPOINT", "KrokiClient"]
