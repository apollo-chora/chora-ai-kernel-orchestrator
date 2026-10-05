"""Kroki adapter — cluster-local deterministic Mermaid→diagram renderer (W8).

Backs the qgen ``render_image`` node's ``mode=="mermaid"`` path. POSTs the
Mermaid source to the in-cluster chora-kroki Service (plaintext ClusterIP,
mesh-gated) and returns the rendered diagram bytes.

Per [[secrets-and-env]] / feedback_no_inline_config the endpoint is read from
``KROKI_ENDPOINT`` (e.g. ``http://chora-kroki.ai-kernel.svc.cluster.local:8000``)
— never hardcoded.
"""

from chora_ai_kernel_orchestrator.adapter.kroki.client import KrokiClient

__all__ = ["KrokiClient"]
