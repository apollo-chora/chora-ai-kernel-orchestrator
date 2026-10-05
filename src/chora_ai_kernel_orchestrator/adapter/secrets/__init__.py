"""Secret Manager adapter for chora-ai-kernel-orchestrator.

Per CLAUDE.md §6 + memory `feedback_no_inline_config`: DSNs and other
secrets MUST come from Secret Manager (Workload Identity Federation in
production). This adapter resolves a Secret Manager secret name to its
latest payload at boot.

The Python orchestrator deliberately avoids importing google-cloud-secret-manager
at module load time so unit tests + dev runs (where the env vars are unset)
incur zero network cost. The import is delayed inside ``resolve_dsn`` to
the path where a secret is actually required.
"""

from chora_ai_kernel_orchestrator.adapter.secrets.resolver import resolve_dsn

__all__ = ["resolve_dsn"]
