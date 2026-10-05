"""Process-wide logging bootstrap (CHO-2368).

Without this, no logging handler is ever configured and every app logger
falls back to Python's ``lastResort`` handler - WARNING+ only, bare message,
no logger name. Every INFO breadcrumb the adapters deliberately emit
(lifespan ``*.started`` / ``*.skipped``, the audit consumer's ``acked`` /
``activate_skipped_status``, inbox dedupe hits) was INVISIBLE in
``kubectl logs``: a consumer could receive, process and ack a message with
zero observable trace. Fail-loud requires the success path to be observable -
[[reusable_gotcha_no_logging_handler_makes_a_silent_skip_invisible]].

Called once from ``main.py`` at import time, BEFORE any adapter logs.
Uvicorn's own loggers (``uvicorn.*``) keep their handlers (propagate=False)
and are unaffected.
"""

from __future__ import annotations

import logging
import os
import sys

ENV_LOG_LEVEL = "CHORA_LOG_LEVEL"
DEFAULT_LEVEL = logging.INFO

_FORMAT = "%(levelname)s %(name)s %(message)s"


def configure_logging(root: logging.Logger | None = None) -> None:
    """Attach one stderr StreamHandler to the root logger at INFO
    (``CHORA_LOG_LEVEL`` overrides; invalid values fall back to INFO).

    Idempotent, and a no-op when the root logger already has handlers
    (embedded runs, test harnesses own their config). ``root`` is injectable
    for tests only - production call sites pass nothing.
    """
    if root is None:
        root = logging.getLogger()
    if root.handlers:
        return

    raw = (os.getenv(ENV_LOG_LEVEL) or "").strip().upper()
    level = getattr(logging, raw, None) if raw else DEFAULT_LEVEL
    if not isinstance(level, int):
        level = DEFAULT_LEVEL

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


__all__ = ["DEFAULT_LEVEL", "ENV_LOG_LEVEL", "configure_logging"]
