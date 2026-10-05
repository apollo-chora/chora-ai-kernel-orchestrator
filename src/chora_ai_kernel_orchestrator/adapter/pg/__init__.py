"""psycopg-backed adapters for chora_ai_kernel domain ports.

Hexagonal: modules here implement domain ports (``domain/**/repository.py``)
against a psycopg ``AsyncConnection``. They depend on the domain, never the
reverse.
"""

from __future__ import annotations
