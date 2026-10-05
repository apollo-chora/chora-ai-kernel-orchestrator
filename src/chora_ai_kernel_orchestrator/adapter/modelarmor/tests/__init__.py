"""Unit tests for the modelarmor adapter (ADR-152).

Stub-driven — no network. The cloud-neutral
:class:`...local_screener.LocalScreener` backs the guardrail port; the
:class:`google.cloud.modelarmor_v1.ModelArmorAsyncClient` path is removed.
"""
