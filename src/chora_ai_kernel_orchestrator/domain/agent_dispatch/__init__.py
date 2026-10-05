"""Pure agent-dispatch domain rules (ADR-253 D3a park + ADR-254 D5 reaper).

No I/O, no clock, no env reads at import time. The adapters in
``adapter/pubsub`` and ``adapter/checkpointer`` carry these types to and from
Postgres and Pub/Sub; the rules themselves are testable with a dict and a
``datetime``.
"""
