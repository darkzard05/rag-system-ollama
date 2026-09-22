"""Phase 2C shim: ``api.stream_events`` → ``api.stream_pipeline``.

Pure re-export — no logic lives here. Canonical module is
``api.stream_pipeline`` (single source of truth for buffers, events,
state machine, and the streaming response handler).
"""

from api.stream_pipeline import (
    STREAM_EVENT_TYPES,
    StreamChunk,
    StreamEvent,
    chunk_to_stream_events,
)

__all__ = [
    "STREAM_EVENT_TYPES",
    "StreamChunk",
    "StreamEvent",
    "chunk_to_stream_events",
]
