"""Phase 2C shim: ``api.stream_buffers`` → ``api.stream_pipeline``.

Pure re-export — no logic lives here. Canonical module is
``api.stream_pipeline`` (single source of truth for buffers, events,
state machine, and the streaming response handler).
"""

from api.stream_pipeline import PriorityStreamBuffer, TokenStreamBuffer

__all__ = ["PriorityStreamBuffer", "TokenStreamBuffer"]
