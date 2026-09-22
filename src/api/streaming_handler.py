"""Phase 2C shim: ``api.streaming_handler`` → ``api.stream_pipeline``.

Pure re-export — no logic lives here. Canonical module is
``api.stream_pipeline`` (single source of truth for buffers, events,
state machine, and the streaming response handler).

``UI_CONTENT_BUFFER_SIZE`` and ``get_streaming_handler`` are kept in this
namespace so the existing ``patch.object(api.streaming_handler, ...)``
call-time seam keeps working byte-identically.
"""

from api import stream_pipeline as _pipeline
from api.stream_pipeline import (
    AdaptiveStreamingController,
    PriorityStreamBuffer,
    ServerSentEventsHandler,
    StreamChunk,
    StreamingMetrics,
    StreamingResponseBuilder,
    StreamingResponseHandler,
    StreamingState,
    StreamingStateContext,
    StreamingStateMachine,
    TokenStreamBuffer,
    create_streaming_state_machine,
    get_adaptive_controller,
    gzip_compress,
)
from api.stream_pipeline import _estimate_tokens as _estimate_tokens
from common.config import (
    UI_CONTENT_BUFFER_SIZE,
    UI_STREAMING_CONTENT_TIMEOUT_MS,
    UI_STREAMING_THOUGHT_BUFFER_SIZE,
    UI_STREAMING_THOUGHT_TIMEOUT_MS,
)

# Contract: all 15 names must stay importable from `from api.streaming_handler import (...)`.
__all__ = ["AdaptiveStreamingController", "PriorityStreamBuffer", "ServerSentEventsHandler", "StreamChunk", "StreamingMetrics", "StreamingResponseBuilder", "StreamingResponseHandler", "StreamingState", "StreamingStateContext", "StreamingStateMachine", "TokenStreamBuffer", "create_streaming_state_machine", "get_adaptive_controller", "get_streaming_handler", "gzip_compress"]  # fmt: skip


def get_streaming_handler() -> StreamingResponseHandler:
    """Build a handler reading this module's ``UI_*`` constants at call time.

    Preserves the ``patch.object(api.streaming_handler, "UI_CONTENT_BUFFER_SIZE", N)``
    seam — creation logic is identical to ``stream_pipeline``.
    """
    return _pipeline.StreamingResponseHandler(
        content_buffer_size=UI_CONTENT_BUFFER_SIZE,
        content_timeout_ms=UI_STREAMING_CONTENT_TIMEOUT_MS,
        thought_buffer_size=UI_STREAMING_THOUGHT_BUFFER_SIZE,
        thought_timeout_ms=UI_STREAMING_THOUGHT_TIMEOUT_MS,
    )
