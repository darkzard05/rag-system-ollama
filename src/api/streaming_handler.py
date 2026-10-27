"""Phase 2C shim: ``api.streaming_handler`` → ``api.stream_pipeline``.

Pure re-export — no logic lives here. Canonical module is
``api.stream_pipeline`` (single source of truth for buffers, events,
state machine, and the streaming response handler).
"""

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
    get_streaming_handler,
    gzip_compress,
)
from api.stream_pipeline import _estimate_tokens as _estimate_tokens

__all__ = ["AdaptiveStreamingController", "PriorityStreamBuffer", "ServerSentEventsHandler", "StreamChunk", "StreamingMetrics", "StreamingResponseBuilder", "StreamingResponseHandler", "StreamingState", "StreamingStateContext", "StreamingStateMachine", "TokenStreamBuffer", "create_streaming_state_machine", "get_adaptive_controller", "get_streaming_handler", "gzip_compress"]  # fmt: skip
