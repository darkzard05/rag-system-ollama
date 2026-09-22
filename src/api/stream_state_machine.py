"""Phase 2C shim: ``api.stream_state_machine`` → ``api.stream_pipeline``.

Pure re-export — no logic lives here. Canonical module is
``api.stream_pipeline`` (single source of truth for buffers, events,
state machine, and the streaming response handler).
"""

from api.stream_pipeline import (
    StreamingState,
    StreamingStateContext,
    StreamingStateMachine,
    create_streaming_state_machine,
)

__all__ = [
    "StreamingState",
    "StreamingStateContext",
    "StreamingStateMachine",
    "create_streaming_state_machine",
]
