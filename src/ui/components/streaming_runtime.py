"""Streaming runtime — backward-compat re-export stub.

All logic has been consolidated into ``streaming_core.py`` (Phase 2B).
This module re-exports every public name so existing
``from ui.components.streaming_runtime import …`` paths and
``patch.object(streaming_runtime_mod, …)`` targets keep working.
"""

from api.stream_pipeline import StreamChunk, get_streaming_handler  # noqa: F401
from common.config import (  # noqa: F401
    UI_STREAMING_HARD_TIMEOUT,
    UI_STREAMING_SETUP_TIMEOUT,
    UI_STREAMING_TIMEOUT,
)
from core.session import SessionManager  # noqa: F401
from ui.components.streaming_core import (  # noqa: F401
    _STATUS_PLACEHOLDER,
    _clock,
    _FinalAnswerExtractor,
    _q_get_with_stop_poll,
    stream_chunks,
    stream_content,
)
