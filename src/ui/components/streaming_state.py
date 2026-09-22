"""Streaming state helpers — backward-compat re-export stub.

All logic has been consolidated into ``streaming_core.py`` (Phase 2B).
This module re-exports every public name so existing
``from ui.components.streaming_state import …`` paths and
``patch.object(streaming_state_mod, …)`` targets keep working.
"""

from core.session import SessionManager  # noqa: F401
from ui.components.common import get_doc_metadata  # noqa: F401
from ui.components.streaming_core import (  # noqa: F401
    _AUX_STATE_KEY,
    _STATUS_PLACEHOLDER,
    _build_process,
    _clear_aux_state,
    _clock,
    _content_generator,
    _FinalAnswerExtractor,
    _target_message_dict,
    _write_aux_state,
    stream_chunks,
)
