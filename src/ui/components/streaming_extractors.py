"""Streaming extractors — backward-compat re-export stub.

All logic has been consolidated into ``streaming_core.py`` (Phase 2B).
This module re-exports every public name so existing
``from ui.components.streaming_extractors import …`` paths keep working.
"""

from ui.components.streaming_core import (  # noqa: F401
    _ESCAPE_DECODE,
    _FA_RE,
    _extract_final_answer_delta,
    _FinalAnswerExtractor,
    _recover_final_answer,
)
