"""Streaming extractors — backward-compat re-export stub.

All logic lives in ``common.final_answer``.
This module re-exports every public name so existing
``from ui.components.streaming_extractors import …`` paths keep working.
"""

from common.final_answer import _ESCAPE_DECODE as _escape_decode
from common.final_answer import _FA_RE as _fa_re
from common.final_answer import FinalAnswerExtractor as _impl_FinalAnswerExtractor
from common.final_answer import extract_final_answer_delta as _extract_impl
from common.final_answer import recover_final_answer as _recover_impl

_ESCAPE_DECODE = _escape_decode
_FA_RE = _fa_re
_FinalAnswerExtractor = _impl_FinalAnswerExtractor
_extract_final_answer_delta = _extract_impl
_recover_final_answer = _recover_impl
