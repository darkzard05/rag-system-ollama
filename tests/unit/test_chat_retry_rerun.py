"""Tests for chat retry / sample question sidebar lock and input state unsticking.

Validates that:
1. Normal chat submission leaves `_ui_rendered_while_generating` as False, maintaining
   the clean-finalize `st.rerun()` skip optimization.
2. Retry (`_retry_stopped_answer`) or sample-question flows that set `is_generating_answer=True`
   before rendering the chat input area record `_ui_rendered_while_generating = True`,
   which causes `_should_skip_final_rerun` to return False, forcing `st.rerun()` upon
   stream completion to unlock the sidebar and input area.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("IS_CI_TEST", "true")

from core.session import SessionManager
from ui.components.chat import (
    _retry_stopped_answer,
    _should_skip_final_rerun,
    render_chat_input_area,
)


def test_should_skip_final_rerun_backward_compatibility() -> None:
    assert _should_skip_final_rerun(cancelled=False, error_text=None) is True
    assert _should_skip_final_rerun(cancelled=True, error_text=None) is False
    assert _should_skip_final_rerun(cancelled=False, error_text="some error") is False
    assert (
        _should_skip_final_rerun(
            cancelled=False, error_text=None, rendered_while_generating=True
        )
        is False
    )
    assert (
        _should_skip_final_rerun(
            cancelled=False, error_text=None, rendered_while_generating=False
        )
        is True
    )


def test_normal_chat_input_does_not_set_rendered_while_generating() -> None:
    sid = "test-sid-normal"
    SessionManager.reset_all_state(sid)
    SessionManager.set("is_generating_answer", False, sid)
    SessionManager.set("last_uploaded_file_name", "doc.pdf", sid)

    mock_chat_input = MagicMock(return_value=None)
    with patch("streamlit.chat_input", mock_chat_input):
        render_chat_input_area()

    assert SessionManager.get("_ui_rendered_while_generating", True, sid) is False


def test_retry_stopped_answer_sets_rendered_while_generating() -> None:
    sid = "test-sid-retry"
    SessionManager.reset_all_state(sid)
    SessionManager.set("last_uploaded_file_name", "doc.pdf", sid)

    # Trigger retry callback which sets is_generating_answer = True
    _retry_stopped_answer(sid, "테스트 질문")
    assert SessionManager.get("is_generating_answer", False, sid) is True

    mock_chat_input = MagicMock(return_value=None)
    with patch("streamlit.chat_input", mock_chat_input):
        render_chat_input_area()

    # Because input area rendered while is_generating_answer was True, flag should be True
    assert SessionManager.get("_ui_rendered_while_generating", False, sid) is True
