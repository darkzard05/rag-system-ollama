import pytest
from unittest.mock import patch
from ui.components.chat import _render_unified_timeline


def test_ui_timeline_guidance_renders_when_only_system_messages_exist(monkeypatch):
    """시스템 로그나 READY_FOR_QUERY만 존재할 때 대화 메시지가 비어있어 온보딩 가이드가 호출되는지 검증합니다."""
    mock_messages = [
        {"role": "system", "msg_type": "log", "content": "READY_FOR_QUERY"},
        {"role": "system", "msg_type": "build_error", "error": "test error"},
    ]
    with (
        patch("core.session.SessionManager.get_messages", return_value=mock_messages),
        patch("ui.components.chat._render_guidance_panel") as mock_guidance,
        patch("streamlit.chat_message"),
        patch("streamlit.caption"),
    ):
        _render_unified_timeline("test_sid")
        mock_guidance.assert_called_once()
