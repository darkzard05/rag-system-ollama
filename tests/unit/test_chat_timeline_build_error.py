"""채팅 타임라인(_render_unified_timeline)의 build_error 메시지 렌더링 검증 단위 테스트."""

from unittest.mock import call, patch

from core.session import SessionManager
from ui.components.chat import _render_unified_timeline
from ui.components.common import AVATARS


def test_timeline_normal_no_build_error(monkeypatch):
    """정상 케이스: build_progress/log만 존재하고 build_error가 없으면 온보딩만 호출되고 ui_error는 호출되지 않음."""
    sid = "sid_normal"
    SessionManager.reset_all_state(sid)
    SessionManager.add_message(
        "system", "build progress", msg_type="build_progress", session_id=sid
    )
    SessionManager.add_message("system", "system log", msg_type="log", session_id=sid)

    with (
        patch(
            "core.session.SessionManager.get_messages",
            return_value=SessionManager.get_messages(sid),
        ),
        patch("ui.components.chat._render_guidance_panel") as mock_guidance,
        patch("ui.components.chat.ui_error") as mock_ui_error,
        patch("streamlit.chat_message"),
        patch("streamlit.caption"),
    ):
        _render_unified_timeline(sid)
        mock_guidance.assert_called_once()
        mock_ui_error.assert_not_called()


def test_timeline_pre_chat_build_error_assistant(monkeypatch):
    """버그 재현 케이스 1: 대화 시작 전 assistant 역할의 build_error(_post_upload_error 경로)가 렌더링됨."""
    sid = "sid_pre_chat_err"
    SessionManager.reset_all_state(sid)
    SessionManager.add_message(
        "assistant", "Invalid PDF file.", msg_type="build_error", session_id=sid
    )

    with (
        patch(
            "core.session.SessionManager.get_messages",
            return_value=SessionManager.get_messages(sid),
        ),
        patch("ui.components.chat._render_guidance_panel") as mock_guidance,
        patch("ui.components.chat.ui_error") as mock_ui_error,
        patch("streamlit.chat_message") as mock_chat_message,
        patch("streamlit.caption"),
    ):
        _render_unified_timeline(sid)
        mock_guidance.assert_called_once()
        mock_chat_message.assert_called_with("assistant", avatar=AVATARS["error"])
        mock_ui_error.assert_called_once_with("Invalid PDF file.")


def test_timeline_post_chat_mixed_build_errors(monkeypatch):
    """버그 재현 케이스 2: 대화 진행 후 assistant 및 system build_error 혼재 시 각각 오류 말풍선으로 렌더링됨."""
    sid = "sid_post_chat_err"
    SessionManager.reset_all_state(sid)
    SessionManager.add_message("user", "Hello", session_id=sid)
    SessionManager.add_message("assistant", "Hi there", session_id=sid)
    SessionManager.add_message(
        "assistant", "Swap failed", msg_type="build_error", session_id=sid
    )
    SessionManager.add_message(
        "system", "", msg_type="build_error", error="Build failed", session_id=sid
    )

    with (
        patch(
            "core.session.SessionManager.get_messages",
            return_value=SessionManager.get_messages(sid),
        ),
        patch("ui.components.chat.ui_error") as mock_ui_error,
        patch("streamlit.chat_message") as mock_chat_message,
        patch("ui.components.chat.render_message") as mock_render_msg,
    ):
        _render_unified_timeline(sid)
        # Should call ui_error for both build_errors
        assert mock_ui_error.call_args_list == [
            call("Swap failed"),
            call("Build failed"),
        ]
        # Verify chat_message was called with appropriate avatars/roles
        mock_chat_message.assert_any_call("assistant", avatar=AVATARS["error"])
        mock_chat_message.assert_any_call("system", avatar=AVATARS["error"])
        # Ensure regular render_message wasn't called for the build_error messages
        for call_item in mock_render_msg.call_args_list:
            assert call_item.kwargs.get("msg_type") != "build_error"
