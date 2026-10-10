"""SSoT: 채팅 오류 표면은 ui_error 단일 경로를 경유한다 (Task 3a/3b).

- 3a: ``chat._draw_streaming_message`` / ``chat.render_chat_input_area`` /
  ``chat_build._render_build_progress_block`` 의 직접 ``st.error`` 호출이
  ``ui_error`` 를 경유하고, 원문 메시지가 ``st.error`` 에 그대로 도달한다.
- 3b: 스트리밍 상태줄은 한 칸 미들닷(``" · "``) 단일 규격을 사용한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.session import SessionManager
from ui.components import chat as chat_mod
from ui.components import common as common_mod


def test_draw_streaming_message_routes_error_through_ui_error() -> None:
    """스트리밍 오류 메시지가 ui_error를 경유해 st.error에 원문 그대로 도달."""
    sid = "ssot_stream_err"
    raw = "ERR-RAW-1"
    msg = {
        "role": "assistant",
        "content": "partial",
        "msg_type": "streaming",
        "error": raw,
    }
    with (
        patch("ui.components.chat.st") as mock_st,
        patch(
            "ui.components.chat.ui_error",
            side_effect=lambda m: mock_st.error(m),
        ) as mock_ui_error,
    ):
        chat_mod._draw_streaming_message(msg, sid)
    mock_ui_error.assert_called_once_with(str(raw))
    assert mock_st.error.call_args is not None
    assert mock_st.error.call_args.args[0] == str(raw)


def test_chat_input_disabled_routes_placeholder_through_ui_error() -> None:
    """비활성 입력창 제출 시 플레이스홀더가 ui_error를 경유해 그대로 도달."""
    placeholder = "PLACEHOLDER-SSOT-1"
    with (
        patch("ui.components.chat.st") as mock_st,
        patch(
            "ui.components.chat.ui_error",
            side_effect=lambda m: mock_st.error(m),
        ) as mock_ui_error,
        patch(
            "ui.components.chat._resolve_chat_input_state",
            return_value=(placeholder, True),
        ),
    ):
        mock_st.session_state.pop.return_value = None
        mock_st.chat_input.return_value = "hello?"
        chat_mod.render_chat_input_area()
    mock_ui_error.assert_called_once_with(placeholder)
    assert mock_st.error.call_args is not None
    assert mock_st.error.call_args.args[0] == placeholder


def test_build_progress_error_routes_message_through_ui_error() -> None:
    """빌드 오류 메시지가 ui_error를 경유해 st.error에 원문 그대로 도달."""
    import ui.components.doc_analysis.panel as panel_mod
    from core.build_state import fail_build, initial_build_state

    sid = "ssot_build_err"
    SessionManager.reset_all_state(sid)
    build = initial_build_state("doc.pdf", "h1")
    fail_build(build, "RAW-BOOM-9")
    SessionManager.set("build_state", build, session_id=sid)
    SessionManager.set("pdf_processing_error", "RAW-BOOM-9", session_id=sid)
    with (
        patch("ui.components.doc_analysis.panel.st") as mock_st,
        patch(
            "ui.components.doc_analysis.panel.ui_error",
            side_effect=lambda m: mock_st.error(m),
        ) as mock_ui_error,
    ):
        mock_st.status.return_value.__enter__.return_value = MagicMock()
        panel_mod.render_document_panel(sid)
    mock_ui_error.assert_called_once_with("RAW-BOOM-9")
    assert mock_st.error.call_args is not None
    assert mock_st.error.call_args.args[0] == "RAW-BOOM-9"


def test_ui_error_passes_message_through_unchanged() -> None:
    """ui_error는 순수 패스스루다: 특수문자·개행이 변경 없이 st.error로 전달."""
    payload = "<script>alert(1)</script>\nsecond line & <b>bold</b>"
    with patch("ui.components.common.st") as mock_st:
        common_mod.ui_error(payload)
    assert mock_st.error.call_args is not None
    assert mock_st.error.call_args.args[0] == payload


def test_proc_phase_caption_uses_single_space_middot() -> None:
    """스트리밍 상태줄은 한 칸 미들닷으로 단계를 연결하고 커서를 유지한다."""
    line = chat_mod._proc_phase_caption("zzz-no-such-phase-qqq", 7.0)
    assert "  ·  " not in line
    assert " · " in line
    assert "▍" in line
    assert line.endswith("7s ▍")


def test_proc_phase_caption_skips_empty_status_text() -> None:
    """빈 status_text가 이중 구분자로 남지 않고, 빈 조각은 건너뛴다."""
    line = chat_mod._proc_phase_caption("", 3.0)
    assert "  ·  " not in line
    assert "· ·" not in line
    assert line.endswith("3s ▍")
    assert common_mod.status_line("a", "", "b") == "a · b"
