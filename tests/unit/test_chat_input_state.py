"""채팅 입력 상태 결정 로직(_resolve_chat_input_state) 단위 테스트.

Phase B: 4분기 → 3분기 (생성 중 / 미준비 / 준비됨) 단순화에 맞춰 기대값 동기화.
"""

from unittest.mock import patch

from core.session import SessionManager
from ui.components.chat import _resolve_chat_input_state, render_chat_input_area

_PLACEHOLDER_GENERATING = "AI가 답변을 생성 중입니다... · ■ 버튼으로 중지할 수 있습니다"
_PLACEHOLDER_NOT_READY = "Upload a PDF and ask a question"
_PLACEHOLDER_READY = "Ask a follow-up question..."


def _reset_session(sid: str) -> None:
    SessionManager.reset_all_state(sid)


def _make_ready(sid: str) -> None:
    SessionManager.set("pdf_processed", True, sid)
    SessionManager.set("rag_engine", object(), sid)
    SessionManager.set("is_building_rag", False, sid)
    SessionManager.set("needs_rag_rebuild", False, sid)
    SessionManager.set("needs_qa_chain_update", False, sid)
    SessionManager.set("pdf_processing_error", None, sid)


def test_generating_enables_input_for_stop_button():
    """생성 중에도 disabled=False를 유지하여 submit_mode=stop의 중지 버튼 표시를 보장한다."""
    sid = "input_state_gen"
    _reset_session(sid)
    SessionManager.set("is_generating_answer", True, sid)

    placeholder, disabled = _resolve_chat_input_state(sid)

    assert disabled is False
    assert placeholder == _PLACEHOLDER_GENERATING


def test_not_ready_disables_input_without_pdf():
    sid = "input_state_no_pdf"
    _reset_session(sid)

    placeholder, disabled = _resolve_chat_input_state(sid)

    assert disabled is True
    assert placeholder == _PLACEHOLDER_NOT_READY


def test_not_ready_disables_input_while_processing():
    sid = "input_state_processing"
    _reset_session(sid)
    SessionManager.set("pdf_file_path", "C:/tmp/doc.pdf", sid)

    placeholder, disabled = _resolve_chat_input_state(sid)

    assert disabled is True
    assert placeholder == _PLACEHOLDER_NOT_READY


def test_not_ready_disables_input_on_error():
    sid = "input_state_error"
    _reset_session(sid)
    SessionManager.set("pdf_file_path", "C:/tmp/doc.pdf", sid)
    SessionManager.set("pdf_processing_error", "파싱 실패", sid)

    placeholder, disabled = _resolve_chat_input_state(sid)

    assert disabled is True
    assert placeholder == _PLACEHOLDER_NOT_READY


def test_ready_enables_input():
    sid = "input_state_ready"
    _reset_session(sid)
    _make_ready(sid)

    placeholder, disabled = _resolve_chat_input_state(sid)

    assert disabled is False
    assert placeholder == _PLACEHOLDER_READY


def test_render_chat_input_area_generating_shows_status_caption() -> None:
    """UX-4: 생성 중 입력 영역에 상태 캡션이 placeholder 텍스트로 표시된다."""
    sid = "input_area_generating"
    _reset_session(sid)
    SessionManager.set_session_id(sid)
    SessionManager.set("is_generating_answer", True, sid)

    with patch("ui.components.chat.st") as mock_st:
        mock_st.chat_input.return_value = None
        render_chat_input_area()

    captions = [call.args[0] for call in mock_st.caption.call_args_list if call.args]
    assert _PLACEHOLDER_GENERATING in captions


def test_render_chat_input_area_idle_has_no_status_caption() -> None:
    """UX-4: 비생성 상태에서는 상태 캡션이 렌더되지 않는다."""
    sid = "input_area_idle"
    _reset_session(sid)
    SessionManager.set_session_id(sid)
    SessionManager.set("is_generating_answer", False, sid)

    with patch("ui.components.chat.st") as mock_st:
        mock_st.chat_input.return_value = None
        render_chat_input_area()

    mock_st.caption.assert_not_called()
