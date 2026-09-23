"""채팅 입력 상태 결정 로직(_resolve_chat_input_state) 단위 테스트.

Phase B: 4분기 → 3분기 (생성 중 / 미준비 / 준비됨) 단순화에 맞춰 기대값 동기화.
"""

from unittest.mock import patch

from core.session import SessionManager
from ui.components.chat import _resolve_chat_input_state, render_chat_input_area

# New 6-state spec in src/ui/components/chat.py::_resolve_chat_input_state
# (Korean literals, no longer driven by t() i18n keys).
_PLACEHOLDER_GENERATING = "답변을 생성하고 있습니다... (중지는 우측 하단 ■ 버튼)"
_PLACEHOLDER_SWAPPING = "AI 모델을 전환하는 중입니다... 잠시만 기다려 주세요."
_PLACEHOLDER_BUILDING = "문서 지식을 분석하고 있습니다... (완료 후 질문 가능)"
_PLACEHOLDER_NO_PDF = "좌측 사이드바에서 PDF 문서를 먼저 업로드해 주세요."
_PLACEHOLDER_NOT_READY = "문서 인덱스를 준비 중입니다. 잠시만 기다려 주세요."
_PLACEHOLDER_READY = "문서 내용에 대해 궁금한 점을 질문해 보세요."
_PLACEHOLDER_FOLLOWUP = "이어서 추가 질문을 입력하세요..."


def _reset_session(sid: str) -> None:
    SessionManager.reset_all_state(sid)


def _make_ready(sid: str) -> None:
    SessionManager.set("last_uploaded_file_name", "doc.pdf", sid)
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
    assert placeholder == _PLACEHOLDER_NO_PDF


def test_not_ready_disables_input_while_processing():
    sid = "input_state_processing"
    _reset_session(sid)
    SessionManager.set("last_uploaded_file_name", "doc.pdf", sid)
    SessionManager.set("is_building_rag", True, sid)

    placeholder, disabled = _resolve_chat_input_state(sid)

    assert disabled is True
    assert placeholder == _PLACEHOLDER_BUILDING


def test_not_ready_disables_input_on_error():
    sid = "input_state_error"
    _reset_session(sid)
    SessionManager.set("last_uploaded_file_name", "doc.pdf", sid)
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
    """generating 입력 영역은 status caption 없이 chat_input placeholder로 안내한다."""
    sid = "input_area_generating"
    _reset_session(sid)
    SessionManager.set_session_id(sid)
    SessionManager.set("is_generating_answer", True, sid)

    with patch("ui.components.chat.st") as mock_st:
        mock_st.chat_input.return_value = None
        render_chat_input_area()

    assert mock_st.chat_input.called, "st.chat_input must be rendered"
    _, kwargs = mock_st.chat_input.call_args
    args = mock_st.chat_input.call_args.args or ()
    placeholder = args[0] if args else kwargs.get("placeholder")
    assert placeholder == _PLACEHOLDER_GENERATING
    assert kwargs.get("disabled") is False
    mock_st.caption.assert_not_called()


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
