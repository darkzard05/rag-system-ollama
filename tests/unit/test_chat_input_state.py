"""채팅 입력 상태 결정 로직(_resolve_chat_input_state) 단위 테스트.

Phase B: 4분기 → 3분기 (생성 중 / 미준비 / 준비됨) 단순화에 맞춰 기대값 동기화.
"""

from unittest.mock import patch

from core.session import SessionManager
from ui.components.chat import _resolve_chat_input_state, render_chat_input_area
from ui.strings import set_lang, t

# Placeholders are i18n-driven (src/ui/components/chat.py::_resolve_chat_input_state
# via t() keys). Tests pin the Korean locale to assert exact strings.
_PLACEHOLDER_GENERATING = "chat_placeholder_generating"
_PLACEHOLDER_SWAPPING = "chat_placeholder_swapping"
_PLACEHOLDER_BUILDING = "chat_placeholder_building"
_PLACEHOLDER_NO_PDF = "chat_placeholder_no_pdf"
_PLACEHOLDER_NOT_READY = "chat_placeholder_not_ready"
_PLACEHOLDER_READY = "chat_placeholder_ready"
_PLACEHOLDER_FOLLOWUP = "chat_placeholder_next"


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

    set_lang("ko")
    try:
        placeholder, disabled = _resolve_chat_input_state(sid)
        expected = t(_PLACEHOLDER_GENERATING)
    finally:
        set_lang("en")

    assert disabled is False
    assert placeholder == expected


def test_not_ready_disables_input_without_pdf():
    sid = "input_state_no_pdf"
    _reset_session(sid)

    set_lang("ko")
    try:
        placeholder, disabled = _resolve_chat_input_state(sid)
        expected = t(_PLACEHOLDER_NO_PDF)
    finally:
        set_lang("en")

    assert disabled is True
    assert placeholder == expected


def test_not_ready_disables_input_while_processing():
    sid = "input_state_processing"
    _reset_session(sid)
    SessionManager.set("last_uploaded_file_name", "doc.pdf", sid)
    SessionManager.set("is_building_rag", True, sid)

    set_lang("ko")
    try:
        placeholder, disabled = _resolve_chat_input_state(sid)
        expected = t(_PLACEHOLDER_BUILDING)
    finally:
        set_lang("en")

    assert disabled is True
    assert placeholder == expected


def test_not_ready_disables_input_on_error():
    sid = "input_state_error"
    _reset_session(sid)
    SessionManager.set("last_uploaded_file_name", "doc.pdf", sid)
    SessionManager.set("pdf_processing_error", "파싱 실패", sid)

    set_lang("ko")
    try:
        placeholder, disabled = _resolve_chat_input_state(sid)
        expected = t(_PLACEHOLDER_NOT_READY)
    finally:
        set_lang("en")

    assert disabled is True
    assert placeholder == expected


def test_ready_enables_input():
    sid = "input_state_ready"
    _reset_session(sid)
    _make_ready(sid)

    set_lang("ko")
    try:
        placeholder, disabled = _resolve_chat_input_state(sid)
        expected = t(_PLACEHOLDER_READY)
    finally:
        set_lang("en")

    assert disabled is False
    assert placeholder == expected


def test_render_chat_input_area_generating_shows_status_caption() -> None:
    """generating 입력 영역은 status caption 없이 chat_input placeholder로 안내한다."""
    sid = "input_area_generating"
    _reset_session(sid)
    SessionManager.set_session_id(sid)
    SessionManager.set("is_generating_answer", True, sid)

    set_lang("ko")
    try:
        with patch("ui.components.chat.st") as mock_st:
            mock_st.chat_input.return_value = None
            render_chat_input_area()
        expected = t(_PLACEHOLDER_GENERATING)
    finally:
        set_lang("en")

    assert mock_st.chat_input.called, "st.chat_input must be rendered"
    _, kwargs = mock_st.chat_input.call_args
    args = mock_st.chat_input.call_args.args or ()
    placeholder = args[0] if args else kwargs.get("placeholder")
    assert placeholder == expected
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
