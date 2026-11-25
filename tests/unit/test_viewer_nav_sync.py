from unittest.mock import MagicMock, patch
from core.session import SessionManager
from ui.components.viewer import render_pdf_controls, _on_page_change
from ui.widget_keys import PDF_NAV_INPUT_KEY


def test_render_pdf_controls_always_uses_dynamic_key_and_value():
    """render_pdf_controls가 current_page에 기반한 dyn_key와 value=current_page를 정확히 바인딩하는지 검증"""
    current_page = 15
    total_pages = 50
    expected_dyn_key = f"{PDF_NAV_INPUT_KEY}_{current_page}"

    mock_session_state = {}
    with patch("ui.components.viewer.st") as mock_st:
        mock_st.session_state = mock_session_state
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        render_pdf_controls(current_page, total_pages)

        # st.number_input 호출 인자 검증
        mock_st.number_input.assert_called_once()
        _, kwargs = mock_st.number_input.call_args
        assert kwargs.get("key") == expected_dyn_key
        assert kwargs.get("value") == current_page
        assert kwargs.get("args") == (total_pages,)


def test_on_page_change_navigates_correctly_with_dynamic_key():
    """동적 키로부터 사용자 입력값을 읽어 올바른 페이지로 네비게이션하는지 검증"""
    SessionManager.reset()
    sid = "test_sync_session"
    SessionManager.set_session_id(sid)
    SessionManager.set("current_page", 2, session_id=sid)

    dyn_key = f"{PDF_NAV_INPUT_KEY}_2"
    user_input_page = 25
    total_pages = 50

    mock_session_state = {dyn_key: user_input_page}
    with (
        patch("ui.components.viewer.st") as mock_st,
        patch("ui.components.viewer.navigate_to_page") as mock_navigate,
    ):
        mock_st.session_state = mock_session_state

        _on_page_change(total_pages)

        mock_navigate.assert_called_once_with(user_input_page)
