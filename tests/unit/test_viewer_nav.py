import streamlit as st

from core.session import SessionManager
from ui.components.viewer import _on_page_change
from ui.widget_keys import PDF_NAV_INPUT_KEY


def test_on_page_change_direct_input_success():
    """st.number_input 조작 시 이전 값이 존재하더라도 신규 입력값으로 정상 이동하는지 검증"""
    SessionManager.reset()
    sid = "test_nav_session"
    SessionManager.set_session_id(sid)
    SessionManager.set("current_page", 1, session_id=sid)

    # 1. 과거 navigate_to_page로 인해 st.session_state에 이전 페이지 번호가 남아있는 상황 모사
    st.session_state[PDF_NAV_INPUT_KEY] = 1

    # 2. 사용자가 1페이지 위젯에서 '7'로 값을 변경한 상황
    active_widget_key = f"{PDF_NAV_INPUT_KEY}_1"
    st.session_state[active_widget_key] = 7

    # 3. 콜백 실행 (신규 규격)
    _on_page_change(active_widget_key, total_pages=10)

    # 4. 검증: 사용자가 입력한 7페이지로 정확히 이동해야 함 (1페이지로 회귀하지 않아야 함)
    assert SessionManager.get("current_page", session_id=sid) == 7


def test_on_page_change_legacy_signature_compat():
    """기존 단일 인자(total_pages) 호출 시에도 하위 호환성을 유지하는지 검증"""
    SessionManager.reset()
    sid = "test_legacy_session"
    SessionManager.set_session_id(sid)
    SessionManager.set("current_page", 2, session_id=sid)

    active_widget_key = f"{PDF_NAV_INPUT_KEY}_2"
    st.session_state[active_widget_key] = 4

    # 단일 인자로 레거시 호출
    _on_page_change(10)

    assert SessionManager.get("current_page", session_id=sid) == 4
