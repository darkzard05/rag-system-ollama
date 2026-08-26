"""UIBridge.sync_session() 인터랙티브 키 보호 동작 검증.

sync_session()은 widget_keys.INTERACTIVE_KEYS에 등록된 키를 스냅샷/복원하여
세션 동기화가 위젯 상태(커서 위치, 입력값 등)를 덮어쓰지 않도록 보장합니다.
"""

from unittest.mock import patch

import pytest

from ui.bridge import SyncRegistry, UIBridge
from ui.widget_keys import INTERACTIVE_KEYS


@pytest.fixture(autouse=True)
def _reset_interactive_keys() -> None:
    """테스트 간 격리를 위해 기본 인터랙티브 키 세트로 복원합니다."""
    SyncRegistry._interactive_keys = set(INTERACTIVE_KEYS)


@patch("ui.bridge.ContextManager.get_current_session_id")
@patch("ui.bridge.SessionManager.sync_to_streamlit")
@patch("ui.bridge.SessionManager.has_pending_ui_sync")
@patch("ui.bridge.st")
def test_sync_session_skips_registered_keys(
    mock_st, mock_has_pending, mock_sync, mock_sid
):
    """INTERACTIVE_KEYS에 등록된 키는 동기화에서 건너뛰고, 나머지는 갱신됩니다."""
    mock_sid.return_value = "test_session"
    mock_has_pending.return_value = True

    def fake_sync(session_id, key=None):
        session_state = mock_st.session_state
        session_state["pdf_uploader"] = "new_pdf"
        session_state["some_state"] = "updated_value"
        session_state["custom_input"] = "new_user_value"

    mock_sync.side_effect = fake_sync

    session_state = {
        "pdf_uploader": "old_pdf",
        "some_state": "old_value",
        "custom_input": "original_user_value",
    }
    mock_st.session_state = session_state

    UIBridge.sync_session()

    # 1. pdf_uploader는 INTERACTIVE_KEYS 기본 세트에 포함되어 건너뛰어야 함
    assert session_state["pdf_uploader"] == "old_pdf"
    # 2. 일반 키는 갱신되어야 함
    assert session_state["some_state"] == "updated_value"


@patch("ui.bridge.ContextManager.get_current_session_id")
@patch("ui.bridge.SessionManager.sync_to_streamlit")
@patch("ui.bridge.SessionManager.has_pending_ui_sync")
@patch("ui.bridge.st")
def test_sync_session_updates_unregistered_keys(
    mock_st, mock_has_pending, mock_sync, mock_sid
):
    """등록되지 않은 키는 동기화에 의해 정상적으로 갱신됩니다."""
    mock_sid.return_value = "test_session"
    mock_has_pending.return_value = True

    def fake_sync(session_id, key=None):
        mock_st.session_state["unregistered_key"] = "new_value"

    mock_sync.side_effect = fake_sync

    session_state = {"unregistered_key": "old_value"}
    mock_st.session_state = session_state

    UIBridge.sync_session()

    assert session_state["unregistered_key"] == "new_value"


@patch("ui.bridge.ContextManager.get_current_session_id")
@patch("ui.bridge.SessionManager.sync_to_streamlit")
@patch("ui.bridge.SessionManager.has_pending_ui_sync")
@patch("ui.bridge.st")
def test_sync_session_skips_when_no_pending_change(
    mock_st, mock_has_pending, mock_sync, mock_sid
):
    """변경 사항이 없으면 동기화(sync_to_streamlit)를 호출하지 않아야 합니다."""
    mock_sid.return_value = "test_session"
    mock_has_pending.return_value = False

    UIBridge.sync_session()

    mock_sync.assert_not_called()


@patch("ui.bridge.ContextManager.get_current_session_id")
@patch("ui.bridge.SessionManager.sync_to_streamlit")
@patch("ui.bridge.SessionManager.has_pending_ui_sync")
@patch("ui.bridge.st")
def test_sync_session_runs_when_pending_change(
    mock_st, mock_has_pending, mock_sync, mock_sid
):
    """변경 사항이 있으면 동기화(sync_to_streamlit)를 호출해야 합니다."""
    mock_sid.return_value = "test_session"
    mock_has_pending.return_value = True

    def fake_sync(session_id, key=None):
        mock_st.session_state["some_state"] = "updated_value"

    mock_sync.side_effect = fake_sync
    mock_st.session_state = {"some_state": "old_value"}

    UIBridge.sync_session()

    mock_sync.assert_called_once_with("test_session")
    assert mock_st.session_state["some_state"] == "updated_value"
