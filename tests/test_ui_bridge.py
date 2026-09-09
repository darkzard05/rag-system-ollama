"""UIBridge.sync_session() 동작 검증.

sync_session()은 SessionManager.sync_to_streamlit()에 위젯 키를 직접 쓰는
대신 동기화를 위임하고, 저장소 키만 미러링합니다. 위젯 키는 절대 스냅샷/복원
되지 않습니다 (Streamlit 1.54는 스크립트 측 위젯 키 대입을 금지).
"""

from unittest.mock import patch

from ui.bridge import UIBridge


@patch("ui.bridge.SessionManager.get_session_id")
@patch("ui.bridge.SessionManager.sync_to_streamlit")
@patch("ui.bridge.SessionManager.has_pending_ui_sync")
def test_sync_session_updates_unregistered_keys(mock_has_pending, mock_sync, mock_sid):
    """등록되지 않은 키는 동기화에 의해 정상적으로 갱신됩니다."""
    mock_sid.return_value = "test_session"
    mock_has_pending.return_value = True

    def fake_sync(session_id, key=None):
        session_state["unregistered_key"] = "new_value"

    mock_sync.side_effect = fake_sync

    session_state = {"unregistered_key": "old_value"}

    UIBridge.sync_session()

    assert session_state["unregistered_key"] == "new_value"


@patch("ui.bridge.SessionManager.get_session_id")
@patch("ui.bridge.SessionManager.sync_to_streamlit")
@patch("ui.bridge.SessionManager.has_pending_ui_sync")
def test_sync_session_skips_when_no_pending_change(
    mock_has_pending, mock_sync, mock_sid
):
    """변경 사항이 없으면 동기화(sync_to_streamlit)를 호출하지 않아야 합니다."""
    mock_sid.return_value = "test_session"
    mock_has_pending.return_value = False

    UIBridge.sync_session()

    mock_sync.assert_not_called()


@patch("ui.bridge.SessionManager.get_session_id")
@patch("ui.bridge.SessionManager.sync_to_streamlit")
@patch("ui.bridge.SessionManager.has_pending_ui_sync")
def test_sync_session_runs_when_pending_change(mock_has_pending, mock_sync, mock_sid):
    """변경 사항이 있으면 동기화(sync_to_streamlit)를 호출해야 합니다."""
    mock_sid.return_value = "test_session"
    mock_has_pending.return_value = True

    def fake_sync(session_id, key=None):
        session_state["some_state"] = "updated_value"

    mock_sync.side_effect = fake_sync

    session_state = {"some_state": "old_value"}

    UIBridge.sync_session()

    mock_sync.assert_called_once_with("test_session")
    assert session_state["some_state"] == "updated_value"


@patch("ui.bridge.SessionManager.get_session_id")
@patch("ui.bridge.SessionManager.sync_to_streamlit")
@patch("ui.bridge.SessionManager.has_pending_ui_sync")
def test_sync_session_delegates_to_sync_to_streamlit(
    mock_has_pending, mock_sync, mock_sid
):
    """sync_session은 위젯 키를 직접 쓰지 않고 sync_to_streamlit에 위임합니다.

    스냅샷/복원 블록이 제거된 후에도 동기화는 저장소 키만 미러링하고
    위젯 키(예: pdf_uploader)는 bridge 코드가 직접 건드리지 않아야 합니다.
    """
    mock_sid.return_value = "test_session"
    mock_has_pending.return_value = True

    def fake_sync(session_id, key=None):
        session_state["some_state"] = "v"

    mock_sync.side_effect = fake_sync

    session_state = {
        "some_state": "old_value",
        "pdf_uploader": "untouched_pdf",
    }

    UIBridge.sync_session()

    # sync_to_streamlit에 위임되었는지 확인
    mock_sync.assert_called_once_with("test_session")
    # 저장소(비위젯) 키는 동기화에 의해 갱신됨
    assert session_state["some_state"] == "v"
    # 위젯 키는 bridge 코드가 직접 쓰지 않아 원래 값이 유지됨
    assert session_state["pdf_uploader"] == "untouched_pdf"
