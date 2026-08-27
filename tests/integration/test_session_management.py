"""
Unified session management tests.
Covers:
- Standalone session operations
- Thread safety and object reassignment
- Message history limits
- Session isolation in API context
"""

import io
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.append(os.path.abspath("src"))

from src.api.api_server import app

from common.constants import MAX_MESSAGE_HISTORY
from core.session import SessionManager
from ui.session_sync import StreamlitSessionSync

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Cleanup sessions before and after each test."""
    SessionManager.reset()
    yield
    SessionManager.reset()


def test_standalone_session_ops():
    """Basic set/get and status logging."""
    sid = "test_standalone"
    SessionManager.set_session_id(sid)
    SessionManager.init_session()

    SessionManager.set("test_key", "hello")
    assert SessionManager.get("test_key") == "hello"

    SessionManager.add_status_log("Log 1")
    assert "Log 1" in SessionManager.get("status_logs")


def test_message_persistence_and_limits():
    """Test message accumulation and MAX_HISTORY constraint."""
    sid = "test_limits"
    SessionManager.set_session_id(sid)
    SessionManager.init_session()

    # 1. Basic persistence
    SessionManager.add_message("user", "Hello")
    SessionManager.add_message("assistant", "Hi")
    msgs = SessionManager.get_messages()
    assert len(msgs) == 2

    # 2. History limit
    for i in range(MAX_MESSAGE_HISTORY + 10):
        SessionManager.add_message("user", f"msg {i}")

    msgs = SessionManager.get_messages()
    assert len(msgs) == MAX_MESSAGE_HISTORY
    assert msgs[-1]["content"] == f"msg {MAX_MESSAGE_HISTORY + 9}"


def test_streamlit_sync_reassigns_mutable_objects():
    """sync_to_streamlit은 긴 리스트를 새 객체로 할당해 Streamlit 변경 감지를 트리거합니다.

    (신규 SessionManager는 메시지 리스트를 제자리(in-place)에서 변경하므로,
    UI 동기화 시점에 리스트 복사본을 할당합니다.)
    """
    sid = "test_reassignment"
    SessionManager.set_session_id(sid)
    SessionManager.init_session()

    for i in range(15):
        SessionManager.add_message("user", f"v{i}")

    fallback_messages = SessionManager.get_messages()
    mock_state = {}

    # 신규 SessionManager는 set_ui_sync로 주입된 StreamlitSessionSync 어댑터를
    # 통해서만 UI 상태를 미러링한다. 테스트에서도 동일하게 어댑터를 설치해야
    # sync_to_streamlit()이 st.session_state로 값을 기록한다.
    SessionManager.set_ui_sync(StreamlitSessionSync)
    try:
        with (
            patch.object(SessionManager, "_is_streamlit_running", return_value=True),
            patch(
                "streamlit.runtime.scriptrunner.get_script_run_ctx",
                return_value=MagicMock(),
            ),
            patch("streamlit.session_state", mock_state),
        ):
            SessionManager.sync_to_streamlit()

        assert "messages" in mock_state
        # 10개 초과 리스트는 복사본(val[:])이 할당되어 객체 재할당이 발생합니다.
        assert mock_state["messages"] is not fallback_messages
        assert mock_state["messages"] == fallback_messages
    finally:
        SessionManager.set_ui_sync(None)


def test_api_session_isolation():
    """Verify that different session IDs in API headers keep data isolated."""
    session_a = "user-a"
    session_b = "user-b"

    # [인증] 테스트용 API 키 등록 및 헤더 설정
    api_key = "sk_admin_test_token_12345"
    from src.api.api_server import TEST_USER, auth_manager

    auth_manager.register_fixed_api_key(TEST_USER, api_key)
    headers = {"Authorization": f"Bearer {api_key}"}

    # User A tries to upload (ignoring actual processing for isolation check)
    pdf_content = b"%PDF-1.4 mock content"
    files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}

    # [우회] 업로드 엔드포인트의 무거운 리소스(임베딩 로더, RAGSystem)를 목킹합니다.
    mock_rag_system = MagicMock()
    mock_rag_system.return_value.build_pipeline = AsyncMock(
        return_value=("인덱싱 완료", False)
    )

    with (
        patch("src.api.api_server.RAGSystem", mock_rag_system),
        patch(
            "src.core.resource_manager.ResourceManager.get_embedder_for_session",
            new_callable=AsyncMock,
        ),
    ):
        response_a = client.post(
            "/api/v1/upload",
            files=files,
            data={"session_id": session_a},
            headers=headers,
        )
        assert response_a.status_code == 200

    # User B queries without uploading
    response_b = client.post(
        "/api/v1/query",
        json={"query": "Where is my file?", "session_id": session_b},
        headers=headers,
    )

    # Should fail for User B
    assert response_b.status_code == 400
    assert "문서를 업로드" in response_b.json()["detail"]
