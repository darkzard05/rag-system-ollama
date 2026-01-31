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

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.append(os.path.abspath("src"))

from src.api.api_server import app

from core.session import SessionManager
from core.thread_safe_session import MAX_MESSAGE_HISTORY, ThreadSafeSessionManager

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Cleanup sessions before and after each test."""
    ThreadSafeSessionManager._fallback_sessions.clear()
    yield
    ThreadSafeSessionManager._fallback_sessions.clear()


def test_standalone_session_ops():
    """Basic set/get and status logging."""
    sid = "test_standalone"
    ThreadSafeSessionManager.set_session_id(sid)
    ThreadSafeSessionManager.init_session()

    SessionManager.set("test_key", "hello")
    assert SessionManager.get("test_key") == "hello"

    SessionManager.add_status_log("Log 1")
    assert "Log 1" in SessionManager.get("status_logs")


def test_message_persistence_and_limits():
    """Test message accumulation and MAX_HISTORY constraint."""
    sid = "test_limits"
    ThreadSafeSessionManager.set_session_id(sid)
    ThreadSafeSessionManager.init_session()

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


def test_object_reassignment_for_streamlit():
    """Ensure mutable objects are reassigned to trigger Streamlit change detection."""
    sid = "test_reassignment"
    ThreadSafeSessionManager.set_session_id(sid)
    ThreadSafeSessionManager.init_session()

    state = ThreadSafeSessionManager._get_state()

    SessionManager.add_message("user", "v1")
    list_v1 = state.get("messages")

    SessionManager.add_message("user", "v2")
    list_v2 = state.get("messages")

    # Must be different objects
    assert list_v1 is not list_v2


def test_api_session_isolation():
    """Verify that different session IDs in API headers keep data isolated."""
    session_a = "user-a"
    session_b = "user-b"

    # [인증] 테스트용 API 키 등록 및 헤더 설정
    api_key = "sk_admin_test_token_12345"
    from src.api.api_server import TEST_USER, auth_manager

    auth_manager._api_keys[api_key] = TEST_USER
    headers = {"Authorization": f"Bearer {api_key}"}

    # User A tries to upload (ignoring actual processing for isolation check)
    pdf_content = b"%PDF-1.4 mock content"
    files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}

    # Note: Using session_id form data as expected by api_server.py
    client.post(
        "/api/v1/upload", files=files, data={"session_id": session_a}, headers=headers
    )

    # User B queries without uploading
    response_b = client.post(
        "/api/v1/query",
        json={"query": "Where is my file?", "session_id": session_b},
        headers=headers,
    )

    # Should fail for User B
    assert response_b.status_code == 400
    assert "문서를 업로드" in response_b.json()["detail"]
