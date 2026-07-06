"""
Session Management Helpers for RAG System tests.
Provides utilities to easily manage and verify session state during tests.
"""

from contextlib import contextmanager
from src.core.session import SessionManager


@contextmanager
def with_session(session_id: str):
    """
    특정 세션 ID를 현재 컨텍스트로 설정하고,
    블록 종료 후 원래 세션으로 복구하는 컨텍스트 매니저입니다.
    """
    original_sid = SessionManager.get_session_id()
    SessionManager.set_session_id(session_id)
    try:
        yield
    finally:
        SessionManager.set_session_id(original_sid)


def assert_session_value(key: str, expected_value, session_id: str = None):
    """
    세션 내 특정 키의 값이 기대값과 일치하는지 검증합니다.
    """
    actual_value = SessionManager.get(key, session_id=session_id)
    assert actual_value == expected_value, (
        f"Session key '{key}' expected {expected_value}, but got {actual_value}"
    )


def assert_session_key_exists(key: str, session_id: str = None):
    """
    세션 내에 특정 키가 존재하는지 검증합니다.
    """
    # create=False를 통해 기본값 생성을 방지하고 존재 여부만 확인
    val = SessionManager.get(key, default=None, session_id=session_id, create=False)
    assert val is not None, f"Session key '{key}' should exist but was not found"
