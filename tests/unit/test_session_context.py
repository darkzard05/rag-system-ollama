# 세션 컨텍스트 관리를 위한 ContextManager 유닛 테스트
import pytest
from src.core.session.context import ContextManager


@pytest.fixture(autouse=True)
def reset_context():
    """테스트 간 ContextManager 전역 상태 정리."""
    ContextManager.set_current_session_id(None)
    yield
    ContextManager.set_current_session_id(None)


def test_context_manager_initial_state():
    # 초기 상태에서는 session_id가 None이어야 함
    assert ContextManager.get_current_session_id() is None


def test_context_manager_set_and_get():
    # session_id를 설정하고 다시 가져올 수 있어야 함
    test_session_id = "test-session-123"
    ContextManager.set_current_session_id(test_session_id)
    assert ContextManager.get_current_session_id() == test_session_id


def test_context_manager_reset():
    # None으로 설정하여 초기화할 수 있어야 함 (또는 None 허용 확인)
    ContextManager.set_current_session_id(None)
    assert ContextManager.get_current_session_id() is None
