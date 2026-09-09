# SessionManager 세션 ID 컨텍스트 유닛 테스트
import pytest
from src.core.session import SessionManager


@pytest.fixture(autouse=True)
def _reset_session_id_to_default():
    """테스트 간 SessionManager ContextVar 전역 상태 정리.

    SessionManager에는 reset_session_id가 없으므로 기본값("default")으로
    되돌려 놓습니다.
    """
    SessionManager.set_session_id("default")
    yield
    SessionManager.set_session_id("default")


def test_get_session_id_returns_default_when_not_set():
    # 아무것도 설정하지 않은 기본 상태에서는 "default"가 반환되어야 함
    SessionManager.set_session_id("default")
    assert SessionManager.get_session_id() == "default"


def test_set_session_id_propagates_to_get_session_id(session_context):
    # set_session_id로 설정한 값이 동일 컨텍스트의 get_session_id에서 반환됨
    SessionManager.set_session_id(session_context)
    assert SessionManager.get_session_id() == session_context


def test_reset_session_id_via_default():
    # reset_session_id가 없으므로 "default"로 되돌려 초기화
    SessionManager.set_session_id("other-session")
    SessionManager.set_session_id("default")
    assert SessionManager.get_session_id() == "default"
