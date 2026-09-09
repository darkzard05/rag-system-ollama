"""Session cleanup — migrated from scripts/verification/verify_session_cleanup.py.

원본 스크립트는 두 세션을 만들고 만료 세션의 ``last_accessed`` 를 과거로
조작해 ``SessionManager.cleanup_expired_sessions`` 가 만료 세션만 제거하는지
print 로 확인했습니다. 여기서는 ``session_context`` fixture 와 전역 상태
리셋(conftest.autouse)을 사용해 동일 흐름을 assert 로 검증합니다.
라이브 모델/네트워크 의존성 없음.
"""

import time

from core.session import SessionManager


def test_cleanup_removes_expired_and_keeps_active(session_context: str) -> None:
    sid_active = session_context
    sid_expired = "expired_" + sid_active

    SessionManager.set("data", "active", session_id=sid_active)
    SessionManager.set("data", "to_be_expired", session_id=sid_expired)

    # 만료 세션의 last_accessed 를 과거로 조작 (TTL 3600초 기준, 5000초 전).
    state_expired = SessionManager._get_state(sid_expired)
    state_expired["last_accessed"] = time.time() - 5000

    SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)

    assert sid_active in SessionManager._fallback_sessions
    assert sid_expired not in SessionManager._fallback_sessions


def test_cleanup_keeps_recently_accessed_sessions(session_context: str) -> None:
    sid = session_context
    SessionManager.set("data", "value", session_id=sid)

    SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)

    assert sid in SessionManager._fallback_sessions
    assert SessionManager.get("data", session_id=sid) == "value"


def test_cleanup_removes_sessions_idle_beyond_ttl(session_context: str) -> None:
    sid_idle = "idle_" + session_context
    SessionManager.set("data", "idle", session_id=sid_idle)

    # 10분(600초) 유휴 상태를 60초 TTL 기준으로 판정 → 제거 대상.
    state = SessionManager._get_state(sid_idle)
    state["last_accessed"] = time.time() - 600

    SessionManager.cleanup_expired_sessions(max_idle_seconds=60)

    assert sid_idle not in SessionManager._fallback_sessions
