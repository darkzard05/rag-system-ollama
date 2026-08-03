"""
세션 식별 정상화 검증 (문제 2A).

스레드 이름 기반의 임의 세션 생성이 제거되어야 합니다.
- 미설정 시 get_session_id()는 "default"를 반환
- 스레드 이름으로 세션이 조작(fabrication)되지 않음
- 명시적 set_session_id()가 우선
"""

import threading

from core.session import SessionManager


def test_unset_returns_default():
    SessionManager.reset()
    assert SessionManager.get_session_id() == "default"


def test_no_thread_name_session_fabrication():
    SessionManager.reset()
    sid1 = SessionManager.get_session_id()
    sid2 = SessionManager.get_session_id()
    assert sid1 == sid2 == "default"
    # 스레드 이름 → 세션 매핑 저장소는 존재하지 않아야 함
    assert not hasattr(SessionManager, "_thread_session_map")


def test_get_session_id_stable_across_threads():
    SessionManager.reset()
    results: list[str] = []

    def worker():
        results.append(SessionManager.get_session_id())

    t = threading.Thread(target=worker, name="SomeUniqueWorkerThread-xyz")
    t.start()
    t.join()

    assert results == ["default"]


def test_explicit_set_wins_over_default():
    SessionManager.reset()
    SessionManager.set_session_id("explicit_sid")
    assert SessionManager.get_session_id() == "explicit_sid"
    SessionManager.reset()
