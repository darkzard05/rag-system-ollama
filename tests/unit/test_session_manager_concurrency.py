import concurrent.futures
import threading
import pytest
from core.session.manager import SessionManager


def test_session_manager_concurrency():
    """세션 관리자의 동시 쓰기 작업 시 락 경합과 데이터 일관성을 테스트합니다."""
    sid = "test_concurrency_sid"
    SessionManager.init_session(sid)

    num_threads = 10
    iterations = 100

    def writer(thread_id):
        for i in range(iterations):
            SessionManager.set(f"key_{thread_id}_{i}", i, session_id=sid)
            SessionManager.add_message("user", f"msg_{thread_id}_{i}", session_id=sid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(writer, t) for t in range(num_threads)]
        concurrent.futures.wait(futures)

    # 일관성 검증
    messages = SessionManager.get_messages(session_id=sid)
    assert len(messages) <= 1000  # MAX_MESSAGE_HISTORY check

    # 세션 상태에 데이터가 제대로 저장되었는지 확인
    for t in range(num_threads):
        assert (
            SessionManager.get(f"key_{t}_{iterations - 1}", session_id=sid)
            == iterations - 1
        )


if __name__ == "__main__":
    pytest.main([__file__])
