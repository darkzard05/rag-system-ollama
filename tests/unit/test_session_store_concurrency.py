# SessionStore의 스레드 안전성을 검증하는 단위 테스트.
import threading

from core.session.store import SessionStore


def test_concurrent_set_get_no_lost_writes():
    """여러 스레드가 서로 다른 키를 동시에 쓸 때 손실 없이 모두 저장되는지 검증합니다."""
    store = SessionStore()
    session_id = "concurrency_test_session"

    num_threads = 20
    keys_per_thread = 5

    def worker(tid: int):
        for i in range(keys_per_thread):
            store.set(f"key_{tid}_{i}", i, session_id)

    threads = [
        threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 스레드 간 손실(lost update) 없이 전부 저장되어야 함
    for tid in range(num_threads):
        for i in range(keys_per_thread):
            assert store.get(f"key_{tid}_{i}", session_id) == i


def test_concurrent_set_get_same_key():
    """여러 스레드가 같은 키를 동시에 갱신해도 예외 없이 최종값이 남는지 검증합니다."""
    store = SessionStore()
    session_id = "concurrency_test_session"
    num_threads = 20

    def worker():
        for _ in range(100):
            store.set("counter", 1, session_id)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.get("counter", session_id) == 1


def test_session_store_isolation():
    """clear/delete가 다른 세션의 데이터에 영향을 주지 않는지 검증합니다."""
    store = SessionStore()
    store.set("key", "v1", "s1")
    store.set("key", "v2", "s2")

    store.delete("key", "s1")
    assert store.get("key", "s1") is None
    assert store.get("key", "s2") == "v2"

    store.clear("s2")
    assert store.get("key", "s2") is None
