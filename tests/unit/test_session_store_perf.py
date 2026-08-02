# SessionStore의 set/get 지연 시간이 세션 크기와 무관하게 안정적인지(O(1)) 검증합니다.
import time

from core.session.store import SessionStore


def test_session_store_set_latency():
    store = SessionStore()
    session_id = "perf_test_session"

    # Test sizes: 10, 100, 1000 entries per session
    sizes = [10, 100, 1000]
    results = {}

    for size in sizes:
        store.clear(session_id)
        for i in range(size):
            store.set(f"msg_{i}", "Hello " * 10, session_id)

        # Measure latency of a single 'set' operation
        start_time = time.perf_counter()
        for _ in range(100):
            store.set("global_status", "Updating...", session_id)
        end_time = time.perf_counter()

        avg_latency = (end_time - start_time) / 100
        results[size] = avg_latency

    # Verify that latency remains stable regardless of session size (O(1))
    assert results[1000] < results[10] * 2, (
        f"Latency should be stable. 10: {results[10]}, 1000: {results[1000]}"
    )
