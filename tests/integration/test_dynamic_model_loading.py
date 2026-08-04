import threading
import time

from core.session import SessionManager


def mock_load_llm(name, duration=0.2):
    time.sleep(duration)
    return f"Object_LLM_{name}"


def mock_load_embedder(name, duration=0.2):
    time.sleep(duration)
    return f"Object_Embedder_{name}"


def test_parallel_loading_is_faster_than_serial():
    """Two models loading in parallel should finish faster than sum of individual sleeps."""
    sid = "test_dynamic"
    SessionManager.init_session(sid)

    selected_llm = "test-llm"
    selected_emb = "test-embedder"

    start_time = time.time()

    t1 = threading.Thread(
        target=lambda: SessionManager.set(
            "llm", mock_load_llm(selected_llm), session_id=sid
        )
    )
    t2 = threading.Thread(
        target=lambda: SessionManager.set(
            "embedder", mock_load_embedder(selected_emb), session_id=sid
        )
    )

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    total_time = time.time() - start_time

    assert SessionManager.get("llm", session_id=sid) == f"Object_LLM_{selected_llm}"
    assert (
        SessionManager.get("embedder", session_id=sid)
        == f"Object_Embedder_{selected_emb}"
    )
    # Serial would be 2 * 0.2s = 0.4s; parallel should be roughly 0.2s
    assert total_time < 0.4, f"Parallel loading took too long: {total_time:.2f}s"
