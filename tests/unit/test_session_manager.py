import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.session.manager import (  # noqa: E402
    MAX_MESSAGE_HISTORY,
    SessionManager,
)


@pytest.fixture(autouse=True)
def reset_session_manager():
    SessionManager.reset()
    yield
    SessionManager.reset()


class _FakeUiSync:
    """UI-sync adapter backed by a plain dict (stands in for StreamlitSessionSync)."""

    def __init__(self, store: dict) -> None:
        self.store = store
        self.writes: list[tuple[str, object]] = []

    def write(self, key: str, val: object) -> None:
        self.store[key] = val
        self.writes.append((key, val))

    def read(self, key: str, default: object = None) -> object:
        return self.store.get(key, default)


def _patch_sync_deps(st_dict):
    """Return combined patchers for sync_to_streamlit testing.

    The streamlit dependency was refactored out of core: UI writes now go
    through a pluggable adapter installed via ``SessionManager.set_ui_sync``
    (see core.session.manager / ui.session_sync). We install a fake adapter
    whose backing store is ``st_dict``, and force ``_is_streamlit_running`` so
    that ``get``/``sync_to_streamlit`` route to the adapter.
    """
    adapter = _FakeUiSync(st_dict)
    return (
        patch.object(SessionManager, "_ui_sync", adapter),
        patch.object(SessionManager, "_is_streamlit_running", return_value=True),
        patch(
            "streamlit.runtime.scriptrunner.get_script_run_ctx",
            return_value=SimpleNamespace(session_id="t"),
        ),
    )


# --- Basic CRUD Tests ---


def test_basic_set_get_delete():
    sid = "test_sid"
    SessionManager.init_session(sid)

    SessionManager.set("key1", "value1", session_id=sid)
    assert SessionManager.get("key1", session_id=sid) == "value1"

    SessionManager.set("key2", 123, session_id=sid)
    assert SessionManager.get("key2", session_id=sid) == 123

    SessionManager.delete("key1", session_id=sid)
    assert SessionManager.get("key1", session_id=sid) is None
    assert SessionManager.get("key2", session_id=sid) == 123


def test_session_isolation():
    sid1 = "sid1"
    sid2 = "sid2"
    SessionManager.init_session(sid1)
    SessionManager.init_session(sid2)

    SessionManager.set("shared_key", "val1", session_id=sid1)
    SessionManager.set("shared_key", "val2", session_id=sid2)

    assert SessionManager.get("shared_key", session_id=sid1) == "val1"
    assert SessionManager.get("shared_key", session_id=sid2) == "val2"


# --- Streamlit Synchronization Tests ---


def test_sync_to_streamlit_copy():
    sid = "sync_sid"
    SessionManager.init_session(sid)

    st_session_state = {}
    p1, p2, p3 = _patch_sync_deps(st_session_state)

    with p1, p2, p3:
        SessionManager.set("messages", [1, 2, 3], session_id=sid)
        SessionManager.sync_to_streamlit(session_id=sid)

    assert "messages" in st_session_state
    assert st_session_state["messages"] == [1, 2, 3]

    test_list = [1, 2, 3]
    test_list.append(4)
    assert st_session_state["messages"] == [1, 2, 3]


def test_dirty_key_tracking():
    sid = "test_session"
    SessionManager.set_session_id(sid)

    st_state = {}
    p1, p2, p3 = _patch_sync_deps(st_state)

    with p1, p2, p3:
        state = SessionManager._get_state(sid)
        assert len(state["_dirty_keys"]) == 0

        SessionManager.set("key1", "value1", session_id=sid)
        assert "key1" in state["_dirty_keys"]

        SessionManager.sync_to_streamlit(session_id=sid)

    assert st_state["key1"] == "value1"
    assert len(state["_dirty_keys"]) == 0


def test_partial_sync():
    sid = "test_session"
    SessionManager.set_session_id(sid)

    st_state = {}
    p1, p2, p3 = _patch_sync_deps(st_state)

    with p1, p2, p3:
        SessionManager.set("key1", "value1", session_id=sid)
        SessionManager.set("key2", "value2", session_id=sid)

        SessionManager.sync_to_streamlit(session_id=sid)

        assert st_state["key1"] == "value1"
        assert st_state["key2"] == "value2"

        SessionManager.set("key1", "new_value1", session_id=sid)
        state = SessionManager._get_state(sid)
        assert "key1" in state["_dirty_keys"]
        assert "key2" not in state["_dirty_keys"]

        SessionManager.sync_to_streamlit(session_id=sid)

        assert st_state.get("key1") == "new_value1"
        assert st_state.get("key2") == "value2"
        assert len(state["_dirty_keys"]) == 0


def test_get_messages_returns_copy_not_shared_reference():
    """Regression: get_messages() must return a copy, not the live stored list.

    Previously get/get_messages returned the actual list reference held by
    session state. A background add_message mutating that list in-place while
    the UI thread iterated the returned reference caused
    RuntimeError('list changed size during iteration'). Returning a copy gives
    the caller an isolated snapshot.
    """
    sid = "copy_sid"
    SessionManager.init_session(sid)
    for i in range(3):
        SessionManager.add_message("user", f"m{i}", session_id=sid)

    snapshot = SessionManager.get_messages(session_id=sid)
    assert isinstance(snapshot, list)
    assert len(snapshot) == 3

    # Mutating the returned snapshot must NOT affect stored state.
    snapshot.append({"role": "user", "content": "injected"})
    stored = SessionManager.get_messages(session_id=sid)
    assert len(stored) == 3, "stored messages leaked via external mutation"

    # The distinct object identity proves isolation.
    assert snapshot is not stored


# --- Lifecycle & Utility Tests ---


def test_cleanup_expired_sessions():
    sid_old = "old_sid"
    sid_new = "new_sid"

    SessionManager.init_session(sid_old)
    SessionManager.init_session(sid_new)

    now = time.time()
    SessionManager._fallback_sessions[sid_old]["last_accessed"] = now - 5000
    SessionManager._fallback_sessions[sid_new]["last_accessed"] = now

    SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)

    assert sid_old not in SessionManager._fallback_sessions
    assert sid_new in SessionManager._fallback_sessions


def test_evict_stuck_generating_session_frees_graph_thread():
    """Evicting a stuck (stale + generating) session must free its graph thread.

    Regression: a broken generation left ``is_generating_answer`` stuck True
    while ``last_accessed`` was not refreshed. Such sessions must be evicted and
    their LangGraph checkpoint thread (keyed by session_id) released, otherwise
    memory leaks for every dead stream.
    """
    stuck = "stuck_sid"
    fresh = "fresh_sid"

    SessionManager.init_session(stuck)
    SessionManager.init_session(fresh)

    now = time.time()
    SessionManager._fallback_sessions[stuck]["is_generating_answer"] = True
    SessionManager._fallback_sessions[stuck]["last_accessed"] = now - (
        SessionManager.MAX_GENERATION_SECONDS + 100
    )
    SessionManager._fallback_sessions[fresh]["is_generating_answer"] = False
    SessionManager._fallback_sessions[fresh]["last_accessed"] = now

    with patch("core.graph_builder.delete_graph_thread") as mock_del_thread:
        SessionManager._evict_oldest_session_locked()

    assert stuck not in SessionManager._fallback_sessions
    assert fresh in SessionManager._fallback_sessions
    mock_del_thread.assert_called_once_with(stuck)


def test_evict_protects_healthy_generating_session():
    """A live (fresh) generation must never be killed; only idle sessions evict.

    ``_evict_oldest_session_locked`` picks the session with the smallest
    ``last_accessed``. The idle session is oldest, so it is evicted while the
    actively streaming session (fresh ``last_accessed``) stays untouched.
    """
    gen = "gen_sid"
    idle = "idle_sid"

    SessionManager.init_session(gen)
    SessionManager.init_session(idle)

    now = time.time()
    SessionManager._fallback_sessions[gen]["is_generating_answer"] = True
    SessionManager._fallback_sessions[gen]["last_accessed"] = now
    SessionManager._fallback_sessions[idle]["is_generating_answer"] = False
    SessionManager._fallback_sessions[idle]["last_accessed"] = now - 5000

    SessionManager._evict_oldest_session_locked()

    assert gen in SessionManager._fallback_sessions
    assert idle not in SessionManager._fallback_sessions


def test_evict_no_deadlock_lock_held():
    """Eviction must complete without deadlock while ``_map_lock`` is held.

    The real caller invokes ``_evict_oldest_session_locked`` already holding
    ``_map_lock``; if it re-acquires the same non-reentrant lock it would
    deadlock. This test exercises that exact call shape.
    """
    SessionManager.init_session("default")
    SessionManager.init_session("stuck")

    now = time.time()
    SessionManager._fallback_sessions["stuck"]["is_generating_answer"] = True
    SessionManager._fallback_sessions["stuck"]["last_accessed"] = now - (
        SessionManager.MAX_GENERATION_SECONDS + 100
    )

    with SessionManager._map_lock:
        SessionManager._evict_oldest_session_locked()

    assert "stuck" not in SessionManager._fallback_sessions
    assert "default" in SessionManager._fallback_sessions


# --- Error Handling & Edge Cases ---


@patch("os.path.exists")
def test_safe_remove_file_retry(mock_exists):
    mock_exists.side_effect = [False, True]

    result = SessionManager.safe_remove_file("/fake/path.pdf")
    assert result is True


def test_add_message_limit():
    SessionManager.init_session()
    sid = SessionManager.get_session_id()

    for i in range(MAX_MESSAGE_HISTORY + 5):
        SessionManager.add_message("user", str(i), session_id=sid)

    msgs = SessionManager.get("messages", session_id=sid)
    assert len(msgs) == MAX_MESSAGE_HISTORY
    assert msgs[0]["content"] == "5"


# --- Multi-Session Tests ---


@pytest.mark.parametrize(
    "sid", ["sid_a", "sid_b", "sid_c"], ids=["session-A", "session-B", "session-C"]
)
def test_multiple_sessions_param(sid):
    SessionManager.init_session(sid)
    SessionManager.set("key", f"val_{sid}", session_id=sid)
    assert SessionManager.get("key", session_id=sid) == f"val_{sid}"


# --- Async Context Propagation Tests ---


def test_session_id_recovery_in_async():
    import asyncio

    sid = "async_test_session"
    SessionManager.set_session_id(sid)

    async def worker():
        return SessionManager.get_session_id()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(worker())
    finally:
        loop.close()

    assert SessionManager.get_session_id() == sid


def test_session_id_context_loss():
    import asyncio

    async def main():
        original_sid = "main_context"
        SessionManager.set_session_id(original_sid)
        assert SessionManager.get_session_id() == original_sid

        await asyncio.sleep(0)
        current_sid = SessionManager.get_session_id()
        assert current_sid == original_sid, (
            f"Expected session ID to persist across await, got {current_sid}"
        )

    asyncio.run(main())


def test_session_isolation_async():
    import asyncio

    async def worker(name, expected_sid):
        for _ in range(5):
            current = SessionManager.get_session_id()
            assert current == expected_sid, (
                f"{name}: expected {expected_sid}, got {current}"
            )
            await asyncio.sleep(0.001)

    async def run_concurrent():
        SessionManager.set_session_id("user_A")
        task1 = asyncio.create_task(worker("user_A", "user_A"))

        SessionManager.set_session_id("user_B")
        task2 = asyncio.create_task(worker("user_B", "user_B"))

        await asyncio.gather(task1, task2)

    asyncio.run(run_concurrent())


# --- Thread-Safety Stress Tests ---


def test_thread_safe_global_state():
    SessionManager.reset()
    SessionManager.set_session_id("thread_test")
    errors = []
    iterations = 500

    def worker(value):
        for i in range(iterations):
            SessionManager.set(f"key{i}", value)
            val = SessionManager.get(f"key{i}")
            if val != value:
                errors.append(f"Expected {value}, got {val}")
                return

    thread1 = threading.Thread(target=worker, args=("value1",))
    thread2 = threading.Thread(target=worker, args=("value2",))

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    assert not errors, f"Thread-safety violation detected: {errors}"
    SessionManager.reset()


# --- Add Message with Various Types ---


def test_add_message_with_types():
    """add_message는 다양한 content 타입을 수용해야 합니다."""
    SessionManager.init_session()
    sid = SessionManager.get_session_id()

    SessionManager.add_message("user", "string_msg")
    SessionManager.add_message("assistant", 123)  # type: ignore[arg-type]
    SessionManager.add_message("user", [1, 2, 3])  # type: ignore[arg-type]
    SessionManager.add_message("assistant", None)  # type: ignore[arg-type]

    messages = SessionManager.get("messages", session_id=sid)
    assert messages[0]["content"] == "string_msg"
    assert messages[1]["content"] == 123
    assert messages[2]["content"] == [1, 2, 3]
    assert messages[3]["content"] is None


# --- Reset State ---


def test_reset_conversation_clears_messages():
    SessionManager.init_session()
    sid = SessionManager.get_session_id()

    SessionManager.add_message("msg1", "user")
    SessionManager.add_message("msg2", "assistant")
    SessionManager.reset_conversation()

    messages = SessionManager.get("messages", session_id=sid)
    assert messages == []
