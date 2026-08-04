import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import src.core.session.manager as _mgr
from src.core.session.manager import (  # noqa: E402
    MAX_MESSAGE_HISTORY,
    SessionManager,
)


@pytest.fixture(autouse=True)
def reset_session_manager():
    SessionManager.reset()
    yield
    SessionManager.reset()


def _make_streamlit_mock(session_state_dict):
    """Create a minimal streamlit mock with a real dict for session_state."""
    return SimpleNamespace(
        session_state=session_state_dict,
        runtime=SimpleNamespace(
            exists=lambda: True,
            scriptrunner=SimpleNamespace(
                get_script_run_ctx=lambda: SimpleNamespace(session_id="t")
            ),
        ),
    )


def _patch_sync_deps(st_dict):
    """Return combined patchers for sync_to_streamlit testing.

    We need to patch:
    1. `_mgr.st` — replace module-level st with mock whose session_state is a real dict.
    2. `_is_streamlit_running` — force True (real streamlit has no runtime here).
    3. `streamlit.runtime.scriptrunner.get_script_run_ctx` — return truthy mock context,
       because `sync_to_streamlit` does a fresh `from streamlit.runtime...` import at runtime.
    """
    mock_st = _make_streamlit_mock(st_dict)
    return (
        patch.object(_mgr, "st", mock_st),
        patch.object(SessionManager, "_is_streamlit_running", return_value=True),
        patch("streamlit.runtime.scriptrunner.get_script_run_ctx"),
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
