import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# Mock streamlit before importing SessionManager
mock_st = MagicMock()
with patch.dict("sys.modules", {"streamlit": mock_st}):
    from src.core.session.manager import (  # noqa: I001
        MAX_MESSAGE_HISTORY,
        SessionManager,
    )


@pytest.fixture(autouse=True)
def reset_session_manager():
    """Resets the SessionManager state before and after each test."""
    SessionManager.reset()
    yield
    SessionManager.reset()


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


@patch("streamlit.runtime.exists", return_value=True)
@patch("streamlit.runtime.scriptrunner.get_script_run_ctx")
def test_sync_to_streamlit_copy(mock_get_ctx, mock_exists):
    sid = "sync_sid"
    SessionManager.init_session(sid)

    mock_ctx = MagicMock()
    mock_ctx.session_id = sid
    mock_get_ctx.return_value = mock_ctx

    st_session_state = {}

    test_list = [1, 2, 3]
    sync_key = "messages"
    SessionManager.set(sync_key, test_list, session_id=sid)

    # import 순서와 무관하게 동작하도록 st.session_state를 직접 patch
    # (manager 모듈의 st가 실제 streamlit이든 mock이든 동일하게 동작)
    with patch("src.core.session.manager.st.session_state", st_session_state):
        SessionManager.sync_to_streamlit(session_id=sid)

    assert sync_key in st_session_state
    assert st_session_state[sync_key] == test_list
    assert st_session_state[sync_key] is not test_list

    test_list.append(4)
    assert st_session_state[sync_key] == [1, 2, 3]

    @patch("streamlit.runtime.scriptrunner.get_script_run_ctx")
    def test_dirty_key_tracking(mock_get_ctx):
        mock_get_ctx.return_value = MagicMock(session_id="test_session")
        sid = "test_session"
        SessionManager.set_session_id(sid)

        with patch.object(SessionManager, "_is_streamlit_running", return_value=True):
            state = SessionManager._get_state(sid)
            assert len(state["_dirty_keys"]) == 0

            SessionManager.set("key1", "value1", session_id=sid)
            assert "key1" in state["_dirty_keys"]

            mock_st_state = {}
            with patch("src.core.session.manager.st.session_state", mock_st_state):
                SessionManager.sync_to_streamlit(session_id=sid)

            assert mock_st_state["key1"] == "value1"
            assert len(state["_dirty_keys"]) == 0

    @patch("streamlit.runtime.scriptrunner.get_script_run_ctx")
    def test_partial_sync(mock_get_ctx):
        mock_get_ctx.return_value = MagicMock(session_id="test_session")
        sid = "test_session"
        SessionManager.set_session_id(sid)

        with patch.object(SessionManager, "_is_streamlit_running", return_value=True):
            SessionManager.set("key1", "value1", session_id=sid)
            SessionManager.set("key2", "value2", session_id=sid)

            mock_st_state = {}
            with patch("src.core.session.manager.st.session_state", mock_st_state):
                SessionManager.sync_to_streamlit(session_id=sid)

            assert mock_st_state["key1"] == "value1"
            assert mock_st_state["key2"] == "value2"

            SessionManager.set("key1", "new_value1", session_id=sid)
            state = SessionManager._get_state(sid)
            assert "key1" in state["_dirty_keys"]
            assert "key2" not in state["_dirty_keys"]

            with patch("src.core.session.manager.st.session_state", mock_st_state):
                SessionManager.sync_to_streamlit(session_id=sid)

            assert mock_st_state.get("key1") == "new_value1"
            assert mock_st_state.get("key2") == "value2"
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


@patch("os.path.exists")
@patch("os.remove")
def test_safe_remove_file_retry(mock_remove, mock_exists):
    mock_exists.return_value = True
    path = "dummy_path.txt"
    mock_remove.side_effect = [PermissionError, None]

    with patch("time.sleep"):
        result = SessionManager.safe_remove_file(path, max_retries=3)

    assert result is True
    assert mock_remove.call_count == 2


def test_add_message_limit():
    sid = "msg_sid"
    SessionManager.init_session(sid)

    for i in range(MAX_MESSAGE_HISTORY + 10):
        SessionManager.add_message("user", f"message {i}", session_id=sid)

    messages = SessionManager.get_messages(session_id=sid)
    assert len(messages) == MAX_MESSAGE_HISTORY
    assert messages[-1]["content"] == f"message {MAX_MESSAGE_HISTORY + 9}"


@pytest.mark.parametrize("sid", ["sid_a", "sid_b", "sid_c"])
def test_multiple_sessions_param(sid):
    SessionManager.init_session(sid)
    SessionManager.set("param_key", f"val_{sid}", session_id=sid)
    assert SessionManager.get("param_key", session_id=sid) == f"val_{sid}"


# --- Async & Threading Tests ---


@pytest.mark.asyncio
async def test_session_id_recovery_in_async(session_context):
    SessionManager.set("key", "value", session_id=session_context)

    async def background_task():
        return SessionManager.get("key", session_id=session_context)

    result = await background_task()
    assert result == "value"


@pytest.mark.asyncio
async def test_session_id_context_loss(session_context):
    SessionManager.set("file_hash", "hash123", session_id=session_context)

    from src.core.session.manager import _session_id_var

    token = _session_id_var.set("default")
    try:
        val = SessionManager.get("file_hash")
        assert val != "hash123"
        val_explicit = SessionManager.get("file_hash", session_id=session_context)
        assert val_explicit == "hash123"
    finally:
        _session_id_var.reset(token)


@pytest.mark.asyncio
async def test_session_isolation_async(session_context):
    other_session = f"other_{session_context}"
    SessionManager.init_session(other_session)

    SessionManager.set("shared_key", "value1", session_id=session_context)
    SessionManager.set("shared_key", "value2", session_id=other_session)

    assert SessionManager.get("shared_key", session_id=session_context) == "value1"
    assert SessionManager.get("shared_key", session_id=other_session) == "value2"


def test_thread_safe_global_state():
    sid = "test_session"
    SessionManager.init_session(sid)

    def worker():
        SessionManager.set("bg_key", "bg_value", session_id=sid)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert SessionManager.get("bg_key", session_id=sid) == "bg_value"


def test_add_message_with_types():
    sid = "type_test"
    SessionManager.init_session(sid)
    # 상태 메시지 테스트
    SessionManager.add_message(
        "system", "분석 시작", msg_type="build_progress", progress=10
    )
    msgs = SessionManager.get_messages(sid)
    assert len(msgs) == 1
    assert msgs[0]["msg_type"] == "build_progress"
    assert msgs[0]["progress"] == 10


def test_update_streaming_message():
    sid = "stream_test"
    SessionManager.init_session(sid)

    # 1. 초기 메시지 추가
    msg_id = "stream_1"
    SessionManager.add_message("assistant", "Hi", msg_type="streaming", msg_id=msg_id)

    # 2. 업데이트
    SessionManager.add_message(
        "assistant", "Hi there", msg_type="streaming", msg_id=msg_id
    )

    msgs = SessionManager.get_messages(sid)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Hi there"
    assert msgs[0]["msg_type"] == "streaming"
