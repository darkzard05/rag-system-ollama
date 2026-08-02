import pytest
from unittest.mock import patch
import streamlit as st

# Patch streamlit.fragment BEFORE importing UIBridge to ensure the decorator is a no-op
with patch("streamlit.fragment", lambda **kwargs: lambda func: func):
    from ui.bridge import UIBridge

from core.session.store import SessionStore


@pytest.fixture
def mock_streamlit():
    """Mocks streamlit session state and fragment decorator."""
    with (
        patch("streamlit.session_state", {}),
        patch("streamlit.fragment", lambda **kwargs: lambda func: func),
    ):
        yield st


@pytest.fixture
def mock_session_id():
    """Mocks get_session_id to return a consistent test ID."""
    with patch("ui.bridge.get_session_id", return_value="test_session_123"):
        yield "test_session_123"


@pytest.fixture
def fresh_store():
    """Provides a fresh SessionStore instance for isolation."""
    return SessionStore()


def test_sync_session_updates_state(mock_streamlit, mock_session_id, fresh_store):
    """
    Verify that a background update in SessionStore is reflected in st.session_state
    via UIBridge.sync_session().
    """
    # Setup: Patch the global session_store used by UIBridge
    with patch("ui.bridge.session_store", fresh_store):
        # 1. Initial state in store
        initial_status = "✅ 시스템 준비 완료"
        fresh_store.set("global_status", initial_status, mock_session_id)

        # Ensure st.session_state is empty or different
        st.session_state["global_status"] = "Old Status"

        # 2. First sync
        UIBridge.sync_session()

        # Verify initial sync
        assert st.session_state["global_status"] == initial_status

        # 3. Simulate background update in SessionStore
        updated_status = "🚀 처리 중..."
        fresh_store.set("global_status", updated_status, mock_session_id)

        # Verify st.session_state is NOT yet updated (simulating async nature)
        assert st.session_state["global_status"] == initial_status

        # 4. Second sync (the heartbeat)
        UIBridge.sync_session()

        # Verify the update is now reflected
        assert st.session_state["global_status"] == updated_status


def test_sync_session_preserves_user_input(
    mock_streamlit, mock_session_id, fresh_store
):
    """
    Verify that keys in USER_INPUT_KEYS are not overwritten by the sync process,
    preserving user input.
    """
    with patch("ui.bridge.session_store", fresh_store):
        # 1. Setup a user input key in st.session_state
        input_key = "main_chat_input"
        user_value = "Hello, this is my query"
        st.session_state[input_key] = user_value

        # 2. Setup a different value for the same key in SessionStore
        # (This simulates a scenario where the store might have a default or old value)
        store_value = "Default or Old Query"
        fresh_store.set(input_key, store_value, mock_session_id)

        # 3. Perform sync
        UIBridge.sync_session()

        # 4. Verify that the user's input was NOT overwritten
        assert st.session_state[input_key] == user_value
        assert st.session_state[input_key] != store_value


def test_sync_session_handles_complex_types(
    mock_streamlit, mock_session_id, fresh_store
):
    """
    Verify that lists and dicts are shallow copied to prevent reference issues.
    """
    with patch("ui.bridge.session_store", fresh_store):
        # Setup a list in store
        logs = ["Log 1", "Log 2"]
        fresh_store.set("status_logs", logs, mock_session_id)

        UIBridge.sync_session()

        # Verify value is correct
        assert st.session_state["status_logs"] == logs
        # Verify it's a copy, not the same reference
        assert st.session_state["status_logs"] is not logs
