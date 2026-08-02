import pytest
from unittest.mock import MagicMock, patch
from core.session.store import SessionStore
from src.core.rag_core import RAGSystem
from src.core.session import SessionManager
from src.core.session.store import SessionStore


def test_session_isolation():
    """Verify that SessionStore provides complete data isolation between different session_ids."""
    store = SessionStore()
    sid1 = "session_1"
    sid2 = "session_2"

    # Set data for session 1
    store.set("user_name", "Alice", session_id=sid1)
    store.set("preference", "dark_mode", session_id=sid1)

    # Set data for session 2
    store.set("user_name", "Bob", session_id=sid2)

    # Verify isolation
    assert store.get("user_name", session_id=sid1) == "Alice"
    assert store.get("preference", session_id=sid1) == "dark_mode"
    assert store.get("user_name", session_id=sid2) == "Bob"
    assert store.get("preference", session_id=sid2) is None


def test_rag_system_di():
    """Verify that RAGSystem correctly interacts with the SessionManager."""
    sid = "test_rag_session"
    rag_system = RAGSystem(session_id=sid)

    # Verify that clear_session uses the session_id
    rag_system.clear_session()

    # Verify the store was cleared for that session (using SessionManager directly)
    assert SessionManager.get("rag_engine", session_id=sid) is None


def test_session_store_persistence():
    """Verify that data persists across different calls to the same session_id."""
    store = SessionStore()
    sid = "persistent_session"

    # First call: set value
    store.set("counter", 1, session_id=sid)

    # Second call: get value
    assert store.get("counter", session_id=sid) == 1

    # Third call: update value
    store.set("counter", 2, session_id=sid)
    assert store.get("counter", session_id=sid) == 2


def test_rag_system_session_id_consistency():
    """Verify RAGSystem uses its initialized session_id for operations."""
    sid = "consistent_session"
    rag_system = RAGSystem(session_id=sid)

    # Set data in a DIFFERENT session
    other_sid = "other_session"
    SessionManager.set("key", "value", session_id=other_sid)

    # Clear the RAGSystem's session
    rag_system.clear_session()

    # Verify other session is untouched
    assert SessionManager.get("key", session_id=other_sid) == "value"
    # Verify RAGSystem's session is cleared
    assert SessionManager.get("key", session_id=sid) is None
