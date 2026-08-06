from src.core.rag_core import RAGSystem
from src.core.session import SessionManager


def test_rag_system_di():
    """Verify that RAGSystem correctly interacts with the SessionManager."""
    sid = "test_rag_session"
    rag_system = RAGSystem(session_id=sid)

    # Verify that clear_session uses the session_id
    rag_system.clear_session()

    # Verify the store was cleared for that session (using SessionManager directly)
    assert SessionManager.get("rag_engine", session_id=sid) is None


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
