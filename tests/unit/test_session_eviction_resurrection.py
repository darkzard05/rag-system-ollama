# tests/unit/test_session_eviction_resurrection.py
"""
Unit test for session deletion resurrection and eviction thrashing prevention (Task 4).
"""

import sys
import os
import pytest

sys.path.append(os.path.abspath("src"))

from core.session import SessionManager


@pytest.fixture(autouse=True)
def reset_sessions():
    SessionManager.reset()
    yield
    SessionManager.reset()


def test_session_deletion_and_get_resurrection_prevention():
    """
    1. Initialize a session and record data.
    2. Delete the session via SessionManager.delete_session().
    3. Verify session is no longer in _fallback_sessions.
    4. Simulate background residual task calling SessionManager.get("is_generating_answer", session_id=sid).
    5. Assert that the session does NOT resurrect in _fallback_sessions and get() returns default.
    """
    sid = "sid_to_delete_123"
    SessionManager.init_session(sid)
    SessionManager.set("is_generating_answer", True, session_id=sid)
    SessionManager.set("test_data", "active", session_id=sid)

    assert sid in SessionManager._fallback_sessions
    assert SessionManager.get("test_data", session_id=sid) == "active"

    # Delete session
    success = SessionManager.delete_session(sid)
    assert success is True
    assert sid not in SessionManager._fallback_sessions

    # Simulate background get call (e.g. residual task or logger checking status)
    val = SessionManager.get("is_generating_answer", default=False, session_id=sid)
    assert val is False

    # Verify session was NOT resurrected in _fallback_sessions
    assert sid not in SessionManager._fallback_sessions
