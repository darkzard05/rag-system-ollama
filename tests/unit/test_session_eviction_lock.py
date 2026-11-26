"""Eviction must not delete a session while a writer holds its lock."""

from core.session import SessionManager
from core.session.manager import MAX_ACTIVE_SESSIONS


def test_eviction_skips_session_with_active_writer() -> None:
    SessionManager.reset()
    try:
        for i in range(MAX_ACTIVE_SESSIONS):
            sid = f"evict-{i:02d}"
            SessionManager.set("last_accessed", 1000.0 + i, session_id=sid)

        victim = "evict-00"
        lock = SessionManager._acquire_lock(victim)
        lock.acquire()
        try:
            SessionManager.init_session("evict-new")

            assert victim in SessionManager._fallback_sessions
            assert SessionManager.get("last_accessed", session_id=victim) == 1000.0
        finally:
            lock.release()
    finally:
        SessionManager.reset()
