"""
delete_session 락 범위 검증 (문제 2D).

물리적 파일 삭제(safe_remove_file)는 _map_lock 밖에서 수행되어야
전역 세션 접근이 블로킹되지 않습니다.
"""

from core.session import SessionManager


def test_file_removal_happens_outside_map_lock(monkeypatch):
    SessionManager.reset()
    sid = "lock_scope_test"
    SessionManager.init_session(sid)
    SessionManager.set("pdf_file_path", "/fake/path.pdf", session_id=sid)

    captured = {}

    def fake_safe_remove_file(path):
        captured["lock_held"] = SessionManager._map_lock.locked()
        captured["path"] = path

    monkeypatch.setattr(SessionManager, "safe_remove_file", fake_safe_remove_file)

    result = SessionManager.delete_session(sid)

    assert result is True
    assert captured["path"] == "/fake/path.pdf"
    assert captured["lock_held"] is False
    assert SessionManager.get("pdf_file_path", session_id=sid) is None


def test_delete_nonexistent_session_returns_false():
    SessionManager.reset()
    assert SessionManager.delete_session("ghost_session_zzz") is False
