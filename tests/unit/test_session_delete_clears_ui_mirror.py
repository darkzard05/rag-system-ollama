"""Deleting a session must clear values mirrored into the UI store."""

import pytest

from core.session import SessionManager


class _FakeAdapter:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def write(self, key: str, value: object) -> None:
        self.store[key] = value

    def read(self, key: str, default: object = None) -> object:
        return self.store.get(key, default)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)


@pytest.fixture(autouse=True)
def _reset_sessions():
    SessionManager.reset()
    yield
    SessionManager.reset()


def test_reset_all_state_does_not_resurrect_old_mirror_values(monkeypatch) -> None:
    """A recreated session must not read the deleted session's UI values."""
    monkeypatch.setattr(
        SessionManager,
        "_is_streamlit_running",
        classmethod(lambda cls: True),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "streamlit.runtime.scriptrunner.get_script_run_ctx",
        lambda: object(),
        raising=False,
    )
    adapter = _FakeAdapter()
    SessionManager.set_ui_sync(adapter)
    sid = "reset_mirror_session"

    SessionManager.set("chat_history", ["old message"], session_id=sid)
    SessionManager.set("doc_stats", {"pages": 9}, session_id=sid)
    SessionManager.sync_to_streamlit(sid)
    assert adapter.store.get("chat_history") == ["old message"]
    assert adapter.store.get("doc_stats") == {"pages": 9}

    SessionManager.reset_all_state(sid)

    assert SessionManager.get("chat_history", default=[], session_id=sid) == []
    assert SessionManager.get("doc_stats", default=None, session_id=sid) is None
