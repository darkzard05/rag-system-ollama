"""A deleted key must not resurrect from the UI mirror before sync consumes it."""

import streamlit.runtime.scriptrunner as scriptrunner

from core.session import SessionManager


class _FakeUI:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def read(self, key: str, default: object = None) -> object:
        return self.store.get(key, default)

    def write(self, key: str, value: object) -> None:
        self.store[key] = value

    def delete(self, key: str) -> None:
        self.store.pop(key, None)


def test_deleted_key_does_not_resurrect_from_ui_mirror(monkeypatch) -> None:
    monkeypatch.setattr(
        SessionManager, "_is_streamlit_running", classmethod(lambda cls: True)
    )
    monkeypatch.setattr(scriptrunner, "get_script_run_ctx", lambda: object())

    fake = _FakeUI()
    SessionManager.reset()
    SessionManager.set_ui_sync(fake)
    sid = "tombstone-ui-mirror"
    try:
        SessionManager.set("k", "v", session_id=sid)
        fake.store["k"] = "v"
        SessionManager.delete("k", session_id=sid)

        assert SessionManager.get("k", session_id=sid) is None
    finally:
        SessionManager.set_ui_sync(None)
        SessionManager.reset()
