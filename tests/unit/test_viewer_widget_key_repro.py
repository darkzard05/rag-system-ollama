"""Repro: viewer token path must not crash on widget-key write (TDD Red->Green).

Official basis: Streamlit docs for st.session_state state that modifying a
widget value via the Session State API after instantiation raises
StreamlitAPIException. The token-consume path in _resolve_pdf_state() runs in
the render body (after number_input exists on reruns), so the nav-input sync
write must swallow StreamlitAPIException like common.navigate_to_page does.
"""

import time

from streamlit.errors import StreamlitAPIException

import ui.components.viewer as viewer_module
from core.session import SessionManager
from ui.widget_keys import PDF_NAV_INPUT_KEY

SID = "test_viewer_widget_key_repro"


class _WidgetInstantiatedState(dict):
    """Fake session_state that mimics post-instantiation widget protection."""

    def __setitem__(self, key: object, value: object) -> None:
        if key == PDF_NAV_INPUT_KEY:
            raise StreamlitAPIException(
                f"st.session_state.{key} cannot be modified after the widget "
                f"with key {key} is instantiated."
            )
        super().__setitem__(key, value)


def _setup(monkeypatch) -> _WidgetInstantiatedState:
    SessionManager.reset_all_state(SID)
    SessionManager.set_session_id(SID)
    SessionManager.set("pdf_file_path", "dummy.pdf", SID)
    SessionManager.set("file_hash", "testhash", SID)
    monkeypatch.setattr(viewer_module, "_get_pdf_total_pages", lambda path: 10)
    fake = _WidgetInstantiatedState()
    monkeypatch.setattr(viewer_module.st, "session_state", fake)
    return fake


def test_token_consume_survives_widget_key_protection(monkeypatch) -> None:
    """Green expectation: token jump resolves even when nav-input write is barred."""
    fake = _setup(monkeypatch)
    SessionManager.set(
        "pdf_target_page", {"page": 5, "source": "manual", "ts": time.time()}, SID
    )

    # Must not raise StreamlitAPIException; the write is best-effort sync.
    state = viewer_module._resolve_pdf_state()

    assert state is not None
    assert state["current_page"] == 5
    assert SessionManager.get("pdf_target_page", session_id=SID) is None
    # Widget write was attempted but safely swallowed; no crash, no leftover.
    assert PDF_NAV_INPUT_KEY not in fake
