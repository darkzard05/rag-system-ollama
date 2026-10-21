from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

import ui.components.viewer as viewer_module
from core.session import SessionManager

SID = "test_first_page_resilience_sid"


@pytest.fixture(autouse=True)
def setup_session(monkeypatch: pytest.MonkeyPatch):
    SessionManager.reset()
    SessionManager.init_session(SID)
    SessionManager.set_session_id(SID)
    SessionManager.set("pdf_file_path", "dummy_test.pdf", SID)
    SessionManager.set("file_hash", "hash_resilient_123", SID)
    fake_session_state: dict[str, object] = {}
    monkeypatch.setattr(viewer_module.st, "session_state", fake_session_state)
    yield
    SessionManager.reset()


def test_resolve_pdf_state_falls_back_to_preview_when_total_pages_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionManager.delete("total_pages", SID)
    monkeypatch.setattr(viewer_module, "_get_pdf_total_pages", lambda path: None)

    fake_preview = {
        "file_hash": "hash_resilient_123",
        "bytes": b"%PDF-1.4 FAST PREVIEW PAGE 1",
    }
    SessionManager.set("first_page_preview", fake_preview, SID)

    state = viewer_module._resolve_pdf_state()

    assert state is not None, "first_page_preview가 있음에도 state가 None을 반환함"
    assert state["current_page"] == 1
    assert state["total_pages"] == 1
    assert state["file_hash"] == "hash_resilient_123"


def test_resolve_pdf_state_returns_none_when_both_total_pages_and_preview_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionManager.delete("total_pages", SID)
    SessionManager.delete("first_page_preview", SID)
    monkeypatch.setattr(viewer_module, "_get_pdf_total_pages", lambda path: None)

    state = viewer_module._resolve_pdf_state()

    assert state is None
