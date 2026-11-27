"""Retrying analysis must drop the error bubble from the failed attempt."""

from __future__ import annotations

import pytest

from core.session import SessionManager
from ui.components.doc_analysis.actions import on_retry


@pytest.fixture
def sid() -> str:
    session_id = "test-retry-clears-build-error"
    SessionManager.reset_all_state(session_id)
    yield session_id
    SessionManager.reset_all_state(session_id)


def test_retry_removes_stale_build_error_but_keeps_chat(sid: str) -> None:
    SessionManager.add_message("user", "What is in this PDF?", session_id=sid)
    SessionManager.add_message(
        "assistant", "Text extraction failed", msg_type="build_error", session_id=sid
    )
    SessionManager.add_message(
        "system", "Preparing analysis...", msg_type="build_progress", session_id=sid
    )

    on_retry(sid)

    messages = SessionManager.get_messages(session_id=sid)
    assert [m["content"] for m in messages] == [
        "What is in this PDF?",
        "Preparing analysis...",
    ]
    assert SessionManager.get("needs_rag_rebuild", False, session_id=sid) is True
