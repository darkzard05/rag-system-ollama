"""A user's cancel request must survive build progress events."""

from __future__ import annotations

import pytest

from core.build_state import BuildEvent, BuildPhase
from core.session import SessionManager


@pytest.fixture
def sid() -> str:
    session_id = "test-cancel-request-survives-progress"
    SessionManager.reset_all_state(session_id)
    yield session_id
    SessionManager.reset_all_state(session_id)


def test_progress_event_keeps_pending_cancel(sid: str) -> None:
    SessionManager.init_build_state("doc.pdf", "hash1", session_id=sid)
    SessionManager.set("rebuild_cancelled", True, session_id=sid)

    SessionManager.update_build_state(
        BuildEvent(BuildPhase.EXTRACTING, 40, "build_phase_extracting"),
        session_id=sid,
    )

    assert SessionManager.get("rebuild_cancelled", False, session_id=sid) is True
    assert SessionManager.get("is_building_rag", False, session_id=sid) is True


def test_cancel_phase_still_sets_the_flag(sid: str) -> None:
    SessionManager.init_build_state("doc.pdf", "hash1", session_id=sid)

    SessionManager.update_build_state(
        BuildEvent(phase=BuildPhase.CANCELLED, pct=40),
        session_id=sid,
    )

    assert SessionManager.get("rebuild_cancelled", False, session_id=sid) is True
