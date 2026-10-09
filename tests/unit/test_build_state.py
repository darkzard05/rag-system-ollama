"""P0: BuildState 단일 진실원천 계약 테스트 (순수 + 세션 이중기록)."""

from __future__ import annotations

import time

from core.build_state import (
    BuildEvent,
    BuildPhase,
    apply_event,
    cancel_build,
    complete_build,
    elapsed_sec,
    fail_build,
    initial_build_state,
    is_terminal,
)
from core.session import SessionManager


def test_initial_state_is_preparing() -> None:
    build = initial_build_state("doc.pdf", "hash1")
    assert build.phase == BuildPhase.PREPARING
    assert build.pct == 0
    assert build.file_name == "doc.pdf"
    assert build.started_at is not None
    assert not is_terminal(build.phase)


def test_apply_event_monotonic_guards_regression() -> None:
    build = initial_build_state("doc.pdf")
    apply_event(build, BuildEvent(BuildPhase.EXTRACTING, 25, "k", time.time()))
    assert (build.phase, build.pct) == (BuildPhase.EXTRACTING, 25)
    # 역행 pct는 무시된다
    apply_event(build, BuildEvent(BuildPhase.EXTRACTING, 10, "k", time.time()))
    assert build.pct == 25


def test_terminal_transitions_set_finished() -> None:
    build = initial_build_state("doc.pdf")
    complete_build(build, cache_used=True)
    assert build.phase == BuildPhase.DONE
    assert build.pct == 100
    assert build.cache_used is True
    assert build.finished_at is not None

    build2 = initial_build_state("doc.pdf")
    fail_build(build2, "boom")
    assert build2.phase == BuildPhase.ERROR
    assert build2.error == "boom"

    build3 = initial_build_state("doc.pdf")
    cancel_build(build3)
    assert build3.phase == BuildPhase.CANCELLED
    # 터미널 이후 이벤트는 무시
    apply_event(build3, BuildEvent(BuildPhase.INDEXING, 75, "k", time.time()))
    assert build3.phase == BuildPhase.CANCELLED


def test_elapsed_sec_without_start_is_zero() -> None:
    build = initial_build_state("doc.pdf")
    build.started_at = None
    assert elapsed_sec(build) == 0


def test_session_dual_write_legacy_keys() -> None:
    sid = "test-build-state-dual"
    SessionManager.reset_all_state(sid)
    build = SessionManager.init_build_state("doc.pdf", "hash1", session_id=sid)
    assert SessionManager.get("rebuild_progress", 0, session_id=sid) == 0
    assert SessionManager.get("is_building_rag", False, session_id=sid) is True
    assert SessionManager.get("rebuild_done", True, session_id=sid) is False

    SessionManager.update_build_state(
        BuildEvent(BuildPhase.INDEXING, 75, "build_phase_indexing", time.time()),
        session_id=sid,
        status_text="Building vector index...",
    )
    assert SessionManager.get("rebuild_progress", 0, session_id=sid) == 75
    assert (
        SessionManager.get("rebuild_status", "", session_id=sid)
        == "Building vector index..."
    )
    stored = SessionManager.get_build_state(session_id=sid)
    assert stored is not None
    assert stored.phase == BuildPhase.INDEXING
    assert stored.pct == 75

    complete_build(build, cache_used=False)
    SessionManager.finalize_build_state(build, session_id=sid)
    assert SessionManager.get("rebuild_done", False, session_id=sid) is True
    assert SessionManager.get("is_building_rag", True, session_id=sid) is False
    SessionManager.reset_all_state(sid)
