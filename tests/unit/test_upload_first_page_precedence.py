"""신규 파일 업로드 시 첫 페이지 페인트 보호를 위한 rerun 미발화 검증."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

import src.main as main_module  # noqa: E402

from core.session import SessionManager  # noqa: E402

SID = "test_first_page_precedence_sid"


@pytest.fixture(autouse=True)
def setup_session():
    SessionManager.reset()
    SessionManager.reset_all_state(SID)
    SessionManager.set_session_id(SID)
    yield
    SessionManager.reset()


def test_new_file_uploaded_dispatches_without_rerun():
    """new_file_uploaded 처리 시 st.rerun이 호출되지 않는다.

    업로드 콜백이 만든 첫 렌더링 프레임이 중단되면 브라우저가 좌측 PDF
    1페이지를 페인트할 틈이 없다. 펜딩 핸들러는 백그라운드 디스패치만
    수행하고 프레임을 커밋해야 한다.
    """
    SessionManager.set("new_file_uploaded", True, session_id=SID)
    SessionManager.set("last_uploaded_file_name", "preview_doc.pdf", session_id=SID)
    SessionManager.set("file_hash", "hash_preview_123", session_id=SID)
    SessionManager.set("pdf_file_path", "/tmp/preview_doc.pdf", session_id=SID)

    with (
        patch("src.main._bg_rebuild_task", MagicMock()) as mock_task,
        patch("common.utils.run_in_background_worker") as mock_worker,
        patch("src.main.st.rerun") as mock_rerun,
    ):
        main_module._handle_pending_tasks()

    mock_task.assert_called_once()
    mock_worker.assert_called_once()
    mock_rerun.assert_not_called()

    assert SessionManager.get("new_file_uploaded", False, session_id=SID) is False
    assert SessionManager.get("is_building_rag", False, session_id=SID) is True
    build_state = SessionManager.get_build_state(session_id=SID)
    assert build_state is not None
    assert build_state.file_name == "preview_doc.pdf"
    assert build_state.file_hash == "hash_preview_123"


def test_embedding_rebuild_still_requests_rerun():
    """needs_rag_rebuild 분기는 기존대로 rerun을 유지한다."""
    SessionManager.set("needs_rag_rebuild", True, session_id=SID)
    SessionManager.set("last_uploaded_file_name", "preview_doc.pdf", session_id=SID)
    SessionManager.set("pdf_file_path", "/tmp/preview_doc.pdf", session_id=SID)

    with (
        patch("src.main._bg_rebuild_task", MagicMock()),
        patch("common.utils.run_in_background_worker"),
        patch("src.main.st.rerun") as mock_rerun,
    ):
        main_module._handle_pending_tasks()

    mock_rerun.assert_called_once()


def test_extracting_rerun_blocked_during_initial_paint_window():
    """업로드 직후 initial_paint_window(0.8s) 내의 조기 EXTRACTING 리런은 억제되어야 함."""
    from core.build_state import BuildPhase, initial_build_state

    SessionManager.set("is_building_rag", True, session_id=SID)
    build = initial_build_state("doc.pdf", "hash123")
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, session_id=SID)

    now = 1000.0
    main_module._BUILD_RERUN_LAST[SID] = (now, "preparing")

    with patch("common.async_worker.request_session_rerun") as mock_rerun:
        with patch("time.monotonic", return_value=now + 0.2):
            main_module._request_build_progress_rerun(SID, "extracting")

        mock_rerun.assert_not_called()


def test_extracting_rerun_allowed_after_initial_paint_window():
    """initial_paint_window(0.8s)가 경과한 후에는 EXTRACTING 리런이 정상 트리거되어야 함."""
    from core.build_state import BuildPhase, initial_build_state

    SessionManager.set("is_building_rag", True, session_id=SID)
    build = initial_build_state("doc.pdf", "hash123")
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, session_id=SID)

    now = 1000.0
    main_module._BUILD_RERUN_LAST[SID] = (now, "preparing")

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        with patch("time.monotonic", return_value=now + 1.0):
            main_module._request_build_progress_rerun(SID, "extracting")

        mock_rerun.assert_called_once_with(SID)
