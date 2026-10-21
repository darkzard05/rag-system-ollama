"""First page precedence and staged build initialization unit tests."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.main as main_module  # noqa: E402
from core.build_state import BuildPhase  # noqa: E402
from core.session import SessionManager  # noqa: E402


class FakeSessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


def _make_dummy_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "FIRST PAGE ISOLATION TEST")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def setup_session():
    sid = "test_first_page_isolation"
    SessionManager.reset()
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)
    yield sid
    SessionManager.reset()


def test_stage_1_upload_does_not_initialize_build_state(setup_session: str):
    """Stage 1: 파일 업로드 콜백 완료 직후 뷰어 미리보기 및 build_state(PREPARING)가 즉시 등록되어야 함."""
    sid = setup_session
    pdf_bytes = _make_dummy_pdf()

    uploaded = io.BytesIO(pdf_bytes)
    uploaded.name = "preview_first.pdf"
    uploaded.type = "application/pdf"  # type: ignore[attr-defined]
    uploaded.size = len(pdf_bytes)  # type: ignore[attr-defined]

    fake_state = FakeSessionState({"pdf_uploader": uploaded})

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("src.main.FilePathConstants.TEMP_DIR", tmp_dir),
        patch("src.main.st.session_state", fake_state),
        patch("services.monitoring.notification_system.SystemNotifier.success"),
    ):
        main_module.on_file_upload()

    # 1. 1페이지 미리보기 바이트 및 기본 뷰어 정보는 준비 완료
    preview = SessionManager.get("first_page_preview", session_id=sid)
    assert preview is not None, "first_page_preview가 등록되어야 함"
    assert preview["bytes"] != b""
    assert SessionManager.get("total_pages", session_id=sid) == 1
    assert SessionManager.get("new_file_uploaded", session_id=sid) is True

    # 2. 첫 프레임에서 1페이지 뷰어와 PREPARING 상태 패널이 함께 렌더링되도록 build_state가 즉시 초기화되어야 함
    build_state = SessionManager.get_build_state(session_id=sid)
    assert build_state is not None, (
        "업로드 즉시 build_state가 PREPARING으로 초기화되어야 함"
    )
    assert build_state.phase == BuildPhase.PREPARING


def test_stage_2_pending_tasks_initializes_build_state_after_first_frame(
    setup_session: str,
):
    """Stage 2: 펜딩 핸들러 단계에서 이미 초기화된 build_state를 유지하며 백그라운드 태스크를 디스패치함."""
    sid = setup_session
    SessionManager.set("new_file_uploaded", True, session_id=sid)
    SessionManager.set("last_uploaded_file_name", "preview_first.pdf", session_id=sid)
    SessionManager.set("file_hash", "hash_test_123", session_id=sid)
    SessionManager.set("pdf_file_path", "/tmp/preview_first.pdf", session_id=sid)
    SessionManager.init_build_state(
        "preview_first.pdf", "hash_test_123", session_id=sid
    )

    build_state_before = SessionManager.get_build_state(session_id=sid)
    assert build_state_before is not None
    assert build_state_before.phase == BuildPhase.PREPARING

    with (
        patch("src.main._bg_rebuild_task", MagicMock()) as mock_task,
        patch("common.utils.run_in_background_worker") as mock_worker,
        patch("src.main.st.rerun") as mock_rerun,
    ):
        main_module._handle_pending_tasks()

    build_state = SessionManager.get_build_state(session_id=sid)
    assert build_state is not None
    assert build_state.phase == BuildPhase.PREPARING
    assert SessionManager.get("is_building_rag", False, session_id=sid) is True
    assert SessionManager.get("new_file_uploaded", False, session_id=sid) is False
    mock_task.assert_called_once()
    mock_worker.assert_called_once()
    mock_rerun.assert_not_called()
