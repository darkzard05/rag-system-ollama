"""Upload staged lifecycle — viewer first, analysis second.

Upload must no longer pre-empt the build on the callback thread. Stage 1
(``on_file_upload``) only publishes the viewer metadata and
``new_file_uploaded``; stage 2 (``_handle_pending_tasks``) initializes the
``BuildState`` and dispatches the rebuild worker.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# 프로젝트 루트 경로 추가 (src를 import path에 올린다)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

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


def _make_dummy_pdf(text: str = "Test Content") -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


class TestUploadStagedLifecycle(unittest.TestCase):
    def setUp(self):
        self.sid = "test_staged_sid"
        SessionManager.reset()
        SessionManager.reset_all_state(self.sid)
        SessionManager.set_session_id(self.sid)

    def tearDown(self):
        SessionManager.reset()

    def _upload_state(self) -> FakeSessionState:
        pdf_bytes = _make_dummy_pdf("Page 1 Content")
        uploaded = io.BytesIO(pdf_bytes)
        uploaded.name = "sample.pdf"
        uploaded.type = "application/pdf"  # type: ignore[attr-defined]
        uploaded.size = len(pdf_bytes)  # type: ignore[attr-defined]
        return FakeSessionState({"pdf_uploader": uploaded})

    def test_stage_1_upload_does_not_initialize_build_state(self):
        """1단계: 업로드 즉시 1페이지 미리보기는 등록되지만 build_state는 생성되지 않아 뷰어 선행 렌더링을 보장함."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("src.main.FilePathConstants.TEMP_DIR", tmp_dir),
            patch("src.main.st.session_state", self._upload_state()),
            patch("services.monitoring.notification_system.SystemNotifier.success"),
        ):
            main_module.on_file_upload()

        preview = SessionManager.get("first_page_preview", session_id=self.sid)
        assert preview is not None
        assert preview["bytes"] != b""
        assert SessionManager.get("total_pages", session_id=self.sid) == 1
        assert SessionManager.get("new_file_uploaded", session_id=self.sid) is True

        # 업로드 콜백 단계(Stage 1)에서는 BuildState가 생성되지 않아야 함 (뷰어 선행 렌더링 보장)
        build_state = SessionManager.get_build_state(session_id=self.sid)
        assert build_state is None
        assert (
            SessionManager.get("is_building_rag", False, session_id=self.sid) is False
        )

    def test_stage_2_handle_pending_tasks_starts_build_naturally(self):
        """2단계: 첫 페이지 렌더 후 펜딩 핸들러가 BuildState를 초기화하고 빌드를 가동함."""
        SessionManager.set("new_file_uploaded", True, session_id=self.sid)
        SessionManager.set("last_uploaded_file_name", "sample.pdf", session_id=self.sid)
        SessionManager.set("file_hash", "hash_abc", session_id=self.sid)
        SessionManager.set("pdf_file_path", "/fake/sample.pdf", session_id=self.sid)

        # 업로드 직후 상태이므로 build_state는 None
        assert SessionManager.get_build_state(session_id=self.sid) is None

        with (
            patch("src.main._bg_rebuild_task", MagicMock()) as mock_task,
            patch("common.utils.run_in_background_worker") as mock_worker,
            patch("src.main.st.rerun") as mock_rerun,
        ):
            main_module._handle_pending_tasks()

        assert SessionManager.get("is_building_rag", False, session_id=self.sid) is True
        build_state = SessionManager.get_build_state(session_id=self.sid)
        assert build_state is not None
        assert build_state.phase == BuildPhase.PREPARING

        mock_task.assert_called_once()
        mock_worker.assert_called_once()
        mock_rerun.assert_not_called()
        assert (
            SessionManager.get("new_file_uploaded", False, session_id=self.sid) is False
        )
