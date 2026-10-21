"""Upload staged lifecycle — viewer first, analysis second.

Upload must no longer pre-empt the build on the callback thread. Stage 1
(``on_file_upload``) only publishes the viewer metadata and
``new_file_uploaded``; stage 2 (``_handle_pending_tasks``) initializes the
``BuildState`` and dispatches the rebuild worker.
"""

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
        uploaded.type = "application/pdf"
        uploaded.size = len(pdf_bytes)
        return FakeSessionState({"pdf_uploader": uploaded})

    def test_stage_1_upload_sets_preview_without_starting_build(self):
        """1단계: 업로드 직후에는 첫 페이지 프리뷰만 세팅되고 분석은 미시작."""
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

        # 첫 페이지 단독 선행 렌더 보장: 빌드 상태는 아직 비활성.
        assert (
            SessionManager.get("is_building_rag", False, session_id=self.sid) is False
        )
        assert SessionManager.get_build_state(session_id=self.sid) is None

    def test_stage_2_handle_pending_tasks_starts_build_naturally(self):
        """2단계: 첫 페이지 렌더 이후 펜딩 태스크 핸들러가 빌드를 개시한다."""
        SessionManager.set("new_file_uploaded", True, session_id=self.sid)
        SessionManager.set("last_uploaded_file_name", "sample.pdf", session_id=self.sid)
        SessionManager.set("file_hash", "hash_abc", session_id=self.sid)
        SessionManager.set("pdf_file_path", "/fake/sample.pdf", session_id=self.sid)

        with (
            patch("src.main._bg_rebuild_task", MagicMock()) as mock_task,
            patch("common.utils.run_in_background_worker") as mock_worker,
            patch("src.main.st.rerun") as mock_rerun,
        ):
            main_module._handle_pending_tasks()

        assert SessionManager.get("is_building_rag", False, session_id=self.sid) is True
        build_state = SessionManager.get_build_state(session_id=self.sid)
        assert build_state is not None
        assert build_state.file_name == "sample.pdf"
        assert build_state.file_hash == "hash_abc"

        mock_task.assert_called_once()
        mock_worker.assert_called_once()
        # 첫 페이지 페인트 프레임 보존: 업로드 직후 강제 rerun 금지.
        mock_rerun.assert_not_called()
        assert (
            SessionManager.get("new_file_uploaded", False, session_id=self.sid) is False
        )
