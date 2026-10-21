"""단일 페이지 추출 시 deflate=False 최적화 및 유효성 검증 단위 테스트."""

import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

import src.main as main_module

from core.session import SessionManager


def _create_sample_pdf_bytes(text: str = "Fast Extract Test") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


class FakeSessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


class TestUploadPreviewNoDeflate(unittest.TestCase):
    def setUp(self):
        self.sid = "test_deflate_sid"
        SessionManager.reset()
        SessionManager.reset_all_state(self.sid)
        SessionManager.set_session_id(self.sid)

    def tearDown(self):
        SessionManager.reset()

    def test_upload_extracts_preview_without_deflate(self):
        """업로드 시 1페이지 추출에서 tobytes가 deflate=False로 호출되어 고속 추출되는지 검증."""
        pdf_bytes = _create_sample_pdf_bytes("First Page Fast Preview")
        uploaded = io.BytesIO(pdf_bytes)
        uploaded.name = "test_no_deflate.pdf"
        uploaded.type = "application/pdf"
        uploaded.size = len(pdf_bytes)

        fake_state = FakeSessionState({"pdf_uploader": uploaded})

        real_tobytes = pymupdf.Document.tobytes
        tobytes_calls = []

        def spy_tobytes(self_doc, *args, **kwargs):
            tobytes_calls.append(kwargs)
            return real_tobytes(self_doc, *args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("src.main.FilePathConstants.TEMP_DIR", tmp_dir),
            patch("src.main.st.session_state", fake_state),
            patch.object(
                pymupdf.Document, "tobytes", side_effect=spy_tobytes, autospec=True
            ),
            patch("services.monitoring.notification_system.SystemNotifier.success"),
        ):
            main_module.on_file_upload()

        # 1. tobytes 호출 인자 중 deflate=False 검증
        assert len(tobytes_calls) >= 1
        assert tobytes_calls[0].get("deflate") is False

        # 2. 세션에 저장된 1페이지 바이트의 무결성 검증
        preview = SessionManager.get("first_page_preview", session_id=self.sid)
        assert preview is not None
        assert preview["bytes"]

        with pymupdf.open(stream=preview["bytes"], filetype="pdf") as doc:
            assert len(doc) == 1
            assert "First Page Fast Preview" in doc[0].get_text()
