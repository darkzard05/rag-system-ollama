from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

import src.main as main_module
from core.session import SessionManager
import ui.components.viewer as viewer_module


def _make_dummy_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "SESSION ROUTING DUMMY PDF")
    data = doc.tobytes()
    doc.close()
    return data


def test_process_uploaded_file_routes_to_target_session():
    """ContextVar가 'default'인 상태에서도 지정된 세션 ID(target_sid)에 모든 파일 메타데이터가 정확히 라우팅되는지 검증."""
    target_sid = "isolated_user_session_777"
    SessionManager.reset()
    SessionManager.set_session_id("default")  # 의도적으로 ContextVar를 default로 설정
    SessionManager.init_session(target_sid)

    pdf_bytes = _make_dummy_pdf()
    uploaded = io.BytesIO(pdf_bytes)
    uploaded.name = "sample_routing.pdf"
    uploaded.type = "application/pdf"
    uploaded.size = len(pdf_bytes)

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("src.main.FilePathConstants.TEMP_DIR", tmp_dir),
        patch("src.main.SessionManager.get_session_id", return_value=target_sid),
        patch("services.monitoring.notification_system.SystemNotifier.success"),
    ):
        main_module._process_uploaded_file(uploaded)

    # 1. target_sid 세션에 파일 경로, 해시, 1페이지 미리보기, 업로드 플래그가 모두 존재해야 함
    assert SessionManager.get("pdf_file_path", session_id=target_sid) is not None
    assert SessionManager.get("file_hash", session_id=target_sid) is not None
    assert SessionManager.get("new_file_uploaded", session_id=target_sid) is True
    assert SessionManager.get("first_page_preview", session_id=target_sid) is not None

    # 2. default 세션에는 누출되지 않아야 함
    assert (
        SessionManager.get("pdf_file_path", session_id="default", create=False) is None
    )
    assert SessionManager.get("file_hash", session_id="default", create=False) is None

    # 3. 뷰어의 상태 해석기가 target_sid 세션에서 즉시 유효한 상태를 반환하는지 검증
    with patch(
        "ui.components.viewer.SessionManager.get_session_id", return_value=target_sid
    ):
        state = viewer_module._resolve_pdf_state()
        assert state is not None
        assert state["current_page"] == 1
        assert state["total_pages"] == 1
