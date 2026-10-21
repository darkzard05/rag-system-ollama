"""PDF 뷰어 첫 페이지 선행 추출 캐싱 및 디스크 I/O 우회 검증 테스트."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

import ui.components.viewer as viewer_mod
from core.session import SessionManager
from ui.components.viewer import render_pdf_viewer


def _ensure_pdf_viewer_module(monkeypatch: pytest.MonkeyPatch, stub: Any) -> Any:
    module = types.ModuleType("streamlit_pdf_viewer")
    module.__dict__["pdf_viewer"] = stub
    sys.modules["streamlit_pdf_viewer"] = module
    return module


def test_first_page_preview_bypasses_disk_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """세션에 1페이지 미리보기 바이트가 있으면 디스크 추출 없이 즉시 렌더링한다."""
    sid = "test-first-page-speed"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)

    fake_bytes = b"%PDF-1.4 FAST-PREVIEW-PAGE1"
    file_hash = "fast_hash_123"
    SessionManager.set(
        "first_page_preview",
        {"file_hash": file_hash, "bytes": fake_bytes},
        session_id=sid,
    )

    calls: dict = {}

    def stub(**kwargs: Any) -> None:
        calls.update(kwargs)
        return None

    _ensure_pdf_viewer_module(monkeypatch, stub)

    mock_extract = MagicMock(
        side_effect=AssertionError("디스크 페이지 추출 함수가 호출되면 안 됨")
    )
    monkeypatch.setattr(viewer_mod, "_get_pdf_page_bytes", mock_extract)

    render_pdf_viewer("/fake/path/doc.pdf", 1, file_hash)

    # 1. 디스크 추출 함수는 일체 호출되지 않음
    mock_extract.assert_not_called()
    # 2. 뷰어 컴포넌트에 선행 추출된 바이트가 그대로 전달됨
    assert calls["input"] == fake_bytes
    assert calls["pages_to_render"] == [1]


def test_fallback_to_disk_when_page_different_or_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2페이지 요청 또는 파일 해시 불일치 시 디스크 추출 경로로 폴스루한다."""
    sid = "test-first-page-fallback"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)

    SessionManager.set(
        "first_page_preview",
        {"file_hash": "different_hash", "bytes": b"P1-DIFFERENT"},
        session_id=sid,
    )

    calls: dict = {}

    def stub(**kwargs: Any) -> None:
        calls.update(kwargs)
        return None

    _ensure_pdf_viewer_module(monkeypatch, stub)

    mock_extract = MagicMock(return_value=b"P2-DISK-BYTES")
    monkeypatch.setattr(viewer_mod, "_get_pdf_page_bytes", mock_extract)

    render_pdf_viewer("/fake/path/doc.pdf", 2, "target_hash")

    mock_extract.assert_called_once_with("/fake/path/doc.pdf", "target_hash", 2)
    assert calls["input"] == b"P2-DISK-BYTES"


def test_fallback_to_disk_when_hash_mismatch_on_first_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1페이지라도 미리보기 해시가 현재 파일과 다르면 디스크 추출을 사용한다."""
    sid = "test-first-page-hash-mismatch"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)

    SessionManager.set(
        "first_page_preview",
        {"file_hash": "stale_hash", "bytes": b"STALE-PREVIEW"},
        session_id=sid,
    )

    calls: dict = {}

    def stub(**kwargs: Any) -> None:
        calls.update(kwargs)
        return None

    _ensure_pdf_viewer_module(monkeypatch, stub)

    mock_extract = MagicMock(return_value=b"FRESH-DISK-BYTES")
    monkeypatch.setattr(viewer_mod, "_get_pdf_page_bytes", mock_extract)

    render_pdf_viewer("/fake/path/doc.pdf", 1, "current_hash")

    mock_extract.assert_called_once_with("/fake/path/doc.pdf", "current_hash", 1)
    assert calls["input"] == b"FRESH-DISK-BYTES"


def test_reset_for_new_file_clears_first_page_preview() -> None:
    """새 파일 전환 시 이전 문서의 미리보기가 남지 않아야 한다 (고스트 방지)."""
    sid = "test-first-page-reset"
    SessionManager.reset_all_state(sid)
    SessionManager.set(
        "first_page_preview",
        {"file_hash": "old_hash", "bytes": b"OLD"},
        session_id=sid,
    )

    SessionManager.reset_for_new_file(session_id=sid)

    assert SessionManager.get("first_page_preview", session_id=sid) is None


def test_upload_caches_first_page_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """업로드 검증 시 1페이지 바이트가 실제로 추출되어 세션에 저장된다."""
    import io

    import pymupdf
    from src.main import on_file_upload

    from core.document_processor import compute_file_hash

    sid = "test-first-page-upload"
    SessionManager.reset_all_state(sid)
    SessionManager.set_session_id(sid)

    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc[0].insert_text((72, 72), "FIRST PAGE MARKER")
    doc[1].insert_text((72, 72), "SECOND PAGE MARKER")
    file_bytes = doc.tobytes()
    doc.close()

    uploaded = io.BytesIO(file_bytes)
    uploaded.name = "preview.pdf"
    uploaded.type = "application/pdf"
    uploaded.size = len(file_bytes)

    monkeypatch.setattr("src.main.st.session_state", {"pdf_uploader": uploaded})
    monkeypatch.setattr("src.main.FilePathConstants.TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(
        "services.monitoring.notification_system.SystemNotifier.success",
        lambda *a, **k: None,
    )

    on_file_upload()

    preview = SessionManager.get("first_page_preview", session_id=sid)
    assert isinstance(preview, dict)
    assert preview["file_hash"] == compute_file_hash("", data=file_bytes)
    assert preview["bytes"]

    with pymupdf.open(stream=preview["bytes"], filetype="pdf") as single:
        assert len(single) == 1
        assert "FIRST PAGE MARKER" in single[0].get_text()
