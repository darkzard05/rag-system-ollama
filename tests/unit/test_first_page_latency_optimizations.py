"""First-page latency optimizations — cache safety, memory fallback, rerun gates.

These tests pin the contracts introduced for the deferred upload persist:

* a missing on-disk PDF must never poison the 300s viewer cache with ``b""``;
* page extraction falls back to the in-memory upload bytes before the atomic
  write lands;
* build-progress reruns are suppressed both inside the initial paint window and
  between phases closer than the minimum rerun interval;
* ``first_page_preview`` stays in the SSoT instead of being mirrored into UI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.main as main_module  # noqa: E402

from core.build_state import BuildPhase, initial_build_state  # noqa: E402
from core.session import SessionManager  # noqa: E402
from ui.components import viewer as viewer_module  # noqa: E402

SID = "test_first_page_latency_sid"


def _make_pdf(pages: int = 2, token: str = "PAGE-TWO-TOKEN") -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 100), f"PAGE {i + 1} {token}")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def setup_session():
    SessionManager.reset()
    SessionManager.reset_all_state(SID)
    SessionManager.set_session_id(SID)
    main_module._PENDING_UPLOAD_BYTES.clear()
    main_module._BUILD_RERUN_LAST.clear()
    yield
    main_module._PENDING_UPLOAD_BYTES.clear()
    main_module._BUILD_RERUN_LAST.clear()
    SessionManager.reset()


def test_get_pdf_bytes_skips_cached_reader_when_file_missing(tmp_path):
    """파일이 없으면 내부 캐시 리더를 호출하지 않아 b"" 가 캐시되지 않는다."""
    missing = str(tmp_path / "not_written_yet.pdf")

    with patch.object(viewer_module, "_read_pdf_bytes_cached") as mock_cached:
        assert viewer_module._get_pdf_bytes(missing) == b""
        mock_cached.assert_not_called()

        # 지연 저장이 끝나 실제 파일이 생기면 정상적으로 바이트가 반환된다.
        target = tmp_path / "written.pdf"
        target.write_bytes(b"%PDF-1.4 real")
        mock_cached.return_value = b"%PDF-1.4 real"
        assert viewer_module._get_pdf_bytes(str(target)) == b"%PDF-1.4 real"
        mock_cached.assert_called_once_with(str(target), target.stat().st_mtime_ns)


def test_get_pdf_page_bytes_skips_cached_reader_when_file_missing(tmp_path):
    """페이지 추출도 파일이 없으면 내부 캐시 리더를 호출하지 않는다."""
    missing = str(tmp_path / "page_not_written.pdf")

    with patch.object(viewer_module, "_read_pdf_page_bytes_cached") as mock_cached:
        assert viewer_module._get_pdf_page_bytes(missing, "hash", 1) == b""
        mock_cached.assert_not_called()

        target = tmp_path / "page_written.pdf"
        target.write_bytes(_make_pdf(pages=1))
        mock_cached.return_value = b"%PDF-1.4 page"
        assert (
            viewer_module._get_pdf_page_bytes(str(target), "hash", 1)
            == b"%PDF-1.4 page"
        )
        mock_cached.assert_called_once_with(
            str(target), "hash", 1, target.stat().st_mtime_ns
        )


def test_get_pdf_page_bytes_uses_pending_memory_before_disk(tmp_path):
    """디스크 기록 전에도 메모리 업로드 바이트로 요청 페이지를 추출한다."""
    pdf_bytes = _make_pdf(pages=2, token="MEMORY-FALLBACK-TOKEN")
    main_module._PENDING_UPLOAD_BYTES[SID] = pdf_bytes
    missing = str(tmp_path / "pending_upload.pdf")

    page_bytes = viewer_module._get_pdf_page_bytes(missing, "hash-memory", 2)
    assert page_bytes, "메모리 폴백이 2페이지를 추출해야 함"

    with pymupdf.open(stream=page_bytes, filetype="pdf") as doc:
        assert len(doc) == 1
        assert "MEMORY-FALLBACK-TOKEN" in doc[0].get_text("text")

    # get_pending_upload_bytes 는 읽기 전용 조회를 제공한다.
    assert main_module.get_pending_upload_bytes(SID) == pdf_bytes
    assert main_module.get_pending_upload_bytes("no-such-session") is None


def test_build_progress_rerun_blocked_inside_initial_paint_window():
    """업로드 직후 첫 페인트 보호 윈도우 안에서는 위상 전이 리런이 0회다."""
    SessionManager.set("is_building_rag", True, session_id=SID)
    build = initial_build_state("doc.pdf", "hash")
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, session_id=SID)

    start = 1000.0
    main_module._BUILD_RERUN_LAST[SID] = (start, "preparing")

    with patch("common.async_worker.request_session_rerun") as mock_rerun:
        with patch("time.monotonic", return_value=start + 1.0):
            main_module._request_build_progress_rerun(SID, "extracting")
        mock_rerun.assert_not_called()

        with patch(
            "time.monotonic",
            return_value=start + main_module._INITIAL_PAINT_WINDOW_S + 0.1,
        ):
            main_module._request_build_progress_rerun(SID, "extracting")
        mock_rerun.assert_called_once_with(SID)


def test_build_progress_rerun_throttled_between_phases():
    """윈도우 밖에서도 최소 리런 간격 미만의 위상 전이는 건너뛴다."""
    SessionManager.set("is_building_rag", True, session_id=SID)
    build = initial_build_state("doc.pdf", "hash")
    build.phase = BuildPhase.CHUNKING
    SessionManager.set("build_state", build, session_id=SID)

    start = 1000.0
    main_module._BUILD_RERUN_LAST[SID] = (start, "extracting")

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        with patch(
            "time.monotonic",
            return_value=start + main_module._MIN_BUILD_RERUN_INTERVAL_S - 0.5,
        ):
            main_module._request_build_progress_rerun(SID, "chunking")
        mock_rerun.assert_not_called()

        with patch(
            "time.monotonic",
            return_value=start + main_module._MIN_BUILD_RERUN_INTERVAL_S + 0.1,
        ):
            main_module._request_build_progress_rerun(SID, "chunking")
        mock_rerun.assert_called_once_with(SID)


def test_first_page_preview_is_not_mirrored_to_ui():
    """대용량 1페이지 바이너리는 UI 미러에서 제외되고 SSoT 조회로 동작한다."""
    assert "first_page_preview" in SessionManager._NON_UI_SYNC_KEYS

    preview = {"file_hash": "hash", "bytes": b"%PDF-1.4 preview"}
    SessionManager.set("first_page_preview", preview, session_id=SID)

    stored = SessionManager.get("first_page_preview", session_id=SID)
    assert stored == preview
    assert stored is not None
