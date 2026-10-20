"""PDF 뷰어 초기화 시 total_pages 세션 캐시 우회 검증.

업로드 단계(main._process_uploaded_file)에서 이미 계산해 세션에 저장한
total_pages 를 _resolve_pdf_state 가 재사용하여 중복 디스크 I/O 를 하지
않는지, 세션에 없을 때만 폴백 후 캐싱하는지 확인한다.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import ui.components.viewer as viewer_module  # noqa: E402
from core.session import SessionManager  # noqa: E402

SID = "test_viewer_cache_bypass_sid"


@pytest.fixture(autouse=True)
def setup_session(monkeypatch: pytest.MonkeyPatch):
    SessionManager.reset()
    SessionManager.init_session(SID)
    SessionManager.set_session_id(SID)
    SessionManager.set("pdf_file_path", "dummy_test.pdf", SID)
    SessionManager.set("file_hash", "test_hash_val", SID)
    fake_session_state: dict[str, object] = {}
    monkeypatch.setattr(viewer_module.st, "session_state", fake_session_state)
    yield
    SessionManager.reset()


def test_resolve_pdf_state_bypasses_disk_io_when_total_pages_in_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """세션에 total_pages가 있으면 디스크 파일 오픈이 0회여야 한다."""
    SessionManager.set("total_pages", 42, SID)

    mock_disk_get = MagicMock()
    monkeypatch.setattr(viewer_module, "_get_pdf_total_pages", mock_disk_get)

    state = viewer_module._resolve_pdf_state()

    assert state is not None
    assert state["total_pages"] == 42
    mock_disk_get.assert_not_called()


def test_resolve_pdf_state_falls_back_and_caches_when_session_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """세션에 total_pages가 없으면 디스크 파싱 후 세션에 캐싱해야 한다."""
    assert SessionManager.get("total_pages") is None

    mock_disk_get = MagicMock(return_value=15)
    monkeypatch.setattr(viewer_module, "_get_pdf_total_pages", mock_disk_get)

    state = viewer_module._resolve_pdf_state()

    assert state is not None
    assert state["total_pages"] == 15
    mock_disk_get.assert_called_once_with(os.path.abspath("dummy_test.pdf"))
    assert SessionManager.get("total_pages") == 15
