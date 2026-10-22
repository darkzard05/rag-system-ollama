"""업로드 즉시 좌측 뷰어 1페이지 및 우측 분석 패널 동시 준비 검증 테스트."""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.main as main_module

import ui.components.viewer as viewer_module
from core.build_state import BuildPhase
from core.session import SessionManager
from ui.components.doc_analysis.viewmodel import to_viewmodel

SID = "test_instant_simultaneous_sid"


class FakeSessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


def _create_sample_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "INSTANT SIMULTANEOUS TEST - PAGE 1")
    doc.new_page().insert_text((72, 72), "INSTANT SIMULTANEOUS TEST - PAGE 2")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def setup_session():
    SessionManager.reset()
    SessionManager.init_session(SID)
    SessionManager.set_session_id(SID)
    yield
    SessionManager.reset()


def test_upload_instantly_prepares_both_viewer_and_panel_for_frame1():
    """업로드 콜백 직후 펜딩 태스크(Frame 2) 실행 없이도 Frame 1에서 양쪽 화면이 모두 준비되는지 검증."""
    pdf_bytes = _create_sample_pdf()
    uploaded = io.BytesIO(pdf_bytes)
    uploaded.name = "instant_test.pdf"
    uploaded.type = "application/pdf"  # type: ignore[attr-defined]
    uploaded.size = len(pdf_bytes)  # type: ignore[attr-defined]

    fake_state = FakeSessionState({"pdf_uploader": uploaded})

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("src.main.FilePathConstants.TEMP_DIR", tmp_dir),
        patch("src.main.st.session_state", fake_state),
        patch("services.monitoring.notification_system.SystemNotifier.success"),
    ):
        # 1. 파일 업로드 실행 (on_file_upload)
        main_module.on_file_upload()

    # [검증 1: 좌측 PDF 뷰어 상태 - Frame 1 렌더 준비 완료 여부]
    preview = SessionManager.get("first_page_preview", session_id=SID)
    assert preview is not None, "1페이지 미리보기 바이트가 등록되어야 함"
    assert preview["bytes"] != b""

    with patch("ui.components.viewer.st.session_state", {}):
        with patch("ui.components.viewer._get_pdf_page_bytes") as mock_page_bytes:
            viewer_state = viewer_module._resolve_pdf_state()
        mock_page_bytes.assert_not_called()
        assert viewer_state is not None, (
            "뷰어 상태 해석기가 즉시 유효한 상태를 반환해야 함"
        )
        assert viewer_state["current_page"] == 1
        assert viewer_state["total_pages"] == 2

    # [검증 2: 우측 분석 패널 상태 - Frame 1 렌더 준비 완료 여부]
    build_state = SessionManager.get_build_state(session_id=SID)
    assert build_state is not None, (
        "업로드 즉시 build_state가 생성되어 우측 패널이 빈 화면으로 조기 반환되지 않아야 함"
    )
    assert build_state.phase == BuildPhase.PREPARING
    assert SessionManager.get("is_building_rag", False, session_id=SID) is True

    # [검증 3: 우측 분석 패널 뷰모델 - PREPARING은 compact, EXTRACTING부터 확장]
    vm = to_viewmodel(build_state)
    assert vm is not None
    assert vm.status_state == "running"
    assert vm.expanded is False, (
        "PREPARING 단계에서는 패널이 접힌 채로 1페이지 뷰어의 선행성을 보장해야 함"
    )
    assert vm.progress is None, (
        "PREPARING 단계에서는 0% 게이지가 숨겨져 뷰어와의 마운트 경쟁이 없어야 함"
    )
    assert vm.stepper is None, "PREPARING 단계에서는 stepper를 노출하지 않아야 함"
