"""빌드 착수 직후 전역 Rerun 억제 검증.

업로드 직후 PDF 뷰어가 마운트되는 동안 첫 진행 이벤트가 전역 Rerun을
발화하면 뷰어 렌더링이 리셋된다. 빌드 시작 시점에 등록한 쿨다운 기준점이
첫 이벤트를 억제하고, 쿨다운 경과 후에는 정상 리런을 허용하는지 확인한다.
"""

import time
from unittest.mock import patch

import pytest
import src.main as main_module

from core.build_state import BuildPhase, initial_build_state
from core.session import SessionManager

SID = "test_early_rerun_suppression_sid"


@pytest.fixture(autouse=True)
def setup_session():
    SessionManager.reset()
    SessionManager.init_session(SID)
    SessionManager.set_session_id(SID)
    SessionManager.set("is_building_rag", True, SID)
    build = initial_build_state("test_doc.pdf", "hash123")
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, SID)
    main_module._BUILD_RERUN_LAST.clear()
    yield
    SessionManager.reset()
    main_module._BUILD_RERUN_LAST.clear()


def test_early_event_is_suppressed_after_build_start():
    """빌드 착수 직후(0.2초 경과) 첫 이벤트는 1.5초 쿨다운에 의해 전역 리런이 차단되는지 검증."""
    start_time = time.monotonic()
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "start")

    with patch("common.async_worker.request_session_rerun") as mock_rerun:
        with patch("time.monotonic", return_value=start_time + 0.2):
            main_module._request_build_progress_rerun(SID, "extracting")

        mock_rerun.assert_not_called()


def test_rerun_allowed_after_cooldown_period():
    """쿨다운 기간(1.5초 초과) 경과 후에는 진행률 갱신 리런이 정상 트리거되는지 검증."""
    start_time = time.monotonic()
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "start")

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        with patch("time.monotonic", return_value=start_time + 1.6):
            main_module._request_build_progress_rerun(SID, "chunking")

        mock_rerun.assert_called_once_with(SID)
