import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import src.main as main_module

from core.build_state import BuildPhase, initial_build_state
from core.session import SessionManager

SID = "test_phase_transition_sid"


@pytest.fixture(autouse=True)
def setup_session():
    SessionManager.reset()
    SessionManager.init_session(SID)
    SessionManager.set_session_id(SID)
    SessionManager.set("is_building_rag", True, SID)
    build = initial_build_state("doc.pdf", "hash_abc")
    build.phase = BuildPhase.PREPARING
    SessionManager.set("build_state", build, SID)
    main_module._BUILD_RERUN_LAST.clear()
    yield
    SessionManager.reset()
    main_module._BUILD_RERUN_LAST.clear()


def test_phase_transition_is_suppressed_during_initial_paint_window():
    """PREPARING→EXTRACTING 전이는 첫 페인트 보호 윈도우(0.8s) 동안 억제된다."""
    start_time = time.monotonic()
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "preparing")

    build = SessionManager.get_build_state(session_id=SID)
    assert build is not None
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, SID)

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        with patch("time.monotonic", return_value=start_time + 0.05):
            main_module._request_build_progress_rerun(SID, "extracting")

        mock_rerun.assert_not_called()
        assert main_module._BUILD_RERUN_LAST[SID][1] == "preparing"


def test_phase_transition_reruns_after_initial_paint_window():
    """윈도우가 끝난 뒤 첫 위상 전이는 즉시 Rerun을 트리거한다."""
    start_time = time.monotonic()
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "preparing")

    build = SessionManager.get_build_state(session_id=SID)
    assert build is not None
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, SID)

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        with patch("time.monotonic", return_value=start_time + 0.9):
            main_module._request_build_progress_rerun(SID, "extracting")

        mock_rerun.assert_called_once_with(SID)
        assert main_module._BUILD_RERUN_LAST[SID][1] == "extracting"


def test_same_phase_progress_never_triggers_global_rerun():
    """동일 위상(EXTRACTING) 내 진행률 갱신은 시간이 지나도 Rerun을 유발하지 않는다."""
    start_time = time.monotonic()
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "extracting")

    build = SessionManager.get_build_state(session_id=SID)
    assert build is not None
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, SID)

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        with patch("time.monotonic", return_value=start_time + 1.0):
            main_module._request_build_progress_rerun(SID, "extracting")

        # 2초가 지나도 동일 위상 내 세부 % 갱신은 전역 Rerun을 유발하지 않는다.
        with patch("time.monotonic", return_value=start_time + 2.1):
            main_module._request_build_progress_rerun(SID, "extracting")

        mock_rerun.assert_not_called()
