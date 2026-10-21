import os
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


def test_phase_transition_triggers_rerun_immediately_without_freeze():
    """위상이 PREPARING에서 EXTRACTING으로 변경될 때 0.8초 대기 없이 즉시 Rerun이 호출되는지 검증."""
    start_time = time.monotonic()
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "preparing")

    build = SessionManager.get_build_state(session_id=SID)
    assert build is not None
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, SID)

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        # 위상 전이는 발생 시점(예: 0.05초 후)에 억제되지 않고 즉각 리런되어야 함
        with patch("time.monotonic", return_value=start_time + 0.05):
            main_module._request_build_progress_rerun(SID, "extracting")

        mock_rerun.assert_called_once_with(SID)
        assert main_module._BUILD_RERUN_LAST[SID][1] == "extracting"


def test_same_phase_progress_is_properly_throttled():
    """동일 위상(EXTRACTING) 내에서의 진행률 갱신은 2.0초 동안 쓰로틀링되는지 검증."""
    start_time = time.monotonic()
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "extracting")

    build = SessionManager.get_build_state(session_id=SID)
    assert build is not None
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, SID)

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        # 1초 경과는 2초 미만이므로 리런 스킵
        with patch("time.monotonic", return_value=start_time + 1.0):
            main_module._request_build_progress_rerun(SID, "extracting")
        mock_rerun.assert_not_called()

        # 2.1초 경과 시에는 리런 허용
        with patch("time.monotonic", return_value=start_time + 2.1):
            main_module._request_build_progress_rerun(SID, "extracting")
        mock_rerun.assert_called_once_with(SID)
