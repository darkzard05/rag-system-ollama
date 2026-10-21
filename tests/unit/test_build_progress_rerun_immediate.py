"""빌드 착수 시 위상 전이 즉각 반응형 Rerun 검증 단위 테스트."""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import patch

import pytest

# src 경로 추가
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

import src.main as main_module

from core.build_state import BuildPhase, initial_build_state
from core.session import SessionManager

SID = "test_rerun_immediate_sid"


@pytest.fixture(autouse=True)
def setup_session():
    SessionManager.reset()
    SessionManager.init_session(SID)
    SessionManager.set_session_id(SID)
    SessionManager.set("is_building_rag", True, SID)
    build = initial_build_state("test_doc.pdf", "hash123")
    build.phase = BuildPhase.PREPARING
    SessionManager.set("build_state", build, SID)
    main_module._BUILD_RERUN_LAST.clear()
    yield
    SessionManager.reset()
    main_module._BUILD_RERUN_LAST.clear()


def test_phase_transition_reruns_immediately_without_cooldown():
    """위상이 변경될 때 1.5초 쿨다운에 걸리지 않고 즉시 리런이 트리거되는지 검증."""
    start_time = time.monotonic()
    # 이전 위상 기록
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "preparing")

    # 0.1초 직후 새 위상(extracting) 이벤트 도착 시뮬레이션
    build = SessionManager.get_build_state(session_id=SID)
    assert build is not None
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, SID)

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        with patch("time.monotonic", return_value=start_time + 0.1):
            main_module._request_build_progress_rerun(SID, "extracting")

        # 위상이 달라졌으므로 즉각적으로 리런이 호출되어야 함 (1.5초 차단 없음)
        mock_rerun.assert_called_once_with(SID)


def test_same_phase_updates_are_throttled():
    """동일 위상 내 세부 진행률은 최소 인터벌(2.0초) 동안 쓰로틀링되는지 검증."""
    start_time = time.monotonic()
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "extracting")

    build = SessionManager.get_build_state(session_id=SID)
    assert build is not None
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, SID)

    with patch("common.async_worker.request_session_rerun") as mock_rerun:
        with patch("time.monotonic", return_value=start_time + 0.5):
            main_module._request_build_progress_rerun(SID, "extracting")

        # 동일 위상 0.5초 경과는 쓰로틀링되어 리런이 호출되지 않아야 함
        mock_rerun.assert_not_called()
