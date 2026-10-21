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


def test_phase_transition_preparing_to_extracting_within_window_blocked():
    """PREPARING에서 EXTRACTING으로의 초기 위상 전이가 0.8초 윈도우 내에서는 억제되어 첫 페이지 페인트를 보장함."""
    start_time = time.monotonic()
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "preparing")

    build = SessionManager.get_build_state(session_id=SID)
    assert build is not None
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, SID)

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        with patch("time.monotonic", return_value=start_time + 0.2):
            main_module._request_build_progress_rerun(SID, "extracting")

        # 초기 0.8초 페인트 보호 윈도우 내이므로 리런이 호출되지 않아야 함
        mock_rerun.assert_not_called()


def test_phase_transition_after_initial_window_reruns_immediately():
    """초기 페인트 윈도우(0.8초) 경과 후의 위상 전이는 즉시 리런이 트리거되는지 검증."""
    start_time = time.monotonic()
    main_module._BUILD_RERUN_LAST[SID] = (start_time, "preparing")

    build = SessionManager.get_build_state(session_id=SID)
    assert build is not None
    build.phase = BuildPhase.EXTRACTING
    SessionManager.set("build_state", build, SID)

    with patch(
        "common.async_worker.request_session_rerun", return_value=True
    ) as mock_rerun:
        with patch("time.monotonic", return_value=start_time + 1.0):
            main_module._request_build_progress_rerun(SID, "extracting")

        # 윈도우 경과 후 위상 전이이므로 즉시 리런 호출
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
