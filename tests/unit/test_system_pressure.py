"""
Phase 1 단위 테스트: host_pressure_exceeded 임계값/psutil 부재 동작 검증.

실제 근본 원인(의존성 누락 아님): 호스트 RAM 90% 초과 시에만 모델 퇴출 발동 +
ImportError/런타임 오류를 조용히 삼킴(ImportError -> False, debug). 이를 수정:
- 임계값을 config(HOST_PRESSURE_THRESHOLD, 기본 85.0)로 외부화
- psutil 부재 시 warning + False
- 런타임 오류 시 warning + False
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from common.system_pressure import host_pressure_exceeded


def test_host_pressure_exceeded_true_above_threshold() -> None:
    """가상 메모리 점유율이 임계값 초과 시 True."""
    fake_psutil = MagicMock()
    fake_psutil.virtual_memory.return_value.percent = 90.0
    with patch.dict("sys.modules", {"psutil": fake_psutil}):
        assert host_pressure_exceeded(threshold=85.0) is True


def test_host_pressure_exceeded_false_below_threshold() -> None:
    """가상 메모리 점유율이 임계값 미만 시 False."""
    fake_psutil = MagicMock()
    fake_psutil.virtual_memory.return_value.percent = 50.0
    with patch.dict("sys.modules", {"psutil": fake_psutil}):
        assert host_pressure_exceeded(threshold=85.0) is False


def test_host_pressure_exceeded_import_error_warns_and_false(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """psutil 부재 시 경고 로그 1회 + False (조용한 삼킴 금지)."""
    with (
        patch.dict("sys.modules", {"psutil": None}),
        caplog.at_level("WARNING", logger="common.system_pressure"),
    ):
        result = host_pressure_exceeded(threshold=85.0)
    assert result is False
    assert any("psutil unavailable" in record.message for record in caplog.records), (
        "psutil 부재 시 경고가 발생해야 함"
    )


def test_host_pressure_exceeded_runtime_error_warns_and_false(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """psutil 런타임 오류 시 경고 로그 + False (debug 격상 확인)."""
    fake_psutil = MagicMock()
    fake_psutil.virtual_memory.side_effect = RuntimeError("boom")
    with (
        patch.dict("sys.modules", {"psutil": fake_psutil}),
        caplog.at_level("WARNING", logger="common.system_pressure"),
    ):
        result = host_pressure_exceeded(threshold=85.0)
    assert result is False
    assert any(
        "Host memory check failed" in record.message for record in caplog.records
    ), "런타임 오류 시 warning이 발생해야 함"
