"""부트 프리워밍이 사용자 상호작용에 즉시 양보하는지 검증."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import src.main as main_module

import common.config as config_module


@pytest.mark.asyncio
async def test_warmup_runs_after_25_poll_intervals(monkeypatch):
    """상호작용이 없으면 100ms × 25회 폴링 뒤 코어 워밍을 시작한다."""
    monkeypatch.setattr(config_module, "is_test_env", lambda: False)
    monkeypatch.setattr(main_module.SessionManager, "get", lambda *a, **k: None)
    sleep_mock = AsyncMock()
    monkeypatch.setattr(main_module.asyncio, "sleep", sleep_mock)
    core_warmup = AsyncMock()

    with patch("core.model_loader._warmup_models", core_warmup):
        await main_module._warmup_models()

    assert sleep_mock.await_count == 25
    core_warmup.assert_awaited_once()


@pytest.mark.asyncio
async def test_warmup_aborts_on_first_user_interaction(monkeypatch):
    """PDF 업로드/빌드/생성 중 하나라도 시작되면 첫 100ms 틱에서 중단한다."""
    monkeypatch.setattr(config_module, "is_test_env", lambda: False)
    monkeypatch.setattr(
        main_module.SessionManager,
        "get",
        lambda key, *a, **k: key == "pdf_file_path",
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr(main_module.asyncio, "sleep", sleep_mock)
    core_warmup = AsyncMock()

    with patch("core.model_loader._warmup_models", core_warmup):
        await main_module._warmup_models()

    assert sleep_mock.await_count == 1
    core_warmup.assert_not_awaited()
