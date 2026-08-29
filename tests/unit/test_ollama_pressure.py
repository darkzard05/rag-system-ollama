"""
F4 검증: Ollama 인지형 리소스 퇴출 폴백.

기본 배포(backend=Ollama)는 별도 프로세스라 torch.cuda 미사용 → 기존 VRAM 퇴출
로직이 영원히 발동하지 않는 dead-code였다. ENABLE_OLLAMA_PRESSURE_FALLBACK +
호스트 RAM > 90% 에서 퇴출이 실제로 발동하는지 검증한다.

헤르메틱: torch.cuda.is_available / common.system_pressure.host_pressure_exceeded /
common.config 를 모두 스텁 — 실 모델/네트워크/GPU 호출 없이 CI 에서 green 이어야 함.
"""

from unittest.mock import MagicMock, patch

import pytest

import common.config as config_mod
from common.system_pressure import (
    ollama_backend_active,
    ollama_pressure_fallback_active,
)
from core.model_loader import ModelManager
from core.resource_manager import ModelPool


def _make_pool(items: int = 1) -> ModelPool:
    """비어있지 않은 풀을 생성해 _evict_one 이 항상 성공하도록."""
    pool = ModelPool(name="model", item_limit=5, byte_limit=1024 * 1024 * 512)
    for i in range(items):
        pool._pool[f"m{i}"] = object()
    return pool


def _patch_backend(monkeypatch, ollama: bool) -> None:
    """DEFAULT_EMBEDDING_MODEL 로 Ollama 백엔드 여부를 강제."""
    monkeypatch.setattr(
        config_mod,
        "DEFAULT_EMBEDDING_MODEL",
        "nomic-embed-text-v2-moe" if ollama else "BAAI/bge-m3",
    )


async def _async_true() -> bool:
    return True


async def _async_none() -> None:
    return None


# ---------------------------------------------------------------------------
# Baseline: 기존(수정 전) 동작 핀 - CUDA 경로 안 탈 때 호스트 RAM 압력 없으면 미발동
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_no_cuda_no_eviction(monkeypatch):
    """torch.cuda.is_available()=False + 플래그 ON + Ollama 여도,
    호스트 RAM 압력이 없으면 퇴출 안 함 — 수정 전 dead-code 계약."""
    _patch_backend(monkeypatch, ollama=True)
    monkeypatch.setattr(config_mod, "ENABLE_OLLAMA_PRESSURE_FALLBACK", True)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("common.system_pressure.host_pressure_exceeded", return_value=False),
        patch("common.system_pressure.ollama_backend_active", return_value=True),
    ):
        pool = _make_pool()
        result = await pool.check_vram_pressure()
    assert result is False


# ---------------------------------------------------------------------------
# Failing-first: 수정 후 동작 증명 - Ollama + RAM>90% → 퇴출 발동
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_fallback_fires_eviction_at_92pct(monkeypatch):
    """torch.cuda=False + RAM 92% + 플래그 ON + Ollama → True 및 _evict_one 호출."""
    _patch_backend(monkeypatch, ollama=True)
    monkeypatch.setattr(config_mod, "ENABLE_OLLAMA_PRESSURE_FALLBACK", True)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("common.system_pressure.host_pressure_exceeded", return_value=True),
        patch("common.system_pressure.ollama_backend_active", return_value=True),
    ):
        pool = _make_pool()
        with patch.object(pool, "_evict_one", new=MagicMock()) as evict:
            evict.side_effect = lambda: _async_true()
            result = await pool.check_vram_pressure()
    assert result is True
    assert evict.called


@pytest.mark.asyncio
async def test_ollama_fallback_no_eviction_at_70pct(monkeypatch):
    """torch.cuda=False + RAM 70% + 플래그 ON + Ollama → False (미만)."""
    _patch_backend(monkeypatch, ollama=True)
    monkeypatch.setattr(config_mod, "ENABLE_OLLAMA_PRESSURE_FALLBACK", True)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("common.system_pressure.host_pressure_exceeded", return_value=False),
        patch("common.system_pressure.ollama_backend_active", return_value=True),
    ):
        pool = _make_pool()
        result = await pool.check_vram_pressure()
    assert result is False


# ---------------------------------------------------------------------------
# Regression: 플래그 OFF 시 92% 에서도 퇴출 안 함 (기존 동작 보존)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_disables_fallback(monkeypatch):
    """플래그 OFF + RAM 92% + Ollama 여도 퇴출 안 함 (구 동작)."""
    _patch_backend(monkeypatch, ollama=True)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("common.system_pressure.host_pressure_exceeded", return_value=True),
        patch("common.system_pressure.ollama_backend_active", return_value=True),
        patch("core.resource_manager.ENABLE_OLLAMA_PRESSURE_FALLBACK", False),
    ):
        pool = _make_pool()
        result = await pool.check_vram_pressure()
    assert result is False


@pytest.mark.asyncio
async def test_hf_backend_not_triggered_by_host_ram(monkeypatch):
    """backend=HF(torch.cuda=False 시뮬)에서 호스트 RAM>90% 는 폴백 무관 퇴출 안 함."""
    _patch_backend(monkeypatch, ollama=False)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("common.system_pressure.host_pressure_exceeded", return_value=True),
        patch("common.system_pressure.ollama_backend_active", return_value=False),
        patch("core.resource_manager.ENABLE_OLLAMA_PRESSURE_FALLBACK", True),
    ):
        pool = _make_pool()
        result = await pool.check_vram_pressure()
    assert result is False


# ---------------------------------------------------------------------------
# ModelManager._check_memory_pressure 동일 계약
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_manager_ollama_fallback_fires(monkeypatch):
    """ModelManager 도 Ollama + RAM>90% + 플래그 ON 에서 _evict_oldest_model 호출."""
    _patch_backend(monkeypatch, ollama=True)
    monkeypatch.setattr(config_mod, "ENABLE_OLLAMA_PRESSURE_FALLBACK", True)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("common.system_pressure.host_pressure_exceeded", return_value=True),
        patch("common.system_pressure.ollama_backend_active", return_value=True),
        patch.object(ModelManager, "_evict_oldest_model", new=MagicMock()) as evict,
    ):
        evict.side_effect = _async_none
        result = await ModelManager._check_memory_pressure()
    assert result is True
    assert evict.called


@pytest.mark.asyncio
async def test_model_manager_flag_off_no_fallback(monkeypatch):
    """ModelManager 플래그 OFF + RAM>90% 에서 퇴출 안 함."""
    _patch_backend(monkeypatch, ollama=True)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("common.system_pressure.host_pressure_exceeded", return_value=True),
        patch("common.system_pressure.ollama_backend_active", return_value=True),
        patch("core.model_loader.ENABLE_OLLAMA_PRESSURE_FALLBACK", False),
    ):
        result = await ModelManager._check_memory_pressure()
    assert result is False


# ---------------------------------------------------------------------------
# Helper 단위
# ---------------------------------------------------------------------------


def test_ollama_backend_active_default_is_ollama(monkeypatch):
    """기본 DEFAULT_EMBEDDING_MODEL(nomic-embed-text-v2-moe) 는 Ollama 백엔드."""
    _patch_backend(monkeypatch, ollama=True)
    assert ollama_backend_active() is True


def test_ollama_backend_active_false_for_hf(monkeypatch):
    """'/' 포함 HF 모델명은 Ollama 백엔드 아님."""
    _patch_backend(monkeypatch, ollama=False)
    assert ollama_backend_active() is False


def test_flag_export_default_false():
    """ENABLE_OLLAMA_PRESSURE_FALLBACK 는 기본 배포에서 False 로 export.

    cfecea0(F3): 핸들 evict 가 Ollama host RAM 을 해방하지 못해 무효한
    퇴출→재로드만 유발하므로 기본 비활성화. 실제 메모리 반납은 Ollama
    keep_alive=0 호출이 필요하다(후속 개선 과제).
    """
    assert ollama_pressure_fallback_active() is False
