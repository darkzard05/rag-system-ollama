"""
호스트 시스템 압력(RAM) 감지를 위한 공유 헬퍼.

Ollama는 별도 프로세스라 torch.cuda 미사용 환경(기본 배포)에서 VRAM 퇴출이
절대 발동하지 않는다. 이 모듈은 호스트 RAM 점유율 기반 폴백 판정을 한 곳에서
제공하여 ModelPool / ModelManager 가 공유한다.
"""

from __future__ import annotations

import logging
import time

from common.config import ENABLE_OLLAMA_PRESSURE_FALLBACK, HOST_PRESSURE_THRESHOLD

logger = logging.getLogger(__name__)

# 호스트 RAM 압력 기반 퇴출 스로틀 상수(초). Ollama처럼 별도 프로세스 백엔드에서는
# 파이썬 측 핸들 퇴출이 호스트 RAM을 줄이지 못해 조건이 영구히 참이 되어 매 호출
# "퇴출→즉시 재로드" 쓰래시가 난다. 마지막 퇴출 이후 이 시간 내 재퇴출을 차단한다.
EVICT_COOLDOWN_SECONDS: float = 30.0

# 풀/매니저별 마지막 퇴출 시각(monotonic). 런타임 단위 스로틀에만 사용.
_LAST_EVICT_TS: dict[str, float] = {}


def eviction_allowed(key: str) -> bool:
    """퇴출 쿨다운을 적용해 현재 퇴출이 허용되는지 반환한다.

    허용 시 타임스탬프를 갱신한다. 쿨다운 내면 False를 반환해 무의미한
    재퇴출/재로드 루프를 차단한다.
    """
    now = time.monotonic()
    last = _LAST_EVICT_TS.get(key, 0.0)
    if now - last < EVICT_COOLDOWN_SECONDS:
        return False
    _LAST_EVICT_TS[key] = now
    return True


def host_pressure_exceeded(threshold: float | None = None) -> bool:
    """
    호스트 가상 메모리 점유율이 임계값을 초과하면 True.

    psutil 부재 시 False 반환 (압력 감지 불가 → 퇴출 유도 안 함).
    """
    if threshold is None:
        threshold = HOST_PRESSURE_THRESHOLD
    try:
        import psutil
    except ImportError:
        logger.warning("psutil unavailable — host pressure detection disabled")
        return False
    try:
        return psutil.virtual_memory().percent > threshold
    except Exception as e:
        logger.warning(f"Host memory check failed: {e}")
        return False


def ollama_pressure_fallback_active() -> bool:
    """ENABLE_OLLAMA_PRESSURE_FALLBACK 플래그가 켜져 있는지."""
    return ENABLE_OLLAMA_PRESSURE_FALLBACK


def ollama_backend_active() -> bool:
    """기본 배포 백엔드가 Ollama인지 (torch.cuda 미사용 HF 경로가 아님)."""
    from common.config import DEFAULT_EMBEDDING_MODEL

    return "/" not in DEFAULT_EMBEDDING_MODEL or DEFAULT_EMBEDDING_MODEL.startswith(
        "ollama:"
    )
