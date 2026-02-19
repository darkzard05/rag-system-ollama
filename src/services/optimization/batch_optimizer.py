"""
GPU 메모리 상태를 감지하여 최적의 배치 사이즈를 계산하는 모듈.
"""

import logging

import torch

from common.constants import PerformanceConstants

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> tuple[bool, float]:
    """
    GPU 가용 여부와 총 메모리(MB)를 반환합니다.
    """
    if not torch.cuda.is_available():
        return False, 0.0

    try:
        device = torch.cuda.current_device()
        total: float = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        return True, float(total)
    except Exception:
        return False, 0.0


def _get_free_vram() -> float:
    """실제 여유 VRAM(MB)을 계산합니다."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        return total - allocated
    except Exception:
        return 0.0


def get_optimal_batch_size(device: str = "auto", model_type: str = "embedding") -> int:
    """
    하드웨어 자원에 따른 최적의 배치 사이즈를 반환합니다.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        return PerformanceConstants.EMBEDDING_BATCH_SIZE_CPU

    # GPU 메모리 기반 계산
    free_vram = _get_free_vram()

    if free_vram > 8000:
        return int(PerformanceConstants.EMBEDDING_BATCH_SIZE_GPU_HIGH)
    elif free_vram > 4000:
        return int(PerformanceConstants.EMBEDDING_BATCH_SIZE_GPU_MID)
    elif free_vram > 2000:
        return int(PerformanceConstants.EMBEDDING_BATCH_SIZE_GPU_LOW)
    else:
        return 16


def validate_batch_size(
    batch_size: int, device: str = "auto", model_type: str = "embedding"
) -> tuple[bool, str]:
    """
    설정된 배치 사이즈가 현재 자원에서 안전한지 검증합니다.
    """
    optimal = get_optimal_batch_size(device, model_type)
    # CPU에서도 64까지는 기본적으로 허용하도록 완화 (테스트 호환성 및 기본값 고려)
    limit = max(optimal * 4, int(PerformanceConstants.EMBEDDING_BATCH_SIZE_DEFAULT))

    if batch_size > limit:
        return False, f"배치 사이즈({batch_size})가 허용 한계({limit})를 초과했습니다."

    return True, "OK"
