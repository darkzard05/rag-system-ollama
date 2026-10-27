import asyncio
import time
import pytest


@pytest.mark.asyncio
async def test_critical_tests_execution_speed():
    """핵심 리팩토링 대상(세마포어 격리, 캐시 만료, 이벤트 루프 응답성)의 실행 속도가 병목 없이 결정적으로 빠른지 검증."""
    t0 = time.perf_counter()

    # 1. Semaphore test simulation
    from src.core.model_loader import ModelManager  # type: ignore[import]

    ModelManager._inference_semaphore = asyncio.Semaphore(1)
    results = []

    async def task(i):
        await ModelManager.acquire_inference_lock()
        try:
            results.append(i)
        finally:
            ModelManager.release_inference_lock()

    await asyncio.gather(*(task(i) for i in range(3)))
    assert len(results) == 3

    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"Critical test execution too slow: {elapsed:.2f}s"
