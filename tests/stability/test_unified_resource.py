# tests/stability/test_unified_resource.py
import asyncio
import pytest
from src.core.model_loader import ModelManager


@pytest.mark.asyncio
async def test_unified_resource_lru_policy():
    """ModelManager의 통합 리소스 관리 및 LRU 정책을 검증합니다."""

    # 1. 테스트용 빌드 함수
    async def mock_build(name):
        return f"resource_{name}"

    # 2. 최대 캐시 크기 확인 (현재 5)
    limit = ModelManager.MAX_CACHED_MODELS

    # 3. 리소스 대량 등록
    for i in range(limit + 2):
        await ModelManager.get_or_build_resource(f"key_{i}", mock_build, f"key_{i}")

    # 4. 가장 오래된 리소스(key_0, key_1)가 해제되었는지 확인
    # ModelManager 내부 _instances 직접 확인
    with ModelManager._global_lock:
        assert "key_0" not in ModelManager._instances
        assert "key_1" not in ModelManager._instances
        assert "key_2" in ModelManager._instances
        assert len(ModelManager._instances) == limit


@pytest.mark.asyncio
async def test_unified_resource_concurrency():
    """동일 리소스에 대한 동시 빌드 요청이 안전하게 처리되는지 확인합니다."""
    build_count = 0

    async def slow_build():
        nonlocal build_count
        await asyncio.sleep(0.5)
        build_count += 1
        return "heavy_resource"

    # 여러 태스크가 동시에 동일 리소스 요청
    tasks = [
        ModelManager.get_or_build_resource("shared_key", slow_build) for _ in range(5)
    ]

    results = await asyncio.gather(*tasks)

    # 5개 요청 모두 동일한 결과를 받았지만, 실제 빌드는 1회만 수행되어야 함
    assert all(r == "heavy_resource" for r in results)
    assert build_count == 1
