import asyncio
import pytest
import unittest.mock as mock
from core.resource_pool import ResourcePool

@pytest.mark.asyncio
async def test_resource_pool_lru_and_async_locks():
    """리소스 풀의 LRU 방출과 비동기 락이 정상 작동하는지 테스트"""
    pool = ResourcePool(max_size=2)
    await pool.clear() # 상태 초기화
    
    await pool.register("hash1", "vs1", "bm1")
    await pool.register("hash2", "vs2", "bm2")
    
    # hash1 조회하여 순서 갱신
    await pool.get("hash1")
    
    # hash3 등록 -> 가장 오래된 hash2 방출
    await pool.register("hash3", "vs3", "bm3")
    
    vs1, _ = await pool.get("hash1")
    vs2, _ = await pool.get("hash2")
    vs3, _ = await pool.get("hash3")
    
    assert vs1 == "vs1"
    assert vs2 is None
    assert vs3 == "vs3"

@pytest.mark.asyncio
async def test_resource_pool_concurrency():
    """동시에 여러 리소스를 등록할 때 데드락 없이 처리되는지 테스트"""
    pool = ResourcePool(max_size=5)
    await pool.clear()
    
    # 락의 안정성을 테스트하기 위해 많은 수의 태스크 실행
    async def task(i):
        # 락 경합 유도
        await pool.register(f"hash_{i}", f"vs_{i}", f"bm_{i}")
        return await pool.get(f"hash_{i}")

    # 비동기 태스크들을 실행하되, 순서 보장을 위해 약간의 간격을 둠 (또는 결과만 확인)
    tasks = [task(i) for i in range(10)]
    await asyncio.gather(*tasks)
    
    # 최종적으로 풀 크기가 max_size(5)를 유지하는지 확인
    assert len(pool._pool) <= 5
    assert len(pool._pool) > 0
    print(f"✅ 동시성 테스트 완료 (최종 풀 크기: {len(pool._pool)})")
