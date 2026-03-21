import asyncio
import pytest
from src.core.resource_pool import ResourcePool


def run_in_new_loop(coro, result_container):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result_container["result"] = loop.run_until_complete(coro)
    except Exception as e:
        result_container["error"] = e
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_resource_pool_concurrency():
    """
    ResourcePool이 여러 비동기 작업 환경에서도 안전하게 작동하는지 검증합니다.
    """
    pool = ResourcePool()

    # 여러 등록 작업이 동시 실행되어도 락이 안전하게 처리되는지 확인
    async def register_task(i):
        await pool.register(f"file_{i}", f"vs_{i}", f"bm_{i}")
        return await pool.get(f"file_{i}")

    tasks = [register_task(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    assert all(r[0] is not None for r in results)
