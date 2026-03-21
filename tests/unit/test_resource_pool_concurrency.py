import asyncio
import threading
import pytest
from src.core.resource_pool import ResourcePool


def run_in_new_loop(coro, result_container):
    """
    제공된 코루틴을 새로운 이벤트 루프에서 실행하고 결과를 컨테이너에 담습니다.
    이것은 Streamlit의 각 세션(스레드)이 독립적인 루프를 가지는 환경을 모사합니다.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result_container["result"] = loop.run_until_complete(coro)
    except Exception as e:
        import traceback
        result_container["error"] = e
        result_container["traceback"] = traceback.format_exc()
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_resource_pool_concurrency():
    """
    단일 이벤트 루프 내에서 여러 비동기 작업이 동시에 ResourcePool에 접근할 때의 안전성을 검증합니다.
    """
    pool = ResourcePool()

    async def register_task(i):
        await pool.register(f"file_{i}", f"vs_{i}", f"bm_{i}")
        return await pool.get(f"file_{i}")

    tasks = [register_task(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    assert all(r[0] is not None for r in results)


def test_resource_pool_multi_thread_concurrency():
    """
    여러 스레드에서 각자의 이벤트 루프를 통해 ResourcePool에 동시 접근할 때 안전하게 작동하는지 검증합니다.
    이것이 `threading.local()` 변경의 핵심 검증 시나리오입니다.
    """
    pool = ResourcePool()
    threads = []
    # 각 스레드 결과를 저장할 컨테이너 리스트
    results = [{} for _ in range(10)]

    for i in range(10):
        # 클로저 문제를 피하기 위해 기본 인자 사용
        async def register_task(idx=i):
            await pool.register(f"file_{idx}", f"vs_{idx}", f"bm_{idx}")
            return await pool.get(f"file_{idx}")

        thread = threading.Thread(target=run_in_new_loop, args=(register_task(), results[i]))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # 에러 발생 여부 확인
    errors = [(r.get('error'), r.get('traceback')) for r in results if 'error' in r]
    assert not errors, f"Errors occurred during threaded execution: {errors}"
    
    # 모든 작업이 결과를 반환했는지 확인
    assert all(r.get("result") is not None for r in results), "Some threads failed to get results"
    
    # 등록된 데이터가 올바른지 확인
    for i, r in enumerate(results):
        vs, bm = r.get("result")
        assert vs == f"vs_{i}"
        assert bm == f"bm_{i}"
