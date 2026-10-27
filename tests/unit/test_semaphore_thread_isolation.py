import asyncio
import time
import pytest
from src.core.model_loader import ModelManager


# Helper to run task as an async coroutine
async def run_inference_task(task_id: int):
    # 비동기 세마포어 획득
    await ModelManager.acquire_inference_lock()
    try:
        results.append(("enter", task_id))
        await asyncio.sleep(0.001)
        results.append(("exit", task_id))
    finally:
        ModelManager.release_inference_lock()


@pytest.mark.asyncio
async def test_semaphore_isolation_across_tasks():
    """
    This test verifies that asyncio.Semaphore in ModelManager
    successfully protects across different async tasks.
    """
    # Ensure semaphore is correctly initialized for this test
    from src.common.config import MAX_CONCURRENT_INFERENCE  # type: ignore[import]

    # ModelManager._inference_semaphore is already asyncio.Semaphore
    # 강제로 1로 설정하여 순차 실행을 보장함
    ModelManager._inference_semaphore = asyncio.Semaphore(1)

    ModelManager._resource_loop_id = 0  # type: ignore[attr-defined]

    global results
    results = []

    start_time = time.time()
    num_tasks = 3

    # async tasks를 동시에 실행
    tasks = [asyncio.create_task(run_inference_task(i)) for i in range(num_tasks)]
    await asyncio.gather(*tasks)

    end_time = time.time()

    duration = end_time - start_time

    # Verification: Execution is deterministic and fast (< 0.2s instead of 3s)
    assert duration < 0.5, f"Expected fast execution, but took {duration:.2f}s"

    # Check if multiple tasks were in the session at the same time
    max_concurrent = 0
    current_concurrent = 0
    # results format: [("enter", tid), ("exit", tid), ...]
    # We can track enter/exit directly in order
    for event_type, tid in results:
        if event_type == "enter":
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
        else:
            current_concurrent -= 1

    assert max_concurrent == 1, (
        f"Expected exactly 1 concurrent entry, but found {max_concurrent}"
    )
    assert len(results) == 6
    for i in range(0, 6, 2):
        assert results[i][0] == "enter"
        assert results[i + 1][0] == "exit"
        assert results[i][1] == results[i + 1][1]
