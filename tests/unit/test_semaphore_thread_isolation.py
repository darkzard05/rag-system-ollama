import asyncio
import time
import pytest
from src.core.model_loader import ModelManager


# Helper to run task as an async coroutine
async def run_inference_task(results, task_id):
    # 비동기 세마포어 획득
    await ModelManager.acquire_inference_lock()
    try:
        results.append(("enter", task_id, time.time()))
        await asyncio.sleep(1.0)
        results.append(("exit", task_id, time.time()))
    finally:
        ModelManager.release_inference_lock()


@pytest.mark.asyncio
async def test_semaphore_isolation_across_tasks():
    """
    This test verifies that asyncio.Semaphore in ModelManager
    successfully protects across different async tasks.
    """
    # Ensure semaphore is correctly initialized for this test
    from common.config import MAX_CONCURRENT_INFERENCE

    # ModelManager._inference_semaphore is already asyncio.Semaphore
    # 강제로 1로 설정하여 순차 실행을 보장함
    ModelManager._inference_semaphore = asyncio.Semaphore(1)

    ModelManager._resource_loop_id = 0

    results = []

    start_time = time.time()
    num_tasks = 3

    # async tasks를 동시에 실행
    tasks = [run_inference_task(results, i) for i in range(num_tasks)]
    await asyncio.gather(*tasks)

    end_time = time.time()

    duration = end_time - start_time
    print(
        f"\n[TEST] Total duration with asyncio.Semaphore across {num_tasks} tasks: {duration:.2f}s"
    )

    # Verification: Sequential execution means duration >= num_tasks * 1.0
    # In config.yml, max_concurrent_inference is 1.
    assert duration >= (num_tasks * 0.9), (
        f"Expected sequential execution, but took {duration:.2f}s"
    )

    # Check if multiple tasks were in the session at the same time
    max_concurrent = 0
    current_concurrent = 0
    sorted_events = sorted(results, key=lambda x: x[2])
    for event_type, tid, timestamp in sorted_events:
        if event_type == "enter":
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
        else:
            current_concurrent -= 1

    print(f"[TEST] Max concurrent entries detected: {max_concurrent}")
    assert max_concurrent == 1, (
        f"Expected exactly 1 concurrent entry, but found {max_concurrent}"
    )
