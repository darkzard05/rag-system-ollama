import asyncio
import threading
import pytest
from common.exceptions import LLMInferenceError
from core.resource_manager import get_resource_manager


def test_cross_loop_inference_lock_no_future_error():
    coordinator = get_resource_manager()
    coordinator.reset()

    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()

    t_a = threading.Thread(target=loop_a.run_forever, daemon=True)
    t_b = threading.Thread(target=loop_b.run_forever, daemon=True)
    t_a.start()
    t_b.start()

    held_event = threading.Event()
    release_event = threading.Event()

    async def task_a():
        await coordinator.acquire_inference_lock(timeout=5.0)
        held_event.set()
        while not release_event.is_set():
            await asyncio.sleep(0.01)
        coordinator.release_inference_lock()

    async def task_b():
        await coordinator.acquire_inference_lock(timeout=0.2)
        coordinator.release_inference_lock()

    fut_a = None
    fut_b = None
    try:
        fut_a = asyncio.run_coroutine_threadsafe(task_a(), loop_a)
        assert held_event.wait(timeout=2.0)

        fut_b = asyncio.run_coroutine_threadsafe(task_b(), loop_b)
        with pytest.raises(LLMInferenceError) as exc_info:
            fut_b.result(timeout=2.0)
        assert exc_info.value.details.get("reason") == "timeout"
    finally:
        release_event.set()
        if fut_a is not None:
            try:
                fut_a.result(timeout=2.0)
            except Exception:
                pass
        loop_a.call_soon_threadsafe(loop_a.stop)
        loop_b.call_soon_threadsafe(loop_b.stop)
        t_a.join(timeout=2.0)
        t_b.join(timeout=2.0)
        loop_a.close()
        loop_b.close()
