import asyncio

import pytest

from core.model_loader import ModelManager
from core.resource_manager import get_resource_manager


@pytest.fixture(autouse=True)
def restore_semaphore():
    """Restores the inference semaphore after each test."""
    rc = get_resource_manager()
    original = rc.inference_semaphore
    yield
    rc.inference_semaphore = original


async def _entry_times(duration: float, count: int = 3) -> list[float]:
    """Runs `count` inference tasks and returns their entry timestamps."""

    async def mock_inference_task() -> float:
        start = asyncio.get_event_loop().time()
        async with ModelManager.inference_session():
            entry = asyncio.get_event_loop().time()
            await asyncio.sleep(duration)
        return entry - start

    return await asyncio.gather(*(mock_inference_task() for _ in range(count)))


@pytest.mark.asyncio
async def test_inference_semaphore_limits_concurrency():
    """With a semaphore of 2, 3 tasks must run in 2 waves (max overlap of 2)."""
    rc = get_resource_manager()
    rc.inference_semaphore = asyncio.Semaphore(2)

    duration = 0.2
    starts = await _entry_times(duration, count=3)
    starts.sort()

    # Third task must wait for the first two slots to free up.
    assert starts[2] >= starts[1] + duration - 0.05
    # First two tasks start near-immediately.
    assert starts[0] < 0.1
    assert starts[1] < 0.1


@pytest.mark.asyncio
async def test_inference_semaphore_serial_when_one():
    """With a semaphore of 1, all tasks must run serially."""
    rc = get_resource_manager()
    rc.inference_semaphore = asyncio.Semaphore(1)

    duration = 0.2
    starts = await _entry_times(duration, count=3)
    starts.sort()

    for i in range(2):
        assert starts[i + 1] >= starts[i] + duration - 0.05
