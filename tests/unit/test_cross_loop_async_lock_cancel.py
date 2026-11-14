import asyncio
import threading

import pytest

from core.resource_manager import (
    BaseResourcePool,
    CrossLoopAsyncLock,
    ResourceCoordinator,
)


@pytest.mark.asyncio
async def test_cross_loop_lock_does_not_leak_on_waiter_cancellation() -> None:
    """대기 중인 태스크가 취소되어도 선행 홀더 해제 후 락이 누수되지 않아야 한다."""
    lock = CrossLoopAsyncLock()
    assert await lock.acquire() is True

    # 락 경합 중 타임아웃(내부적으로 waiter 태스크 취소) 발생
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(lock.acquire(), timeout=0.05)

    # 선행 홀더가 락을 해제한 뒤 새 대기자(다음 요청)는 즉시 락을 획득할 수 있어야 함
    lock.release()
    reacquired = await asyncio.wait_for(lock.acquire(), timeout=1.0)
    assert reacquired is True
    lock.release()


@pytest.mark.asyncio
async def test_cross_loop_lock_immediate_cancel_before_acquire() -> None:
    """실행 전 또는 직후 취소된 태스크도 락을 점유한 채 남기지 않아야 한다."""
    lock = CrossLoopAsyncLock()
    task = asyncio.create_task(lock.acquire())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # 직후 다른 태스크가 정상 획득/해제 가능해야 함
    async with lock:
        assert lock.locked() is True
    assert lock.locked() is False


@pytest.mark.asyncio
async def test_resource_coordinator_recovers_after_contended_waiter_cancelled() -> None:
    """ResourceCoordinator.get_or_build 대기 중 취소 후에도 동일 키 빌드가 데드락되지 않아야 한다."""
    coordinator = ResourceCoordinator()
    coordinator.reset()
    pool: BaseResourcePool[str] = BaseResourcePool(
        "Test", item_limit=4, byte_limit=1024 * 1024
    )

    build_entered = threading.Event()
    allow_finish = threading.Event()

    def slow_build() -> str:
        build_entered.set()
        allow_finish.wait(timeout=5.0)
        return "built_value"

    holder_task = asyncio.create_task(
        coordinator.get_or_build(pool, "key-1", slow_build)
    )
    try:
        entered = await asyncio.to_thread(build_entered.wait, 2.0)
        assert entered is True

        # 동일 키 대기 중 취소/타임아웃 발생
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                coordinator.get_or_build(pool, "key-1", lambda: "unused"),
                timeout=0.05,
            )

        # 선행 빌드 완료 후 후속 요청이 영구 데드락 없이 캐시된 값을 받아야 함
        allow_finish.set()
        assert await asyncio.wait_for(holder_task, timeout=1.0) == "built_value"

        recovered = await asyncio.wait_for(
            coordinator.get_or_build(pool, "key-1", lambda: "fallback"),
            timeout=1.0,
        )
        assert recovered == "built_value"
    finally:
        allow_finish.set()
        if not holder_task.done():
            await holder_task
        coordinator.reset()
