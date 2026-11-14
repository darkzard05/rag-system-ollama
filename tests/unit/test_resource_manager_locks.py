import asyncio
import threading
from collections.abc import Iterator

import pytest

from core.resource_manager import (
    _BUILD_LOCK_MAX_ENTRIES,
    BaseResourcePool,
    CrossLoopAsyncLock,
    ResourceCoordinator,
)


@pytest.fixture
def coordinator() -> Iterator[ResourceCoordinator]:
    coord = ResourceCoordinator()
    coord.reset()
    yield coord
    coord.reset()


def test_cleanup_build_locks_does_not_cause_double_build_toctou(
    coordinator: ResourceCoordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        pool: BaseResourcePool[str] = BaseResourcePool(
            "Test", item_limit=10, byte_limit=1024 * 1024
        )
        build_count = 0
        orig_acquire = CrossLoopAsyncLock.acquire
        task1_cleaned = asyncio.Event()
        task2_entered = asyncio.Event()
        call_seq = 0

        async def patched_acquire(self: CrossLoopAsyncLock) -> bool:
            nonlocal call_seq
            call_seq += 1
            seq = call_seq
            if seq == 1:
                coordinator._cleanup_build_locks()
                task1_cleaned.set()
                await asyncio.wait_for(task2_entered.wait(), timeout=2.0)
            elif seq == 2:
                task2_entered.set()
            return await orig_acquire(self)

        monkeypatch.setattr(CrossLoopAsyncLock, "acquire", patched_acquire)

        async def slow_build() -> str:
            nonlocal build_count
            build_count += 1
            await asyncio.sleep(0)
            return "built_resource"

        async def run_task1() -> str:
            return await coordinator.get_or_build(pool, "target_key", slow_build)

        async def run_task2() -> str:
            await asyncio.wait_for(task1_cleaned.wait(), timeout=2.0)
            return await coordinator.get_or_build(pool, "target_key", slow_build)

        res1, res2 = await asyncio.gather(run_task1(), run_task2())
        assert res1 == "built_resource"
        assert res2 == "built_resource"
        assert build_count == 1

    asyncio.run(_run())


def test_evict_excess_build_locks_spares_referenced_lock(
    coordinator: ResourceCoordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        pool: BaseResourcePool[str] = BaseResourcePool(
            "Test", item_limit=100, byte_limit=10 * 1024 * 1024
        )
        build_count = 0
        orig_acquire = CrossLoopAsyncLock.acquire
        task1_evicted = asyncio.Event()
        task2_entered = asyncio.Event()
        call_seq = 0

        async def patched_acquire(self: CrossLoopAsyncLock) -> bool:
            nonlocal call_seq
            call_seq += 1
            seq = call_seq
            if seq == 1:
                for idx in range(_BUILD_LOCK_MAX_ENTRIES + 2):
                    coordinator._get_build_lock(f"filler_{idx}")
                task1_evicted.set()
                await asyncio.wait_for(task2_entered.wait(), timeout=2.0)
            elif seq == 2:
                task2_entered.set()
            return await orig_acquire(self)

        monkeypatch.setattr(CrossLoopAsyncLock, "acquire", patched_acquire)

        async def slow_build() -> str:
            nonlocal build_count
            build_count += 1
            await asyncio.sleep(0)
            return "built_resource"

        async def run_task1() -> str:
            return await coordinator.get_or_build(pool, "target_key", slow_build)

        async def run_task2() -> str:
            await asyncio.wait_for(task1_evicted.wait(), timeout=2.0)
            return await coordinator.get_or_build(pool, "target_key", slow_build)

        res1, res2 = await asyncio.gather(run_task1(), run_task2())
        assert res1 == "built_resource"
        assert res2 == "built_resource"
        assert build_count == 1

    asyncio.run(_run())


def test_concurrent_lock_cleanup_and_failure_tracking_thread_safe(
    coordinator: ResourceCoordinator,
) -> None:
    num_threads = 8
    barrier = threading.Barrier(num_threads)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(tid: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for step in range(100):
                key = f"k_{tid}_{step % 12}"
                coordinator._get_build_lock(key)
                coordinator._record_build_failure(key)
                coordinator._purge_expired_build_failures(0.0)
                if step % 5 == 0:
                    coordinator._cleanup_build_locks()
                if step % 7 == 0:
                    asyncio.run(coordinator.models.put(f"m_{tid}_{step % 3}", object()))
        except BaseException as exc:
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()

    assert errors == []
