"""
CoordCacheManager 이벤트 루프 독립성 검증 (문제 1C).

각 asyncio.run() 호출은 별도의 이벤트 루프를 생성합니다.
save는 루프 A, get은 루프 B에서 실행해도 owner-루프 라우팅으로
"attached to a different loop" 오류 없이 동작해야 합니다.
"""

import asyncio

from cache.coord_cache import _WRITE_BEHIND_QUEUE_MAX, CoordCacheManager

FILE_HASH = "cross_loop_test_hash"
PAGE_NUM = 3
COORDS = [
    {"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0, "word": "test"},
    {"x0": 5.0, "y0": 6.0, "x1": 7.0, "y1": 8.0, "word": "loop"},
]


def test_save_then_get_across_different_loops():
    manager = CoordCacheManager()
    try:
        # 루프 A: 저장
        asyncio.run(manager.clear_cache(FILE_HASH))
        ok = asyncio.run(manager.save_coords(FILE_HASH, PAGE_NUM, COORDS))
        assert ok

        # 루프 B: write-behind 워커 반영 대기 후 조회
        async def _flush_and_get():
            for _ in range(50):
                result = await manager.get_coords(FILE_HASH, PAGE_NUM)
                if result is not None:
                    return result
                await asyncio.sleep(0.05)
            return None

        result = asyncio.run(_flush_and_get())
        assert result == COORDS

        # 루프 C: 배치 조회
        batch = asyncio.run(manager.get_coords_batch(FILE_HASH, [PAGE_NUM]))
        assert PAGE_NUM in batch
        assert batch[PAGE_NUM] == COORDS
    finally:
        asyncio.run(manager.clear_cache(FILE_HASH))
        asyncio.run(manager.close())


def test_missing_cache_returns_none():
    manager = CoordCacheManager()
    try:
        missing = asyncio.run(manager.get_coords("no_such_hash_zzz", 1))
        assert missing is None
    finally:
        asyncio.run(manager.close())


def test_close_from_owner_loop_does_not_crash():
    """owner 루프 위에서 close()를 호출해도 자기-join 크래시가 없어야 합니다 (C1)."""
    manager = CoordCacheManager()
    try:
        ok = asyncio.run(manager.save_coords(FILE_HASH, PAGE_NUM, COORDS))
        assert ok

        assert manager._owner_loop is not None
        # owner 루프 스레드 위에서 close() 실행 (self-join 가드 검증)
        future = asyncio.run_coroutine_threadsafe(manager.close(), manager._owner_loop)
        future.result(timeout=10)

        assert manager._owner_thread is None
        assert manager._owner_loop is None
    finally:
        asyncio.run(manager.close())


def test_save_coords_no_block_when_queue_full():
    """큐가 가득 찬 상태에서 save_coords가 블로킹 없이 반환해야 합니다 (C3)."""
    manager = CoordCacheManager()
    try:
        q = manager._queue
        for i in range(_WRITE_BEHIND_QUEUE_MAX):
            q.put_nowait((f"fill_{i}", 1, [{"word": "x"}]))
        assert q.full()

        # 가득 찬 상태에서도 블로킹 없이 반환되어야 함
        ok = asyncio.run(manager.save_coords(FILE_HASH, PAGE_NUM, COORDS))
        assert ok is True
        assert q.qsize() <= _WRITE_BEHIND_QUEUE_MAX
    finally:
        asyncio.run(manager.close())
