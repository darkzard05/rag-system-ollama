"""
Phase 2 단위 테스트: 좌표 캐시 QueueFull 시 조용한 드롭 금지.

실제 근본 원인: _save_coords_impl가 asyncio.QueueFull 시 경고 후 좌표를 버림(drop).
수정: QueueFull 시 즉시 SQLite 기록으로 폴백(_insert_coords_impl via _submit)하여
제품 차별화 기능(하이라이트) 좌표 유실을 방지.
"""

import asyncio

import pytest

from cache.coord_cache import CoordCacheManager

FILE_HASH = "queuefull_fallback_hash"
PAGE_NUM = 7
COORDS = [
    {"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0, "word": "fallback"},
]


def test_queue_full_falls_back_to_sqlite_no_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """큐가 가득 차도 좌표가 SQLite에 기록되어 조회 가능해야 함(드롭 금지)."""
    manager = CoordCacheManager()
    try:
        asyncio.run(manager.clear_cache(FILE_HASH))

        # 큐를 1칸으로 축소하여 즉시 QueueFull 유도
        manager._queue = asyncio.Queue(maxsize=1)
        # 워커가 즉시 소비하지 못하도록 백프레시어 경로 강제:
        # 워커 시작 전 저장 → 큐 가득 → 폴백 경로 태움
        manager._worker_task = None

        ok = asyncio.run(manager.save_coords(FILE_HASH, PAGE_NUM, COORDS))
        assert ok is True

        # 폴백 경로가 SQLite에 즉시 기록했으므로 워커 대기 불필요
        async def _get():
            for _ in range(50):
                result = await manager.get_coords(FILE_HASH, PAGE_NUM)
                if result is not None:
                    return result
                await asyncio.sleep(0.05)
            return None

        result = asyncio.run(_get())
        assert result == COORDS, "QueueFull 후에도 좌표가 유실되면 안 됨"

        # 기존 드롭 경고("생략")가 발생하지 않아야 함
        assert not any("생략" in record.message for record in caplog.records), (
            "조용한 드롭 경고가 발생하면 안 됨"
        )
    finally:
        asyncio.run(manager.clear_cache(FILE_HASH))
        asyncio.run(manager.close())
