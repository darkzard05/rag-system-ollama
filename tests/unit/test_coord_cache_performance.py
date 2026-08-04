"""
CoordCacheManager 왕복 성능 검증.

save_coords → get_coords 왕복을 반복하며 (1) 매 왕복마다 저장한
페이로드가 정확히 복원되는지, (2) 전체 소요 시간이 관대한 상한을
넘지 않는지 검증합니다. 타이트한 타이밍보다 정확성 검증을 우선합니다.
"""

import asyncio
import time

from cache.coord_cache import CoordCacheManager

PERF_HASH = "test_perf_hash"
ITERATIONS = 20
MAX_TOTAL_SECONDS = 30.0
_FLUSH_ATTEMPTS = 200


def _payload(index):
    """인덱스별 고유 페이로드 (dict 기반 — JSON 왕복 후 정확 일치)."""
    return [
        {"x0": float(index), "y0": float(j), "x1": 0.0, "y1": 0.0, "word": f"w{j}"}
        for j in range(10)
    ]


async def _flush_and_get(manager, file_hash, page_num):
    """write-behind 워커가 DB에 반영할 때까지 폴링 후 좌표를 조회합니다."""
    for _ in range(_FLUSH_ATTEMPTS):
        result = await manager.get_coords(file_hash, page_num)
        if result is not None:
            return result
        await asyncio.sleep(0.01)
    return None


async def _round_trips(manager):
    """N회의 save→get 왕복을 순차 실행하며 매번 정확성을 검증합니다."""
    for i in range(ITERATIONS):
        page_num = i + 1
        payload = _payload(i)
        ok = await manager.save_coords(PERF_HASH, page_num, payload)
        assert ok, "저장이 성공해야 합니다."

        result = await _flush_and_get(manager, PERF_HASH, page_num)
        assert result == payload, f"왕복 {i}의 페이로드가 정확히 복원되어야 합니다."


def test_coord_cache_round_trip_performance():
    """N회 save→get 왕복이 정확하고 총 시간이 상한 이내여야 합니다."""
    manager = CoordCacheManager()
    try:
        asyncio.run(manager.clear_cache(PERF_HASH))
        start = time.perf_counter()
        asyncio.run(_round_trips(manager))
        duration = time.perf_counter() - start

        assert duration < MAX_TOTAL_SECONDS, (
            f"왕복 성능이 상한을 초과했습니다: {duration:.2f}s"
        )
    finally:
        asyncio.run(manager.clear_cache(PERF_HASH))
        asyncio.run(manager.close())
