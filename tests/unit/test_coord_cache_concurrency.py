"""
CoordCacheManager 동시성 검증.

SQLite 기반 좌표 캐시에 대해 동일 키 (file_hash, page_num)로 여러
save_coords가 동시에 실행되어도 데이터 손상 없이 유효한 좌표 리스트를
읽을 수 있는지, 그리고 저장한 페이로드가 정확히 왕복되는지 검증합니다.
모든 public API는 async이며 owner 루프 라우팅으로 루프 독립적입니다.
"""

import asyncio

from cache.coord_cache import CoordCacheManager

CONCURRENCY_HASH = "test_concurrency_hash"
ROUND_TRIP_HASH = "test_round_trip_hash"
PAGE_NUM = 1
NUM_TASKS = 20


async def _concurrent_saves(manager, file_hash, page_num):
    """동일 키에 대해 여러 save_coords 코루틴을 동시에 실행합니다."""
    tasks = [
        asyncio.create_task(manager.save_coords(file_hash, page_num, _task_data(i)))
        for i in range(NUM_TASKS)
    ]
    return await asyncio.gather(*tasks)


def _task_data(task_index):
    """태스크 인덱스 기반 좌표 데이터 (JSON 왕복 후 튜플 → 리스트)."""
    return [(float(task_index), float(j), 0.0, 0.0, f"w{j}") for j in range(10)]


async def _flush_and_get(manager, file_hash, page_num):
    """write-behind 워커가 DB에 반영할 때까지 폴링 후 좌표를 조회합니다."""
    for _ in range(200):
        result = await manager.get_coords(file_hash, page_num)
        if result is not None:
            return result
        await asyncio.sleep(0.05)
    return None


def test_coord_cache_concurrent_writes():
    """동일 키 동시 저장 후에도 손상 없는 유효한 좌표 데이터를 읽어야 합니다."""
    manager = CoordCacheManager()
    writer_ids = {float(i) for i in range(NUM_TASKS)}
    try:
        asyncio.run(manager.clear_cache(CONCURRENCY_HASH))
        ok_results = asyncio.run(_concurrent_saves(manager, CONCURRENCY_HASH, PAGE_NUM))
        assert all(ok_results), "모든 동시 저장이 성공해야 합니다."

        data = asyncio.run(_flush_and_get(manager, CONCURRENCY_HASH, PAGE_NUM))
        assert data is not None, "동시 저장 후 좌표 데이터를 조회할 수 있어야 합니다."
        assert isinstance(data, list)
        assert len(data) == 10
        for entry in data:
            assert len(entry) >= 4, "좌표 항목은 최소 4개 요소를 가져야 합니다."
            assert isinstance(entry[0], (int, float))
            assert isinstance(entry[1], (int, float))
            assert entry[2] == 0.0
            assert entry[3] == 0.0
            assert isinstance(entry[4], str)

        # 모든 좌표가 단일 작성자(task index)의 데이터여야 함 (혼합/손상 방지)
        first_elements = {entry[0] for entry in data}
        assert len(first_elements) == 1, (
            "한 페이로드 내 좌표의 작성자가 일치해야 합니다."
        )
        assert first_elements.pop() in writer_ids, (
            "알 수 없는 작성자 데이터가 있어서는 안 됩니다."
        )
    finally:
        asyncio.run(manager.clear_cache(CONCURRENCY_HASH))
        asyncio.run(manager.close())


def test_coord_cache_round_trip_integrity():
    """신규 키에 저장한 페이로드가 정확히 동일하게 복원되어야 합니다."""
    manager = CoordCacheManager()
    page_num = 7
    payload = [
        {"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0, "word": "round"},
        {"x0": 5.0, "y0": 6.0, "x1": 7.0, "y1": 8.0, "word": "trip"},
    ]
    try:
        asyncio.run(manager.clear_cache(ROUND_TRIP_HASH))
        ok = asyncio.run(manager.save_coords(ROUND_TRIP_HASH, page_num, payload))
        assert ok, "페이로드 저장이 성공해야 합니다."

        data = asyncio.run(_flush_and_get(manager, ROUND_TRIP_HASH, page_num))
        assert data == payload, "저장한 페이로드가 정확히 동일하게 복원되어야 합니다."
    finally:
        asyncio.run(manager.clear_cache(ROUND_TRIP_HASH))
        asyncio.run(manager.close())
