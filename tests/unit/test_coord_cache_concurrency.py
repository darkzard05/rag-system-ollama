import concurrent.futures
import orjson
from pathlib import Path
import pytest
from cache.coord_cache import CoordCacheManager, COORD_CACHE_DIR


def test_coord_cache_concurrent_writes():
    """
    여러 스레드에서 동일한 캐시 파일에 동시에 쓰기를 시도할 때
    데이터 손상이 발생하지 않고 유효한 JSON이 유지되는지 테스트합니다.
    """
    manager = CoordCacheManager()
    file_hash = "test_concurrency_hash"
    page_num = 1

    # 테스트 전 캐시 삭제
    manager.clear_cache(file_hash)

    num_threads = 20
    iterations_per_thread = 50

    def writer(thread_id):
        for i in range(iterations_per_thread):
            # 각 스레드가 고유한 데이터를 쓰려고 시도
            data = [(float(thread_id), float(i), 0.0, 0.0)] * 10
            manager.save_coords(file_hash, page_num, data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(writer, t) for t in range(num_threads)]
        concurrent.futures.wait(futures)

    # 최종 결과 검증
    cache_path = COORD_CACHE_DIR / f"{file_hash}_p{page_num}.json"
    assert cache_path.exists(), "캐시 파일이 생성되어야 합니다."

    try:
        with open(cache_path, "rb") as f:
            content = f.read()
            # 유효한 JSON인지 확인
            data = orjson.loads(content)
            assert isinstance(data, list), "데이터는 리스트 형태여야 합니다."
            assert len(data) == 10, "데이터 길이가 예상과 다릅니다."
    except Exception as e:
        pytest.fail(f"동시 쓰기 후 파일이 손상되었습니다: {e}")


def test_coord_cache_concurrent_read_write():
    """
    읽기와 쓰기가 동시에 일어날 때 읽기 작업이 손상된 데이터를 읽지 않는지 테스트합니다.
    """
    manager = CoordCacheManager()
    file_hash = "test_rw_concurrency_hash"
    page_num = 1
    manager.clear_cache(file_hash)

    stop_event = False

    def writer():
        nonlocal stop_event
        i = 0
        while not stop_event:
            data = [(float(i), 0.0, 0.0, 0.0)] * 100
            manager.save_coords(file_hash, page_num, data)
            i += 1

    def reader():
        nonlocal stop_event
        while not stop_event:
            data = manager.get_coords(file_hash, page_num)
            if data is not None:
                # 데이터가 있다면 유효한 형태인지 확인
                assert isinstance(data, list)
                assert len(data) == 100

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        writer_future = executor.submit(writer)
        reader_futures = [executor.submit(reader) for _ in range(3)]

        # 일정 시간 동안 테스트 수행
        import time

        time.sleep(2)
        stop_event = True

        concurrent.futures.wait([writer_future] + reader_futures)


if __name__ == "__main__":
    pytest.main([__file__])
