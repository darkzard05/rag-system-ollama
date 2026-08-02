import time
import pytest
from cache.coord_cache import CoordCacheManager


def test_coord_cache_write_performance():
    manager = CoordCacheManager()
    file_hash = "test_perf_hash"
    page_num = 1
    manager.clear_cache(file_hash)

    iterations = 500
    start_time = time.time()

    for i in range(iterations):
        data = [(float(i), 0.0, 0.0, 0.0)] * 10
        manager.save_coords(file_hash, page_num, data)

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n{iterations} iterations took {duration:.4f} seconds")

    # 500 iterations should easily take less than 2 seconds even with disk I/O
    assert duration < 2.0, f"Performance too slow: {duration:.4f}s"


if __name__ == "__main__":
    pytest.main([__file__])
