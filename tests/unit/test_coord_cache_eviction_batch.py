"""
CoordCacheManager eviction 배치 정리·close 재사용 검증 (T14 / R1b-07, R1b-05).

- R1b-07: 크기 상한 초과 시 LIMIT 10(10행/사이클) 대신
  SUM(length(coords)) 기반 배치 삭제로 한 사이클 안에 정리되어야 합니다.
- R1b-05: close() 후 재사용 시 eviction 루프가 다시 예약되어야 합니다.
"""

import asyncio
import sqlite3
import time
from unittest.mock import patch

import orjson

from cache.coord_cache import COORD_CACHE_DB, CoordCacheManager

_KEEP_HASHES = [f"keep_{i}" for i in range(5)]
# 현실적 규모: keep ≪ _EVICTION_BATCH_SIZE(128) ≪ old — 한 번의 배치(128행)가
# 단일 DELETE로 전체 테이블을 휩쓸어 최신(keep) 행을 삭제하지 못하는 규모.
_OLD_HASHES = [f"old_{i}" for i in range(600)]


def _payload(size: int) -> list[dict[str, float | str]]:
    """대략 size개 단어의 좌표 페이로드를 생성합니다."""
    return [
        {"x0": float(i), "y0": float(i), "x1": 1.0, "y1": 1.0, "word": "w"}
        for i in range(size)
    ]


def _insert_rows(rows: list[tuple[str, int, bytes, float]]) -> None:
    """스키마 존재를 전제로 좌표 행을 직접 삽입합니다 (raw sqlite3)."""
    with sqlite3.connect(COORD_CACHE_DB) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO coords "
            "(file_hash, page_num, coords, created_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def _read_state() -> tuple[int, int, set[str]]:
    """테이블의 (행 수, 페이로드 합계 바이트, file_hash 집합)을 반환합니다."""
    with sqlite3.connect(COORD_CACHE_DB) as conn:
        count, total = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(length(coords)), 0) FROM coords"
        ).fetchone()
        hashes = {r[0] for r in conn.execute("SELECT file_hash FROM coords")}
    return count, total, hashes


def test_batch_eviction_cleans_over_limit_in_one_cycle():
    """상한 초과 시 한 사이클 안에 오래된 행을 배치 삭제해야 합니다."""
    manager = CoordCacheManager()
    try:
        asyncio.run(manager.clear_cache())

        now = time.time()
        rows: list[tuple[str, int, bytes, float]] = []
        for i, h in enumerate(_KEEP_HASHES):
            rows.append((h, 1, orjson.dumps(_payload(3)), now))
        for i, h in enumerate(_OLD_HASHES):
            rows.append((h, 1, orjson.dumps(_payload(200)), now - 100))
        _insert_rows(rows)

        # 상한을 4KB로 축소해 초과 상태를 재현 (old 600행 ~5.7MB, keep 5행 ~0.5KB)
        limit_bytes = 4096
        small_mb = limit_bytes / (1024 * 1024)
        with patch("cache.coord_cache.MAX_CACHE_SIZE_MB", small_mb):
            asyncio.run(manager._submit(manager._evict_old_entries()))

        count, total, hashes = _read_state()
        assert count == len(_KEEP_HASHES), (
            f"한 사이클에 초과분을 정리해야 합니다 (남은 {count}행)"
        )
        assert total <= limit_bytes, f"남은 페이로드가 상한을 초과했습니다: {total}B"
        assert hashes == set(_KEEP_HASHES), "최신 keep 행만 남아야 합니다"
    finally:
        asyncio.run(manager.clear_cache())
        asyncio.run(manager.close())


def test_eviction_loop_restarts_after_close():
    """close() 후 재사용 시 eviction 루프가 다시 예약되어야 합니다."""
    manager = CoordCacheManager()
    try:
        asyncio.run(manager._submit(manager._ensure_schema()))
        assert manager._eviction_started is True
        assert manager._eviction_task is not None
        first_task = manager._eviction_task

        asyncio.run(manager.close())
        assert manager._eviction_task is None

        # close 후 재사용 → _ensure_schema 조기 반환 없이 eviction 루프 재시작
        asyncio.run(manager._submit(manager._ensure_schema()))
        assert manager._eviction_started is True
        assert manager._eviction_task is not None
        assert manager._eviction_task is not first_task
        assert not manager._eviction_task.done()
    finally:
        asyncio.run(manager.close())
