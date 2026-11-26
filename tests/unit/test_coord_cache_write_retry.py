"""A transient SQLite write failure must not discard a page's coordinates."""

import asyncio
from unittest.mock import patch

import aiosqlite
import pytest

from cache.coord_cache import CoordCacheManager

FILE_HASH = "transient_write_retry_hash"
PAGE = 3
PAYLOAD = [{"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0, "word": "retry"}]


@pytest.mark.asyncio
async def test_write_behind_retries_transient_lock() -> None:
    manager = CoordCacheManager()
    await manager.clear_cache(FILE_HASH)
    original = manager._insert_coords_impl
    calls = {"n": 0}

    async def flaky_insert(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise aiosqlite.OperationalError("database is locked")
        return await original(*args, **kwargs)

    try:
        with patch.object(manager, "_insert_coords_impl", side_effect=flaky_insert):
            assert await manager.save_coords(FILE_HASH, PAGE, PAYLOAD)
            data = None
            for _ in range(200):
                data = await manager.get_coords(FILE_HASH, PAGE)
                if data is not None:
                    break
                await asyncio.sleep(0.05)
        assert data == PAYLOAD
        assert calls["n"] >= 2
    finally:
        await manager.clear_cache(FILE_HASH)
        await manager.close()
