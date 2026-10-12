"""Repro: clear_cache must drop annotation_memo; close must drain queue."""

import asyncio
import contextlib
import importlib

import pytest

from cache.coord_cache import coord_cache


@pytest.fixture
def isolated_coord_db(tmp_path, monkeypatch):
    db_dir = tmp_path / "coord_cache"
    db_dir.mkdir()
    mod = importlib.import_module("cache.coord_cache")
    with contextlib.suppress(Exception):
        asyncio.run(coord_cache.close())
    coord_cache._conn = None
    coord_cache._schema_ready = False
    coord_cache._conn_healthy = False
    monkeypatch.setattr(mod, "COORD_CACHE_DIR", db_dir)
    monkeypatch.setattr(mod, "COORD_CACHE_DB", db_dir / "coords.db")
    yield db_dir
    with contextlib.suppress(Exception):
        asyncio.run(coord_cache.close())


def test_clear_cache_drops_annotation_memo(isolated_coord_db):
    annots = [{"page": 1, "x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}]
    assert coord_cache.save_annotation_memo_sync("h1", 1, "n1", annots) is True
    assert coord_cache.get_annotation_memo_sync("h1", 1, "n1") == annots

    asyncio.run(coord_cache.clear_cache("h1"))

    assert coord_cache.get_annotation_memo_sync("h1", 1, "n1") is None


def test_clear_cache_all_drops_annotation_memo(isolated_coord_db):
    annots = [{"page": 1}]
    assert coord_cache.save_annotation_memo_sync("hA", 1, "nA", annots) is True
    asyncio.run(coord_cache.clear_cache())
    assert coord_cache.get_annotation_memo_sync("hA", 1, "nA") is None


def test_close_drains_write_behind_queue(isolated_coord_db, monkeypatch):
    """Slow the worker so the queue backlogs deterministically, then close."""
    import cache.coord_cache as mod

    real_insert = coord_cache._insert_coords_impl

    async def _slow_insert(file_hash, page_num, coords):
        await asyncio.sleep(0.05)
        await real_insert(file_hash, page_num, coords)

    monkeypatch.setattr(coord_cache, "_insert_coords_impl", _slow_insert)

    async def _fill_and_close():
        for p in range(20):
            ok = await coord_cache.save_coords("h2", p, [{"x0": 1, "word": "w"}])
            assert ok is True
        await coord_cache.close()

    asyncio.run(_fill_and_close())

    # Reopen (next _submit restarts owner loop) and verify nothing was lost.
    async def _read_back():
        return await coord_cache.get_coords_batch("h2", list(range(20)))

    got = asyncio.run(_read_back())
    assert len(got) == 20, f"lost pages: have={sorted(got)}"
    _ = mod  # keep import explicit for monkeypatched constants
