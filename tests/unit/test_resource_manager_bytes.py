import asyncio
import pytest
from unittest.mock import MagicMock
from core.resource_manager import ResourceManager, get_resource_manager


class MockFaissIndex:
    def __init__(self, ntotal, d):
        self.ntotal = ntotal
        self.d = d


class MockResource:
    def __init__(self, index):
        self.index = index


@pytest.mark.asyncio
async def test_resource_manager_byte_eviction():
    # Get singleton instance and clear it
    rm = get_resource_manager()
    await rm.clear_all()

    # Set a small byte limit for testing (e.g., 100 bytes)
    rm.retrievers.byte_limit = 100
    rm.retrievers.item_limit = 100  # Ensure item limit doesn't trigger first

    # Resource 1: 40 bytes (10 * 1 * 4)
    res1 = MockResource(MockFaissIndex(10, 1))
    await rm.retrievers.put("res1", res1)
    assert "res1" in rm.retrievers._pool
    assert rm.retrievers._current_bytes == 40

    # Resource 2: 40 bytes (10 * 1 * 4)
    res2 = MockResource(MockFaissIndex(10, 1))
    await rm.retrievers.put("res2", res2)
    assert "res2" in rm.retrievers._pool
    assert rm.retrievers._current_bytes == 80

    # Resource 3: 40 bytes (10 * 1 * 4)
    # Total would be 120 > 100. Should evict res1.
    res3 = MockResource(MockFaissIndex(10, 1))
    await rm.retrievers.put("res3", res3)

    assert "res1" not in rm.retrievers._pool
    assert "res2" in rm.retrievers._pool
    assert "res3" in rm.retrievers._pool
    assert rm.retrievers._current_bytes == 80


@pytest.mark.asyncio
async def test_resource_manager_large_resource_eviction():
    rm = get_resource_manager()
    await rm.clear_all()
    rm.retrievers.byte_limit = 100
    rm.retrievers.item_limit = 100

    # Resource 1: 40 bytes
    res1 = MockResource(MockFaissIndex(10, 1))
    await rm.retrievers.put("res1", res1)

    # Resource 2: 80 bytes (20 * 1 * 4)
    # Total would be 120 > 100. Should evict res1.
    res2 = MockResource(MockFaissIndex(20, 1))
    await rm.retrievers.put("res2", res2)

    assert "res1" not in rm.retrievers._pool
    assert "res2" in rm.retrievers._pool
    assert rm.retrievers._current_bytes == 80


@pytest.mark.asyncio
async def test_resource_manager_update_size():
    rm = get_resource_manager()
    await rm.clear_all()
    rm.retrievers.byte_limit = 100
    rm.retrievers.item_limit = 100

    # Resource 1: 40 bytes
    res1 = MockResource(MockFaissIndex(10, 1))
    await rm.retrievers.put("res1", res1)
    assert rm.retrievers._current_bytes == 40

    # Update Resource 1 to 80 bytes
    res1_updated = MockResource(MockFaissIndex(20, 1))
    await rm.retrievers.put("res1", res1_updated)
    assert rm.retrievers._current_bytes == 80
