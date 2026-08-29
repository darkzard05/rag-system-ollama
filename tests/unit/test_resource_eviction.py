import asyncio
from unittest.mock import MagicMock, patch

import pytest

from core.resource_manager import (
    BaseResourcePool,
    ModelPool,
    ResourceCoordinator,
    RetrieverPool,
)


class MockResource:
    """Mock resource that simulates a FAISS index with specific size."""

    def __init__(self, ntotal=100, d=128, name="mock"):
        self.name = name
        self.ntotal = ntotal
        self.d = d
        # Simulate FAISS index attributes
        self.index = self


# --- Tests for BaseResourcePool ---


@pytest.mark.asyncio
async def test_lru_item_limit_eviction():
    """Verify that the oldest unpinned item is evicted when item_limit is reached."""
    pool: BaseResourcePool = BaseResourcePool("Test", item_limit=2, byte_limit=10**9)

    await pool.put("res1", MockResource(name="res1"))
    await pool.put("res2", MockResource(name="res2"))

    # Adding 3rd item should evict res1
    await pool.put("res3", MockResource(name="res3"))

    assert pool.get("res1") is None
    assert pool.get("res2") is not None
    assert pool.get("res3") is not None


@pytest.mark.asyncio
async def test_byte_limit_eviction():
    """Verify that items are evicted when byte_limit is exceeded."""
    # Set a small byte limit: 1000 bytes
    # MockResource size: ntotal * d * 4 = 100 * 128 * 4 = 51,200 bytes
    pool: BaseResourcePool = BaseResourcePool("Test", item_limit=10, byte_limit=60000)

    res1 = MockResource(ntotal=100, d=128)  # ~51KB
    res2 = MockResource(ntotal=100, d=128)  # ~51KB

    await pool.put("res1", res1)
    # Adding res2 should push total to ~102KB, exceeding 60KB limit -> evict res1
    await pool.put("res2", res2)

    assert pool.get("res1") is None
    assert pool.get("res2") is not None


@pytest.mark.asyncio
async def test_pinning_prevents_eviction():
    """Verify that pinned resources are not evicted even if they are the oldest."""
    pool: BaseResourcePool = BaseResourcePool("Test", item_limit=2, byte_limit=10**9)

    await pool.put("res1", MockResource(name="res1"))
    await pool.put("res2", MockResource(name="res2"))

    pool.pin("res1")  # Pin the oldest

    # Adding 3rd item should normally evict res1, but res1 is pinned -> evict res2
    await pool.put("res3", MockResource(name="res3"))

    assert pool.get("res1") is not None
    assert pool.get("res2") is None
    assert pool.get("res3") is not None

    pool.unpin("res1")
    await pool.put("res4", MockResource(name="res4"))
    assert pool.get("res1") is None  # Now it can be evicted


@pytest.mark.asyncio
async def test_concurrent_put_respects_capacity():
    """Regression: concurrent put() must not overfill byte/item limits.

    The previous implementation released the lock between the capacity check
    and the insert, letting multiple awaiters pass the check simultaneously and
    exceed ``byte_limit`` (VRAM OOM risk). The whole check→evict→insert is now
    atomic, so even under parallel load ``_current_bytes`` stays within bounds.
    """
    # Each resource ~51KB; item_limit 3, byte_limit 150KB -> at most 2-3 fit.
    pool: BaseResourcePool = BaseResourcePool("Conc", item_limit=3, byte_limit=150000)

    async def worker(idx: int) -> None:
        await pool.put(f"res{idx}", MockResource(name=f"res{idx}"))

    # Fire 12 concurrent puts; the pool must evict to stay within limits.
    await asyncio.gather(*[worker(i) for i in range(12)])

    assert len(pool._pool) <= pool.item_limit
    assert pool._current_bytes <= pool.byte_limit
    # No resource should be silently double-counted or leaked.
    assert pool._current_bytes >= 0


# --- Tests for RetrieverPool (RAM Pressure) ---


@pytest.mark.asyncio
async def test_retriever_memory_pressure_eviction():
    """Verify that RetrieverPool evicts resources when system RAM pressure is high."""
    pool = RetrieverPool("Retriever", item_limit=10, byte_limit=10**9)
    await pool.put("res1", MockResource(name="res1"))
    await pool.put("res2", MockResource(name="res2"))

    # Mock psutil.virtual_memory().percent to be > 85%
    with patch("psutil.virtual_memory") as mock_vm:
        # First call > 85% to trigger, second call > 80% to keep evicting, third < 80% to stop
        mock_vm.side_effect = [
            MagicMock(percent=90),  # Initial check
            MagicMock(percent=85),  # While loop check 1
            MagicMock(percent=75),  # While loop check 2 -> Stop
        ]

        evicted = await pool.check_memory_pressure()

        assert evicted is True
        # One or more should have been evicted. Since res1 was oldest...
        assert pool.get("res1") is None


@pytest.mark.asyncio
async def test_retriever_single_doc_no_eviction_under_pressure():
    """Regression: a single-document pool must NOT evict its only doc under
    memory pressure (would trigger an immediate full re-parse rebuild loop)."""
    pool = RetrieverPool("Retriever", item_limit=10, byte_limit=10**9)
    await pool.put("only_doc", MockResource(name="only_doc"))

    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value = MagicMock(percent=97.8)
        evicted = await pool.check_memory_pressure()

    assert evicted is False
    assert pool.get("only_doc") is not None


# --- Tests for ModelPool (VRAM Pressure) ---


@pytest.mark.asyncio
async def test_model_vram_pressure_eviction():
    """Verify that ModelPool evicts resources when VRAM pressure is high."""
    pool = ModelPool("Model", item_limit=10, byte_limit=10**9)
    await pool.put("res1", MockResource(name="res1"))

    # Mock torch.cuda to simulate VRAM pressure
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties") as mock_props,
        patch("torch.cuda.memory_reserved") as mock_res,
        patch("torch.cuda.empty_cache"),
        patch("torch.cuda.ipc_collect"),
    ):
        # Total = 10GB, Reserved = 9.5GB (95%)
        mock_props.return_value.total_memory = 10 * 1024**3
        mock_res.return_value = 9.5 * 1024**3

        evicted = await pool.check_vram_pressure()

        assert evicted is True
        assert pool.get("res1") is None


# --- ResourceCoordinator Integration ---


@pytest.mark.asyncio
async def test_coordinator_proactive_eviction():
    """Verify that ResourceCoordinator triggers pressure checks during get_or_build."""
    coord = ResourceCoordinator()
    coord.reset()

    # Mock the pressure check methods to return True
    with (
        patch.object(
            RetrieverPool, "check_memory_pressure", return_value=True
        ) as mock_mem,
        patch.object(ModelPool, "check_vram_pressure", return_value=True) as mock_vram,
    ):
        # Trigger retriever check
        await coord.get_or_build(coord.retrievers, "test_res", lambda: MockResource())
        mock_mem.assert_called_once()

        # Trigger model check
        await coord.get_or_build(coord.models, "test_model", lambda: MockResource())
        mock_vram.assert_called_once()
