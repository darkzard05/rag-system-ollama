import asyncio
import psutil
import os
import logging
import sys
from core.resource_manager import ResourceCoordinator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leak_test")


class MockRetriever:
    def __init__(self, ntotal, d):
        self.ntotal = ntotal
        self.d = d
        # Allocate actual memory to simulate FAISS index
        self.data = bytearray(ntotal * d * 4)


async def main():
    rc = ResourceCoordinator()
    rc.reset()
    pool = rc.retrievers

    # Set limits to trigger eviction quickly
    pool.item_limit = 10
    pool.byte_limit = 150 * 1024 * 1024  # 150MB

    process = psutil.Process(os.getpid())
    initial_rss = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Initial RSS: {initial_rss:.2f} MB")

    # 1. Load many retrievers to trigger eviction
    # Each retriever ~ 10MB (2.5M * 1 * 4 bytes)
    for i in range(50):
        res = MockRetriever(ntotal=2500000, d=1)
        await pool.put(f"doc_{i}", res)

        if i % 10 == 0:
            current_rss = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Loaded {i} docs, RSS: {current_rss:.2f} MB")

    final_rss = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Final RSS after loading 50 docs: {final_rss:.2f} MB")

    # Verify that RSS didn't grow linearly (50 * 10MB = 500MB)
    # It should be around initial + (item_limit * 10MB)
    expected_max = initial_rss + (pool.item_limit * 10) + 100  # with some buffer
    if final_rss < expected_max:
        logger.info(
            f"SUCCESS: Memory stabilized via eviction. {final_rss:.2f} MB < {expected_max:.2f} MB"
        )
    else:
        logger.error(
            f"FAILURE: Memory grew too much. RSS: {final_rss:.2f} MB, Expected < {expected_max:.2f} MB"
        )
        sys.exit(1)

    # 2. Test Pinning
    logger.info("Testing pinning...")
    pinned_key = "pinned_doc"
    pinned_res = MockRetriever(ntotal=2500000, d=1)
    await pool.put(pinned_key, pinned_res)
    pool.pin(pinned_key)

    # Load more to force eviction of others
    for i in range(20):
        await pool.put(f"extra_{i}", MockRetriever(ntotal=2500000, d=1))

    if pool.get(pinned_key) is not None:
        logger.info("SUCCESS: Pinned resource was not evicted.")
    else:
        logger.error("FAILURE: Pinned resource was evicted!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
