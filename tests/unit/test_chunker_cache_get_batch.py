"""RED: _get_embeddings cache lookups must dedup + run concurrently.

Covers src/core/semantic_chunker.py _get_embeddings step 1 (sequential await
cache.get per input, duplicate inputs re-queried).
"""

import asyncio

import numpy as np
import pytest

from core.semantic_chunker import EmbeddingBasedSemanticChunker


class MockEmbeddings:
    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension
        self.model_name = "repro-model"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype="float32")
            vec[sum(ord(c) for c in text[:32]) % self.dimension] = 1.0
            results.append(vec.tolist())
        return results


class DelayTrackingCache:
    """Artificial latency per get to expose sequential awaits."""

    def __init__(self, delay: float = 0.02) -> None:
        self._store: dict[str, dict[str, object]] = {}
        self.delay = delay
        self.get_calls = 0
        self.max_in_flight = 0
        self._in_flight = 0

    async def get(self, key: str) -> dict[str, object] | None:
        self.get_calls += 1
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self._in_flight -= 1
        return self._store.get(key)

    async def set(
        self, key: str, value: dict[str, object], persist_to_disk: bool = True
    ) -> None:
        self._store[key] = value


def _make_chunker(cache: DelayTrackingCache) -> EmbeddingBasedSemanticChunker:
    return EmbeddingBasedSemanticChunker(
        embedder=MockEmbeddings(),  # type: ignore[arg-type]
        batch_size=64,
        cache_manager=cache,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_cache_get_dedups_and_runs_concurrently() -> None:
    cache = DelayTrackingCache(delay=0.02)
    chunker = _make_chunker(cache)
    texts = ["same text"] * 20 + [f"unique text {i}" for i in range(10)]

    result = await chunker._get_embeddings(texts)

    assert result.shape == (len(texts), 16)
    # 1 unique ("same text") + 10 uniques = 11 lookups max.
    assert cache.get_calls == 11, (
        f"sequential/duplicate get suspected: {cache.get_calls}"
    )
    assert cache.max_in_flight > 1, "cache lookups are fully sequential"
