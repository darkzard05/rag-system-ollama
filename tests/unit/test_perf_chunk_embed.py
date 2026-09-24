"""chunk-embed 성능 TDD: _get_embeddings warm-cache + hit율 로그 검증.

Baseline: reports/perf_baseline_20260924_091935.md
- t_embed ≈ 0ms on cache hit, M_unique = len(missing_texts),
  캐시 효율 = 1 - M_unique / 총청크수, 캐시키 emb:{model}:{xxhash}.
"""

import asyncio
import logging
import time

import numpy as np
import pytest

from core.semantic_chunker import EmbeddingBasedSemanticChunker


class FakeEmbedder:
    """호출 횟수를 세는 비-Ollama 가짜 임베더 (HF 분할 경로 365-368)."""

    def __init__(self, dimension: int = 16, delay: float = 0.05) -> None:
        self.dimension = dimension
        self.model_name = "fake-perf-model"
        self.delay = delay
        self.embed_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import time as _time

        self.embed_calls += 1
        _time.sleep(self.delay)
        results: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype="float32")
            vec[sum(ord(c) for c in text[:32]) % self.dimension] = 1.0
            results.append(vec.tolist())
        return results


class FakeCacheManager:
    """persist_to_disk 플래그를 기록하는 가짜 캐시 매니저."""

    def __init__(self, set_delay: float = 0.0) -> None:
        self._store: dict[str, dict[str, object]] = {}
        self.persist_flags: list[bool] = []
        self.set_delay = set_delay
        self.max_in_flight = 0
        self._in_flight = 0

    async def get(self, key: str) -> dict[str, object] | None:
        return self._store.get(key)

    async def set(
        self, key: str, value: dict[str, object], persist_to_disk: bool = True
    ) -> None:
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self.set_delay > 0:
                await asyncio.sleep(self.set_delay)
        finally:
            self._in_flight -= 1
        self.persist_flags.append(persist_to_disk)
        self._store[key] = value


def _make_chunker(
    embedder: FakeEmbedder, cache: FakeCacheManager
) -> EmbeddingBasedSemanticChunker:
    return EmbeddingBasedSemanticChunker(
        embedder=embedder,  # type: ignore[arg-type]
        batch_size=64,
        cache_manager=cache,  # type: ignore[arg-type]
    )


TEXTS = [
    "GraphRAG chunk embed perf test alpha.",
    "GraphRAG chunk embed perf test beta.",
    "GraphRAG chunk embed perf test gamma.",
    "GraphRAG chunk embed perf test delta.",
]


@pytest.mark.asyncio
async def test_cache_save_is_batched_outside_single_wait() -> None:
    """RED-1: miss분 캐시 저장은 단일 wait_for + gather 병렬 배치.

    Baseline: miss가 많을수록(저hit율)逐차 per-item wait_for가 t_embed 지배.
    6건 miss × set 0.1s →逐차 0.6s 이상, 병렬 배치 ≈ 0.1s.
    동시성 구조로 판정(max_in_flight)하여 타이머 플레이크를 제거한다.
    디스크 영속 계약(test_chunking_does_not_write_response_cache_to_disk)은
    유지되므로 persist_to_disk=True를 함께 고정한다.
    """
    embedder = FakeEmbedder(delay=0.0)
    cache = FakeCacheManager(set_delay=0.1)
    chunker = _make_chunker(embedder, cache)
    texts = [f"batch save perf text number {i}." for i in range(6)]
    result = await chunker._get_embeddings(texts)
    assert result.shape == (len(texts), embedder.dimension)
    assert len(cache.persist_flags) == len(texts)
    assert all(flag is True for flag in cache.persist_flags)
    assert cache.max_in_flight == len(texts), (
        f"逐차 저장 의심: max_in_flight={cache.max_in_flight}"
    )


@pytest.mark.asyncio
async def test_miss_logs_hit_rate(caplog: pytest.LogCaptureFixture) -> None:
    """RED-2: miss 경로에서 hit율(newly_embedded/len) 로그 출력."""
    embedder = FakeEmbedder()
    cache = FakeCacheManager()
    chunker = _make_chunker(embedder, cache)
    # 2건만 사전 워밍 → 4건 중 2건 miss → hit율 50.0%.
    await chunker._get_embeddings(TEXTS[:2])
    cache.persist_flags.clear()
    with caplog.at_level(logging.DEBUG, logger="core.semantic_chunker"):
        await chunker._get_embeddings(TEXTS)
    assert "hit율" in caplog.text
    assert "50.0%" in caplog.text


@pytest.mark.asyncio
async def test_warm_cache_skips_embed_and_is_fast() -> None:
    """가드: warm-cache에서 embed_documents 미호출 + wall-time 상한."""
    embedder = FakeEmbedder(delay=0.05)
    cache = FakeCacheManager()
    chunker = _make_chunker(embedder, cache)
    first = await chunker._get_embeddings(TEXTS)
    assert first.shape == (len(TEXTS), embedder.dimension)
    assert embedder.embed_calls == 1
    t0 = time.perf_counter()
    second = await chunker._get_embeddings(TEXTS)
    warm_ms = (time.perf_counter() - t0) * 1000
    assert embedder.embed_calls == 1, "warm-cache에서 재임베딩 금지"
    assert np.allclose(first, second)
    assert warm_ms < 2000.0, f"warm-cache wall-time 초과: {warm_ms:.1f}ms"


def test_as_valid_vector_guards() -> None:
    """가드: _as_valid_vector 유효성 판별 유지."""
    assert EmbeddingBasedSemanticChunker._as_valid_vector([1.0, 2.0]) is not None
    assert EmbeddingBasedSemanticChunker._as_valid_vector([]) is None
    assert EmbeddingBasedSemanticChunker._as_valid_vector([[1.0]]) is None
    assert EmbeddingBasedSemanticChunker._as_valid_vector("not-a-vector") is None


def test_resolve_expected_dim_prefers_new() -> None:
    """가드: _resolve_expected_dim 신규 임베딩 우선 유지."""
    vec = np.zeros(16, dtype="float32")
    assert (
        EmbeddingBasedSemanticChunker._resolve_expected_dim([None], [(0, "t", vec)])
        == 16
    )
    assert EmbeddingBasedSemanticChunker._resolve_expected_dim([vec], []) == 16
    assert EmbeddingBasedSemanticChunker._resolve_expected_dim([None], []) is None
