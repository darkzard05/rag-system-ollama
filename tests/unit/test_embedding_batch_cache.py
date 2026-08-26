"""
Phase 1 성능 리팩터 테스트.

배치 임베딩 + 영구 캐시 히트 경로를 검증한다.
- Ollama 임베더는 누락분 전체를 단일 ``embed_documents`` 호출로 묶어야 한다 (HTTP 왕복 1회).
- 동일 텍스트(콘텐츠 해시 키) 재임베딩 시 영구 디스크 캐시에서 히트하여 Ollama 왕복을 건너뛴다.
임베더는 목으로 대체하므로 실제 Ollama 호출은 발생하지 않는다.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from langchain_ollama import OllamaEmbeddings

from core.semantic_chunker import EmbeddingBasedSemanticChunker
from services.optimization.caching_optimizer import CacheManager


class OllamaLikeEmbeddings(OllamaEmbeddings):
    """langchain_ollama.OllamaEmbeddings 서브클래스 목 (실제 Ollama 호출 없음).

    ``embed_documents`` 는 전달된 전체 텍스트를 단일 HTTP 요청으로 보내는
    Ollama 동작을 모사하므로, 배치 사이즈와 무관하게 한 번의 호출로 묶인다.
    ``OllamaEmbeddings`` 를 상속하므로 ``_is_ollama_embedder`` 판별이 True 가 되어
    청커가 단일 배치 경로를 탄다.
    """

    def __init__(self, dimension: int = 32, model: str = "mock-embed") -> None:
        super().__init__(model=model)
        # OllamaEmbeddings 는 pydantic 모델이므로 비필드 속성은 object.__setattr__ 로 설정.
        # call_count/last_batch_size 는 += 재할당을 피하기 위해 가변 컨테이너에 보관.
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_calls", [0])
        object.__setattr__(self, "_last_batch", [0])

    @property
    def call_count(self) -> int:
        return self._calls[0]

    @property
    def last_batch_size(self) -> int:
        return self._last_batch[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._calls[0] += 1
        self._last_batch[0] = len(texts)
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype="float32")
        idx = sum(ord(c) for c in text[:50]) % self.dimension
        vec[idx] = 1.0
        return vec.tolist()


def _make_persistent_cache(tmp_dir: str) -> CacheManager:
    """청킹 임베딩 전용 영구 캐시 (L1 메모리 + L3 디스크)."""
    return CacheManager(
        enable_memory_cache=True,
        enable_semantic_cache=False,
        enable_disk_cache=True,
        disk_cache_dir=str(Path(tmp_dir) / "embedding_cache"),
    )


@pytest.mark.asyncio
async def test_ollama_embedder_coalesces_into_single_call():
    """Ollama 임베더는 누락 텍스트 전체를 한 번의 embed_documents 호출로 묶는다."""
    embedder = OllamaLikeEmbeddings(dimension=32)
    cache = _make_persistent_cache(tempfile.mkdtemp())

    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        batch_size=4,  # 작은 배치 사이즈여도 Ollama는 한 번에 묶어야 함
        cache_manager=cache,
        min_chunk_size=0,
        max_chunk_size=500,
    )

    texts = [
        f"Distinct sentence number {i} for embedding batch test." for i in range(10)
    ]
    vectors = await chunker._get_embeddings(texts)

    # 1. 반환 행렬 크기/차원 보존
    assert vectors.shape == (10, 32)

    # 2. Ollama는 한 번의 HTTP 요청(단일 embed_documents 호출)로 전체를 처리
    assert embedder.call_count == 1
    assert embedder.last_batch_size == 10


@pytest.mark.asyncio
async def test_identical_text_reuses_cache_without_ollama_roundtrip():
    """동일 텍스트 재임베딩 시 영구 캐시에서 히트하여 Ollama 왕복을 건너뛴다."""
    tmp_dir = tempfile.mkdtemp()
    embedder = OllamaLikeEmbeddings(dimension=32)
    cache = _make_persistent_cache(tmp_dir)

    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        batch_size=8,
        cache_manager=cache,
        min_chunk_size=0,
        max_chunk_size=500,
    )

    texts = ["same content to be embedded and cached."]
    first = await chunker._get_embeddings(texts)
    assert embedder.call_count == 1

    # L1 메모리 캐시만 비워도 L3 디스크 캐시에서 복구되는지 검증
    # (전체 clear()는 디스크까지 지우므로 영구 캐시 보존을 테스트할 수 없음)
    assert cache.memory_cache is not None
    await cache.memory_cache.clear()

    second = await chunker._get_embeddings(texts)

    # 1. 디스크 캐시 히트로 임베더 호출이 추가로 발생하지 않음
    assert embedder.call_count == 1
    assert np.allclose(first, second)


@pytest.mark.asyncio
async def test_split_documents_hits_persistent_cache_across_instances():
    """재수집 시 신규 청커 인스턴스도 영구 캐시에서 히트한다 (프로세스 재시작 모사)."""
    tmp_dir = tempfile.mkdtemp()
    text = " ".join(
        f"Sentence {i} about a stable topic that should be cached." for i in range(12)
    )

    # 첫 수집: 실제 임베딩 생성 + 디스크 캐시 기록
    first_embedder = OllamaLikeEmbeddings(dimension=32)
    first = EmbeddingBasedSemanticChunker(
        embedder=first_embedder,
        cache_manager=_make_persistent_cache(tmp_dir),
        min_chunk_size=0,
        max_chunk_size=500,
    )
    chunks1 = await first.split_text(text)
    assert first_embedder.call_count == 1

    # 두 번째 수집: 새 임베더/새 청커지만 동일 디스크 캐시 공유
    second_embedder = OllamaLikeEmbeddings(dimension=32)
    second = EmbeddingBasedSemanticChunker(
        embedder=second_embedder,
        cache_manager=_make_persistent_cache(tmp_dir),
        min_chunk_size=0,
        max_chunk_size=500,
    )
    chunks2 = await second.split_text(text)

    # 임베더를 한 번도 호출하지 않고 분할 결과가 동일해야 함
    assert second_embedder.call_count == 0
    assert len(chunks1) == len(chunks2)
    assert [c["text"] for c in chunks1] == [c["text"] for c in chunks2]
