"""임베딩 캐시 오염 시 벡터 정렬 무결성 보장 테스트 (Bug A).

Bug A: ``_get_embeddings``가 차원 불일치/None 캐시 벡터를 제거(drop)하여
임베딩 행렬의 행 수가 입력 텍스트 수보다 작아지는 결함을 검증합니다.
축소된 행렬을 ``split_text``의 ``zip(sentences, embeddings, strict=False)``가
소비하면 일부 문장에 "vector" 키가 빠져 잘못된 청크 벡터/KeyError가 발생합니다.
"""

import numpy as np
import pytest
import xxhash
from langchain_core.embeddings import FakeEmbeddings

from core.semantic_chunker import EmbeddingBasedSemanticChunker

SENTENCES = [
    "The quick brown fox jumps over the lazy dog at dawn.",
    "Semantic chunking splits long documents into meaningful units.",
    "Embedding vectors capture the semantic meaning of each sentence.",
]


class PoisonedCacheManager:
    """오염된 임베딩을 사전 주입할 수 있는 테스트용 캐시 매니저."""

    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return self._store.get(key)

    async def set(
        self,
        key: str,
        value: object,
        ttl_seconds: float = 0,
        persist_to_disk: bool = True,
    ) -> None:
        self._store[key] = value


def _emb_key(model_name: str, text: str) -> str:
    norm_text = " ".join(text.split())
    return f"emb:{model_name}:{xxhash.xxh64(norm_text.encode()).hexdigest()}"


def _make_chunker(
    cache: PoisonedCacheManager, buffer_size: int = 10, dimension: int = 32
) -> EmbeddingBasedSemanticChunker:
    embedder = FakeEmbeddings(size=dimension)
    return EmbeddingBasedSemanticChunker(
        embedder=embedder,
        buffer_size=buffer_size,
        batch_size=64,
        cache_manager=cache,
        min_chunk_size=0,
        max_chunk_size=500,
    )


@pytest.mark.asyncio
async def test_get_embeddings_returns_row_per_input_with_poisoned_cache():
    """오염된(차원 불일치) 캐시 벡터가 있어도 행 수는 입력 텍스트 수와 동일해야 합니다."""
    cache = PoisonedCacheManager()
    chunker = _make_chunker(cache)
    await cache.set(
        _emb_key(chunker.model_name, SENTENCES[1]),
        {"vector": [0.25, 0.5, 0.75, 1.0], "cache_version": "1.0"},
    )

    matrix = await chunker._get_embeddings(SENTENCES)

    assert matrix.shape[0] == len(SENTENCES), (
        f"_get_embeddings must return exactly {len(SENTENCES)} rows (one per input text), "
        f"got {matrix.shape[0]} — cached dim-mismatched vectors were DROPPED instead of re-embedded."
    )
    assert matrix.shape[1] == 32


@pytest.mark.asyncio
async def test_get_embeddings_repairs_none_cached_vector():
    """None 벡터 캐시 항목은 재임베딩으로 복구되어 행 수가 유지되어야 합니다."""
    cache = PoisonedCacheManager()
    chunker = _make_chunker(cache)
    two_sentences = SENTENCES[:2]
    await cache.set(
        _emb_key(chunker.model_name, two_sentences[1]),
        {"vector": None, "cache_version": "1.0"},
    )

    matrix = await chunker._get_embeddings(two_sentences)

    assert matrix.shape[0] == len(two_sentences), (
        f"_get_embeddings must return exactly {len(two_sentences)} rows, "
        f"got {matrix.shape[0]} — None cached vector was not repaired."
    )


@pytest.mark.asyncio
async def test_split_text_attaches_vector_to_every_sentence_with_poisoned_cache():
    """오염된 캐시에서도 모든 문장에 "vector"가 부착되어야 합니다 (누락 키 금지)."""
    cache = PoisonedCacheManager()
    chunker = _make_chunker(cache, buffer_size=10)
    text = " ".join(SENTENCES)
    await cache.set(
        _emb_key(chunker.model_name, SENTENCES[1]),
        {"vector": [0.25, 0.5, 0.75, 1.0], "cache_version": "1.0"},
    )

    result = await chunker.split_text(text)

    assert len(result) == len(SENTENCES)
    for i, sentence in enumerate(result):
        assert "vector" in sentence, (
            f"sentence {i} is missing the 'vector' key — "
            "zip(sentences, embeddings, strict=False) left it unset because "
            "_get_embeddings returned fewer rows than sentences."
        )
        assert np.asarray(sentence["vector"]).shape == (32,), (
            f"sentence {i} has wrong vector dim: {np.asarray(sentence['vector']).shape}"
        )
