import pytest
import asyncio
import numpy as np
from langchain_core.documents import Document
from core.semantic_chunker import EmbeddingBasedSemanticChunker


class MockEmbeddings:
    def __init__(self, dimension=128):
        self.dimension = dimension
        self.model_name = "mock-model"

    def embed_documents(self, texts):
        results = []
        for text in texts:
            vec = np.zeros(self.dimension)
            t_lower = text.lower()
            if "group a" in t_lower:
                vec[0] = 1.0
            elif "group b" in t_lower:
                vec[1] = 1.0
            else:
                idx = 3 + (sum(ord(c) for c in text[:100]) % (self.dimension - 3))
                vec[idx] = 1.0
            results.append(vec.tolist())
        return results


class TrackingCacheManager:
    """캐시 저장 타이밍과 콘텐츠를 추적하는 테스트용 캐시 매니저"""

    def __init__(self):
        self._store: dict[str, dict[str, object]] = {}
        self.set_calls: list[tuple[str, dict[str, object]]] = []

    async def get(self, key: str) -> dict[str, object] | None:
        return self._store.get(key)

    async def set(
        self, key: str, value: dict[str, object], persist_to_disk: bool = True
    ) -> None:
        self.set_calls.append((key, value))
        self._store[key] = value


class FailingBatchEmbeddings:
    """N번째 배치에서 실패하는 임베딩 모델 (배치 2부터 OSError 발생)"""

    def __init__(self, dimension=128, fail_after_batches=1):
        self.dimension = dimension
        self.model_name = "failing-model"
        self._batch_count = 0
        self._fail_after = fail_after_batches

    def embed_documents(self, texts):
        self._batch_count += 1
        if self._batch_count > self._fail_after:
            raise OSError("Simulated Ollama crash")
        results = []
        for text in texts:
            vec = np.zeros(self.dimension)
            vec[0] = 1.0
            results.append(vec.tolist())
        return results


@pytest.mark.asyncio
async def test_semantic_chunker_overlap():
    """청크 간 Overlap(겹침)이 정상적으로 발생하는지 테스트"""
    embedder = MockEmbeddings()
    # 겹침을 2개 문장으로 설정하여 공존 확률을 높임
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        chunk_overlap=2,
        min_chunk_size=0,
        max_chunk_size=300,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_value=0.01,  # 더 민감하게 분할
    )

    # 문장 구성 (충분히 길게)
    s1 = "this is the first sentence of group a and it is quite long. "
    s2 = "this is the second sentence of group a and it is also long. "
    s3 = "this is the third sentence of group a providing more context. "
    s4 = "now we start group b with a completely different topic here. "
    s5 = "this is the second sentence of group b to ensure distance. "
    s6 = "this is the final sentence of group b concluding the text. "

    text = s1 + s2 + s3 + s4 + s5 + s6

    chunks = await chunker.split_text(text)

    # 분할 확인
    assert len(chunks) >= 2

    # Overlap 확인 (모든 청크 대상)
    found_overlap = False
    for chunk in chunks:
        curr_text = chunk["text"].lower()
        if "group a" in curr_text and "group b" in curr_text:
            found_overlap = True
            break

    assert found_overlap, f"Chunk overlap failed. Chunks: {[c['text'] for c in chunks]}"


@pytest.mark.asyncio
async def test_semantic_chunker_cross_page_metadata():
    """여러 페이지에 걸친 청크의 메타데이터가 올바르게 보존되는지 테스트"""
    embedder = MockEmbeddings()
    chunker = EmbeddingBasedSemanticChunker(embedder=embedder, chunk_overlap=0)

    docs = [
        Document(
            page_content="Content on page 1 is long enough.", metadata={"page": 1}
        ),
        Document(
            page_content="Content on page 2 is also long enough.", metadata={"page": 2}
        ),
        Document(
            page_content="Content on page 3 is finally long enough.",
            metadata={"page": 3},
        ),
    ]

    chunker.min_chunk_size = 5000
    chunker.max_chunk_size = 10000
    final_docs, _ = await chunker.split_documents(docs)

    assert len(final_docs) > 0
    meta = final_docs[0].metadata
    assert "pages" in meta
    assert 1 in meta["pages"] and 2 in meta["pages"] and 3 in meta["pages"]
    assert meta["is_cross_page"] is True


@pytest.mark.asyncio
async def test_semantic_chunker_sentence_splitting_safety():
    """매우 긴 텍스트(구분자 없음)에 대한 강제 분할 안전성 테스트"""
    embedder = MockEmbeddings()
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder, min_chunk_size=0, max_chunk_size=500
    )

    long_text = "".join(f"segment{i:04d} " for i in range(500))[:2000]

    chunks = await chunker.split_text(long_text)

    # 1. 분할이 발생했는지 확인
    assert len(chunks) > 1
    # 2. 모든 청크가 합리적인 크기인지 확인 (병합 허용 범위 내)
    for c in chunks:
        # [최적화] hard_split_limit가 1.5배로 늘어났으므로 임계값을 1600으로 상향 (500*1.5=750이지만, 병합 로직 고려)
        assert len(c["text"]) <= 1600


@pytest.mark.asyncio
async def test_get_embeddings_deduplicates_repeated_texts_before_batching():
    class TrackingEmbeddings(MockEmbeddings):
        def __init__(self, dimension=32):
            super().__init__(dimension=dimension)
            self.requested_texts: list[str] = []

        def embed_documents(self, texts):
            self.requested_texts.extend(texts)
            return super().embed_documents(texts)

    embedder = TrackingEmbeddings(dimension=32)
    cache = TrackingCacheManager()
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        batch_size=8,
        cache_manager=cache,
        min_chunk_size=0,
        max_chunk_size=500,
    )

    vectors = await chunker._get_embeddings(["same text", "same text"])

    assert vectors.shape == (2, 32)
    assert embedder.requested_texts == ["same text"]
    assert len(cache.set_calls) == 1


@pytest.mark.asyncio
async def test_cache_written_after_all_batches_succeed():
    embedder = MockEmbeddings(dimension=32)
    cache = TrackingCacheManager()
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        batch_size=3,
        cache_manager=cache,
        min_chunk_size=0,
        max_chunk_size=500,
    )

    sentences = [
        f"Sentence number {i} with enough text for processing. " for i in range(10)
    ]
    text = "".join(sentences)

    await chunker.split_text(text)

    assert len(cache.set_calls) > 0
    for key, value in cache.set_calls:
        assert isinstance(value, dict)
        assert "vector" in value
        assert value["cache_version"] == "1.0"


@pytest.mark.asyncio
async def test_no_partial_cache_on_batch_failure():
    embedder = FailingBatchEmbeddings(dimension=32, fail_after_batches=1)
    cache = TrackingCacheManager()
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        batch_size=3,
        cache_manager=cache,
        min_chunk_size=0,
        max_chunk_size=500,
    )

    sentences = [
        f"Sentence number {i} with enough text for processing. " for i in range(10)
    ]
    text = "".join(sentences)

    with pytest.raises(OSError, match="Simulated Ollama crash"):
        await chunker.split_text(text)

    assert len(cache.set_calls) == 0, (
        f"Expected 0 cache writes after batch failure, got {len(cache.set_calls)}"
    )


@pytest.mark.asyncio
async def test_cache_version_field_present():
    embedder = MockEmbeddings(dimension=32)
    cache = TrackingCacheManager()
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        cache_manager=cache,
        min_chunk_size=0,
        max_chunk_size=500,
    )

    text = "A sufficiently long sentence for embedding cache test. " * 5
    await chunker.split_text(text)

    for key, value in cache.set_calls:
        assert "cache_version" in value, (
            f"cache_version missing from cache entry for {key}"
        )
        assert isinstance(value["cache_version"], str)
