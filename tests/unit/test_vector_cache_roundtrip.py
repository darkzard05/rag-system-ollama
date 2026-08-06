"""VectorStoreCache 저장→로드 왕복(round-trip) 회귀 테스트.

저장된 캐시가 첫 로드 시 스스로 파괴(self-destruct)되는 버그를 검증합니다.
- 테스트 1: 실제 save() → load() 왕복이 정상 동작해야 함.
  (현재 RED: faiss_index/index.faiss 등에 .meta가 없어 CacheIntegrityError →
  _purge_cache(rmtree) → (None, None, None) 반환)
- 테스트 2: 로드 실패 시 캐시 디렉터리를 삭제하지 않아야 함.
  (현재 RED: _purge_cache가 rmtree로 캐시 디렉터리를 제거)
"""

import os

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from cache.vector_cache import VectorStoreCache
from common.text_utils import bm25_tokenizer
from security.cache_security import CacheSecurityManager


class MockEmbeddings(Embeddings):
    """128차원 고정 벡터를 반환하는 오프라인 임베더."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 128] * len(texts)

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 128


SAMPLE_DOCS = [
    Document(
        page_content="GraphRAG combines knowledge graphs with retrieval augmented generation",
        metadata={"page": 1},
    ),
    Document(
        page_content="Vector store caching prevents redundant indexing",
        metadata={"page": 2},
    ),
]


def _make_cache(tmp_path: object) -> VectorStoreCache:
    """실제 보안 검증(meta 생성/해시 검증)을 수행하는 캐시 인스턴스를 구성합니다."""
    cache = VectorStoreCache(
        "fake.pdf",
        "test-embedding-model",
        cache_dir=str(tmp_path),
        file_hash="fakehash",
    )
    cache.security_manager = CacheSecurityManager(
        security_level="medium",
        hmac_secret=None,
        trusted_paths=[],
        check_permissions=False,
    )
    return cache


def _save_cache(cache: VectorStoreCache) -> None:
    """작은 FAISS 스토어 + BM25 리트리버를 만들어 실제 save()를 호출합니다."""
    vector_store = FAISS.from_documents(SAMPLE_DOCS, MockEmbeddings())
    bm25_retriever = BM25Retriever.from_documents(
        SAMPLE_DOCS, preprocess_func=bm25_tokenizer
    )
    cache.save(SAMPLE_DOCS, vector_store, bm25_retriever)


def test_save_then_load_roundtrip(tmp_path):
    """Given: save()로 저장된 캐시 / When: 동일 인스턴스로 load() / Then: 전체 복원."""
    cache = _make_cache(tmp_path)
    _save_cache(cache)

    loaded_docs, vector_store, bm25_retriever = cache.load(MockEmbeddings())

    assert loaded_docs is not None
    assert vector_store is not None
    assert bm25_retriever is not None
    assert [d.page_content for d in loaded_docs] == [
        d.page_content for d in SAMPLE_DOCS
    ]
    assert [d.metadata for d in loaded_docs] == [d.metadata for d in SAMPLE_DOCS]
    results = vector_store.similarity_search("caching", k=1)
    assert len(results) == 1
    assert results[0].page_content in {d.page_content for d in SAMPLE_DOCS}


def test_load_does_not_purge_cache_dir_on_integrity_failure(tmp_path):
    """Given: 저장된 캐시의 doc_splits.json이 변조됨 / When: load() 실패 / Then: 디렉터리 유지."""
    cache = _make_cache(tmp_path)
    _save_cache(cache)
    cache_dir = cache.cache_dir
    assert os.path.isdir(cache_dir)

    with open(cache.doc_splits_path, "wb") as f:
        f.write(b"corrupted")

    loaded = cache.load(MockEmbeddings())

    assert loaded == (None, None, None)
    assert os.path.isdir(cache_dir)


def test_cache_key_includes_config_hash(tmp_path, monkeypatch):
    """Given: 청킹 설정이 다른 두 캐시 인스턴스 / When: 동일 파일+모델로 생성 / Then: cache_dir이 달라야 함."""
    import cache.vector_cache as vc_module

    base_splitter = {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "separators": ["\\n\\n"],
    }
    monkeypatch.setattr(vc_module, "TEXT_SPLITTER_CONFIG", dict(base_splitter))
    monkeypatch.setattr(vc_module, "SEMANTIC_CHUNKER_CONFIG", {"enabled": True})

    cache_a = VectorStoreCache(
        "fake.pdf", "model-x", cache_dir=str(tmp_path), file_hash="hash-a"
    )

    monkeypatch.setattr(
        vc_module, "TEXT_SPLITTER_CONFIG", {**base_splitter, "chunk_size": 1200}
    )

    cache_b = VectorStoreCache(
        "fake.pdf", "model-x", cache_dir=str(tmp_path), file_hash="hash-a"
    )

    assert cache_a.cache_dir != cache_b.cache_dir
