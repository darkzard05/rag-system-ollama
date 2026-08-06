"""F8: 캐시 신뢰 검증 fail-open → fail-closed 회귀 테스트.

- test 1 (RED): security_level=medium에서 신뢰되지 않은 캐시 디렉터리의 load()는
  (None, None, None)을 반환하고 _cache_invalid를 설정해야 함.
- test 2 (RED): 상대 trusted_path는 CWD가 아닌 PROJECT_ROOT 기준으로 해석되어야 함.
- test 3 (lock): medium 레벨에서 불신 경로는 raise가 아닌 False 반환 (호출측 준수 의무).
- test 4 (guard): 신뢰 경로의 정상 save→load 왕복은 여전히 동작해야 함.
"""

from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from cache.vector_cache import VectorStoreCache
from common.config import PROJECT_ROOT
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


def _make_cache(tmp_path: Path) -> VectorStoreCache:
    cache = VectorStoreCache(
        "fake.pdf",
        "test-embedding-model",
        cache_dir=str(tmp_path),
        file_hash="fakehash",
    )
    return cache


def _save_cache(cache: VectorStoreCache) -> None:
    vector_store = FAISS.from_documents(SAMPLE_DOCS, MockEmbeddings())
    bm25_retriever = BM25Retriever.from_documents(
        SAMPLE_DOCS, preprocess_func=bm25_tokenizer
    )
    cache.save(SAMPLE_DOCS, vector_store, bm25_retriever)


def _untrusted_manager(tmp_path: Path) -> CacheSecurityManager:
    return CacheSecurityManager(
        security_level="medium",
        hmac_secret=None,
        trusted_paths=[str(tmp_path.parent / "elsewhere")],
        check_permissions=False,
    )


def test_load_rejects_untrusted_cache_dir(tmp_path):
    """Given: 캐시 디렉터리를 포함하지 않는 신뢰 경로 / When: save 후 load / Then: 실패 폐쇄."""
    cache = _make_cache(tmp_path)
    cache.security_manager = _untrusted_manager(tmp_path)
    _save_cache(cache)

    loaded = cache.load(MockEmbeddings())

    assert loaded == (None, None, None)
    assert cache._cache_invalid is True


def test_trusted_path_relative_resolved_against_project_root():
    """Given: 상대 trusted_path ".model_cache" / Then: PROJECT_ROOT 기준으로 해석."""
    mgr = CacheSecurityManager(
        security_level="medium",
        trusted_paths=[".model_cache"],
        hmac_secret=None,
        check_permissions=False,
    )

    assert mgr.trusted_paths[0] == (PROJECT_ROOT / ".model_cache").resolve()


def test_medium_level_untrusted_returns_false_not_raise(tmp_path):
    """Lock: medium에서 불신 경로는 CacheTrustError 대신 False를 반환."""
    mgr = _untrusted_manager(tmp_path)

    assert mgr.verify_cache_trust(str(tmp_path)) is False


def test_load_accepts_trusted_cache_dir(tmp_path):
    """Guard: 신뢰 경로의 정상 save→load 왕복은 문서를 반환."""
    cache = _make_cache(tmp_path)
    cache.security_manager = CacheSecurityManager(
        security_level="medium",
        hmac_secret=None,
        trusted_paths=[str(tmp_path)],
        check_permissions=False,
    )
    _save_cache(cache)

    loaded_docs, vector_store, bm25_retriever = cache.load(MockEmbeddings())

    assert loaded_docs is not None
    assert vector_store is not None
    assert bm25_retriever is not None
    assert [d.page_content for d in loaded_docs] == [
        d.page_content for d in SAMPLE_DOCS
    ]
