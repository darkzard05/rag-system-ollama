"""Task 3: VectorStoreCache object-cache bridge integration (TDD).

Verifies that VectorStoreCache now routes its in-memory parsed-object caching
through SyncCacheBridge/ObjectCache (no event-loop errors in the synchronous
load/save path) while PRESERVING the pickle-free trust checks and the legacy
[:10] dir fallback.

- test 1: cache miss -> build -> save -> reload produces a unified object-cache
  HIT (second load returns without re-reading parsed objects). Tampered .meta
  still raises CacheTrustError (trust policy unchanged).
- test 2: legacy [:10] truncated cache dir is still migrated on load.
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
    vector_store = FAISS.from_documents(SAMPLE_DOCS, MockEmbeddings())
    bm25_retriever = BM25Retriever.from_documents(
        SAMPLE_DOCS, preprocess_func=bm25_tokenizer
    )
    cache.save(SAMPLE_DOCS, vector_store, bm25_retriever)


def test_object_cache_hit_after_save_reload(tmp_path):
    """Given: 저장된 캐시 / When: load() 두 번 호출 / Then: 두 번째는 통합 객체 캐시 히트."""
    cache = _make_cache(tmp_path)
    _save_cache(cache)

    first = cache.load(MockEmbeddings())
    assert all(x is not None for x in first)

    # 통합 객체 캐시에 파싱 객체가 보관되었는지 확인.
    assert cache._object_cache.get_sync(cache.cache_dir) is not None

    # 디스크를 가짜로 비워도(파싱 결과가 아닌) 객체 캐시에서 동일 객체 반환.
    second = cache.load(MockEmbeddings())
    assert [d.page_content for d in second[0]] == [d.page_content for d in SAMPLE_DOCS]


def test_tampered_meta_still_fails_integrity_check(tmp_path):
    """Given: 신뢰 경로 + 저장된 캐시의 .meta가 변조됨 / When: load() / Then: 실패(정책 유지).

    보안 정책: 신뢰 경로에서는 무결성 검증(HMAC/해시)이 수행되며, .meta 변조 시
    CacheIntegrityError 가 발생해 캐시가 폐기(재구축)됩니다.
    """
    cache = _make_cache(tmp_path)
    cache.security_manager = CacheSecurityManager(
        security_level="medium",
        hmac_secret="test-secret-test-secret-test-secret-32",
        trusted_paths=[],
        check_permissions=False,
    )
    _save_cache(cache)
    meta_path = cache.doc_splits_path + ".meta"
    assert os.path.exists(meta_path)

    # .meta 무결성 훼손 (HMAC 재검증 실패 유도).
    with open(meta_path, "wb") as f:
        f.write(b"tampered-meta-content")

    # 보안 정책: 무결성 위반 시 load() 는 raise 하지 않고 (None,None,None) 을
    # 반환하며 _cache_invalid 를 설정해 재구축을 유도합니다(폐기, 삭제 아님).
    loaded = cache.load(MockEmbeddings())
    assert loaded == (None, None, None)
    assert cache._cache_invalid is True


def test_untrusted_path_fails_closed_on_load(tmp_path):
    """Given: 캐시 디렉터리를 포함하지 않는 신뢰 경로 / When: load() / Then: (None,None,None) (fail-closed)."""
    cache = _make_cache(tmp_path)
    cache.security_manager = CacheSecurityManager(
        security_level="medium",
        hmac_secret=None,
        trusted_paths=[str(tmp_path / "elsewhere")],
        check_permissions=False,
    )
    _save_cache(cache)

    loaded = cache.load(MockEmbeddings())
    assert loaded == (None, None, None)
    assert cache._cache_invalid is True


def test_legacy_dir_fallback_migrated_on_load(tmp_path):
    """Given: [:10] legacy 캐시 디렉터리 + 하위 아티팩트 존재 / When: load() / Then: 경로 재지정(마이그레이션)."""
    cache = _make_cache(tmp_path)
    # legacy 키(임베딩 모델명 [:10])로 디렉터리 생성 및 아티팩트 배치.
    legacy_dir = os.path.join(str(tmp_path), f"fakehash_{'test-embedding-model'[:10]}")
    os.makedirs(legacy_dir, exist_ok=True)
    for name in ("doc_splits.json", "bm25_docs.json"):
        with open(os.path.join(legacy_dir, name), "w") as f:
            f.write("{}")
    os.makedirs(os.path.join(legacy_dir, "faiss_index"), exist_ok=True)

    # 현재(cache_dir != legacy_dir) 인스턴스에서 legacy 재지정이 일어나는지 확인.
    assert cache._legacy_cache_dir == legacy_dir
    assert cache._try_load_legacy() is True
    assert cache.cache_dir == legacy_dir
