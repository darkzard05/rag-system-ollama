"""R3a-03: 벡터 캐시 서명에 인덱스 전략(index_params)·정규화 배선 포함 검증.

`_build_config_signature`가 index_params(use_l2_norm/hnsw_m/quantization_threshold/
nprobe)와 semantic chunker 정규화 배선을 서명에 포함하지 않아, 설정 변경 시
스테일 인덱스 캐시가 히트되는 결함을 회귀 방지합니다.
"""

import json
import os

import cache.vector_cache as vc_module
from cache.vector_cache import VectorStoreCache

_BASE_SPLITTER = {
    "chunk_size": 500,
    "chunk_overlap": 100,
    "separators": ["\n\n"],
}
_DEFAULT_INDEX_PARAMS = {
    "use_l2_norm": True,
    "hnsw_m": 32,
    "quantization_threshold": 5000,
    "nprobe": 16,
}


def _make_cache(tmp_path: object, file_hash: str = "hash-a") -> VectorStoreCache:
    return VectorStoreCache(
        "fake.pdf", "model-x", cache_dir=str(tmp_path), file_hash=file_hash
    )


def _patch_base(monkeypatch) -> None:
    monkeypatch.setattr(vc_module, "TEXT_SPLITTER_CONFIG", dict(_BASE_SPLITTER))
    monkeypatch.setattr(vc_module, "SEMANTIC_CHUNKER_CONFIG", {"enabled": True})


def _patch_index_params(monkeypatch, index_params: dict) -> None:
    monkeypatch.setattr(
        vc_module,
        "VECTOR_STORE_CONFIG",
        {"index_params": index_params},
        raising=False,
    )


def test_signature_changes_when_use_l2_norm_changes(tmp_path, monkeypatch):
    """Given: use_l2_norm 토글 / When: 캐시 인스턴스 생성 / Then: cache_dir이 달라야 함."""
    _patch_base(monkeypatch)
    _patch_index_params(monkeypatch, {**_DEFAULT_INDEX_PARAMS, "use_l2_norm": True})
    cache_a = _make_cache(tmp_path)

    _patch_index_params(monkeypatch, {**_DEFAULT_INDEX_PARAMS, "use_l2_norm": False})
    cache_b = _make_cache(tmp_path)

    assert cache_a.cache_dir != cache_b.cache_dir


def test_signature_changes_when_hnsw_m_changes(tmp_path, monkeypatch):
    """Given: hnsw_m 변경 / When: 캐시 인스턴스 생성 / Then: cache_dir이 달라야 함."""
    _patch_base(monkeypatch)
    _patch_index_params(monkeypatch, {**_DEFAULT_INDEX_PARAMS, "hnsw_m": 32})
    cache_a = _make_cache(tmp_path)

    _patch_index_params(monkeypatch, {**_DEFAULT_INDEX_PARAMS, "hnsw_m": 64})
    cache_b = _make_cache(tmp_path)

    assert cache_a.cache_dir != cache_b.cache_dir


def test_signature_changes_when_quantization_threshold_changes(tmp_path, monkeypatch):
    """Given: quantization_threshold 변경 / When: 캐시 인스턴스 생성 / Then: cache_dir이 달라야 함."""
    _patch_base(monkeypatch)
    _patch_index_params(
        monkeypatch, {**_DEFAULT_INDEX_PARAMS, "quantization_threshold": 5000}
    )
    cache_a = _make_cache(tmp_path)

    _patch_index_params(
        monkeypatch, {**_DEFAULT_INDEX_PARAMS, "quantization_threshold": 10000}
    )
    cache_b = _make_cache(tmp_path)

    assert cache_a.cache_dir != cache_b.cache_dir


def test_signature_changes_when_nprobe_changes(tmp_path, monkeypatch):
    """Given: nprobe 변경(T6 추가 키) / When: 캐시 인스턴스 생성 / Then: cache_dir이 달라야 함."""
    _patch_base(monkeypatch)
    _patch_index_params(monkeypatch, {**_DEFAULT_INDEX_PARAMS, "nprobe": 16})
    cache_a = _make_cache(tmp_path)

    _patch_index_params(monkeypatch, {**_DEFAULT_INDEX_PARAMS, "nprobe": 32})
    cache_b = _make_cache(tmp_path)

    assert cache_a.cache_dir != cache_b.cache_dir


def test_signature_changes_when_chunker_normalize_wiring_changes(tmp_path, monkeypatch):
    """Given: semantic chunker 정규화 배선(T8) 변경 / When: 캐시 인스턴스 생성 / Then: 서명이 달라야 함."""
    _patch_base(monkeypatch)
    _patch_index_params(monkeypatch, dict(_DEFAULT_INDEX_PARAMS))
    monkeypatch.setattr(vc_module, "SEMANTIC_CHUNKER_NORMALIZE", True, raising=False)
    cache_a = _make_cache(tmp_path)

    monkeypatch.setattr(vc_module, "SEMANTIC_CHUNKER_NORMALIZE", False, raising=False)
    cache_b = _make_cache(tmp_path)

    assert cache_a.cache_dir != cache_b.cache_dir


def test_signature_stable_for_equivalent_index_params(tmp_path, monkeypatch):
    """Given: index_params 미설정(빈 dict)과 명시 기본값 / When: 캐시 생성 / Then: 동일 서명 (타입 안정)."""
    _patch_base(monkeypatch)
    monkeypatch.setattr(vc_module, "VECTOR_STORE_CONFIG", {}, raising=False)
    cache_a = _make_cache(tmp_path)

    monkeypatch.setattr(
        vc_module,
        "VECTOR_STORE_CONFIG",
        {"index_params": dict(_DEFAULT_INDEX_PARAMS)},
        raising=False,
    )
    cache_b = _make_cache(tmp_path)

    assert cache_a.cache_dir == cache_b.cache_dir


def test_load_rejects_older_schema_version(tmp_path):
    """Given: schema 2로 저장된 캐시 / When: load() / Then: 재구축 유도 (캐시 디렉터리 삭제 없음)."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    from common.text_utils import bm25_tokenizer
    from security.cache_security import CacheSecurityManager

    docs = [Document(page_content="hello cache schema eviction", metadata={"page": 1})]

    class _MockEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 128] * len(texts)

        def embed_query(self, text: str) -> list[float]:
            return [0.1] * 128

    cache = VectorStoreCache(
        "fake.pdf", "test-model", cache_dir=str(tmp_path), file_hash="fakehash"
    )
    cache.security_manager = CacheSecurityManager(
        security_level="medium",
        hmac_secret=None,
        trusted_paths=[],
        check_permissions=False,
    )
    vector_store = FAISS.from_documents(docs, _MockEmbeddings())
    bm25 = BM25Retriever.from_documents(docs, preprocess_func=bm25_tokenizer)
    cache.save(docs, vector_store, bm25)

    # 저장된 VERSION.json의 schema_version을 구버전으로 다운그레이드하고,
    # 무결성 검증을 우회하기 위해 faiss_index 내 .meta를 제거한다 (스키마 검증만 격리).
    version_path = os.path.join(cache.faiss_index_path, "VERSION.json")
    with open(version_path) as f:
        manifest = json.load(f)
    manifest["schema_version"] = 2
    with open(version_path, "w") as f:
        json.dump(manifest, f)
    for artifact in os.listdir(cache.faiss_index_path):
        meta_path = os.path.join(cache.faiss_index_path, artifact) + ".meta"
        if os.path.exists(meta_path):
            os.remove(meta_path)

    loaded = cache.load(_MockEmbeddings())

    assert loaded == (None, None, None)
    assert cache._cache_invalid is True
    assert os.path.isdir(cache.cache_dir)
