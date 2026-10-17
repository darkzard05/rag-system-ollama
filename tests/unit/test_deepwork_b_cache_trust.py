"""Phase B — cache trust verification (fail-closed on every read path).

Covers:
- B1: vector memory hit blocked after on-disk tamper (fingerprint + trust).
- B1-positive: untouched disk still serves from memory (optimization intact).
- B4: HMAC downgrade rejected when secret is set but .meta lacks HMAC.
- B3: expired annotation memo not served (sync + async paths).
- B2: query cache value validation (version/confidence/response gates).
"""

from __future__ import annotations

import sqlite3
import time

import pytest
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import core.graph._graph_core as graph_core_module
from cache.coord_cache import COORD_CACHE_DB, memo_norm_hash
from cache.vector_cache import VectorStoreCache, clear_vector_object_cache
from common.text_utils import bm25_tokenizer
from core.graph import _generate as gen_module
from core.graph import _retrieval as retrieval_module
from core.graph._graph_core import (
    _is_valid_query_cache_value,
)
from security.cache_security import (
    CacheIntegrityError,
    CacheSecurityManager,
)


class MockEmbeddings(Embeddings):
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
    cache = VectorStoreCache(
        "fake.pdf",
        "test-embedding-model",
        cache_dir=str(tmp_path),
        file_hash="fakehash",
    )
    cache.security_manager = CacheSecurityManager(
        security_level="medium",
        hmac_secret=None,
        trusted_paths=[str(tmp_path)],
        check_permissions=False,
    )
    return cache


def _save(cache: VectorStoreCache) -> None:
    vector_store = FAISS.from_documents(SAMPLE_DOCS, MockEmbeddings())
    bm25 = BM25Retriever.from_documents(SAMPLE_DOCS, preprocess_func=bm25_tokenizer)
    cache.save(SAMPLE_DOCS, vector_store, bm25)


@pytest.fixture(autouse=True)
def _isolate_object_cache():
    clear_vector_object_cache()
    yield
    clear_vector_object_cache()


def test_b1_memory_hit_blocked_after_disk_tamper(tmp_path) -> None:
    cache = _make_cache(tmp_path)
    _save(cache)
    first = cache.load(MockEmbeddings())
    assert all(x is not None for x in first)

    # Attacker swaps doc_splits on disk (memory still holds the old object).
    with open(cache.doc_splits_path, "w", encoding="utf-8") as f:
        f.write('[{"page_content": "PWNED", "metadata": {}}]')

    loaded = cache.load(MockEmbeddings())
    assert loaded == (None, None, None)
    assert cache._cache_invalid is True


def test_b1_memory_hit_intact_without_tamper(tmp_path) -> None:
    cache = _make_cache(tmp_path)
    _save(cache)
    assert all(x is not None for x in cache.load(MockEmbeddings()))
    second = cache.load(MockEmbeddings())
    assert [d.page_content for d in second[0]] == [d.page_content for d in SAMPLE_DOCS]


def test_b4_hmac_missing_rejected_when_secret_set(tmp_path) -> None:
    mgr = CacheSecurityManager(
        security_level="medium",
        hmac_secret="test-secret-test-secret-test-secret-32",
        trusted_paths=[str(tmp_path)],
        check_permissions=False,
    )
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"payload")
    meta = mgr.create_metadata_for_file(str(target), description="t")
    assert meta.integrity_hmac  # sanity: secret set => HMAC present
    stripped = meta.model_copy(update={"integrity_hmac": None})
    mgr.save_cache_metadata(str(target) + ".meta", stripped)
    with pytest.raises(CacheIntegrityError):
        mgr.verify_cache_integrity(str(target))


def test_b4_no_secret_keeps_hash_only_path(tmp_path) -> None:
    mgr = CacheSecurityManager(
        security_level="medium",
        hmac_secret=None,
        trusted_paths=[str(tmp_path)],
        check_permissions=False,
    )
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"payload")
    meta = mgr.create_metadata_for_file(str(target), description="t")
    mgr.save_cache_metadata(str(target) + ".meta", meta)
    assert mgr.verify_cache_integrity(str(target)) is True


def _stale_memo_row() -> None:
    from cache.coord_cache import coord_cache

    coord_cache.save_annotation_memo_sync(
        "dw-b3-fh", 7, memo_norm_hash("hello deepwork"), [{"x": 1}]
    )
    stale = time.time() - 8 * 86400
    with sqlite3.connect(COORD_CACHE_DB) as conn:
        conn.execute(
            "UPDATE annotation_memo SET created_at = ? "
            "WHERE file_hash = ? AND page_num = ?",
            (stale, "dw-b3-fh", 7),
        )
        conn.commit()


def test_b3_expired_memo_not_served_sync() -> None:
    from cache.coord_cache import coord_cache

    _stale_memo_row()
    assert (
        coord_cache.get_annotation_memo_sync(
            "dw-b3-fh", 7, memo_norm_hash("hello deepwork")
        )
        is None
    )


@pytest.mark.asyncio
async def test_b3_expired_memo_not_served_async() -> None:
    from cache.coord_cache import coord_cache

    _stale_memo_row()
    assert (
        await coord_cache.get_annotation_memo(
            "dw-b3-fh", 7, memo_norm_hash("hello deepwork")
        )
        is None
    )


def test_b3_fresh_memo_still_served() -> None:
    from cache.coord_cache import coord_cache

    coord_cache.save_annotation_memo_sync(
        "dw-b3-fh2", 8, memo_norm_hash("world deepwork"), [{"x": 2}]
    )
    assert coord_cache.get_annotation_memo_sync(
        "dw-b3-fh2", 8, memo_norm_hash("world deepwork")
    ) == [{"x": 2}]


def test_b2_version_constant_shared() -> None:
    assert gen_module._QUERY_CACHE_VALUE_VERSION == "1.0"
    assert retrieval_module._is_valid_query_cache_value is not None
    assert graph_core_module._QUERY_CACHE_VALUE_VERSION == "1.0"


def test_b2_hit_predicate() -> None:
    good = {"response": "ans", "confidence": 0.9, "version": "1.0"}
    assert _is_valid_query_cache_value(good, 0.85) is True
    assert (
        _is_valid_query_cache_value({**good, "version": "0.9"}, 0.85) is False
    )  # stale schema
    assert _is_valid_query_cache_value({**good} | {"version": None}, 0.85) is False
    assert (
        _is_valid_query_cache_value({**good, "confidence": 0.5}, 0.85) is False
    )  # low conf
    assert (
        _is_valid_query_cache_value({**good, "response": "  "}, 0.85) is False
    )  # blank
    assert _is_valid_query_cache_value(None, 0.85) is False
    assert _is_valid_query_cache_value({"response": "x"}, 0.85) is False
