"""R3a-06: 벡터 캐시 저장 시 pickle(index.pkl) 아티팩트 미생성 검증 (TDD).

langchain의 `FAISS.save_local`은 `index.faiss`와 함께 pickle `index.pkl`
(docstore + index_to_docstore_id 2-튜플)을 디스크에 작성하지만, 프로젝트 로드
경로(vector_cache.py load)는 index.pkl을 읽지 않고 index.faiss + JSON으로
재구성한다. 즉 index.pkl은 데드 아티팩트 + 잠재 보안 표면이다.

본 테스트는 저장 포맷을 `faiss.write_index` + JSON docstore 직렬화로 전환한
후에는 저장된 캐시 디렉터리에 `index.pkl`이 존재하지 않아야 함을 단언한다.
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
    """128차원 고정 벡터를 반환하는 오프라인 임베더 (모델 불필요)."""

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
        trusted_paths=[str(tmp_path)],
        check_permissions=False,
    )
    return cache


def test_save_creates_no_pickle_artifact(tmp_path):
    """Given: save()로 소형 FAISS 스토어 저장 / When: 캐시 디렉터리 검사 / Then: index.pkl 없음."""
    cache = _make_cache(tmp_path)
    vector_store = FAISS.from_documents(SAMPLE_DOCS, MockEmbeddings())
    bm25_retriever = BM25Retriever.from_documents(
        SAMPLE_DOCS, preprocess_func=bm25_tokenizer
    )

    cache.save(SAMPLE_DOCS, vector_store, bm25_retriever)

    assert os.path.isdir(cache.faiss_index_path)
    artifacts = os.listdir(cache.faiss_index_path)

    # 저장 포맷 전환 전(RED): save_local이 index.pkl을 생성하므로 아래 단언이 실패.
    assert "index.pkl" not in artifacts
    # 바이너리 인덱스는 write_index로 저장되어야 함.
    assert "index.faiss" in artifacts
    # docstore 매핑은 pickle이 아닌 JSON으로 직렬화되어야 함 (pickle-free 정책).
    assert "index_to_docstore_id.json" in artifacts

    # 캐시 디렉터리 전체에서 .pkl 아티팩트가 하나도 없어야 함.
    for root, _, files in os.walk(cache.cache_dir):
        for fname in files:
            assert not fname.endswith(".pkl"), (
                f"pickle 아티팩트 발견: {os.path.join(root, fname)}"
            )


def test_save_then_load_roundtrip_without_pickle(tmp_path):
    """Given: pickle-free 포맷으로 저장 / When: load() / Then: docstore 매핑이 그대로 복원."""
    cache = _make_cache(tmp_path)
    vector_store = FAISS.from_documents(SAMPLE_DOCS, MockEmbeddings())
    bm25_retriever = BM25Retriever.from_documents(
        SAMPLE_DOCS, preprocess_func=bm25_tokenizer
    )
    cache.save(SAMPLE_DOCS, vector_store, bm25_retriever)

    loaded_docs, loaded_store, loaded_bm25 = cache.load(MockEmbeddings())

    assert loaded_docs is not None
    assert loaded_store is not None
    assert loaded_bm25 is not None
    assert [d.page_content for d in loaded_docs] == [
        d.page_content for d in SAMPLE_DOCS
    ]
    results = loaded_store.similarity_search("caching", k=1)
    assert len(results) == 1
    assert results[0].page_content in {d.page_content for d in SAMPLE_DOCS}
