import os
import shutil
import tempfile
import unittest.mock as mock

import faiss
import numpy as np
import pytest
from langchain_core.documents import Document

from cache.vector_cache import VectorStoreCache, _serialize_docs


class MockEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 128] * len(texts)

    def embed_query(self, text):
        return [0.1] * 128

    def __call__(self, text):  # Callable 지원 추가
        return self.embed_query(text)


@pytest.fixture
def temp_cache_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


def test_faiss_reconstruction_without_pickle(temp_cache_dir):
    """Pickle 파일 없이 FAISS 인덱스와 JSON 메타데이터만으로 복원이 가능한지 테스트"""

    # 1. 가짜 데이터 준비
    docs = [Document(page_content="test content", metadata={"page": 1})]
    file_hash = "fakehash"
    model_name = "test-model"

    # 캐시 객체 생성 (실제 경로 확인 위함)
    cache = VectorStoreCache(
        "fake.pdf", model_name, cache_dir=temp_cache_dir, file_hash=file_hash
    )

    # 2. 수동으로 캐시 파일 생성 (구조에 맞게!)
    os.makedirs(cache.cache_dir, exist_ok=True)
    os.makedirs(cache.faiss_index_path, exist_ok=True)

    # FAISS 인덱스 바이너리만 저장
    d = 128
    index = faiss.IndexFlatL2(d)
    index.add(np.array([[0.1] * d], dtype="float32"))
    faiss.write_index(index, os.path.join(cache.faiss_index_path, "index.faiss"))

    # 문서 조각 및 BM25 데이터(JSON) 저장
    import orjson

    with open(cache.doc_splits_path, "wb") as f:
        f.write(orjson.dumps(_serialize_docs(docs)))
    with open(cache.bm25_retriever_path, "wb") as f:
        f.write(orjson.dumps(_serialize_docs(docs)))

    # 3. 로드 시도
    embedder = MockEmbeddings()
    with (
        mock.patch.object(cache.security_manager, "verify_cache_trust"),
        mock.patch.object(cache.security_manager, "verify_cache_integrity"),
    ):
        loaded_docs, vector_store, bm25 = cache.load(embedder)

        # 4. 검증
        assert loaded_docs is not None
        assert len(loaded_docs) == 1
        assert vector_store is not None
        # FAISS 내부적으로 문서가 복원되었는지 확인
        res = vector_store.similarity_search("hello", k=1)
        assert len(res) == 1
        assert res[0].page_content == "test content"
        print("✅ Pickle 파일 없이도 FAISS 및 문서 복원 성공!")
