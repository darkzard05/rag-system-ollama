# create_vector_store 호출 시 중복 임베딩 수행 여부를 검증하는 테스트
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document
from src.core.retriever_factory import create_vector_store


@pytest.fixture
def faiss_mock(monkeypatch):
    """FAISS, torch 모듈을 모킹하여 실제 인덱스 구축을 우회."""
    import faiss

    mock_index = MagicMock()
    mock_index.d = 128

    monkeypatch.setattr(faiss, "index_factory", lambda *a, **k: mock_index)
    monkeypatch.setattr(faiss, "normalize_L2", lambda *a, **k: None)
    monkeypatch.setattr(faiss, "downcast_index", lambda idx: mock_index)
    monkeypatch.setattr(faiss, "METRIC_INNER_PRODUCT", 0)

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    import src.core.retriever_factory as rf

    # SessionManager.add_status_log는 실제 세션이 필요하므로 모킹
    session_mock = MagicMock()
    session_mock.add_status_log = MagicMock()
    monkeypatch.setattr(rf, "SessionManager", session_mock)


def test_create_vector_store_no_redundant_embedding(faiss_mock):
    """벡터가 전달되었을 때 embedder.embed_documents가 호출되지 않는지 검증"""
    mock_docs = [
        Document(page_content="테스트 문서 1", metadata={"page": 1}),
        Document(page_content="테스트 문서 2", metadata={"page": 2}),
    ]
    mock_embedder = MagicMock()
    mock_vectors = [np.random.rand(128).astype("float32") for _ in mock_docs]

    create_vector_store(mock_docs, mock_embedder, vectors=mock_vectors)

    mock_embedder.embed_documents.assert_not_called()
