# Contract test for create_vector_store precomputed-vectors enforcement (F2).
#
# Baseline (old behavior) + failing-first proof (new behavior) for the
# "vectors must be precomputed" contract.
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

from common.exceptions import VectorStoreError
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

    session_mock = MagicMock()
    session_mock.add_status_log = MagicMock()
    monkeypatch.setattr(rf, "SessionManager", session_mock)


def _make_docs(n: int = 2) -> list[Document]:
    return [
        Document(page_content=f"테스트 문서 {i}", metadata={"page": i})
        for i in range(1, n + 1)
    ]


# --- CONTRACT PROOF: the new "vectors must be precomputed" enforcement ---
def test_vectors_none_raises_vector_store_error(faiss_mock):
    """FIX CONTRACT: vectors=None must raise, never silently re-embed."""
    docs = _make_docs(1)
    mock_embedder = MagicMock()

    with pytest.raises(VectorStoreError) as exc_info:
        create_vector_store(docs, mock_embedder, vectors=None)

    assert exc_info.value.details.get("reason") == (
        "vectors required; callers must pass precomputed embeddings"
    )
    mock_embedder.embed_documents.assert_not_called()


def test_vectors_precomputed_never_calls_embedder(faiss_mock):
    """FIX CONTRACT: provided vectors must never trigger re-embedding."""
    docs = _make_docs(2)
    mock_embedder = MagicMock()
    precomputed = [np.random.rand(128).astype("float32") for _ in docs]

    create_vector_store(docs, mock_embedder, vectors=precomputed)

    mock_embedder.embed_documents.assert_not_called()
