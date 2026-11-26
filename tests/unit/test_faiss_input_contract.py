"""FAISS construction validates its inputs and does not mutate caller vectors."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

from common.exceptions import VectorStoreError
from core.retriever_factory import create_vector_store


def _docs(n: int) -> list[Document]:
    return [Document(page_content=f"doc {i}") for i in range(n)]


def test_vector_count_must_match_document_count() -> None:
    vectors = np.ones((3, 4), dtype="float32")
    with pytest.raises(VectorStoreError):
        create_vector_store(_docs(1), MagicMock(), vectors=vectors)


def test_normalization_does_not_mutate_caller_array() -> None:
    vectors = np.array([[3.0, 4.0, 0.0, 0.0]] * 2, dtype="float32")
    original = vectors.copy()
    create_vector_store(_docs(2), MagicMock(), vectors=vectors)
    np.testing.assert_array_equal(vectors, original)
