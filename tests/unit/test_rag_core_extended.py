import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from rag_core import (
    _load_pdf_docs,
    _split_documents,
    _create_vector_store,
    _create_bm25_retriever,
    _create_ensemble_retriever,
)


@patch("rag_core.PyMuPDFLoader")
@patch("rag_core.tempfile.NamedTemporaryFile")
@patch("rag_core.os.remove")
def test_load_pdf_docs(mock_remove, mock_tempfile, mock_loader):
    """Tests the PDF loading function."""
    # 1. Arrange
    mock_file = MagicMock()
    mock_tempfile.return_value.__enter__.return_value = mock_file
    mock_loader.return_value.load.return_value = [Document(page_content="test")]
    pdf_bytes = b"dummy pdf bytes"

    # 2. Act
    docs = _load_pdf_docs(pdf_bytes)

    # 3. Assert
    mock_tempfile.assert_called_once_with(suffix=".pdf", delete=False)
    mock_file.write.assert_called_once_with(pdf_bytes)
    mock_loader.assert_called_once()
    mock_remove.assert_called_once()
    assert len(docs) == 1
    assert docs[0].page_content == "test"


@patch("rag_core.RecursiveCharacterTextSplitter")
def test_split_documents(mock_splitter):
    """Tests the document splitting function."""
    # 1. Arrange
    docs = [Document(page_content="a" * 2000)]
    mock_splitter.return_value.split_documents.return_value = [
        Document(page_content="a" * 1500),
        Document(page_content="a" * 500),
    ]

    # 2. Act
    split_docs = _split_documents(docs)

    # 3. Assert
    mock_splitter.assert_called_once()
    mock_splitter.return_value.split_documents.assert_called_once_with(docs)
    assert len(split_docs) == 2


@patch("rag_core.FAISS")
def test_create_vector_store(mock_faiss):
    """Tests the FAISS vector store creation."""
    # 1. Arrange
    docs = [Document(page_content="test")]
    embedder = MagicMock()

    # 2. Act
    _create_vector_store(docs, embedder)

    # 3. Assert
    mock_faiss.from_documents.assert_called_once_with(docs, embedder)


@patch("rag_core.BM25Retriever")
def test_create_bm25_retriever(mock_bm25):
    """Tests the BM25 retriever creation."""
    # 1. Arrange
    docs = [Document(page_content="test")]

    # 2. Act
    retriever = _create_bm25_retriever(docs)

    # 3. Assert
    mock_bm25.from_documents.assert_called_once_with(docs)
    assert retriever.k == 5  # Check if k is set from config


@patch("rag_core.EnsembleRetriever")
def test_create_ensemble_retriever(mock_ensemble):
    """Tests the Ensemble retriever creation."""
    # 1. Arrange
    vector_store = MagicMock()
    bm25_retriever = MagicMock()
    faiss_retriever = MagicMock()
    vector_store.as_retriever.return_value = faiss_retriever

    # 2. Act
    _create_ensemble_retriever(vector_store, bm25_retriever)

    # 3. Assert
    mock_ensemble.assert_called_once_with(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
    )
