import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

# Test target
from rag_core import build_rag_pipeline

# To check the final state
from session import SessionManager


@pytest.fixture(autouse=True)
def setup_session():
    """Ensure each test starts with a clean session."""
    SessionManager.init_session()
    yield
    SessionManager.reset_all_state()


def test_build_rag_pipeline_cache_miss(mocker):
    """
    Tests the entire RAG pipeline build process when there is no cache.
    It checks if the functions for loading, splitting, embedding, and retrieving
    are called in the correct order.
    """
    # 1. Arrange: Mock all external and heavy functions
    mocker.patch("rag_core.load_pdf_docs", return_value=[Document(page_content="...")])
    mocker.patch(
        "rag_core.split_documents", return_value=[Document(page_content="...")]
    )
    mocker.patch("rag_core.create_vector_store", return_value=MagicMock())
    mocker.patch("rag_core.create_bm25_retriever", return_value=MagicMock())
    mocker.patch("rag_core.create_ensemble_retriever", return_value=MagicMock())
    mocker.patch("rag_core.build_graph", return_value=MagicMock())

    # Mock the VectorStoreCache methods
    mock_cache_load = mocker.patch(
        "rag_core.VectorStoreCache.load", return_value=(None, None)
    )
    mock_cache_save = mocker.patch("rag_core.VectorStoreCache.save")

    # Dummy inputs
    dummy_file_name = "test.pdf"
    dummy_file_bytes = b"dummy pdf content"
    mock_llm = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.model_name = "dummy-embedding-model"

    # 2. Act: Call the function to be tested
    success_message, cache_used = build_rag_pipeline(
        uploaded_file_name=dummy_file_name,
        file_bytes=dummy_file_bytes,
        llm=mock_llm,
        embedder=mock_embedder,
    )

    # 3. Assert: Verify the process flow and end state

    # Check if cache was checked and then saved (cache miss flow)
    mock_cache_load.assert_called_once()
    mock_cache_save.assert_called_once()
    assert cache_used is False

    # Check if all processing steps were called
    # (We get the patched objects for assertion)
    rag_core = __import__("rag_core")
    rag_core.load_pdf_docs.assert_called_once_with(dummy_file_bytes)
    rag_core.split_documents.assert_called_once()
    rag_core.create_vector_store.assert_called_once()
    rag_core.create_bm25_retriever.assert_called_once()
    rag_core.create_ensemble_retriever.assert_called_once()
    rag_core.build_graph.assert_called_once()

    # Check if the session was updated correctly
    assert SessionManager.get("pdf_processed") is True
    assert SessionManager.get("qa_chain") is not None
    assert "완료" in success_message


def test_build_rag_pipeline_cache_hit(mocker):
    """
    Tests the RAG pipeline build process when a cache is available.
    It checks that document processing steps are skipped.
    """
    # 1. Arrange: Mock functions and simulate a cache hit
    mock_load_pdf_docs = mocker.patch("rag_core.load_pdf_docs")
    mock_split_documents = mocker.patch("rag_core.split_documents")
    mock_create_vector_store = mocker.patch("rag_core.create_vector_store")
    mock_cache_save = mocker.patch("rag_core.VectorStoreCache.save")

    # Simulate that cache.load() returns valid data
    mock_docs = [Document(page_content="cached content")]
    mock_vs = MagicMock()
    mock_cache_load = mocker.patch(
        "rag_core.VectorStoreCache.load", return_value=(mock_docs, mock_vs)
    )

    # These should still be called
    mock_create_bm25 = mocker.patch(
        "rag_core.create_bm25_retriever", return_value=MagicMock()
    )
    mock_create_ensemble = mocker.patch(
        "rag_core.create_ensemble_retriever", return_value=MagicMock()
    )
    mock_build_graph = mocker.patch("rag_core.build_graph", return_value=MagicMock())

    # Dummy inputs
    dummy_file_name = "test.pdf"
    dummy_file_bytes = b"dummy pdf content"
    mock_llm = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.model_name = "dummy-embedding-model"

    # 2. Act: Call the function
    success_message, cache_used = build_rag_pipeline(
        uploaded_file_name=dummy_file_name,
        file_bytes=dummy_file_bytes,
        llm=mock_llm,
        embedder=mock_embedder,
    )

    # 3. Assert: Verify the process flow for a cache hit
    mock_cache_load.assert_called_once()
    assert cache_used is True

    # Assert that expensive processing steps were SKIPPED
    mock_load_pdf_docs.assert_not_called()
    mock_split_documents.assert_not_called()
    mock_create_vector_store.assert_not_called()
    mock_cache_save.assert_not_called()

    # Assert that the rest of the pipeline was still built
    mock_create_bm25.assert_called_once()
    mock_create_ensemble.assert_called_once()
    mock_build_graph.assert_called_once()

    # Check session state
    assert SessionManager.get("pdf_processed") is True
    assert SessionManager.get("qa_chain") is not None
    assert "캐시를 불러왔습니다" in success_message
