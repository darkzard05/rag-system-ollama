"""
Mock Factory for RAG System tests.
Provides standardized mock objects to avoid repetitive mocking setup and reduce coupling to implementation details.
"""

from unittest.mock import MagicMock, AsyncMock
from langchain_core.documents import Document


def create_mock_document(
    content: str = "테스트 문서 내용", metadata: dict = None
) -> Document:
    """표준 테스트 문서 객체를 생성합니다."""
    if metadata is None:
        metadata = {"page": 1, "file_hash": "test_hash", "has_coordinates": True}
    return Document(page_content=content, metadata=metadata)


def create_mock_llm():
    """표준 모킹 LLM을 생성합니다."""
    mock_llm = AsyncMock()
    # 기본 ainvoke 응답 설정
    mock_llm.ainvoke.return_value = {
        "response": "모킹된 답변입니다.",
        "thought": "모킹된 생각 과정입니다.",
        "relevant_docs": [create_mock_document()],
        "performance": {"latency": 0.1},
    }
    return mock_llm


def create_mock_embedder():
    """표준 모킹 임베더를 생성합니다."""
    mock_embedder = MagicMock()
    mock_embedder.model = "test-embedding-model"
    # embed_documents mock
    mock_embedder.embed_documents.return_value = [[0.1] * 128]
    # embed_query mock
    mock_embedder.embed_query.return_value = [0.1] * 128
    return mock_embedder


def create_mock_vector_store():
    """표준 모킹 벡터 스토어를 생성합니다."""
    mock_vs = MagicMock()

    # as_retriever mock
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever

    # similarity_search mock
    mock_vs.similarity_search.return_value = [create_mock_document()]

    return mock_vs


def create_mock_bm25_retriever():
    """표준 모킹 BM25 리트리버를 생성합니다."""
    mock_bm25 = MagicMock()
    mock_bm25.k = 5
    mock_bm25.get_relevant_documents.return_value = [create_mock_document()]
    return mock_bm25
